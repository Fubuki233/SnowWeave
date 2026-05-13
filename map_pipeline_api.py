"""
FastAPI server for the SnowGlobe map pipeline.

Generation lives in SnowWeave/dependencies/agent-sprite-forge/scripts/asf.py.
This API only manages async tasks and returns a Godot-friendly artifact bundle.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


SNOWWEAVE_ROOT = Path(__file__).resolve().parent
DEPENDENCIES_ROOT = SNOWWEAVE_ROOT / "dependencies"
ASF_SCRIPTS = DEPENDENCIES_ROOT / "agent-sprite-forge" / "scripts"
GODOT_PROJECT = Path(r"D:\SnowGlobe\SnowGlobe\snow-globe")
DEFAULT_OUT = SNOWWEAVE_ROOT / "out" / "maps"
FMG_REF_ROOT = SNOWWEAVE_ROOT / "out" / "fmg_refs"
SNOWWEAVE_SCRIPTS = SNOWWEAVE_ROOT / "scripts"
API_KEY_ENV_NAMES = ("OPENROUTER_API_KEY", "NAGA_API_KEY", "OPENAI_API_KEY")
DEFAULT_FMG_BACKEND_URL = "http://127.0.0.1:8765"
DEFAULT_FMG_CHUNK_SIZE = 4096

sys.path.insert(0, str(ASF_SCRIPTS))
import asf  # type: ignore  # noqa: E402


app = FastAPI(title="MapPipeline", version="4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_tasks: dict[str, dict[str, Any]] = {}
_events: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()
_condition = threading.Condition(_lock)


def _task_id() -> str:
    task_id = uuid.uuid4().hex[:10]
    with _condition:
        _tasks[task_id] = {"status": "pending", "progress": 0, "message": ""}
        _events[task_id] = []
    return task_id


def _update(task_id: str, **updates: Any) -> None:
    with _condition:
        if task_id in _tasks:
            _tasks[task_id].update(updates)
            _condition.notify_all()


def _emit(task_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    with _condition:
        if task_id not in _tasks:
            raise KeyError(task_id)
        _tasks[task_id].update(payload)
        event = {
            "seq": len(_events[task_id]),
            "task_id": task_id,
            "type": event_type,
            **payload,
        }
        _events[task_id].append(event)
        _condition.notify_all()
        return event


def _artifact_rel_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _artifact_entry(path: Path, output_dir: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "source_path": str(path.resolve()),
        "relative_path": _artifact_rel_path(path, output_dir),
        "mime": mime,
        "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _path_values(result: dict[str, Any]) -> list[Path]:
    paths = [
        result.get("base_image"),
        result.get("dressed_image"),
        result.get("preview_image"),
        result.get("preview_annotated_image"),
        result.get("prop_alpha_overlay_image"),
        result.get("placements_json"),
        result.get("prop_manifest_json"),
        result.get("preview_report"),
        result.get("prop_alpha_overlay_report"),
    ]
    for prop in result.get("props", []):
        if isinstance(prop, dict):
            paths.append(prop.get("image"))
    return [Path(str(path)) for path in paths if path]


def _attach_godot_files(result: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(result["output_dir"]))
    files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in _path_values(result):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        entry = _artifact_entry(resolved, output_dir)
        if entry:
            files.append(entry)
    result["godot_project"] = str(GODOT_PROJECT)
    result["godot_cache_hint"] = "user://map_pipeline_cache/<task_id>"
    result["files"] = files
    return result


def _read_user_env_from_registry(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value).strip() or None
    except OSError:
        return None


def _api_key_from_environment() -> str | None:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    for name in API_KEY_ENV_NAMES:
        value = _read_user_env_from_registry(name)
        if value:
            os.environ[name] = value
            return value
    return None


def _run_json_script(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _prepare_fmg_reference(
    *,
    output_dir: Path,
    seed: str | None,
    chunk_id: str,
    chunk_size: int,
    backend_url: str,
    bundle_zip: str | None,
) -> dict[str, Any]:
    if chunk_size != DEFAULT_FMG_CHUNK_SIZE:
        raise ValueError("FMG SnowWeave integration requires fixed 4096px chunks.")

    ref_dir = output_dir / "fmg-reference"
    script = SNOWWEAVE_SCRIPTS / "fmg_unpack_atlas_bundle.py"
    command = [
        str(script),
        "--output-dir", str(ref_dir),
        "--chunk-size", str(chunk_size),
        "--backend-url", backend_url,
        "--force",
    ]
    if seed:
        command.extend(["--seed", seed])
    if bundle_zip:
        command.extend(["--zip", bundle_zip])

    index = _run_json_script(command)
    chunks = index.get("chunks")
    if not isinstance(chunks, dict) or chunk_id not in chunks:
        raise ValueError(f"FMG chunk not found: {chunk_id}")
    chunk = chunks[chunk_id]
    if not isinstance(chunk, dict):
        raise ValueError(f"Invalid FMG chunk entry: {chunk_id}")

    return {
        "index_path": str((ref_dir / "fmg_reference_index.json").resolve()),
        "index": index,
        "chunk_id": chunk_id,
        "chunk": chunk,
        "image_path": str(chunk["png"]),
        "legend_context": str(chunk.get("legend_context") or ""),
    }


def _run_pipeline(
    task_id: str,
    *,
    prompt: str,
    output_dir: Path,
    image_paths: list[str],
    prompt_context: str,
    reference_metadata: dict[str, Any],
    use_fmg_reference: bool,
    fmg_seed: str | None,
    fmg_chunk_id: str,
    fmg_chunk_size: int,
    fmg_backend_url: str,
    fmg_bundle_zip: str | None,
    model: str,
    diff: float,
    min_area: int,
    map_mode: str,
    no_shadow_suppression: bool,
    no_edge_delta: bool,
    edge_threshold: float,
    edge_grow_radius: int,
    edge_support_radius: int,
    no_fill_holes: bool,
    matting_backend: str,
    no_rembg_alpha_matting: bool,
    no_constrain_rembg_to_diff_mask: bool,
) -> None:
    try:
        _emit(
            task_id,
            "started",
            status="running",
            progress=5,
            message="Generating map bundle with ASF",
        )
        effective_image_paths = list(image_paths)
        effective_prompt_context = prompt_context
        effective_reference_metadata = dict(reference_metadata)
        if use_fmg_reference:
            _emit(
                task_id,
                "fmg_reference",
                status="running",
                progress=8,
                message=f"Preparing FMG chunk reference {fmg_chunk_id}",
            )
            fmg_reference = _prepare_fmg_reference(
                output_dir=output_dir,
                seed=fmg_seed,
                chunk_id=fmg_chunk_id,
                chunk_size=fmg_chunk_size,
                backend_url=fmg_backend_url,
                bundle_zip=fmg_bundle_zip,
            )
            effective_image_paths = [fmg_reference["image_path"], *effective_image_paths]
            contexts = [effective_prompt_context, fmg_reference["legend_context"]]
            effective_prompt_context = "\n\n".join(context for context in contexts if context.strip())
            effective_reference_metadata["fmg_reference"] = {
                "index_path": fmg_reference["index_path"],
                "chunk_id": fmg_reference["chunk_id"],
                "chunk": fmg_reference["chunk"],
            }

        result = asf.generate_map_pipeline(
            prompt=prompt,
            output_dir=output_dir,
            image_paths=effective_image_paths,
            prompt_context=effective_prompt_context,
            reference_metadata=effective_reference_metadata,
            model=model,
            map_mode=map_mode,
            api_key=_api_key_from_environment(),
            sub_diff_threshold=diff,
            sub_min_component_area=min_area,
            sub_no_shadow_suppression=no_shadow_suppression,
            sub_no_edge_delta=no_edge_delta,
            sub_edge_threshold=edge_threshold,
            sub_edge_grow_radius=edge_grow_radius,
            sub_edge_support_radius=edge_support_radius,
            sub_no_fill_holes=no_fill_holes,
            sub_matting_backend=matting_backend,
            sub_no_rembg_alpha_matting=no_rembg_alpha_matting,
            sub_no_constrain_rembg_to_diff_mask=no_constrain_rembg_to_diff_mask,
        )
        result = _attach_godot_files(result)
        _emit(
            task_id,
            "completed",
            status="completed",
            progress=100,
            message=f"Done ({len(result.get('props', []))} props)",
            result=result,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _emit(task_id, "failed", status="failed", progress=100, message=str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "4.1",
        "tasks": len(_tasks),
        "dependencies_root": str(DEPENDENCIES_ROOT),
        "output_root": str(DEFAULT_OUT),
        "godot_project": str(GODOT_PROJECT),
        "api_key_configured": bool(_api_key_from_environment()),
    }


@app.post("/generate")
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip() or "2D top-down RPG map"
    image_path = payload.get("image")
    if image_path and not Path(str(image_path)).exists():
        raise HTTPException(400, f"Image not found: {image_path}")
    use_fmg_reference = bool(payload.get("use_fmg_reference", False))
    if not prompt and not image_path and not use_fmg_reference:
        raise HTTPException(400, "Need prompt or image")

    task_id = _task_id()
    output_name = str(payload.get("output_name") or "").strip()
    if not output_name:
        output_name = f"map-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    output_dir = DEFAULT_OUT / output_name
    _emit(
        task_id,
        "queued",
        status="pending",
        progress=0,
        message="Queued",
        output_dir=str(output_dir),
    )
    thread = threading.Thread(
        target=_run_pipeline,
        kwargs={
            "task_id": task_id,
            "prompt": prompt,
            "output_dir": output_dir,
            "image_paths": [str(image_path)] if image_path else [],
            "prompt_context": str(payload.get("prompt_context") or ""),
            "reference_metadata": {},
            "use_fmg_reference": use_fmg_reference,
            "fmg_seed": str(payload.get("fmg_seed") or "").strip() or None,
            "fmg_chunk_id": str(payload.get("fmg_chunk_id") or "chunk_0_0"),
            "fmg_chunk_size": int(payload.get("fmg_chunk_size") or DEFAULT_FMG_CHUNK_SIZE),
            "fmg_backend_url": str(payload.get("fmg_backend_url") or DEFAULT_FMG_BACKEND_URL),
            "fmg_bundle_zip": str(payload.get("fmg_bundle_zip") or "").strip() or None,
            "model": str(payload.get("model") or "gemini3.1flash"),
            "diff": float(payload.get("sub_diff_threshold", 30)),
            "min_area": int(payload.get("sub_min_component_area", 100)),
            "map_mode": str(payload.get("map_mode") or "auto"),
            "no_shadow_suppression": bool(payload.get("sub_no_shadow_suppression", False)),
            "no_edge_delta": bool(payload.get("sub_no_edge_delta", True)),
            "edge_threshold": float(payload.get("sub_edge_threshold", 18)),
            "edge_grow_radius": int(payload.get("sub_edge_grow_radius", 2)),
            "edge_support_radius": int(payload.get("sub_edge_support_radius", 10)),
            "no_fill_holes": bool(payload.get("sub_no_fill_holes", False)),
            "matting_backend": str(payload.get("sub_matting_backend", "auto")),
            "no_rembg_alpha_matting": bool(payload.get("sub_no_rembg_alpha_matting", False)),
            "no_constrain_rembg_to_diff_mask": bool(payload.get("sub_no_constrain_rembg_to_diff_mask", False)),
        },
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "status": "pending", "output_dir": str(output_dir)}


@app.get("/status/{task_id}")
def status(task_id: str) -> dict[str, Any]:
    with _lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "not found")
    return task


@app.get("/events/{task_id}")
def events(task_id: str, after: int = -1, timeout: float = 45.0) -> dict[str, Any]:
    """Return the next task event after `after`, waiting up to `timeout` seconds.

    This is intentionally simple long polling rather than SSE so Godot can use
    the same HttpClient request/response path it already uses elsewhere.
    """
    timeout = max(1.0, min(timeout, 120.0))
    with _condition:
        if task_id not in _tasks:
            raise HTTPException(404, "not found")

        def has_next() -> bool:
            return any(event["seq"] > after for event in _events[task_id])

        _condition.wait_for(has_next, timeout=timeout)
        for event in _events[task_id]:
            if event["seq"] > after:
                return event
        task = _tasks[task_id]
        return {
            "seq": after,
            "task_id": task_id,
            "type": "timeout",
            "status": task.get("status", "pending"),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
        }


@app.post("/load")
def load(payload: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(str(payload["output_dir"])) / "pipeline_result.json"
    if not result_path.exists():
        raise HTTPException(404, str(result_path))
    return _attach_godot_files(json.loads(result_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"MapPipeline API -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
