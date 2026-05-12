"""
地图标注器 — 两阶段 VLM grounding 管线
Phase 1: 碰撞/遮蔽物标注（bbox 回归）
Phase 2: 物品标注（用 Godot 物品目录）

流程：
  1. 发原图给 VLM（不加网格）
  2. Phase 1: VLM 返回 bbox + 碰撞类型
  3. Phase 2: VLM 返回 bbox + 物品 ID
  4. 导出 JSON / .tscn / 可视化

可扩展：
  - 自定义物品目录
  - 自定义 VLM prompt
  - 不支持网格模式

⚠️ 需要依赖：openai, Pillow
"""

import os
import json
import re
import sys
import argparse
import contextlib
import io
from pathlib import Path
from base64 import b64encode
from io import BytesIO
from openai import OpenAI
from PIL import Image, ImageDraw

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
#  物品目录（从 Godot AIItem 导出，可运行时覆盖）
# ============================================================

DEFAULT_ITEM_CATALOG = {
    "plants": [
        {"id": "oak_tree",      "name": "橡树",     "desc": "高大的阔叶树，可砍伐获得木材和橡子"},
        {"id": "pine_tree",     "name": "松树",     "desc": "常绿针叶树，可砍伐获得木材和松果"},
        {"id": "durian_tree",   "name": "榴莲树",   "desc": "热带果树，可采集榴莲果实"},
        {"id": "strawberry",    "name": "草莓",     "desc": "矮小草本植物，可采集草莓"},
        {"id": "bush",          "name": "灌木丛",   "desc": "低矮的灌木，可能隐藏物品或小动物"},
        {"id": "flower",        "name": "花丛",     "desc": "鲜艳的花朵，可采集作为装饰或材料"},
        {"id": "dead_tree",     "name": "枯树",     "desc": "枯死的树干，可砍伐获得少量木材"},
    ],
    "buildings": [
        {"id": "house",         "name": "房屋",     "desc": "木质或石质建筑，可能有NPC居住"},
        {"id": "fence",         "name": "栅栏",     "desc": "木质或石质围栏，标记区域边界"},
        {"id": "well",          "name": "水井",     "desc": "石头砌成的水井，可以取水"},
        {"id": "bridge",        "name": "桥梁",     "desc": "横跨水域的通道"},
        {"id": "sign_post",     "name": "路标",     "desc": "指示方向的木牌"},
        {"id": "farmland",      "name": "农田",     "desc": "被开垦的可种植土地"},
    ],
    "terrain": [
        {"id": "water",         "name": "水域",     "desc": "水体，不可行走"},
        {"id": "rock",          "name": "岩石",     "desc": "大型石头障碍物，不可行走"},
        {"id": "cliff",         "name": "悬崖",     "desc": "陡峭的高差边界，不可穿越"},
        {"id": "path",          "name": "小路",     "desc": "人工铺设的道路"},
        {"id": "grass",         "name": "草地",     "desc": "可通行的绿色草地"},
        {"id": "sand",          "name": "沙地",     "desc": "沙滩或沙漠地形"},
    ],
    "objects": [
        {"id": "chest",         "name": "宝箱",     "desc": "可能包含物品"},
        {"id": "barrel",        "name": "木桶",     "desc": "可能装有液体或食物"},
        {"id": "campfire",      "name": "篝火",     "desc": "可用于烹饪或休息"},
        {"id": "torch",         "name": "火把",     "desc": "发光的照明物"},
        {"id": "mushroom",      "name": "蘑菇",     "desc": "可采集的真菌，部分有毒"},
    ],
}


# ============================================================
#  地图标注器
# ============================================================

class MapGridAnnotator:
    """地图标注器 — 两阶段 VLM grounding（无网格，纯 bbox）"""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen3-vl-flash"
        self.item_catalog = dict(DEFAULT_ITEM_CATALOG)

    # ---- 物品目录 ----

    def load_item_catalog(self, catalog_path: str):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.item_catalog = json.load(f)
        print(f"📋 加载物品目录: {catalog_path}")

    def build_catalog_text(self) -> str:
        lines = []
        for category, items in self.item_catalog.items():
            lines.append(f"\n【{category}】")
            for item in items:
                lines.append(f"  - {item['id']} ({item['name']}): {item['desc']}")
        return "\n".join(lines)

    # ---- 原图 base64 ----

    def _image_url(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=92)
        data = b64encode(buf.getvalue()).decode()
        print(f"   🖼️ 原图: {w}×{h}, base64: {len(data)/1024:.0f} KB")
        return f"data:image/jpeg;base64,{data}", w, h

    # ---- VLM 调用 ----

    def _call_vlm(self, image_url: str, prompt: str, img_w: int, img_h: int) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "min_pixels": 128 * 28 * 28,   # ≥65536, 官方示例用这个
                     "max_pixels": 2560 * 32 * 32,
                     "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ]
            }],
            temperature=0.2,
            timeout=120,
        )
        content = completion.choices[0].message.content
        print(f"   📥 响应 {len(content)} 字符")
        return content

    def _norm_bbox_to_pixels(self, bbox: list, img_w: int, img_h: int) -> list:
        """Qwen3-VL 返回 0-1000 归一化坐标 → 真实像素"""
        if len(bbox) < 4:
            return bbox
        return [
            bbox[0] / 1000 * img_w,
            bbox[1] / 1000 * img_h,
            bbox[2] / 1000 * img_w,
            bbox[3] / 1000 * img_h,
        ]

    def _denormalize_objects(self, objects: list, img_w: int, img_h: int) -> list:
        """将 VLM 返回的归一化坐标转为像素坐标"""
        result = []
        for obj in objects:
            bbox = obj.get("bbox_2d", [])
            if len(bbox) >= 4:
                obj = dict(obj)
                obj["bbox_2d"] = self._norm_bbox_to_pixels(bbox, img_w, img_h)
            result.append(obj)
        return result

    def _extract_json(self, text: str) -> dict:
        # 1) 直接解析完整文本
        try:
            return json.loads(text)
        except:
            pass

        # 2) 提取 markdown 代码块
        m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass

        # 3) 从末尾往前找：去掉尾随的非 JSON 内容
        #    有些模型在 JSON 后面多写文字
        end = len(text)
        while end > 0:
            try:
                return json.loads(text[:end])
            except json.JSONDecodeError as e:
                # 如果错误位置接近末尾，可能是模型在 JSON 后多写了内容
                if e.pos > end - 20:
                    end = e.pos
                    continue
                break

        # 4) 括号匹配提取
        start = text.find('{')
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception as e:
                            print(f"   ⚠️ JSON 解析失败: {e}")
                            print(f"   末尾: ...{text[max(0,i-200):i+1]}")
                            raise
        raise ValueError(f"无法提取 JSON, 前200字符: {text[:200]}")

    # ================================================================
    #  Phase 1: 碰撞标注
    # ================================================================

    def annotate_collision(self, image_path: str) -> dict:
        print("🔍 Phase 1: 碰撞标注 (grounding)...")
        image_url, w, h = self._image_url(image_path)

        prompt = """你是游戏关卡设计师。这是一张等轴/俯视视角的游戏地图。
这是一个多层 TILEMAP 系统——地面层和覆盖物层是分开的。

**任务**: 检测图中的物理碰撞实体和上层遮挡实体，输出 bounding box 和分层类型。
overlay 虽然地面可走，但属于上层遮挡，必须标注出来。
暂时不要标注水体；桥梁、道路、桥面都允许通过，不属于碰撞或 overlay。

**碰撞类型**:
- overlay  : 上方有覆盖物但地面可走（树冠、房顶、屋檐、遮阳棚、凉亭顶）
            判断：图中能看到透出下方地面颜色 → overlay
            注意：overlay 是可通行的遮挡层，不是 non-walkable，但仍然必须输出。
- obstacle : 实体障碍（树干、墙壁、岩石、灌木丛——从地面到顶完全实心）
            判断：看不到下方地面，完全遮住 → obstacle
- cliff    : 悬崖
- path     : 人工道路
- building : 建筑内部空地

**不要输出**:
- water / river / lake / stream：暂时不作为碰撞标注。
- bridge：桥梁和桥面允许通过，不作为 obstacle，也不作为 overlay。

**bbox 规则**:
- 只使用 bbox_2d。
- 对斜向、弯曲、细长的不规则区域，尤其是悬崖边界、道路，不要用一个大框包住整体。
- 必须沿着形状走向拆成多个较小 bbox 段，每段只覆盖真实不可通行/对应类型区域。
- 河流从地图对角穿过时也不要输出 water bbox；保持可通行。
- 单个 bbox 应尽量贴合局部形状；如果矩形内明显包含大量可行走陆地，请继续拆小。

**输出 JSON**:
{
  "objects": [
    {"bbox_2d": [120, 80, 200, 180], "label": "overlay", "confidence": 0.9},
    {"bbox_2d": [300, 200, 360, 320], "label": "obstacle", "confidence": 0.8}
  ],
  "summary": "森林地图，树冠覆盖大范围，中部有岩石"
}
bbox_2d = [x1, y1, x2, y2] 左上和右下像素坐标。
仅返回 JSON。"""

        raw = self._extract_json(self._call_vlm(image_url, prompt, w, h))
        raw_objects = self._denormalize_objects(raw.get("objects", []), w, h)
        print(f"   📐 归一化→像素 bbox 示例: {json.dumps(raw_objects[:2], ensure_ascii=False)}")
        return {"objects": raw_objects, "summary": raw.get("summary", "")}

    # ================================================================
    #  Phase 2: 物品标注
    # ================================================================

    def annotate_items(self, image_path: str) -> dict:
        print("🔍 Phase 2: 物品标注 (grounding)...")
        image_url, w, h = self._image_url(image_path)

        catalog = self.build_catalog_text()
        prompt = f"""你是游戏关卡设计师。这是一张等轴/俯视视角的游戏地图。

**可用物品目录**:
{catalog}

**任务**: 检测图中出现的所有物品，输出 bounding box 和物品 ID。

**输出 JSON**:
{{
  "objects": [
    {{"bbox_2d": [120, 80, 200, 180], "label": "oak_tree", "confidence": 0.9}},
    {{"bbox_2d": [400, 300, 450, 360], "label": "rock", "confidence": 0.8}}
  ],
  "summary": "地图上方有橡树，中部有岩石"
}}
bbox_2d = [x1, y1, x2, y2] 左上和右下像素坐标。
仅返回 JSON。"""

        raw = self._extract_json(self._call_vlm(image_url, prompt, w, h))
        raw_objects = self._denormalize_objects(raw.get("objects", []), w, h)
        return {"objects": raw_objects, "summary": raw.get("summary", "")}

    # ================================================================
    #  完整两阶段管线
    # ================================================================

    def annotate_full(self, image_path: str) -> dict:
        print("=" * 60)
        print(f"🌍 地图两阶段标注: {image_path}")
        print("=" * 60)

        collision = self.annotate_collision(image_path)
        items = self.annotate_items(image_path)

        img = Image.open(image_path)
        result = {
            "image": str(Path(image_path).resolve()),
            "image_size": {"width": img.width, "height": img.height},
            "collision_objects": collision["objects"],
            "item_objects": items["objects"],
            "collision_summary": collision["summary"],
            "item_summary": items["summary"],
        }

        nc, ni = len(result["collision_objects"]), len(result["item_objects"])
        print(f"\n✅ 标注完成: {nc} 碰撞 + {ni} 物品")
        return result

    # ---- 导出 ----

    def export_json(self, result: dict, output_path: str):
        export = {
            "version": "2.0",
            "image": result["image"],
            "image_size": result["image_size"],
            "collision_objects": result["collision_objects"],
            "item_objects": result["item_objects"],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON: {output_path}")

    def export_scene(self, result: dict, output_path: str):
        lines = [
            f'[gd_scene load_steps=1 format=3 uid="uid://auto_{os.urandom(4).hex()}"]',
            '',
            '[node name="ImportedMap" type="Node2D"]',
            f'; 图片: {result["image_size"]["width"]}x{result["image_size"]["height"]}',
            '',
        ]
        for obj in result.get("item_objects", []):
            b = obj.get("bbox_2d", [0, 0, 0, 0])
            label = obj.get("label", "?")
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            lines += [
                f'[node name="{label}_{cx:.0f}_{cy:.0f}" type="Node2D" parent="."]',
                f'position = Vector2({cx:.0f}, {cy:.0f})',
                f'; item_id = "{label}", bbox={b}',
                '',
            ]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"🎮 场景: {output_path}")

    # ---- 可视化 ----

    COLLISION_COLORS = {
        "walkable":  (0, 255, 0, 60),
        "overlay":   (0, 180, 0, 90),
        "obstacle":  (255, 0, 0, 80),
        "water":     (0, 100, 255, 80),
        "cliff":     (255, 165, 0, 80),
        "path":      (200, 200, 0, 60),
        "building":  (128, 0, 128, 80),
    }

    def visualize(self, image_path: str, result: dict, output_path: str):
        img = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for obj in result.get("collision_objects", []):
            b = obj.get("bbox_2d", [])
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            color = self.COLLISION_COLORS.get(obj.get("label", ""), (255, 0, 0, 80))
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=(200, 200, 200, 120), width=2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tag = f"{obj.get('label','?')[:6]}"
            draw.text((cx - 20, cy - 8), tag, fill=(255, 255, 0, 255))

        for obj in result.get("item_objects", []):
            b = obj.get("bbox_2d", [])
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255, 220), width=1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tag = f"{obj.get('label','?')[:8]}"
            draw.text((cx - 20, cy + 6), tag, fill=(255, 255, 255, 255))

        result_img = Image.alpha_composite(img, overlay).convert('RGB')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_img.save(output_path)
        print(f"🎨 可视化: {output_path}")


# ============================================================
#  便捷函数
# ============================================================

def annotate_and_export(image_path: str, output_dir: str = "map_outputs",
                        api_key: str = None):
    annotator = MapGridAnnotator(api_key)
    result = annotator.annotate_full(image_path)

    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    annotator.export_json(result, f"{output_dir}/{name}.json")
    annotator.export_scene(result, f"{output_dir}/{name}.tscn")
    annotator.visualize(image_path, result, f"{output_dir}/{name}_viz.png")
    return result


def annotate_collision_json_only(image_path: str, output_dir: str = "map_outputs",
                                 api_key: str = None, quiet: bool = False) -> str:
    """
    独立给 Godot 调用的碰撞 JSON 导出方法。
    不跑物品标注，不导出 tscn，不导出可视化图片，不改变 annotate_and_export 全流程。
    """
    sink = io.StringIO()
    redirect = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()

    with redirect:
        annotator = MapGridAnnotator(api_key)
        collision = annotator.annotate_collision(image_path)
        img = Image.open(image_path)

        export = {
            "version": "2.0",
            "image": str(Path(image_path).resolve()),
            "image_size": {"width": img.width, "height": img.height},
            "collision_objects": collision["objects"],
            "item_objects": [],
        }

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"{Path(image_path).stem}_collision.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

    if quiet:
        print(f"GODOT_COLLISION_COUNT={len(export['collision_objects'])}")
        print(f"GODOT_OUTPUT_JSON={output_path.resolve()}")
    else:
        print(f"💾 Collision JSON: {output_path}")
    return str(output_path)


def annotate_full_export_for_godot_collision_import(image_path: str, output_dir: str = "map_outputs",
                                                    api_key: str = None, quiet: bool = False) -> str:
    """
    给 Godot 调用的完整导出入口。
    Python 正常跑完整流程并生成 JSON / TSCN / 可视化图；Godot 只解析输出 JSON 里的 collision_objects。
    """
    sink = io.StringIO()
    redirect = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()

    with redirect:
        result = annotate_and_export(image_path, output_dir, api_key)

    output_path = Path(output_dir) / f"{Path(image_path).stem}.json"
    if quiet:
        print(f"GODOT_COLLISION_COUNT={len(result.get('collision_objects', []))}")
        print(f"GODOT_OUTPUT_JSON={output_path.resolve()}")
    else:
        print(f"💾 Godot import JSON: {output_path}")
    return str(output_path)


def _try_run_collision_json_cli() -> bool:
    """只在 Godot/命令行显式请求时启用，避免破坏原来的 __main__ 自检流程。"""
    cli_flags = {"-h", "--help", "--collision-only", "--godot-quiet", "--output-dir", "--api-key"}
    if not any(arg in cli_flags or arg.startswith("--output-dir=") or arg.startswith("--api-key=")
               for arg in sys.argv[1:]):
        return False

    parser = argparse.ArgumentParser(description="地图完整导出；Godot 只解析碰撞")
    parser.add_argument("image", help="待标注地图图片")
    parser.add_argument("--output-dir", default="map_outputs", help="输出目录")
    parser.add_argument("--api-key", default=None, help="DashScope API Key；默认读取 DASHSCOPE_API_KEY")
    parser.add_argument("--collision-only", action="store_true", help="兼容 Godot 调用：完整导出，但 Godot 只解析碰撞")
    parser.add_argument("--godot-quiet", action="store_true", help="只输出 Godot 可解析的 ASCII 结果")
    args = parser.parse_args()

    annotate_full_export_for_godot_collision_import(args.image, args.output_dir, "sk-a07e48f51c64489e93dc6bcea0fb7ba0", args.godot_quiet)
    return True


# ============================================================
#  自检
# ============================================================

if __name__ == "__main__":
    if _try_run_collision_json_cli():
        raise SystemExit(0)

    print("=" * 60)
    print("🗺️  地图标注器 (grounding) — 自检")
    print("=" * 60)

    ok = True
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError:
        print("❌ pip install Pillow")
        ok = False
    try:
        from openai import OpenAI
        print("✅ openai")
    except ImportError:
        print("❌ pip install openai")
        ok = False
    if not ok:
        exit(1)

    print("\n📖 用法:")
    print("result = annotate_and_export('input.jpg')\n")
    result = annotate_and_export(image_path='D:\\SnowGlobe\\SnowWeave\\test\\input.jpg', api_key="sk-a07e48f51c64489e93dc6bcea0fb7ba0")
