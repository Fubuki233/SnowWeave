# FMG to SnowWeave Map Pipeline

## 简介

这个流程把 `Fantasy-Map-Generator` 导出的 4K 分片地图作为 SnowWeave basemap 的参考图，再通过 Agent Sprite Forge 生成可导入 Godot 的 layered raster map。

当前目标：

- 使用 FMG atlas chunk 作为地形、水路、道路拓扑参考。
- basemap 可美化为 clean HD RPG 地图，但必须保持地形拓扑、水网、道路网络、海岸线、高程分布不变。
- props 分为 `plants` 和 `manmade` 两类分别生成 dressed reference、分别抠图。
- 同类 props 使用 alpha overlap 处理近邻/重复：
  - alpha overlap < 15%：保留两个。
  - IoU > 45% 或候选被覆盖 > 65%：判重复，删除小的。
  - 中等重叠：合并为 cluster prop。
- 最终输出普通预览和四色 alpha overlay 调试图。

## 环境

使用 conda 环境：

```powershell
conda activate snowglobe
cd D:\SnowGlobe
```

需要配置一个图像模型 API key：

```powershell
$env:OPENROUTER_API_KEY="..."
```

也支持：

```powershell
$env:NAGA_API_KEY="..."
$env:OPENAI_API_KEY="..."
```

## 启动服务

### 1. Fantasy Map Generator 前端

```powershell
cd D:\SnowGlobe\Fantasy-Map-Generator
npm run dev -- --host 127.0.0.1 --port 5170
```

### 2. Fantasy backend

```powershell
cd D:\SnowGlobe\Fantasy-Map-Generator
uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

SnowWeave 的直接 CLI pipeline 不需要启动 `map_pipeline_api.py`。Godot UI/API 调用时才需要启动：

```powershell
cd D:\SnowGlobe
python SnowWeave\map_pipeline_api.py --host 127.0.0.1 --port 8766
```

## 完整 Pipeline 命令

直接跑完整生图、分类 dressed、prop 抠图、合并和预览：

```powershell
conda activate snowglobe
cd D:\SnowGlobe

python SnowWeave\scripts\generate_fmg_chunk_pipeline.py `
  --prompt "top-down clean HD fantasy island RPG exploration basemap, natural terrain, roads, rivers, lakes, cliffs, no buildings" `
  --output-name fmg-test-full-chunk-00 `
  --model gemini3.1flash `
  --map-mode scene_mode `
  --fmg-seed 12345 `
  --fmg-chunk-id chunk_0_0 `
  --fmg-chunk-size 4096 `
  --fmg-backend-url http://127.0.0.1:8765
```

输出目录：

```text
SnowWeave\out\maps\fmg-test-full-chunk-00
```

## 使用已有 FMG bundle

如果已经有 `atlas-bundle.zip`，可以跳过 backend 拉取：

```powershell
python SnowWeave\scripts\generate_fmg_chunk_pipeline.py `
  --prompt "top-down clean HD fantasy island RPG exploration basemap, natural terrain, roads, rivers, lakes, cliffs, no buildings" `
  --output-name fmg-test-full-chunk-00 `
  --model gemini3.1flash `
  --map-mode scene_mode `
  --fmg-bundle-zip D:\path\to\atlas-bundle.zip `
  --fmg-chunk-id chunk_0_0
```

## 只解包 FMG 分片

```powershell
python SnowWeave\scripts\fmg_unpack_atlas_bundle.py `
  --backend-url http://127.0.0.1:8765 `
  --seed 12345 `
  --chunk-size 4096 `
  --output-dir SnowWeave\out\fmg_refs\seed-12345 `
  --force
```

生成：

```text
fmg_reference_index.json
atlas.png
manifest.json
chunks\chunk_0_0.png
chunks\chunk_0_0.json
...
```

## 单独重跑 Prop 抠图

Plants：

```powershell
python SnowWeave\dependencies\agent-sprite-forge\skills\generate2dmap\scripts\extract_props_by_subtraction.py `
  --base SnowWeave\out\maps\fmg-test-full-chunk-00\base\base-1.png `
  --dressed SnowWeave\out\maps\fmg-test-full-chunk-00\dressed-plants\dressed-plants-1.png `
  --output-dir SnowWeave\out\maps\fmg-test-full-chunk-00\props\plants `
  --manifest prop-pack.plants.manifest.json `
  --diff-threshold 30 `
  --min-component-area 100 `
  --matting-backend auto
```

Manmade：

```powershell
python SnowWeave\dependencies\agent-sprite-forge\skills\generate2dmap\scripts\extract_props_by_subtraction.py `
  --base SnowWeave\out\maps\fmg-test-full-chunk-00\base\base-1.png `
  --dressed SnowWeave\out\maps\fmg-test-full-chunk-00\dressed-manmade\dressed-manmade-1.png `
  --output-dir SnowWeave\out\maps\fmg-test-full-chunk-00\props\manmade `
  --manifest prop-pack.manmade.manifest.json `
  --diff-threshold 30 `
  --min-component-area 100 `
  --matting-backend auto
```

注意：裸 `extract_props_by_subtraction.py` 只做单次抠图。同类 overlap 合并、分类 manifest 合并和四色 overlay 是完整 ASF pipeline 里的后处理。

## 输出结构

完整 pipeline 输出示例：

```text
fmg-reference\
  fmg_reference_index.json
  chunks\chunk_0_0.png
  chunks\chunk_0_0.json

base\
  base-1.png
  base.prompt.txt
  base.manifest.json
  base.response.json

dressed-plants\
  dressed-plants-1.png
  dressed-plants.prompt.txt

dressed-manmade\
  dressed-manmade-1.png
  dressed-manmade.prompt.txt

props\
  plants\
    prop-pack.plants.manifest.json
    prop-*.png
  manmade\
    prop-pack.manmade.manifest.json
    prop-*.png
  prop-pack.manifest.json

placements.plants.json
placements.manmade.json
placements.json
layered-preview.png
layered-preview.annotated.png
prop-alpha-overlay.png
prop-alpha-overlay.report.json
pipeline_result.json
```

## 关键文件说明

- `base/base.prompt.txt`：basemap 生成 prompt，包含 FMG 图例解释和地形保持约束。
- `dressed-plants/dressed-plants.prompt.txt`：只允许植物/自然有机 props。
- `dressed-manmade/dressed-manmade.prompt.txt`：只允许人造 props。
- `props/*/prop-pack.*.manifest.json`：分类 prop 抠图结果。
- `props/prop-pack.manifest.json`：合并后的 prop manifest。
- `placements.json`：最终 Godot/preview 使用的 prop 放置数据。
- `layered-preview.png`：base + props 合成预览。
- `prop-alpha-overlay.png`：四色半透明 alpha 调试图，相邻 prop 尽量不同色。
- `pipeline_result.json`：API/Godot 读取的最终结果。

## 2x2 分片顺序

后续全量生成使用固定顺序：

```text
chunk_0_0 -> chunk_1_0 -> chunk_0_1 -> chunk_1_1
```

后三张 chunk 生成前，应先用已生成 basemap 替换当前 FMG reference 的重叠区域：

```powershell
python SnowWeave\scripts\fmg_stitch_chunk_reference.py `
  --index SnowWeave\out\fmg_refs\seed-12345\fmg_reference_index.json `
  --chunk-id chunk_1_0 `
  --generated chunk_0_0=SnowWeave\out\maps\chunk_0_0\base\base-1.png `
  --output SnowWeave\out\fmg_refs\seed-12345\chunks\chunk_1_0-reference-stitched.png
```

重叠区是故意设计的，不裁掉。后续 chunk 应使用 stitched reference，让模型沿着已生成 basemap 的风格继续绘制。

