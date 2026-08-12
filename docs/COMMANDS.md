# 项目命令手册

这份文档把常用命令集中放在一起。默认约定：测试图片和视频放在 `data/raw/`，运行产物写入 `artifacts/`，统一模型参数配置优先使用 `configs/model_params.example.json`。迁移到另一台内网机器后，可以复制为 `configs/model_params.json` 再按现场路径修改。

## 1. 环境初始化

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,inference]"
```

Linux/内网训练机：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,inference]"
```

如果只运行 MCP 编排，不做本地推理，可以先安装：

```bash
python -m pip install -e ".[dev]"
```

## 2. 配置 MCP 环境变量

本机不需要配置 MCP。到另一台能访问内网 MCP 的机器后再设置：

```bash
export INTRANET_MCP_URL="https://your-intranet-mcp-server/mcp"
export INTRANET_MCP_TOKEN="your-token"
```

PowerShell：

```powershell
$env:INTRANET_MCP_URL = "https://your-intranet-mcp-server/mcp"
$env:INTRANET_MCP_TOKEN = "your-token"
```

## 3. 检查与规划远程训练/部署

只校验配置，不连接远程：

```bash
llm-pipeline validate -c configs/model_params.example.json
```

只生成执行计划，不连接远程：

```bash
llm-pipeline plan -c configs/model_params.example.json
```

查看远程 MCP 已暴露的工具：

```bash
llm-pipeline tools -c configs/model_params.example.json
```

执行下载、训练、部署、组合和数据处理：

```bash
llm-pipeline run -c configs/model_params.example.json
```

## 4. 图片推理

把测试图片放到 `data/raw/`，例如 `data/raw/example.jpg`：

```bash
infer-image --config configs/model_params.example.json --input data/raw/example.jpg --output-dir artifacts/inference
```

输出：

- `artifacts/inference/example.json`
- `artifacts/inference/example.annotated.jpg`

## 5. 视频推理

把测试视频放到 `data/raw/`，例如 `data/raw/example.mp4`：

```bash
infer-video --config configs/model_params.example.json --input data/raw/example.mp4 --output-dir artifacts/inference --frame-stride 2
```

输出：

- `artifacts/inference/example.rank-00.jsonl`
- `artifacts/inference/example.annotated.mp4`

`--frame-stride 2` 表示每 2 帧推理 1 帧；如果要逐帧推理，设为 `1`。

## 6. 多卡视频批处理

多卡建议按摄像头或视频分配 GPU。对单个长视频做采样帧分片时可以用：

```bash
torchrun --standalone --nproc-per-node=4 scripts/infer_video.py \
  --config configs/model_params.example.json \
  --input data/raw/example.mp4 \
  --output-dir artifacts/inference \
  --frame-stride 2
```

多进程模式会分别输出 `rank-00.jsonl`、`rank-01.jsonl` 等文件；为避免多个进程竞争视频写入器，多卡模式默认不合并标注视频。

## 7. 生产检测器配置

`configs/model_params.example.json` 当前默认用 Grounding DINO 做开放词表探索推理，适合类别探索和半自动标注。等 YOLOX 训练并导出 TorchScript 后，把 `stages` 改成：

```json
[
  {
    "type": "torchscript_detector",
    "name": "yolox_production",
    "model_path": "/models/assembly-line-detector/model.ts",
    "labels_path": "/models/assembly-line-detector/labels.txt",
    "input_size": [960, 960],
    "confidence_threshold": 0.35
  }
]
```

如果仍需要不确定实体探索，保留 YOLO-World 或 Grounding DINO 作为旁路/回退模型，不建议让开放词表模型无条件参与每帧生产告警。
