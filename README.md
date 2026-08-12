# awesome-llm-apps

一个轻量、配置驱动的工业视频检测工程骨架。目标部署拓扑为“执行机 + 执行机可访问的训练/部署服务器”；本机不连接 MCP。当前场景聚焦固定第三人称机位下的电装生产流水线目标检测。

## 结构

```text
src/awesome_llm_apps/   核心编排、MCP 客户端、图片/视频推理
scripts/                环境检查和运行入口
configs/                不含密钥的环境与流水线配置
data/                   本地数据挂载点（内容默认不入 Git）
artifacts/              运行产物（默认不入 Git）
tests/                  最小单元测试
docs/                   接入信息清单与协议约定
```

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
$env:INTRANET_MCP_URL = "https://你的内网服务/mcp"
$env:INTRANET_MCP_TOKEN = "你的令牌"
llm-pipeline validate -c configs/pipeline.example.toml
llm-pipeline plan -c configs/pipeline.example.toml
llm-pipeline run -c configs/pipeline.example.toml
```

`plan` 不连接远端；`run` 才会发起 MCP 请求。当前执行顺序为：下载各模型 → 训练需要训练的模型 → 部署 → 组合模型 → 处理数据集。失败时立即停止，避免在未知状态下继续部署。

## 安全边界

- URL 和令牌只从环境变量读取，不进入版本库。
- 配置中的本地数据路径仅作为传给远端 MCP 的参数；本项目不会自动上传数据。
- 首次接入建议先运行 `tools` 查看远端工具，再用 `plan` 核对参数。
- 服务端工具名和参数结构可在 TOML 中替换，核心代码无需修改。

接入前需要提供的信息见 [docs/INTEGRATION_CHECKLIST.md](docs/INTEGRATION_CHECKLIST.md)。

模型筛选结论见 [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)。生产推荐链为固定类别检测器（优先 YOLOX）→ 可选 ByteTrack → 事件规则；Grounding DINO 只用于离线类别探索和辅助标注。

## 图片与视频推理

开发探索配置使用 Grounding DINO：

```powershell
pip install -e ".[inference]"
infer-image --config configs/inference.example.toml --input data/raw/example.jpg
infer-video --config configs/inference.example.toml --input data/raw/example.mp4 --frame-stride 2
```

生产检测器应导出为 TorchScript，并返回 `N x 6` 张量：`x1,y1,x2,y2,score,class_id`。然后修改 `configs/inference.production.example.toml` 中的模型与标签路径：

```powershell
infer-video --config configs/inference.production.example.toml --input /data/line-01.mp4
```

单进程会输出 JSONL 和标注后 MP4。多卡用于批量视频/摄像头分片：

```bash
torchrun --standalone --nproc-per-node=4 scripts/infer_video.py \
  --config configs/inference.production.example.toml \
  --input /data/line-01.mp4
```

多卡模式按采样帧分片并分别输出 `rank-XX.jsonl`，避免多个进程竞争同一个视频写入器。生产实时流更推荐按摄像头固定分配 GPU，而不是拆散单路帧序列。
