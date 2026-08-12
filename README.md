# industrial-video-detection

一个轻量、配置驱动的工业视频目标检测工程骨架。目标部署拓扑为“执行机器 + 执行机器可访问的内网训练/部署服务器”；本机不连接 MCP。当前场景聚焦第三人称固定机位下的电装生产流水线目标检测，覆盖工人、设备操作区、工具、工件、线束、原料和料盒等对象。

远程 GitHub 仓库名是 `2026.8.12`；本项目内部包名使用 `industrial_video_detection`，避免继续沿用无关的 `awesome_llm_apps`。

## 结构

```text
src/industrial_video_detection/   核心编排、MCP 客户端、图片/视频推理
scripts/                          环境检查和运行入口
configs/                          不含密钥的模型、部署、推理配置
data/                             测试数据挂载点，内容默认不入 Git
artifacts/                        运行产物，默认不入 Git
tests/                            最小单元测试
docs/                             模型评估、命令手册和接入信息清单
```

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,inference]"
ivd-pipeline validate -c configs/model_params.example.json
ivd-pipeline plan -c configs/model_params.example.json
```

`plan` 不连接远端；`run` 才会发起 MCP 请求。迁移到另一台内网机器后再设置：

```powershell
$env:INTRANET_MCP_URL = "https://你的内网服务/mcp"
$env:INTRANET_MCP_TOKEN = "你的令牌"
ivd-pipeline tools -c configs/model_params.example.json
ivd-pipeline run -c configs/model_params.example.json
```

`llm-pipeline` 作为兼容别名暂时保留。

## 模型路线

模型筛选结论见 [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)。当前推荐路线：

```text
零样本/开放探索层：Grounding DINO / OWLv2 / YOLO-World / PET-DINO
        ↓ 人工复核与类别收敛
生产检测层：YOLOX 固定类别检测器
        ↓ 可选 ByteTrack
事件规则层：操作顺序、区域进入、物料拿取、漏装/误放等业务逻辑
```

零样本目标检测模型已经纳入测试清单。YOLO-World 和 OWLv2 用于文本零样本/开放词表测试；PET-DINO 用于视觉提示/少样本同类召回；Grounding DINO 用于文本提示辅助标注。生产主链仍建议优先用现场数据训练后的 YOLOX。

## 统一配置

模型下载、训练、部署、组合、数据集路径和推理参数统一放在：

```text
configs/model_params.example.json
```

零样本文本检测本地测试配置放在：

```text
configs/zero_shot.example.json
```

建议在目标机器上复制为 `configs/model_params.json` 后修改现场路径、类别、阈值、GPU 数量和导出位置。旧的 TOML 示例仍保留，方便单独测试。

## 图片与视频推理

测试数据放在 `data/raw/`。命令集中在 [docs/COMMANDS.md](docs/COMMANDS.md)。

```powershell
infer-image --config configs/model_params.example.json --input data/raw/example.jpg --output-dir artifacts/inference
infer-video --config configs/model_params.example.json --input data/raw/example.mp4 --output-dir artifacts/inference --frame-stride 2
```

零样本基线测试：

```powershell
infer-image --config configs/zero_shot.example.json --input data/raw/example.jpg --output-dir artifacts/zero-shot
infer-video --config configs/zero_shot.example.json --input data/raw/example.mp4 --output-dir artifacts/zero-shot --frame-stride 5
```

生产检测器应导出为 TorchScript，并返回 `N x 6` 张量：`x1,y1,x2,y2,score,class_id`。然后把 `configs/model_params.example.json` 中的 `stages` 改为 `torchscript_detector`，指向导出的模型和标签文件。

## 安全边界

- URL 和令牌只从环境变量读取，不进入版本库。
- `data/` 和 `artifacts/` 中的大文件默认不入 Git。
- 配置中的本地数据路径只作为参数传给远端 MCP；项目不会自动上传数据。
- 首次接入建议先运行 `ivd-pipeline tools` 查看远端工具，再用 `ivd-pipeline plan` 核对参数。
