# awesome-llm-apps

一个轻量、配置驱动的工业视频检测工程骨架。目标部署拓扑为“执行机器 + 执行机器可访问的内网训练/部署服务器”；本机不连接 MCP。当前场景聚焦第三人称固定机位下的电装生产流水线目标检测，覆盖工人、设备操作区、工具、工件、线束、原料和料盒等对象。

## 结构

```text
src/awesome_llm_apps/   核心编排、MCP 客户端、图片/视频推理
scripts/                环境检查和运行入口
configs/                不含密钥的模型、部署、推理配置
data/                   测试数据挂载点，内容默认不入 Git
artifacts/              运行产物，默认不入 Git
tests/                  最小单元测试
docs/                   模型评估、命令手册和接入信息清单
```

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,inference]"
llm-pipeline validate -c configs/model_params.example.json
llm-pipeline plan -c configs/model_params.example.json
```

`plan` 不连接远端；`run` 才会发起 MCP 请求。迁移到另一台内网机器后再设置：

```powershell
$env:INTRANET_MCP_URL = "https://你的内网服务/mcp"
$env:INTRANET_MCP_TOKEN = "你的令牌"
llm-pipeline tools -c configs/model_params.example.json
llm-pipeline run -c configs/model_params.example.json
```

## 模型路线

模型筛选结论见 [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)。当前推荐路线：

```text
Grounding DINO / YOLO-World / PET-DINO 进行类别探索、未知对象发现、视觉提示辅助标注
        ↓ 人工复核与类别收敛
YOLOX 固定类别检测器作为生产主链
        ↓ 可选 ByteTrack
事件规则层处理操作顺序、区域进入、物料拿取、漏装/误放等业务逻辑
```

PET-DINO 和 YOLO-World 已保留在方案中，但定位为探索、旁路和辅助标注，不建议首期无条件放进逐帧生产告警主链。

## 统一配置

模型下载、训练、部署、组合、数据集路径和推理参数统一放在：

```text
configs/model_params.example.json
```

建议在目标机器上复制为 `configs/model_params.json` 后修改现场路径、类别、阈值、GPU 数量和导出位置。旧的 TOML 示例仍保留，方便单独测试。

## 图片与视频推理

测试数据放在 `data/raw/`。命令集中在 [docs/COMMANDS.md](docs/COMMANDS.md)。

```powershell
infer-image --config configs/model_params.example.json --input data/raw/example.jpg --output-dir artifacts/inference
infer-video --config configs/model_params.example.json --input data/raw/example.mp4 --output-dir artifacts/inference --frame-stride 2
```

生产检测器应导出为 TorchScript，并返回 `N x 6` 张量：`x1,y1,x2,y2,score,class_id`。然后把 `configs/model_params.example.json` 中的 `stages` 改为 `torchscript_detector`，指向导出的模型和标签文件。

## 安全边界

- URL 和令牌只从环境变量读取，不进入版本库。
- `data/` 和 `artifacts/` 中的大文件默认不入 Git。
- 配置中的本地数据路径只作为参数传给远端 MCP；项目不会自动上传数据。
- 首次接入建议先运行 `llm-pipeline tools` 查看远端工具，再用 `llm-pipeline plan` 核对参数。
