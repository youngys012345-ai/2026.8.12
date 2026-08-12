# 工业电装流水线模型收敛评估

评估范围：第三人称固定视角；工人操作设备、工具、工件、线束和原料；目标是边界框检测，测试数据由使用方提供；运行环境为多张 RTX 4090、PyTorch + CUDA。本文把“代码公开、权重公开、许可可接受、环境可维护”分别判断，开源许可结论仍应由项目法务最终确认。

## 推荐架构

```text
Grounding DINO（离线类别探索/辅助标注）
                  ↓ 人工复核形成固定类别数据集
YOLOX 或已取得企业授权的 YOLOE（生产检测） → ByteTrack（可选时序关联） → 规则/事件层
```

固定机位使背景、拍摄距离和目标集合相对稳定。生产阶段使用经过现场数据训练的固定类别检测器，通常比每帧运行开放词汇模型更快、更稳定，也更容易定义漏检率、误报率和延迟验收线。这里的“串联”应是检测 → 跟踪 → 事件规则，而不是把多个检测器逐帧重复串联。

## 保留

### 1. YOLOX + ByteTrack：生产主线首选

- YOLOX 是 PyTorch 检测器；ByteTrack 官方实现与 YOLOX 生态直接配套，适合固定类别视频目标检测和轨迹连续化。
- YOLOX **代码仓库**明确采用 Apache-2.0；官方预训练权重是否自动适用同一许可目前缺少足够明确的单独声明。生产项目应优先用自有现场数据训练最终权重，并保存所有初始化权重的来源、版本和许可快照。
- 对多卡 4090，可用 DDP 训练；推理端通常按摄像头/视频分卡，避免逐帧跨卡通信。需要单路极致性能时再导出 TensorRT。
- 首期类别建议围绕可观察对象定义，例如 `worker`、设备操作区、工具、物料箱、线束/工件；“正在操作”属于人与设备/物料的时空关系，应放到后续事件层，不能只靠目标框标签可靠表达。
- 风险：必须使用现场数据重训；小零件和遮挡需要高分辨率、切片或专门的小目标策略。

### 2. Grounding DINO：只用于辅助标注和类别探索

- 官方代码和可下载权重均存在，Apache-2.0；Hugging Face Transformers 也有实现，迁移成本较低。
- 适合在数据准备阶段用文本提示召回未知或新增物料，再由人工复核。
- 不建议作为每帧生产主模型：固定类别场景没有必要持续支付文本编码/融合开销，且开放词汇输出的阈值和同义词漂移不利于稳定告警。

### 3. OmDet-Turbo：实时开放词汇备选，不作为首选训练主线

- 官方 PyTorch 仓库、Tiny 权重、ONNX 导出和 Apache-2.0 许可均可获取。
- 当类别经常变化且必须在线文本提示时值得实测；附件中的 100.2 FPS 是 A100/TensorRT/特定条件结果，不能外推为 4090 端到端视频 FPS。
- 官方工程更偏预训练权重推理和导出；对于现场固定类别训练，YOLOX 的工程路径更直接。

### 4. SAM 2 / Grounded SAM 2：按需保留

- 代码、训练代码和检查点可获取，SAM 2 主体为 Apache-2.0。
- 只有在需要像素级区域、精细遮挡轮廓或辅助标注时才加入；纯边界框检测首期不需要。

## 从首期剔除

- **LocateAnything-3B/GGUF**：模型明确限非商业研究；3B 模型也不适合该固定类别逐帧链路。
- **Grounding DINO 1.5、DINO-X、T-Rex2**：公开项目主要是 API 调用链，不等于完整本地权重和私有化训练部署链；与内网数据边界冲突。
- **PET-DINO**：视觉示例提示解决“给样例找同类”，不是固定类别生产检测的优先问题；属于很新的研究实现，先不承担主线风险。
- **HeadCLIP、AnomalyCLIP、AdaCLIP、VCP-CLIP、FE-CLIP、DLVP-CLIP、FB-CLIP、MoECLIP、AnomalyVFM**：核心任务是异常/缺陷分数或掩码，而当前需求是人员、设备、原料等对象检测。未来若明确增加表面缺陷检测，应作为独立支线评估；其中 AnomalyCLIP 的代码和权重可获取，但不改变任务不匹配的问题。
- **Florence-2**：OCR、描述、多任务能力会增加推理成本，当前边界框任务用不到。
- **GLIP/GLIPv2、Detic、RegionCLIP、ViLD、F-VLM、CoDet、CLIPSelf、X-Decoder/OpenSeeD、OWLv2**：能获取其中不少代码/权重，但属于较老、研究型或重依赖栈的开放词汇路线；在已有 Grounding DINO 辅助标注 + 固定类生产检测器的情况下没有首期增益。
- **PP-YOLOE+/PP-PicoDet**：本身成熟且 Apache-2.0，但核心环境指定 PyTorch；引入 PaddlePaddle 会形成第二套 CUDA/训练/部署依赖。仅在后续端侧设备明确采用 Paddle Inference 时重评。
- **YOLO-World**：开放词汇能力对固定类别生产推理不是必需；官方工程依赖 MMYOLO/MMDetection，维护面更大。
- **Ultralytics YOLOE/YOLOE-26**：工程可部署，但 Ultralytics 当前采用 AGPL-3.0/企业双许可。工业闭源系统应先确认义务或取得企业许可，不能按附件所述无条件商用。
- **PP-PicoDet/YOLOE Tiny 的无 GPU 方案**：目标环境是多卡 4090，与实际部署条件不符。

## 实测验收建议

用你提供的数据按摄像头、日期和产线划分训练/验证/测试，避免相邻视频帧跨集合泄漏。除 mAP 外必须分别记录关键类别召回率、每小时误报次数、小目标召回、遮挡召回、端到端 FPS/P95 延迟、显存和连续运行稳定性。第一轮建议比较 `YOLOX-S/M/L` 的速度—精度曲线，再决定是否启用切片、高分辨率或 ByteTrack。

## 主要官方来源

- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- SAM 2: https://github.com/facebookresearch/sam2
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- OmDet-Turbo: https://github.com/om-ai-lab/OmDet
- YOLOX / ByteTrack: https://github.com/Megvii-BaseDetection/YOLOX and https://github.com/FoundationVision/ByteTrack
- Ultralytics license: https://github.com/ultralytics/ultralytics
- LocateAnything model terms: https://huggingface.co/nvidia/LocateAnything-3B
- AnomalyCLIP: https://github.com/zqhang/AnomalyCLIP
- PaddleDetection: https://github.com/PaddlePaddle/PaddleDetection
