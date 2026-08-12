# 工业电装流水线模型收敛评估

评估范围：第三人称固定视角；工业电装生产流水线；工人操作设备、工具、工件、线束、原料、料盒和工位区域；核心任务是目标检测，必要时引入视觉提示或开放词表模型做类别探索、辅助标注和未知实体发现。运行环境按多张 RTX 4090、PyTorch + CUDA 设计。开源许可结论仍建议在生产落地前由项目法务或负责人最终确认。

## 为什么 PET-DINO 不应直接剔除

之前把 PET-DINO 剔除，是基于“固定类别生产检测主链”的保守判断：如果类别已经固定、标注充分、指标要稳定，YOLOX 这类固定类别检测器更适合作为生产主模型。但你补充的前提改变了判断：

- 你的场景中有些实体名称未必一开始就能确定，例如某些工装、夹具、线束、半成品、原料盒和局部设备区域。
- 第三人称固定视角让视觉提示更可行：可以从同一机位截取稳定样例，用“给样例找同类”的方式减少文本类名不准带来的漂移。
- 视觉提示模型适合做数据建设阶段的少样本召回、相似物发现和辅助标注，可以帮你更快形成最终固定类别训练集。

所以 PET-DINO 不应进入生产主链的首选，但应保留为探索链路：用少量视觉样例发现同类目标，再人工复核并沉淀成 YOLOX 训练数据。

## 推荐架构

```text
开放探索层：Grounding DINO / YOLO-World / PET-DINO
        ↓ 人工复核与类别收敛
生产检测层：YOLOX 固定类别检测器
        ↓ 可选
时序关联层：ByteTrack
        ↓
规则与事件层：操作顺序、区域进入、物料拿取、工具使用、漏装/误放等
```

这里的“串联”建议理解为任务链串联，而不是把多个检测器无条件逐帧堆叠。生产告警链路应尽量稳定；开放词表和视觉提示模型更适合在离线探索、旁路复核、未知实体召回或低频抽帧检查中使用。

## 保留模型

### 1. YOLOX + ByteTrack：生产主线

- YOLOX 是 PyTorch 检测器，适合在固定机位、固定或半固定类别上用现场数据训练。
- ByteTrack 适合在检测器后串联，用于轨迹、计数、短时漏检平滑和操作过程关联。
- 对多卡 4090，训练可用 DDP；推理更建议按摄像头或视频分配 GPU，而不是把单路实时流拆成跨卡逐帧通信。
- 首期类别建议从可观察对象定义：`worker`、`hand`、`tool`、`part`、`wire_harness`、`material_box`、`equipment_operation_area`。像“正在操作”“漏装”“误放”更适合放到事件层，不要只靠单个框标签表达。

### 2. YOLO-World：开放词表实体探索与未知对象旁路

- YOLO-World 对你的场景是有价值的，尤其当你还不能准确命名实体，或者同一对象在不同工位有不同叫法时。
- 推荐用途：离线抽帧检测、候选类名探索、未知目标发现、标注前预检、生产链路的低频旁路复核。
- 工程风险：官方依赖 MMYOLO/MMDetection，部署面比 YOLOX 更大；阈值和 prompt 设计需要现场验证。
- 不建议作为第一版生产逐帧唯一主模型：开放词表输出会受 prompt、同义词和阈值影响，告警验收线更难稳定。

### 3. PET-DINO：视觉提示/少样本同类召回

- PET-DINO 的价值在于视觉提示，而不是文本提示。它适合用少量样例图提示模型寻找同类物体。
- 推荐用途：工装、夹具、原料、半成品、线束局部等难命名或形态相近目标的探索；辅助标注；新物料上线时的快速召回。
- 工程风险：研究实现较新，训练/部署链路成熟度需要实测；不建议首期承担生产实时主链 SLA。
- 落地方式：把视觉样例放到 `data/raw/` 或远端数据目录下的 prompt/support 子目录，作为离线探索任务输入；输出候选框后人工复核，再进入固定类别训练集。

### 4. Grounding DINO：文本提示辅助标注

- 代码和可下载权重成熟，适合用文本 prompt 先扫一批候选目标。
- 推荐用途：类别探索、半自动标注、和 YOLO-World/PET-DINO 的结果互相对照。
- 不建议作为每帧生产主模型：固定类别生产检测没有必要长期支付文本融合开销，开放词汇输出也更难直接绑定稳定告警阈值。

### 5. OmDet-Turbo：实时开放词表备选

- 有 PyTorch、ONNX 和 Apache-2.0 路线，可作为开放词表方向的备选。
- 只有当 YOLO-World 在你的 4090 环境中速度或效果不满足时，再纳入同场景实测。

### 6. SAM 2 / Grounded SAM 2：像素级支线

- 当前任务是目标检测，不需要首期引入像素级分割。
- 如果后续需要精细遮挡轮廓、抓取区域、装配区域边界或辅助标注，再启用。

## 暂不纳入首期

- **LocateAnything-3B/GGUF**：模型条款偏研究用途，且 3B 级别不适合作为固定类别逐帧检测链路。
- **Grounding DINO 1.5、DINO-X、T-Rex2**：公开使用形态更偏 API 或平台能力，不等于完整本地权重、私有化训练和内网部署链。
- **HeadCLIP、AnomalyCLIP、AdaCLIP、VCP-CLIP、FE-CLIP、DLVP-CLIP、FB-CLIP、MoECLIP、AnomalyVFM**：核心更偏异常/缺陷分数或分割，不是当前“人员、设备、工具、原料目标检测”的主任务。
- **Florence-2**：多任务能力强，但 OCR、描述、通用视觉问答会增加推理成本；当前边界框检测首期用不到。
- **GLIP/GLIPv2、Detic、RegionCLIP、ViLD、F-VLM、CoDet、CLIPSelf、X-Decoder/OpenSeeD、OWLv2**：多数属于较老、研究型或重依赖开放词汇路线；在 YOLO-World、Grounding DINO、PET-DINO 已覆盖探索需求后，首期增益不明显。
- **PP-YOLOE+/PP-PicoDet**：成熟且许可友好，但你的环境是 PyTorch + CUDA；引入 PaddlePaddle 会增加第二套训练/部署栈。
- **Ultralytics YOLOE/YOLOE-26**：工程可用，但 AGPL-3.0/企业双许可需要先确认闭源工业系统的合规边界。

## 实测验收建议

请把测试数据按摄像头、日期、班次、产线拆分，避免相邻视频帧跨训练/验证/测试集泄漏。除 mAP 外，建议记录：

- 关键类别召回率，尤其是手、工具、小零件、线束和料盒。
- 每小时误报次数和漏报样例。
- 小目标召回、遮挡召回、不同光照/班次稳定性。
- 端到端 FPS、P95 延迟、显存占用和连续运行稳定性。
- 开放词表/视觉提示模型输出进入人工复核后的有效标注率。

## 当前推荐优先级

1. 用 Grounding DINO、YOLO-World、PET-DINO 对 `data/raw/` 中的抽帧做探索，收敛类别与标注规则。
2. 人工复核后形成固定类别数据集。
3. 训练 YOLOX-S/M/L，对比速度和精度。
4. 按需要串联 ByteTrack 与事件规则。
5. 保留 YOLO-World/PET-DINO 作为未知对象发现和新物料上线时的旁路工具。

## 主要官方来源

- YOLOX / ByteTrack: https://github.com/Megvii-BaseDetection/YOLOX and https://github.com/FoundationVision/ByteTrack
- YOLO-World: https://github.com/AILab-CVC/YOLO-World
- PET-DINO: https://github.com/IDEA-Research/PET-DINO
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- SAM 2: https://github.com/facebookresearch/sam2
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- OmDet-Turbo: https://github.com/om-ai-lab/OmDet
- Ultralytics license: https://github.com/ultralytics/ultralytics
- LocateAnything model terms: https://huggingface.co/nvidia/LocateAnything-3B
- PaddleDetection: https://github.com/PaddlePaddle/PaddleDetection
