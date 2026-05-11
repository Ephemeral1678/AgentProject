# 具身智能视频生成 - 光流预测方案 (Flow Strategy)

基于光流变形(Flow Warping)的具身智能视频生成模型，结合CLIP文本引导实现文本条件下的机器人未来帧预测。

## 1. 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.8 | 推荐 3.10 |
| PyTorch | >= 2.0 | 需匹配CUDA版本 |
| CUDA | >= 11.8 | 推荐 12.x |
| GPU显存 | >= 16GB | 推荐 24GB+ |
| 操作系统 | Linux | 已在AutoDL测试通过 |

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**: transformers 仅用于 CLIP 文本编码器。如果 CLIP 加载失败（如网络问题），会自动降级为内置的 `SimpleTextEncoder`（BiLSTM），无需额外依赖。

## 3. 数据准备

将数据集放置在以下路径：

```
/root/release/
├── train/              # 训练集 (250条轨迹)
│   ├── 1_1/
│   │   ├── video.mp4
│   │   ├── action.txt
│   │   ├── joint.txt
│   │   └── instruction.txt
│   ├── 1_2/
│   │   └── ...
│   └── ...
└── test/               # 测试集 (100条轨迹)
    ├── 1_1/
    │   ├── video.mp4
    │   ├── action.txt
    │   ├── joint.txt
    │   └── instruction.txt
    ├── 1_2/
    │   └── ...
    └── ...
```

**每条轨迹包含4个文件：**
- `video.mp4`: 输入视频（前16帧作为上下文）
- `action.txt`: 动作数据（26维，CSV格式，含行号列）
- `joint.txt`: 关节数据（26维，CSV格式，含行号列）
- `instruction.txt`: 文本指令

## 4. 模型架构

### 核心模块

| 模块 | 名称 | 架构 |
|------|------|------|
| **Module A** | TextConditionedFlowVideoPredictor | U-Net编码器 + ConvLSTM时序 + 光流解码器 + FiLM文本调制 |
| **Module B** | TextConditionedActionPredictor | CNN视觉编码器 + Transformer Encoder/Decoder + CrossAttention |
| **Module C** | TextConditionedJointPredictor | 双层LSTM + FiLM调制 |

### 文本编码
- **首选**: CLIP ViT-B/32（需transformers库）
- **后备**: SimpleTextEncoder（字符嵌入 + BiLSTM）

### 视频生成原理
采用光流变形策略：
```
输出帧 = mask × warp(当前帧, 光流) + (1 - mask) × 残差
```
自回归逐帧预测50帧，保留纹理清晰度并补偿遮挡区域。

### 训练策略
- **渐进式预测长度**: Epoch 1~8:5帧 → 9~15:10帧 → 16~30:25帧 → 31~80:50帧
- **Scheduled Sampling**: Teacher Forcing比率从0.8逐步降至0（epoch 30后完全自回归）
- **损失函数**: MSE + L1 + SSIM + CLIP对齐损失(epoch>=20启用)
- **分组学习率**: 动作5e-5 / 关节1e-4 / 视频2e-4 / 文本1e-4

## 5. 使用方法

### 5.1 完整流程（训练+推理+评估）

```bash
python train_flow.py --mode full
```

### 5.2 仅训练

```bash
python train_flow.py --mode train
```

### 5.3 从断点续训

```bash
python train_flow.py --mode train --resume
```

### 5.4 推理生成

```bash
# 使用最优模型(best_model.pt)
python train_flow.py --mode infer

# 指定特定checkpoint
python train_flow.py --mode infer --ckpt ckpt_e20.pt

# 自定义输出目录名
python train_flow.py --mode infer --ckpt ckpt_e60.pt --outdir submission_v2
```

### 5.5 仅质量检查（不生成视频）

```bash
python train_flow.py --mode check
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | 运行模式: `full`(全部) / `train`(训练) / `infer`(推理) / `check`(检查) |
| `--ckpt` | None | 指定checkpoint文件(如 `ckpt_e20.pt`)，默认用 `best_model.pt` |
| `--outdir` | None | 输出目录名，默认根据ckpt自动命名或使用 `submission` |
| `--resume` | False | 从上次保存的best_model继续训练 |

## 6. 输出格式

### 训练输出 (`checkpoints/`)

```
checkpoints/
├── best_model.pt        # 最优模型权重(loss最低)
├── ckpt_e20.pt          # 第20轮快照
├── ckpt_e40.pt          # 第40轮快照
├── ckpt_e60.pt          # 第60轮快照
├── ckpt_e80.pt          # 第80轮快照
└── loss_history.json    # 完整训练loss记录
```

### 推理输出 (`submission/`)

```
submission/
├── 1_1/
│   ├── video.mp4         # 50帧预测视频
│   ├── action.txt        # 51行×27列动作数据(1上下文+50预测)
│   ├── joint.txt         # 51行×27列关节数据(1上下文+50预测)
│   └── instruction.txt   # 原始文本指令
├── 1_2/
│   └── ...
└── ... (共100个文件夹)
```

每条轨迹的输出格式：
- **video.mp4**: 包含模型生成的50个预测时间步
- **action.txt**: 51行，第1行为最后一帧上下文，后50行为模型预测
- **joint.txt**: 同action.txt
- **instruction.txt**: 原始文本指令
- 每行27列：第1列为索引号，第2~27列为26维特征值

## 7. 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 256×256 | 平衡质量和显存占用 |
| 上下文帧数 | 16 | 使用输入视频的前16帧 |
| 预测帧数 | 50 | 自回归预测未来50帧 |
| 总训练轮数 | 80 | 可在代码中修改 `NUM_EPOCHS` |
| Batch Size | 1 | 高分辨率下使用单样本训练 |
| Hidden Dim | 512 | Transformer/LSTM隐藏层维度 |
| Text Embed Dim | 512(CLIP) / 256(后备) | 文本嵌入维度 |

## 8. 模型权重获取

### 网盘信息
链接: https://pan.baidu.com/s/1SSbVkL8om9Ie9SMT1clV8g 提取码: nd5z 

### 包含的权重文件

```
checkpoints/
├── best_model.pt        # 最优模型权重 (推荐使用)
├── ckpt_e20.pt          # 第20轮快照
├── ckpt_e40.pt          # 第40轮快照
├── ckpt_e60.pt          # 第60轮快照
├── ckpt_e80.pt          # 第80轮快照
└── loss_history.json    # 完整训练loss记录
```

### 下载后的放置位置

将解压后的 `checkpoints/` 文件夹放在项目根目录下（与 `train_flow.py` 同级），或在运行时通过修改代码中的 `CHECKPOINT_DIR` 变量指定路径。



## 9. 开源模型说明

本项目使用的 **CLIP ViT-B/32** 是 OpenAI 发布的开源视觉-语言模型（来自 HuggingFace `openai/clip-vit-base-patch32`），仅作为**文本编码器**用于将指令文本转换为向量表示。**未对该模型进行任何微调或修改。**

### 各组件来源说明

| 组件 | 来源 |
|------|------|
| CLIP ViT-B/32 | HuggingFace开源预训练 (`openai/clip-vit-base-patch32`) |
| SimpleTextEncoder（后备编码器） | 本项目自行实现 | 
| 光流预测器 (Module A) | 本项目从零构建 (U-Net + ConvLSTM + FiLM) | 
| 动作预测器 (Module B) | 本项目从零构建 (CNN + Transformer) |
| 关节预测器 (Module C) | 本项目从零构建 (LSTM + FiLM) |



## 10. 注意事项

1. **HuggingFace镜像**: 代码默认使用 `hf-mirror.com` 国内镜像下载CLIP模型。如在其他环境运行，可修改 `_HF_MIRROR` 变量。
2. **数据盘检测**: 代码自动检测 `/root/autodl-tmp` 数据盘。如存在，所有输出（模型、结果）优先写入数据盘。
3. **显存需求**: 训练阶段推荐24GB+显存，推理阶段8GB即可。
4. **CLIP模型缓存**: 首次运行会从HuggingFace下载CLIP ViT-B/32模型（约600MB），后续运行直接读取缓存。
5. **Jupyter Notebook**: 如在Jupyter中运行，shell命令需加 `!` 前缀，且不能用 `&&` 连接多条命令。
