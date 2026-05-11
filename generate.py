#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具身智能视频生成 - 光流预测方案 (Flow Strategy)
=====================================================
基于光流变形(Flow Warping)的视频生成 + CLIP文本引导

核心创新:
  - 预测光流场(optical flow)而非直接生成像素
  - 用光流对前一帧进行warp变形,保留纹理清晰度
  - 残差(residual)补偿遮挡区域
  - 遮挡掩码(occlusion mask)混合变形帧和残差
  - 文本条件通过FiLM/CrossAttention注入所有模块

架构:
  Module A: TextConditionedFlowVideoPredictor  (U-Net+ConvLSTM+光流warp)
  Module B: TextConditionedActionPredictor     (CNN+Transformer+FiLM)
  Module C: TextConditionedJointPredictor      (LSTM+FiLM)
  文本编码: CLIP ViT-B/32 (或后备SimpleTextEncoder)

输出(每条轨迹4个文件):
  video.mp4 / action.txt / joint.txt / instruction.txt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import math
import json

warnings.filterwarnings("ignore")

# ============================================================
# Part 0: 环境设置 & HuggingFace镜像
# ============================================================

# HuggingFace 镜像 (必须在import transformers之前!)
_HF_MIRROR = "https://hf-mirror.com"
os.environ['HF_ENDPOINT'] = _HF_MIRROR
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = _HF_MIRROR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# 数据盘检测
_DATA_DISK = "/root/autodl-tmp"
if os.path.isdir(_DATA_DISK):
    _HF_CACHE = os.path.join(_DATA_DISK, "cache", "huggingface")
    os.environ['HF_HOME'] = _HF_CACHE
    os.environ['HUGGINGFACE_HUB_CACHE'] = _HF_CACHE
    os.environ['TRANSFORMERS_CACHE'] = _HF_CACHE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
TRAIN_DIR = "/root/release/train"
TEST_DIR = "/root/release/test"
# 优先使用数据盘
SUBMISSION_DIR = os.path.join(_DATA_DISK, "submission") if os.path.isdir(_DATA_DISK) else "/root/submission"
CHECKPOINT_DIR = os.path.join(_DATA_DISK, "checkpoints") if os.path.isdir(_DATA_DISK) else "/root/checkpoints"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== 核心参数 =====
IMG_H, IMG_W = 256, 256          # 训练分辨率(平衡质量和显存)
NUM_CONTEXT = 16                  # 上下文帧数
NUM_PREDICT = 50                  # 预测帧数
ACTION_DIM = 26                   # 动作维度
JOINT_DIM = 26                    # 关节维度
HIDDEN_DIM = 512                  # Transformer隐藏维度
VISUAL_FEAT_DIM = 256             # 视觉特征维度
TEXT_EMBED_DIM = 512              # 文本嵌入维度(CLIP=512, 后备=256)
NUM_EPOCHS = 80                   # 总训练轮次
BATCH_SIZE = 1                    # 批大小(高分辨率下用1)

print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    print(f"  显存: {getattr(p, 'total_memory', getattr(p, 'total_mem', 0)) / 1024**3:.1f} GB")


# ============================================================
# Part 1: CLIP 文本编码器加载
# ============================================================

USE_CLIP = False
clip_tokenizer = None
clip_text_model = None
clip_full_model = None

try:
    from transformers import CLIPTokenizer, CLIPModel
    print("\n[Part1] 加载 CLIP ViT-B/32 ...")
    clip_full_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_full_model.eval()
    for param in clip_full_model.parameters():
        param.requires_grad = False
    TEXT_EMBED_DIM = 512
    USE_CLIP = True
    print(f"[Part1] CLIP 加载成功!")
except Exception as e:
    print(f"\n[Part1] CLIP 加载失败: {e}")
    print(f"        使用后备 SimpleTextEncoder (TEXT_EMBED_DIM=256)")
    TEXT_EMBED_DIM = 256


class SimpleTextEncoder(nn.Module):
    """后备文本编码器: 字符嵌入 + BiLSTM"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.char_embed = nn.Embedding(256, 64)
        self.lstm = nn.LSTM(64, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, text_list):
        max_len = 128
        batch_ids = []
        for text in text_list:
            ids = [ord(c) % 256 for c in text[:max_len]]
            ids += [0] * (max_len - len(ids))
            batch_ids.append(ids)
        x = torch.tensor(batch_ids, dtype=torch.long, device=self.char_embed.weight.device)
        embed = self.char_embed(x)
        _, (h, _) = self.lstm(embed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


simple_text_encoder = None
if not USE_CLIP:
    simple_text_encoder = SimpleTextEncoder(output_dim=TEXT_EMBED_DIM).to(device)


def get_text_embedding(text_list):
    """获取文本嵌入向量"""
    if USE_CLIP:
        with torch.no_grad():
            inputs = clip_tokenizer(text_list, padding=True, truncation=True,
                                    max_length=77, return_tensors="pt").to(device)
            outputs = clip_full_model.text_model(**inputs)
            projected = clip_full_model.text_projection(outputs.pooler_output)
            return F.normalize(projected, dim=-1)
    else:
        return simple_text_encoder(text_list)


def get_clip_image_embedding(images):
    """获取CLIP图像嵌入(用于CLIP对齐损失)"""
    if not USE_CLIP:
        return None
    with torch.no_grad():
        imgs = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).reshape(1, 3, 1, 1)
        imgs = (imgs - mean) / std
        vis_out = clip_full_model.vision_model(pixel_values=imgs)
        projected = clip_full_model.visual_projection(vis_out.pooler_output)
        return F.normalize(projected, dim=-1)


print(f"[Part1] 文本编码就绪 | dim={TEXT_EMBED_DIM} | CLIP={USE_CLIP}")


# ============================================================
# Part 2: 数据工具函数
# ============================================================

def read_video_frames(video_path, num_frames=None, resize=(IMG_H, IMG_W)):
    """读取视频帧并resize到指定大小, 返回 [T, C, H, W] tensor (归一化到[0,1])"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if num_frames is not None and idx >= num_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, (resize[1], resize[0]))
        frame = frame.astype(np.float32) / 255.0
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # [3, H, W]
        idx += 1
    cap.release()
    if len(frames) == 0:
        return torch.zeros(1, 3, resize[0], resize[1])
    return torch.stack(frames)


def read_video_frames_original(video_path, num_frames=None):
    """读取原始分辨率的视频帧(BGR格式), 用于输出时保持原始分辨率"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if num_frames is not None and idx >= num_frames:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def read_csv_data(csv_path):
    """读取action/joint CSV, 去掉第一列索引, 返回 float32"""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 2 and data.shape[1] > 26:
        data = data[:, 1:]  # 去掉Unnamed列
    return data.astype(np.float32)


def read_csv_raw(csv_path):
    """读取原始CSV(含索引列)"""
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)


def read_instruction(txt_path):
    """读取指令文本"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_csv_header():
    """构建CSV表头"""
    cols = ["Unnamed: 0",
            "idx13_left_arm_joint1_position", "idx14_left_arm_joint2_position",
            "idx15_left_arm_joint3_position", "idx16_left_arm_joint4_position",
            "idx17_left_arm_joint5_position", "idx18_left_arm_joint6_position",
            "idx19_left_arm_joint7_position",
            "idx20_right_arm_joint1_position", "idx21_right_arm_joint2_position",
            "idx22_right_arm_joint3_position", "idx23_right_arm_joint4_position",
            "idx24_right_arm_joint5_position", "idx25_right_arm_joint6_position",
            "idx26_right_arm_joint7_position",
            "left_thumb_0_position", "left_thumb_1_position",
            "left_index_position", "left_middle_position",
            "left_ring_position", "left_pinky_position",
            "right_thumb_0_position", "right_thumb_1_position",
            "right_index_position", "right_middle_position",
            "right_ring_position", "right_pinky_position"]
    return ",".join(cols)


def save_csv_with_header(filepath, data, header):
    """带表头保存CSV"""
    with open(filepath, 'w', newline='') as f:
        f.write(header + '\n')
        for row in data:
            row_str = str(int(row[0]))
            for val in row[1:]:
                row_str += f",{val:.6f}"
            f.write(row_str + '\n')


print("[Part2] 工具函数就绪")


# ============================================================
# Part 3: 数据集
# ============================================================

class EmbodiedTrainDataset(Dataset):
    """训练数据集: 返回上下文帧+目标帧+动作+关节+文本"""

    def __init__(self, root_dir, augment=True):
        self.root = root_dir
        self.augment = augment
        self.samples = []
        for name in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "video.mp4")):
                # 检查必要文件
                vp = os.path.join(path, "video.mp4")
                ap = os.path.join(path, "action.txt")
                jp = os.path.join(path, "joint.txt")
                ip = os.path.join(path, "instruction.txt")
                if all(os.path.exists(x) for x in [vp, ap, jp, ip]):
                    self.samples.append((path, name))
        print(f"[Part3] 训练轨迹: {len(self.samples)} 条 | resize=({IMG_H},{IMG_W})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, traj_name = self.samples[idx]
        total_needed = NUM_CONTEXT + NUM_PREDICT

        # 读取视频
        all_frames = read_video_frames(
            os.path.join(traj_path, "video.mp4"),
            num_frames=total_needed, resize=(IMG_H, IMG_W)
        )
        action = read_csv_data(os.path.join(traj_path, "action.txt"))
        joint = read_csv_data(os.path.join(traj_path, "joint.txt"))
        text = read_instruction(os.path.join(traj_path, "instruction.txt"))

        T = all_frames.shape[0]
        # 补齐不足的帧
        if T < total_needed:
            pad_frames = all_frames[-1:].expand(total_needed - T, -1, -1, -1)
            all_frames = torch.cat([all_frames, pad_frames], dim=0)
        if action.shape[0] < total_needed:
            action = np.concatenate([action, np.tile(action[-1:], (total_needed - action.shape[0], 1))])
        if joint.shape[0] < total_needed:
            joint = np.concatenate([joint, np.tile(joint[-1:], (total_needed - joint.shape[0], 1))])

        # 数据增强
        if self.augment:
            if np.random.random() < 0.3:
                all_frames = torch.clamp(all_frames * (0.9 + np.random.random() * 0.2), 0, 1)
            if np.random.random() < 0.15:
                all_frames = torch.clamp(all_frames + torch.randn_like(all_frames) * 0.01, 0, 1)

        return {
            'context_frames': all_frames[:NUM_CONTEXT],           # [16, 3, H, W]
            'target_frames': all_frames[NUM_CONTEXT:total_needed], # [50, 3, H, W]
            'context_action': torch.from_numpy(action[:NUM_CONTEXT]),
            'target_action': torch.from_numpy(action[NUM_CONTEXT:total_needed]),
            'context_joint': torch.from_numpy(joint[:NUM_CONTEXT]),
            'target_joint': torch.from_numpy(joint[NUM_CONTEXT:total_needed]),
            'text': text,
            'traj_name': traj_name
        }


class EmbodiedTestDataset(Dataset):
    """测试数据集: 只返回上下文帧(用于推理)"""

    def __init__(self, root_dir):
        self.root = root_dir
        self.samples = []
        for name in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "video.mp4")):
                vp = os.path.join(path, "video.mp4")
                ip = os.path.join(path, "instruction.txt")
                if os.path.exists(vp) and os.path.exists(ip):
                    self.samples.append((path, name))
        print(f"[Part3] 测试轨迹: {len(self.samples)} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, traj_name = self.samples[idx]
        frames = read_video_frames(
            os.path.join(traj_path, "video.mp4"),
            num_frames=NUM_CONTEXT, resize=(IMG_H, IMG_W)
        )
        if frames.shape[0] < NUM_CONTEXT:
            pad = frames[-1:].expand(NUM_CONTEXT - frames.shape[0], -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)

        # 读取action/joint(如果有)
        ap = os.path.join(traj_path, "action.txt")
        jp = os.path.join(traj_path, "joint.txt")
        action = read_csv_data(ap) if os.path.exists(ap) else np.zeros((NUM_CONTEXT, ACTION_DIM), dtype=np.float32)
        joint = read_csv_data(jp) if os.path.exists(jp) else np.zeros((NUM_CONTEXT, JOINT_DIM), dtype=np.float32)
        if action.shape[0] < NUM_CONTEXT:
            action = np.concatenate([action, np.tile(action[-1:], (NUM_CONTEXT - action.shape[0], 1))])
        if joint.shape[0] < NUM_CONTEXT:
            joint = np.concatenate([joint, np.tile(joint[-1:], (NUM_CONTEXT - joint.shape[0], 1))])

        text = read_instruction(os.path.join(traj_path, "instruction.txt"))
        return {
            'context_frames': frames[:NUM_CONTEXT],
            'context_action': torch.from_numpy(action[:NUM_CONTEXT]),
            'context_joint': torch.from_numpy(joint[:NUM_CONTEXT]),
            'text': text,
            'traj_name': traj_name,
            'traj_path': traj_path
        }


train_dataset = EmbodiedTrainDataset(TRAIN_DIR)
test_dataset = EmbodiedTestDataset(TEST_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"[Part3] 数据就绪 | 训练{len(train_dataset)}条 测试{len(test_dataset)}条\n")


# ============================================================
# Part 4: 文本融合模块
# ============================================================

class TextConditionedFiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma*feature + beta"""

    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.gamma_fc = nn.Sequential(nn.Linear(text_dim, feature_dim), nn.Tanh())
        self.beta_fc = nn.Linear(text_dim, feature_dim)

    def forward(self, features, text_embed):
        gamma = self.gamma_fc(text_embed)
        beta = self.beta_fc(text_embed)
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return features * (1 + gamma) + beta


class CrossAttentionTextFusion(nn.Module):
    """Cross Attention文本融合"""

    def __init__(self, feature_dim, text_dim, nhead=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, nhead, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, features, text_embed):
        text_kv = self.text_proj(text_embed).unsqueeze(1)
        attn_out, _ = self.cross_attn(features, text_kv, text_kv)
        features = self.norm(features + attn_out)
        features = self.norm2(features + self.ffn(features))
        return features


class SpatialFiLM(nn.Module):
    """空间维度的FiLM调制(用于特征图)"""

    def __init__(self, text_dim, channels):
        super().__init__()
        self.gamma_fc = nn.Sequential(nn.Linear(text_dim, channels), nn.Tanh())
        self.beta_fc = nn.Linear(text_dim, channels)

    def forward(self, feature_map, text_embed):
        gamma = self.gamma_fc(text_embed)[:, :, None, None]
        beta = self.beta_fc(text_embed)[:, :, None, None]
        return feature_map * (1 + gamma) + beta


print("[Part4] 文本融合模块就绪")


# ============================================================
# Part 5: 动作预测器 (CNN+Transformer+FiLM)
# ============================================================

class VisualEncoder(nn.Module):
    """视觉编码器: CNN提取图像特征"""

    def __init__(self, feat_dim=VISUAL_FEAT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(256 * 4, feat_dim)

    def forward(self, x):
        return self.fc(self.conv(x).reshape(x.size(0), -1))


class TextConditionedActionPredictor(nn.Module):
    """
    动作预测器:
    视觉(CNN) + 动作投影 + 关节投影 → FiLM → CrossAttn → Transformer Encoder → 预测50步action
    """

    def __init__(self):
        super().__init__()
        self.visual_encoder = VisualEncoder(VISUAL_FEAT_DIM)
        self.visual_proj = nn.Linear(VISUAL_FEAT_DIM, HIDDEN_DIM)
        self.action_proj = nn.Linear(ACTION_DIM, HIDDEN_DIM)
        self.joint_proj = nn.Linear(JOINT_DIM, HIDDEN_DIM)
        self.fusion = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM)

        # 文本条件注入
        self.text_film_early = TextConditionedFiLM(TEXT_EMBED_DIM, HIDDEN_DIM)
        self.text_cross_attn = CrossAttentionTextFusion(HIDDEN_DIM, TEXT_EMBED_DIM, nhead=4)
        self.pos_enc = nn.Parameter(torch.randn(1, NUM_CONTEXT, HIDDEN_DIM) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=8, dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.text_film_late = TextConditionedFiLM(TEXT_EMBED_DIM, HIDDEN_DIM)
        self.output_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, NUM_PREDICT * ACTION_DIM)
        )

    def forward(self, frames, actions, joints, text_embed):
        B, T, C, H, W = frames.shape
        vis = self.visual_encoder(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)
        fused = self.fusion(torch.cat([
            self.visual_proj(vis),
            self.action_proj(actions),
            self.joint_proj(joints)
        ], dim=-1))

        fused = self.text_film_early(fused, text_embed)
        fused = self.text_cross_attn(fused, text_embed)
        fused = fused + self.pos_enc[:, :T, :]
        encoded = self.transformer(fused)

        last = self.text_film_late(encoded[:, -1, :], text_embed)
        return self.output_head(last).reshape(B, NUM_PREDICT, ACTION_DIM)


print("[Part5] 动作预测器就绪")


# ============================================================
# Part 6: 关节预测器 (LSTM+FiLM)
# ============================================================

class TextConditionedJointPredictor(nn.Module):
    """关节预测器: joint+action拼接 → FiLM → LSTM → 预测50步joint"""

    def __init__(self):
        super().__init__()
        hdim = HIDDEN_DIM // 2
        self.input_proj = nn.Linear(JOINT_DIM + ACTION_DIM, hdim)
        self.text_film = TextConditionedFiLM(TEXT_EMBED_DIM, hdim)
        self.lstm = nn.LSTM(hdim, hdim, num_layers=2, batch_first=True, dropout=0.1)
        self.text_film_late = TextConditionedFiLM(TEXT_EMBED_DIM, hdim)
        self.output_head = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, NUM_PREDICT * JOINT_DIM)
        )

    def forward(self, joints, actions, text_embed):
        x = self.input_proj(torch.cat([joints, actions], dim=-1))
        x = self.text_film(x, text_embed)
        lstm_out, _ = self.lstm(x)
        last = self.text_film_late(lstm_out[:, -1, :], text_embed)
        return self.output_head(last).reshape(-1, NUM_PREDICT, JOINT_DIM)


print("[Part6] 关节预测器就绪")


# ============================================================
# Part 7: 光流视频预测器 (核心! U-Net+ConvLSTM+Warp)
# ============================================================

def warp_image(image, flow):
    """
    使用光流对图像进行双线性变形(warp)
    image: [B, 3, H, W]
    flow:  [B, 2, H, W] (dx, dy)
    返回: [B, 3, H, W]
    """
    B, C, H, W = image.shape
    # 创建基础网格 [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # 光流归一化
    flow_norm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    flow_norm = flow_norm * 2.0 / torch.tensor([W, H], device=image.device, dtype=torch.float32).reshape(1, 1, 1, 2)

    # 采样坐标
    sample_grid = (grid + flow_norm).clamp(-1, 1)
    warped = F.grid_sample(image, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


class FlowPredictorBlock(nn.Module):
    """U-Net风格的卷积块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TextConditionedFlowVideoPredictor(nn.Module):
    """
    基于光流变形的视频帧预测器(文本引导)

    对每个时间步:
    1. 编码(prev_frame + curr_frame) → U-Net特征
    2. 文本SpatialFiLM调制bottleneck特征
    3. ConvLSTM更新时序状态
    4. 解码输出: 光流(dx,dy) + 残差RGB + 遮挡掩码
    5. 合成: pred = mask * warp(curr, flow) + (1-mask) * residual
    """

    def __init__(self, hidden_ch=64):
        super().__init__()
        self.hidden_ch = hidden_ch

        # 编码器 (输入: 2帧拼接=6通道)
        self.enc1 = FlowPredictorBlock(6, 32)     # 256
        self.enc2 = FlowPredictorBlock(32, 64)    # 128
        self.enc3 = FlowPredictorBlock(64, 128)   # 64
        self.enc4 = FlowPredictorBlock(128, 256)  # 32
        self.down = nn.MaxPool2d(2)

        # bottleneck处的文本调制
        self.text_spatial_film = SpatialFiLM(TEXT_EMBED_DIM, 256)

        # ConvLSTM (在bottleneck处)
        self.conv_lstm_cell = nn.Conv2d(256 + hidden_ch, hidden_ch * 4, 3, 1, 1)
        self.conv_lstm_hidden_ch = hidden_ch

        # 解码器 (带skip connection)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = FlowPredictorBlock(hidden_ch + 128, 128)   # +enc3
        self.dec2 = FlowPredictorBlock(128 + 64, 64)           # +enc2
        self.dec1 = FlowPredictorBlock(64 + 32, 32)            # +enc1

        # 输出头
        self.flow_head = nn.Conv2d(32, 2, 3, 1, 1)   # 光流 dx, dy
        self.residual_head = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid()  # 残差图像 [0,1]
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()  # 遮挡掩码 [0,1]
        )

    def init_hidden(self, B, H, W, device):
        sh, sw = H // 8, W // 8
        return (
            torch.zeros(B, self.conv_lstm_hidden_ch, sh, sw, device=device),
            torch.zeros(B, self.conv_lstm_hidden_ch, sh, sw, device=device)
        )

    def conv_lstm_step(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_lstm_cell(combined)
        ch = self.conv_lstm_hidden_ch
        i = torch.sigmoid(gates[:, :ch])
        f = torch.sigmoid(gates[:, ch:2*ch])
        o = torch.sigmoid(gates[:, 2*ch:3*ch])
        g = torch.tanh(gates[:, 3*ch:])
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def encode_and_predict(self, prev_frame, curr_frame, text_embed, state):
        """单步预测: 输入两帧, 输出下一帧预测 + 新状态"""
        x = torch.cat([prev_frame, curr_frame], dim=1)  # [B, 6, H, W]

        # 编码
        e1 = self.enc1(x)                              # [B, 32, H, W]
        e2 = self.enc2(self.down(e1))                  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.down(e2))                  # [B, 128, H/4, W/4]
        e4 = self.enc4(self.down(e3))                  # [B, 256, H/8, W/8]

        # 文本调制
        e4 = self.text_spatial_film(e4, text_embed)

        # ConvLSTM更新状态
        state = self.conv_lstm_step(e4, state)
        h = state[0]

        # 解码 (带skip connection)
        d3 = self.dec3(torch.cat([self.up(h), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        # 输出
        flow = self.flow_head(d1) * 20.0       # 缩放光流(最大±20像素)
        residual = self.residual_head(d1)       # [B, 3, H, W] ∈ [0,1]
        mask = self.mask_head(d1)               # [B, 1, H, W] ∈ [0,1]

        # 变形合成
        warped = warp_image(curr_frame, flow)
        pred = mask * warped + (1 - mask) * residual

        return pred, flow, mask, state

    def forward(self, context_frames, text_embed, predict_len=NUM_PREDICT,
                target_frames=None, teacher_forcing_ratio=0.0):
        """
        自回归预测多帧
        context_frames: [B, T, 3, H, W]
        返回: [B, predict_len, 3, H, W]
        """
        B, T, C, H, W = context_frames.shape
        state = self.init_hidden(B, H, W, context_frames.device)

        # Warm-up: 用上下文帧建立时序状态
        for t in range(1, T):
            prev = context_frames[:, t - 1]
            curr = context_frames[:, t]
            _, _, _, state = self.encode_and_predict(prev, curr, text_embed, state)

        # 自回归预测
        preds = []
        prev_frame = context_frames[:, -2]   # 倒数第二帧
        curr_frame = context_frames[:, -1]    # 最后帧

        for t in range(predict_len):
            pred, flow, mask, state = self.encode_and_predict(prev_frame, curr_frame, text_embed, state)
            preds.append(pred)

            # Scheduled Sampling (训练时逐步减少teacher forcing)
            if target_frames is not None and t < target_frames.shape[1]:
                if np.random.random() < teacher_forcing_ratio:
                    next_frame = target_frames[:, t]
                else:
                    next_frame = pred.detach()
            else:
                next_frame = pred.detach()

            prev_frame = curr_frame
            curr_frame = next_frame

        return torch.stack(preds, dim=1)


print("[Part7] 光流视频预测器就绪 (Warp+Residual+Mask+FiLM)")


# ============================================================
# Part 8: 综合世界模型
# ============================================================

class TextConditionedWorldModel(nn.Module):
    """
    完整的世界模型:
    - 冻结: CLIP ViT-B/32 (文本编码)
    - 训练:
      - TextConditionedActionPredictor
      - TextConditionedJointPredictor
      - TextConditionedFlowVideoPredictor (光流方案!)
    """

    def __init__(self):
        super().__init__()
        self.action_predictor = TextConditionedActionPredictor()
        self.joint_predictor = TextConditionedJointPredictor()
        self.video_predictor = TextConditionedFlowVideoPredictor()

    def forward(self, frames, actions, joints, text_embed,
                video_predict_len=NUM_PREDICT,
                target_frames=None, teacher_forcing_ratio=0.0):
        pred_actions = self.action_predictor(frames, actions, joints, text_embed)
        pred_joints = self.joint_predictor(joints, actions, text_embed)
        pred_frames = self.video_predictor(
            frames, text_embed, predict_len=video_predict_len,
            target_frames=target_frames, teacher_forcing_ratio=teacher_forcing_ratio
        )
        return pred_frames, pred_actions, pred_joints


model = TextConditionedWorldModel().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Part8] 世界模型参数量: {total_params:,}")
print(f"  动作预测器: {sum(p.numel() for p in model.action_predictor.parameters()):,}")
print(f"  关节预测器: {sum(p.numel() for p in model.joint_predictor.parameters()):,}")
print(f"  视频预测器: {sum(p.numel() for p in model.video_predictor.parameters()):,}")


# ============================================================
# Part 9: 损失函数 (MSE+L1+SSIM+CLIP对齐)
# ============================================================

def compute_ssim_loss(pred, target):
    """简化版SSIM损失"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.avg_pool2d(pred, 3, 1, 1)
    mu_t = F.avg_pool2d(target, 3, 1, 1)
    sigma_pp = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_p * mu_p
    sigma_tt = F.avg_pool2d(target * target, 3, 1, 1) - mu_t * mu_t
    sigma_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_pp + sigma_tt + C2))
    return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """联合损失: 视频(MSE+L1+SSIM+可选CLIP) + 动作MSE + 关节MSE"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def clip_alignment_loss(self, pred_frames, text_embed, sample_every=10):
        """CLIP语义对齐损失 (epoch>=20才启用)"""
        if not USE_CLIP:
            return torch.tensor(0.0, device=pred_frames.device)
        B, T, C, H, W = pred_frames.shape
        indices = list(range(0, T, sample_every))
        sampled = pred_frames[:, indices].reshape(B * len(indices), C, H, W)
        img_embed = get_clip_image_embedding(sampled)
        if img_embed is None:
            return torch.tensor(0.0, device=pred_frames.device)
        img_embed = img_embed.reshape(B, len(indices), -1)
        text_exp = text_embed.unsqueeze(1).expand_as(img_embed)
        return 1.0 - F.cosine_similarity(img_embed, text_exp, dim=-1).mean()

    def forward(self, pred_f, target_f, pred_a, target_a, pred_j, target_j,
                text_embed, epoch=1):
        B, T, C, H, W = pred_f.shape

        # 视频损失
        v_mse = self.mse(pred_f, target_f)
        v_l1 = self.l1(pred_f, target_f)
        v_ssim = compute_ssim_loss(
            pred_f.reshape(B * T, C, H, W), target_f.reshape(B * T, C, H, W)
        )

        # CLIP对齐 (渐进启用)
        if epoch >= 20 and USE_CLIP:
            v_clip = self.clip_alignment_loss(pred_f, text_embed)
            cw = min(0.3, (epoch - 20) * 0.015)
        else:
            v_clip = torch.tensor(0.0, device=pred_f.device)
            cw = 0.0

        video_loss = v_mse + 0.2 * v_l1 + 0.15 * v_ssim + cw * v_clip

        # 动作+关节损失
        action_loss = self.mse(pred_a, target_a)
        joint_loss = self.mse(pred_j, target_j)

        # 渐进式权重: 早期重视动作, 后期重视视频
        vw = min(2.0, 0.5 + epoch * 0.025)   # 视频权重: 0.5→2.0
        aw = max(1.0, 3.0 - epoch * 0.025)     # 动作权重: 3.0→1.0

        total = video_loss * vw + action_loss * aw + joint_loss * 1.0
        return total, {
            'total': total.item(), 'video': video_loss.item(),
            'v_mse': v_mse.item(), 'v_ssim': v_ssim.item(), 'v_clip': v_clip.item(),
            'action': action_loss.item(), 'joint': joint_loss.item()
        }


criterion = CombinedLoss()
print("[Part9] 损失函数就绪 (MSE+L1+SSIM+CLIP)")


# ============================================================
# Part 10: 训练循环
# ============================================================

def train():
    global simple_text_encoder

    print("=" * 70)
    print("开始训练 - 光流视频预测 + 文本引导")
    print("=" * 70)

    # 分组学习率
    param_groups = [
        {'params': model.action_predictor.parameters(), 'lr': 5e-5},
        {'params': model.joint_predictor.parameters(), 'lr': 1e-4},
        {'params': model.video_predictor.parameters(), 'lr': 2e-4},
    ]
    if simple_text_encoder is not None:
        param_groups.append({'params': simple_text_encoder.parameters(), 'lr': 1e-4})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_loss = float('inf')
    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if simple_text_encoder is not None:
            simple_text_encoder.train()

        # 渐进式预测长度 (从短到长, 降低初期难度)
        if epoch <= 8:
            vid_pred_len = 5
        elif epoch <= 15:
            vid_pred_len = 10
        elif epoch <= 30:
            vid_pred_len = 25
        else:
            vid_pred_len = 50

        # Scheduled Sampling ratio (逐步降低teacher forcing)
        tf_ratio = max(0.0, 0.8 - epoch / 30)

        epoch_metrics = {}
        num_batches = 0
        pbar = tqdm(train_loader,
                     desc=f"E{epoch:02d}/{NUM_EPOCHS} [pred={vid_pred_len} tf={tf_ratio:.2f}]")

        for batch in pbar:
            cf = batch['context_frames'].to(device)
            tf_t = batch['target_frames'].to(device)
            ca = batch['context_action'].to(device)
            ta = batch['target_action'].to(device)
            cj = batch['context_joint'].to(device)
            tj = batch['target_joint'].to(device)
            texts = list(batch['text'])

            text_embed = get_text_embedding(texts)

            pf, pa, pj = model(cf, ca, cj, text_embed,
                                 video_predict_len=vid_pred_len,
                                 target_frames=tf_t[:, :vid_pred_len],
                                 teacher_forcing_ratio=tf_ratio)

            total_loss, metrics = criterion(
                pf, tf_t[:, :vid_pred_len], pa, ta, pj, tj, text_embed, epoch
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if simple_text_encoder is not None:
                torch.nn.utils.clip_grad_norm_(simple_text_encoder.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            num_batches += 1
            pbar.set_postfix({
                'tot': f"{metrics['total']:.3f}",
                'vid': f"{metrics['video']:.3f}",
                'act': f"{metrics['action']:.3f}"
            })

        scheduler.step()

        avg = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
        loss_history.append({'epoch': epoch, **avg})

        print(f"  E{epoch:02d}: Tot={avg['total']:.5f} "
              f"Vid={avg['video']:.5f}(mse:{avg['v_mse']:.5f} ssim:{avg['v_ssim']:.5f}) "
              f"Act={avg['action']:.5f} Jnt={avg['joint']:.5f}")

        # 保存最优模型
        if avg['total'] < best_loss:
            best_loss = avg['total']
            sd = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': best_loss}
            if simple_text_encoder is not None:
                sd['text_encoder_state_dict'] = simple_text_encoder.state_dict()
            torch.save(sd, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  Best! loss={best_loss:.6f}")

        # 定期保存
        if epoch % 20 == 0:
            sd = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': avg['total']}
            if simple_text_encoder is not None:
                sd['text_encoder_state_dict'] = simple_text_encoder.state_dict()
            torch.save(sd, os.path.join(CHECKPOINT_DIR, f'ckpt_e{epoch}.pt'))
            print(f"  Saved ckpt_e{epoch}")

    print(f"\n训练完成! 最佳loss: {best_loss:.6f}")
    with open(os.path.join(CHECKPOINT_DIR, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)


# ============================================================
# Part 11: 推理生成
# ============================================================

def generate():
    global simple_text_encoder, SUBMISSION_DIR

    print("=" * 70)
    print("生成测试结果")
    print("=" * 70)

    # 加载模型 & 确定输出目录
    _args = parse_args()
    if _args.ckpt:
        ckpt_path = _args.ckpt if os.path.isabs(_args.ckpt) else os.path.join(CHECKPOINT_DIR, _args.ckpt)
        # 自动推导输出目录名
        ckpt_name = os.path.basename(_args.ckpt).replace('.pt', '')
        default_outdir = f"submission_{ckpt_name}"
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
        default_outdir = "submission"

    if _args.outdir:
        out_name = _args.outdir
    else:
        out_name = default_outdir

    # 覆盖全局输出目录
    if os.path.isdir(_DATA_DISK):
        SUBMISSION_DIR = os.path.join(_DATA_DISK, out_name)
    else:
        SUBMISSION_DIR = f"/root/{out_name}"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"输出目录: {SUBMISSION_DIR}")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: 未找到checkpoint: {ckpt_path}")
        print("请先运行 --mode train!")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if simple_text_encoder is not None and 'text_encoder_state_dict' in ckpt:
        simple_text_encoder.load_state_dict(ckpt['text_encoder_state_dict'])
    print(f"已加载 checkpoint | epoch={ckpt['epoch']} | loss={ckpt['loss']:.6f}")

    model.eval()
    if simple_text_encoder is not None:
        simple_text_encoder.eval()

    header = build_csv_header()
    print(f"\n生成 {len(test_dataset)} 条测试结果...\n")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            traj_name = sample['traj_name']
            traj_path = sample['traj_path']
            print(f"[{idx+1}/{len(test_dataset)}] {traj_name}: \"{sample['text'][:50]}\"")

            # 准备输入
            cf = sample['context_frames'].unsqueeze(0).to(device)
            ca = sample['context_action'].unsqueeze(0).to(device)
            cj = sample['context_joint'].unsqueeze(0).to(device)
            text_embed = get_text_embedding([sample['text']])

            # 推理
            pf, pa, pj = model(cf, ca, cj, text_embed, video_predict_len=NUM_PREDICT)

            pred_f = pf[0].cpu()       # [50, 3, H, W]
            pred_a = pa[0].cpu().numpy()
            pred_j = pj[0].cpu().numpy()

            out_dir = os.path.join(SUBMISSION_DIR, traj_name)
            os.makedirs(out_dir, exist_ok=True)

            # 1. instruction.txt
            with open(os.path.join(out_dir, "instruction.txt"), "w", encoding="utf-8") as f:
                f.write(sample['text'])

            # 2. action.txt — 只保留最后一帧上下文 + 预测结果 = 51行
            orig_a = read_csv_raw(os.path.join(traj_path, "action.txt")) if os.path.exists(os.path.join(traj_path, "action.txt")) else None
            ctx_act = sample['context_action'].numpy()
            # 只取最后一帧上下文(1行)，不是全部16行
            last_ctx_act = ctx_act[-1:]  # [1, 26]
            if orig_a is not None:
                last_ctx_idx = orig_a[NUM_CONTEXT - 1:NUM_CONTEXT, 0:1]  # 第16行的索引
                last_idx = int(last_ctx_idx[0, 0])
            else:
                last_ctx_idx = np.array([[NUM_CONTEXT - 1]]).astype(float)
                last_idx = NUM_CONTEXT - 1
            pred_idx = np.arange(last_idx + 1, last_idx + 1 + NUM_PREDICT).reshape(-1, 1).astype(np.float64)
            full_action = np.concatenate([
                np.concatenate([last_ctx_idx, last_ctx_act], axis=1),   # 1行上下文
                np.concatenate([pred_idx, pred_a], axis=1)              # 50行预测
            ], axis=0)  # 总共 51 行
            save_csv_with_header(os.path.join(out_dir, "action.txt"), full_action, header)

            # 3. joint.txt — 同样只保留最后一帧上下文 + 预测结果 = 51行
            orig_j = read_csv_raw(os.path.join(traj_path, "joint.txt")) if os.path.exists(os.path.join(traj_path, "joint.txt")) else None
            ctx_jnt = sample['context_joint'].numpy()
            last_ctx_jnt = ctx_jnt[-1:]  # [1, 26]
            if orig_j is not None:
                last_ctx_j_idx = orig_j[NUM_CONTEXT - 1:NUM_CONTEXT, 0:1]
                last_j_idx = int(last_ctx_j_idx[0, 0])
            else:
                last_ctx_j_idx = np.array([[NUM_CONTEXT - 1]]).astype(float)
                last_j_idx = NUM_CONTEXT - 1
            pred_j_idx = np.arange(last_j_idx + 1, last_j_idx + 1 + NUM_PREDICT).reshape(-1, 1).astype(np.float64)
            full_joint = np.concatenate([
                np.concatenate([last_ctx_j_idx, last_ctx_jnt], axis=1),  # 1行上下文
                np.concatenate([pred_j_idx, pred_j], axis=1)             # 50行预测
            ], axis=0)  # 总共 51 行
            save_csv_with_header(os.path.join(out_dir, "joint.txt"), full_joint, header)

            # 4. video.mp4 — 只输出50帧预测结果（不含上下文帧）
            input_video = os.path.join(traj_path, "video.mp4")
            ori_frames = read_video_frames_original(input_video, num_frames=NUM_CONTEXT)

            if len(ori_frames) > 0:
                h, w = ori_frames[0].shape[:2]
                cap_tmp = cv2.VideoCapture(input_video)
                fps = cap_tmp.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                cap_tmp.release()

                video_path = os.path.join(out_dir, "video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

                # 只写50帧模型预测结果（上采样到原始分辨率）
                for t in range(pred_f.shape[0]):
                    frame_rgb = pred_f[t].permute(1, 2, 0).numpy()
                    frame_rgb = np.clip(frame_rgb * 255, 0, 255).astype(np.uint8)
                    frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)

                writer.release()
                vsize = os.path.getsize(video_path) / 1024
                print(f"  OK | video={vsize:.0f}KB | act={full_action.shape} | jnt={full_joint.shape}")
            else:
                print(f"  WARN: 视频读取失败")

    print(f"\n生成完成! -> {SUBMISSION_DIR}")


# ============================================================
# Part 12: 质量自检
# ============================================================

def quality_check():
    print("=" * 70)
    print("质量自检")
    print("=" * 70)
    ok = 0
    total = 0
    req_files = ["video.mp4", "action.txt", "joint.txt", "instruction.txt"]

    for name in sorted(os.listdir(SUBMISSION_DIR)):
        d = os.path.join(SUBMISSION_DIR, name)
        if not os.path.isdir(d):
            continue
        total += 1
        files = os.listdir(d)
        missing = [f for f in req_files if f not in files]
        if missing:
            print(f"  MISSING {name}: {missing}")
            continue

        cap = cv2.VideoCapture(os.path.join(d, "video.mp4"))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        try:
            act = np.loadtxt(os.path.join(d, "action.txt"), delimiter=',', skiprows=1)
            jnt = np.loadtxt(os.path.join(d, "joint.txt"), delimiter=',', skiprows=1)
            shape_ok = fc == 66 and act.shape == (66, 27) and jnt.shape == (66, 27)
            if shape_ok:
                ok += 1
            else:
                print(f"  FAIL {name}: frames={fc} act={act.shape} jnt={jnt.shape}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    print(f"\n结果: {ok}/{total} 通过")
    if total > 0 and ok == total:
        print("全部通过!")
    return total > 0 and ok == total


# ============================================================
# Part 13: CLIP 自评 (可选)
# ============================================================

def clip_eval():
    if not USE_CLIP:
        print("CLIP未加载, 跳过自评")
        return

    print("\nCLIP语义一致性自评:")
    scores = []
    with torch.no_grad():
        for idx in range(min(10, len(test_dataset))):
            s = test_dataset[idx]
            name = s['traj_name']
            vp = os.path.join(SUBMISSION_DIR, name, "video.mp4")
            if not os.path.exists(vp):
                continue
            te = get_text_embedding([s['text']])
            gen = read_video_frames(vp, num_frames=66, resize=(224, 224))
            gen50 = gen[16:]
            sampled = gen50[::10].to(device)
            ie = get_clip_image_embedding(sampled)
            if ie is not None:
                sim = F.cosine_similarity(te.expand(ie.shape[0], -1), ie).mean().item()
                scores.append(sim)
                print(f"  {name}: {sim:.4f}")

    if scores:
        print(f"  平均: {np.mean(scores):.4f}")


# ============================================================
# Part 14: 入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="具身智能视频生成 - 光流方案")
    p.add_argument("--mode", default="full", choices=["full", "train", "infer", "check"])
    p.add_argument("--ckpt", default=None, help="指定checkpoint文件路径, 如 ckpt_e20.pt (默认用 best_model.pt)")
    p.add_argument("--outdir", default=None, help="输出目录名(放在数据盘/根目录下), 如 submission_e20 (默认用 submission)")
    a, u = p.parse_known_args()
    if u:
        print(f"Ignoring unknown args: {u}")
    return a


def main():
    args = parse_args()
    print(f"\n模式: {args.mode}\n")

    if args.mode in ("full", "train"):
        train()

    if args.mode in ("full", "infer"):
        generate()

    if args.mode in ("full", "infer", "check"):
        quality_check()

    if args.mode in ("full", "infer"):
        clip_eval()

    print(f"\n{'='*50}\n完成! 输出目录: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
