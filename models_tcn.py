"""
全序列 TCN 教师/学生分类模型（LUPI 蒸馏用）。

输入形状: [B, T, C]
输出形状: (logits, z)，其中 z 的维度为 latent_dim。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 仅支持奇数 kernel，保证对称 padding 后长度不变
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length with symmetric padding")
        # 使用对称 padding + 空洞卷积来扩大感受野
        # 同时利用t-k和t+k的输入，稳定训练
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 如果通道数不一致，shortcut 需要投影到新通道数
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差支路
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差相加，稳定训练
        out = out + identity
        return self.relu(out)


class TCNEncoderFullSequence(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        # 通道配置不能为空
        if not channels:
            raise ValueError("channels must be a non-empty list")
        # 输入投影到第一个通道数
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks: List[nn.Module] = []
        in_ch = channels[0]
        # 按层堆叠空洞卷积，dilation 逐层指数增长
        for i, out_ch in enumerate(channels):
            dilation = dilation_base**i
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        # 时间维全局池化，得到固定长度表征
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return x


class ClassifierMLP(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int = 3, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        # 简单两层 MLP 分类器
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TeacherClassifierTCNFull(nn.Module):
    def __init__(
        self,
        input_dim: int = 18,
        latent_dim: int = 32,
        num_classes: int = 3,
        channels: List[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.7,
    ) -> None:
        super().__init__()
        # 保证最后一层通道数与 latent_dim 对齐
        channels = channels or [32, 32]
        if channels[-1] != latent_dim:
            channels = [*channels, latent_dim]
        # 全序列 TCN 编码器，直接处理完整时间窗
        self.encoder = TCNEncoderFullSequence(
            input_dim=input_dim,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation_base=dilation_base,
        )
        # 分类头
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码得到 z，再进行分类
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


class StudentClassifierTCNFull(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        latent_dim: int = 32,
        num_classes: int = 3,
        channels: List[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # 保证最后一层通道数与 latent_dim 对齐
        channels = channels or [32, 32]
        if channels[-1] != latent_dim:
            channels = [*channels, latent_dim]
        # 学生端编码器（不使用特权特征）
        self.encoder = TCNEncoderFullSequence(
            input_dim=input_dim,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation_base=dilation_base,
        )
        # 学生端分类头
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码得到 z，再进行分类
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z
