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


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class AttentionTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: int = 64,
        tcn_layers: int = 3,
        tcn_kernel: int = 3,
        tcn_dropout: float = 0.15,
        dilation_base: int = 2,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dim: int = 128,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.tcn = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=tcn_kernel,
                    dilation=dilation_base**i,
                    dropout=tcn_dropout,
                )
                for i in range(max(tcn_layers, 1))
            ]
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(ff_dim, channels),
        )
        self.ln2 = nn.LayerNorm(channels)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(channels * 2, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x.transpose(1, 2))
        for blk in self.tcn:
            h = blk(h)
        h = h.transpose(1, 2)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.ff(h))
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        z = torch.cat([h_mean, h_max], dim=1)
        logits = self.classifier(z)
        return logits, z


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.use_bottleneck = in_ch > bottleneck
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()
        branch_in = bottleneck if self.use_bottleneck else in_ch
        self.b1 = nn.Conv1d(branch_in, out_ch, kernel_size=1, padding=0, bias=False)
        self.b3 = nn.Conv1d(branch_in, out_ch, kernel_size=3, padding=1, bias=False)
        self.b5 = nn.Conv1d(branch_in, out_ch, kernel_size=5, padding=2, bias=False)
        self.bp = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        out = torch.cat([self.b1(z), self.b3(z), self.b5(z), self.bp(x)], dim=1)
        out = self.bn(out)
        out = self.act(out)
        return self.drop(out)


class InceptionTimeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        out_ch: int = 16,
        n_blocks: int = 6,
        bottleneck: int = 16,
        dropout: float = 0.2,
        classifier_hidden: int = 96,
        classifier_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = input_dim
        for i in range(max(n_blocks, 1)):
            blk = InceptionBlock1D(in_ch=in_ch, out_ch=out_ch, bottleneck=bottleneck, dropout=dropout)
            blocks.append(blk)
            in_ch = out_ch * 4
            if (i + 1) % 3 == 0:
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(input_dim if i == 2 else out_ch * 4, out_ch * 4, kernel_size=1, bias=False),
                        nn.BatchNorm1d(out_ch * 4),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.n_blocks = max(n_blocks, 1)
        self.out_dim = out_ch * 8
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.out_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        residual_source = h
        block_count = 0
        for module in self.blocks:
            if isinstance(module, InceptionBlock1D):
                h = module(h)
                block_count += 1
                if block_count % 3 == 0:
                    continue
            else:
                h = torch.relu(h + module(residual_source))
                residual_source = h
        h_mean = h.mean(dim=2)
        h_max = h.max(dim=2).values
        z = torch.cat([h_mean, h_max], dim=1)
        logits = self.classifier(z)
        return logits, z


__all__ = [
    "TCNResidualBlock",
    "TCNEncoderFullSequence",
    "ClassifierMLP",
    "TeacherClassifierTCNFull",
    "StudentClassifierTCNFull",
    "ResidualTCNBlock",
    "AttentionTCNClassifier",
    "InceptionBlock1D",
    "InceptionTimeClassifier",
]
