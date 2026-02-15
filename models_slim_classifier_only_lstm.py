"""
Slim Teacher-Student models for LUPI (Learning Using Privileged Information) with
LSTM temporal aggregation to preserve segment order.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SegmentEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_channels: List[int] = [16, 32, 64],
        block_dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers: List[nn.Module] = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels:
            # 如果输入维度和输出维度不一致，需要进行下采样，确保shortcut的维度一致
            if in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm1d(out_ch),
                )
                layers.append(
                    ResBlock1D(in_ch, out_ch, stride=2, downsample=downsample, dropout=block_dropout)
                )
            else:
                layers.append(ResBlock1D(in_ch, out_ch, stride=1, dropout=block_dropout))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc_out(x)


class ClassifierMLP(nn.Module):
    def __init__(self, latent_dim: int = 32, num_classes: int = 3, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
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


class TemporalConvAggregator(nn.Module):
    """Multi-layer temporal conv aggregator over segment latents.

    Input:  latent_seq [B, S, C]
    Output: aggregated [B, C] via mean pooling across S
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1, num_layers: int = 1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length with symmetric padding")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        padding = kernel_size // 2
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        # latent_seq: [B, S, C] -> [B, C, S]
        x = latent_seq.transpose(1, 2)
        for block in self.blocks:
            out = block(x)
            x = out + x  # residual per layer
        x = x.transpose(1, 2)  # [B, S, C]
        return x.mean(dim=1)  # mean pooling over S


class TemporalConvAggregator1Layer(TemporalConvAggregator):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__(channels=channels, kernel_size=kernel_size, dropout=dropout, num_layers=1)


class TemporalConvAggregator2Layer(TemporalConvAggregator):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__(channels=channels, kernel_size=kernel_size, dropout=dropout, num_layers=2)


class TeacherClassifier(nn.Module):
    """Teacher model with full 18-dim features (privileged information)."""

    def __init__(
        self,
        input_dim: int = 18,            # 输入特征维度 (教师模型包含隐私信息，维度为18)
        latent_dim: int = 128,          # 潜在空间维度 (SegmentEncoder输出的特征向量维度)
        num_classes: int = 3,           # 分类类别数 (例如: 正常, 故障A, 故障B)
        num_segments: int = 20,         # 时间序列切分段数 (控制局部特征提取的粒度)
        sequence_length: int = 1000,    # 输入序列总长度 (例如: 窗口大小为20)
        hidden_channels: List[int] = [16, 32, 64],  # ResBlock每层的通道数 (决定ResBlock的数量和深度)
        block_dropout: float = 0.5,     # ResBlock内部的Dropout比例
        classifier_hidden: int = 64,    # 分类器MLP的隐藏层维度
        classifier_dropout: float = 0.7, # 分类器内部的Dropout比例
        aggregator_layers: int = 1,     # LSTM层数 (保留顺序信息)
    ):
        super().__init__()
        self.num_segments = num_segments
        self.sequence_length = sequence_length
        self.segment_length = sequence_length // num_segments
        self.encoder = SegmentEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            block_dropout=block_dropout,
        )
        self.temporal_aggregator = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=aggregator_layers,
            batch_first=True,
        )
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim, num_classes=num_classes, hidden_dim=classifier_hidden, dropout=classifier_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent sequence and aggregate with LSTM to keep order."""
        segments = x.split(self.segment_length, dim=1)
        latents = [self.encoder(seg) for seg in segments if seg.size(1) == self.segment_length]
        latent_seq = torch.stack(latents, dim=1)
        agg_out, _ = self.temporal_aggregator(latent_seq)
        return agg_out[:, -1, :]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


class TeacherClassifierTCN(nn.Module):
    """Teacher model with full 18-dim features, using 1-layer TCN + mean pooling aggregator."""

    def __init__(
        self,
        input_dim: int = 18,
        latent_dim: int = 128,
        num_classes: int = 3,
        num_segments: int = 20,
        sequence_length: int = 1000,
        hidden_channels: List[int] = [16, 32, 64],
        block_dropout: float = 0.5,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.7,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_layers: int = 2,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.sequence_length = sequence_length
        self.segment_length = sequence_length // num_segments
        self.encoder = SegmentEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            block_dropout=block_dropout,
        )
        if tcn_layers == 1:
            self.temporal_aggregator = TemporalConvAggregator1Layer(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout
            )
        elif tcn_layers == 2:
            self.temporal_aggregator = TemporalConvAggregator2Layer(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout
            )
        else:
            self.temporal_aggregator = TemporalConvAggregator(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout, num_layers=tcn_layers
            )
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim, num_classes=num_classes, hidden_dim=classifier_hidden, dropout=classifier_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        segments = x.split(self.segment_length, dim=1)
        latents = [self.encoder(seg) for seg in segments if seg.size(1) == self.segment_length]
        latent_seq = torch.stack(latents, dim=1)  # [B, S, C]
        return self.temporal_aggregator(latent_seq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


class StudentClassifier(nn.Module):
    """Student model with partial 14-dim features (no privileged information)."""

    def __init__(
        self,
        input_dim: int = 14,            # 输入特征维度 (学生模型通常使用公开特征，维度为14)
        latent_dim: int = 128,          # 潜在空间维度 (需与教师模型保持一致以进行对齐)
        num_classes: int = 3,           # 分类类别数
        num_segments: int = 10,         # 切分段数
        sequence_length: int = 1000,    # 输入序列长度
        hidden_channels: List[int] = [16, 32, 64],  # ResBlock通道配置
        block_dropout: float = 0.5,     # Encoder Dropout
        classifier_hidden: int = 64,    # Classifier隐藏层
        classifier_dropout: float = 0.5, # Classifier Dropout
        aggregator_layers: int = 1,     # LSTM层数 (保留顺序信息)
    ):
        super().__init__()
        self.num_segments = num_segments
        self.sequence_length = sequence_length
        self.segment_length = sequence_length // num_segments
        self.encoder = SegmentEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            block_dropout=block_dropout,
        )
        self.temporal_aggregator = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=aggregator_layers,
            batch_first=True,
        )
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim, num_classes=num_classes, hidden_dim=classifier_hidden, dropout=classifier_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent sequence and aggregate with LSTM to keep order."""
        segments = x.split(self.segment_length, dim=1)
        latents = [self.encoder(seg) for seg in segments if seg.size(1) == self.segment_length]
        latent_seq = torch.stack(latents, dim=1)
        agg_out, _ = self.temporal_aggregator(latent_seq)
        return agg_out[:, -1, :]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


class StudentClassifierTCN(nn.Module):
    """Student model with partial 14-dim features, using 1-layer TCN + mean pooling aggregator."""

    def __init__(
        self,
        input_dim: int = 14,
        latent_dim: int = 128,
        num_classes: int = 3,
        num_segments: int = 10,
        sequence_length: int = 1000,
        hidden_channels: List[int] = [16, 32, 64],
        block_dropout: float = 0.5,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.5,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_layers: int = 2,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.sequence_length = sequence_length
        self.segment_length = sequence_length // num_segments
        self.encoder = SegmentEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            block_dropout=block_dropout,
        )
        if tcn_layers == 1:
            self.temporal_aggregator = TemporalConvAggregator1Layer(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout
            )
        elif tcn_layers == 2:
            self.temporal_aggregator = TemporalConvAggregator2Layer(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout
            )
        else:
            self.temporal_aggregator = TemporalConvAggregator(
                channels=latent_dim, kernel_size=tcn_kernel_size, dropout=tcn_dropout, num_layers=tcn_layers
            )
        self.classifier = ClassifierMLP(
            latent_dim=latent_dim, num_classes=num_classes, hidden_dim=classifier_hidden, dropout=classifier_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        segments = x.split(self.segment_length, dim=1)
        latents = [self.encoder(seg) for seg in segments if seg.size(1) == self.segment_length]
        latent_seq = torch.stack(latents, dim=1)
        return self.temporal_aggregator(latent_seq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z
