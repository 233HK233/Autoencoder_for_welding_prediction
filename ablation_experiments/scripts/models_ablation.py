#!/usr/bin/env python3
"""Models used by ablation experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LSTMClassifier(nn.Module):
    """Sequence classifier returning (logits, z) to match distillation interface."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.latent_dim = out_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.latent_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _pooled_representation(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.lstm(x)
        seq_mean = seq_out.mean(dim=1)
        seq_max = seq_out.max(dim=1).values
        return torch.cat([seq_mean, seq_max], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._pooled_representation(x)
        logits = self.classifier(z)
        return logits, z

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the post-fc1 representation for inference-only analysis."""
        z = self._pooled_representation(x)
        h = self.classifier[0](z)
        h = self.classifier[1](h)
        h = self.classifier[2](h)
        h = self.classifier[3](h)
        logits = self.classifier[4](h)
        return logits, h


class GRUClassifier(nn.Module):
    """Sequence classifier returning (logits, z) with a GRU backbone."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.latent_dim = out_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.latent_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        seq_out, _ = self.gru(x)
        seq_mean = seq_out.mean(dim=1)
        seq_max = seq_out.max(dim=1).values
        z = torch.cat([seq_mean, seq_max], dim=1)
        logits = self.classifier(z)
        return logits, z


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max positional length {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderClassifier(nn.Module):
    """Transformer encoder classifier returning (logits, z)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.2,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.latent_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.latent_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        z = torch.cat([h_mean, h_max], dim=1)
        logits = self.classifier(z)
        return logits, z


__all__ = ["LSTMClassifier", "GRUClassifier", "SinusoidalPositionalEncoding", "TransformerEncoderClassifier"]
