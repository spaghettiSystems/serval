from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_flash_attention import *
import torchmetrics

import pytorch_lightning as pl
from dataset_map import *

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


class MLPHead(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_expansion=2, layers=3, dropout_rate=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        self.layers.append(nn.Linear(input_dim, input_dim * hidden_expansion))
        self.norms.append(nn.LayerNorm(input_dim * hidden_expansion))

        for _ in range(layers - 2):
            self.layers.append(
                nn.Linear(input_dim * hidden_expansion, input_dim * hidden_expansion)
            )
            self.norms.append(nn.LayerNorm(input_dim * hidden_expansion))

        self.layers.append(nn.Linear(input_dim * hidden_expansion, output_dim))

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = self.dropout(F.gelu(norm(layer(x))))

        x = self.layers[-1](x)
        return x


class ImprovedMLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=None,
        dropout_rate=0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.add_layer(prev_dim, hidden_dim)
            prev_dim = hidden_dim

        self.final_layer = nn.Linear(prev_dim, output_dim)
        self.initialize_final_layer(self.final_layer)

        self.dropout = nn.Dropout(dropout_rate)

    def add_layer(self, input_dim, output_dim):
        layer = nn.Linear(input_dim, output_dim)
        self.initialize_layer(layer)
        self.layers.append(layer)
        self.norms.append(nn.LayerNorm(output_dim))

    @staticmethod
    def initialize_layer(layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def initialize_final_layer(self, layer):
        nn.init.normal_(layer.weight, std=1 / layer.weight.shape[1] ** 0.5)

        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)

        x = self.final_layer(x)
        return x


class MLPWrapper(nn.Module):
    def __init__(self, model, input_dim, output_dim, hidden_dims, dropout_rate=0.1):
        super().__init__()
        self.model = model

        self.model = torch.compile(self.model)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.head = ImprovedMLPHead(input_dim, output_dim, hidden_dims, dropout_rate)

    def forward(self, x):
        features = self.model(x).last_hidden_state

        features = features.detach()
        return self.head(features), features


class WhisperEncoderWithHead(nn.Module):
    def __init__(
        self,
        model_name="openai/whisper-medium",
        num_labels=8,
    ):
        super(WhisperEncoderWithHead, self).__init__()

        model = WhisperFlashAttentionForConditionalGeneration.from_pretrained(
            "openai/whisper-medium"
        )
        self.encoder = model.get_encoder()

        self.encoder.gradient_checkpointing_enable()

        for param in self.encoder.conv1.parameters():
            param.requires_grad = False

        for param in self.encoder.conv2.parameters():
            param.requires_grad = False

        for param in self.encoder.embed_positions.parameters():
            param.requires_grad = False

        self.encoder.train(True)

        self.head = nn.Linear(1024, 8)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.len_dataset = 0

    def forward(self, x, attn_mask=None):
        hidden = (
            self.encoder(x).last_hidden_state
            if attn_mask is None
            else self.encoder(x, attention_mask=attn_mask).last_hidden_state
        )
        return self.head(hidden), hidden
