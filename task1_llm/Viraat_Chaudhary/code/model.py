"""A post-normalized causal decoder; attention and layer normalization are explicit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class DecoderConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float
    epsilon: float = 1e-5

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.d_model,
                self.num_heads,
                self.num_layers,
                self.d_ff,
            )
            <= 0
        ):
            raise ValueError("Decoder dimensions must be positive.")
        if self.d_model % self.num_heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid head division or dropout.")

    @classmethod
    def from_experiment(cls, config: dict, vocab_size: int):
        m = config["model"]
        return cls(
            vocab_size,
            config["data"]["sequence_length"],
            m["d_model"],
            m["num_heads"],
            m["num_layers"],
            m["d_ff"],
            m["dropout"],
            m["layer_norm_epsilon"],
        )


class LayerNormalization(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))
        self.epsilon = epsilon

    def forward(self, x):
        # Accumulate statistics in float32 during reduced-precision training.
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = value.mean(dim=-1, keepdim=True)
        centered = value - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.scale + self.shift).to(x.dtype)


class CausalAttention(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.heads, self.head_width = config.num_heads, config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.probability_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "allowed",
            torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x, return_weights=False):
        batch, length, width = x.shape
        projected = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.float().masked_fill(~self.allowed[:length, :length], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        attended = self.probability_dropout(probabilities) @ v
        joined = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.output_dropout(self.output(joined))
        return (output, probabilities) if return_weights else output


class PostNormBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.attention = CausalAttention(config)
        self.attention_norm = LayerNormalization(config.d_model, config.epsilon)
        self.expand = nn.Linear(config.d_model, config.d_ff)
        self.contract = nn.Linear(config.d_ff, config.d_model)
        self.feedforward_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = LayerNormalization(config.d_model, config.epsilon)

    def forward(self, x):
        # Post-norm: normalization follows each residual addition.
        x = self.attention_norm(x + self.attention(x))
        feedforward = self.contract(torch.relu(self.expand(x)))
        return self.feedforward_norm(x + self.feedforward_dropout(feedforward))


class CharGPT(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([PostNormBlock(config) for _ in range(config.num_layers)])
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        # The output head is intentionally independent of the token embedding.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.context_length:
            raise ValueError(
                "Expected nonempty [batch, length] indices within the configured context."
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_head(hidden)

    @torch.inference_mode()
    def generate(
        self, prompt_ids, new_characters: int, temperature: float = 1.0, greedy: bool = False
    ):
        if new_characters < 0 or (not greedy and temperature <= 0):
            raise ValueError("Invalid generation length or temperature.")
        was_training = self.training
        self.eval()
        generated = prompt_ids.clone()
        try:
            for _ in range(new_characters):
                logits = self(generated[:, -self.config.context_length :])[:, -1, :].float()
                next_id = (
                    logits.argmax(-1, keepdim=True)
                    if greedy
                    else torch.multinomial(torch.softmax(logits / temperature, -1), 1)
                )
                generated = torch.cat((generated, next_id), dim=1)
            return generated
        finally:
            self.train(was_training)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configuration(self):
        return asdict(self.config)
