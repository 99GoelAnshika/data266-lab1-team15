from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    max_sequence_length: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float
    use_bias: bool
    layer_norm_epsilon: float
    tie_token_and_output_embeddings: bool

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("Vocabulary size must be greater than one.")
        if self.max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be exactly divisible by num_heads."
            )
        if self.num_layers <= 0:
            raise ValueError("The model must contain at least one block.")
        if self.d_ff <= 0:
            raise ValueError("Feed-forward dimension must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be in the range [0, 1).")

    @property
    def head_dimension(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_experiment_config(
        cls,
        experiment_config: dict[str, Any],
        vocab_size: int,
    ) -> "GPTConfig":
        model_config = experiment_config["model"]
        data_config = experiment_config["data"]

        return cls(
            vocab_size=vocab_size,
            max_sequence_length=int(data_config["sequence_length"]),
            d_model=int(model_config["d_model"]),
            num_heads=int(model_config["num_heads"]),
            num_layers=int(model_config["num_layers"]),
            d_ff=int(model_config["d_ff"]),
            dropout=float(model_config["dropout"]),
            use_bias=bool(model_config["use_bias"]),
            layer_norm_epsilon=float(
                model_config["layer_norm_epsilon"]
            ),
            tie_token_and_output_embeddings=bool(
                model_config["tie_token_and_output_embeddings"]
            ),
        )


class ManualLayerNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        normalized = (inputs - mean) * torch.rsqrt(
            variance + self.epsilon
        )
        return self.weight * normalized + self.bias


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dimension = config.head_dimension

        self.query_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )
        self.key_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )
        self.value_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )
        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.use_bias,
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(
                config.max_sequence_length,
                config.max_sequence_length,
                dtype=torch.bool,
            )
        ).view(
            1,
            1,
            config.max_sequence_length,
            config.max_sequence_length,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length, _ = inputs.shape

        queries = self._split_heads(
            self.query_projection(inputs)
        )
        keys = self._split_heads(
            self.key_projection(inputs)
        )
        values = self._split_heads(
            self.value_projection(inputs)
        )

        attention_scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )
        attention_scores = attention_scores / math.sqrt(
            self.head_dimension
        )

        active_mask = self.causal_mask[
            :,
            :,
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~active_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )
        dropped_attention = self.attention_dropout(
            attention_weights
        )

        attended_values = torch.matmul(
            dropped_attention,
            values,
        )
        merged_values = self._merge_heads(attended_values)
        outputs = self.output_projection(merged_values)
        outputs = self.output_dropout(outputs)

        if return_attention:
            return outputs, attention_weights

        return outputs, None


class ManualGELU(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coefficient = math.sqrt(2.0 / math.pi)
        cubic_term = 0.044715 * inputs.pow(3)
        return 0.5 * inputs * (
            1.0 + torch.tanh(coefficient * (inputs + cubic_term))
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.use_bias,
        )
        self.activation = ManualGELU()
        self.output_projection = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=config.use_bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.activation(hidden)
        hidden = self.output_projection(hidden)
        return self.dropout(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = ManualLayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
        )
        self.attention = CausalMultiHeadSelfAttention(config)
        self.feed_forward_norm = ManualLayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
        )
        self.feed_forward = FeedForwardNetwork(config)

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized_inputs = self.attention_norm(inputs)
        attention_outputs, attention_weights = self.attention(
            normalized_inputs,
            return_attention=return_attention,
        )
        residual = inputs + attention_outputs

        normalized_residual = self.feed_forward_norm(residual)
        feed_forward_outputs = self.feed_forward(
            normalized_residual
        )
        outputs = residual + feed_forward_outputs

        return outputs, attention_weights


class CharacterGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length,
            config.d_model,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = ManualLayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
        )
        self.language_model_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=config.use_bias,
        )

        self.apply(self._initialize_weights)
        self._scale_residual_projections()

        if config.tie_token_and_output_embeddings:
            self.language_model_head.weight = (
                self.token_embeddings.weight
            )

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        residual_standard_deviation = 0.02 / math.sqrt(
            2.0 * self.config.num_layers
        )

        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )
            nn.init.normal_(
                block.feed_forward.output_projection.weight,
                mean=0.0,
                std=residual_standard_deviation,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]."
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model "
                f"limit {self.config.max_sequence_length}."
            )

        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError(
                "targets must have the same shape as input_ids."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embeddings(input_ids)
        position_vectors = self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(
            token_vectors + position_vectors.unsqueeze(0)
        )

        collected_attentions: list[torch.Tensor] = []

        for block in self.blocks:
            hidden_states, attention_weights = block(
                hidden_states,
                return_attention=return_attentions,
            )
            if attention_weights is not None:
                collected_attentions.append(attention_weights)

        hidden_states = self.final_norm(hidden_states)
        logits = self.language_model_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    batch_size * sequence_length,
                    self.config.vocab_size,
                ),
                targets.reshape(batch_size * sequence_length),
            )

        return {
            "logits": logits,
            "loss": loss,
            "attentions": (
                collected_attentions
                if return_attentions
                else None
            ),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_characters: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        if max_new_characters < 0:
            raise ValueError(
                "max_new_characters cannot be negative."
            )
        if not greedy and temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive for sampling."
            )

        was_training = self.training
        self.eval()
        generated = input_ids

        for _ in range(max_new_characters):
            context = generated[
                :,
                -self.config.max_sequence_length :,
            ]

            outputs = self(context)
            next_character_logits = outputs["logits"][:, -1, :]

            if greedy:
                next_character = torch.argmax(
                    next_character_logits,
                    dim=-1,
                    keepdim=True,
                )
            else:
                probabilities = torch.softmax(
                    next_character_logits / temperature,
                    dim=-1,
                )
                next_character = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            generated = torch.cat(
                [generated, next_character],
                dim=1,
            )

        if was_training:
            self.train()

        return generated

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_model_from_experiment_config(
    experiment_config: dict[str, Any],
    vocab_size: int,
) -> CharacterGPT:
    model_config = GPTConfig.from_experiment_config(
        experiment_config=experiment_config,
        vocab_size=vocab_size,
    )
    return CharacterGPT(model_config)