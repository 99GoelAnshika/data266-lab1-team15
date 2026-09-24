import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

import model as model_module
from model import (
    CausalMultiHeadSelfAttention,
    CharacterGPT,
    GPTConfig,
    ManualLayerNorm,
)


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=13,
        max_sequence_length=8,
        d_model=12,
        num_heads=3,
        num_layers=2,
        d_ff=24,
        dropout=0.0,
        use_bias=True,
        layer_norm_epsilon=1.0e-5,
        tie_token_and_output_embeddings=True,
    )


def test_manual_layer_norm_matches_reference_operation() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(2, 4, 6)
    layer = ManualLayerNorm(6, epsilon=1.0e-5)

    with torch.no_grad():
        layer.weight.copy_(torch.randn(6))
        layer.bias.copy_(torch.randn(6))

    actual = layer(inputs)
    expected = F.layer_norm(
        inputs,
        normalized_shape=(6,),
        weight=layer.weight,
        bias=layer.bias,
        eps=1.0e-5,
    )

    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_attention_weights_cannot_access_future_positions() -> None:
    torch.manual_seed(2)
    attention = CausalMultiHeadSelfAttention(tiny_config())
    attention.eval()

    inputs = torch.randn(2, 6, 12)
    _, weights = attention(inputs, return_attention=True)

    assert weights is not None
    assert weights.shape == (2, 3, 6, 6)

    future_mask = torch.triu(
        torch.ones(6, 6, dtype=torch.bool),
        diagonal=1,
    )
    future_weights = weights[..., future_mask]

    assert torch.count_nonzero(future_weights).item() == 0
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
        atol=1.0e-6,
    )


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(3)
    model = CharacterGPT(tiny_config())
    model.eval()

    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_future = torch.tensor([[1, 2, 3, 9, 8, 7]])

    original_logits = model(original)["logits"]
    changed_logits = model(changed_future)["logits"]

    assert torch.allclose(
        original_logits[:, :3],
        changed_logits[:, :3],
        atol=1.0e-6,
    )
    assert not torch.allclose(
        original_logits[:, 3:],
        changed_logits[:, 3:],
    )


def test_forward_loss_shape_and_gradients() -> None:
    torch.manual_seed(4)
    model = CharacterGPT(tiny_config())

    inputs = torch.randint(0, 13, (2, 8))
    targets = torch.randint(0, 13, (2, 8))
    outputs = model(inputs, targets)

    assert outputs["logits"].shape == (2, 8, 13)
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])

    outputs["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_embedding_and_output_weights_are_tied() -> None:
    model = CharacterGPT(tiny_config())

    assert (
        model.token_embeddings.weight.data_ptr()
        == model.language_model_head.weight.data_ptr()
    )
    assert model.parameter_count() > 0


def test_greedy_generation_has_expected_length_and_range() -> None:
    torch.manual_seed(5)
    model = CharacterGPT(tiny_config())
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    generated = model.generate(
        prompt,
        max_new_characters=5,
        greedy=True,
    )

    assert generated.shape == (1, 11)
    assert torch.equal(generated[:, :6], prompt)
    assert generated.min().item() >= 0
    assert generated.max().item() < tiny_config().vocab_size


def test_no_prebuilt_transformer_or_attention_modules() -> None:
    model = CharacterGPT(tiny_config())

    banned_types = (
        nn.MultiheadAttention,
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    )

    assert not any(
        isinstance(module, banned_types)
        for module in model.modules()
    )

    source = inspect.getsource(model_module)
    assert "scaled_dot_product_attention" not in source