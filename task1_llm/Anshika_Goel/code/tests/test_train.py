import math

import torch
from torch.utils.data import TensorDataset

from data import SequenceTensorDataset
from model import CharacterGPT, GPTConfig
from train import (
    build_optimizer,
    evaluate,
    make_train_loader,
    scheduled_learning_rate,
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


def test_learning_rate_warmup_and_cosine_decay() -> None:
    maximum = 3.0e-4
    minimum = 3.0e-5
    total_steps = 100
    warmup_steps = 10

    rates = [
        scheduled_learning_rate(
            optimization_step=step,
            total_optimization_steps=total_steps,
            warmup_steps=warmup_steps,
            maximum_learning_rate=maximum,
            minimum_learning_rate=minimum,
        )
        for step in range(total_steps)
    ]

    assert math.isclose(rates[0], maximum / warmup_steps)
    assert math.isclose(rates[warmup_steps - 1], maximum)
    assert rates[warmup_steps] < maximum
    assert math.isclose(rates[-1], minimum)
    assert all(
        0.0 < rate <= maximum
        for rate in rates[:warmup_steps]
    )
    assert all(
        minimum <= rate <= maximum
        for rate in rates[warmup_steps:]
    )
    assert rates[:warmup_steps] == sorted(rates[:warmup_steps])
    assert rates[warmup_steps:] == sorted(
        rates[warmup_steps:],
        reverse=True,
    )


def test_optimizer_assigns_every_parameter_once() -> None:
    model = CharacterGPT(tiny_config())
    optimizer = build_optimizer(
        model=model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    )

    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    model_parameter_ids = [
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(optimizer.param_groups) == 2
    assert len(optimizer_parameter_ids) == len(
        set(optimizer_parameter_ids)
    )
    assert set(optimizer_parameter_ids) == set(
        model_parameter_ids
    )
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_training_loader_order_is_seeded() -> None:
    dataset = TensorDataset(torch.arange(20))

    first_loader = make_train_loader(
        dataset,
        batch_size=4,
        seed=123,
        num_workers=0,
        pin_memory=False,
    )
    second_loader = make_train_loader(
        dataset,
        batch_size=4,
        seed=123,
        num_workers=0,
        pin_memory=False,
    )
    different_loader = make_train_loader(
        dataset,
        batch_size=4,
        seed=456,
        num_workers=0,
        pin_memory=False,
    )

    first_order = torch.cat(
        [batch[0] for batch in first_loader]
    )
    second_order = torch.cat(
        [batch[0] for batch in second_loader]
    )
    different_order = torch.cat(
        [batch[0] for batch in different_loader]
    )

    assert torch.equal(first_order, second_order)
    assert not torch.equal(first_order, different_order)


def test_validation_metrics_are_finite() -> None:
    torch.manual_seed(7)
    model = CharacterGPT(tiny_config())
    sequence_tensor = torch.randint(0, 13, (4, 9))
    dataset = SequenceTensorDataset(sequence_tensor)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    metrics = evaluate(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        use_mixed_precision=False,
    )

    assert math.isfinite(float(metrics["loss"]))
    assert 0.0 <= float(metrics["top1_accuracy"]) <= 1.0
    assert metrics["tokens"] == 32
    assert float(metrics["seconds"]) > 0.0
    assert float(metrics["tokens_per_second"]) > 0.0