import math

import pytest
import torch

from evaluate_generate import (
    distinct_n,
    diversity_summary,
    repeated_ngram_rate,
)
from model import CharacterGPT, GPTConfig
from train import set_reproducible_seed


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


def test_distinct_n_uses_all_sequences_without_crossing_boundaries() -> None:
    sequences = [
        [1, 2, 1],
        [1, 2, 3],
    ]

    assert distinct_n(sequences, n=1) == pytest.approx(0.5)
    assert distinct_n(sequences, n=2) == pytest.approx(0.75)
    assert distinct_n(sequences, n=3) == pytest.approx(1.0)


def test_repeated_ngram_rate_counts_duplicate_ngrams() -> None:
    sequence = [1, 2, 3, 4, 1, 2, 3, 4]

    rate = repeated_ngram_rate(sequence, n=4)

    assert rate == pytest.approx(0.2)


def test_repeated_ngram_rate_is_zero_for_short_sequence() -> None:
    assert repeated_ngram_rate([1, 2, 3], n=4) == 0.0


def test_invalid_ngram_sizes_are_rejected() -> None:
    with pytest.raises(ValueError):
        distinct_n([[1, 2, 3]], n=0)

    with pytest.raises(ValueError):
        repeated_ngram_rate([1, 2, 3], n=-1)


def test_diversity_summary_reports_expected_values() -> None:
    sequences = [
        [1, 2, 3, 4, 1, 2, 3, 4],
        [5, 6, 7],
    ]

    summary = diversity_summary(sequences)

    assert summary["sample_count"] == 2
    assert summary["generated_tokens"] == 11
    assert summary["distinct_1"] == pytest.approx(7 / 11)
    assert summary["distinct_2"] == pytest.approx(6 / 9)
    assert summary["distinct_3"] == pytest.approx(5 / 7)
    assert summary["repeated_4gram_rate"] == pytest.approx(0.1)
    assert (
        summary["minimum_sample_repeated_4gram_rate"]
        == pytest.approx(0.0)
    )
    assert (
        summary["maximum_sample_repeated_4gram_rate"]
        == pytest.approx(0.2)
    )


def test_sampled_generation_is_reproducible_with_fixed_seed() -> None:
    set_reproducible_seed(99)
    model = CharacterGPT(tiny_config())
    model.eval()

    prompt = torch.tensor([[1, 2, 3]])

    set_reproducible_seed(123)
    first = model.generate(
        prompt,
        max_new_characters=5,
        temperature=0.8,
        greedy=False,
    )

    set_reproducible_seed(123)
    second = model.generate(
        prompt,
        max_new_characters=5,
        temperature=0.8,
        greedy=False,
    )

    assert first.shape == (1, 8)
    assert torch.equal(first, second)
    assert math.isfinite(float(first.float().mean()))