"""Checks target leakage, gradients, metric definitions and real failure evidence."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import CharacterSequences, CharacterVocabulary, collect_text, make_rows
from evaluate_generate import distinct_n, repeated_ngram_rate
from model import CharGPT, DecoderConfig, LayerNormalization
from report import validate_cases
from train import learning_rate, loader


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(26615)


def tiny():
    return CharGPT(DecoderConfig(16, 12, 16, 4, 2, 32, 0.0))


def test_character_mapping_shift_and_unknown():
    vocabulary = CharacterVocabulary.from_text("abcabcabc")
    assert vocabulary.decode(vocabulary.encode("cab")) == "cab"
    assert vocabulary.encode("z").tolist() == [0]
    rows = make_rows("abcabcabc", 2, 3, vocabulary)
    x, y = CharacterSequences(rows)[0]
    assert x.tolist() == vocabulary.encode("abc").tolist()
    assert y.tolist() == vocabulary.encode("bca").tolist()
    assert x.dtype == y.dtype == torch.long


def test_story_records_are_split_before_windowing():
    stories = [{"text": "abcdef"}, {"text": "ghijkl"}, {"text": "mnopqr"}]
    train, index, count = collect_text(stories, 0, 8, "text", "|")
    validation, end, _ = collect_text(stories, index, 4, "text", "|")
    assert (train, index, count) == ("abcdef|g", 2, 2)
    assert validation == "mnop" and end == 3


def test_manual_normalization_matches_reference_and_gradient():
    norm = LayerNormalization(8, 1e-5).double()
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    assert torch.allclose(norm(x), F.layer_norm(x, (8,), norm.scale, norm.shift, 1e-5), atol=1e-10)
    assert torch.autograd.gradcheck(norm, (x,))


def test_future_characters_cannot_change_past_logits():
    model = tiny().eval()
    original = torch.randint(0, 16, (2, 12))
    altered = original.clone()
    altered[:, 6:] = (altered[:, 6:] + 1) % 16
    assert torch.allclose(model(original)[:, :6], model(altered)[:, :6], atol=1e-6)


def test_mask_probabilities_and_independent_output_weights():
    model = tiny().eval()
    _, weights = model.blocks[0].attention(torch.randn(2, 12, 16), return_weights=True)
    assert torch.count_nonzero(weights.triu(1)) == 0
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert model.output_head.weight.data_ptr() != model.token_embedding.weight.data_ptr()


def test_all_parameters_receive_finite_gradients():
    model = tiny()
    x, y = torch.randint(0, 16, (2, 12)), torch.randint(0, 16, (2, 12))
    F.cross_entropy(model(x).reshape(-1, 16), y.reshape(-1)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_bfloat16_autocast_preserves_finite_loss_and_gradients():
    model = tiny()
    x, y = torch.randint(0, 16, (2, 12)), torch.randint(0, 16, (2, 12))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, 16), y.reshape(-1))
    assert logits.dtype == torch.bfloat16 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_tiny_batch_can_be_learned():
    model = tiny()
    x, y = torch.randint(0, 16, (2, 12)), torch.randint(0, 16, (2, 12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    start = F.cross_entropy(model(x).reshape(-1, 16), y.reshape(-1)).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, 16), y.reshape(-1))
        loss.backward()
        optimizer.step()
    assert F.cross_entropy(model(x).reshape(-1, 16), y.reshape(-1)).item() < start * 0.3


def test_diversity_does_not_cross_sample_boundaries():
    assert distinct_n([[1, 2, 1], [1, 2, 3]], 2) == pytest.approx(3 / 4)
    assert repeated_ngram_rate([1] * 6) == pytest.approx(2 / 3)
    assert distinct_n([[1]], 2) == repeated_ngram_rate([1, 2, 3]) == 0


def test_warmup_cosine_and_full_epoch_last_batch():
    rates = [learning_rate(s, 10, 2, 1.0, 0.1) for s in range(10)]
    assert rates[:2] == [0.5, 1.0]
    assert rates[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))
    dataset = CharacterSequences(np.zeros((10, 5), dtype=np.uint16))
    assert sum(len(x) for x, _ in loader(dataset, 4, seed=15)) == 10


def test_generation_restores_mode_and_is_reproducible():
    model = tiny().train()
    prompt = torch.tensor([[1, 2, 3]])
    torch.manual_seed(42)
    a = model.generate(prompt, 5)
    torch.manual_seed(42)
    b = model.generate(prompt, 5)
    assert torch.equal(a, b) and model.training and a.shape == (1, 8)


def test_fabricated_failure_excerpt_is_rejected():
    samples = [{"sample_id": str(i), "continuation": "real generated words"} for i in range(3)]
    cases = [
        {
            "sample_id": str(i),
            "excerpt": "invented",
            "failure_type": "grammar",
            "observation": "bad grammar",
            "testable_fix": "measure another temperature",
        }
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="exact substrings"):
        validate_cases(samples, cases)
    for case in cases:
        case["excerpt"] = "generated words"
    assert len(validate_cases(samples, cases)) == 3


def test_checkpoint_reload_reproduces_logits(tmp_path):
    model = tiny().eval()
    inputs = torch.randint(0, 16, (2, 12))
    path = tmp_path / "weights.pt"
    torch.save({"configuration": model.configuration(), "weights": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = CharGPT(DecoderConfig(**saved["configuration"])).eval()
    restored.load_state_dict(saved["weights"])
    assert torch.equal(model(inputs), restored(inputs))
