"""Quick CPU checks without datasets/checkpoints. Synthetic only."""
import numpy as np
import torch
from models import build_model
from metrics import full_metrics, counts_metrics, bootstrap_intervals
from runtime_utils import member_root, read_json, MODEL_NAMES


def main():
    torch.set_num_threads(2)
    root = member_root()
    for name in MODEL_NAMES:
        config = read_json(root / f'configs/{name}.json')
        config.update(embedding_dim=8, hidden_dim=8, channels=8, head_dim=4)
        model = build_model(config, 32).eval()
        ids, lengths = torch.tensor([[3, 4, 0, 0], [2, 0, 0, 0]]), torch.tensor([2, 1])
        assert torch.allclose(model(ids, lengths), model(torch.nn.functional.pad(ids, (0, 8)), lengths), atol=1e-6)
        model.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ids, lengths), torch.tensor([1., 0.]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metrics = full_metrics([0, 0, 1, 1], [.1, .8, .3, .9])
    assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]
    assert np.isclose(metrics['brier_score'], .2875) and np.isclose(metrics['ece_15_bins'], .425)
    protocol = read_json(root / 'configs/evaluation_protocol.json')
    protocol['bootstrap'].update(replicates=32, replicate_size=4)
    p = np.asarray([.1, .8, .3, .9])
    result = bootstrap_intervals([0, 0, 1, 1], {n: p for n in MODEL_NAMES}, protocol)
    assert result['intervals']['baseline'] == result['intervals']['experiment_1']
    print('PASS: synthetic CPU model, padding, gradient and metric checks. No real accuracy claim.')


if __name__ == '__main__':
    main()
