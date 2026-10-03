"""CPU/GPU sentiment evaluator using a saved checkpoint and frozen tokenizer."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from models import build_model
from data_runtime import verify_preprocessing
from runtime_utils import MODEL_NAMES, member_root, read_json, sha256, write_json
from yelp_data import build_preprocessor, encode_tokens


def evaluate_texts(root, config_path, texts, device=None):
    root, config_path = Path(root), Path(config_path)
    verify_preprocessing(root)
    config = read_json(config_path)
    name = config['name']
    manifest = read_json(root / f'outputs/metrics/{name}_training_manifest.json')
    checkpoint_path = root / manifest['checkpoint']
    if sha256(config_path) != manifest['config_sha256'] or sha256(checkpoint_path) != manifest['checkpoint_sha256']:
        raise RuntimeError('Evaluator checkpoint/config integrity failed.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vocabulary = read_json(root / 'outputs/metrics/vocabulary.json')['token_to_index']
    protocol = read_json(root / 'configs/data_protocol.json')
    tokenizer = build_preprocessor(read_json(root / 'configs/english_stopwords.json'))
    encoded = [encode_tokens(tokenizer(text), vocabulary, protocol['sequence_length']) for text in texts]
    ids = torch.from_numpy(np.stack([e[0] for e in encoded]).astype(np.int64)).to(device)
    lengths = torch.tensor([e[1]['lengths'] for e in encoded], dtype=torch.long, device=device)
    model = build_model(config, len(vocabulary)).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        p = torch.sigmoid(model(ids, lengths).float()).cpu().numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError('Evaluator emitted invalid probabilities.')
    return [{'text': text, 'probability_positive': float(prob),
             'label': 'positive' if prob >= .5 else 'negative'} for text, prob in zip(texts, p)]


def smoke_all(root):
    root = Path(root)
    texts = ['The food was delicious and the service was friendly.',
             'The food was not good and the service was terrible.',
             r'Great food.\nClean café.', '']
    records = {}
    for name in MODEL_NAMES:
        records[name] = evaluate_texts(root, root / f'configs/{name}.json', texts, device='cpu')
    result = {'status': 'PASS', 'scope': 'saved-checkpoint CPU inference and probability validity',
              'examples_are_synthetic': True, 'not_a_test_accuracy_measurement': True,
              'predictions': records}
    write_json(root / 'outputs/metrics/evaluator_smoke.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--text', action='append')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = member_root()
    config = Path(args.config)
    if not config.is_absolute():
        config = Path.cwd() / config
    if not config.is_file():
        raise FileNotFoundError('Provide the relative config path from your working directory.')
    texts = args.text or ['Great food and friendly service.', 'The meal was not good.']
    if not args.smoke and not args.text:
        parser.error('Use --smoke or supply --text.')
    result = evaluate_texts(root, config, texts, device=args.device)
    print(json.dumps({'synthetic_smoke': bool(args.smoke), 'predictions': result}, indent=2))


if __name__ == '__main__':
    main()
