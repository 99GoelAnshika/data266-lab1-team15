"""Generate fixed, direct predictions from Viraat's own trained checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from checkpointing import file_sha256, verify_training_checkpoint
from data import build_eval_transform, discover_images
import train

MEMBER = Path(__file__).resolve().parents[1]
ROOT = MEMBER.parents[1]


def load_trained(run_id, checkpoint='latest.pt', device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = MEMBER / 'configs/cyclegan_viraat.json'
    config, digest = train.load_config(config_path)
    train.validate_canonical_config(config)
    path = MEMBER / 'checkpoints' / run_id / checkpoint
    verify_training_checkpoint(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload['config_sha256'] != digest or payload['metadata'].get('run_id') != run_id:
        raise RuntimeError('Checkpoint/config/run identity mismatch')
    networks = train.build_networks(config=config, device=device)
    for name, network in networks.items():
        network.load_state_dict(payload['network_state_dicts'][name], strict=True)
        network.eval()
    return config, networks, payload, path, device


def tensor_to_pil(x):
    # Range conversion and JPEG encoding only; no aesthetic postprocessing.
    pixels = x.detach().float().cpu().squeeze(0).clamp(-1, 1).add(1).div(2)
    return to_pil_image(pixels).convert('RGB')


def generate(run_id, checkpoint='latest.pt', n=300):
    config, networks, payload, ckpt, device = load_trained(run_id, checkpoint)
    paths_a = discover_images(ROOT / config['domains']['domain_a_directory'])
    paths_b = discover_images(ROOT / config['domains']['domain_b_directory'])
    if len(paths_a) < n or len(paths_b) < n:
        raise RuntimeError('Too few real input images for the fixed evaluation protocol')
    if payload['epoch_completed'] != config['training']['epochs']:
        raise RuntimeError('Default inference requires the completed planned run; avoid choosing checkpoints by test scores')
    dest = MEMBER / 'outputs' / run_id / 'predictions'
    if dest.exists():
        raise FileExistsError('Predictions already exist. Preserve them; do not silently overwrite.')
    transform = build_eval_transform(image_size=config['preprocessing']['evaluation_size'])
    manifest = {'run_id': run_id, 'checkpoint': ckpt.name, 'checkpoint_sha256': file_sha256(ckpt),
        'config_sha256': payload['config_sha256'], 'epoch': payload['epoch_completed'],
        'global_step': payload['global_step'], 'n_eval': n,
        'selection': 'first 300 sorted source filenames, as in the team evaluation protocol',
        'preprocessing': 'bicubic resize to 256; RGB normalized to [-1,1]',
        'encoding': 'JPEG quality=100, subsampling=0; direct generator output', 'records': []}
    start = time.perf_counter()
    with torch.inference_mode():
        for direction, sources, generator_name in [
            ('A2B', paths_a[:n], 'generator_a_to_b'),
            ('B2A', paths_b[:n], 'generator_b_to_a')]:
            folder = dest / ('pred_' + direction)
            folder.mkdir(parents=True, exist_ok=False)
            for index, source in enumerate(sources):
                with Image.open(source) as image:
                    x = transform(image.convert('RGB')).unsqueeze(0).to(device)
                output = networks[generator_name](x)
                if not torch.isfinite(output).all():
                    raise FloatingPointError('Nonfinite prediction: ' + source.name)
                target = folder / source.name
                tensor_to_pil(output).save(target, 'JPEG', quality=100, subsampling=0, optimize=False)
                manifest['records'].append({'direction': direction, 'source': source.name,
                    'prediction': target.name, 'prediction_sha256': file_sha256(target)})
                if (index + 1) % 50 == 0:
                    print(direction, index + 1, '/', n, flush=True)
    manifest['inference_seconds'] = time.perf_counter() - start
    (dest / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Private workspace avoids changing the teammate's existing class-evaluator inputs.
    workspace = MEMBER / 'outputs' / run_id / 'evaluator_workspace' / 'Part 3' / 'Data'
    for name, sources in [('monet_jpg', paths_a[:n]), ('photo_jpg', paths_b[:n]),
                          ('pred_A2B', sorted((dest / 'pred_A2B').glob('*.jpg'))),
                          ('pred_B2A', sorted((dest / 'pred_B2A').glob('*.jpg')))]:
        folder = workspace / name
        folder.mkdir(parents=True, exist_ok=False)
        for source in sources:
            shutil.copy2(source, folder / source.name)
    print('Predictions:', dest)
    print('Unchanged instructor evaluator working directory:', workspace.parents[1])
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='viraat_resizeconv6_run001')
    parser.add_argument('--checkpoint', default='latest.pt')
    args = parser.parse_args()
    generate(args.run_id, args.checkpoint)
