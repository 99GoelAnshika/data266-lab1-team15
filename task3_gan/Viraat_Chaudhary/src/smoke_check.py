"""One-command CPU smoke test of actual training and completed-epoch resume."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image


def main():
    member=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='viraat-task3-smoke-') as tmp:
        root=Path(tmp)
        dest=root/'task3_gan/Viraat_Chaudhary'
        shutil.copytree(member/'src',dest/'src',ignore=shutil.ignore_patterns('__pycache__','*.ipynb'))
        (dest/'configs').mkdir()
        config=json.loads((member/'configs/cyclegan_viraat.json').read_text())
        config['model']['generator']['base_channels']=8
        config['model']['discriminator']['base_channels']=8
        config['training'].update(epochs=2,constant_lr_epochs=1,linear_decay_epochs=1,batch_size=1)
        config['preprocessing'].update(train_resize=64,train_crop=64,evaluation_size=64)
        config['dataloader'].update(num_workers=0,persistent_workers=False)
        config_path=dest/'configs/cyclegan_viraat.json'
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        rng=np.random.default_rng(266)
        for domain in ['monet_jpg','photo_jpg']:
            d=root/'task3_gan/data'/domain;d.mkdir(parents=True)
            for i in range(2):
                Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(d/f'{i}.jpg')
        command=[sys.executable,'-u',str(dest/'src/train.py'),'--config',str(config_path),
                 '--run-id','synthetic_smoke','--mode','smoke','--device','cpu',
                 '--smoke-epochs','1','--sample-count','1']
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for extra in [[],['--resume']]:
            result=subprocess.run(command+extra,cwd=root,env=env,text=True,capture_output=True)
            if result.returncode:
                print(result.stdout);print(result.stderr)
                raise RuntimeError('Training/resume smoke test failed')
        latest=dest/'checkpoints/synthetic_smoke/latest.pt'
        payload=torch.load(latest,map_location='cpu',weights_only=False)
        assert payload['epoch_completed']==2 and payload['global_step']==4
        events=[json.loads(s) for s in (dest/'logs/synthetic_smoke/events.jsonl').read_text().splitlines()]
        assert any(e['event_type']=='run_resume' for e in events)
        assert len([e for e in events if e['event_type']=='epoch_end'])==2
        assert all(v==0 for e in events if e['event_type']=='epoch_end'
                   for v in e['payload']['nonfinite_gradient_sums'].values())
    print('PASS: real CPU training CLI, checkpoints, resume, four optimizer steps, finite gradients.')
    print('Synthetic smoke evidence is temporary and is not used as production results.')


if __name__=='__main__':
    main()
