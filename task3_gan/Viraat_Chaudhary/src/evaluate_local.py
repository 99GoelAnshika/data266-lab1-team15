"""Both-direction local metrics, real human-audit forms, and training plots.

The instructor evaluation notebook was not in the uploaded ZIP. This
script supplies a local reproduction, not an unverified official score.
Use the unchanged class evaluator before creating the official submission.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
from torch import nn

from data import build_eval_transform, discover_images
from infer import MEMBER, ROOT, load_trained, tensor_to_pil
from metric_utils import (fid_sample_space, class_mifid, kid,
                          generative_precision_recall, normalize, agreement)


def save_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_extractor(device):
    # Pretrained networks are used only to measure images, never to generate them.
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                transform_input=False)
    model.fc = nn.Identity()
    return model.eval().to(device), transforms.Compose([
        transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])


@torch.inference_mode()
def activations(paths, model, transform, device, batch_size=16):
    batches = []
    for i in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[i:i+batch_size]:
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        values = model(torch.stack(tensors).to(device))
        batches.append(values.float().cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def human_package(run_id, sources, predictions):
    folder = MEMBER / 'outputs' / run_id / 'human_audit'
    if folder.exists():
        return  # Never overwrite forms a person may already have completed.
    folder.mkdir(parents=True)
    rng = np.random.default_rng(266)
    chosen = [(direction, int(i)) for direction in ['A2B','B2A']
              for i in rng.choice(300, 15, replace=False)]
    rng.shuffle(chosen)
    rows, private = [], []
    for n, (direction, i) in enumerate(chosen, 1):
        audit_id = f'AUDIT_{n:03d}'
        canvas = Image.new('RGB',(512,280),'white')
        for x, path in [(0, sources[direction][i]), (256,predictions[direction][i])]:
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB').resize((256,256)), (x,24))
        draw = ImageDraw.Draw(canvas)
        draw.text((8,5),'Input',fill='black'); draw.text((264,5),'Translation',fill='black')
        canvas.save(folder / (audit_id + '.png'))
        rows.append({'audit_id':audit_id,'style':'','content':'','artifacts':''})
        private.append({'audit_id':audit_id,'direction':direction,'source':sources[direction][i].name})
    for rater in [1,2]:
        save_csv(folder / f'rater{rater}_scores.csv', rows)
    save_csv(folder / 'private_manifest.csv', private)
    (folder / 'instructions.md').write_text(
        'Two real raters independently score all 30 anonymized image panels.\n'
        'Use integer 1-5 for style match, content preservation, and freedom from artifacts '
        '(5 means best for each criterion). Do not reveal the model author or checkpoint.\n'
        'Do not simulate scores. Fill each rater CSV independently; keep private_manifest.csv from raters.\n')


def analyze_human(run_id):
    folder = MEMBER / 'outputs' / run_id / 'human_audit'
    with (folder/'rater1_scores.csv').open() as f:
        a = list(csv.DictReader(f))
    with (folder/'rater2_scores.csv').open() as f:
        b = list(csv.DictReader(f))
    if len(a)!=30 or len(b)!=30 or [r['audit_id'] for r in a] != [r['audit_id'] for r in b]:
        raise ValueError('Both real raters must score the same 30 audit IDs')
    private = {r['audit_id']:r['direction'] for r in csv.DictReader((folder/'private_manifest.csv').open())}
    output = {}
    for direction in ['A2B','B2A','both']:
        output[direction] = {}
        indices = [i for i,r in enumerate(a) if direction=='both' or private[r['audit_id']]==direction]
        for criterion in ['style','content','artifacts']:
            scores_a, scores_b = [], []
            for i in indices:
                x,y = int(a[i][criterion]), int(b[i][criterion])
                if not (1<=x<=5 and 1<=y<=5):
                    raise ValueError('Human scores must be integers 1-5')
                scores_a.append(x);scores_b.append(y)
            observed,kappa = agreement(scores_a,scores_b)
            output[direction][criterion] = {'mean_score':float(np.mean(scores_a+scores_b)),
                'agreement_fraction':observed,'cohens_kappa':kappa}
    (folder/'human_audit_results.json').write_text(json.dumps(output,indent=2)+'\n')
    for report_name in ['metrics_report.csv','full_metrics_report.csv']:
        report=MEMBER/report_name
        if not report.exists():
            continue
        with report.open() as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            if row['status']!='pending_real_raters':
                continue
            scores=output[row['direction']]
            mapping={'human_style_score':'style','human_content_score':'content',
                     'human_artifacts_score':'artifacts'}
            if row['metric'] in mapping:
                row['value']=scores[mapping[row['metric']]]['mean_score']
            else:
                ids=[i for i,r in enumerate(a) if private[r['audit_id']]==row['direction']]
                aa=[int(a[i][c]) for i in ids for c in ['style','content','artifacts']]
                bb=[int(b[i][c]) for i in ids for c in ['style','content','artifacts']]
                fraction,kappa=agreement(aa,bb)
                row['value']=fraction if row['metric']=='human_inter_rater_agreement' else kappa
            row['status']='computed_real_raters' if row['value'] is not None else 'undefined_constant_ratings'
            row['evidence']=f'outputs/{run_id}/human_audit/human_audit_results.json'
        save_csv(report,rows)
    return output


def training_summary(run_id, metric_rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log = MEMBER / 'logs' / run_id
    manifest = json.loads((log/'run_manifest.json').read_text())['manifest']
    events = [json.loads(s) for s in (log/'events.jsonl').read_text().splitlines() if s]
    epochs = [e['payload'] for e in events if e['event_type']=='epoch_end']
    final = epochs[-1]
    total_time = sum(e['duration_seconds'] for e in epochs)
    peak = max(e['peak_cuda_memory_mib'] for e in epochs)
    values = {'parameter_count':manifest['total_trainable_parameters'],
        'training_seconds':total_time,
        'images_per_second':sum(e['image_count'] for e in epochs)/total_time,
        'peak_cuda_memory_mib':peak,
        'nonfinite_gradient_count':sum(sum(e['nonfinite_gradient_sums'].values()) for e in epochs)}
    for k,v in values.items():
        metric_rows.append({'direction':'both','metric':k,'value':v,'status':'computed','evidence':'logs/'+run_id+'/events.jsonl'})
    for k,v in final['metric_means'].items():
        if k.startswith(('loss_','grad_norm_')):
            metric_rows.append({'direction':'both','metric':'final_epoch_'+k,'value':v,
                                'status':'computed','evidence':'logs/'+run_id+'/events.jsonl'})
    groups = {'generator_losses':['loss_generator_total','loss_gan_a_to_b','loss_gan_b_to_a'],
        'discriminator_losses':['loss_discriminator_a','loss_discriminator_b'],
        'cycle_identity_losses':['loss_cycle_a','loss_cycle_b','loss_identity_a','loss_identity_b'],
        'gradient_norms':['grad_norm_generator','grad_norm_discriminator_a','grad_norm_discriminator_b']}
    for name,keys in groups.items():
        fig,ax = plt.subplots(figsize=(8,4.5))
        for key in keys:
            ax.plot([e['epoch'] for e in epochs],[e['metric_means'][key] for e in epochs],label=key)
        ax.set(xlabel='Epoch',ylabel='Epoch mean',title=name.replace('_',' ').title())
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
    save_csv(output/'training_epoch_metrics.csv', [dict(epoch=e['epoch'],
        learning_rate=e['learning_rate'],duration_seconds=e['duration_seconds'],**e['metric_means']) for e in epochs])
    (output/'training_summary.json').write_text(json.dumps({'hardware':manifest['runtime'],
        'completed_epochs':len(epochs),**values},indent=2)+'\n')


def evaluate(run_id):
    import lpips

    config, networks, payload, ckpt, device = load_trained(run_id)
    pred_root = MEMBER/'outputs'/run_id/'predictions'
    inference = json.loads((pred_root/'inference_manifest.json').read_text())
    from checkpointing import file_sha256
    if inference['checkpoint_sha256'] != file_sha256(ckpt):
        raise RuntimeError('Predictions do not match the current checkpoint')
    paths_a = discover_images(ROOT/config['domains']['domain_a_directory'])[:300]
    paths_b = discover_images(ROOT/config['domains']['domain_b_directory'])[:300]
    predicted = {d:discover_images(pred_root/('pred_'+d)) for d in ['A2B','B2A']}
    sources = {'A2B':paths_a,'B2A':paths_b}
    for d in sources:
        if len(predicted[d])!=300 or [p.name for p in predicted[d]] != [p.name for p in sources[d]]:
            raise RuntimeError('Fixed source/prediction ordering mismatch')
    output = MEMBER/'outputs'/run_id/'metrics'
    if output.exists():
        raise FileExistsError('Metric artifacts already exist; preserve prior evidence')
    output.mkdir(parents=True)
    extractor,feature_tf = feature_extractor(device)
    features = {}
    for key,paths in [('real_a',paths_a),('real_b',paths_b),('pred_A2B',predicted['A2B']),('pred_B2A',predicted['B2A'])]:
        print('Extracting',key,flush=True)
        features[key] = activations(paths,extractor,feature_tf,device)
    np.savez_compressed(output/'inception_features.npz',**features)
    rows, directional = [], {}
    for direction,target,source in [('A2B','real_b','real_a'),('B2A','real_a','real_b')]:
        x,y = features[target],features['pred_'+direction]
        kmean,kstd = kid(x,y)
        precision,recall = generative_precision_recall(x,y)
        source_cos = np.sum(normalize(features[source])*normalize(y),axis=1)
        values = {'fid':fid_sample_space(x,y),'class_mifid':class_mifid(x,y),
            'kid_mean':kmean,'kid_std':kstd,'generative_precision':precision,'generative_recall':recall,
            'content_preservation_cosine_similarity':float(source_cos.mean())}
        directional[direction] = values
        for k,v in values.items():
            rows.append({'direction':direction,'metric':k,'value':v,'status':'computed',
                         'evidence':f'outputs/{run_id}/metrics/inception_features.npz'})
    del extractor
    if device.type=='cuda':
        torch.cuda.empty_cache()
    perceptual = lpips.LPIPS(net='alex').eval().to(device)
    transform = build_eval_transform(image_size=256)
    per_sample = []
    ga,gb = networks['generator_a_to_b'], networks['generator_b_to_a']
    with torch.inference_mode():
        for direction,forward,reverse in [('A2B',ga,gb),('B2A',gb,ga)]:
            for source in sources[direction]:
                with Image.open(source) as image:
                    x=transform(image.convert('RGB')).unsqueeze(0).to(device)
                generated=forward(x);recovered=reverse(generated)
                l1=float((recovered-x).abs().mean()/2)  # RGB [0,1] scale, as baseline.
                perceptual_error=float(perceptual(x,recovered).mean())
                per_sample.append({'direction':direction,'source':source.name,
                    'cycle_l1_rgb01':l1,'cycle_lpips_alex':perceptual_error})
                if len(per_sample)%100==0:
                    print('Cycle metrics:',len(per_sample),'/600',flush=True)
    save_csv(output/'cycle_metrics_per_sample.csv',per_sample)
    for direction in ['A2B','B2A']:
        selected=[r for r in per_sample if r['direction']==direction]
        for key in ['cycle_l1_rgb01','cycle_lpips_alex']:
            value=float(np.mean([r[key] for r in selected]))
            directional[direction][key]=value
            rows.append({'direction':direction,'metric':key,'value':value,'status':'computed',
                         'evidence':f'outputs/{run_id}/metrics/cycle_metrics_per_sample.csv'})
    local_fid=float(np.mean([v['fid'] for v in directional.values()]))
    local_mifid=float(np.mean([v['class_mifid'] for v in directional.values()]))
    save_csv(output/'submission_local_reproduction.csv',[{'ID':1,'FID':local_fid,'MiFID':local_mifid}])
    training_summary(run_id,rows,output)
    human_package(run_id,sources,predicted)
    for direction in ['A2B','B2A']:
        for metric in ['human_style_score','human_content_score','human_artifacts_score',
                       'human_inter_rater_agreement','human_cohens_kappa']:
            rows.append({'direction':direction,'metric':metric,'value':'','status':'pending_real_raters',
                         'evidence':f'outputs/{run_id}/human_audit'})
    for metric in ['kaggle_public_score','kaggle_private_score','kaggle_rank']:
        rows.append({'direction':'both','metric':metric,'value':'','status':'pending_official_submission','evidence':''})
    save_csv(MEMBER/'metrics_report.csv',rows)
    save_csv(MEMBER/'full_metrics_report.csv',rows)
    summary={'run_id':run_id,'epoch':payload['epoch_completed'],'checkpoint_sha256':inference['checkpoint_sha256'],
        'protocol':'team reference: first 300 sorted images, Inception-v3 ImageNet features, fixed ordering',
        'directions':directional,'local_average_fid':local_fid,'local_average_class_mifid':local_mifid,
        'official_evaluator_verified':False,'real_human_audit_completed':False}
    (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Local metrics ready. Run the unchanged instructor evaluator before using an official submission.csv.')
    print('Two real raters must complete the human-audit forms; their results are still pending.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='viraat_resizeconv6_run001')
    parser.add_argument('--human-only',action='store_true')
    args=parser.parse_args()
    if args.human_only:
        print(json.dumps(analyze_human(args.run_id),indent=2))
    else:
        evaluate(args.run_id)
