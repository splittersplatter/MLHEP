"""Validation-only weak-DisCo search. Run locally; no test predictions are read."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from study import ROOT, partitions, inject, prepare, train_mlp, score_model, metrics_for, region

STRENGTHS = [0.003]
SEEDS = [11, 22, 33]
FRACTIONS = [0.005, 0.01]


def select_strength(rows):
    """Predeclared exploratory criterion: >=90% SIC, <=80% JS in BOTH regimes."""
    summary=[]
    for strength in STRENGTHS:
        item={'lambda':strength,'eligible':True,'regimes':[]}
        for fraction in FRACTIONS:
            base=[r for r in rows if r['lambda']==0 and r['fraction']==fraction]
            runs=[r for r in rows if r['lambda']==strength and r['fraction']==fraction]
            if len(base)!=len(SEEDS) or len(runs)!=len(SEEDS):
                item['eligible']=False
                continue
            if any(r['sic'] is None or r['js_divergence'] is None for r in base+runs):
                item['eligible']=False
                continue
            b_sic=np.mean([r['sic'] for r in base]); b_js=np.mean([r['js_divergence'] for r in base])
            sic=np.mean([r['sic'] for r in runs]); js=np.mean([r['js_divergence'] for r in runs])
            ratio=float(sic/b_sic) if b_sic>0 else 0.
            jsratio=float(js/b_js) if b_js>0 else float('inf')
            item['regimes'].append(dict(fraction=fraction,sic=float(sic),js=float(js),sic_retained=ratio,js_ratio=jsratio))
            item['eligible'] &= ratio>=.9 and jsratio<=.8
        summary.append(item)
    eligible=[r for r in summary if r['eligible'] and len(r['regimes'])==2]
    best=min(eligible,key=lambda r:(np.mean([x['js_ratio'] for x in r['regimes']]),r['lambda'])) if eligible else None
    return dict(selected_lambda=best['lambda'] if best else None,candidates=summary,
                interpretation='Fixed lambda 0.003 chosen by user; eligible indicates whether validation targets are met, not a selection among strengths.')


def find_development_cache():
    caches = []
    for candidate in (ROOT/'feature_cache').glob('jets_*.npz'):
        with np.load(candidate) as saved:
            metadata = json.loads(str(saved['metadata']))
            if metadata.get('events') == 200000 and metadata.get('degree') == 4 and metadata.get('version') == 2:
                caches.append(candidate)
    if len(caches) != 1:
        raise ValueError(f'Expected one 200k feature cache, found {len(caches)}')
    return caches[0]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results_tuning')
    parser.add_argument('--max-new-fits',type=int,default=6,help='Limit this invocation; rerun to resume')
    args=parser.parse_args()
    if args.max_new_fits<0: parser.error('max-new-fits must be nonnegative')
    source=ROOT/'results_study'
    original=json.loads((source/'config.json').read_text())
    cache=find_development_cache()
    if not cache.exists(): raise FileNotFoundError(f'Required existing cache missing: {cache}. No raw-data extraction is attempted.')
    with np.load(cache) as saved:
        meta=json.loads(str(saved['metadata']))
        if meta['events']!=200000 or meta['degree']!=4 or meta['version']!=2:
            raise ValueError('Unexpected cache provenance')
        data={k:saved[k] for k in ('x','mass','truth','row','jet_mass')}
    args.epochs=original['epochs']
    protocol=dict(version=1,lambdas=STRENGTHS,seeds=SEEDS,fractions=FRACTIONS,epochs=args.epochs,
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        criterion='Fixed lambda 0.003: assess whether mean validation SIC >=90% baseline and mean JS <=80% baseline in each fraction.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('study.py','tune_disco.py')})
    args.output.mkdir(parents=True,exist_ok=True)
    config=args.output/'protocol.json'
    encoded=json.dumps(protocol,indent=2,sort_keys=True)
    if config.exists() and config.read_text()!=encoded: raise ValueError('Protocol changed: use a new output directory')
    config.write_text(encoded)
    torch.set_num_threads(2)
    rows=[]; new_fits=0
    for seed in SEEDS:
        train_pool,valid_pool,excluded_test=partitions(data,seed)
        sr=region(data['mass']); truth=data['truth']
        valid_eval=valid_pool[(truth[valid_pool]==0)|(sr[valid_pool]==1)]
        for fraction in FRACTIONS:
            base_dir=source/f'seed{seed}_f{fraction:g}_mass_mlp'
            # Open ONLY validation arrays; test arrays in the archive are never loaded.
            with np.load(base_dir/'predictions.npz') as saved:
                assert np.array_equal(saved['validation_rows'],data['row'][valid_eval])
                base_score=saved['validation_score']
            threshold=float(np.quantile(base_score[sr[valid_eval]==0],.95))
            base=metrics_for(data,valid_eval,base_score,threshold,seed+300,199)
            rows.append(dict(seed=seed,fraction=fraction,**{'lambda':0.},**base))
            train=inject(train_pool,truth,sr,fraction,seed)
            valid=inject(valid_pool,truth,sr,fraction,seed+1)
            assert not(set(train)&set(valid_eval) or set(train)&set(excluded_test) or set(valid_eval)&set(excluded_test))
            x,preprocessing=prepare(data,train,True)
            for strength in STRENGTHS:
                directory=args.output/f'seed{seed}_f{fraction:g}_lambda{strength:g}'
                done=directory/'metrics.json'
                if done.exists(): rows.append(json.loads(done.read_text())); continue
                if new_fits>=args.max_new_fits: continue
                print(f'Fit {new_fits+1}: seed={seed}, SR signal={fraction:.1%}, lambda={strength}',flush=True)
                model,history=train_mlp(x,data,train,valid,seed,'sideband',strength,args)
                score=score_model(model,x[valid_eval])
                threshold=float(np.quantile(score[sr[valid_eval]==0],.95))
                result=metrics_for(data,valid_eval,score,threshold,seed+300,199)
                # All-tied/collapsed output is a failed candidate, never an infinite SIC.
                for key,value in result.items():
                    if isinstance(value,float) and not np.isfinite(value): result[key]=None
                row=dict(seed=seed,fraction=fraction,**{'lambda':strength},threshold=threshold,**result)
                directory.mkdir(exist_ok=True)
                torch.save(model.state_dict(),directory/'model.pt')
                np.savez_compressed(directory/'preprocessing.npz',**preprocessing)
                np.savez_compressed(directory/'validation.npz',rows=data['row'][valid_eval],score=score,
                                    mass=data['mass'][valid_eval],truth=truth[valid_eval],train_rows=data['row'][train])
                np.savetxt(directory/'training.csv',history,delimiter=',',header='epoch,train_bce,train_dcor2,val_bce,val_dcor2,val_objective')
                done.write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row);new_fits+=1
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (args.output/'validation_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    completed=sum(r['lambda']!=0 for r in rows)
    result=select_strength(rows)
    result['completed_fits']=completed
    if completed!=6: result['selected_lambda']=None
    (args.output/'selection.json').write_text(json.dumps(result,indent=2))
    lines=['# Fixed-lambda DisCo validation',f'Completed: {completed}/6 new fits.',
           'Fixed lambda: 0.003 (specified by user)',
           'Qualification compares the fixed lambda against the unregularized baseline; no lambda is selected.',
           '', '| Lambda | SR injection | SIC retained / baseline | JS / baseline | Qualifies in both regimes |',
           '|---|---|---|---|---|']
    for candidate in result['candidates']:
        for regime in candidate['regimes']:
            lines.append(f'| {candidate["lambda"]} | {regime["fraction"]:.1%} | {regime["sic_retained"]:.3f} | {regime["js_ratio"]:.3f} | {candidate["eligible"]} |')
    lines+=['','If no lambda qualifies, do not claim a successful fix or relax the rule after seeing results.',
            'Truth labels are used for simulation validation metrics, not the sideband training penalty.',
            'Shared validation samples and checkpoint selection make these tuning results optimistic; no statistical significance is claimed.']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print(f'Completed {completed}/6. Read {args.output / "REPORT.md"}',flush=True)

if __name__=='__main__': main()
