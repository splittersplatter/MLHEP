"""Frozen-model confirmation on raw rows 200000:300000. No training or tuning."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT, extract, region, score_model, BaselineEFPNetwork

SEEDS=[11,22,33]
FRACTIONS=[.005,.01]
MODELS=['mass_mlp','disco','efp_mlp']


def files(seed,fraction,name):
    if name=='disco':
        folder=ROOT/'results_tuning'/f'seed{seed}_f{fraction:g}_lambda0.003'
        threshold=json.loads((folder/'metrics.json').read_text())['threshold']
    else:
        folder=ROOT/'results_study'/f'seed{seed}_f{fraction:g}_{name}'
        rows=json.loads((folder/'metrics.json').read_text())
        threshold=next(r['threshold'] for r in rows if r['split']=='validation' and r['cut']==.05)
    return folder,float(threshold)


def predict(data,folder,name):
    with np.load(folder/'preprocessing.npz') as p:
        raw=np.log(np.maximum(data['x'],1e-12))
        if name!='efp_mlp': raw=np.column_stack((raw,np.log1p(np.maximum(data['jet_mass'],0))))
        active=p['active'];x=((raw[:,active]-p['mean'][active])/p['scale'][active]).astype('float32')
    model=BaselineEFPNetwork(x.shape[1])
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    return score_model(model,x)


def measures(mass,truth,selected):
    bg=truth==0;sr=region(mass)==1;signal=(truth==1)&sr
    eb=selected[bg&sr].mean();es=selected[signal].mean()
    a=np.histogram(mass[bg],np.linspace(3100,3900,17))[0]
    b=np.histogram(mass[bg&selected],np.linspace(3100,3900,17))[0]
    js=float(jensenshannon(a,b,base=2)**2) if b.sum() else np.nan
    return float(es/np.sqrt(eb)) if eb>0 else np.nan,js,float(es),float(eb)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    selection=json.loads((ROOT/'results_tuning/selection.json').read_text())
    if selection['completed_fits']!=24 or selection['selected_lambda']!=.003:
        raise ValueError('Expected completed tuning selecting lambda=0.003')
    source=json.loads((ROOT/'results_study/config.json').read_text())
    spec=argparse.Namespace(data=Path(source['data']),events=100000,start_row=200000,degree=4,chunk_size=128)
    output=ROOT/'results_confirmation';output.mkdir(exist_ok=True)
    fingerprints={}
    for seed in SEEDS:
        for fraction in FRACTIONS:
            for name in MODELS:
                folder,threshold=files(seed,fraction,name)
                for filename in ['model.pt','preprocessing.npz','metrics.json']:
                    path=folder/filename;fingerprints[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=dict(source_start=200000,source_stop=300000,seeds=SEEDS,fractions=FRACTIONS,lambda_value=.003,
        rule='Mean SIC ratio >=0.90 and mean JS ratio <=0.80 at both fractions',
        bootstrap='300 paired stratified event bootstrap replicates; models/seeds fixed; 95% percentile intervals',
        files=fingerprints,source_size=spec.data.stat().st_size,source_mtime=spec.data.stat().st_mtime_ns,
        code={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['confirm_disco.py','study.py','phase1.py']})
    encoded=json.dumps(protocol,indent=2,sort_keys=True);path=output/'protocol.json'
    if path.exists() and path.read_text()!=encoded: raise ValueError('Frozen protocol/artifacts changed; do not overwrite confirmation')
    if args.check_only:
        # Verify inference round trip against already-used validation outputs, never fresh data.
        with np.load(ROOT/'feature_cache/jets_ee22264f96aec21c.npz') as old:
            old_rows=old['row']
            for seed in SEEDS:
                for fraction in FRACTIONS:
                    for name in MODELS:
                        folder,_=files(seed,fraction,name)
                        archive=folder/('validation.npz' if name=='disco' else 'predictions.npz')
                        with np.load(archive) as saved:
                            rowkey,scorekey=('rows','score') if name=='disco' else ('validation_rows','validation_score')
                            rows=saved[rowkey][:32];expected=saved[scorekey][:32]
                        ids=np.searchsorted(old_rows,rows)
                        assert np.array_equal(old_rows[ids],rows)
                        data={k:old[k][ids] for k in ['x','jet_mass']}
                        assert np.allclose(predict(data,folder,name),expected,atol=2e-6),str(folder)
        print('All 18 model reload checks passed; no fresh events evaluated.');return
    path.write_text(encoded)
    # Separate 10k-row caches allow interrupted extraction to resume cheaply.
    blocks=[]
    for start in range(200000,300000,10000):
        spec.start_row=start;spec.events=10000
        blocks.append(extract(spec))
    data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    assert data['row'].min()>=200000 and data['row'].max()<300000
    keep=(data['truth']==0)|(region(data['mass'])==1)
    data={k:v[keep] for k,v in data.items()}
    mass,truth=data['mass'],data['truth'];results=[];decisions={};confirmation=[]
    for fraction in FRACTIONS:
        for seed in SEEDS:
            for name in MODELS:
                folder,threshold=files(seed,fraction,name)
                print(f'Evaluating frozen {name}, seed {seed}, training fraction {fraction}',flush=True)
                score=predict(data,folder,name);selected=score>threshold
                decisions[(fraction,seed,name)]=selected
                sic,js,es,eb=measures(mass,truth,selected)
                results.append(dict(fraction=fraction,seed=seed,model=name,sic=sic,js=js,signal_efficiency=es,background_efficiency=eb))
                np.savez_compressed(output/f'f{fraction:g}_seed{seed}_{name}.npz',rows=data['row'],mass=mass,truth=truth,score=score,threshold=threshold)
        # Paired event bootstrap: SAME resampled rows for both methods and all seeds.
        point={name:np.mean([[r['sic'],r['js']] for r in results if r['fraction']==fraction and r['model']==name],axis=0) for name in MODELS}
        ratios=point['disco']/point['mass_mlp'];samples=[]
        strata=[np.flatnonzero((truth==label)&(region(mass)==sr)) for label,sr in [(0,0),(0,1),(1,1)]]
        rng=np.random.default_rng(773)
        for _ in range(300):
            ids=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata])
            means=[]
            for name in ['mass_mlp','disco']:
                means.append(np.mean([measures(mass[ids],truth[ids],decisions[(fraction,s,name)][ids])[:2] for s in SEEDS],axis=0))
            samples.append(means[1]/means[0])
        low,high=np.quantile(samples,[.025,.975],axis=0)
        confirmation.append(dict(fraction=fraction,sic_ratio=float(ratios[0]),js_ratio=float(ratios[1]),
            sic_interval=[float(low[0]),float(high[0])],js_interval=[float(low[1]),float(high[1])],
            point_targets_met=bool(ratios[0]>=.9 and ratios[1]<=.8),
            intervals_clear_targets=bool(low[0]>=.9 and high[1]<=.8)))
        fig,ax=plt.subplots(figsize=(10,5));bins=np.linspace(3100,3900,17);bg=truth==0
        n=np.histogram(mass[bg],bins)[0]
        for name in MODELS:
            rates=np.array([np.histogram(mass[bg&decisions[(fraction,s,name)]],bins)[0]/n for s in SEEDS])
            ax.errorbar((bins[1:]+bins[:-1])/2,rates.mean(0),yerr=rates.std(0),fmt='.-',label=name)
        ax.axvspan(3300,3700,color='gray',alpha=.1);ax.axhline(.05,color='black',ls=':')
        ax.set(xlabel='Dijet mass [GeV]',ylabel='Background selection efficiency',title=f'Fresh confirmation; training signal {fraction:.1%}; error bars: seed SD')
        ax.legend();fig.tight_layout();fig.savefig(output/f'efficiency_f{fraction:g}.png',dpi=160);plt.close(fig)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (output/'confirmation.json').write_text(json.dumps(confirmation,indent=2,allow_nan=False))
    lines=['# Fresh-data confirmation',f'Raw source rows: 200000-299999. Selected evaluation events: {len(mass)}.',
        'Frozen lambda=0.003; no training, checkpoint selection or threshold tuning on this sample.',
        '', '| Training signal | SIC retained (95% interval) | JS ratio (95% interval) | Point targets met | Intervals clear targets |',
        '|---|---|---|---|---|']
    for r in confirmation:
        lines.append(f'| {r["fraction"]:.1%} | {r["sic_ratio"]:.3f} {r["sic_interval"]} | {r["js_ratio"]:.3f} {r["js_interval"]} | {r["point_targets_met"]} | {r["intervals_clear_targets"]} |')
    lines+=['','Both signal fractions must meet the point targets for the predefined empirical criterion.',
        'Intervals are paired stratified event-bootstrap intervals conditional on the three frozen models and event class/region counts. They do not include training or simulator uncertainty and are not simultaneous confidence bands.',
        'The same fresh events are shared across models and fractions. Do not treat seeds as independent test samples.',
        'Raw JS contains finite-count bias. A lower JS is not proof of independence or absence of a fake resonance.',
        'If confirmation fails, report failure; do not retune on these events. See metrics.csv for the EFP-only reference.']
    (output/'REPORT.md').write_text('\n'.join(lines));print(f'Done: {output/"REPORT.md"}')

if __name__=='__main__': main()
