"""Audit saved confirmation metrics and frozen file hashes without retraining."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from confirm_disco import measures, files
ROOT=Path(__file__).resolve().parent

def main():
    protocol=json.loads((ROOT/'results_confirmation/protocol.json').read_text())
    for name,digest in {**protocol['files'],**protocol['code']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    original=list(csv.DictReader((ROOT/'results_confirmation/metrics.csv').open()))
    audited=[];reference=None
    for row in original:
        fraction=float(row['fraction']);seed=int(row['seed']);name=row['model']
        with np.load(ROOT/'results_confirmation'/f'f{fraction:g}_seed{seed}_{name}.npz') as data:
            ids=data['rows'];assert ids.min()>=200000 and ids.max()<300000
            assert len(np.unique(ids))==len(ids)
            if reference is None:reference=ids.copy()
            assert np.array_equal(ids,reference)
            folder,threshold=files(seed,fraction,name)
            assert np.isclose(float(data['threshold']),threshold,rtol=0,atol=1e-12)
            assert np.isfinite(data['score']).all()
            metrics=measures(data['mass'],data['truth'],data['score']>threshold)
            for key,value in zip(['sic','js','signal_efficiency','background_efficiency'],metrics):
                assert np.isclose(float(row[key]),value,rtol=1e-10,atol=1e-12),(name,key)
            archive=folder/('validation.npz' if name=='disco' else 'predictions.npz')
            with np.load(archive) as previous:
                for key in ['train_rows','rows' if name=='disco' else 'validation_rows']:
                    assert not np.intersect1d(ids,previous[key]).size
        audited.append(row)
    confirmed=json.loads((ROOT/'results_confirmation/confirmation.json').read_text())
    lines=['# Audited results table','', 'Metrics recomputed from all 18 saved confirmation prediction files. Frozen code/model/preprocessing/threshold-file SHA-256 hashes match the confirmation protocol. Fresh row IDs are identical across models, unique, and disjoint from saved training/validation IDs.', '',
        '| Training signal | Model | Mean SIC | Mean JS | Mean signal efficiency | Mean background efficiency |',
        '|---|---|---:|---:|---:|---:|']
    for fraction in [.005,.01]:
        means={}
        for name in ['mass_mlp','disco','efp_mlp']:
            group=[r for r in audited if float(r['fraction'])==fraction and r['model']==name]
            values=[np.mean([float(r[k]) for r in group]) for k in ['sic','js','signal_efficiency','background_efficiency']]
            means[name]=values
            lines.append(f'| {fraction:.1%} | {name} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        c=next(r for r in confirmed if r['fraction']==fraction)
        assert np.isclose(means['disco'][0]/means['mass_mlp'][0],c['sic_ratio'])
        assert np.isclose(means['disco'][1]/means['mass_mlp'][1],c['js_ratio'])
    lines+=['','| Training signal | SIC retained, 95% interval | JS reduction, 95% interval |','|---|---|---|']
    for r in confirmed:
        lo,hi=r['sic_interval'];jl,jh=r['js_interval']
        lines.append(f'| {r["fraction"]:.1%} | {100*r["sic_ratio"]:.2f}% [{100*lo:.2f}, {100*hi:.2f}] | {100*(1-r["js_ratio"]):.2f}% [{100*(1-jh):.2f}, {100*(1-jl):.2f}] |')
    lines+=['','Ratios are ratios of three-seed means, not means of per-seed ratios. SIC is a sensitivity proxy, not discovery significance. JS uses base-2 logarithms and 16 bins over 3100-3900 GeV.',
        'Bootstrap intervals are transcribed from the saved confirmation output, not recomputed by this quick audit. They are conditional paired stratified event-bootstrap intervals for fixed models, not training/simulator uncertainty or simultaneous coverage.',
        'Important: seed 11 at 1% injection has worse JS with DisCo; improvements are aggregate. EFP-only is a competitive simpler reference at 1%, but unstable at 0.5%.',
        'Full numerical per-seed values: results_confirmation/metrics.csv. Full interval regeneration: confirm_disco.py (uses cached features, frozen inference, and 300 paired bootstrap replicates).']
    (ROOT/'AUDITED_RESULTS.md').write_text('\n'.join(lines))
    print('PASS: 18 prediction metrics, frozen hashes, thresholds, fresh rows, split separation, and aggregate ratios. AUDITED_RESULTS.md generated.')

if __name__=='__main__':main()
