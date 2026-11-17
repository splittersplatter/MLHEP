from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'report/figures'
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
colors={'mass_mlp':'#c45b22','disco':'#087f8c','efp_mlp':'#5442a3'}
labels={'mass_mlp':'Mass-feature MLP','disco':'Sideband DisCo','efp_mlp':'EFP-only MLP'}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,1.05));ax.axis('off')
texts=['Particles\nHDF5 chunks','Two jets\nEFPs + masses','SR vs SB\nBCE + DisCo','Validation\nselect lambda','Fresh events\nfrozen models']
for i,t in enumerate(texts):
    x=.09+i*.205;ax.text(x,.55,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.4',fc='#edf3f5',ec='#46606b'),fontsize=10,transform=ax.transAxes)
    if i<4:ax.annotate('',xy=(x+.13,.55),xytext=(x+.077,.55),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#46606b'))
save(fig,'fig1_workflow')
bins=np.linspace(3100,3900,17);centers=(bins[1:]+bins[:-1])/2
fig,axes=plt.subplots(1,2,figsize=(7.2,2.1))
for name in ['efp_mlp','mass_mlp']:
    d=np.load(ROOT/f'results_study/seed11_f0.01_{name}/predictions.npz');b=d['test_truth']==0;m=d['test_mass'][b]
    rec=json.loads((ROOT/f'results_study/seed11_f0.01_{name}/metrics.json').read_text());cut=next(r['threshold'] for r in rec if r['split']=='test' and r['cut']==.05)
    sel=d['test_score'][b]>cut
    if name=='efp_mlp':axes[0].hist(m,bins=bins,density=True,histtype='step',color='black',label='All QCD')
    axes[0].hist(m[sel],bins=bins,density=True,histtype='step',color=colors[name],label=labels[name])
    axes[1].plot(centers,np.histogram(m[sel],bins)[0]/np.histogram(m,bins)[0],'.-',color=colors[name],label=labels[name])
for ax in axes:ax.axvspan(3300,3700,color='gray',alpha=.1);ax.set_xlabel('Dijet mass [GeV]');ax.grid(alpha=.2)
axes[0].set_ylabel('Density / GeV');axes[0].legend();axes[1].set_ylabel('QCD selection efficiency');fig.tight_layout();save(fig,'fig2_baseline')
rows=list(csv.DictReader((ROOT/'results_tuning/validation_summary.csv').open()));old=list(csv.DictReader((ROOT/'results_study/summary.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(7.2,2.35));lams=[.001,.003,.01,.03,.1,1]
for f,c in [(.005,'#126b9c'),(.01,'#b85519')]:
    bs=np.mean([float(r['sic']) for r in rows if float(r['fraction'])==f and float(r['lambda'])==0]);bj=np.mean([float(r['js_divergence']) for r in rows if float(r['fraction'])==f and float(r['lambda'])==0])
    sr=[];jr=[]
    for l in lams:
        group=[r for r in rows if float(r['fraction'])==f and float(r['lambda'])==l] if l<.1 else [r for r in old if float(r['fraction'])==f and r['model']==f'sideband_{l:g}' and int(r['seed']) in [11,22,33] and r['split']=='validation' and float(r['cut'])==.05]
        sr.append(np.mean([float(r['sic']) for r in group])/bs);jr.append(np.mean([float(r['js_divergence']) for r in group])/bj)
    axes[0].plot(lams,sr,'o-',color=c,label=f'{f:.1%} signal');axes[1].plot(lams,jr,'o-',color=c)
for ax in axes:ax.set_xscale('log');ax.set_xlabel('DisCo strength lambda');ax.axvline(.003,color='gray',ls=':');ax.grid(alpha=.2)
axes[0].axhline(.9,color='black',ls='--');axes[1].axhline(.8,color='black',ls='--');axes[0].set_ylabel('SIC / baseline');axes[1].set_ylabel('JS / baseline');axes[0].legend();fig.tight_layout();save(fig,'fig3_lambda')
fig,axes=plt.subplots(1,2,figsize=(7.2,2.25))
for f,ax in zip([.005,.01],axes):
    for name in labels:
        rates=[]
        for s in [11,22,33]:
            d=np.load(ROOT/f'results_confirmation/f{f:g}_seed{s}_{name}.npz');b=d['truth']==0;sel=d['score']>d['threshold'];m=d['mass'];rates.append(np.histogram(m[b&sel],bins)[0]/np.histogram(m[b],bins)[0])
        rates=np.array(rates);ax.errorbar(centers,rates.mean(0),yerr=rates.std(0),fmt='.-',lw=1,color=colors[name],label=labels[name])
    ax.axvspan(3300,3700,color='gray',alpha=.1);ax.set(title=f'{f:.1%} training signal',xlabel='Dijet mass [GeV]');ax.grid(alpha=.2)
axes[0].set_ylabel('QCD selection efficiency');axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'fig4_confirmation')
rows=list(csv.DictReader((ROOT/'results_confirmation/metrics.csv').open()));fig,axes=plt.subplots(1,2,figsize=(7.2,2.1))
for f,ax in zip([.005,.01],axes):
    for name in labels:
        group=[r for r in rows if float(r['fraction'])==f and r['model']==name];x=[float(r['js']) for r in group];y=[float(r['sic']) for r in group]
        ax.scatter(x,y,color=colors[name],s=16,alpha=.7);ax.scatter(np.mean(x),np.mean(y),color=colors[name],marker='D',s=45,label=labels[name])
    ax.set(title=f'{f:.1%} training signal',xlabel='Background JS divergence');ax.grid(alpha=.2)
axes[0].set_ylabel('SR sensitivity improvement SIC');axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'fig5_tradeoff')
# Exact counts from stored events, not estimates.
d=np.load(ROOT/'results_confirmation/f0.005_seed11_mass_mlp.npz');sr=(d['mass']>=3300)&(d['mass']<3700)
print('confirmation counts',len(d['rows']),int((d['truth']==0).sum()),int((d['truth']==1).sum()),int(((d['truth']==0)&sr).sum()))
