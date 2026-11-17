from pathlib import Path
import json,csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Spacer,Image,Table,TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'MLHEP_Project_Report.pdf'
for name,file in [('TimesCustom','times.ttf'),('TimesBold','timesbd.ttf'),('TimesItalic','timesi.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('TimesCustom',normal='TimesCustom',bold='TimesBold',italic='TimesItalic',boldItalic='TimesBold')
W,H=A4;M=40;BW=W-2*M
styles={
'body':ParagraphStyle('body',fontName='TimesCustom',fontSize=10.5,leading=12.5,spaceAfter=6),
'head':ParagraphStyle('head',fontName='TimesBold',fontSize=12,leading=14,spaceBefore=7,spaceAfter=6),
'sub':ParagraphStyle('sub',fontName='TimesBold',fontSize=10.5,leading=12.5,spaceBefore=4,spaceAfter=4),
'caption':ParagraphStyle('caption',fontName='TimesCustom',fontSize=10,leading=11.5,spaceAfter=7),
'cell':ParagraphStyle('cell',fontName='TimesCustom',fontSize=10,leading=11.3),
'eq':ParagraphStyle('eq',fontName='TimesCustom',fontSize=10.5,leading=13,alignment=1,spaceAfter=6),
 'title':ParagraphStyle('title',fontName='TimesBold',fontSize=17,leading=19,spaceAfter=8,alignment=1)}
pages=[];page=[];textlog=[]
def p(t,style='body'):
    page.append(Paragraph(t,styles[style]));textlog.append(t)
def heading(t):p(t,'head')
def sub(t):p(t,'sub')
def fig(name,height,caption):
    from PIL import Image as PIL
    path=ROOT/'figures'/f'{name}.png';im=PIL.open(path);width=height*im.width/im.height
    if width>BW:height*=BW/width;width=BW
    page.append(Image(str(path),width=width,height=height));p(caption,'caption')
def table(rows,widths,caption):
    p(caption,'caption');t=Table([[Paragraph(str(v),styles['cell']) for v in row] for row in rows],colWidths=[BW*w for w in widths])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef1')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#b7c2c8')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));page.append(t);page.append(Spacer(1,5))
def nextpage():
    global page
    pages.append(page);page=[]
p('Controlling Mass Sculpting in CWoLa Hunting<br/>with Distance Correlation','title')
p('Nishanth S (EP23BTECH11024) · Arav P (EP23BTECH11003)<br/>Dhruv Goyal (EP23BTECH11008)','eq')
sub('Abstract')
p('Jet-substructure classifiers can enhance a resonant signal while distorting the background mass spectrum used to search for it. We investigate this trade-off in CWoLa hunting using Energy Flow Polynomials (EFPs) and individual jet masses from the LHC Olympics R&amp;D simulation. A PyTorch classifier is regularized with a distance-correlation (DisCo) penalty applied only to mass-sideband events. An initial 105-fit comparison exposed excessive sensitivity loss at strong penalties; a further 24-fit validation study selected λ = 0.003. Without retraining or adjusting thresholds, we evaluated 18 saved models on 100,000 previously unused raw events. Relative to the mass-feature baseline, mean background Jensen–Shannon distortion fell by 45.3% and 58.6% at training signal fractions of 0.5% and 1.0%, while mean sensitivity improvement was retained at 94.3% and 106.5%. Both predefined targets cleared conditional event-bootstrap intervals. These results support a useful, but neither universal nor complete, decorrelation trade-off in this benchmark.')
heading('1. Problem Statement')
p('A narrow resonance decaying to jets can appear as an excess in their combined invariant mass, m<sub>JJ</sub>. CWoLa hunting learns to distinguish a central signal region (SR) from neighboring sidebands (SB), without signal/background training targets [1]. If background features vary with mass, the selection efficiency ε<sub>B</sub>(m) also varies, changing the selected spectrum through dN<sub>sel</sub>/dm = ε<sub>B</sub>(m) dN<sub>B</sub>/dm. This mass sculpting can complicate background estimation; it need not produce a narrow fake bump.')
p('The Kitchen Sink study combines many jet observables and compares weakly supervised detectors [2]. EFPs provide a systematic jet-substructure basis [3], while DisCo supplies a differentiable measure of nonlinear dependence [4]. We combine these established ideas in a reduced-feature study: <b>can sideband-only score decorrelation reduce background distortion without losing useful signal sensitivity?</b> Our contribution is the controlled comparison and independently confirmed trade-off, not a new EFP basis, a new loss estimator, or reproduction of the full Kitchen Sink feature set.')
fig('fig1_workflow',66,'<b>Figure 1.</b> Analysis sequence. Truth labels construct simulated mixtures and evaluate performance; the selected classifier and sideband penalty do not use truth as training targets.')
heading('2. Methodology')
sub('2.1 Dataset and physics preprocessing')
p('We use events_anomalydetection.h5 from the LHCO R&amp;D release [5,6]: 1 million QCD events and 100,000 simulated W′ → XY → 4q events, randomly shuffled. The resonance mass is 3.5 TeV, with daughter masses 500 and 100 GeV. Each row contains up to 700 massless reconstructed particles as (p<sub>T</sub>, η, φ), zero padding, and a truth bit. The sample has no pileup or multiparton interactions [6].')
nextpage()
sub('2.1 Dataset and physics preprocessing continued')
p('We read 128-row HDF5 chunks with h5py and the Blosc filter, remove zero-p<sub>T</sub> padding, and construct (E,p<sub>x</sub>,p<sub>y</sub>,p<sub>z</sub>) = p<sub>T</sub>(cosh η, cos φ, sin φ, sinh η). FastJet clusters anti-k<sub>T</sub> jets with R = 1 [7,8]. We retain the two highest-p<sub>T</sub> jets when the leading jet exceeds 1,200 GeV and compute m<sub>JJ</sub><sup>2</sup> = (p<sub>J1</sub> + p<sub>J2</sub>)<sup>2</sup>. Jets are then ordered by increasing mass. Sequential EFP evaluation and compact caches avoid the original full-file RAM allocation and Windows multiprocessing failures.')
p('EnergyFlow computes normalized hadronic EFPs with β = 1 and degree ≤ 4 separately on each jet. Removing constant columns leaves 70 EFP inputs. The mass-feature variant adds the lighter jet mass and the mass difference, giving 72 inputs; m<sub>JJ</sub> itself is never an input. We use log(max(EFP,10<super>−12</super>)) and log(1 + jet mass feature), followed by training-only standardization. This is an EFP-plus-mass study, without N-subjettiness or the full paper’s EFP basis.')
table([['Stage','Raw rows or sample','Use'],['Development','0–199,999; 69,698 selected','Training, tuning and exploratory evaluation'],['Earlier held-out evaluation','9,926 background + 2,590 SR signal','Exploratory after first results were inspected'],['Fresh confirmation','200,000–299,999; 33,313 evaluated','26,486 background + 6,827 SR signal; no fitting']], [.22,.40,.38],'<b>Table 1.</b> Data accounting. Fresh confirmation contains 12,999 SR background events. Signal outside the SR is excluded from evaluation.')
sub('2.2 CWoLa labels and separation of information')
p('The SR is [3,300,3,700) GeV with label y = 1; the SB are [3,100,3,300) and [3,700,3,900) GeV with y = 0. These are region labels, not signal truth. A fixed stratified quarter of eligible development events with source row ≥ 50,000 forms the earlier test pool. The rest is split 75:25 into training and validation per seed. Signal is injected only in the SR with f = S/(B<sub>SR</sub> + S), at f = 0, 0.005 or 0.01. Counts are rounded separately per split and signal subsets are nested. All available held-out SR signal estimates sensitivity; evaluation abundance is not the training injection fraction.')
sub('2.3 Architecture and regularized objective')
p('The MLP has 128, 64 and 32 ReLU hidden units and one output logit z; s = sigmoid(z). Balanced binary cross-entropy assigns class weight N/(2N<sub>y</sub>). Adam uses learning rate 0.001, batch size 512, at most 15 epochs and patience 5. Each model selects its checkpoint using validation BCE plus its own weighted DisCo penalty. The objective is')
p('L = L<sub>balanced BCE</sub> + λ R̂²(s,m<sub>JJ</sub>), &nbsp; R̂² = ⟨AB⟩ / √(⟨A²⟩⟨B²⟩).','eq')
p('Here a<sub>ij</sub> = |s<sub>i</sub> − s<sub>j</sub>| and A<sub>ij</sub> = a<sub>ij</sub> − ā<sub>i·</sub> − ā<sub>·j</sub> + ā<sub>··</sub>; B is the analogous centered mass-distance matrix. This biased empirical squared distance correlation costs O(n²). A denominator floor prevents numerical division by zero. We use up to 128 eligible events per shuffled training batch and a fixed validation subset of up to 512. The selected method penalizes both SB jointly with uniform weights. An oracle comparison instead masks true background across the full window. Constant scores have zero penalty, so sensitivity must be checked alongside decorrelation.')
nextpage()
heading('3. Results')
sub('3.1 Controlled comparisons and evaluation criteria')
table([['Configuration','Inputs and penalty','Purpose'],['EFP-only MLP','70 EFPs; λ = 0','Simpler feature ablation'],['Mass-feature MLP','70 EFPs + 2 masses; λ = 0','Matched neural baseline'],['Boosted trees','Same 72 inputs; 150 iterations, 15 leaves, L2 = 1','Non-neural reference'],['Oracle / sideband DisCo','Same MLP; truth-background / SB penalty','Oracle benchmark / practical variant']], [.25,.44,.31],'<b>Table 2.</b> Experimental design. Trees use fixed settings; no test-based hyperparameter selection.')
p('The initial grid comprised seven configurations × three injections × five seeds (105 fits), including λ = 0.1 and 1 for each penalty mask. At 1% injection, the ordinary mass-feature MLP achieved mean SR SIC 2.94, but sideband DisCo at λ = 0.1 fell to 1.17. This failure motivated weaker regularization; it was not hidden by reporting only the eventual setting. The EFP-only initial null baseline showed little sculpting, whereas tree and signal-trained mass-feature models showed larger distortions (Figure 2).')
fig('fig2_baseline',142,'<b>Figure 2.</b> Exploratory baseline evaluation, predetermined seed 11 and 1% training injection. Only QCD is plotted. The cut is the validation-SB 95th score percentile. These diagnostic lines show a slope change, not a calibrated resonance excess; bin fluctuations are not individually significant claims.')
sub('3.2 Validation selection of the DisCo strength')
p('We measure SR sensitivity with SIC = ε<sub>S</sub>/√ε<sub>B</sub>, a proxy rather than discovery significance. Distortion is base-2 Jensen–Shannon divergence between normalized selected/all-QCD mass histograms in 16 bins from 3,100 to 3,900 GeV. For P and Q, JS = ½ KL(P∥M) + ½ KL(Q∥M), with M = (P + Q)/2. Cuts use validation-SB quantiles; achieved SR background efficiency is measured, not forced to 5%. Exploratory checks also used ROC/AUC, score–mass distance correlation, and 199 random-selection permutations.')
fig('fig3_lambda',145,'<b>Figure 3.</b> Validation strength scan: ratios of three-seed means to the matched baseline at each injection. Dashed lines mark SIC ≥ 0.90 and JS ≤ 0.80; dotted line marks λ = 0.003. Weak strengths are the 24-fit refinement; 0.1 and 1 use the same three seeds from the earlier study. Lines connect tested settings only.')
p('Four weaker strengths (0.001, 0.003, 0.01, 0.03), two injections and three seeds required 24 additional fits. Before that sweep, we fixed a rule: retain ≥ 90% of mean baseline SIC and reduce mean JS by ≥ 20% at <b>both</b> injections. Only λ = 0.003 qualified, retaining 98.9% and 106.1% of validation SIC while reducing JS by 39.9% and 58.0%. The rule is an engineering target, not a statistical discovery test. All confirmation models, preprocessing and cut thresholds were then frozen.')
nextpage()
sub('3.3 Confirmation on previously unused events')
p('Confirmation used 100,000 new raw events and 18 frozen models: three seeds × two training fractions × three methods. No retraining, model selection or threshold adjustment used these events. Table 3 reports averages across the three models; ratios below are ratios of means. Figures 4–5 show the physical effect and the seed-to-seed variation.')
table([['Training f','Method','Mean SIC','Mean JS','εS','εB in SR'],['0.5%','Baseline','2.554','0.03072','0.627','0.0603'],['0.5%','DisCo 0.003','2.409','0.01681','0.603','0.0626'],['0.5%','EFP only','0.915','0.00212','0.210','0.0515'],['1.0%','Baseline','2.877','0.03032','0.706','0.0602'],['1.0%','DisCo 0.003','3.066','0.01256','0.726','0.0564'],['1.0%','EFP only','2.870','0.00202','0.664','0.0538']], [.14,.24,.16,.17,.14,.15],'<b>Table 3.</b> Audited fresh-data performance. All score cuts were fixed from validation SB. “Baseline” denotes the mass-feature MLP.')
fig('fig4_confirmation',148,'<b>Figure 4.</b> Fresh-QCD selection efficiency versus mass. Points average three frozen seeds; bars are seed standard deviations, not independent-event confidence intervals. Shading marks the SR. EFP-only remains an important low-distortion reference.')
fig('fig5_tradeoff',142,'<b>Figure 5.</b> Fresh sensitivity–distortion trade-off. Small points are individual frozen seeds; diamonds are means. Higher SIC and lower JS are desirable. The broad EFP-only spread at 0.5% reveals unstable sensitivity despite low distortion. Seed averages are not an ensemble score.')
p('At 0.5% injection, DisCo retained <b>94.31%</b> of baseline SIC (95% interval 92.96–95.92%) and reduced JS by <b>45.29%</b> (34.56–50.71%). At 1.0%, retention was <b>106.54%</b> (104.23–108.93%) and JS reduction <b>58.59%</b> (47.09–61.56%). Both point estimates and conditional intervals clear the predefined targets. The 300-replicate paired bootstrap resamples the same events for both methods and all seeds within fixed background-SB, background-SR and signal-SR strata.')
nextpage()
sub('3.4 Interpretation and limitations')
p('The confirmation supports a useful aggregate trade-off, not uniform improvement. At 1% injection, seed 11 has higher JS with DisCo (0.02924 versus 0.02602); the reduction is driven by the three-seed average. The EFP-only reference is competitive at 1%, with much smaller JS, but at 0.5% two seeds retain almost no signal while a third is effective. This makes feature choice and optimization stability part of the conclusion: adding a custom loss is not automatically better than simplifying the inputs.')
p('The bootstrap intervals condition on the frozen models and observed class/region counts. They exclude training, simulator and detector systematic uncertainty, are not simultaneous confidence bands, and do not turn three shared-data seeds into independent samples. Raw JS has finite-count bias. A lower JS neither proves statistical independence nor demonstrates removal of a fake discovery. We did not fit a background model or compute a bump p-value.')
p('The analysis uses one simulated signal topology, a reduced degree-4 EFP basis, and SR-only injection. Excluding signal tails from SB idealizes the weak-supervision problem. Sideband decorrelation need not extrapolate into the SR in other datasets. The earlier test set was treated as exploratory after inspection; the fresh confirmation block was kept separate during refinement. Our claim is therefore specific to this benchmark and protocol.')
sub('3.5 Milestones and reproducible artifacts')
p('The work produced a memory-bounded physics pipeline; tested DisCo implementation; 105 initial fits and 24 refinement fits; 18 frozen confirmation evaluations; model/preprocessing archives; row-level predictions; configuration and hash manifests; and auditable figures and tables. No new simulated dataset was generated. Development progressed from repairing loading and clustering assumptions to comparing feature sets, diagnosing overly strong penalties, selecting a weaker loss and confirming it without further fitting.')
p('A compact reproducible_demo bundle runs mathematical/split tests, reloads all 18 models against stored validation predictions, and recomputes confirmation metrics and ratios from prediction archives. Frozen model/code hashes, thresholds, unique fresh row IDs and separation from training/validation rows passed the audit. The command is <font name="TimesBold">python reproducible_demo/run_gauntlet.py</font>; adding <font name="TimesBold">--replay-confirmation</font> reruns frozen inference and bootstrap from caches. The quick audit does not recalculate intervals. Raw HDF5 remains external, package versions are pinned, and clean-environment retraining on another machine has not been verified.')
heading('4. LLM / Agentic AI usage for help with the project')
p('Gemini and OpenAI Codex assisted with topic exploration and refinement from a general mass-sculpting demonstration to a comparison of ordinary and DisCo-regularized CWoLa classifiers. Codex also assisted substantially with preprocessing and jet reconstruction, code generation and debugging, loss and experiment implementation, automated runs, verification and interpretation of saved results, reproducibility checks, figures, and drafting this report. Some longer experiments were executed locally by the team using the prepared scripts. The disclosure includes these implementation and writing contributions; the reported numerical claims are tied to saved outputs and explicit audits.')
heading('5. Workload Distribution')
page.append(Spacer(1,28))
heading('6. Summary &amp; Outlook')
p('A validation-selected sideband DisCo penalty at λ = 0.003 reduced mean background distortion by approximately 45–59% while retaining approximately 94–107% of baseline mean SIC on independent simulated events. Stronger penalties frequently destroyed sensitivity; the successful result required explicit trade-off selection and frozen confirmation. The EFP-only comparison prevents an overbroad claim of superiority. Next steps are additional signal topologies and seeds, realistic signal leakage into SB, detector/generator variations, and a calibrated background-fit study. These would test whether the observed gain survives beyond this controlled benchmark.')
nextpage()
heading('References')
refs=[
('[1]','J. H. Collins, K. Howe and B. Nachman. Anomaly Detection for Resonant New Physics with Machine Learning. Physical Review Letters 121, 241803 (2018).','https://arxiv.org/abs/1805.02664'),
('[2]','R. Das et al. Kitchen Sink Anomaly Detection. arXiv:2604.20965 (2026).','https://arxiv.org/abs/2604.20965'),
('[3]','P. T. Komiske, E. M. Metodiev and J. Thaler. Energy flow polynomials: A complete linear basis for jet substructure. JHEP 04 (2018) 013.','https://arxiv.org/abs/1712.07124'),
('[4]','G. Kasieczka and D. Shih. DisCo Fever: Robust Networks Through Distance Correlation. Physical Review Letters 125, 122001 (2020).','https://arxiv.org/abs/2001.05310'),
('[5]','G. Kasieczka et al. The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics. arXiv:2101.08320 (2021).','https://arxiv.org/abs/2101.08320'),
('[6]','G. Kasieczka, B. Nachman and D. Shih. R&amp;D Dataset for LHC Olympics 2020 Anomaly Detection Challenge. Zenodo record 4536377. File used: events_anomalydetection.h5.','https://zenodo.org/records/4536377'),
('[7]','M. Cacciari, G. P. Salam and G. Soyez. The anti-k<sub>T</sub> jet clustering algorithm. JHEP 04 (2008) 063.','https://arxiv.org/abs/0802.1189'),
('[8]','M. Cacciari, G. P. Salam and G. Soyez. FastJet user manual. European Physical Journal C 72, 1896 (2012).','https://arxiv.org/abs/1111.6097')]
for number,text,url in refs:p(f'{number} {text}<br/><link href="{url}" color="#174e6d">{url}</link>');page.append(Spacer(1,5))
nextpage()
c=canvas.Canvas(str(OUT),pagesize=A4);c.setTitle('Controlling Mass Sculpting in CWoLa Hunting with Distance Correlation');c.setAuthor('Nishanth S; Arav P; Dhruv Goyal')
for i,items in enumerate(pages,1):
    y=H-34
    if i>1:
        c.setFont('TimesCustom',10);c.drawString(M,H-20,'ML for HEP | CWoLa and distance correlation')
    for item in items:
        w,h=item.wrap(BW,H)
        before=getattr(item,'getSpaceBefore',lambda:0)();after=getattr(item,'getSpaceAfter',lambda:0)()
        y-=before
        if y-h<38:raise RuntimeError(f'Page {i} overflow by {38-(y-h):.1f} pt before {str(item)[:80]}')
        item.drawOn(c,M+(BW-w)/2,y-h);y-=h+after
    print('page',i,'bottom',round(y,1))
    c.setFont('TimesCustom',10);c.drawCentredString(W/2,20,str(i));c.showPage()
c.save();(ROOT/'report_text.html').write_text('<html><body>'+'\n'.join('<p>'+t+'</p>' for t in textlog)+'</body></html>',encoding='utf-8');print(OUT)
