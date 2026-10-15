from pathlib import Path
import sys, json, gzip, hashlib, tarfile, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[2]
DEFAULT_CODE = DRAFT / 'code/deep'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--development-code', type=Path, default=DEFAULT_CODE/'code')
parser.add_argument('--campaign-code', type=Path, default=DEFAULT_CODE/'campaign_C')
parser.add_argument('--development-data', type=Path, default=DRAFT/'data/deep_development')
parser.add_argument('--development-results', type=Path, default=None,
                    help='Optional directory of additional loose development JSON files')
parser.add_argument('--campaign-data', type=Path, default=DRAFT/'data/deep_campaign_C')
parser.add_argument('--output-dir', type=Path, default=DRAFT/'figures/20260930_submission/deep')
parser.add_argument('--review-dir', type=Path, default=DRAFT/'review/submission_cleanup_20260930/figures/deep')
ARGS, _ = parser.parse_known_args()
DEV_CODE, CAM = ARGS.development_code.resolve(), ARGS.campaign_code.resolve()
DEV_DATA, CAM_DATA = ARGS.development_data.resolve(), ARGS.campaign_data.resolve()
OUT, REVIEW = ARGS.output_dir.resolve(), ARGS.review_dir.resolve()
ANALYSIS = CAM/'analysis.json' if (CAM/'analysis.json').exists() else CAM/'results/analysis.json'
sys.path.insert(0, str(DEV_CODE))
sys.path.insert(0, str(CAM))
from flatwin import best_grok_window
from endpoints import run_fields, series, at, knee_step, wilson
from matplotlib.text import Text

BLUE, ORANGE, GREEN, PURPLE, GRAY, RED = '#0072B2', '#D55E00', '#009E73', '#7851A9', '#888888', '#C23B45'
REFIT, TRAIN_ACC, AGOP_C, WINDOW, LAYER = '#CC79A7', '#000000', '#E69F00', '#EBEBEB', '#444444'
FIRST, LAST = '-.', '-'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.5, 'axes.labelsize':8.5,
    'axes.titlesize':9, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#AAAAAA', 'axes.linewidth':0.6, 'xtick.color':'black',
    'ytick.color':'black', 'grid.color':'#DDDDDD', 'grid.linewidth':0.45,
    'text.color':'black', 'axes.labelcolor':'black', 'xtick.labelsize':8, 'ytick.labelsize':8, 'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'})
RECEIPT = {'source_files':[], 'archive_members':[], 'derived':{}, 'outputs':[]}
def sha(data): return hashlib.sha256(data).hexdigest()
def source(path):
    path=Path(path)
    if not any(r['path']==str(path) for r in RECEIPT['source_files']):
        RECEIPT['source_files'].append({'path':str(path),'sha256':sha(path.read_bytes())})
def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_ticks, drawn_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks.update((tick.label1, tick.label2))
            for tick in axis._update_ticks():
                drawn_ticks.update((tick.label1, tick.label2))
    labels = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
              and (t not in all_ticks or t in drawn_ticks)]
    fonts = [t.get_fontsize() for t in labels]
    boxes = [t.get_window_extent(renderer) for t in labels]
    bounds = fig.bbox
    outside = [t.get_text() for t,b in zip(labels,boxes)
               if b.x0 < bounds.x0-.5 or b.y0 < bounds.y0-.5
               or b.x1 > bounds.x1+.5 or b.y1 > bounds.y1+.5]
    assert min(fonts) >= 8, (name, min(fonts))
    assert not outside, (name, 'text outside canvas', outside)
    RECEIPT.setdefault('validation', {})[name] = {
        'width_inches':float(fig.get_figwidth()), 'height_inches':float(fig.get_figheight()),
        'minimum_font_points':float(min(fonts)), 'all_text_inside_canvas':True,
        'label_count':len(labels), 'font_family':'DejaVu Sans'}
    for ext in ('pdf','png'):
        path=OUT/f'{name}.{ext}'
        fig.savefig(path, dpi=240)
        RECEIPT['outputs'].append({'path':str(path), 'sha256':sha(path.read_bytes())})
    plt.close(fig)
def field(obs,key,ax,color,ls='-',lw=1.35):
    x,y=series(obs,key); ax.plot(x/1000,y,color=color,ls=ls,lw=lw)
def setup_axes(axs, a,b,x0,x1):
    for ax in axs:
        ax.axvspan(a/1000,b/1000,color=WINDOW,lw=0,zorder=0)
        ax.set_xlim(x0/1000,x1/1000); ax.grid(True,axis='y'); ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(3))
def assert_visible_ce(ax, x, y, name):
    x=np.asarray(x); y=np.asarray(y); xlo,xhi=ax.get_xlim(); ylo,yhi=ax.get_ylim()
    keep=(x>=xlo)&(x<=xhi)&np.isfinite(x)&np.isfinite(y)
    assert keep.any(), (name, 'no finite CE points inside displayed x limits')
    v=y[keep]
    assert v.min()>=ylo and v.max()<=yhi, (name, float(v.min()), float(v.max()), (ylo,yhi))
    RECEIPT.setdefault('validation',{}).setdefault('visible_ce',[]).append({
        'run':name,'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
        'n_finite_visible':int(keep.sum()),'minimum':float(v.min()),'maximum':float(v.max()),'within_limits':True})
def random_handles():
    return (Line2D([],[],color=GRAY,ls='--'),Line2D([],[],color=GRAY,ls=':'))
def read_development(name):
    manifest={line.split('\t')[0]:line.split('\t')[1].strip() for line in (DEV_CODE/'RUN_MANIFEST.tsv').read_text().splitlines() if line and not line.startswith('#')}
    expected=manifest['runs/'+name+'.json']
    for path in sorted(DEV_DATA.rglob('*.tar.gz')):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                if not member.isfile() or Path(member.name).name not in (name+'.json',name+'.json.gz'): continue
                packed=tf.extractfile(member).read()
                data=gzip.decompress(packed) if member.name.endswith('.gz') else packed
                if sha(data)!=expected: continue
                source(path)
                RECEIPT['archive_members'].append({'archive':str(path),'member':member.name,'uncompressed_sha256':sha(data),'manifest_match':True})
                return json.loads(data)
    roots = [DEV_DATA] + ([ARGS.development_results.resolve()] if ARGS.development_results else [])
    candidates = [p for root in roots for suffix in ('.json.gz', '.json')
                  for p in sorted(root.rglob(name+suffix))]
    for path in candidates:
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        if sha(data)==expected:
            source(path)
            RECEIPT['archive_members'].append({'path':str(path),'uncompressed_sha256':sha(data),'manifest_match':True})
            return json.loads(data)
    raise FileNotFoundError(name)

def development_overview():
    names=['seedD2_r0.5_s2','T6_D3_cascade_r0.9_0.45_s2_T100k','T18_D4_r1.0_1.0_0.4_s1_T120k','T21_D5_r1.0_1.0_1.0_0.35_s0_T150k']
    fig,axs=plt.subplots(3,4,figsize=(6.75,4.5),sharey='row')
    meta=[]; ce_curves=[]
    for j,name in enumerate(names):
        d=read_development(name); obs=d['obs']; L=np.asarray(d['loss']); cm=(L[:-1]+L[1:])/2
        s,te=series(obs,'test_acc'); k=knee_step(d['cfg'],obs)
        a,b,share=best_grok_window(cm,s,te,k)
        t10=int(s[np.flatnonzero(te>=.1)[0]]); t90=int(s[np.flatnonzero(te<.9)[-1]+1])
        mu=cm[a:b+1].mean(); p=d['cfg']['p']; depth=d['cfg']['depth']
        setup_axes(axs[:,j],a,b,min(a,t10)-6000,max(b,t90)+7000)
        for ax in axs[:,j]:
            ax.axvline(t10/1000,color=GRAY,ls=':',lw=.8)
            ax.axvline(t90/1000,color=GRAY,ls=':',lw=.8)
        axs[0,j].plot(np.arange(len(cm))/1000,cm/math.log(p),color=BLUE,lw=1.4)
        ce_curves.append((axs[0,j],np.arange(len(cm))/1000,cm/math.log(p),name))
        axs[0,j].set_title(f'Depth {depth}\n{share:.0%} of rise in W')
        axs[0,j].text(.04,.08,f'Level {mu/math.log(p):.2f}',transform=axs[0,j].transAxes,fontsize=8)
        field(obs,'train_acc',axs[1,j],TRAIN_ACC,'--',lw=1)
        field(obs,'test_acc',axs[1,j],ORANGE)
        for key,ls in [('agopH1_4_ho',FIRST),('agop30_ho',LAST)]:
            field(obs,key,axs[2,j],AGOP_C,ls)
            field(obs,key+'_rand',axs[2,j],GRAY,'--' if ls==FIRST else ':',.9)
        axs[2,j].set_xlabel('Updates (thousands)')
        meta.append({'run':name,'cfg':d['cfg'],'window':[a,b],'share':float(share),'CE_mean_over_logp':float(mu/math.log(p)),
                     'CE_range_over_midrange':float((cm[a:b+1].max()-cm[a:b+1].min())/((cm[a:b+1].max()+cm[a:b+1].min())/2)),
                     't10':t10,'t90':t90,'selection':'same exemplar and outcome-aware best window as original overview'})
    axs[0,0].set_ylim(.30,.64); axs[1,0].set_ylim(-.02,1.04); axs[2,0].set_ylim(0,.65)
    for args in ce_curves: assert_visible_ce(*args)
    axs[0,0].set_ylabel('Training CE / log p\n(cycle mean)')
    axs[1,0].set_ylabel('Accuracy')
    axs[2,0].set_ylabel('Hidden AGOP alignment\n(legacy observer)')
    handles=[Line2D([],[],color=TRAIN_ACC,ls='--'),Line2D([],[],color=ORANGE),Line2D([],[],color=AGOP_C,ls=FIRST),Line2D([],[],color=AGOP_C,ls=LAST),random_handles()]
    fig.legend(handles=handles,labels=['Train acc.','Held-out','First hidden, top 4','Last hidden, top 30','Random-frame means'],handler_map={tuple:HandlerTuple(ndivide=None)},loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,.005),fontsize=8)
    fig.subplots_adjust(left=.105,right=.990,top=.91,bottom=.22,hspace=.31,wspace=.22)
    RECEIPT['derived']['development_overview']=meta
    save(fig,'deep_development_overview')

def campaign_summary(analysis):
    rows=list(analysis['rows'].values()); groups=['D3','D4','D5','D4p23','D4w512','D3ngd']
    labels=['D3','D4','D5','D4\np=23','D4\nw=512','D3*']
    colors=[BLUE,GREEN,PURPLE,ORANGE,GRAY,GRAY]
    fig,axs=plt.subplots(2,2,figsize=(6.75,3.8))
    specs=[('share','Share of 0.1 to 0.9 rise',.75,(-.02,1.14)),('dAH_last',r'Last-hidden $\Delta A_H$',.02,(0,.105)),('final_test','Final held-out accuracy',.99,(.90,1.012)),('agop_first_gain_over_random','First-hidden AGOP\ngain over random',0,(-.08,.24))]
    stats={}; count_labels=[]
    for gi,g in enumerate(groups):
        rs=sorted([r for r in rows if r['cohort']==g],key=lambda r:(r['seed'],r['stage']))
        n=len(rs); k=sum(r['main']['primary'] for r in rs)
        offsets=np.linspace(-.20,.20,n); offsets=offsets[np.argsort(np.argsort([r['id'] for r in rs]))]
        stats[g]={'n':n,'k':k,'wilson':list(wilson(k,n)),'primary_failures':{key:sum(not r['main']['primary_checks'][key] for r in rs) for key in ('share','F02','final','fit')}}
        for ai,(key,ylabel,threshold,ylim) in enumerate(specs):
            ax=axs.flat[ai]; vals=np.array([r['main'][key] for r in rs])
            ax.scatter(gi+offsets,vals,s=11,color=(LAYER,GREEN,ORANGE,AGOP_C)[ai],alpha=.65,linewidths=0,zorder=3)
            med=np.median(vals); ax.plot([gi-.25,gi+.25],[med,med],color='black',lw=1.5,zorder=4)
            if ai==0: count_labels.append(ax.text(gi,1.055,f'{k}/{n}',ha='center',va='bottom',fontsize=8))
            stats[g][key]={'min':float(vals.min()),'median':float(med),'max':float(vals.max())}
    for ai,(key,ylabel,threshold,ylim) in enumerate(specs):
        ax=axs.flat[ai]
        ax.set_ylim(*ylim); ax.set_xlim(-.5,5.5); ax.set_xticks(range(6),labels,fontsize=8)
        ax.set_ylabel(ylabel); ax.axhline(threshold,color='#555555',ls='--',lw=.9)
        ax.axvline(2.5,color='#AAAAAA',lw=.7); ax.grid(True,axis='y')
        ax.set_title(chr(97+ai)+') '+('Recovery share; joint passes above' if ai==0 else 'Last-hidden Fourier gain' if ai==1 else 'Final held-out accuracy' if ai==2 else 'First-hidden AGOP gain, q=4'),loc='left',fontsize=8)
    axs[0,0].set_yticks([0,.25,.5,.75,1])
    fig.subplots_adjust(left=.105,right=.985,top=.94,bottom=.16,hspace=.72,wspace=.38)
    fig.text(.54,.01,'D3*: normalized GD; other cohorts: Polar. Black bars: medians.',ha='center',va='bottom',fontsize=8)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    boxes=[text.get_window_extent(renderer) for text in count_labels]
    gaps=[b.x0-a.x1 for a,b in zip(boxes,boxes[1:])]
    assert min(gaps)>1, ('overlapping primary count labels',gaps)
    left_title=next(t for t in axs[0,0].texts if t.get_text()==axs[0,0].get_title(loc='left')) if axs[0,0].texts and any(t.get_text()==axs[0,0].get_title(loc='left') for t in axs[0,0].texts) else axs[0,0]._left_title
    assert max(b.y1 for b in boxes)<left_title.get_window_extent(renderer).y0, 'count labels overlap title'
    RECEIPT.setdefault('validation',{})['summary_count_labels']={'minimum_gap_pixels':float(min(gaps)),'title_clear':True}
    RECEIPT['derived']['campaign_summary']=stats
    save(fig,'deep_campaign_summary')

def campaign_trajectories(analysis):
    fig,axs=plt.subplots(5,3,figsize=(6.75,6.8),sharey='row')
    meta=[]; ce_curves=[]
    for j,depth in enumerate((3,4,5)):
        chain=f'C2_D{depth}_s001'; archive=CAM_DATA/f'campaignC_C2_D{depth}.tar'; source(archive)
        with tarfile.open(archive) as tf:
            packed=tf.extractfile(chain+'/main.json.gz').read(); data=gzip.decompress(packed)
            prov=json.loads(tf.extractfile(chain+'/chain.json').read())
        d=json.loads(data); obs=d['obs']; f=analysis['rows'][chain]['main']; verified=run_fields(d)
        for key in ('W','share','dAH_last','final_test','primary','agop_first_gain_over_random'):
            assert verified[key]==f[key], (chain,key,verified[key],f[key])
        RECEIPT['archive_members'].append({'archive':str(archive),'member':chain+'/main.json.gz','compressed_sha256':sha(packed),'uncompressed_sha256':sha(data),'recomputed_analysis_match':True})
        a,b=f['W']; t10,t90=f['t10'],f['t90']; L=np.asarray(d['loss']); cm=(L[:-1]+L[1:])/2; lp=math.log(d['cfg']['p'])
        setup_axes(axs[:,j],a,b,min(a,t10)-6000,max(b,t90)+6000)
        axs[0,j].plot(np.arange(len(cm))/1000,cm/lp,color=BLUE,lw=1.4)
        ce_curves.append((axs[0,j],np.arange(len(cm))/1000,cm/lp,chain))
        axs[0,j].hlines([f['CE_peak']/1.03/lp,f['CE_peak']/lp],a/1000,b/1000,colors=GRAY,linestyles='--',lw=.8)
        axs[0,j].set_title(f'Depth {depth}, seed 1\n{f["share"]:.0%} of rise in W')
        for key,color,ls in [('train_acc',TRAIN_ACC,'--'),('test_acc',ORANGE,'-'),('v1_ridge_acc_te',REFIT,':')]:field(obs,key,axs[1,j],color,ls=ls,lw=1 if key=='train_acc' else 1.35)
        for name,ls in [('first',FIRST),('last',LAST)]:
            field(obs,f'v1_{name}_agop_purity',axs[2,j],AGOP_C,ls)
            field(obs,f'v1_{name}_agop_rand_med',axs[2,j],GRAY,'--' if name=='first' else ':',.85)
            x,rmin=series(obs,f'v1_{name}_agop_rand_min'); _,rmax=series(obs,f'v1_{name}_agop_rand_max')
            axs[2,j].fill_between(x/1000,rmin,rmax,color=GRAY,alpha=.10,lw=0)
        field(obs,'v1_first_AH',axs[3,j],GREEN,FIRST)
        field(obs,'v1_last_AH',axs[3,j],GREEN,LAST)
        field(obs,'v1_ridge_mse_te',axs[4,j],REFIT,':')
        axs[4,j].set_xlabel('Updates (thousands)')
        meta.append({'chain':chain,'window':[a,b],'fields_recomputed':True,'primary':f['primary'],'share':f['share'],'cfg':d['cfg'],
            'v1_ridge_acc_te_W':[at(*series(obs,'v1_ridge_acc_te'),t) for t in (a,b)],
            'v1_ridge_mse_te_W':[at(*series(obs,'v1_ridge_mse_te'),t) for t in (a,b)],
            'v1_first_AH_W':[at(*series(obs,'v1_first_AH'),t) for t in (a,b)],
            'v1_last_AH_W':[at(*series(obs,'v1_last_AH'),t) for t in (a,b)],
            'selection':'lowest-index C2 seed per depth, not selected by outcome'})
    axs[0,0].set_ylim(.30,.61); axs[1,0].set_ylim(-.02,1.04); axs[2,0].set_ylim(.10,.72); axs[3,0].set_ylim(0,.83); axs[4,0].set_ylim(0,1.35)
    for args in ce_curves: assert_visible_ce(*args)
    axs[0,0].set_ylabel('Training CE / log p\n(cycle mean)'); axs[1,0].set_ylabel('Accuracy')
    axs[2,0].set_ylabel('Hidden AGOP purity\n(class-centered, q=4)'); axs[3,0].set_ylabel('Hidden Fourier\n$A_H$'); axs[4,0].set_ylabel('Refit held-out MSE')
    handles=[Line2D([],[],color=TRAIN_ACC,ls='--'),Line2D([],[],color=ORANGE),Line2D([],[],color=REFIT,ls=':'),
             Line2D([],[],color=LAYER,ls=FIRST),Line2D([],[],color=LAYER,ls=LAST),random_handles()]
    fig.legend(handles=handles,labels=['Train acc.','Held-out','Refit','First hidden','Last hidden','Random medians'],handler_map={tuple:HandlerTuple(ndivide=None)},loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,.005),fontsize=8)
    fig.subplots_adjust(left=.115,right=.990,top=.94,bottom=.12,hspace=.33,wspace=.22)
    RECEIPT['derived']['campaign_trajectories']=meta
    save(fig,'deep_campaign_trajectories')

def campaign_trajectories_main(analysis):
    fig, axs = plt.subplots(3, 3, figsize=(6.75, 3.0), sharey='row', sharex='col')
    meta = []
    ce_curves = []
    for j, depth in enumerate((3, 4, 5)):
        chain = f'C2_D{depth}_s001'
        archive = CAM_DATA/f'campaignC_C2_D{depth}.tar'
        source(archive)
        with tarfile.open(archive) as tf:
            data = gzip.decompress(tf.extractfile(chain+'/main.json.gz').read())
        d = json.loads(data)
        obs = d['obs']
        f = analysis['rows'][chain]['main']
        a, b = f['W']
        t10, t90 = f['t10'], f['t90']
        L = np.asarray(d['loss'])
        cm = (L[:-1]+L[1:])/2
        lp = math.log(d['cfg']['p'])
        setup_axes(axs[:, j], a, b, min(a,t10)-6000, max(b,t90)+6000)
        axs[0,j].plot(np.arange(len(cm))/1000,cm/lp,color=BLUE,lw=1.2)
        ce_curves.append((axs[0,j],np.arange(len(cm))/1000,cm/lp,chain+'_main'))
        axs[0,j].hlines([f['CE_peak']/1.03/lp,f['CE_peak']/lp],a/1000,b/1000,colors=GRAY,linestyles='--',lw=.8)
        axs[0,j].set_title(f'Depth {depth}, seed 1', fontsize=9)
        field(obs,'train_acc',axs[1,j],TRAIN_ACC,ls='--',lw=1.0)
        field(obs,'test_acc',axs[1,j],ORANGE,lw=1.2)
        field(obs,'v1_ridge_acc_te',axs[1,j],REFIT,ls=':',lw=1.2)
        field(obs,'v1_first_AH',axs[2,j],GREEN,ls=FIRST,lw=1.2)
        field(obs,'v1_last_AH',axs[2,j],GREEN,ls=LAST,lw=1.2)
        axs[2,j].set_xlabel('Updates (thousands)', labelpad=2)
        meta.append({'chain':chain, 'window':[a,b], 'payload_sha256':sha(data),
                     'selection':'same lowest-index C2 seeds as full trajectory sheet'})
    axs[0,0].set_ylim(.30,.61)
    axs[1,0].set_ylim(-.02,1.04)
    axs[2,0].set_ylim(0,.83)
    for args in ce_curves:
        assert_visible_ce(*args)
    axs[0,0].set_ylabel('Training CE / log p\n(cycle mean)',labelpad=3)
    axs[1,0].set_ylabel('Accuracy',labelpad=3)
    axs[2,0].set_ylabel('Hidden Fourier\n$A_H$',labelpad=3)
    handles = [Line2D([],[],color=TRAIN_ACC,ls='--'),Line2D([],[],color=ORANGE),Line2D([],[],color=REFIT,ls=':'),
               Line2D([],[],color=LAYER,ls=FIRST),Line2D([],[],color=LAYER,ls=LAST)]
    fig.legend(handles, ['Train acc.','Held-out','Refit','First hidden','Last hidden'],
               loc='lower center',bbox_to_anchor=(.55,.006),ncol=5,frameon=False,
               fontsize=8,columnspacing=1.2,handlelength=2.1)
    fig.subplots_adjust(left=.105,right=.990,top=.905,bottom=.23,hspace=.23,wspace=.23)
    RECEIPT['derived']['campaign_trajectories_main'] = meta
    save(fig,'deep_campaign_trajectories_main')

if __name__=='__main__':
    for p in [DEV_CODE/'RUN_MANIFEST.tsv',DEV_CODE/'flatwin.py',DEV_CODE/'make_figs_deep.py',ANALYSIS,CAM/'endpoints.py',CAM/'PREREGISTRATION.md',CAM/'manifest.json',CAM/'natplat_campaign.py']:
        source(p)
    analysis=json.loads(ANALYSIS.read_text())
    development_overview(); campaign_summary(analysis); campaign_trajectories(analysis); campaign_trajectories_main(analysis)
    source(__file__)
    (REVIEW/'provenance.json').write_text(json.dumps(RECEIPT,indent=2)+'\n')
    print('Created 4 PDF/PNG figures; verified 4 development input hashes and 3 campaign endpoint recomputations.')
