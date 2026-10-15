import numpy as np, json, glob, os, sys, collections, statistics as st
def cm(x):
    y=x.copy(); y[:-1]=0.5*(x[:-1]+x[1:]); return y
def analyze(npz):
    z=np.load(npz); j=json.load(open(npz[:-4]+'.json')); m=j['meta']; cfg=m['config']
    ce=z['train_ce']; acc=z['test_accuracy']; tr=z['train_accuracy']; os_=z['obs_step'].astype(int); AH=z['obs_hidden_AH_add']
    c=cm(ce); T=len(ce)-1
    def first(th):
        i=np.flatnonzero(acc>=th); return int(i[0]) if len(i) else None
    t10,t50,t90=first(.1),first(.5),first(.9)
    r=dict(cid=m['config_id'], p=cfg.get('p'), tf=cfg.get('train_fraction'), lw=cfg.get('decay_W'), ev=cfg.get('eta_V'), lv=cfg.get('decay_V'),
           T=T, t10=t10, t90=t90, final=float(acc[-1]), logp=np.log(cfg.get('p')))
    if t10 is not None and t90 is not None and t90>t10:
        seg=c[t10:t90+1]
        r['d']=100*(seg[-1]-seg[0])/seg[0]
        rm=np.maximum.accumulate(seg); r['dd']=100*float(np.max((rm-seg)/rm))
        r['up']=100*float(np.max(seg)/seg[0]-1)
        r['v']=100*float(seg.max()/seg.min()-1)
        r['fit']=float(tr[t10:t90+1].min()); r['ce10']=float(c[t10])
        r['dAH']=float(np.interp(t90,os_,AH)-np.interp(t10,os_,AH))
    return r
rows=[]
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d+'/*.npz')):
        try: rows.append(analyze(f))
        except Exception as e: print('fail',f,e)
by=collections.defaultdict(list)
for r in rows: by[r['cid']].append(r)
print(f"{'config':38s} n  rec  med_d(t10->t90)% frac_d>=0  med_drawdown% med_v% fit  CE10/logp  dAH")
out=[]
for cid,R in by.items():
    rr=[r for r in R if 'd' in r]
    if not rr: out.append((cid,len(R),0,None,None,None,None,None,None,None)); continue
    out.append((cid,len(R),len(rr),st.median([r['d'] for r in rr]),sum(r['d']>=0 for r in rr)/len(rr),st.median([r['dd'] for r in rr]),
                st.median([r['v'] for r in rr]),min(r['fit'] for r in rr),st.median([r['ce10']/r['logp'] for r in rr]),st.median([r['dAH'] for r in rr])))
out.sort(key=lambda x:(-(x[3] if x[3] is not None else -1e9)))
for o in out:
    if o[3] is None: print(f"{o[0]:38s} {o[1]:2d} {o[2]:3d}  (no 10->90 rise)"); continue
    print(f"{o[0]:38s} {o[1]:2d} {o[2]:3d}  {o[3]:8.2f}         {o[4]:4.2f}      {o[5]:7.2f}     {o[6]:6.2f} {o[7]:4.2f}  {o[8]:6.3f}   {o[9]:.3f}")
json.dump(rows, open(os.environ.get('SCAN_OUT', 'scan_rows.json'), 'w'), default=float)
