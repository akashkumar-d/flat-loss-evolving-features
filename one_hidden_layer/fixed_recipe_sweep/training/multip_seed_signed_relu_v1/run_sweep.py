from pathlib import Path
import concurrent.futures as futures
import hashlib,json,multiprocessing,os,time,traceback
import numpy as np
import core

HERE=Path(__file__).resolve().parent
ROSTER=[dict(name=f'p{p}_seed{rep+1}',p=p,width=512,replicate=rep+1,
             init_seed=2026091201+3*rep+j,split_seed=2026191201+3*rep+j,
             steps=40000,train_fraction=.8,initial_scale=24.,eta_w=.03,eta_v=.03,
             decay_w=.005,decay_v=.2)
        for rep in range(3) for j,p in enumerate([13,17,23])]
THREADS=['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def first_suffix(mask):
    if not len(mask) or not mask[-1]:return None
    bad=np.flatnonzero(~mask)
    return int(bad[-1]+1) if len(bad) else 0


def window_summary(trace,diags,left,right,ntr,nte):
    if len(trace['step'])<=right:
        return dict(start=left,end=right,status='unavailable',joint_window_pass=None)
    ce=trace['train_ce'][left:right+1]
    dl=next(r for r in diags if r['step']==left);dr=next(r for r in diags if r['step']==right)
    rel=float(np.ptp(ce)/ce.mean()) if ce.mean()>0 else None
    fitted=bool(np.all(trace['train_strict_count'][left:right+1]==ntr))
    gain=float(dr['fourier_energy']-dl['fourier_energy'])
    ts=float(trace['test_strict_count'][left]/nte);tf=float(trace['test_strict_count'][right]/nte)
    certified=float(dr['separated_certified']/nte)
    passed=bool(fitted and rel is not None and rel<=.10 and gain>=.01 and ts<=.50 and tf>=.90 and certified>=.90)
    return dict(start=left,end=right,status='available',relative_ce_range=rel,
        ce_min=float(ce.min()),ce_max=float(ce.max()),all_train_fit=fitted,
        hidden_energy_gain=gain,test_start_fraction=ts,test_end_fraction=tf,
        certified_end_fraction=certified,joint_window_pass=passed)


def run_one(cfg,global_deadline):
    assert all(os.environ.get(k)=='1' for k in THREADS)
    start=time.monotonic();deadline=min(global_deadline,start+900)
    folder=HERE/cfg['name'];folder.mkdir()
    outputs={};diags=[];checkpoints=[];saved_components={};error=None
    observed=-1;updates=0;status='unrun';trace={};ntr=int(.8*cfg['p']**2);nte=cfg['p']**2-ntr
    dump(folder/'LAUNCH.json',dict(config=cfg,work_seconds=900,global_seconds_remaining=max(0,global_deadline-start)))

    def save(path,**arrays):
        with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
        outputs[str(path)]=sha(path)

    try:
        if time.monotonic()<deadline:
            p=cfg['p'];x,y,tr,te,w,v=core.initialize(p,cfg['width'],cfg['init_seed'],cfg['split_seed'])
            assert len(tr)==ntr and len(te)==nte
            assert np.array_equal(np.sort(np.r_[tr,te]),np.arange(p*p))
            trace={k:np.empty(cfg['steps']+1,dtype=int if 'count' in k else float) for k in
                   ['train_ce','test_ce','train_strict_count','test_strict_count','train_margin','test_margin']}
            for step in range(cfg['steps']+1):
                rec,pre,h,f,residual=core.forward(x,y,tr,te,w,v)
                assert all(np.isfinite(val) for val in rec.values()),'Nonfinite observed scalar'
                for key,value in rec.items():trace[key][step]=value
                observed=step
                terminal=(step==cfg['steps']) or time.monotonic()>=deadline
                if step%100==0 or terminal:
                    diag,a,z=core.diagnostics(h,f,y,te,p)
                    assert diag['separated_certified']<=rec['test_strict_count']
                    diags.append(dict(step=step,**diag))
                    if step in [10000,20000]:saved_components[step]=(a.copy(),z.copy())
                if step%1000==0 or terminal:
                    cp=folder/f'checkpoint_{step:05d}.npz'
                    save(cp,W=w,V=v,train_idx=tr,test_idx=te)
                    checkpoints.append(dict(step=step,path=str(cp)))
                if step%5000==0 or terminal:
                    print(json.dumps(dict(run=cfg['name'],step=step,train_ce=rec['train_ce'],
                        test_correct=rec['test_strict_count'],n_test=nte,
                        elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
                if terminal:
                    status='complete' if step==cfg['steps'] else 'partial'
                    break
                c,d=core.forces(x[tr],pre[tr],h[tr],residual,v)
                wn,vn,rank=core.update(w,v,c,d)
                assert np.isfinite(wn).all() and np.isfinite(vn).all(),'Nonfinite proposed parameters'
                w,v=wn,vn;updates+=1
    except Exception as exc:
        status='failed';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    windows=[];summary={}
    try:
        if observed>=0:
            trace={k:value[:observed+1] for k,value in trace.items()}
            trace['step']=np.arange(observed+1)
            save(folder/'trajectory.npz',**trace)
            if diags:
                arrays={k:np.array([np.nan if r[k] is None else r[k] for r in diags]) for k in diags[0]}
                save(folder/'diagnostics.npz',**arrays)
            if updates==observed and not any(cp['step']==observed for cp in checkpoints):
                cp=folder/f'checkpoint_{observed:05d}.npz';save(cp,W=w,V=v,train_idx=tr,test_idx=te)
                checkpoints.append(dict(step=observed,path=str(cp)))
            if all(s in saved_components for s in [10000,20000]):
                crosses={k:np.empty((2,2),dtype=int if 'correct' in k else float) for k in
                         ['train_ce','test_ce','train_correct','test_correct']}
                for i,rs in enumerate([10000,20000]):
                    for j,zs in enumerate([10000,20000]):
                        result=core.table_metrics(saved_components[rs][0]+saved_components[zs][1],y,tr,te)
                        for key,value in result.items():crosses[key][i,j]=value
                        if i==j:
                            assert result['train_correct']==trace['train_strict_count'][rs]
                            assert result['test_correct']==trace['test_strict_count'][rs]
                            assert abs(result['train_ce']-trace['train_ce'][rs])<=2e-10
                            assert abs(result['test_ce']-trace['test_ce'][rs])<=2e-10
                save(folder/'crossed_components.npz',step=np.array([10000,20000]),**crosses)
            windows=[window_summary(trace,diags,l,r,ntr,nte) for l,r in [(10000,20000),(15000,20000)]]
            fitted=first_suffix(trace['train_strict_count']==ntr)
            recovered=first_suffix(trace['test_strict_count']/nte>=.90)
            bad_interval=(fitted is not None and fitted+1000<=observed and
                          bool(np.all(trace['test_strict_count'][fitted:fitted+1001]/nte<=.10)))
            delayed=bool(status=='complete' and bad_interval and recovered is not None and recovered>=fitted+1000)
            summary=dict(train_ce=float(trace['train_ce'][-1]),test_ce=float(trace['test_ce'][-1]),
                train_correct=int(trace['train_strict_count'][-1]),test_correct=int(trace['test_strict_count'][-1]),
                first_train_full_suffix=fitted,first_test90_suffix=recovered,
                first_test100_suffix=first_suffix(trace['test_strict_count']==nte),
                initial_fitted_bad_test_1000=bool(bad_interval),delayed_recovery=delayed,
                hidden_energy=diags[-1]['fourier_energy'] if diags else None,
                separated_certified=diags[-1]['separated_certified'] if diags else None)
    except Exception as exc:
        status='failed';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),previous_error=error)
    result=dict(status=status,config=cfg,p=cfg['p'],n_train=ntr,n_test=nte,
        completed_steps=max(0,observed),observed_states=observed+1,updates_completed=updates,
        pending_unobserved_update=updates>observed if observed>=0 else False,
        checkpoints=checkpoints,windows=windows,final=summary,error=error,
        output_sha256=outputs,elapsed_seconds=time.monotonic()-start,
        initialization_theorem_proved=False,causal_optimizer_theorem_proved=False)
    dump(folder/'RESULTS.json',result)
    dump(folder/'CLOSED.json',dict(status=status,results_sha256=sha(folder/'RESULTS.json')))
    print(json.dumps(dict(run=cfg['name'],status=status,completed_steps=max(0,observed),
        final=summary,windows=windows,elapsed_seconds=result['elapsed_seconds'])),flush=True)
    return result


def main():
    assert all(os.environ.get(k)=='1' for k in THREADS)
    start=time.monotonic();deadline=start+1800
    sources={str(HERE/k):sha(HERE/k) for k in ['SCOPE.md','core.py','test_core.py','test_runner.py','run_sweep.py','REVIEW.md']}
    original=HERE.parent/'fitted_polar_head_grid_v1/run_diagnostic.py';sources[str(original)]=sha(original)
    dump(HERE/'LAUNCH.json',dict(roster=ROSTER,sources=sources,max_workers=3,
        per_arm_work_seconds=900,global_work_seconds=1800,total_seconds=1860,
        declared_new_updates=360000,seeded_gaussian_rng='NumPy default_rng',original_run_included=False))
    collected={};failures=[]
    with futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(run_one,cfg,deadline):cfg for cfg in ROSTER}
        for job in futures.as_completed(jobs):
            cfg=jobs[job]
            try:collected[cfg['name']]=job.result()
            except Exception as exc:
                failures.append(dict(run=cfg['name'],message=str(exc),traceback=traceback.format_exc()))
                collected[cfg['name']]=dict(status='failed',config=cfg,p=cfg['p'],n_train=int(.8*cfg['p']**2),
                    n_test=cfg['p']**2-int(.8*cfg['p']**2),completed_steps=0,observed_states=0,
                    updates_completed=None,output_sha256={},windows=[],final={},error=failures[-1])
    runs=[collected[cfg['name']] for cfg in ROSTER]
    outputs={};binding_errors=[]
    for cfg in ROSTER:
        for name in ['RESULTS.json','CLOSED.json']:
            path=HERE/cfg['name']/name
            if path.exists():outputs[str(path)]=sha(path)
    for path,expected in sources.items():
        if sha(path)!=expected:binding_errors.append(path)
    elapsed=time.monotonic()-start
    status='complete' if all(r['status']=='complete' for r in runs) and not failures and not binding_errors else 'partial_or_failed'
    if elapsed>1860:status='resource_overrun'
    result=dict(status=status,roster=ROSTER,runs=runs,declared_runs=9,
        completed_runs=sum(r['status']=='complete' for r in runs),
        total_new_updates=sum(r['updates_completed'] or 0 for r in runs),
        output_sha256=outputs,source_sha256=sources,failures=failures,source_binding_errors=binding_errors,
        elapsed_seconds=elapsed,independent_seed_pairs=True,iid_entry_sampling=False,
        initialization_theorem_proved=False,causal_optimizer_theorem_proved=False)
    dump(HERE/'RESULTS.json',result)
    dump(HERE/'CLOSED.json',dict(status=status,results_sha256=sha(HERE/'RESULTS.json')))
    print(json.dumps(dict(status=status,completed_runs=result['completed_runs'],
        total_new_updates=result['total_new_updates'],elapsed_seconds=elapsed)),flush=True)


if __name__=='__main__':main()
