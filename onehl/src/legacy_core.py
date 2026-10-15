import math
import numpy as np


def stable_ce_residual(logits, labels):
    relative=logits-logits[np.arange(len(labels)),labels,None]
    relative[np.arange(len(labels)),labels]=-np.inf
    largest=relative.max(1)
    log_rivals=largest+np.log(np.exp(relative-largest[:,None]).sum(1))
    ce=np.logaddexp(0.,log_rivals)
    rival_probs=np.exp(relative-ce[:,None])
    residual=-rival_probs
    residual[np.arange(len(labels)),labels]=rival_probs.sum(1)
    return ce,residual


def forces(x,pre,h,residual,v):
    n=len(x)
    return ((residual@v)*(pre>0)).T@x/n,residual.T@h/n


def polar_direction(force):
    scale=float(np.abs(force).max())
    if scale==0:return np.zeros_like(force),0
    u,s,vh=np.linalg.svd(force/scale,full_matrices=False)
    keep=s>np.finfo(np.float64).eps*max(force.shape)*s[0]
    rank=int(keep.sum())
    return (u[:,keep]@vh[keep])/math.sqrt(rank) if rank else np.zeros_like(force),rank


def head_direction(force):
    scale=float(np.abs(force).max())
    if scale==0:return np.zeros_like(force)
    normalized=force/scale
    return normalized/np.linalg.norm(normalized)


def update(w,v,c,d):
    q,rank=polar_direction(c)
    return w+.03*(q-.005*w),v+.03*(head_direction(d)-.2*v),rank


def forward(x,y,tr,te,w,v):
    pre=x@w.T;h=np.maximum(pre,0.);f=h@v.T
    ce,residual=stable_ce_residual(f[tr],y[tr]);test_ce,_=stable_ce_residual(f[te],y[te])
    other=f.copy();other[np.arange(len(y)),y]=-np.inf
    margins=f[np.arange(len(y)),y]-other.max(1)
    values=dict(train_ce=float(ce.mean()),test_ce=float(test_ce.mean()),
        train_strict_count=int(np.sum(margins[tr]>0)),test_strict_count=int(np.sum(margins[te]>0)),
        train_margin=float(margins[tr].min()),test_margin=float(margins[te].min()))
    return values,pre,h,f,residual


def diagnostics(h,f,y,te,p):
    centered=h-h.mean(0);hm=np.stack([centered[y==s].mean(0) for s in range(p)])
    denom=float(np.sum(centered**2))
    hidden_energy=float(np.sum(hm[y]**2)/denom) if denom else 0.
    f=f-f.mean(1,keepdims=True)
    means=np.stack([f[y==s].mean(0) for s in range(p)])
    a=means[y]-f.mean(0);z=f-a
    rule=means-f.mean(0)
    g=rule[np.arange(p),np.arange(p),None]-rule
    g[np.arange(p),np.arange(p)]=np.inf
    gamma=g.min(1)
    complement=z[np.arange(len(y)),y,None]-z
    complement[np.arange(len(y)),y]=np.inf
    beta=-complement.min(1)
    lower=gamma[y]-beta
    actual=f[np.arange(len(y)),y,None]-f
    actual[np.arange(len(y)),y]=np.inf
    worst=actual.min(1)
    ratio=None
    if np.all(gamma>0):
        ratio=float(np.max((np.maximum(-complement,0.)/g[y])[te]))
    fnorm=float(np.sum(f*f))
    rec=dict(fourier_energy=hidden_energy,logit_fourier_energy=float(np.sum(a*a)/fnorm) if fnorm else 0.,
        rule_min=float(gamma.min()),harmful_max=float(beta[te].max()),
        global_lower=float(gamma.min()-beta[te].max()),row_lower_min=float(lower[te].min()),
        separated_certified=int(np.sum(lower[te]>0)),test_min_margin=float(worst[te].min()),
        ratio_max=ratio)
    assert np.all(lower<=worst+2e-10)
    return rec,a,z


def table_metrics(f,y,tr,te):
    a,_=stable_ce_residual(f[tr],y[tr]);b,_=stable_ce_residual(f[te],y[te])
    others=f.copy();others[np.arange(len(y)),y]=-np.inf
    margin=f[np.arange(len(y)),y]-others.max(1)
    return dict(train_ce=float(a.mean()),test_ce=float(b.mean()),
        train_correct=int(np.sum(margin[tr]>0)),test_correct=int(np.sum(margin[te]>0)))


def initialize(p,width,init_seed,split_seed):
    rng=np.random.default_rng(init_seed)
    w=24.*rng.normal(size=(width,2*p))/math.sqrt(p)
    v=rng.normal(size=(p,width))/(24.*math.sqrt(width))
    perm=np.random.default_rng(split_seed).permutation(p*p);n=int(.8*p*p)
    tr=np.sort(perm[:n]);te=np.sort(perm[n:])
    first=np.repeat(np.arange(p),p);second=np.tile(np.arange(p),p)
    x=np.concatenate([np.eye(p)[first],np.eye(p)[second]],1);y=(first+second)%p
    return x,y,tr,te,w,v
