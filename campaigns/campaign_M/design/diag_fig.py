import sys, math, numpy as np
sys.path.insert(0, '/home/claude/kt/campaign_L')
import endpoints_i as E
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linewidth': 0.5, 'lines.linewidth': 1.6, 'legend.frameon': False})
BLUES = ['#9ecae1', '#3182bd', '#08306b']; ORANGES = ['#fdae6b', '#e6550d', '#7f2704']; PURPLE = '#7B3294'

def curves(fn, every=200):
    d = E.load(fn); c = d['cfg']; lp = math.log(c['p'])
    L = np.asarray(d['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2
    t = np.arange(len(cm)); k = np.arange(0, len(cm), every)
    s, te = E.series(d['obs'], 'test_acc'); sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    sn, nv = E.series(d['obs'], 'normV'); ki = np.where(nv >= 0.99 / c['lam_v'])[0]; knee = int(sn[ki[0]]) if len(ki) else 0
    pk = knee + int(np.argmax(cm[knee:]))
    return dict(t=t[k] / 1e3, ce=cm[k], s=s / 1e3, te=te, sr=sr / 1e3, rf=rf, pk=pk / 1e3, cepk=cm[pk])

D = '/home/claude/v7work/data/D'; L = '/home/claude/kt/campaign_L/collected/results'
fig, ax = plt.subplots(2, 2, figsize=(9.2, 4.6), sharex='col', gridspec_kw=dict(hspace=0.12, wspace=0.18))
for j, r in enumerate(('0.90', '0.45', '0.20')):
    c = curves(f'{D}/D3_p61_r{r}_s1/run.json.gz')
    ax[0, 0].plot(c['t'], c['ce'], color=BLUES[2 - j], label=f'r = {float(r):g}')
    ax[0, 0].plot([c['pk']], [c['cepk']], 'o', ms=4, color=BLUES[2 - j])
    ax[1, 0].plot(c['s'], c['te'], color=ORANGES[2 - j], label=f'held-out, r = {float(r):g}')
ax[0, 0].set_title('Fraction 0.8 (campaign D, seed 1): last-layer rate r varied', fontsize=9)
c = curves(f'{L}/L_D3_p61_lv0.2000_lw0.015_s1/run.json.gz')
ax[0, 1].plot(c['t'], c['ce'], color=BLUES[1], label='r = 0.45')
ax[0, 1].plot([c['pk']], [c['cepk']], 'o', ms=4, color=BLUES[1])
ax[1, 1].plot(c['s'], c['te'], color=ORANGES[1], label='held-out')
ax[1, 1].plot(c['sr'], c['rf'], color=PURPLE, label='ridge refit')
ax[0, 1].set_title('Fraction 0.5 (campaign L, seed 1): r = 0.45 as in the paper', fontsize=9)
for a in ax[:, 1]: a.axvline(c['pk'], color='0.5', lw=0.8, ls='--')
ax[0, 1].text(c['pk'] + 3, 0.05, 'CE peak', color='0.4', fontsize=8)
ax[0, 0].set_ylabel('Training CE / log p\n(cycle mean)'); ax[1, 0].set_ylabel('Accuracy')
for a in ax[0]: a.set_ylim(0, 1.0)
for a in ax[1]: a.set_ylim(-0.02, 1.02); a.set_xlabel('Updates (thousands)')
ax[0, 0].set_xlim(0, 60); ax[0, 1].set_xlim(0, 160)
ax[0, 0].legend(loc='lower right', fontsize=8); ax[1, 1].legend(loc='center right', fontsize=8)
ax[1, 0].legend(loc='lower right', fontsize=8)
fig.suptitle(r'Depth 3, p = 61, $\lambda_V = 0.2$, $\lambda_W = 0.015$: the CE peak moves with r, the held-out rise hardly does', fontsize=10, y=0.995)
fig.savefig(sys.argv[1], dpi=170, bbox_inches='tight')
print('ok')
