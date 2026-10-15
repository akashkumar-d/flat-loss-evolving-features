import glob, os, sys
import paperdefs_tf as P, tf_figs as F
T = '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_T/collected/results'
U = '/mnt/user-data/uploads/modular arithmetic/natural_plateau_shared/campaign_U/collected/results'
V = '/home/claude/kt/campaign_V/collected/results'
def ld(root, pat): return [P.load(f) for f in sorted(glob.glob(os.path.join(root, pat, 'run.json.gz')))]
c1 = ld(V, 'V_C1_*') + ld(U, 'U_L1_*') + ld(T, 'T_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2_s*')
e1a = ld(V, 'V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5_s*')
e1b = ld(V, 'V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.5_s*')
ad = ld(U, 'U_A1_adamw_wd1_s*')
cols = [F.column(c1, 'C1: plain head, frac 0.3', xmax=10000), F.column(e1a, 'RMS, λ_V 0.5, η_W 0.1', xmax=10000),
        F.column(e1b, 'RMS, λ_V 0.5, η_W 0.03', xmax=16000), F.column(ad, 'AdamW, plain, frac 0.3', xmax=10000)]
F.draw(cols, 'figs/existing_paperdefs', rows=('raw', 'err', 'acc', 'ah'), ce_log=True)
print('ok')
