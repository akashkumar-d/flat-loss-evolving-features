import glob, os
import paperdefs_tf as P, tf_figs as F
E = '/home/claude/kt/campaign_W/explore/runs'
V = '/home/claude/kt/campaign_V/collected/results'
def ld(pat): return [P.load(f) for f in sorted(glob.glob(pat))]
cols = [F.column(ld(f'{V}/V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5_s1/run.json.gz'), 'Polar, λ_V 0.5 (V, float32)', xmax=8000, band='raw10'),
        F.column(ld(f'{E}/P_rms_lv0.3_ew0.1_s1.json.gz'), 'Polar, λ_V 0.3', xmax=8000, band='raw10'),
        F.column(ld(f'{E}/P_rms_lv0.25_ew0.1_s[12].json.gz'), 'Polar, λ_V 0.25', xmax=8000, band='raw10'),
        F.column(ld(f'{E}/P_ngd_rms_lv0.25_ew0.1_s1.json.gz'), 'NGD, λ_V 0.25', xmax=8000, band='raw10')]
F.draw(cols, 'figs/pilots_rms_frac0.2', rows=('ce', 'err', 'acc', 'ah'),
       suptitle='RMS head, frac 0.2, η_W 0.1. Gray: raw CE within 10% and every training pair correct')
print('ok')
