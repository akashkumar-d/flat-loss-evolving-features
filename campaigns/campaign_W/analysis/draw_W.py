import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import tf_figs as F
import paperdefs_tf as P
res, vres, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
ld = lambda root, st, seeds: [P.load(os.path.join(root, f'{st}_s{i}', 'run.json.gz')) for i in seeds]
cols = [F.column(ld(res, 'W_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275', (1, 2, 3)), 'Polar, $\\lambda_V$ 0.275, $\\eta_W$ 0.1', 12000, band='raw10'),
        F.column(ld(res, 'W_R4_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.25', (3, 4, 5)), 'Polar, $\\lambda_V$ 0.25, $\\eta_W$ 0.1', 12000, band='raw10'),
        F.column(ld(res, 'W_R1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.3', (1, 2, 3)), 'Polar, $\\lambda_V$ 0.3, $\\eta_W$ 0.03', 30000, band='raw10'),
        F.column(ld(res, 'W_N1_f0.2_rms_ngd_lam_w0.01_eta_w0.1_lam_v0.25', (1, 2, 3)), 'NGD, $\\lambda_V$ 0.25, $\\eta_W$ 0.1', 12000, band='raw10')]
F.draw(cols, os.path.join(out, 'W_frac02'), rows=('ce', 'raw', 'err', 'acc', 'ah'),
       suptitle='Transformer, p = 113, training fraction 0.2, RMS head, float64 (campaign W); grey: raw CE within 10% with every training pair fit')
cols = [F.column(ld(vres, 'V_C1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2', (6, 7, 8)), 'C1, float32 (campaign V)', 12000, band='raw10'),
        F.column(ld(res, 'W_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2', (6, 7, 8)), 'C1, float64 (campaign W)', 12000, band='raw10')]
F.draw(cols, os.path.join(out, 'W_C1_precision'), rows=('raw', 'err', 'acc'),
       suptitle='C1 (plain head, fraction 0.3), same seeds: the float32 CE floor near 3e-7 is numerical')
print('ok')
