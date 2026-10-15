#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

ARGS = sys.argv[1:]
CAMPAIGNS, DATA, AQ, OUT = ARGS[:4]


def size(flag, default):
    if flag in ARGS:
        i = ARGS.index(flag); return float(ARGS[i + 1]), float(ARGS[i + 2])
    return default


DEEP, CONTRAST, PRIMES = size('--deep', (5.6, 3.4)), size('--contrast', (6.75, 3.8)), size('--primes', (6.75, 5.0))
sys.argv = [sys.argv[0], CAMPAIGNS, DATA]
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('PAPER_UNIFORM_DIR', _here))
import paper_uniform as PU


def main():
    out = Path(OUT); rep = {}
    PU.style()
    RN = json.load(open(f'{PU.REL}/campaign_N/analysis/analysis_N.json'))
    rep['N_D3_p47_main'] = PU.draw(PU.deep_cols(RN, 3, 47), out, 'N_D3_p47_main', ('ce', 'acc', 'ah'), DEEP[1], width=DEEP[0])
    for D, p in ((3, 61), (4, 47), (4, 61)):
        rep[f'N_D{D}_p{p}_app'] = PU.draw(PU.deep_cols(RN, D, p), out, f'N_D{D}_p{p}_app', ('ce', 'acc', 'ah'), DEEP[1],
                                          width=DEEP[0])
    PU.style()
    AZ = json.load(open(f'{PU.REL}/campaign_Z/analysis/analysis_Z.json'))['cells']
    zdir = os.path.join(PU.DATA, 'campaign_Z')
    cols = [PU.tf_col("Polar (the paper's rule)", AZ['Y_G2'], zdir, 9000),
            PU.tf_col('Normalized gradient (NGD)', AZ['C_NGD_G2'], zdir, 9000),
            PU.tf_col('AdamW', AZ['C_ADAMW_G2'], zdir, 9000)]
    rep['TF_contrast_main'] = PU.draw(cols, out, 'TF_contrast_main', ('ce', 'acc', 'ah'), CONTRAST[1], width=CONTRAST[0])
    Q = json.load(open(AQ))['cells']
    cols = [PU.tf_col('p = 61, mirror-free, fraction 0.55', Q['P_p61_mf'], '/'),
            PU.tf_col('p = 89, fraction 0.25', Q['P_p89'], '/'),
            PU.tf_col('p = 113, fraction 0.2', Q['P_p113'], '/'),
            PU.tf_col('p = 149, fraction 0.16', Q['P_p149'], '/')]
    rep['TF_acrossp_app'] = PU.draw(cols, out, 'TF_acrossp_app', ('ce', 'acc', 'err', 'ah'), PRIMES[1], width=PRIMES[0],
                                    grid_below=True)
    rep['sizes'] = dict(deep=DEEP, contrast=CONTRAST, primes=PRIMES)
    (out / 'paper_v4_figs_stats.json').write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
