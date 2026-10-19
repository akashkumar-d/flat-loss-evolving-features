#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

CAMPAIGNS, DATA, AQ, OUT = sys.argv[1:5]
sys.argv = [sys.argv[0], CAMPAIGNS, DATA]
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('PAPER_UNIFORM_DIR', os.path.join(_here, 'lib')))
import paper_uniform as PU
sys.argv = [sys.argv[0], CAMPAIGNS, DATA, AQ, OUT]
import paper_q_figs as PQ


def main():
    out = Path(OUT); rep = {}
    PU.style()
    AZ = json.load(open(f'{PU.REL}/transformer/gain_recipe_contrasts_two_blocks/analysis/analysis_Z.json'))['cells']
    zdir = os.path.join(PU.DATA, 'campaign_Z')
    cols = [PU.tf_col("Polar (the paper's rule)", AZ['Y_G2'], zdir, 9000),
            PU.tf_col('Normalized gradient (NGD)', AZ['C_NGD_G2'], zdir, 9000),
            PU.tf_col('AdamW', AZ['C_ADAMW_G2'], zdir, 9000)]
    rep['TF_contrast_main'] = PU.draw(cols, out, 'TF_contrast_main', ('ce', 'acc', 'ah'), 3.8, width=6.75, refit=False)
    Q = json.load(open(AQ))['cells']
    cols = [PU.tf_col('p = 61, mirror-free, fraction 0.55', Q['P_p61_mf'], '/'),
            PU.tf_col('p = 89, fraction 0.25', Q['P_p89'], '/'),
            PU.tf_col('p = 113, fraction 0.2', Q['P_p113'], '/'),
            PU.tf_col('p = 149, fraction 0.16', Q['P_p149'], '/')]
    rep['TF_acrossp_app'] = PU.draw(cols, out, 'TF_acrossp_app', ('ce', 'acc', 'err', 'ah'), 5.0, width=6.75,
                                    grid_below=True, refit=False)
    PU.style()
    rep['TF_two_blocks_app'] = PQ.two_blocks_row(PU.tf_col('Two blocks, fraction 0.2', Q['B_base'], '/'), out,
                                                 'TF_two_blocks_app', refit=False)
    (out / 'paper_v7_figs_stats.json').write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
