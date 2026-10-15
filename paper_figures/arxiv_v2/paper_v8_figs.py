#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

CAMPAIGNS, DATA, AQ, OUT = sys.argv[1:5]
sys.argv = [sys.argv[0], CAMPAIGNS, DATA]
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('PAPER_UNIFORM_DIR', _here))
import paper_uniform as PU


def main():
    out = Path(OUT); PU.style()
    Q = json.load(open(AQ))['cells']
    cols = [PU.tf_col('One block, p = 89', Q['P_p89'], '/'),
            PU.tf_col('One block, p = 113', Q['P_p113'], '/'),
            PU.tf_col('Two blocks, p = 113', Q['B_base'], '/')]
    rep = {'TF_main': PU.draw(cols, out, 'TF_main', ('ce', 'acc', 'ah'), 3.8, width=6.75, grid_below=True, refit=False)}
    (out / 'paper_v8_figs_stats.json').write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
