#!/usr/bin/env python3
import glob, os, sys
from pathlib import Path

BIAS_DIR, COMPACT_DIR = sys.argv[1], sys.argv[2]
QD = sys.argv[sys.argv.index('--qd') + 1] if '--qd' in sys.argv else None
HEIGHT = float(sys.argv[sys.argv.index('--height') + 1]) if '--height' in sys.argv else 2.25
OLD = '--old' in sys.argv
for flag in ('--qd', '--height'):
    if flag in sys.argv: i = sys.argv.index(flag); del sys.argv[i:i + 2]
if OLD: sys.argv.remove('--old')
sys.argv = [sys.argv[0], BIAS_DIR] + sys.argv[3:]
sys.path.insert(0, COMPACT_DIR)
import bias_figs_compact as BC
BF = BC.BF
BC.HEIGHT = HEIGHT

if not OLD:
    _init = BF.Store.__init__

    def init(self, data, dev2, bias):
        _init(self, data, dev2, bias)
        for d in sorted(glob.glob(os.path.join(QD, 'QD_*'))):
            f = os.path.join(d, 'run.json.gz')
            if os.path.exists(f): self._add(os.path.basename(d), 'qd', f)
    BF.Store.__init__ = init
    spec, rows, height, paper, creator = BF.SPECS['fig3_depths_2to5']
    BF.SPECS['fig3_depths_2to5'] = (lambda st: [('Depth 2', [f'QD_D2_s{i:03d}' for i in range(1, 41)],
                                                 [f'QD_B_D2_s{i:03d}' for i in range(1, 41)])] + spec(st)[1:],
                                    rows, height, paper, creator)

if __name__ == '__main__':
    BF.main()
