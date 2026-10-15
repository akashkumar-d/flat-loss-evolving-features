#!/bin/bash
cd "$(dirname "$0")"
lane() { for s in "$@"; do f=cfgs/F_H_p31_lw0.0150_s$s.json; o=runs/F_H_p31_s$s.json.gz; [ -f "$o" ] || python3 mlp_fourier.py $f $o 13000 100 >> runs/lane.log 2>&1; done; }
lane 3 4 5 6 & lane 7 8 9 10 & wait
