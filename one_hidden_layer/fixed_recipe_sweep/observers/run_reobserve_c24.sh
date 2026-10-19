#!/usr/bin/env bash
SRC="${1:?usage: run_reobserve_c24.sh <folder with p*_seed* runs> <out_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:?usage: run_reobserve_c24.sh <folder with p*_seed* runs> <out_dir>}"; mkdir -p "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ls -d "$SRC"/p*_seed* | xargs -P 4 -I{} sh -c 'r=$(basename "{}"); [ -s "'"$OUT"'/$r.jsonl" ] || python3 "'"$HERE"'/reobserve_c24.py" "{}" "'"$OUT"'/$r.jsonl" > "'"$OUT"'/$r.log" 2>&1'
echo "done: $(ls "$OUT"/*.jsonl | wc -l) jsonl"
