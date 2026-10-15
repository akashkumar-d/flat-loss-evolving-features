#!/bin/bash
MAXJOBS=${MAXJOBS:-2}
cd "$(dirname "$0")"
for id in "$@"; do
  while [ "$(pgrep -c -f '^python3 tf_runner.py one')" -ge "$MAXJOBS" ]; do sleep 20; done
  if [ -f "runs/$id/run.json.gz" ]; then echo "$(date -u +%FT%TZ) skip $id (done)"; continue; fi
  CAMPAIGN_THREADS=1 nohup python3 tf_runner.py one "configs/$id.json" "runs/$id/run.json.gz" > "logs/$id.log" 2>&1 &
  echo "$(date -u +%FT%TZ) started $id"; sleep 5
done
echo "$(date -u +%FT%TZ) queue empty"
