#!/bin/bash
# Serial runner for the long measurements.
#
# Written as a file rather than passed to bash -c because the inline version
# failed silently: it piped each step through `tail -100`, which withholds
# everything until the step exits, so when the chain died there was a zero-byte
# log and no way to tell whether the step had failed or never started. Every
# step here streams unbuffered straight to the log and is bracketed by a marker
# and a timestamp, so an interrupted run says where it stopped.
#
# It also detaches itself into its own session on first entry. nohup was not
# enough: two launches died at 7 and 9 minutes with no exception, no non-zero
# rc and no END marker, while an identical measurement run in the foreground
# completed in 12 minutes. Free memory was 65 % and the system log had no
# jetsam kill, so it was not the machine running out -- it was the launching
# shell's process group being reaped and taking the children with it. setsid
# puts this script somewhere that cannot happen.
set -u
if [ "${QUEUE_DETACHED:-}" != "1" ]; then
  export QUEUE_DETACHED=1
  exec python3 -c 'import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
       "$0" "$@"
fi
cd "$(dirname "$0")/.."
export PYTHONPATH=src:tools
export PYTHONUNBUFFERED=1

step() {
  echo "=== BEGIN $1 :: $(date '+%F %T') ==="
  shift
  "$@"
  local rc=$?
  echo "=== END rc=$rc :: $(date '+%F %T') ==="
  return $rc
}

# Wait for any straddling_attachment already running, so the arms are not
# competing for the same ten cores and timing out against each other.
while pgrep -f 'straddling_attachment.py' > /dev/null; do sleep 30; done

# void 135 carries both arms. The first pass ran --arms union only and without
# --write, so it measured +0.0026 on 11/12 and left nothing on disk and no
# control. A family without a control arm is not a result here: the union arm
# adds 1072 tables, and "more tables help" is the reading the more_old arm
# exists to rule out.
step "void 135 (both arms)" \
  python3.12 tools/straddling_attachment.py \
    --family "void 135" --arms union,more_old \
    --out results/architecture_sweep/VOID_LIFT.json --write

step "void permuted 135" \
  python3.12 tools/straddling_attachment.py \
    --family "void permuted 135" --arms union \
    --out results/architecture_sweep/VOID_PERMUTED_LIFT.json --write

# The full stack. backbone and sidechain were 78.8% additive; whether void adds
# on top of both, or is the same buriedness signal arriving by a third route, is
# what decides if it ships.
step "geometry 528 (backbone + sidechain + void)" \
  python3.12 tools/straddling_attachment.py \
    --family "geometry 528" --arms union,more_old \
    --out results/architecture_sweep/GEOMETRY_LIFT.json --write

step "pLM-NN training-fold embed" \
  python3.12 tools/plmnn_by_stratum.py --embed

step "pLM-NN stratify" \
  python3.12 tools/plmnn_by_stratum.py --stratify

echo "=== QUEUE DONE :: $(date '+%F %T') ==="
