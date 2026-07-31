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

while pgrep -f 'straddling_attachment.py' > /dev/null; do sleep 30; done

# The permuted arm is built first so the two measurements run back to back with
# no build between them.
step "build the permuted displacement matrix" \
  python3.12 tools/displacement_wires.py --permuted

step "displacement 144 (both arms)" \
  python3.12 tools/straddling_attachment.py \
    --family "displacement 144" --arms union,more_old \
    --out results/architecture_sweep/DISPLACEMENT_LIFT.json --write

step "displacement permuted 144" \
  python3.12 tools/straddling_attachment.py \
    --family "displacement permuted 144" --arms union \
    --out results/architecture_sweep/DISPLACEMENT_PERMUTED_LIFT.json --write

echo "=== QUEUE DONE :: $(date '+%F %T') ==="
