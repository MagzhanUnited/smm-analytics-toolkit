#!/bin/bash
# Guard for the fine-tuning queue: checks every 30 minutes (the user's choice, 26.11.2026) and exits (which notifies the operator) when
# the queue finishes, a run fails, the queue process disappears or swap grows. If macOS reports a
# serious or critical thermal state, training is paused for three minutes and then resumed.
ML="${ML_ROOT:-$HOME/Desktop/projects/aitu/smm-ml}"
LOG="$ML/runs/queue.log"; GUARD="$ML/runs/watchdog.log"; SENSORS="$ML/tools/sensors"
echo "$(date '+%F %T') watchdog started" >> "$GUARD"
INTERVAL="${WATCHDOG_INTERVAL:-1800}"
while true; do
  sleep "$INTERVAL"
  if grep -q "queue finished" "$LOG" 2>/dev/null; then echo "queue finished"; exit 0; fi
  if grep -E "exit [1-9][0-9]* " "$LOG"; then echo "a run failed"; exit 1; fi
  if ! pgrep -f run_queue.sh >/dev/null; then echo "queue process is gone"; exit 4; fi
  state=$("$SENSORS" 2>/dev/null | head -1); hottest=$("$SENSORS" 2>/dev/null | sed -n 3p)
  case "$state" in
    *serious*|*critical*)
      echo "$(date '+%F %T') $state ($hottest): training paused for 3 minutes" >> "$GUARD"
      pkill -STOP -f "finetune.py --name"; sleep 180; pkill -CONT -f "finetune.py --name"
      echo "$(date '+%F %T') training resumed" >> "$GUARD" ;;
  esac
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+).*/\1/')
  if [ "${swap:-0}" -gt 6000 ]; then echo "swap in use: ${swap} MB"; exit 3; fi
  pid=$(pgrep -f "finetune.py --name" | head -1)
  echo "$(date '+%F %T') $state | $hottest | $(footprint "$pid" 2>/dev/null | grep 'phys_footprint:' | tr -s ' ') | swap ${swap} MB" >> "$GUARD"
done
