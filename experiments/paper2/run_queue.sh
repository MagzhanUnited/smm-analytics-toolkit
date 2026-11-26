#!/bin/bash
# Fine-tuning queue of the second paper. Finished runs are skipped, so the script can be restarted.
#   caffeinate -i experiments/paper2/run_queue.sh
#
# Protocol (fixed 26.11.2026, 05:45): every run has a fixed number of epochs that depends only on the
# number of training reviews (40 for 250, 10 for 1,000, 6 for 4,000, 5 for 16,000 and more), the model is
# validated after every epoch, and the epoch with the best validation macro-F1 is kept. There is no early
# stopping (patience = epochs). Both loss variants (plain and class-weighted cross-entropy) are trained
# at every size.
# Why the night version was replaced: it gave 16,000 reviews four epochs and 64,000 three, while the full
# set had five, so the points of the learning curve had unequal budgets (validation macro-F1 was still
# rising at the last epoch of the 64,000 run); and the class-weighted run on the full set was stopped by
# a patience of one epoch after a single dip at epoch 3 of 5. The three affected runs were moved to
# runs/superseded/ and are repeated here. The cleaned-text replication of the binary task was dropped:
# the raw-text run already reproduces the published score.
# Order changed 26.11.2026, 08:00: training on this graphics processor is not deterministic (the same seed
# and settings gave validation macro-F1 0.5576 and 0.5653 after the first epoch in two runs), so one run
# per loss variant cannot decide which variant is the main one. Both variants now get three seeds on the
# full set before anything else; the ten-epoch run comes last and uses the variant with the higher mean
# validation macro-F1 over the three seeds. Before it, while nothing else runs, the timings of the
# baselines are measured again (baseline_timing.py): the first ones were taken during a training run.
# Changed 26.11.2026, 10:30 (before any repetition on the full set started): the full runs of seeds 2 and 3
# save their models as seed 1 did, and after each of them the identifier probe and its neutral-word control
# run on the idle graphics processor (probe_transitions.py), so that the statements about identifiers can be
# checked on three seeds of both loss variants and not on one. Training itself is unchanged.
# Changed 26.11.2026, 11:40: a third probe is added for every saved model (probe_strings.py: five random identifiers
# per format and controls with several words and with digit groups), because an independent review showed that the
# effect of the fixed identifier strings on the linear model comes from single tokens such as "123". The models of
# seed 1 get this probe too. Training itself is unchanged.
cd "$(dirname "$0")" || exit 1
ML="${ML_ROOT:-$HOME/Desktop/projects/aitu/smm-ml}"
PY="$ML/venv/bin/python"
LOG="$ML/runs/queue.log"
mkdir -p "$ML/runs"
run() {  # run <name> <args...>
  local name="$1"; shift
  echo "=== $(date '+%F %T') start $name" >> "$LOG"
  "$PY" -I -B finetune.py --name "$name" "$@" >> "$LOG" 2>&1
  echo "=== $(date '+%F %T') exit $? $name" >> "$LOG"
}
variant() {  # sets tag and weight for <plain|weighted>
  if [ "$1" = "weighted" ]; then tag="_weighted"; weight="balanced"; else tag=""; weight="none"; fi
}
curve() {  # curve <seed> <plain|weighted>
  local s="$1" tag weight; variant "$2"
  run "xlmr_p3_n250${tag}_s$s"   --task p3 --train-size 250   --seed "$s" --epochs 40 --patience 40 --batch 16 --no-battery --class-weight "$weight"
  run "xlmr_p3_n1000${tag}_s$s"  --task p3 --train-size 1000  --seed "$s" --epochs 10 --patience 10 --batch 16 --no-battery --class-weight "$weight"
  run "xlmr_p3_n4000${tag}_s$s"  --task p3 --train-size 4000  --seed "$s" --epochs 6  --patience 6  --no-battery --class-weight "$weight"
  run "xlmr_p3_n16000${tag}_s$s" --task p3 --train-size 16000 --seed "$s" --epochs 5  --patience 5  --no-battery --class-weight "$weight"
  run "xlmr_p3_n64000${tag}_s$s" --task p3 --train-size 64000 --seed "$s" --epochs 5  --patience 5  --no-battery --class-weight "$weight"
}
full() {  # full <seed> <plain|weighted> [extra args]
  local s="$1" tag weight; variant "$2"; shift 2
  run "xlmr_p3_full${tag}_s$s" --task p3 --seed "$s" --epochs 5 --patience 5 --class-weight "$weight" "$@"
}
stored() {  # stored <key of results/identifier_transitions.json> <run>: true when the probe of the run is in the file
  "$PY" -I -B -c "import json, sys; sys.exit(0 if sys.argv[3] in json.load(open(sys.argv[1])).get(sys.argv[2], {}) else 1)" results/identifier_transitions.json "$1" "$2" 2>/dev/null
}
probe() {  # probe <run with a saved model>: the three probes of a saved model on the idle graphics processor; a stored probe is skipped
  local name="$1" s1="-" s2="-" s3="-"
  if [ ! -d "$ML/runs/$name/model" ]; then echo "=== $(date '+%F %T') probe skipped, no saved model: $name" >> "$LOG"; return; fi
  if ! stored models "$name"; then "$PY" -I -B probe_transitions.py --model "$name" --device mps >> "$LOG" 2>&1; s1=$?; fi
  if ! stored controls "$name"; then "$PY" -I -B probe_transitions.py --model "$name" --device mps --control >> "$LOG" 2>&1; s2=$?; fi
  if [ ! -f "$ML/runs/appended_strings/$name.npz" ]; then "$PY" -I -B probe_strings.py --model "$name" --device mps >> "$LOG" 2>&1; s3=$?; fi
  if [ "$s1$s2$s3" != "---" ]; then echo "=== $(date '+%F %T') probe of $name: identifiers status $s1, control status $s2, appended strings status $s3" >> "$LOG"; fi
}
# a run started by the previous copy of this script may still be training: let it finish first
while pgrep -f "finetune.py --name" >/dev/null; do sleep 5; done
echo "=== $(date '+%F %T') queue restarted: appended-string probe added for every saved model" >> "$LOG"
full 1 plain --save-model            # done at night (five epochs, the last one the best)
run xlmr_pc_full_s1 --task pc --seed 1 --epochs 5 --patience 5   # done at night (five epochs)
curve 1 plain
curve 1 weighted
full 1 weighted --save-model
probe xlmr_p3_full_s1; probe xlmr_p3_full_weighted_s1   # models saved at night: only the probe that is not stored yet runs
curve 2 plain
curve 2 weighted
curve 3 plain
curve 3 weighted
full 2 plain --save-model;    probe xlmr_p3_full_s2
full 2 weighted --save-model; probe xlmr_p3_full_weighted_s2
full 3 plain --save-model;    probe xlmr_p3_full_s3
full 3 weighted --save-model; probe xlmr_p3_full_weighted_s3
# nothing is training now: measure the time and throughput of the baselines on an idle machine
echo "=== $(date '+%F %T') start baseline_timing" >> "$LOG"
"$PY" -I -B baseline_timing.py >> "$LOG" 2>&1
echo "=== $(date '+%F %T') exit $? baseline_timing" >> "$LOG"
# the main loss variant (rule fixed 26.11.2026, 08:35, before the seed repetitions): the higher mean validation macro-F1
# over the three seeds; if the two means differ by less than the spread of either variant over its seeds, the higher
# mean validation accuracy decides. build_paper.py applies the same rule.
selected=$("$PY" -I -B -c "
import json, statistics, sys
runs = sys.argv[1]
def summary(tag):
    rows = [json.load(open(f'{runs}/xlmr_p3_full{tag}_s{seed}/metrics.json')) for seed in (1, 2, 3)]
    f1 = [r['best_valid_macro_f1'] for r in rows]
    acc = [r['history'][r['best_epoch'] - 1]['valid_accuracy'] for r in rows]
    return statistics.mean(f1), max(f1) - min(f1), statistics.mean(acc)
plain, weighted = summary(''), summary('_weighted')
tied = abs(plain[0] - weighted[0]) < max(plain[1], weighted[1])
key = 2 if tied else 0
print('weighted' if weighted[key] > plain[key] else 'plain')" "$ML/runs")
echo "=== $(date '+%F %T') selected loss variant (mean validation macro-F1 over three seeds, accuracy if tied): $selected" >> "$LOG"
# validation macro-F1 of the plain run was still rising after five epochs, so one run gets ten epochs
if [ "$selected" = "plain" ] || [ "$selected" = "weighted" ]; then
  variant "$selected"
  run "xlmr_p3_full${tag}_e10_s1" --task p3 --seed 1 --epochs 10 --patience 10 --class-weight "$weight"
fi
echo "=== $(date '+%F %T') queue finished" >> "$LOG"
