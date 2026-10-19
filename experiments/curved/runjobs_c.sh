#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../tools/env.sh"

while read -r geo t n_trials chunk dummy; do
  f=$BF_ROOT/res/curved/c_${geo}_${t}_${n_trials}_${chunk}.npy
  [ -f "$f" ] || echo "$geo $t $n_trials $chunk $dummy"
done < $HERE/jobs_c.txt > $BF_ROOT/todo_c.txt

timeout -k 2 150 sh -c "cat $BF_ROOT/todo_c.txt | xargs -P 4 -L 1 sh -c 'OMP_NUM_THREADS=1 python3 $HERE/run_curved.py \$0 \$1 \$2 \$3 \$4'"
echo "attempted $(wc -l < $BF_ROOT/todo_c.txt); files now: $(ls $BF_ROOT/res/curved | wc -l)"
