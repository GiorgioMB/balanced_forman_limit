#!/bin/bash
while read -r a b c d e; do f=/tmp/bf/res/curved/c_${a}_${b}_${c}_${d}.npy; [ -f "$f" ] || echo "$a $b $c $d $e"; done < /tmp/bf/jobs_c.txt > /tmp/bf/todo_c.txt
timeout -k 2 150 sh -c "cat /tmp/bf/todo_c.txt | xargs -P 4 -L 1 sh -c 'OMP_NUM_THREADS=1 python3 /tmp/bf/run_curved.py \$0 \$1 \$2 \$3 \$4'"
echo "attempted $(wc -l < /tmp/bf/todo_c.txt); files now: $(ls /tmp/bf/res/curved | wc -l)"
