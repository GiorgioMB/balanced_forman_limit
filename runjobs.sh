#!/bin/bash
# runjobs.sh jobsfile script resdir prefix: runs missing jobs (4 parallel), within ~150 s
while read -r a b c d e; do
  f=$3/${4}_${a}_${b}_${c}_${e}.npy
  [ -f "$f" ] || echo "$a $b $c $d $e"
done < "$1" > /tmp/bf/todo.txt
timeout -k 2 150 sh -c "cat /tmp/bf/todo.txt | xargs -P 4 -L 1 sh -c 'OMP_NUM_THREADS=1 python3 $2 \$0 \$1 \$2 \$3 \$4'"
echo "attempted $(wc -l < /tmp/bf/todo.txt); files now: $(ls $3 | wc -l)"
