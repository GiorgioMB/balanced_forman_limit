cd "$(dirname "$0")" && . ../../tools/env.sh
export OMP_NUM_THREADS=1

i=0
for job in "$@"; do
  s=${job%:*}
  seed=${job#*:}
  SEED=$seed timeout -k 1 8 python3 an_e6.py $s > /tmp/rew_$i.log 2>&1 &
  i=$((i+1))
done
wait

cat /tmp/rew_*.log | cut -c1-200
rm -f /tmp/rew_*.log
