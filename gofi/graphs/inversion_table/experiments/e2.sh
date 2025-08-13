#!/bin/bash

# Privzeto N
N=15

# Če je argument podan, ga uporabi
if [ ! -z "$1" ]; then
    N=$1
fi

for n in $(seq 1 $N)
do
    m=$((5 * n))
    screen -dmS "simple_$m" bash -c "source ~/venv/bin/activate && python3 -i ~/gofi/gofi/graphs/inversion_table/experiments/e2_shallow_wide.py $m --lambda_lr simple --verbose 2"
    screen -dmS "cosine_$m" bash -c "source ~/venv/bin/activate && python3 -i ~/gofi/gofi/graphs/inversion_table/experiments/e2_shallow_wide.py $m --lambda_lr cosine --verbose 2"
done
