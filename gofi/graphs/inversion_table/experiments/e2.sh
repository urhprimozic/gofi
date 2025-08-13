#!/bin/bash

for n in {1..15}
do
    m=$((5*n))
    screen -dmS "simple_$m" bash -c "source ~/venv/bin/activate && python3 -i ~/gofi/gofi/graphs/inversion_table/experiments/e2_shallow_wide.py $m --lambda_lr simple --verbose 2"
done
