#!/bin/bash

for n in {5..50}
do
    screen -dmS "simple_$n" bash -c "source /venv/bin/activate && python3 -i /gofi/gofi/graphs/inversion_table/experiments/e2_shallow_wide.py $n --lambda_lr simple"
done
