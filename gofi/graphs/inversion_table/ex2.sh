#!/bin/bash

# Arrays of values
Ns=(8 16 32 64)
LSs=(128 512 1024 2048)

# Loop over values
for n in "${Ns[@]}"; do
  for ls in "${LSs[@]}"; do
    session_name="n_${n}_ls_${ls}"
    echo "Starting screen session: $session_name"
    screen -dmS "$session_name" bash -c "bash -i -c 'source ~/venv/bin/activate && python3 run.py $n --layer_size $ls --lr 0.0001 --T 10 --verbose 1 --grad_clipping 2 --loss stochastic --scheduler cosine_wr; exec bash"
  done
done
