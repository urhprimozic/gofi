#!/bin/bash

#Ns=(5 6)
#LSs=(64 128)

Ns=(8 16 32 64)
LSs=(128 512 1024 2048)

for n in "${Ns[@]}"; do
  for ls in "${LSs[@]}"; do
    session_name="n_${n}_ls_${ls}"
    echo "Starting screen session: $session_name"
    screen -dmS "$session_name" bash -c "
      source ~/venv/bin/activate
      python3 run.py $n --layer_size $ls --lr 0.0001 --T 10 --verbose 1 --loss_sample_size 64
      echo 'Python command finished. Press Ctrl+A D to detach.'
      exec bash
    "
  done
done
