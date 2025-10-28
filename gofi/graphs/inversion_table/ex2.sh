#!/bin/bash

# Ns
Ns=(8 10 16 20 32 )
# 64)

# Corresponding layer sizes for each N (as space-separated strings)
LSs_list=(
  "64"          # for N=8
  "128 256"      # for N=10
  "256 512"  # for N=16
  "512"         # for N=20
  "1024"    # for N=32
  #"512 1024"        # for N=64
)

for i in "${!Ns[@]}"; do
  n=${Ns[$i]}
  LSs=(${LSs_list[$i]})  # convert string to array

  for ls in "${LSs[@]}"; do
    session_name="ex2_n_${n}_ls_${ls}"
    echo "Starting screen session: $session_name"
    screen -dmS "$session_name" bash -c "
      source ~/venv/bin/activate
      python3 run.py $n --layer_size $ls --lr 0.0001 --T 10 --verbose 1 --name $session_name --loss_sample_size 64
      echo 'Python command finished. Press Ctrl+A D to detach.'
      exec bash
    "
  done
done
