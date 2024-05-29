#!/bin/zsh

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

port=6035

# Only one dataset specified here
declare -a run_args=(
    "bicycle"
    "bonsai"
    "counter"
    "kitchen"
    "room"
    "stump"
    "garden"
    "flowers"
    "treehill"
  )

declare -a prune_percents=(0.0 0.2 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.9 0.95 0.98)


# Loop over the arguments array
for arg in "${run_args[@]}"; do
  for i in "${(@i)prune_percents}"; do
    prune_percent="${prune_percents: i}"

    for prune_type in "${prune_types[@]}"; do
      # Wait for an available GPU
      while true; do
        gpu_id=0
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting train.py with dataset '$arg', prune_percent '$i', prune_type 'mini_splatting' on port $port"
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python train.py \
            -s "path_to_dataset/$arg" \
            -m "output/mini_splatting/360v2/${arg}_${i}" \
            --prune_method "mini_splatting" \
            --prune_percent $i \
            --eval \
            --wandb_run_name "${arg}_${i}"\
            --port $port > "logs/mini_splatting/360v2/train_${arg}_${i}.log" 2>&1
          # you need to create the log folder first
          # Increment the port number for the next run
          ((port++))
          # Allow some time for the process to initialize and potentially use GPU memory
          sleep 60
          break
        else
          echo "No GPU available at the moment. Retrying in 1 minute."
          sleep 60
        fi
        # done
        
      done  # End loop over v_pow values
    done
  done
done

# Wait for all background processes to finish
wait
echo "All runs completed."
