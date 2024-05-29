#!/bin/zsh

# Function to get the id of an available GPU
# get_available_gpu() {
#   local mem_threshold=5000
#   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
#   $2 < threshold { print $1; exit }
#   '
# }

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

# prune percentage for the first prune
declare -a prune_percents=(0.6) 


# Check that prune_percents and prune_decays arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ]; then
  echo "The number of prune percents does not match the number of prune decays."
  exit 1
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do
  for i in "${(@i)prune_percents}"; do
    prune_percent="${prune_percents: i}"
    prune_decay="${prune_decays: i}"
      # Wait for an available GPU
      while true; do
        gpu_id=1
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting train.py lp-3dgs with dataset '$arg', prune_type 'mini_splatting', prune_decay on port $port"
          CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
            -s "path_to_dataset/$arg" \
            -m "output/mini_splatting/${arg}_mask" \
            --prune_method "mini_splatting" \
            --eval \
            --use_importance_mask \
            --wandb_run_name "$arg"\
            --port $port > "logs/mini_splatting/train_${arg}_mask.log" 2>&1 
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

# Wait for all background processes to finish
wait
echo "All runs completed."
