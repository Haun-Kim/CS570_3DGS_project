# LP-3DGS: Learning to Prune 3D Gaussian Splatting


# Overview

This work aims to lead the 3D Gaussian Splatting model to learn the best model size on the given scene. We add a trainable mask on the importance score defined by previous work and make the model find the best prune ratio, avoiding testing the prune ratio for many runs. The details could be found in the paper.

# Running

## Environment
You can use conda to create the environment.
```
conda env create --file environment.yml
conda activate lp_3dgs
```

## Train
File "train.py" including training the model, rendering the images and doing the evaluation. Most of the parameters are the same as [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The parameters should be noticed are
#### --source_path / -s
Path to the source directory containing a COLMAP or Synthetic NeRF data set.
#### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
#### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
#### --prune_method
  The prune method to use, could be "3dgs", "rad_splat", "mint_spaltting", "compact_3dgs". Default is "3dgs", in this case, the model will not be pruned
#### --prune_percent
  The prune percent you want to be, only applicable to "rad_splat" and "mini_splatting"
#### --use_importance_mask
  Whether to use a mask on the importance score, store true
#### --prune_use_percent_iterations
  The iteration to prune the model using prune percent, only effective when use_importance_mask is false
#### --prune_iterations
  The iteration starts to train the trainable mask, only effective when use_importance_mask is true
#### --train_mask_iters
  The total iterations to train the mask, default is 500
#### --gumbel_temp
  The scaling parameter of gumbel sigmoid function, default is 0.5

Beside the python codes, we aslo provides zsh scripts in "scripts" folder, Two of the scripts are used to train lp-3dgs using score of RadSplat and Mini-Splatting, the rest two are used to sweep the prune percent parameter

render.py and metrics.py could aslo be used to render and evaluate seperately, see [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for details.
