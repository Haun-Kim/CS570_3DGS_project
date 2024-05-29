#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from scene.gaussian_model import GaussianModel
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from mask import PruneMask
import gc

class MiniSplatting(GaussianModel):

    def __init__(self, sh_degree : int):
        self.importance_mask = PruneMask()
        super().__init__(sh_degree)
    
    def opacity_with_mask_activation(self, opacity):
        return (torch.sigmoid(opacity).squeeze() * self.importance_mask.get_mask).unsqueeze(1)
        # return (torch.sigmoid(opacity).squeeze() * self.importance_mask.get_ste_mask).unsqueeze(1)
    
    def get_score_before_render(self, opt, scene=None, pipe=None, background=None, render=None):
        self.importance_mask.important_score = self.prune_list(scene, pipe, background, render)
        self.opacity_activation = self.opacity_with_mask_activation

    # Use or not
    def oneupSHdegree(self, opt, iteration):
        if iteration > opt.densify_until_iter and self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def sample_gaussians(self, sample_ratio, imp_list):
        probability_list = imp_list / imp_list.sum()
        target_samples = int(self._xyz.shape[0] * sample_ratio)
        valid_samples = torch.count_nonzero(probability_list)
        target_samples = target_samples if target_samples <= valid_samples else valid_samples
        sample_idx = torch.multinomial(probability_list, target_samples)
        prune_mask = torch.zeros(self._xyz.shape[0], device="cuda").scatter_(0, sample_idx, 1.).bool()
        self.prune_points(~prune_mask)
        
        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, imp_list):
        sorted_tensor, _ = torch.sort(imp_list, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (imp_list <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_after_render(self, opt, iteration, scene=None, pipe=None, background=None, render=None, use_importance_mask=False):
        if iteration == opt.prune_use_percent_iterations and use_importance_mask == False:
            # sampling
            imp_list = self.prune_list(scene, pipe, background, render)
            preserving_ratio = 1 - opt.prune_percent
            self.sample_gaussians(preserving_ratio, imp_list)
        if use_importance_mask:
            if iteration == opt.prune_iterations[0]:
                self.set_mask(opt)
            if iteration == opt.prune_iterations[0]+opt.train_mask_iters:
                prune_mask = (self.importance_mask.get_prune_mask < 0.5).squeeze()
                self.prune_points(prune_mask)
                torch.save(self.importance_mask._mask, "_mask.pt")
                torch.save(self.importance_mask.important_score, "important_score.pt")
                self.opacity_activation = torch.sigmoid
        
        # if iteration == opt.simplification_iteration2:
        #     # direct pruning
        #     imp_list = self.prune_list(scene, pipe, background, render)
        #     self.prune_gaussians(1-opt.preserving_ratio, imp_list)
    
    def addtional_loss(self, opt):
        # return 1e-6 * torch.norm(self.importance_mask.get_prune_mask)
        return 0.0005 * torch.mean((torch.sigmoid(self.importance_mask._mask)))

    @torch.no_grad()
    def prune_list(self, scene, pipe, background, count_render):
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop()
        imp_list = count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]

        for _ in range(len(viewpoint_stack)):
            viewpoint_cam = viewpoint_stack.pop()
            imp_list += count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]
            gc.collect()

        return imp_list

    def set_mask(self, opt):
        self.importance_mask.set_mask(self._opacity.shape, opt)
