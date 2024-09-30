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

class RadSplat(GaussianModel):

    def __init__(self, sh_degree : int):
        self.importance_mask = PruneMask()
        super().__init__(sh_degree)

    def opacity_with_mask_activation(self, opacity):
        return (torch.sigmoid(opacity).squeeze() * self.importance_mask.get_mask).unsqueeze(1)
        # return (torch.sigmoid(opacity).squeeze() * self.importance_mask.get_ste_mask).unsqueeze(1)

    def get_score_before_render(self, opt, scene=None, pipe=None, background=None, render=None):
        self.importance_mask.important_score = self.prune_list(scene, pipe, background, render)
        self.opacity_activation = self.opacity_with_mask_activation

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def prune_after_render(self, opt, iteration, scene=None, pipe=None, background=None, render=None, use_importance_mask=False):
        if iteration in opt.prune_use_percent_iterations and use_importance_mask == False:
            imp_list = self.prune_list(scene, pipe, background, render)
            # prune_mask = imp_list < opt.rad_prune_threshold
            # self.prune_points(prune_mask)
            self.prune_gaussians(opt.prune_percent, imp_list)
        if use_importance_mask:
            if iteration == opt.prune_iterations[0]:
                self.set_mask(opt)
            if iteration == opt.prune_iterations[0]+opt.train_mask_iters:
                prune_mask = (self.importance_mask.get_prune_mask < 0.5).squeeze()
                # prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= 0.01).squeeze(),
                #                               (self.get_opacity < min_opacity).squeeze())
                self.prune_points(prune_mask)
                torch.save(self.importance_mask._mask, "_maskrad.pt")
                torch.save(self.importance_mask.important_score, "important_scorerad.pt")
                self.opacity_activation = torch.sigmoid


    def addtional_loss(self, opt):
        # return 0.0005 * torch.mean(self.importance_mask.get_prune_mask)
        return 0.0005 * torch.mean((torch.sigmoid(self.importance_mask._mask)))

    @torch.no_grad()
    def prune_list(self, scene, pipe, background, count_render):
        viewpoint_stack = scene.getTrainCameras().copy()
        imp_list = torch.zeros(self._xyz.shape[0], device="cuda")

        for viewpoint_cam in viewpoint_stack:
            imp_list = torch.maximum(imp_list, count_render(viewpoint_cam, self, pipe, background, mw_score=True)["important_score"])
            gc.collect()

        return imp_list

    def set_mask(self, opt):
        self.importance_mask.set_mask(self._opacity.shape, opt)