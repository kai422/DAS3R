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
# from lietorch import SO3, SE3, Sim3, LieGroupParameter
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_tensor_from_camera
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import rotation2quad, get_tensor_from_camera, depth_to_pts3d

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.enable_test = True

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.Q,
            self.T,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        self.Q, self.T) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def compute_relative_world_to_camera(self, R1, t1, R2, t2):
        # Create a row of zeros with a one at the end, for homogeneous coordinates
        zero_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

        # Compute the inverse of the first extrinsic matrix
        E1_inv = np.hstack([R1.T, -R1.T @ t1.reshape(-1, 1)])  # Transpose and reshape for correct dimensions
        E1_inv = np.vstack([E1_inv, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the second extrinsic matrix
        E2 = np.hstack([R2, -R2 @ t2.reshape(-1, 1)])  # No need to transpose R2
        E2 = np.vstack([E2, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the relative transformation
        E_rel = E2 @ E1_inv

        return E_rel

    def init_test_RT_seq(self, cam_list):
        if len(cam_list[1.0]) == 0:
            self.enable_test = False
            return 
        quats =[]
        trans = []
        for cam in cam_list[1.0]:
            pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            quat = pose[:4]
            tran = pose[4:]
            quats.append(quat)
            trans.append(tran)
        quats = torch.stack(quats)
        trans = torch.stack(trans)
        self.test_Q = quats.cuda().requires_grad_(True)
        self.test_T = trans.cuda().requires_grad_(True)

    def init_RT_seq(self, cam_list):
        quats =[]
        trans = []
        for cam in cam_list[1.0]:
            pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            quat = pose[:4]
            tran = pose[4:]
            quats.append(quat)
            trans.append(tran)
        quats = torch.stack(quats)
        trans = torch.stack(trans)
        self.Q = quats.cuda().requires_grad_(True)
        self.T = trans.cuda().requires_grad_(True)

    def init_fov(self, cam_list):
        cam = cam_list[1.0][0]
        self.FoVx = torch.tensor(cam.FoVx).cuda().requires_grad_(True)
        self.FoVy = torch.tensor(cam.FoVy).cuda().requires_grad_(True)



    def get_RT(self, idx):
        quat = self.Q[idx]
        tran = self.T[idx]
        pose = torch.cat((quat, tran), dim=0)
        return pose

    def get_P(self):
        pose = torch.cat((self.Q, self.T), dim=1)
        return pose
    
    def get_RT_test(self, idx):
        quat = self.test_Q[idx]
        tran = self.test_T[idx]
        pose = torch.cat((quat, tran), dim=0)
        return pose

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        conf_lr_init = 3e-3
        conf_lr_final = 3e-4

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._conf_static], 'lr': conf_lr_init, "name": "conf_static"},
        ]

        cam_lr_init_Q = 0.00003
        cam_lr_final_Q = 0.000003
        cam_lr_init_T = 0.00003
        cam_lr_final_T = 0.000003
        l_cam = [
                {'params': [self.Q],'lr': cam_lr_init_Q, "name": "pose_Q"},
                {'params': [self.T],'lr': cam_lr_init_T, "name": "pose_T"},
                {'params': [self.FoVx],'lr': 0.0001, "name": "fovX"},
                {'params': [self.FoVy],'lr': 0.0001, "name": "fovY"}
        ]
        # l_cam = [{'params': [self.P],'lr': training_args.rotation_lr, "name": "pose"},]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.optimizer_cam = torch.optim.Adam(l_cam, lr=0.0, eps=1e-15)

        if self.enable_test:
            l_cam_test = [
                    {'params': [self.test_Q],'lr': cam_lr_init_Q, "name": "test_pose_Q"},
                    {'params': [self.test_T],'lr': cam_lr_init_T, "name": "test_pose_T"},
            ]
            self.optimizer_cam_test = torch.optim.Adam(l_cam_test, lr=0.0, eps=1e-15)


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.Q_scheduler_args = get_expon_lr_func(
                                                    # lr_init=0,
                                                    # lr_final=0,
                                                    lr_init=cam_lr_init_Q,
                                                    lr_final=cam_lr_final_Q,
                                                    # lr_init=training_args.position_lr_init*self.spatial_lr_scale*10,
                                                    # lr_final=training_args.position_lr_final*self.spatial_lr_scale*10,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)
        
        self.T_scheduler_args = get_expon_lr_func(
                                                    # lr_init=0,
                                                    # lr_final=0,
                                                    lr_init=cam_lr_init_T,
                                                    lr_final=cam_lr_final_T,
                                                    # lr_init=training_args.position_lr_init*self.spatial_lr_scale*10,
                                                    # lr_final=training_args.position_lr_final*self.spatial_lr_scale*10,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)
        
        self.conf_static_scheduler_args = get_expon_lr_func(
                                                    lr_init=conf_lr_init,
                                                    lr_final=conf_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer_cam.param_groups:
            if param_group["name"] == "pose_Q":
                lr = self.Q_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "pose_T":
                lr = self.T_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "test_pose_Q":
                lr = self.Q_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "test_pose_T":
                lr = self.T_scheduler_args(iteration)
                param_group['lr'] = lr

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "conf_static" or param_group["name"] == "conf":
                lr = self.conf_static_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('conf_static')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        conf_static = self._conf_static.reshape(-1, 1)[self.aggregated_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, conf_static, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        conf_static = np.asarray(plydata.elements[0]["conf_static"])[..., np.newaxis]


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._conf_static = nn.Parameter(torch.tensor(conf_static, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # breakpoint()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    

    def create_from_cameras(self, train_cameras, spatial_lr_scale : float, conf_thre = 1.0):
        self.spatial_lr_scale = spatial_lr_scale
        poses = []
        confidences = []
        dynamic_masks = []
        dyna_avg = []
        rgbs = []
        depth_maps = []
        K = []
        camera0 = train_cameras[1.0][0]
        W = camera0.image_width
        H = camera0.image_height

        for camera in train_cameras[1.0]:
            camera.uid
            intr = camera.intr
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            height = intr.height
            width = intr.width   
            intr = torch.tensor([[focal_length_x, 0, width / 2],
                              [0, focal_length_y, height / 2],
                              [0, 0, 1]], device="cuda")
            K.append(intr)
            poses.append(camera.original_pose)
            depth_maps.append(camera.depth_map)
            confidences.append(camera.conf_map)
            dynamic_masks.append(camera.dynamic_mask)
            dyna_avg.append(camera.dyna_avg_map)
            rgbs.append(camera.original_image)

        K = torch.stack(K)
        rgbs = torch.stack(rgbs)
        depth_maps = torch.stack(depth_maps)
        confidences = torch.stack(confidences)
        dynamic_masks = torch.stack(dynamic_masks)
        dyna_avg = torch.stack(dyna_avg)
        poses = torch.stack(poses).cuda()




        p3d = depth_to_pts3d(K, poses, W, H, depth_maps).float()
        p3d_color = rgbs.permute(0,2,3,1).reshape(-1, 3)
        pts_4_3dgs = p3d.reshape(-1, 3)
        
        dyna = dyna_avg
        conf_static = 1 - torch.tensor(dyna)
        # confidences = conf_static * confidences
        confidence = torch.tensor(confidences).reshape(-1)

        confidence_masks = confidence > torch.tensor(conf_thre).log()
        print(f"Ratio of confidence masks: {confidence_masks.float().mean().item():.4f}")
        self.aggregated_mask = confidence_masks
        print(f"Ratio of aggreagted masks: {self.aggregated_mask.float().mean().item():.4f}")
        print(f"Number of points before: {pts_4_3dgs.shape[0]}")
        pts_4_3dgs = pts_4_3dgs[self.aggregated_mask]
        color_4_3dgs = p3d_color.reshape(-1, 3)[self.aggregated_mask]


        fused_point_cloud = pts_4_3dgs
        fused_color = RGB2SH(color_4_3dgs)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(pts_4_3dgs), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opa = 1/len(train_cameras[1.0])
        opacities = inverse_sigmoid(opa * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = inverse_sigmoid(conf_static_4_3dgs.reshape(-1, 1))


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._conf_static = nn.Parameter(conf_static.requires_grad_(True))
