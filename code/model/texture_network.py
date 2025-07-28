import torch
from model.embedder import *
import torch.nn as nn


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            condition_in=53,
            bottleneck=8,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + bottleneck + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bottleneck = bottleneck
        if bottleneck != 0:
            self.condition_bottleneck = nn.Linear(condition_in, bottleneck)

    def forward(self, points, normals, feature_vectors, jaw_pose=None):
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)

        if self.embedpnts_fn is not None:
            points = self.embedpnts_fn(points) 
        
        if self.bottleneck != 0:
            num_pixels = int(points.shape[0] / jaw_pose.shape[0])
            jaw_pose = jaw_pose.unsqueeze(1).expand(-1, num_pixels, -1).reshape(points.shape[0], -1)
            jaw_pose = self.condition_bottleneck(jaw_pose)
            rendering_input = torch.cat([points, normals, jaw_pose, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)

        return x
    
class RenderingNetwork_Hand(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            dims,
            condition_in=53,
            bottleneck=8,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + bottleneck] + dims + [d_out]

        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bottleneck = bottleneck
        if bottleneck != 0:
            self.condition_bottleneck = nn.Linear(condition_in, bottleneck)

    def forward(self, points, normals, jaw_pose=None):
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)

        if self.embedpnts_fn is not None:
            points = self.embedpnts_fn(points) 
        
        if self.bottleneck != 0:
            num_pixels = int(points.shape[0] / jaw_pose.shape[0])
            jaw_pose = jaw_pose.unsqueeze(1).expand(-1, num_pixels, -1).reshape(points.shape[0], -1)
            jaw_pose = self.condition_bottleneck(jaw_pose)
            rendering_input = torch.cat([points, normals, jaw_pose], dim=-1)
        else:
            rendering_input = torch.cat([points, normals], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)

        return x
    
# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

from .fc import FC
from .embedder import get_embedder, generate_ide_fn
import numpy as np
import torch
import tinycudann as tcnn
# import nvdiffrec.render.renderutils.ops as ru
# import nvdiffrast.torch as dr


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

class NeuralShader(torch.nn.Module):

    def __init__(self,
                 activation='relu',
                 last_activation=None,
                 fourier_features='positional',
                 disentangle_network_params=None,
                 aabb=None,
                 device='cpu'):

        super().__init__()
        self.device = device
        self.aabb = aabb
        # ==============================================================================================
        # PE
        # ==============================================================================================
        self.fourier_feature_transform, channels = get_embedder(4)
        self.inp_size = channels
        # ==============================================================================================
        # create MLP
        # ==============================================================================================
        # self.material_mlp_ch = disentangle_network_params['material_mlp_ch']
        # self.material_mlp = FC(self.inp_size, self.material_mlp_ch, disentangle_network_params["material_mlp_dims"], activation, last_activation).to(device) #sigmoid
        
        # self.light_mlp = FC(38, 3, disentangle_network_params["light_mlp_dims"], activation=activation, last_activation=None, bias=True).to(device) 
        self.dir_enc_func = generate_ide_fn(deg_view=4, device=self.device)
        self.dir_enc_func_normals = generate_ide_fn(deg_view=4, device=self.device)
        
        self.color_mlp = FC(self.inp_size + 38, 3, disentangle_network_params["color_mlp_dims"], activation=activation, last_activation=None, bias=True).to(device) 
        
        print(disentangle_network_params)

        # Store the config
        self._config = {
            "activation":activation,
            "last_activation":last_activation,
            "fourier_features":fourier_features,
            "disentangle_network_params":disentangle_network_params,
            "aabb":aabb,
        }

    # def forward(self, position, normal_bend, skin_mask=None):
    #     bz, h, w, ch = position.shape
    #     pe_input = self.apply_pe(position=position)

    #     # view_dir = view_direction[:, None, None, :]
    #     # normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

    #     # ==============================================================================================
    #     # Albedo ; roughness; specular intensity 
    #     # ==============================================================================================   
    #     all_tex = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32)) 
    #     kd = all_tex[..., :3].view(bz, h, w, ch) 

    #     # ========= diffuse shading ===========
    #     kr_max = torch.ones((bz, h, w, 1))
    #     kr_max = kr_max.to(self.device)                    
    #     enc_nd_kr_max = self.dir_enc_func(normal_bend.view(-1, 3), kr_max.view(-1, 1))
    #     shading = self.light_mlp(enc_nd_kr_max)
    #     shading = shading.view(bz, h, w, 3) 
        
    #     shaded_color = shading * kd
        
    #     buffers = {
    #                 'shading': shaded_color,
    #                 # 'normal_bend': normal_bend
    #                 }

    #     return shaded_color, kd, buffers
    
    def forward(self, position, normal_bend, skin_mask=None):
        bz, h, w, ch = position.shape
        pe_input = self.apply_pe(position=position)

        # view_dir = view_direction[:, None, None, :]
        # normal_bend = self.get_shading_normals(deformed_position, view_dir, gbuffer, mesh)

        # ==============================================================================================
        # Albedo ; roughness; specular intensity 
        # ==============================================================================================   
        # all_tex = self.material_mlp(pe_input.view(-1, self.inp_size).to(torch.float32)) 
        # kd = all_tex[..., :3].view(bz, h, w, ch) 

        # ========= diffuse shading ===========
        kr_max = torch.ones((bz, h, w, 1))
        kr_max = kr_max.to(self.device)                    
        enc_nd_kr_max = self.dir_enc_func(normal_bend.view(-1, 3), kr_max.view(-1, 1))
        # shading = self.light_mlp(enc_nd_kr_max)
        # shading = shading.view(bz, h, w, 3) 
        
        input = torch.cat([pe_input.view(-1, self.inp_size).to(torch.float32), enc_nd_kr_max], dim=-1)
        color = self.color_mlp(input)
        color = color.view(bz, h, w, 3) 

        return color

    # # ==============================================================================================
    # # prepare the final color output
    # # ==============================================================================================    
    # def shade(self, gbuffer, cameras, mesh, lgt):

    #     positions = gbuffer["canonical_position"]
    #     batch_size, H, W, ch = positions.shape

    #     # view_direction = torch.cat([v.center.unsqueeze(0) for v in views['camera']], dim=0)
    #     view_direction = torch.cat([v.center.unsqueeze(0) for v in cameras], dim=0)

    #     skin_mask_bool = None

    #     ### compute the final color, and c-buffers 
    #     pred_color, albedo, buffers = self.forward(positions, gbuffer, view_direction, mesh, light=lgt,
    #                                         deformed_position=gbuffer["position"], skin_mask=skin_mask_bool)
    #     pred_color = pred_color.view(positions.shape) 
    #     albedo = albedo.view(positions.shape)
    #     normal = buffers['normal_bend'].view(positions.shape)

    #     ### !! we mask directly with alpha values from the rasterizer !! ###
    #     pred_color_masked = torch.lerp(torch.zeros((batch_size, H, W, 4)).to(self.device), 
    #                                 torch.concat([pred_color, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask"].float())
    #     albedo_masked = torch.lerp(torch.zeros((batch_size, H, W, 4)).to(self.device), 
    #                                 torch.concat([albedo, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask"].float())
    #     pred_normal_masked = torch.lerp(torch.zeros((batch_size, H, W, 4)).to(self.device), 
    #                                 torch.concat([normal, torch.ones_like(pred_color[..., 0:1]).to(self.device)], axis=3), gbuffer["mask"].float())
    
    #     ### we antialias the final color here (!)
    #     pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
    #     pred_normal_masked = dr.antialias(pred_normal_masked.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())

    #     buffers["albedo"] = albedo_masked[..., :3]
    #     buffers["pred_normal_masked"] = pred_normal_masked[..., :3]
    #     return pred_color_masked[..., :3], buffers, pred_color_masked[..., -1:]

    # # ==============================================================================================
    # # misc functions
    # # ==============================================================================================
    # def get_shading_normals(self, position, view_dir, gbuffer, mesh):
    #     ''' flip the backward facing normals
    #     '''
    #     normal = ru.prepare_shading_normal(position, view_dir, None, 
    #                                        gbuffer["vertex_normals"], gbuffer["tangent_normals"], gbuffer["face_normals"], two_sided_shading=True, opengl=True, use_python=False)
    #     gbuffer["normal"] =  dr.antialias(normal.contiguous(), gbuffer["rast"], gbuffer["deformed_verts_clip_space"], mesh.indices.int())
    #     return gbuffer["normal"]
    
    def apply_pe(self, position):
        ## normalize PE input 
        # position = (position.view(-1, 3) - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
        # position = torch.clamp(position, min=0, max=1)
        pe_input = self.fourier_feature_transform(position.contiguous()).to(torch.float32)
        return pe_input

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)

        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'], strict=False)

        return shader

    def save(self, path):
        data = {
            'version': 2,
            'config': self._config,
            'state_dict': self.state_dict()
        }

        torch.save(data, path)