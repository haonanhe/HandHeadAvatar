import torch
import torch.nn as nn
import numpy as np
import math
from utils import rend_util
from utils import general as utils
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from flame.lbs import inverse_skinning_pts
from flame.FLAME import FLAME
from pytorch3d import ops

class RigidDeformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 condition_in,
                 dims,
                 multires,
                 geometric_init=True,
                 weight_norm=True):
        super().__init__()

    def forward(self, points, deformer_condition, transformations):
        # deformer_condition is not used, for compatibility
        lbs_weights = torch.zeros_like(points)
        lbs_weights[:, 1] = 1.0
        pts_c = inverse_skinning_pts(points, transformations, lbs_weights, dtype=torch.float32)
        others = {}
        return pts_c, others


class BackwardDeformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 condition_in,
                 dims,
                 multires,
                 geometric_init=True,
                 weight_norm=True):
        super().__init__()
        dims = [d_in + condition_in] + dims + [d_out]
        self.condition_in = condition_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.constant_(lin.weight, 0)
                torch.nn.init.constant_(lin.bias, 0)
            elif multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, points, conditions, transformations):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.condition_in != 0:
            num_pixels = int(points.shape[0] / conditions.shape[0])
            conditions = conditions.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.condition_in)
            x = torch.cat([points, conditions], dim=1)
        else:
            x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        lbs_weights = torch.nn.functional.softmax(20 * x, dim=1)
        pts_c = inverse_skinning_pts(points, transformations, lbs_weights, dtype=torch.float32)
        others = {'lbs_weight': lbs_weights}
        return pts_c, others

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            condition_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in + condition_in] + dims + [d_out + feature_vector_size]
        self.condition_in = condition_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + condition_in

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, condition):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        if self.condition_in > 0:
            num_pixels = int(input.shape[0] / condition.shape[0])
            condition = condition.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.condition_in)
            input = torch.cat([input, condition], dim=1)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x, condition):
        x.requires_grad_(True)
        y = self.forward(x, condition)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0 and mode in ['normal']:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        pi = math.pi
        if mode == 'spherical_harmonics':
            self.SH_lighting = nn.Parameter(torch.Tensor([np.sqrt(4 * np.pi), 0., 0., 0., 0., 0., 0., 0., 0.]))
            self.constant_factor = torch.tensor(
                [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                 ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
                 ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                 (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
                 (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                 (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float().cuda()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def add_SHlight(self, normal, sh_coeff):
        '''
            normal: [bz, 3]
            sh_coeff: [9, 1]
        '''
        N = normal
        # assume white illumination
        sh_coeff = sh_coeff.unsqueeze(-1).expand(-1, 3)
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1], dim=1)  # [bz, 9]
        sh = sh * self.constant_factor[None, :]  # [bz, 9]
        shading = torch.sum(sh_coeff[None, :, :] * sh[:, :, None], 1)  # [1, 9, 3] * [bz, 9, 1] -> [bz, 3]
        return shading

    def forward(self, points, normals, feature_vectors):
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)

        if self.mode == 'normal':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)

        elif self.mode == 'spherical_harmonics':
            rendering_input = torch.cat([points, feature_vectors], dim=-1)

        else:
            raise Exception('Rendering Mode is not implemented')

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        if self.mode == 'spherical_harmonics':
            shading = self.add_SHlight(normals, self.SH_lighting)
            color = (x + 1.) / 2.
            color = color * shading
            color = color * 2. - 1.
            others = {'shading': shading, 'albedo': x}
            return color, others

        others = {}
        return x, others

class IDRNetwork(nn.Module):
    def __init__(self, conf, num_training_frames, shape_params):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.use_latent = conf.get_bool('use_latent')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.deformer_network = utils.get_class(conf.get_string('deformer_class'))(**conf.get_config('deformer_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', n_shape=100, n_exp=50, shape_params=shape_params).cuda()
        if self.use_latent:
            self.latent_codes = nn.Embedding(num_training_frames, 32)
            torch.nn.init.uniform_(
                self.latent_codes.weight.data,
                0.0,
                1.0,
            )

    def query_sdf(self, pnts_p, network_condition, deformer_condition, transformations):
        pnts_c, others = self.deformer_network(pnts_p, deformer_condition, transformations)
        output = self.implicit_network(pnts_c, network_condition)
        return output[:, 0], pnts_c, output[:, 1:], others

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        object_mask = input["object_mask"].reshape(-1)
        if self.use_latent:
            latent_code = self.latent_codes(input["idx"].cuda()).squeeze(1)
            network_condition = torch.cat([expression, latent_code], dim=1)
            deformer_condition = flame_pose[:, 3:9]
        else:
            network_condition = deformer_condition = None
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)
        self.implicit_network.eval()
        self.deformer_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.query_sdf(pnts_p=x,
                                                                                              network_condition=network_condition,
                                                                                              deformer_condition=deformer_condition,
                                                                                              transformations=transformations)[0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()
        self.deformer_network.train()

        # TODO check the following line's effect
        # commenting this line should make no difference if not training cameras
        # points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output, _, _, others = self.query_sdf(points, network_condition, deformer_condition, transformations)
        sdf_output = sdf_output.unsqueeze(1)
        if 'lbs_weight' in others:
            lbs_weight = others['lbs_weight']
            surface_mask = network_object_mask & object_mask if self.training else network_object_mask
            _, index_batch, _ = ops.knn_points(points[surface_mask].unsqueeze(0), verts, K=1, return_nn=True)
            index_batch = index_batch[0, :, 0]
            gt_lbs_weight = self.FLAMEServer.lbs_weights[index_batch, :]
            gt_lbs_weight = gt_lbs_weight[:, :3]

        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            surface_sdf_values = surface_output.detach()

            grad_theta = self.implicit_network.gradient(eikonal_points, network_condition).squeeze(1)
            surface_points_grad = self.gradient(surface_points, network_condition, deformer_condition, transformations, create_graph=False, retain_graph=True).clone().detach()

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.zeros_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], others = self.get_rbg_value(differentiable_surface_points, network_condition, deformer_condition, transformations, is_training=self.training)
            normal_values[surface_mask] = others['normals']

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }
        if 'lbs_weight' in others:
            skinning_values = torch.ones_like(points).float().cuda()
            gt_skinning_values = torch.ones_like(points).float().cuda()
            skinning_values[surface_mask] = lbs_weight[surface_mask]
            gt_skinning_values[surface_mask] = gt_lbs_weight
            output['lbs_weight'] = skinning_values
            output['gt_lbs_weight'] = gt_skinning_values

        if 'shading' in others:
            shading_values = torch.ones_like(points).float().cuda()
            alebdo_values = torch.ones_like(points).float().cuda()
            shading_values[surface_mask] = others['shading']
            alebdo_values[surface_mask] = others['albedo']
            output['shading_values'] = shading_values
            output['albedo_values'] = alebdo_values
        return output

    def get_rbg_value(self, points, network_condition, deformer_condition, transformations, is_training=True):
        points.requires_grad_(True)
        sdf, pnts_c, feature_vectors, others = self.query_sdf(points, network_condition, deformer_condition, transformations)
        gradients = self.gradient(points, network_condition, deformer_condition, transformations, sdf=sdf, create_graph=is_training, retain_graph=is_training)

        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)

        rgb_vals, rendering_others = self.rendering_network(pnts_c, normals, feature_vectors)

        others['normals'] = normals
        for k, v in rendering_others.items():
            others[k] = v
        return rgb_vals, others

    def gradient(self, x, network_condition, deformer_condition, transformations, sdf=None, create_graph=True, retain_graph=True):
        x.requires_grad_(True)
        if sdf is None:
            y = self.query_sdf(x, network_condition, deformer_condition, transformations)[0]
        else:
            y = sdf
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]
        return gradients