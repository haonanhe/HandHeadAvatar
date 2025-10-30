import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pickle

from trimesh.voxel.morphology import surface
from utils import rend_util
from utils import general as utils
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from flame.FLAME_metahuman_pca import FLAME
from pytorch3d import ops
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from model.geometry_network import GeometryNetwork
from model.texture_network import RenderingNetwork

from model.monosdf_loss import compute_scale_and_shift

from model.texture_network import NeuralShader

from external.body_models import MANOLayer
from pysdf import SDF
from kaolin.metrics.trianglemesh import point_to_mesh_distance

def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1 # this actually also change points_cam_projected

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    # (points2d[:,:,-1] == points_cam[:,:,-1]) = True
    return points2d

def read_mano_uv_obj(filename):
    vt, ft, f = [], [], []
    for content in open(filename):
        if content.startswith('#'):
            continue
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
            f.append([int(a.split('/')[0]) for a in contents[1:] if a])
    
    #NOTE: the obj file is 1-indexed, thus we need to minus 1
    vt, ft, f = np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1, np.array(f, dtype=np.longlong) - 1     #Neural Actor encoder.py line 1241
    
    #invert the v coordinate
    vt[:, 1] = 1 - vt[:, 1]
    
    return vt, ft, f

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights

def get_normal(points, verts, faces):
    points = points.reshape(1, -1, 3)
    verts = verts.reshape(1, -1, 3)
    faces = faces.reshape(1, -1, 3)

    normals = Meshes(verts, faces).verts_normals_padded()

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points.contiguous(), triangles)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_normals = torch.gather(
        normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)

    pts_norm = (closest_normals*bary_weights[:, :, None]).sum(
        1).unsqueeze(0) * torch.tensor([1.0, 1.0, 1.0]).type_as(normals)
    pts_norm = F.normalize(pts_norm, dim=2)

    return pts_norm.view(-1, 3)


class HHAvatar(nn.Module):
    def __init__(self, conf, shape_params, gt_w_seg):
        super().__init__()
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', n_shape=100, n_exp=50,
                                 shape_params=shape_params).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp,
                             full_pose=self.FLAMEServer.canonical_pose,
                             scale=4)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        self.feature_vector_size = conf.get_int('feature_vector_size')
       
        self.geometry_network = GeometryNetwork(self.feature_vector_size, **conf.get_config('geometry_network'))
        self.deformer_class = conf.get_string('deformer_class').split('.')[-1]
        self.deformer_network = utils.get_class(conf.get_string('deformer_class'))(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))

        self.nonrigid_deformer = utils.get_class(conf.get_string('nonrigid_class'))(**conf.get_config('nonrigid_deformer'))

        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.distance = 1.0
        
        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.gt_w_seg = gt_w_seg
        
        self.device = torch.device('cuda')
        
        self.MANOServer = MANOLayer(model_path="../code/mano_model/data/mano",
                                    is_rhand=True,
                                    batch_size=1,
                                    flat_hand_mean=False,
                                    dtype=torch.float32,
                                    use_pca=False,)
        vt, ft, f = read_mano_uv_obj('../code/mano_model/data/MANO_UV_right.obj')
        self.mano_faces = torch.tensor(f).to(self.device)                               #NOTE: this is same as mano_layer[hand_type].faces
        self.mano_face_uv = torch.tensor(vt[ft], dtype=torch.float32).to(self.device)  
        
        with open("../code/mano_model/data/contact_zones.pkl", "rb") as f:
            contact_zones = pickle.load(f)
        contact_zones = contact_zones["contact_zones"]
        contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])
        contact_idx = torch.from_numpy(contact_idx).to(self.device)
        # self.contact_idx = contact_idx[19:] #fingers
        self.contact_idx = contact_idx[19:47] #index_fingers

        import sys
        sys.path.append('../preprocess/submodules/DECA')
        from decalib.deca import DECA
        from decalib.utils import util
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=self.device, image_size=256, uv_size=256)
        # self.deca = DECA(config=deca_cfg, device=self.device, image_size=512, uv_size=512)
        # self.deca = DECA(config=deca_cfg, device=device, image_size=224, uv_size=224)

        color_mlp_dims = [128, 128, 128, 128]
        disentangle_network_params = {
            'color_mlp_dims': color_mlp_dims
        }
        # Create the optimizer for the neural shader
        self.shader = NeuralShader(fourier_features='positional',
                            activation='relu',
                            last_activation=torch.nn.Sigmoid(), 
                            disentangle_network_params=disentangle_network_params,
                            aabb=None,
                            device=self.device)
        
    def query_sdf(self, pnts_p, network_condition, deformer_condition, pose_feature, betas, transformations, transl=None, nonrigid_params=None):
        
        pnts_c, others = self.deformer_network(pnts_p, deformer_condition, pose_feature, betas, transformations, transl)
        
        nonrigid_output = self.nonrigid_deformer(pnts_p, transl=transl, nonrigid_params=nonrigid_params)
        nonrigid_deformation = nonrigid_output['nonrigid_deformation']
        nonrigid_dir = nonrigid_output['nonrigid_dir']
        
        pnts_c = pnts_c - nonrigid_deformation

        others['nonrigid_deformation'] = nonrigid_deformation
        others['nonrigid_dir'] = nonrigid_dir
        
        output = self.geometry_network(pnts_c, network_condition)
        sdf = output[:, 0]
        feature = output[:, 1:]
        return sdf, pnts_c, feature, others

    def query_sdf_hand(self, pnts_p, verts=None):
        
        def pysdf_func(pnts_p, verts=verts):
            verts = verts.reshape(-1, 3)
            pnts_p = pnts_p.reshape(-1, 3)
            f = SDF(verts.squeeze(0).detach().cpu().numpy(), self.MANOServer.faces.astype(np.int64))
            mano_sdf = f(pnts_p.detach().cpu().numpy())
            mano_sdf = torch.from_numpy(mano_sdf).to(pnts_p.device)#.unsqueeze(-1)
            mano_sdf = mano_sdf * -1
            return mano_sdf

        mano_sdf = pysdf_func(pnts_p, verts=verts)
        mano_normal = get_normal(pnts_p, verts.squeeze(0), self.mano_faces)
        
        sdf = mano_sdf.reshape(-1,)
        pnts_c = pnts_p.clone().reshape(-1, 3)
        feature = None
        others = {'mano_normal': mano_normal.reshape(-1, 3)}
        
        return sdf, pnts_c, feature, others
    
    def query_sdf_flame_bbox(self, network_condition):
        
        min_coords = torch.min(self.FLAMEServer.canonical_verts.reshape(-1, 3), dim=0)[0]
        max_coords = torch.max(self.FLAMEServer.canonical_verts.reshape(-1, 3), dim=0)[0]
        
        n_points = 1000
        points = min_coords + (max_coords - min_coords) * torch.rand(n_points, 3).cuda()
        points = points.reshape(-1, 3)
        
        f = SDF(self.FLAMEServer.canonical_verts.reshape(-1, 3).detach().cpu().numpy(), self.FLAMEServer.faces_tensor.detach().cpu().numpy().astype(np.int64))
        flame_sdf = f(points.detach().cpu().numpy())
        flame_sdf = torch.from_numpy(flame_sdf).cuda()
        flame_sdf = flame_sdf * -1
        
        output = self.geometry_network(points, network_condition)
        head_sdf = output[:, 0]
        feature = output[:, 1:]
        
        return head_sdf, flame_sdf
    
        
    def forward(self, input, return_sdf=False):

        # Parse model input
        uv = input["uv"]
        intrinsics = input["intrinsics"]
        pose = input["cam_pose"]
        
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        flame_transl = input['flame_transl'][0]
        flame_scale = input['flame_scale'][0]
        
        object_mask = input["object_mask"].reshape(-1)
        head_mask = input["head_mask"].reshape(-1)
        hand_mask = input["hand_mask"].reshape(-1)
        
        nonrigid_params = input['nonrigid_params']
        
        ###################################################### HEAD ######################################################
        if "latent_code" in input:
            latent_code = input["latent_code"]
        else:
            latent_code = None

        if self.geometry_network.condition_in == 32:
            network_condition = latent_code
        elif self.geometry_network.condition_in == 50:
            network_condition = expression
        elif self.geometry_network.condition_in == 56:
            network_condition = torch.cat([flame_pose[:, 3:9], expression], dim=1)
        elif self.geometry_network.condition_in == 82:
            network_condition = torch.cat([expression, latent_code], dim=1)
        elif self.geometry_network.condition_in == 88:
            network_condition = torch.cat([flame_pose[:, 3:9], expression, latent_code], dim=1)
        elif self.geometry_network.condition_in == 0:
            network_condition = None

        if self.deformer_network.condition_in == 6:
            deformer_condition = flame_pose[:, 3:9]
        elif self.deformer_network.condition_in == 6+50:
            deformer_condition = torch.cat([flame_pose[:, 3:9], expression], dim=1)
        elif self.deformer_network.condition_in == 0:
            deformer_condition = None

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        verts_head, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, scale=flame_scale.reshape(-1), transl=flame_transl.reshape(-1,1,3))

        self.geometry_network.eval()
        self.deformer_network.eval()
        self.nonrigid_deformer.eval()
        with torch.no_grad():
            sdf_function = lambda x: self.query_sdf(pnts_p=x,
                                                    network_condition=network_condition,
                                                    deformer_condition=deformer_condition,
                                                    pose_feature=pose_feature,
                                                    betas=expression,
                                                    transformations=transformations,
                                                    transl=flame_transl,
                                                    nonrigid_params=nonrigid_params
                                                    )[0]
            points_head, pred_head_mask, dists = self.ray_tracer(sdf=sdf_function,
                                                                 cam_loc=cam_loc,
                                                                 object_mask=head_mask,
                                                                 ray_directions=ray_dirs)
        self.geometry_network.train()
        self.deformer_network.train()
        self.nonrigid_deformer.train()

        points_head = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output_head, canonical_points, _, others = self.query_sdf(points_head, network_condition, deformer_condition,
                                                                 pose_feature=pose_feature, betas=expression,
                                                                 transformations=transformations, 
                                                                 transl=flame_transl, 
                                                                 nonrigid_params=nonrigid_params)
        sdf_output_head = sdf_output_head.unsqueeze(1)
        nonrigid_deformation_head = others['nonrigid_deformation']
        lbs_weight = gt_lbs_weight = None
        posedirs = gt_posedirs = None
        shapedirs = gt_shapedirs = None
        nonrigid_dir = gt_nonrigid_dir = None
        if 'lbs_weight' in others and self.deformer_class == 'BackwardDeformer':
            surface_mask = pred_head_mask & head_mask if self.training else pred_head_mask
            lbs_weight = others['lbs_weight'][surface_mask]
            posedirs = others['posedirs'][surface_mask]
            shapedirs = others['shapedirs'][surface_mask]
            nonrigid_dir = others['nonrigid_dir'][surface_mask]
            _, index_batch, _ = ops.knn_points(points_head[surface_mask].unsqueeze(0), verts_head, K=1, return_nn=True)
            index_batch = index_batch[0, :, 0]
            gt_lbs_weight = self.FLAMEServer.lbs_weights[index_batch, :]
            gt_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, 100:] 
            gt_nonrigid_dir = self.FLAMEServer.nonrigid_dir[index_batch, :, :] 
            gt_posedirs = torch.transpose(self.FLAMEServer.posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        if self.training:
            surface_mask = pred_head_mask & head_mask
            surface_points = points_head[surface_mask]
            surface_canonical_points = canonical_points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs.reshape(-1, 3)[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output_head[surface_mask]
            nonrigid_deformation_head = nonrigid_deformation_head#[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points_head.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)
            grad_theta = self.geometry_network.gradient(eikonal_points, network_condition).squeeze(1)

            surface_sdf_values = surface_output.detach()

            surface_points_grad = self.gradient(surface_points, network_condition, deformer_condition, pose_feature,
                                                expression, transformations, create_graph=False, retain_graph=True, 
                                                transl=flame_transl, nonrigid_params=nonrigid_params
                                                ).clone().detach()

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = pred_head_mask
            differentiable_surface_points = points_head[surface_mask]
            grad_theta = None


        rgb_values_head = torch.ones_like(points_head).float().cuda()
        normal_values_head = torch.zeros_like(points_head).float().cuda()
        depth_values_head = torch.zeros_like(points_head[:, 0]).float().cuda()
        flame_distance_values = torch.zeros(points_head.shape[0]).float().cuda()
        
        depth_values_head[surface_mask] = rend_util.get_depth(differentiable_surface_points.reshape(batch_size, -1, 3), pose).reshape(-1)

        if "depth" in input:
            # the following is to obtain consistent depth scaling for evaluation (plotting).
            if (not 'depth_scale' in input) or input['depth_scale'] is None:
                depth_scale, depth_shift = compute_scale_and_shift(depth_values_head[None, :, None], input["depth"][:, :, None], (head_mask & surface_mask)[None, :, None])
                # depth_scale = depth_scale.item()
                # depth_shift = depth_shift.item()
                depth_scale = None
                depth_shift = None
            else:
                depth_scale = input['depth_scale']
                depth_shift = input['depth_shift']
            depth_values_copy = torch.zeros_like(points_head[:, 0]).float().cuda()
            depth_values_copy[surface_mask] = depth_values_head[surface_mask] #* depth_scale + depth_shift
            depth_values_head = depth_values_copy
        if differentiable_surface_points.shape[0] > 0:
            rgb_values_head[surface_mask], others = self.get_rbg_value(differentiable_surface_points, network_condition, deformer_condition, pose_feature, expression, transformations, is_training=self.training,
                                                                  jaw_pose=torch.cat([expression, flame_pose[:, 6:9]], dim=1),
                                                                  latent_code=latent_code, 
                                                                  transl=flame_transl,
                                                                  nonrigid_params=nonrigid_params
                                                                  )
            normal_values_head[surface_mask] = others['normals']

        # calculate on observation space
        knn_v = verts_head.clone()
        flame_distance, index_batch, _ = ops.knn_points(differentiable_surface_points.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        index_batch_values = torch.ones(points_head.shape[0]).long().cuda()
        index_batch_values[surface_mask] = index_batch
        flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)
        
        output = {
            'points_head': points_head,
            'rgb_values_head': rgb_values_head,
            'normal_values_head': normal_values_head,
            'depth_values_head': depth_values_head,
            'sdf_output_head': sdf_output_head,
            'pred_head_mask': pred_head_mask,
            'object_mask': object_mask,
            'head_mask': head_mask,
            'grad_theta': grad_theta,
            'expression': expression,
            'flame_pose': flame_pose,
            'cam_pose': pose,
            'index_batch': index_batch_values,
            'flame_distance': flame_distance_values,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_deform_dir': self.FLAMEServer.nonrigid_dir,
            'nonrigid_deformation_head': nonrigid_deformation_head,
            'verts_head': verts_head
        }
        if 'depth' in input:
            output['depth_scale'] = depth_scale
            output["depth_shift"] = depth_shift

        if lbs_weight is not None and gt_lbs_weight is not None:
            skinning_values = torch.ones(points_head.shape[0], 6 if self.ghostbone else 5).float().cuda()
            gt_skinning_values = torch.ones(points_head.shape[0], 6 if self.ghostbone else 5).float().cuda()
            skinning_values[surface_mask] = lbs_weight#[..., :3]
            gt_skinning_values[surface_mask] = gt_lbs_weight#[..., :3]
            output['lbs_weight'] = skinning_values
            output['gt_lbs_weight'] = gt_skinning_values
        if posedirs is not None and gt_posedirs is not None:
            posedirs_values = torch.ones(points_head.shape[0], 36, 3).float().cuda()
            gt_posedirs_values = torch.ones(points_head.shape[0], 36, 3).float().cuda()
            posedirs_values[surface_mask] = posedirs
            gt_posedirs_values[surface_mask] = gt_posedirs
            output['posedirs'] = posedirs_values
            output['gt_posedirs'] = gt_posedirs_values
        if shapedirs is not None and gt_shapedirs is not None:
            shapedirs_values = torch.ones(points_head.shape[0], 3, 50).float().cuda()
            gt_shapedirs_values = torch.ones(points_head.shape[0], 3, 50).float().cuda()
            shapedirs_values[surface_mask] = shapedirs
            gt_shapedirs_values[surface_mask] = gt_shapedirs
            output['shapedirs'] = shapedirs_values
            output['gt_shapedirs'] = gt_shapedirs_values
        if nonrigid_dir is not None and gt_nonrigid_dir is not None:
            nonrigid_dir_values = torch.ones(points_head.shape[0], 3, 30).float().cuda()
            gt_nonrigid_dir_values = torch.ones(points_head.shape[0], 3, 30).float().cuda()
            nonrigid_dir_values[surface_mask] = nonrigid_dir
            gt_nonrigid_dir_values[surface_mask] = gt_nonrigid_dir
            output['nonrigid_dir'] = nonrigid_dir_values
            output['gt_nonrigid_dir'] = gt_nonrigid_dir_values    
        
        ###################################################### HAND ######################################################
        mano_global_orient = input["mano_global_orient"]
        mano_hand_pose = input["mano_hand_pose"].reshape(-1, 15, 3, 3)
        mano_betas = input["mano_betas"]
        mano_transl = input["mano_transl"]
        mano_scale = input["mano_scale"]
        
        w2c_p = input["w2c_p"]
        cam_intrinsics = input["cam_intrinsics"].squeeze(0)

        pred_mano_params = {
            'global_orient': mano_global_orient,
            'hand_pose': mano_hand_pose,
            'betas': mano_betas,
            'transl': mano_transl,
            'scale': mano_scale
        }
        mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        verts_hand = mano_output.vertices.clone()
        
        if not self.training:
            with torch.no_grad():
                sdf_function_hand = lambda x: self.query_sdf_hand(pnts_p=x, 
                                                                verts=verts_hand,
                                                                )[0]
                points_hand, pred_hand_mask, dists_hand = self.ray_tracer.forward_hand(sdf=sdf_function_hand,
                                                                                        cam_loc=cam_loc,
                                                                                        object_mask=hand_mask,
                                                                                        ray_directions=ray_dirs)

            points_hand = (cam_loc.unsqueeze(1) + dists_hand.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
            output['points_hand'] = points_hand
            

        trans_verts_hand = projection(verts_hand, cam_intrinsics, w2c_p)
        trans_verts_hand_cam = projection(verts_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
        landmarks3d_p_hand = mano_output.joints.clone()
        trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
        
        depth_image_hand, mask_image_hand = self.deca.render_hand.render_depth(trans_verts_hand, trans_verts_hand_cam)
        pred_color_masked, normal_image_hand = self.deca.render_hand.render_neural_colors(self.shader, verts_hand, trans_verts_hand)
        
        depth_image_hand = depth_image_hand.permute(0,2,3,1)
        mask_image_hand = mask_image_hand.permute(0,2,3,1)
        depth_image_hand = depth_image_hand.clone().reshape(-1)
        mask_image_hand = mask_image_hand.clone().reshape(-1).bool()
        depth_image_hand[mask_image_hand] = depth_image_hand[mask_image_hand]

        gbuffer_mask = mask_image_hand.clone().reshape(-1)
        pred_color_masked = pred_color_masked.reshape(-1,3)
        normal_image_hand = normal_image_hand.reshape(-1,3)
        depth_image_hand = depth_image_hand.reshape(-1)
        
        # depth_image_hand = depth_image_hand * depth_scale + depth_shift
        
        if 'split_indx' in input.keys():
            indx = input['split_indx']
            gbuffer_mask = torch.index_select(gbuffer_mask, 0, indx)
            pred_color_masked = torch.index_select(pred_color_masked, 0, indx)
            normal_image_hand = torch.index_select(normal_image_hand, 0, indx)
            depth_image_hand = torch.index_select(depth_image_hand, 0, indx)
        if 'sampling_idx' in input.keys():
            sampling_idx = input["sampling_idx"].reshape(-1)
            bbox_inside = input["bbox_inside"].reshape(-1)
            sampling_idx_bbox = input["sampling_idx_bbox"].reshape(-1)
            gbuffer_mask = torch.cat([gbuffer_mask[sampling_idx], gbuffer_mask[bbox_inside][sampling_idx_bbox]], 0)
        
        pred_hand_mask = gbuffer_mask.clone().bool() #& pred_hand_mask
        surface_mask_hand = pred_hand_mask & hand_mask if self.training else pred_hand_mask
        
        rgb_values_hand = torch.ones_like(points_head).float().cuda()
        normal_values_hand = torch.zeros_like(points_head).float().cuda()
        depth_values_hand = torch.zeros_like(points_head[:, 0]).float().cuda()
        if 'sampling_idx' in input.keys():
            sampling_idx = input["sampling_idx"].reshape(-1)
            bbox_inside = input["bbox_inside"].reshape(-1)
            sampling_idx_bbox = input["sampling_idx_bbox"].reshape(-1)
            rgb_values_hand[surface_mask_hand] = torch.cat([pred_color_masked[sampling_idx], pred_color_masked[bbox_inside][sampling_idx_bbox, :]], 0)[surface_mask_hand]
            normal_values_hand[surface_mask_hand] = torch.cat([normal_image_hand[sampling_idx], normal_image_hand[bbox_inside][sampling_idx_bbox, :]], 0)[surface_mask_hand]
            depth_values_hand[surface_mask_hand] = torch.cat([depth_image_hand[sampling_idx], depth_image_hand[bbox_inside][sampling_idx_bbox]], 0)[surface_mask_hand]
        else:
            rgb_values_hand[surface_mask_hand] = pred_color_masked[surface_mask_hand]
            normal_values_hand[surface_mask_hand] = normal_image_hand[surface_mask_hand]
            depth_values_hand[surface_mask_hand] = depth_image_hand[surface_mask_hand]

        output['rgb_values_hand'] = rgb_values_hand
        output['normal_values_hand'] = normal_values_hand
        output['depth_values_hand'] = depth_values_hand
        # output['sdf_output_hand'] = sdf_output_hand
        output['pred_hand_mask'] = pred_hand_mask
        output['hand_mask'] = hand_mask
        output['hand_pose'] = mano_hand_pose

        output['rgb_image_hand'] = pred_color_masked
        output['normal_image_hand'] = normal_image_hand
        output['depth_image_hand'] = depth_image_hand
        output['mask_image_hand'] = mask_image_hand
        output['hand_lmk'] = trans_landmarks2d_hand
        output['mano_transl'] = mano_transl
        output['verts_hand'] = verts_hand
        
        output['img_res'] = input['img_res']
        
        if 'optimize_mano_pose' in input.keys():
            output['optimize_mano_pose'] = input['optimize_mano_pose']
        ##################################################################################################################
        if self.training:
            sdf_sampleflamebbox_tohead, sdf_sampleflamebbox_toflame = self.query_sdf_flame_bbox(network_condition) # for points in the flame mesh bbox
            output['sdf_sampleflamebbox_tohead'] = sdf_sampleflamebbox_tohead
            output['sdf_sampleflamebbox_toflame'] = sdf_sampleflamebbox_toflame

        if 'optimize_contact' in input.keys() and input['optimize_contact'] == True:
            optimize_contact = True
        else: 
            optimize_contact = False
        output['optimize_contact'] = optimize_contact
        
        if self.training and optimize_contact:
            if (not 'sdf_hand_mask' in input) or input['sdf_hand_mask'] is None:
                surface_output_values = torch.ones(points_head.shape[0], 1).float().cuda()
                surface_output_values[surface_mask] = surface_output
                output['surface_output'] = surface_output_values # sdf value of the surface of head
                
                meshes_hand = Meshes(verts_hand.reshape(1,-1,3), self.mano_faces.reshape(1,-1,3))
                samples_hand = sample_points_from_meshes(meshes_hand, num_samples=100000, return_normals=False, return_textures=False)
                samples_hand = samples_hand.reshape(-1,3)
                
                # # Add Gaussian noise scaled to 1e-1
                # noise_hand = torch.randn_like(samples_hand) * 1e-2
                # samples_hand = samples_hand + noise_hand.to(samples_hand.device)
                
                sdf_sampleonhand_tohead, _, _, others_sampleonhand_tohead = self.query_sdf(samples_hand, network_condition, deformer_condition,
                                                                                pose_feature=pose_feature, betas=expression,
                                                                                transformations=transformations, 
                                                                                transl=flame_transl, 
                                                                                nonrigid_params=nonrigid_params)
                output['sdf_sampleonhand_tohead'] = sdf_sampleonhand_tohead
                output['nonrigid_deformation_sampleonhand_tohead'] = others_sampleonhand_tohead['nonrigid_deformation']
                
                sdf_sampleonhand, _, _, _ = self.query_sdf_hand(samples_hand, verts=verts_hand)
                output['sdf_sampleonhand'] = sdf_sampleonhand 
            

                sdf_onhead_tohand, _, _, _ = self.query_sdf_hand(points_head, verts=verts_hand)
                output['sdf_onhead_tohand'] = sdf_onhead_tohand 
    
                # sdf_onhead_tohand_values = torch.ones(points_head.shape[0], 1).float().cuda()
                # sdf_onhead_tohand_values[surface_mask] = sdf_onhead_tohand.unsqueeze(-1)[surface_mask]
                # output['sdf_onhead_tohand_surface'] = sdf_onhead_tohand_values
            
                meshes_head = Meshes(verts_head.reshape(1,-1,3), self.FLAMEServer.faces_tensor.reshape(1,-1,3))
                samples_head = sample_points_from_meshes(meshes_head, num_samples=10000, return_normals=False, return_textures=False)
                samples_head = samples_head.reshape(-1,3)
                
                # # Add Gaussian noise scaled to 1e-1
                # noise = torch.randn_like(samples_head) * 1e-2
                # samples_head = samples_head + noise.to(samples_head.device)
                
                sdf_sampleonhead, _, _, others_sampleonhead = self.query_sdf(samples_head, network_condition, deformer_condition,
                                                                pose_feature=pose_feature, betas=expression,
                                                                transformations=transformations, 
                                                                transl=flame_transl, 
                                                                nonrigid_params=nonrigid_params)
                output['sdf_sampleonhead'] = sdf_sampleonhead
                output['nonrigid_deformation_sampleonhead'] = others_sampleonhead['nonrigid_deformation']
      
                sdf_sampleonhead_tohand, _, _, _ = self.query_sdf_hand(samples_head, verts=verts_hand)
                output['sdf_sampleonhead_tohand'] = sdf_sampleonhead_tohand

                # output['sdf_hand_mask'] = (sdf_sampleonhand_tohead.reshape(-1) < 0) &  (sdf_sampleonhand.reshape(-1) < 0)
                # margin = 1e-5
                margin = 1e-6
                # margin = 5e-7
                # surface_mask = (sdf_sampleonhand_tohead < 0) & (-margin < sdf_sampleonhand) & (sdf_sampleonhand < 0)
                surface_mask = (sdf_sampleonhand_tohead < 0) & (-margin < sdf_sampleonhand) & (sdf_sampleonhand < margin)
                # surface_mask = (sdf_sampleonhand_tohead < 0) & (sdf_sampleonhand < 0)
                output['sdf_hand_mask'] = surface_mask
                # output['sdf_head_mask'] = (sdf_sampleonhead_tohand.reshape(-1) < 0) &  (sdf_sampleonhead.reshape(-1) < 0)
                output['samples_hand'] = samples_hand
                # output['samples_head'] = samples_head
                
            else:
                samples_hand = input['samples_hand'].cuda()
                # samples_head = input['samples_head']
                sdf_hand_mask = input['sdf_hand_mask'].cuda()
                # sdf_head_mask = input['sdf_head_mask']
                
                surface_output_values = torch.ones(points_head.shape[0], 1).float().cuda()
                surface_output_values[surface_mask] = surface_output
                output['surface_output'] = surface_output_values # sdf value of the surface of head
                
                meshes_hand = Meshes(verts_hand.reshape(1,-1,3), self.mano_faces.reshape(1,-1,3))
                samples_hand = sample_points_from_meshes(meshes_hand, num_samples=100000, return_normals=False, return_textures=False)
                samples_hand = samples_hand.reshape(-1,3)
                
                # # Add Gaussian noise scaled to 1e-1
                # noise_hand = torch.randn_like(samples_hand) * 1e-2
                # samples_hand = samples_hand + noise_hand.to(samples_hand.device)
                
                sdf_sampleonhand_tohead, _, _, others_sampleonhand_tohead = self.query_sdf(samples_hand, network_condition, deformer_condition,
                                                                                pose_feature=pose_feature, betas=expression,
                                                                                transformations=transformations, 
                                                                                transl=flame_transl, 
                                                                                nonrigid_params=nonrigid_params)
                output['sdf_sampleonhand_tohead'] = sdf_sampleonhand_tohead
                output['nonrigid_deformation_sampleonhand_tohead'] = others_sampleonhand_tohead['nonrigid_deformation']
                
                sdf_sampleonhand, _, _, _ = self.query_sdf_hand(samples_hand, verts=verts_hand)
                output['sdf_sampleonhand'] = sdf_sampleonhand 
            

                sdf_onhead_tohand, _, _, _ = self.query_sdf_hand(points_head, verts=verts_hand)
                output['sdf_onhead_tohand'] = sdf_onhead_tohand 
    
                # sdf_onhead_tohand_values = torch.ones(points_head.shape[0], 1).float().cuda()
                # sdf_onhead_tohand_values[surface_mask] = sdf_onhead_tohand.unsqueeze(-1)[surface_mask]
                # output['sdf_onhead_tohand_surface'] = sdf_onhead_tohand_values
            
                meshes_head = Meshes(verts_head.reshape(1,-1,3), self.FLAMEServer.faces_tensor.reshape(1,-1,3))
                samples_head = sample_points_from_meshes(meshes_head, num_samples=10000, return_normals=False, return_textures=False)
                samples_head = samples_head.reshape(-1,3)
                
                # Add Gaussian noise scaled to 1e-1
                noise = torch.randn_like(samples_head) * 1e-2
                samples_head = samples_head + noise.to(samples_head.device)
                
                sdf_sampleonhead, _, _, others_sampleonhead = self.query_sdf(samples_head, network_condition, deformer_condition,
                                                                pose_feature=pose_feature, betas=expression,
                                                                transformations=transformations, 
                                                                transl=flame_transl, 
                                                                nonrigid_params=nonrigid_params)
                output['sdf_sampleonhead'] = sdf_sampleonhead
                output['nonrigid_deformation_sampleonhead'] = others_sampleonhead['nonrigid_deformation']
      
                sdf_sampleonhead_tohand, _, _, _ = self.query_sdf_hand(samples_head, verts=verts_hand)
                output['sdf_sampleonhead_tohand'] = sdf_sampleonhead_tohand


                output['sdf_hand_mask'] = sdf_hand_mask
                # output['sdf_head_mask'] = sdf_head_mask
                output['samples_hand'] = samples_hand
                # output['samples_head'] = samples_head

        if not return_sdf:
            return output
        else:
            return output, sdf_function

    def get_rbg_value(self, points, network_condition, deformer_condition, pose_feature, betas, transformations, jaw_pose=None, latent_code=None, is_training=True, transl=None, nonrigid_params=None):
        others = {}
        if self.deformer_class != 'ForwardDeformer':
            points.requires_grad_(True)
            sdf, pnts_c, feature_vectors, others = self.query_sdf(points, network_condition, deformer_condition, pose_feature, betas, transformations, transl=transl, nonrigid_params=nonrigid_params)
            gradients = self.gradient(points, network_condition, deformer_condition, pose_feature, betas, transformations, sdf=sdf, create_graph=is_training, retain_graph=is_training, transl=transl, nonrigid_params=nonrigid_params)
        else:
            pnts_c = points
            # others = {}
            _, gradients, feature_vectors = self.forward_gradient(pnts_c, network_condition, pose_feature, betas, transformations, create_graph=is_training, retain_graph=is_training)

        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        rgb_vals = self.rendering_network(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose)

        others['normals'] = normals
        others['pnts_c'] = pnts_c
        return rgb_vals, others
    

    def gradient(self, x, network_condition, deformer_condition, pose_feature, betas, transformations, sdf=None, create_graph=True, retain_graph=True, transl=None, nonrigid_params=None):
        x.requires_grad_(True)
        if sdf is None:
            y = self.query_sdf(x, network_condition, deformer_condition, pose_feature, betas, transformations, transl=transl, nonrigid_params=nonrigid_params)[0]
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

    def forward_gradient(self, pnts_c, network_condition, pose_feature, betas, transformations, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            # d_out = d_out.double()*scale
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.geometry_network(pnts_c, network_condition)
        sdf = output[:, :1]
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]

        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1), feature


    def get_differentiable_x(self, pnts_c, network_condition, pose_feature, betas, transformations, view_dirs, cam_loc):
        # canonical_x : num_points, 3
        # cam_loc: num_points, 3
        # view_dirs: num_points, 3
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()

        pnts_c = pnts_c.detach()
        pnts_c.requires_grad_(True)
        deformed_x = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        sdf = self.geometry_network(pnts_c, network_condition)[:, 0:1]
        dirs = deformed_x - cam_loc
        cross_product = torch.cross(view_dirs, dirs)
        constant = torch.cat([cross_product[:, 0:2], sdf], dim=1)
        # constant: num_points, 3
        num_dim = constant.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(constant, requires_grad=False, device=constant.device)
            d_out[:, i] = 1
            # d_out = d_out.double()*scale
            grad = torch.autograd.grad(
                outputs=constant,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()
        # grad_inv: num_points, 3, 3

        differentiable_x = pnts_c.detach() - torch.einsum('bij,bj->bi', grads_inv, constant - constant.detach())
        return differentiable_x

    def get_rbg_value_hand(self, points, is_training=True, verts=None):
        points.requires_grad_(True)
        sdf, pnts_c, feature_vectors, others = self.query_sdf_hand(points, verts=verts)
        # gradients = self.gradient_hand(points, sdf=sdf, create_graph=is_training, retain_graph=is_training)
        others['pnts_c'] = pnts_c
        
        mano_normal = others['mano_normal']
        normals = mano_normal
        normals = nn.functional.normalize(mano_normal, dim=-1, eps=1e-6)
       
        # rgb_vals = self.rendering_network_hand(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose)
        rgb_vals = torch.zeros_like(mano_normal).to(mano_normal.device)

        others['normals'] = normals
        return rgb_vals, others

    def gradient_hand(self, x, sdf=None, create_graph=True, retain_graph=True, verts=None):
        x.requires_grad_(True)
        if sdf is None:
            # y = self.query_sdf_hand(x, network_condition, deformer_condition, pose_feature, betas, transformations, verts=verts)[0]
            sdf, pnts_c, feature_vectors, others = self.query_sdf_hand(x,  verts=verts)#[-1]['sdf_residue']
            y = others['sdf_residue']
            mano_normal = others['mano_normal']
            # y = others['sdf_full']

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=create_graph,
                retain_graph=retain_graph,
                only_inputs=True)[0]

            gradients = gradients + mano_normal

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