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
from flame.lbs import inverse_skinning_pts, forward_skinning_pts, forward_pts, inverse_pts
from flame.FLAME_metahuman_pca import FLAME
from pytorch3d import ops
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from model.broyden import broyden

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
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    points = points.reshape(1, -1, 3)
    verts = verts.reshape(1, -1, 3)
    faces = faces.reshape(1, -1, 3)

    Bsize = points.shape[0]

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

class BackwardDeformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 condition_in,
                 dims,
                 multires,
                 FLAMEServer=None,
                 geometric_init=True,
                 weight_norm=True,
                 skinning_only=False,
                 num_exp=50,
                 ghostbone=True):
        super().__init__()
        dims = [d_in + condition_in] + dims + [d_out]
        self.condition_in = condition_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)

        self.skinning_only = skinning_only
        # dims[-1] = 50*3 + 36*3 + 5 # 50 expression blendshapes + 4 pose correctives + 5 skinning weight

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        # self.softplus = nn.Softplus(beta=100)
        # self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out - 5)
        # self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        # self.skinning = nn.Linear(dims[self.num_layers - 2], 5)
        # torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        # torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        # self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        # torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        # torch.nn.init.constant_(self.skinning.bias, 0.0)
        # torch.nn.init.constant_(self.skinning.weight, 0.0)
        
        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)
        
        self.ghostbone = ghostbone

    def forward(self, points, conditions, pose_feature, betas, transformations, transl=None, scale=None):
        if transl is not None:
            points = points - transl
        # if scale is not None:
        #     points = points / scale
            
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.condition_in != 0:
            num_pixels = int(points.shape[0] / conditions.shape[0])
            conditions = conditions.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.condition_in)
            x = torch.cat([points, conditions], dim=1)
        else:
            x = points

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3].reshape(-1, 4*9, 3)
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + 50 * 3].reshape(-1, 3, 50)
        # deform_dir = blendshapes[:, 36 * 3 + 50 * 3: 36 * 3 + 50 * 3 + 30 * 3].reshape(-1, 3, 30)
        if self.skinning_only:
            posedirs = torch.zeros_like(posedirs)
            shapedirs = torch.zeros_like(shapedirs)

        # lbs_weight = self.skinning(self.softplus(self.skinning_linear(x))).reshape(-1, 5)
        lbs_weight = self.skinning(self.softplus(self.skinning_linear(x))).reshape(-1, 6 if self.ghostbone else 5)

        lbs_weights = torch.nn.functional.softmax(20 * lbs_weight, dim=1)
        pts_c = inverse_pts(points, betas.expand(points.shape[0], -1), transformations.expand(points.shape[0], -1, -1, -1), pose_feature.squeeze(0).expand(points.shape[0], -1), shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        
        # nonrigid_deformation = torch.einsum('ml,mkl->mk', [deform_params.expand(points.shape[0], -1), deform_dir])
        # pts_c = pts_c - nonrigid_deformation

        others = {'lbs_weight': lbs_weights, 'posedirs': posedirs, 'shapedirs': shapedirs, 'x': x}
        return pts_c, others

class NonrigidDeformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 condition_in,
                 dims,
                 multires,
                 weight_norm=True,):
        super().__init__()
        # dims = [d_in + condition_in + 32] + dims + [d_out]
        dims = [d_in] + dims + [d_out]
        # dims = [d_in + 32] + dims + [d_out]
        self.condition_in = condition_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.lin_out = nn.Linear(dims[self.num_layers - 2], d_out)
        torch.nn.init.constant_(self.lin_out.bias, 0.0)
        torch.nn.init.constant_(self.lin_out.weight, 0.0)

        self.softplus = nn.Softplus(beta=100)
        
        param_dims = [30] + [512, 512, 512, 512] + [30]
        self.param_num_layers = len(param_dims)

        for l in range(0, self.param_num_layers - 2):
            out_dim = param_dims[l + 1]
            lin = nn.Linear(param_dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "param_lin" + str(l), lin)

        self.param_lin_out = nn.Linear(param_dims[self.param_num_layers - 2], param_dims[-1])
        torch.nn.init.constant_(self.param_lin_out.bias, 0.0)
        torch.nn.init.constant_(self.param_lin_out.weight, 0.0)
        
        
    def forward(self, points, latent_code=None, transl=None, scale=None, deform_params=None):
        if transl is not None:
            points = points - transl.reshape(-1, 3)

        if self.embed_fn is not None:
            points = self.embed_fn(points)

        # num_pixels = int(points.shape[0] / latent_code.shape[0])
        # latent_code = latent_code.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, latent_code.shape[-1])
        # x = torch.cat([points, latent_code], dim=1)

        x = points

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        deform_dir = self.lin_out(x).reshape(-1, 3, 30)

        ###
        for l in range(0, self.param_num_layers - 2):
            lin = getattr(self, "param_lin" + str(l))
            deform_params = lin(deform_params)
            deform_params = self.softplus(deform_params)

        deform_params = self.param_lin_out(deform_params).reshape(-1, 30)
        ###
        nonrigid_deformation = torch.einsum('ml,mkl->mk', [deform_params.expand(points.shape[0], -1), deform_dir])

        return {'nonrigid_deformation': nonrigid_deformation, 'deform_dir': deform_dir}

# class NonrigidDeformer(nn.Module):
#     def __init__(self,
#                  d_in,
#                  d_out,
#                  condition_in,
#                  dims,
#                  multires,
#                  weight_norm=True,):
#         super().__init__()
#         # dims = [d_in + condition_in + 32] + dims + [d_out]
#         dims = [128] + dims + [d_out]
#         # dims = [d_in + 32] + dims + [d_out]
#         # self.condition_in = condition_in
#         # self.embed_fn = None
#         # if multires > 0:
#         #     embed_fn, input_ch = get_embedder(multires)
#         #     self.embed_fn = embed_fn
#         #     dims[0] += input_ch - 3
        
#         self.softplus = nn.Softplus(beta=100)
#         self.blendshapes = nn.Linear(128, 90)
#         torch.nn.init.constant_(self.blendshapes.bias, 0.0)
#         torch.nn.init.constant_(self.blendshapes.weight, 0.0)

#         param_dims = [30] + [128] + [30]
#         self.param_num_layers = len(param_dims)

#         for l in range(0, self.param_num_layers - 2):
#             out_dim = param_dims[l + 1]
#             lin = nn.Linear(param_dims[l], out_dim)

#             if multires > 0 and l == 0:
#                 torch.nn.init.constant_(lin.bias, 0.0)
#                 torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
#                 torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
#             else:
#                 torch.nn.init.constant_(lin.bias, 0.0)
#                 torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "param_lin" + str(l), lin)

#         self.param_lin_out = nn.Linear(param_dims[self.param_num_layers - 2], param_dims[-1])
#         torch.nn.init.constant_(self.param_lin_out.bias, 0.0)
#         torch.nn.init.constant_(self.param_lin_out.weight, 0.0)
        
#     def forward(self, x, latent_code=None, transl=None, scale=None, deform_params=None):
#         deform_dir = self.blendshapes(x).reshape(-1, 3, 30)
        
#         ###
#         params = deform_params
#         for l in range(0, self.param_num_layers - 2):
#             lin = getattr(self, "param_lin" + str(l))
#             params = lin(params)
#             params = self.softplus(params)

#         params = self.param_lin_out(params).reshape(-1, 30)
#         ###
        
#         nonrigid_deformation = torch.einsum('ml,mkl->mk', [params.expand(x.shape[0], -1), deform_dir])

#         return {'nonrigid_deformation': nonrigid_deformation, 'deform_dir': deform_dir}

class IMavatar(nn.Module):
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
        # self.geometry_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('geometry_network'))
        self.geometry_network = GeometryNetwork(self.feature_vector_size, **conf.get_config('geometry_network'))
        self.deformer_class = conf.get_string('deformer_class').split('.')[-1]
        self.deformer_network = utils.get_class(conf.get_string('deformer_class'))(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))

        self.nonrigid_deformer_network = utils.get_class(conf.get_string('deformer_class_nonrigid'))(**conf.get_config('nonrigid_deformer_network'))

        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.distance = 1.0
        
        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.gt_w_seg = gt_w_seg
        
        self.MANOServer = MANOLayer(model_path="../code/mano_model/data/mano",
                                    is_rhand=True,
                                    batch_size=1,
                                    flat_hand_mean=False,
                                    dtype=torch.float32,
                                    use_pca=False,)
        
        vt, ft, f = read_mano_uv_obj('../code/mano_model/data/MANO_UV_right.obj')
        self.mano_faces = torch.tensor(f).cuda()                                 #NOTE: this is same as mano_layer[hand_type].faces
        self.mano_face_uv = torch.tensor(vt[ft], dtype=torch.float32).cuda()  
        
        vt, ft, f = read_mano_uv_obj('../code/mano_model/data/MANO_UV_left.obj')
        self.mano_faces_left = torch.tensor(f).cuda()                                 #NOTE: this is same as mano_layer[hand_type].faces
        self.mano_face_uv_left = torch.tensor(vt[ft], dtype=torch.float32).cuda()  
        
        self.device = torch.device('cuda')
        
        # with open("/home/haonan/Codes/IMavatar/code/mano_model/data/contact_zones.pkl", "rb") as f:
        with open("../code/mano_model/data/contact_zones.pkl", "rb") as f:
            contact_zones = pickle.load(f)
        contact_zones = contact_zones["contact_zones"]
        contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])
        contact_idx = torch.from_numpy(contact_idx).to(self.device)
        # self.contact_idx = contact_idx[19:] #fingers
        self.contact_idx = contact_idx[19:47] #index_fingers

        import sys
        # sys.path.append('/home/haonan/Codes/IMavatar/preprocess/submodules/DECA')
        sys.path.append('/home/haonan/data/IMavatar_iccv/preprocess/submodules/DECA')
        from decalib.deca import DECA
        from decalib.utils import util
        from decalib.utils.config import cfg as deca_cfg
        device = 'cuda'
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device, image_size=256, uv_size=256)
        # self.deca = DECA(config=deca_cfg, device=device, image_size=224, uv_size=224)

        # light_mlp_ch = 3
        # light_mlp_dims = [64, 64]
        # material_mlp_dims = [128, 128, 128, 128]
        # material_mlp_ch = 3
        color_mlp_dims = [128, 128, 128, 128]
        disentangle_network_params = {
            # "material_mlp_ch": material_mlp_ch,
            # "light_mlp_ch":light_mlp_ch,
            # "material_mlp_dims":material_mlp_dims,
            # "light_mlp_dims":light_mlp_dims
            'color_mlp_dims': color_mlp_dims
        }
        # Create the optimizer for the neural shader
        self.shader = NeuralShader(fourier_features='positional',
                            activation='relu',
                            last_activation=torch.nn.Sigmoid(), 
                            disentangle_network_params=disentangle_network_params,
                            aabb=None,
                            device=device)
        
    def query_sdf(self, pnts_p, network_condition, deformer_condition, pose_feature, betas, transformations, verts=None, use_verts=False, transl=None, scale=None, deform_params=None):
        pnts_c, others = self.deformer_network(pnts_p, deformer_condition, pose_feature, betas, transformations, transl, scale)
        
        # if self.training:
        #     breakpoint()
        # nonrigid_output = self.nonrigid_deformer_network(others['x'], transl=transl, deform_params=deform_params)
        nonrigid_output = self.nonrigid_deformer_network(pnts_p, transl=transl, deform_params=deform_params)
        nonrigid_deformation = nonrigid_output['nonrigid_deformation']
        deform_dir = nonrigid_output['deform_dir']
        
        # grads = [p.requires_grad for p in self.nonrigid_deformer_network.parameters()] 
        
        pnts_c = pnts_c - nonrigid_deformation

        others['nonrigid_deformation'] = nonrigid_deformation
        others['deform_dir'] = deform_dir
        
        output = self.geometry_network(pnts_c, network_condition)
        sdf = output[:, 0]
        feature = output[:, 1:]
        return sdf, pnts_c, feature, others

    # def query_sdf_detach(self, pnts_p, network_condition, deformer_condition, pose_feature, betas, transformations, verts=None, use_verts=False, transl=None, scale=None, deform_params=None):
    #     pnts_c, others = self.deformer_network(pnts_p, deformer_condition, pose_feature, betas, transformations, transl, scale, deform_params)
        
    #     output = self.nonrigid_deformer_network(pnts_p, transl=transl, deform_params=deform_params)
    #     nonrigid_deformation = output['nonrigid_deformation']
    #     deform_dir = output['deform_dir']

    #     pnts_c = pnts_c.clone().detach() - nonrigid_deformation

    #     others['nonrigid_deformation'] = nonrigid_deformation
    #     others['deform_dir'] = deform_dir
        
    #     output = self.geometry_network(pnts_c, network_condition)
    #     sdf = output[:, 0]
    #     feature = output[:, 1:]
    #     return sdf, pnts_c, feature, others
 

    def query_sdf_hand(self, pnts_p, verts=None, use_verts=False):
        # ForwardDeformer
        # filter points: if points are too far away from flame vertices, return sdf = 1 (outside)
        others = {}

        def pysdf_func(pnts_p, use_verts=False, verts=verts):
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
    
    def query_sdf_hand_left(self, pnts_p, verts=None, use_verts=False):
        # ForwardDeformer
        # filter points: if points are too far away from flame vertices, return sdf = 1 (outside)
        others = {}

        def pysdf_func(pnts_p, use_verts=False, verts=verts):
            verts = verts.reshape(-1, 3)
            pnts_p = pnts_p.reshape(-1, 3)
            f = SDF(verts.squeeze(0).detach().cpu().numpy(), self.MANOServer.faces.astype(np.int64))
            mano_sdf = f(pnts_p.detach().cpu().numpy())
            mano_sdf = torch.from_numpy(mano_sdf).to(pnts_p.device)#.unsqueeze(-1)
            mano_sdf = mano_sdf * -1
            return mano_sdf

        mano_sdf = pysdf_func(pnts_p, verts=verts)
        mano_normal = get_normal(pnts_p, verts.squeeze(0), self.mano_faces_left)
        
        sdf = mano_sdf.reshape(-1,)
        pnts_c = pnts_p.clone().reshape(-1, 3)
        feature = None
        others = {'mano_normal': mano_normal.reshape(-1, 3)}
        
        return sdf, pnts_c, feature, others
    
    def query_sdf_flame(self, network_condition):
        
        # min_coords = torch.tensor([-1,-1,-1]).float().cuda()
        # max_coords = torch.tensor([1,1,1]).float().cuda()
        
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
        sdf = output[:, 0]
        feature = output[:, 1:]
        
        return sdf, flame_sdf
    
        
    def forward(self, input, return_sdf=False):

        # Parse model input
        
        deform_params = input['deform_params']
        
        uv = input["uv"]

        intrinsics = input["intrinsics"]
        pose = input["cam_pose"]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        flame_transl = input['flame_transl'][0]
        flame_scale = input['flame_scale'][0]
        
        object_mask = input["object_mask"].reshape(-1)
        head_mask = input["head_mask"].reshape(-1)
        left_hand_mask = input["left_hand_mask"].reshape(-1)
        right_hand_mask = input["right_hand_mask"].reshape(-1)
        hand_mask = (left_hand_mask.bool() | right_hand_mask.bool()).float()
        if "latent_code" in input:
            latent_code = input["latent_code"]
        else:
            latent_code = None

        if self.geometry_network.condition_in == 32:
            network_condition = latent_code
        elif self.geometry_network.condition_in == 82:
            network_condition = torch.cat([expression, latent_code], dim=1)
        elif self.geometry_network.condition_in == 88:
            network_condition = torch.cat([flame_pose[:, 3:9], expression, latent_code], dim=1)
        elif self.geometry_network.condition_in == 50:
            network_condition = expression
        elif self.geometry_network.condition_in == 56:
            network_condition = torch.cat([flame_pose[:, 3:9], expression], dim=1)
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

        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose, scale=flame_scale.reshape(-1), transl=flame_transl.reshape(-1,1,3))
        # if flame_transl is not None:
        #     verts = verts * flame_scale.reshape(1,1,1)
        #     verts = verts + flame_transl.reshape(1,1,3)

        self.geometry_network.eval()
        self.deformer_network.eval()
        self.nonrigid_deformer_network.eval()
        with torch.no_grad():
            sdf_function = lambda x, use_verts=False: self.query_sdf(pnts_p=x, use_verts=False,
                                                                    network_condition=network_condition,
                                                                    deformer_condition=deformer_condition,
                                                                    pose_feature=pose_feature,
                                                                    betas=expression,
                                                                    transformations=transformations,
                                                                    verts=verts,
                                                                    transl=flame_transl,
                                                                    deform_params=deform_params
                                                                    # scale=flame_scale
                                                                    )[0]
            points, network_object_mask, dists = self.ray_tracer(sdf=sdf_function,
                                                                 cam_loc=cam_loc,
                                                                 object_mask=head_mask,
                                                                 ray_directions=ray_dirs)
        self.geometry_network.train()
        self.deformer_network.train()
        self.nonrigid_deformer_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output, canonical_points, _, others = self.query_sdf(points, network_condition, deformer_condition,
                                                                 pose_feature=pose_feature, betas=expression,
                                                                 transformations=transformations, 
                                                                 transl=flame_transl, 
                                                                 scale=None,
                                                                 deform_params=deform_params)
        sdf_output = sdf_output.unsqueeze(1)
        nonrigid_deformation = others['nonrigid_deformation']
        lbs_weight = gt_lbs_weight = None
        posedirs = gt_posedirs = None
        shapedirs = gt_shapedirs = None
        deform_dir = gt_deform_dir = None
        displacement = gt_displacement = None
        if 'lbs_weight' in others and self.deformer_class == 'BackwardDeformer':

            surface_mask = network_object_mask & head_mask if self.training else network_object_mask
            lbs_weight = others['lbs_weight'][surface_mask]
            posedirs = others['posedirs'][surface_mask]
            shapedirs = others['shapedirs'][surface_mask]
            deform_dir = others['deform_dir'][surface_mask]
            _, index_batch, _ = ops.knn_points(points[surface_mask].unsqueeze(0), verts, K=1, return_nn=True)
            index_batch = index_batch[0, :, 0]
            gt_lbs_weight = self.FLAMEServer.lbs_weights[index_batch, :]
            # gt_lbs_weight = gt_lbs_weight[:, :3]
            # gt_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, 100:] * flame_scale.reshape(-1)
            # gt_posedirs = torch.transpose(self.FLAMEServer.posedirs.reshape(36, -1, 3) * flame_scale.reshape(-1), 0, 1)[index_batch, :, :]
            gt_shapedirs = self.FLAMEServer.shapedirs[index_batch, :, 100:] 
            gt_deform_dir = self.FLAMEServer.deform_dir[index_batch, :, :] 
            gt_posedirs = torch.transpose(self.FLAMEServer.posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        if self.training:
            surface_mask = network_object_mask & head_mask
            surface_points = points[surface_mask]
            surface_canonical_points = canonical_points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs.reshape(-1, 3)[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            nonrigid_deformation = nonrigid_deformation#[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)
            grad_theta = self.geometry_network.gradient(eikonal_points, network_condition).squeeze(1)

            surface_sdf_values = surface_output.detach()


            if self.deformer_class != 'ForwardDeformer':
                surface_points_grad = self.gradient(surface_points, network_condition, deformer_condition, pose_feature,
                                                    expression, transformations, create_graph=False, retain_graph=True, 
                                                    transl=flame_transl, deform_params=deform_params
                                                    ).clone().detach()

                differentiable_surface_points = self.sample_network(surface_output,
                                                                    surface_sdf_values,
                                                                    surface_points_grad,
                                                                    surface_dists,
                                                                    surface_cam_loc,
                                                                    surface_ray_dirs)
            else:
                differentiable_surface_points = self.get_differentiable_x(pnts_c=surface_canonical_points,
                                                                          network_condition=network_condition,
                                                                          pose_feature=pose_feature,
                                                                          betas=expression,
                                                                          transformations=transformations,
                                                                          view_dirs=surface_ray_dirs,
                                                                          cam_loc=surface_cam_loc)

        else:
            surface_mask = network_object_mask
            if self.deformer_class != 'ForwardDeformer':
                differentiable_surface_points = points[surface_mask]
            else:
                differentiable_surface_points = canonical_points[surface_mask]
            grad_theta = None


        rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.zeros_like(points).float().cuda()
        depth_values = torch.zeros_like(points[:, 0]).float().cuda()
        # if self.rendering_network.mode == 'spherical_harmonics':
        #     shading_values = torch.ones_like(points).float().cuda() * 2.
        #     alebdo_values = torch.ones_like(points).float().cuda()
        depth_values[surface_mask] = rend_util.get_depth(differentiable_surface_points.reshape(batch_size, -1, 3), pose).reshape(-1)
        # depth_values[surface_mask] = (differentiable_surface_points - cam_loc).norm(dim=-1)

        if "depth" in input:
            # the following is to obtain consistent depth scaling for evaluation (plotting).
            if (not 'depth_scale' in input) or input['depth_scale'] is None:
                scale, shift = compute_scale_and_shift(depth_values[None, :, None], input["depth"][:, :, None], (head_mask & surface_mask)[None, :, None])
                scale = scale.item()
                shift = shift.item()
            else:
                scale = input['depth_scale']
                shift = input['depth_shift']
            depth_values_copy = torch.zeros_like(points[:, 0]).float().cuda()
            depth_values_copy[surface_mask] = depth_values[surface_mask] * scale + shift
            depth_values = depth_values_copy
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], others = self.get_rbg_value(differentiable_surface_points, network_condition, deformer_condition, pose_feature, expression, transformations, is_training=self.training,
                                                                  jaw_pose=torch.cat([expression, flame_pose[:, 6:9]], dim=1),
                                                                  latent_code=latent_code, 
                                                                  transl=flame_transl,
                                                                  deform_params=deform_params
                                                                  )
            normal_values[surface_mask] = others['normals']
            # if self.rendering_network.mode == 'spherical_harmonics':
            #     shading_values[surface_mask] = others['shading']
            #     alebdo_values[surface_mask] = others['albedo']
            
        flame_distance_values = torch.zeros(points.shape[0]).float().cuda()
        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # canonical_verts, _, _ = \
        # self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp,
        #                 full_pose=self.FLAMEServer.canonical_pose,
        #                 scale=flame_scale)
        # knn_v = canonical_verts.clone()

        knn_v = verts.clone()
        flame_distance, index_batch, _ = ops.knn_points(differentiable_surface_points.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        index_batch_values = torch.ones(points.shape[0]).long().cuda()
        index_batch_values[surface_mask] = index_batch
        flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)
        
        # flame_distance_values = torch.zeros(points.shape[0]).float().cuda()
        # knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        # flame_distance, index_batch, _ = ops.knn_points(differentiable_surface_points.unsqueeze(0), knn_v, K=1, return_nn=True)
        # index_batch = index_batch.reshape(-1)
        # index_batch_values = torch.ones(points.shape[0]).long().cuda()
        # index_batch_values[surface_mask] = index_batch
        # flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'normal_values': normal_values,
            'depth_values': depth_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'hand_mask': hand_mask,
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
            'flame_deform_dir': self.FLAMEServer.deform_dir,
            'nonrigid_deformation': nonrigid_deformation,
            'verts': verts
            # 'flame_posedirs': self.FLAMEServer.posedirs * flame_scale.reshape(-1),
            # 'flame_shapedirs': self.FLAMEServer.shapedirs * flame_scale.reshape(-1)
        }
        if 'depth' in input:
            output['depth_scale'] = scale
            output["depth_shift"] = shift
            
        # if lbs_weight is not None:
        #     skinning_values = torch.ones(points.shape[0], 6 if self.ghostbone else 5).float().cuda()
        #     skinning_values[surface_mask] = lbs_weight
        #     output['lbs_weight'] = skinning_values
        # if posedirs is not None:
        #     posedirs_values = torch.ones(points.shape[0], 36, 3).float().cuda()
        #     posedirs_values[surface_mask] = posedirs
        #     output['posedirs'] = posedirs_values
        # if shapedirs is not None:
        #     shapedirs_values = torch.ones(points.shape[0], 3, 50).float().cuda()
        #     shapedirs_values[surface_mask] = shapedirs
        #     output['shapedirs'] = shapedirs_values
            
        if lbs_weight is not None and gt_lbs_weight is not None:
            # skinning_values = torch.ones_like(points).float().cuda()
            # gt_skinning_values = torch.ones_like(points).float().cuda()
            skinning_values = torch.ones(points.shape[0], 6 if self.ghostbone else 5).float().cuda()
            gt_skinning_values = torch.ones(points.shape[0], 6 if self.ghostbone else 5).float().cuda()
            skinning_values[surface_mask] = lbs_weight#[..., :3]
            gt_skinning_values[surface_mask] = gt_lbs_weight#[..., :3]
            output['lbs_weight'] = skinning_values
            
            # if self.gt_w_seg:
            #     hair = semantics[:, :, 6].reshape(-1) == 1
            #     gt_skinning_values[hair, :] = 0.
            #     gt_skinning_values[hair, 2 if ghostbone else 1] = 1.
            # if self.ghostbone and self.gt_w_seg:
            #     # cloth deforms with ghost bone (identity)
            #     cloth = semantics[:, :, 7].reshape(-1) == 1
            #     gt_skinning_values[cloth, :] = 0.
            #     gt_skinning_values[cloth, 0] = 1.
            output['gt_lbs_weight'] = gt_skinning_values
        if posedirs is not None and gt_posedirs is not None:
            posedirs_values = torch.ones(points.shape[0], 36, 3).float().cuda()
            gt_posedirs_values = torch.ones(points.shape[0], 36, 3).float().cuda()
            posedirs_values[surface_mask] = posedirs
            gt_posedirs_values[surface_mask] = gt_posedirs
            output['posedirs'] = posedirs_values
            
            # if self.gt_w_seg:
            #     # mouth interior and eye glasses doesn't deform
            #     mouth = semantics[:, :, 3].reshape(-1) == 1
            #     gt_posedirs_values[mouth, :] = 0.0
            output['gt_posedirs'] = gt_posedirs_values
        if shapedirs is not None and gt_shapedirs is not None:
            shapedirs_values = torch.ones(points.shape[0], 3, 50).float().cuda()
            gt_shapedirs_values = torch.ones(points.shape[0], 3, 50).float().cuda()
            shapedirs_values[surface_mask] = shapedirs
            gt_shapedirs_values[surface_mask] = gt_shapedirs
            output['shapedirs'] = shapedirs_values
            
            # disable_shapedirs_for_mouth_and_cloth = False
            # if disable_shapedirs_for_mouth_and_cloth:
            #     # I accidentally deleted these when cleaning the code...
            #     # So this is why I don't see teeth anymore...QAQ
            #     # Most of suppmat experiments used this code block, but it doesn't necessarily help in all cases.
            #     if self.gt_w_seg:
            #         # mouth interior and eye glasses doesn't deform
            #         mouth = semantics[:, :, 3].reshape(-1) == 1
            #         gt_shapedirs_values[mouth, :] = 0.0
            #     if self.ghostbone and self.gt_w_seg:
            #         # cloth doesn't deform with facial expressions
            #         cloth = semantics[:, :, 7].reshape(-1) == 1
            #         gt_shapedirs_values[cloth, :] = 0.
            output['gt_shapedirs'] = gt_shapedirs_values
        if deform_dir is not None and gt_deform_dir is not None:
            deform_dir_values = torch.ones(points.shape[0], 3, 30).float().cuda()
            gt_deform_dir_values = torch.ones(points.shape[0], 3, 30).float().cuda()
            deform_dir_values[surface_mask] = deform_dir
            gt_deform_dir_values[surface_mask] = gt_deform_dir
            output['deform_dir'] = deform_dir_values
            
            output['gt_deform_dir'] = gt_deform_dir_values    
        
        # if displacement is not None and gt_displacement is not None:
        #     displacement_values = torch.ones(points.shape[0], 3).float().cuda()
        #     gt_displacement_values = torch.ones(points.shape[0], 3).float().cuda()
        #     displacement_values[surface_mask] = displacement
        #     gt_displacement_values[surface_mask] = gt_displacement
        #     output['displacement'] = displacement_values
        #     output['gt_displacement'] = gt_displacement_values

            # flame_distance_values = torch.ones(points.shape[0]).float().cuda()
            # flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)
            # output['flame_distance'] = flame_distance_values

        # if self.rendering_network.mode == 'spherical_harmonics':
        #     output['shading_values'] = shading_values
        #     output['albedo_values'] = alebdo_values
        
        ###################################################### HAND ######################################################
        # Left hand params
        mano_left_global_orient = input["mano_left_global_orient"]
        mano_left_hand_pose = input["mano_left_hand_pose"].reshape(-1, 15, 3, 3)
        mano_left_betas = input["mano_left_betas"]
        mano_left_transl = input["mano_left_transl"]
        mano_left_scale = input["mano_left_scale"]

        # Right hand params
        mano_right_global_orient = input["mano_right_global_orient"]
        mano_right_hand_pose = input["mano_right_hand_pose"].reshape(-1, 15, 3, 3)
        mano_right_betas = input["mano_right_betas"]
        mano_right_transl = input["mano_right_transl"]
        mano_right_scale = input["mano_right_scale"]

        w2c_p = input["w2c_p"]
        cam_intrinsics = input["cam_intrinsics"].squeeze(0)

        # Predicted MANO params (left)
        pred_mano_left_params = {
            'global_orient': mano_left_global_orient,
            'hand_pose': mano_left_hand_pose,
            'betas': mano_left_betas,
            'transl': mano_left_transl,
            'scale': mano_left_scale
        }

        # Predicted MANO params (right)
        pred_mano_right_params = {
            'global_orient': mano_right_global_orient,
            'hand_pose': mano_right_hand_pose,
            'betas': mano_right_betas,
            'transl': mano_right_transl,
            'scale': mano_right_scale
        }

        # Forward pass for left and right hands
        mano_left_output = self.MANOServer(**{k: v.float() for k, v in pred_mano_left_params.items()}, pose2rot=False)
        mano_right_output = self.MANOServer(**{k: v.float() for k, v in pred_mano_right_params.items()}, pose2rot=False)

        verts_left_hand = mano_left_output.vertices.clone()
        verts_right_hand = mano_right_output.vertices.clone()
        
        landmarks3d_p_left_hand = mano_left_output.joints.clone()
        landmarks3d_p_right_hand = mano_right_output.joints.clone()
        
        verts_left_hand[:,:,0] = -1*verts_left_hand[:,:,0]
        landmarks3d_p_left_hand[:,:,0] = -1*landmarks3d_p_left_hand[:,:,0]
        
 
        with torch.no_grad():
            # Left hand SDF function
            sdf_function_left_hand = lambda x, use_verts=False: self.query_sdf_hand_left(
                pnts_p=x,
                verts=verts_left_hand,
            )[0]

            # Right hand SDF function
            sdf_function_right_hand = lambda x, use_verts=False: self.query_sdf_hand(
                pnts_p=x,
                verts=verts_right_hand,
            )[0]

            # Trace rays for left hand
            points_left_hand, network_object_mask_left_hand, dists_left_hand = self.ray_tracer.forward(
                sdf=sdf_function_left_hand,
                cam_loc=cam_loc,
                object_mask=left_hand_mask,
                ray_directions=ray_dirs
            )

            # Trace rays for right hand
            points_right_hand, network_object_mask_right_hand, dists_right_hand = self.ray_tracer.forward(
                sdf=sdf_function_right_hand,
                cam_loc=cam_loc,
                object_mask=right_hand_mask,
                ray_directions=ray_dirs
            )

        # Compute 3D point locations in camera space (left hand)
        points_left_hand = (
            cam_loc.unsqueeze(1) + 
            dists_left_hand.reshape(batch_size, num_pixels, 1) * ray_dirs
        ).reshape(-1, 3)

        # Compute 3D point locations in camera space (right hand)
        points_right_hand = (
            cam_loc.unsqueeze(1) + 
            dists_right_hand.reshape(batch_size, num_pixels, 1) * ray_dirs
        ).reshape(-1, 3)


        # Query SDF at traced points (left hand)
        sdf_output_left_hand, canonical_points_left_hand, _, others_left_hand = self.query_sdf_hand_left(
            points_left_hand, verts=verts_left_hand
        )

        # Query SDF at traced points (right hand)
        sdf_output_right_hand, canonical_points_right_hand, _, others_right_hand = self.query_sdf_hand(
            points_right_hand, verts=verts_right_hand
        )

        # Store outputs
        output['sdf_output_left_hand'] = sdf_output_left_hand
        output['sdf_output_right_hand'] = sdf_output_right_hand
        
        
        # Left Hand Rendering
        trans_verts_left_hand = projection(verts_left_hand, cam_intrinsics, w2c_p)
        trans_verts_left_hand_cam = projection(verts_left_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
        depth_image_left_hand, mask_image_left_hand = self.deca.render_hand.render_depth(
            trans_verts_left_hand, trans_verts_left_hand_cam
        )
        depth_image_left_hand = depth_image_left_hand.permute(0, 2, 3, 1)
        mask_image_left_hand = mask_image_left_hand.permute(0, 2, 3, 1)

        pred_color_masked_left, normal_image_left_hand = self.deca.render_hand_left.render_neural_colors(
            self.shader, verts_left_hand, trans_verts_left_hand
        )

        landmarks3d_p_left_hand = mano_left_output.joints.clone()
        trans_landmarks2d_left_hand = projection(landmarks3d_p_left_hand, cam_intrinsics, w2c_p)

        # Right Hand Rendering
        trans_verts_right_hand = projection(verts_right_hand, cam_intrinsics, w2c_p)
        trans_verts_right_hand_cam = projection(verts_right_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
        depth_image_right_hand, mask_image_right_hand = self.deca.render_hand.render_depth(
            trans_verts_right_hand, trans_verts_right_hand_cam
        )
        depth_image_right_hand = depth_image_right_hand.permute(0, 2, 3, 1)
        mask_image_right_hand = mask_image_right_hand.permute(0, 2, 3, 1)

        pred_color_masked_right, normal_image_right_hand = self.deca.render_hand.render_neural_colors(
            self.shader, verts_right_hand, trans_verts_right_hand
        )

        landmarks3d_p_right_hand = mano_right_output.joints.clone()
        trans_landmarks2d_right_hand = projection(landmarks3d_p_right_hand, cam_intrinsics, w2c_p)


        # Process depth and masks for left hand
        depth_image_left_hand = depth_image_left_hand.clone().reshape(-1)
        gt_depth_image_left_hand = input["depth_image_left_hand"].clone().reshape(-1)
        mask_image_left_hand = mask_image_left_hand.clone().reshape(-1).bool()
        gbuffer_mask_left = mask_image_left_hand.clone()

        pred_color_masked_left = pred_color_masked_left.reshape(-1, 3)
        normal_image_left_hand = normal_image_left_hand.reshape(-1, 3)
        depth_image_left_hand = depth_image_left_hand.reshape(-1)
        
        # normal_image_left_hand[:,0] = -1 * normal_image_left_hand[:,0]
        # normal_image_left_hand[:,1] = 1 - normal_image_left_hand[:,1]
        # normal_image_left_hand[:,2] = 1 - normal_image_left_hand[:,2]
        # normal_image_left_hand = 1 - normal_image_left_hand

        # Process depth and masks for right hand
        depth_image_right_hand = depth_image_right_hand.clone().reshape(-1)
        gt_depth_image_right_hand = input["depth_image_right_hand"].clone().reshape(-1)
        mask_image_right_hand = mask_image_right_hand.clone().reshape(-1).bool()
        gbuffer_mask_right = mask_image_right_hand.clone()

        pred_color_masked_right = pred_color_masked_right.reshape(-1, 3)
        normal_image_right_hand = normal_image_right_hand.reshape(-1, 3)
        depth_image_right_hand = depth_image_right_hand.reshape(-1)


        # Apply scale/shift
        depth_image_left_hand = depth_image_left_hand * scale + shift
        depth_image_right_hand = depth_image_right_hand * scale + shift


        # Indexing logic (if needed)
        if 'split_indx' in input.keys():
            indx = input['split_indx']
            gbuffer_mask_left = torch.index_select(gbuffer_mask_left, 0, indx)
            pred_color_masked_left = torch.index_select(pred_color_masked_left, 0, indx)
            normal_image_left_hand = torch.index_select(normal_image_left_hand, 0, indx)
            depth_image_left_hand = torch.index_select(depth_image_left_hand, 0, indx)

            gbuffer_mask_right = torch.index_select(gbuffer_mask_right, 0, indx)
            pred_color_masked_right = torch.index_select(pred_color_masked_right, 0, indx)
            normal_image_right_hand = torch.index_select(normal_image_right_hand, 0, indx)
            depth_image_right_hand = torch.index_select(depth_image_right_hand, 0, indx)

        if 'sampling_idx' in input.keys():
            sampling_idx = input["sampling_idx"].reshape(-1)
            bbox_inside = input["bbox_inside"].reshape(-1)
            sampling_idx_bbox = input["sampling_idx_bbox"].reshape(-1)

            gbuffer_mask_left = torch.cat([gbuffer_mask_left[sampling_idx], gbuffer_mask_left[bbox_inside][sampling_idx_bbox]], 0)
            gbuffer_mask_right = torch.cat([gbuffer_mask_right[sampling_idx], gbuffer_mask_right[bbox_inside][sampling_idx_bbox]], 0)


        # Final network object mask per hand
        network_object_mask_left_hand = gbuffer_mask_left.clone().bool() #& left_hand_mask
        surface_mask_left_hand = network_object_mask_left_hand & left_hand_mask if self.training else network_object_mask_left_hand

        network_object_mask_right_hand = gbuffer_mask_right.clone().bool() #& right_hand_mask
        surface_mask_right_hand = network_object_mask_right_hand & right_hand_mask if self.training else network_object_mask_right_hand


        # Initialize RGB, Normal, Depth values
        # rgb_values_left_hand = torch.ones_like(points_left_hand).float().cuda()
        normal_values_left_hand = torch.zeros_like(points_left_hand).float().cuda()
        # depth_values_left_hand = torch.zeros_like(points_left_hand[:, 0]).unsqueeze(-1).float().cuda()

        # rgb_values_right_hand = torch.ones_like(points_right_hand).float().cuda()
        normal_values_right_hand = torch.zeros_like(points_right_hand).float().cuda()
        # depth_values_right_hand = torch.zeros_like(points_right_hand[:, 0]).unsqueeze(-1).float().cuda()

        # depth_image_left_hand = depth_image_left_hand.unsqueeze(-1)
        # depth_image_right_hand = depth_image_right_hand.unsqueeze(-1)
        
        # Populate based on surface mask
        if 'sampling_idx' in input.keys():
            sampling_idx = input["sampling_idx"].reshape(-1)
            bbox_inside = input["bbox_inside"].reshape(-1)
            sampling_idx_bbox = input["sampling_idx_bbox"].reshape(-1)

            # rgb_values_left_hand[surface_mask_left_hand] = torch.cat([
            #     pred_color_masked_left[sampling_idx],
            #     pred_color_masked_left[bbox_inside][sampling_idx_bbox, :]
            # ], 0)[surface_mask_left_hand]

            normal_values_left_hand[surface_mask_left_hand] = torch.cat([
                normal_image_left_hand[sampling_idx],
                normal_image_left_hand[bbox_inside][sampling_idx_bbox, :]
            ], 0)[surface_mask_left_hand]

            # depth_values_left_hand[surface_mask_left_hand] = torch.cat([
            #     depth_image_left_hand[sampling_idx],
            #     depth_image_left_hand[bbox_inside][sampling_idx_bbox]
            # ], 0)[surface_mask_left_hand]


            # rgb_values_right_hand[surface_mask_right_hand] = torch.cat([
            #     pred_color_masked_right[sampling_idx],
            #     pred_color_masked_right[bbox_inside][sampling_idx_bbox, :]
            # ], 0)[surface_mask_right_hand]

            normal_values_right_hand[surface_mask_right_hand] = torch.cat([
                normal_image_right_hand[sampling_idx],
                normal_image_right_hand[bbox_inside][sampling_idx_bbox, :]
            ], 0)[surface_mask_right_hand]

            # depth_values_right_hand[surface_mask_right_hand] = torch.cat([
            #     depth_image_right_hand[sampling_idx],
            #     depth_image_right_hand[bbox_inside][sampling_idx_bbox]
            # ], 0)[surface_mask_right_hand]

        else:
            # rgb_values_left_hand[surface_mask_left_hand] = pred_color_masked_left[surface_mask_left_hand]
            normal_values_left_hand[surface_mask_left_hand] = normal_image_left_hand[surface_mask_left_hand]
            # depth_values_left_hand[surface_mask_left_hand] = depth_image_left_hand[surface_mask_left_hand]

            # rgb_values_right_hand[surface_mask_right_hand] = pred_color_masked_right[surface_mask_right_hand]
            normal_values_right_hand[surface_mask_right_hand] = normal_image_right_hand[surface_mask_right_hand]
            # depth_values_right_hand[surface_mask_right_hand] = depth_image_right_hand[surface_mask_right_hand]

        # # Output dictionary updates
        # output.update({
        #     # Left hand outputs
        #     'points_left_hand': points_left_hand,
        #     'rgb_values_left_hand': rgb_values_left,
        #     'normal_values_left_hand': normal_values_left,
        #     'depth_values_left_hand': depth_values_left,
        #     'sdf_output_left_hand': sdf_output_left_hand,
        #     'network_object_mask_left_hand': network_object_mask_left_hand,
        #     'left_hand_mask': left_hand_mask,
        #     'left_hand_pose': mano_left_hand_pose,
        #     'rgb_image_left_hand': pred_color_masked_left,
        #     'normal_image_left_hand': normal_image_left_hand,
        #     'depth_image_left_hand': depth_image_left_hand,
        #     'mask_image_left_hand': mask_image_left_hand,
        #     'hand_lmk_left': trans_landmarks2d_left_hand,
        #     'mano_left_transl': mano_left_transl,

        #     # Right hand outputs
        #     'points_right_hand': points_right_hand,
        #     'rgb_values_right_hand': rgb_values_right,
        #     'normal_values_right_hand': normal_values_right,
        #     'depth_values_right_hand': depth_values_right,
        #     'sdf_output_right_hand': sdf_output_right_hand,
        #     'network_object_mask_right_hand': network_object_mask_right_hand,
        #     'right_hand_mask': right_hand_mask,
        #     'right_hand_pose': mano_right_hand_pose,
        #     'rgb_image_right_hand': pred_color_masked_right,
        #     'normal_image_right_hand': normal_image_right_hand,
        #     'depth_image_right_hand': depth_image_right_hand,
        #     'mask_image_right_hand': mask_image_right_hand,
        #     'hand_lmk_right': trans_landmarks2d_right_hand,
        #     'mano_right_transl': mano_right_transl,
        # })
        
        # Assume batch_size and num_samples are known
        batch_size = mano_left_hand_pose.shape[0]
        num_samples = int(points_left_hand.shape[0] // batch_size)
        total_samples = num_samples  # or define as needed

        # Initialize empty buffers
        points_hand = torch.zeros(batch_size * total_samples, 3).cuda().float()
        # rgb_values_hand = torch.zeros(batch_size * total_samples, 3).cuda().float()
        normal_values_hand = torch.zeros(batch_size * total_samples, 3).cuda().float()
        # depth_values_hand = torch.zeros(batch_size * total_samples, 1).cuda().float()

        # Fill in values using network_object_mask for each hand
        points_hand[network_object_mask_left_hand] = points_left_hand[network_object_mask_left_hand]
        points_hand[network_object_mask_right_hand] = points_right_hand[network_object_mask_right_hand]

        # rgb_values_hand[network_object_mask_left_hand] = rgb_values_left_hand[network_object_mask_left_hand]
        # rgb_values_hand[network_object_mask_right_hand] = rgb_values_right_hand[network_object_mask_right_hand]

        normal_values_hand[network_object_mask_left_hand] = normal_values_left_hand[network_object_mask_left_hand]
        normal_values_hand[network_object_mask_right_hand] = normal_values_right_hand[network_object_mask_right_hand]

        # depth_values_hand[network_object_mask_left_hand] = depth_values_left_hand[network_object_mask_left_hand]
        # depth_values_hand[network_object_mask_right_hand] = depth_values_right_hand[network_object_mask_right_hand]

        # Combine masks
        network_object_mask_hand = network_object_mask_left_hand | network_object_mask_right_hand
        hand_mask = left_hand_mask | right_hand_mask  # assuming left_hand_mask/right_hand_mask are boolean tensors

        # Optional: reshape if needed
        points_hand = points_hand.reshape(batch_size, total_samples, 3)
        # rgb_values_hand = rgb_values_hand.reshape(batch_size, total_samples, 3)
        normal_values_hand = normal_values_hand.reshape(batch_size, total_samples, 3)
        # depth_values_hand = depth_values_hand.reshape(batch_size, total_samples, 1)

        # Assign to output dict
        output['points_hand'] = points_hand
        # output['rgb_values_hand'] = rgb_values_hand
        output['normal_values_hand'] = normal_values_hand
        # output['depth_values_hand'] = depth_values_hand
        output['network_object_mask_hand'] = network_object_mask_hand
        output['hand_mask'] = hand_mask
        
        output['sdf_output_left_hand'] = sdf_output_left_hand
        output['sdf_output_right_hand'] = sdf_output_right_hand

        # output['rgb_image_hand'] = rgb_image_hand
        # output['normal_image_hand'] = normal_image_hand
        # output['depth_image_hand'] = depth_image_hand
        # output['mask_image_hand'] = mask_image_hand
        # output['hand_lmk'] = hand_lmk
                
        # output['left_hand_pose'] = mano_left_hand_pose
        # output['mano_left_transl'] = mano_left_transl
        # output['right_hand_pose'] = mano_right_hand_pose
        # output['mano_right_transl'] = mano_right_transl
        
        output['img_res'] = input['img_res']
        # output['verts_hand_tips'] = verts_hand_tips
        
        output['verts_hand_left'] = verts_left_hand
        output['verts_hand_right'] = verts_right_hand
        
        ##################################################################################################################
            
        if "depth" in input:
            max_depth = 6.0
            min_depth = 3.0
            batch_size = pose.shape[0]
            num_samples_head = int(rgb_values.shape[0] / batch_size)
            depth_head = torch.ones(batch_size * num_samples_head).cuda().float() * max_depth
            depth_head[network_object_mask] = rend_util.get_depth(points.reshape(batch_size, num_samples_head, 3), pose).reshape(-1)[network_object_mask]
            depth_head = (depth_head.reshape(batch_size, num_samples_head, 1) - min_depth) / (max_depth - min_depth)

            # num_samples_hand = int(rgb_values_hand.shape[0]*rgb_values_hand.shape[1] / batch_size)
            num_samples_hand = int(normal_values_hand.shape[0]*normal_values_hand.shape[1] / batch_size)
            depth_hand = torch.ones(batch_size * num_samples_hand).cuda().float() * max_depth
            depth_hand[network_object_mask_hand] = rend_util.get_depth(points_hand.reshape(batch_size, num_samples_hand, 3), pose).reshape(-1)[network_object_mask_hand]
            depth_hand = (depth_hand.reshape(batch_size, num_samples_hand, 1) - min_depth) / (max_depth - min_depth)

            # depth_mask_head = (depth_values_hand > depth_values_head).int() # head before hand
            # depth_mask_hand = (depth_values_hand < depth_values_head).int() # hand before head
            depth_head = depth_head.reshape(-1)
            depth_hand = depth_hand.reshape(-1)
            depth_head_mask = (depth_head < depth_hand).bool()
            depth_hand_mask = (depth_head > depth_hand).bool()
            output['depth_mask_head'] = depth_head_mask
            output['depth_mask_hand'] = depth_hand_mask
        
        if self.training:
            optimize_contact = input['optimize_contact']
            output['optimize_contact'] = optimize_contact
        
            output['surface_output'] = surface_output
    
            # with torch.no_grad():
            #     sdf_output_headsurf_tohand, _, _, _ = self.query_sdf_hand(points, verts=verts_hand)
            #     output['sdf_output_headsurf_tohand'] = sdf_output_headsurf_tohand #[surface_mask]
            
            with torch.no_grad():
                sdf_output_headsurf_toleft, _, _, _ = self.query_sdf_hand(points, verts=verts_left_hand)
                output['sdf_output_headsurf_toleft'] = sdf_output_headsurf_toleft #[surface_mask]
                
                sdf_output_headsurf_toright, _, _, _ = self.query_sdf_hand(points, verts=verts_right_hand)
                output['sdf_output_headsurf_toright'] = sdf_output_headsurf_toright #[surface_mask]
            
            # sdf_output_handsurf_tohead, _, _, others = self.query_sdf(points_hand, network_condition, deformer_condition,
            #                                                         pose_feature=pose_feature, betas=expression,
            #                                                         transformations=transformations, 
            #                                                         transl=flame_transl, 
            #                                                         scale=None,
            #                                                         deform_params=deform_params)
            # output['sdf_output_handsurf_tohead'] = sdf_output_handsurf_tohead
            # output['nonrigid_deformation_handsurf_tohead'] = others['nonrigid_deformation']
            
            
            # meshes = Meshes(verts_hand.reshape(1,-1,3), self.mano_faces.reshape(1,-1,3))
            # samples = sample_points_from_meshes(meshes, num_samples=3000, return_normals=False, return_textures=False)
            # samples = samples.reshape(-1,3)
            # sdf_output_handmesh_tohead, _, _, others = self.query_sdf(samples, network_condition, deformer_condition,
            #                                                         pose_feature=pose_feature, betas=expression,
            #                                                         transformations=transformations, 
            #                                                         transl=flame_transl, 
            #                                                         scale=None,
            #                                                         deform_params=deform_params)
            # output['sdf_output_handmesh_tohead'] = sdf_output_handmesh_tohead
            # output['nonrigid_deformation_handmesh_tohead'] = others['nonrigid_deformation']
            
            with torch.no_grad():
                # SDF from head surface to left hand
                sdf_output_headsurf_toleft, _, _, _ = self.query_sdf_hand_left(points, verts=verts_left_hand)
                output['sdf_output_headsurf_toleft'] = sdf_output_headsurf_toleft  # [surface_mask]

                # SDF from head surface to right hand
                sdf_output_headsurf_toright, _, _, _ = self.query_sdf_hand(points, verts=verts_right_hand)
                output['sdf_output_headsurf_toright'] = sdf_output_headsurf_toright  # [surface_mask]


            # SDF from hand surface to head (left hand)
            sdf_output_leftsurf_tohead, _, _, others_left = self.query_sdf(
                points_left_hand, network_condition, deformer_condition,
                pose_feature=pose_feature, betas=expression,
                transformations=transformations,
                transl=flame_transl,
                scale=None,
                deform_params=deform_params
            )
            output['sdf_output_leftsurf_tohead'] = sdf_output_leftsurf_tohead
            output['nonrigid_deformation_leftsurf_tohead'] = others_left['nonrigid_deformation']

            # SDF from hand surface to head (right hand)
            sdf_output_rightsurf_tohead, _, _, others_right = self.query_sdf(
                points_right_hand, network_condition, deformer_condition,
                pose_feature=pose_feature, betas=expression,
                transformations=transformations,
                transl=flame_transl,
                scale=None,
                deform_params=deform_params
            )
            output['sdf_output_rightsurf_tohead'] = sdf_output_rightsurf_tohead
            output['nonrigid_deformation_rightsurf_tohead'] = others_right['nonrigid_deformation']


            # Sample points from left hand mesh
            meshes_left = Meshes(verts_left_hand.reshape(1, -1, 3), self.mano_faces_left.reshape(1, -1, 3))
            samples_left = sample_points_from_meshes(meshes_left, num_samples=3000, return_normals=False, return_textures=False)
            samples_left = samples_left.reshape(-1, 3)

            # SDF from left hand mesh to head
            sdf_output_leftmesh_tohead, _, _, others_leftmesh = self.query_sdf(
                samples_left, network_condition, deformer_condition,
                pose_feature=pose_feature, betas=expression,
                transformations=transformations,
                transl=flame_transl,
                scale=None,
                deform_params=deform_params
            )
            output['sdf_output_leftmesh_tohead'] = sdf_output_leftmesh_tohead
            output['nonrigid_deformation_leftmesh_tohead'] = others_leftmesh['nonrigid_deformation']


            # Sample points from right hand mesh
            meshes_right = Meshes(verts_right_hand.reshape(1, -1, 3), self.mano_faces.reshape(1, -1, 3))
            samples_right = sample_points_from_meshes(meshes_right, num_samples=3000, return_normals=False, return_textures=False)
            samples_right = samples_right.reshape(-1, 3)

            # SDF from right hand mesh to head
            sdf_output_rightmesh_tohead, _, _, others_rightmesh = self.query_sdf(
                samples_right, network_condition, deformer_condition,
                pose_feature=pose_feature, betas=expression,
                transformations=transformations,
                transl=flame_transl,
                scale=None,
                deform_params=deform_params
            )
            output['sdf_output_rightmesh_tohead'] = sdf_output_rightmesh_tohead
            output['nonrigid_deformation_rightmesh_tohead'] = others_rightmesh['nonrigid_deformation']

            ### only head
            implicit_sdf, flame_sdf = self.query_sdf_flame(network_condition)
            output['implicit_sdf'] = implicit_sdf
            output['flame_sdf'] = flame_sdf
            
            meshes = Meshes(verts.reshape(1,-1,3), self.FLAMEServer.faces_tensor.reshape(1,-1,3))
            samples = sample_points_from_meshes(meshes, num_samples=3000, return_normals=False, return_textures=False)
            samples = samples.reshape(-1,3)
            
            # Add Gaussian noise scaled to 1e-1
            noise = torch.randn_like(samples) * 1e-2
            samples = samples + noise.to(samples.device)
            
            sdf_output_reg, _, _, others = self.query_sdf(samples, network_condition, deformer_condition,
                                                            pose_feature=pose_feature, betas=expression,
                                                            transformations=transformations, 
                                                            transl=flame_transl, 
                                                            scale=None,
                                                            deform_params=deform_params)
            output['sdf_output_reg'] = sdf_output_reg
            output['nonrigid_deformation_reg'] = others['nonrigid_deformation']
            with torch.no_grad():
                sdf_output_reg_toleft, _, _, _ = self.query_sdf_hand(samples, verts=verts_left_hand)
                output['sdf_output_reg_toleft'] = sdf_output_reg_toleft
                
                sdf_output_reg_toright, _, _, _ = self.query_sdf_hand(samples, verts=verts_right_hand)
                output['sdf_output_reg_toright'] = sdf_output_reg_toright
        
        # ## vis contact points
        # if True:
        #     # optimize_contact = input['optimize_contact']
        #     # output['optimize_contact'] = optimize_contact
        
        #     # output['surface_output'] = surface_output
    
        #     # with torch.no_grad():
        #     #     sdf_output_headsurf_tohand, _, _, _ = self.query_sdf_hand(points, verts=verts_hand)
        #     #     output['sdf_output_headsurf_tohand'] = sdf_output_headsurf_tohand #[surface_mask]
            
        #     # sdf_output_handsurf_tohead, _, _, others = self.query_sdf(points_hand, network_condition, deformer_condition,
        #     #                                                         pose_feature=pose_feature, betas=expression,
        #     #                                                         transformations=transformations, 
        #     #                                                         transl=flame_transl, 
        #     #                                                         scale=None,
        #     #                                                         deform_params=deform_params)
        #     # output['sdf_output_handsurf_tohead'] = sdf_output_handsurf_tohead
        #     # output['nonrigid_deformation_handsurf_tohead'] = others['nonrigid_deformation']
            
            
        #     meshes = Meshes(verts_hand.reshape(1,-1,3), self.mano_faces.reshape(1,-1,3))
        #     samples = sample_points_from_meshes(meshes, num_samples=100000, return_normals=False, return_textures=False)
        #     samples = samples.reshape(-1,3)
        #     sdf_output_handmesh_tohead, _, _, others = self.query_sdf(samples, network_condition, deformer_condition,
        #                                                             pose_feature=pose_feature, betas=expression,
        #                                                             transformations=transformations, 
        #                                                             transl=flame_transl, 
        #                                                             scale=None,
        #                                                             deform_params=deform_params)
        #     # sdf_output_handmesh_tohead, _, _, others = self.query_sdf(verts_hand.reshape(-1,3), network_condition, deformer_condition,
        #     #                                                         pose_feature=pose_feature, betas=expression,
        #     #                                                         transformations=transformations, 
        #     #                                                         transl=flame_transl, 
        #     #                                                         scale=None,
        #     #                                                         deform_params=deform_params)
        #     # output['sdf_output_handmesh_tohead'] = sdf_output_handmesh_tohead
        #     # output['nonrigid_deformation_handmesh_tohead'] = others['nonrigid_deformation']
            
        #     ### save vis contact point
        #     colors = torch.tensor([
        #         [0.5, 0.5, 0.5],  # Green
        #     ], dtype=torch.float32).repeat(samples.shape[0],1).float().cuda()
            
        #     red_color = torch.tensor([
        #         [1, 0, 0],  # Red
        #     ], dtype=torch.float32).float().cuda()

        #     if_contact = (sdf_output_handmesh_tohead < 0)
        #     colors[if_contact] = red_color
        #     # Save the points with colors to a .ply file
        #     from pytorch3d.io import save_ply
        #     save_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/Emory/IMavatar_Emory_hc_pca_10_500000_handmesh_sample5000_fist_flamesdfloss01/rgb/eval/rgb/epoch_60/contact_points'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     import open3d as o3d
        #     point_cloud = o3d.geometry.PointCloud()
        #     point_cloud.points = o3d.utility.Vector3dVector(samples.detach().cpu().numpy())
        #     point_cloud.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy()*255)
        #     img_name = input["img_name"]
        #     o3d.io.write_point_cloud(os.path.join(save_dir, "contact_points_colored_%06d.ply"%int(img_name)), point_cloud)
    
    
    
    
    
    
    
        #     # import trimesh
        #     # mesh = trimesh.Trimesh(vertices=verts_hand.reshape(-1,3).detach().cpu().numpy(),
        #     #                        faces=[[0, 1, 2]])
        #     # ray_mesh = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)                                 
        #     # locations, index_ray, index_tri = ray_mesh.intersects_location(
        #     #     ray_origin,
        #     #     ray_direction
        #     # )
            
        #     # import bvh_ray_tracing
        #     # m = bvh_ray_tracing.BVH()
        #     # triangles = verts_hand.reshape(-1,3)[self.mano_faces.reshape(-1,3)]
        #     # distances, closest_points, closest_faces, closest_bcs = m(triangles, cam_loc.reshape(-1,3).contiguous(), ray_dirs.reshape(-1,3).contiguous())
        #     # # bvh = bvh_ray_tracing.BVHBuilder(verts_hand.reshape(-1,3), self.mano_faces.reshape(-1,3))
        #     # # bvh.build()
        #     # # intersection_points = bvh.ray_trace(cam_loc.reshape(-1,3), ray_dirs.reshape(-1,3))
        #     # breakpoint()
            
        
                
        if not return_sdf:
            return output
        else:
            return output, sdf_function, sdf_function_left_hand, sdf_function_right_hand

    def get_rbg_value(self, points, network_condition, deformer_condition, pose_feature, betas, transformations, jaw_pose=None, latent_code=None, is_training=True, transl=None, scale=None, deform_params=None):
        others = {}
        if self.deformer_class != 'ForwardDeformer':
            points.requires_grad_(True)
            sdf, pnts_c, feature_vectors, others = self.query_sdf(points, network_condition, deformer_condition, pose_feature, betas, transformations, transl=transl, scale=scale, deform_params=deform_params)
            gradients = self.gradient(points, network_condition, deformer_condition, pose_feature, betas, transformations, sdf=sdf, create_graph=is_training, retain_graph=is_training, transl=transl, scale=scale, deform_params=deform_params)
        else:
            pnts_c = points
            # others = {}
            _, gradients, feature_vectors = self.forward_gradient(pnts_c, network_condition, pose_feature, betas, transformations, create_graph=is_training, retain_graph=is_training)

        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        # rgb_vals, rendering_others = self.rendering_network(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose, latent_code=latent_code)
        rgb_vals = self.rendering_network(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose)

        others['normals'] = normals
        # for k, v in rendering_others.items():
        #     others[k] = v
        return rgb_vals, others

    def gradient(self, x, network_condition, deformer_condition, pose_feature, betas, transformations, sdf=None, create_graph=True, retain_graph=True, transl=None, scale=None, deform_params=None):
        x.requires_grad_(True)
        if sdf is None:
            y = self.query_sdf(x, network_condition, deformer_condition, pose_feature, betas, transformations, transl=transl, scale=scale, deform_params=deform_params)[0]
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