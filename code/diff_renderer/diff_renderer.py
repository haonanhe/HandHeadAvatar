from .core import (
    Mesh, Renderer, Camera
)
import sys 
sys.path.append('/home/haonan/data/hold_soft/generator/src/diff_renderer')
import nvdiffrec.render.light as light
import nvdiffrec.render.renderutils.ops as ru
import nvdiffrast.torch as dr

import mano 
import torch
import numpy as np
import torch.nn as nn

from pytorch3d.io import load_obj

def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1

    i = points_cam_projected[..., 0] * K[0][0] + K[0][2]
    j = points_cam_projected[..., 1] * K[1][1] + K[1][2]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d

class Diff_Renderer(nn.Module):
    def __init__(self, K, device, obj_file=None):
        super().__init__()
        
        self.renderer = Renderer(device=device)
        self.channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
        print("Rasterizing:", self.channels_gbuffer)
        
        if obj_file is None:
            mano_model_path = '/home/haonan/data/hold_soft/generator/assets/MANO_RIGHT.pkl'
            rh_model = mano.model.load(
                    model_path=mano_model_path,
                    is_right= True, 
                    num_pca_comps=45, 
                    batch_size=1, 
                    flat_hand_mean=True).to(device)
            faces_new = np.array([[92, 38, 234],
                                [234, 38, 239],
                                [38, 122, 239],
                                [239, 122, 279],
                                [122, 118, 279],
                                [279, 118, 215],
                                [118, 117, 215],
                                [215, 117, 214],
                                [117, 119, 214],
                                [214, 119, 121],
                                [119, 120, 121],
                                [121, 120, 78],
                                [120, 108, 78],
                                [78, 108, 79]])
            faces = np.concatenate([rh_model.faces, faces_new], axis=0).astype(int)
            self.cano_mesh = Mesh(rh_model.v_template, faces, device=device)
            self.cano_mesh.compute_connectivity()
        
        else:
            # obj_file = '/home/haonan/Codes/IMavatar/code/mano_model/data/hand_head_uv_template.obj'
            verts, faces, aux = load_obj(obj_file)
            self.cano_mesh = Mesh(verts.numpy(), faces.verts_idx.numpy(), device=device)
            self.cano_mesh.compute_connectivity()
        
        self.K = K
        self.device = device
        
    # def render(self, vertices,):
    #     extrinsic = torch.eye(4)
    #     R = extrinsic[:3, :3]
    #     t = extrinsic[:3, 3]
    #     camera = Camera(self.K, R, t, device=self.device)
        
    #     num_frames = vertices.shape[0]
    #     cameras = list([camera for i in range(num_frames)])
        
    #     cano_mesh = self.cano_mesh.to(vertices.device)
    #     normals = cano_mesh.fetch_all_normals(vertices, cano_mesh)
        
    #     gbuffers = self.renderer.render_batch(cameras, vertices.contiguous(), normals, 
    #                                 channels=self.channels_gbuffer, with_antialiasing=True, 
    #                                 canonical_v=cano_mesh.vertices, canonical_idx=cano_mesh.indices) 
        
    #     pred_depth = gbuffers['position'][:,:,:,-1][:,:,:,None]

    #     mask = gbuffers['mask']
        
    #     return pred_depth, mask

    def render(self, vertices, extrinsic):
        # extrinsic = torch.eye(4)
        # R = extrinsic[:3, :3]
        # t = extrinsic[:3, 3]
        # camera = Camera(self.K, R, t, device=self.device)
        
        # num_frames = vertices.shape[0]
        # cameras = list([camera for i in range(num_frames)])

        num_frames = vertices.shape[0]
        cameras = []
        for i in range(num_frames):
            R = extrinsic[i, :3, :3]
            t = extrinsic[i, :3, 3]
            camera = Camera(self.K, R, t, device=self.device)
            cameras.append(camera)
        
        cano_mesh = self.cano_mesh.to(vertices.device)
        normals = cano_mesh.fetch_all_normals(vertices, cano_mesh)
        
        # vertices[:,:,2] = vertices[:,:,2] - vertices[:,:,2].min()

        # min_z = vertices[:, :, 2].min()
        # trans_vertices = vertices.clone()
        # trans_vertices[:, :, 2] = vertices[:, :, 2] - min_z

        gbuffers = self.renderer.render_batch(cameras, vertices.contiguous(), normals, 
                                    channels=self.channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=cano_mesh.vertices, canonical_idx=cano_mesh.indices) 
        
        pred_depth = gbuffers['position'][:,:,:,-1][:,:,:,None]

        mask = gbuffers['mask']
        
        # masked_pred_depth = pred_depth[mask.bool()]
        # pred_depth = (pred_depth - masked_pred_depth.min()) / (masked_pred_depth.max() - masked_pred_depth.min())
        # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
        # pred_depth = (pred_depth - pred_depth.min()) / pred_depth.max()
        # pred_depth = (pred_depth - pred_depth.min())

        return pred_depth, mask