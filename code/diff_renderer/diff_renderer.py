from .core import (
    Mesh, Renderer, Camera
)
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
            mano_model_path = '../mano_model/data/mano/MANO_RIGHT.pkl'
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
            verts, faces, aux = load_obj(obj_file)
            self.cano_mesh = Mesh(verts.numpy(), faces.verts_idx.numpy(), device=device)
            self.cano_mesh.compute_connectivity()
        
        self.K = K
        self.device = device
        
    def render(self, vertices, extrinsic):
        num_frames = vertices.shape[0]
        cameras = []
        for i in range(num_frames):
            R = extrinsic[i, :3, :3]
            t = extrinsic[i, :3, 3]
            camera = Camera(self.K, R, t, device=self.device)
            cameras.append(camera)
        
        cano_mesh = self.cano_mesh.to(vertices.device)
        normals = cano_mesh.fetch_all_normals(vertices, cano_mesh)
        
        gbuffers = self.renderer.render_batch(cameras, vertices.contiguous(), normals, 
                                    channels=self.channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=cano_mesh.vertices, canonical_idx=cano_mesh.indices) 
        
        pred_depth = gbuffers['position'][:,:,:,-1][:,:,:,None]

        mask = gbuffers['mask']
        
        return pred_depth, mask