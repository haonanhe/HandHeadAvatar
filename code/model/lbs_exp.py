from external.body_models import MANOLayer
from external.lbs import inverse_skinning_pts, inverse_pts

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d import ops
import math
import os

class MANODeformer:
    def __init__(
        self,
        max_dist=0.1,
        K=1,
        betas=None,
        server=None,
    ):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.server = server
        # params_canoical = self.server.param_canonical.clone()
        # params_canoical[:, -body_specs.shape_dim :] = (
        #     torch.tensor(betas).float().to(self.server.param_canonical.device)
        # )
        # cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(
        #     params_canoical,
        #     [1, 3, body_specs.full_pose_dim, body_specs.shape_dim],
        #     dim=1,
        # )
        # cano_global_orient = torch.zeros(3).unsqueeze(0)
        # cano_pose = self.server.default_hand_pose[2:].unsqueeze(0)#_tensor #torch.zeros(15, 3)
        # cano_beta = torch.zeros(10).unsqueeze(0)
        device = 'cuda'
        # batch_size = 1
        # cano_global_orient = torch.eye(3, device=device).view(
        #         1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        # cano_pose = torch.eye(3, device=device).view(
        #         1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        # cano_beta = torch.zeros(
        #         [batch_size, 10], device=device)
        # pred_mano_params = {
        #     'global_orient': cano_global_orient,
        #     'hand_pose': cano_pose,
        #     'betas': cano_beta,
        # }
        self.server = server.to(device)
        # mano_output = self.server(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        mano_output = self.server(absolute=True)
        self.skin_weights = mano_output.weights
        self.shapedirs = self.server.shapedirs
        self.posedirs = self.server.posedirs
        self.shapedirs = self.shapedirs.reshape(1, -1, 10*3)
        self.posedirs = self.posedirs.reshape(1, -1, 135*3)

        # output = self.server(cano_scale, cano_transl, cano_thetas, cano_betas)
        # #  canonical vertices
        # self.verts = output["verts"]
        # self.skin_weights = output["skin_weights"]

    def forward(self, x, tfs, return_weights=False, inverse=True, verts=None):
        """
        # transform query points from one space to another given tfs

        if not inverse:
            cano -> deform
        else:
            deform -> cano

        if tfs is None:
            use canonical pose tfs
        else:
            use the given tfs
        """
        assert len(x.shape) == 3
        assert len(tfs.shape) == 4
        assert x.shape[0] == tfs.shape[0]
        assert tfs.shape[2] == 4
        assert tfs.shape[3] == 4
        # curr_verts = self.verts.repeat(x.shape[0], 1, 1)
        skin_weights = self.skin_weights.repeat(x.shape[0], 1, 1)
        # if x.shape[0] == 0:
        #     return x
        # if verts is None:
        #     weights, outlier_mask = self.query_skinning_weights_multi(
        #         x, verts=curr_verts, skin_weights=skin_weights
        #     )
        # else:
        weights, outlier_mask = self.query_skinning_weights_multi(
            x, verts=verts, skin_weights=skin_weights
        )
        if return_weights:
            return weights
        x_transformed = skinning(x, weights, tfs, inverse=inverse)
        others = {'outlier_mask': outlier_mask}
        return x_transformed, others

    def forward_skinning(self, xc, cond, tfs):
        num_images = xc.shape[0]
        verts = self.verts.repeat(num_images, 1, 1)
        skin_weights = self.skin_weights.repeat(num_images, 1, 1)
        # cano -> deformed
        # query skining weights in cano
        weights, _ = self.query_skinning_weights_multi(
            xc, verts=verts, skin_weights=skin_weights
        )

        # LBS
        x_transformed = skinning(xc, weights, tfs, inverse=False)
        return x_transformed

    def query_skinning_weights_multi(self, pts, verts, skin_weights):
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts, verts, K=self.K, return_nn=True
        )
        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)

        num_parts = skin_weights.shape[2]

        # Expand index_batch for all parts
        expanded_index = index_batch[:, :, :, None].repeat(1, 1, 1, num_parts)
        skin_weights = skin_weights[:, :, None, :].repeat(1, 1, self.K, 1)
        weights_k = torch.gather(skin_weights, 1, expanded_index)
        # Multiply weights by their respective confidences and sum along the K dimension
        weights = (weights_k * weights_conf.unsqueeze(-1)).sum(dim=2).detach()

        distance_batch = distance_batch.min(dim=2).values
        outlier_mask = distance_batch > self.max_dist
        return weights, outlier_mask

    def query_weights(self, xc):
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse=False):
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc[0], cond)

        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum("bpn,bnij->bpij", w.double(), tfs.double())
            p_h = torch.einsum("bpij,bpj->bpi", tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum(
                "bpn, bnij, bpj->bpi", w.double(), tfs.double(), p_h.double()
            ).float()

        return p_h[:, :, :3]

MANOServer = MANOLayer(model_path="/home/haonan/Codes/IMavatar/code/mano_model/data/mano",
                        is_rhand=True,
                        batch_size=1,
                        flat_hand_mean=False,
                        dtype=torch.float32,
                        use_pca=False,)
manoDeformer = MANODeformer(betas=None, server=MANOServer)

# pred_mano_params = {
#     'global_orient': mano_global_orient,
#     'hand_pose': mano_hand_pose,
#     'betas': mano_betas,
# }
# mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
# verts = mano_output.vertices.clone()
# transformations = mano_output.T.clone()
# pose_feature = mano_output.pose_feature.clone()

points = MANOServer.canonical_verts
