"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import torch
from cv2 import solve
from torch import nn
from torch.nn import functional as F
from model.monosdf_loss import ScaleAndShiftInvariantLoss

class Loss(nn.Module):
    def __init__(self, mask_weight, depth_weight, lbs_weight, flame_distance_weight, alpha, expression_reg_weight, pose_reg_weight, cam_reg_weight, eikonal_weight, gt_w_seg=False):
        super().__init__()
        
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
        self.lbs_weight = lbs_weight
        self.flame_distance_weight = flame_distance_weight
        self.expression_reg_weight = expression_reg_weight
        self.cam_reg_weight = cam_reg_weight
        self.pose_reg_weight = pose_reg_weight
        self.gt_w_seg = gt_w_seg
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.eikonal_weight = eikonal_weight
        self.cosine_loss = nn.CosineSimilarity(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
    #     if (network_object_mask & object_mask).sum() == 0:
    #         return torch.tensor(0.0).cuda().float()

    #     rgb_values = rgb_values[network_object_mask & object_mask]
    #     rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
    #     rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
    #     return rgb_loss
    
    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask, network_occ_mask=None, depth_mask=None):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        mask = network_object_mask & object_mask 
        
        if network_occ_mask is not None and depth_mask is not None:
            mask = mask & (~network_occ_mask | depth_mask)

        rgb_values = rgb_values[mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_flame_distance_loss(self, flame_distance, semantic_gt, network_object_mask):
        object_skin_mask = semantic_gt[:, :, 0].reshape(-1) == 1
        if (network_object_mask & object_skin_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        flame_distance = flame_distance[network_object_mask & object_skin_mask]
        flame_distance_loss = torch.mean(flame_distance * flame_distance)

        return flame_distance_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, flame_distance, network_object_mask, object_mask):

        flame_distance_mask = flame_distance < 0.001
        if (network_object_mask & object_mask & flame_distance_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        lbs_weight = lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(object_mask.shape[0])
        return lbs_loss
    
    # def get_lbs_loss(self, lbs_weight, gt_lbs_weight, network_object_mask, object_mask):
    #     if (network_object_mask & object_mask).sum() == 0:
    #         return torch.tensor(0.0).cuda().float()

    #     lbs_weight = lbs_weight[network_object_mask & object_mask]
    #     gt_lbs_weight = gt_lbs_weight[network_object_mask & object_mask]
    #     # surface_distance = surface_distance[network_object_mask & object_mask]
    #     lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(object_mask.shape[0])
    #     return lbs_loss


    # def get_mask_loss(self, sdf_output, network_object_mask, object_mask, valid_mask):
    #     mask = (~(network_object_mask & object_mask)) 
    #     if valid_mask is not None:
    #         mask = mask & valid_mask
    #     if mask.sum() == 0:
    #         return torch.tensor(0.0).cuda().float()
    #     sdf_pred = -self.alpha * sdf_output[mask]
    #     gt = object_mask[mask].float()
    #     mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(-1), gt, reduction='sum') / float(object_mask.shape[0])
    #     return mask_loss
    
    # def get_mask_loss(self, sdf_output, network_object_mask, object_mask, hand_mask):
    #     ## TODO fix hand mask
    #     mask = (~(network_object_mask & object_mask)) & ~hand_mask
    #     if mask.sum() == 0:
    #         return torch.tensor(0.0).cuda().float()
    #     sdf_pred = -self.alpha * sdf_output[mask]
    #     gt = object_mask[mask].float()
    #     mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
    #     return mask_loss
    
    def get_mask_loss(self, sdf_output, network_object_mask, object_mask, occ_mask=None):
        ## TODO fix hand mask
        mask = (~(network_object_mask & object_mask)) #& ~occ_mask
        
        if occ_mask is not None:
            mask = mask & ~occ_mask
            
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss


    # def get_depth_loss(self, predicted_depth, depth, network_object_mask, object_mask, head_mask):
    #     # def silog_loss(prediction, target, variance_focus: float = 0.85, valid_mask=None) -> float:
    #     #     """
    #     #     Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    #     #     more information about scale-invariant loss.
    #     #
    #     #     Args:
    #     #         prediction (Tensor): Prediction.
    #     #         target (Tensor): Target.
    #     #         variance_focus (float): Variance focus for the SILog computation.
    #     #
    #     #     Returns:
    #     #         float: SILog loss.
    #     #     """
    #     #     ##TODO be cautious, this seems to only work with single batch?
    #     #     # let's only compute the loss on non-null pixels from the ground-truth depth-map
    #     #     non_zero_mask = (target > 0) & (prediction > 0)
    #     #     if valid_mask is not None:
    #     #         valid_mask = valid_mask.view(non_zero_mask.shape[0], non_zero_mask.shape[1], non_zero_mask.shape[2])
    #     #         non_zero_mask = non_zero_mask & valid_mask.bool()  # [:,None]
    #     #
    #     #     # SILog
    #     #     d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    #     #     loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2))
    #     #
    #     #     return loss
    #     # return silog_loss(predicted_depth, depth.squeeze(0))
    #     mask = network_object_mask & head_mask
    #     return self.depth_loss(predicted_depth[None, :, None], depth[:, :, None], mask[None, :, None])
    
    # def get_depth_loss(self, predicted_depth, depth, network_object_mask, head_mask):
    #     mask = network_object_mask & head_mask      #     return loss
    #     # return
    #     if mask.sum() == 0:
    #         return torch.tensor(0.0).cuda().float()
    #     # return self.depth_loss(predicted_depth[None, :, None], depth[:, :, None], mask[None, :, None])
    #     # return l1_distance(predicted_depth.reshape(-1), depth.reshape(-1), mask.reshape(-1))
    #     mask = mask.reshape(-1)
    #     predicted_depth = predicted_depth.reshape(-1)[mask]
    #     depth = depth.reshape(-1)[mask]
    #     return self.l1_loss(predicted_depth, depth) / float(mask.shape[0])

    def get_expression_reg_weight(self, pred, gt):
        return self.l2_loss(pred, gt)

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, semantics, surface_mask, ghostbone):
        bz = index_batch.shape[0]
        index_batch = index_batch[surface_mask]
        output = {}
        if ghostbone:
            gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        gt_skinning_values = torch.ones(bz, 6 if ghostbone else 5).float().cuda()
        gt_skinning_values[surface_mask] = gt_lbs_weight
        #hair deforms with head
        if self.gt_w_seg:
            hair = semantics[:, :, 6].reshape(-1) == 1
            gt_skinning_values[hair, :] = 0.
            gt_skinning_values[hair, 2 if ghostbone else 1] = 1.
        if ghostbone and self.gt_w_seg:
            # cloth deforms with ghost bone (identity)
            cloth = semantics[:, :, 7].reshape(-1) == 1
            gt_skinning_values[cloth, :] = 0.
            gt_skinning_values[cloth, 0] = 1.
        output['gt_lbs_weight'] = gt_skinning_values

        gt_posedirs_values = torch.ones(bz, 36, 3).float().cuda()
        gt_posedirs_values[surface_mask] = gt_posedirs
        if self.gt_w_seg:
            # mouth interior and eye glasses doesn't deform
            mouth = semantics[:, :, 3].reshape(-1) == 1
            gt_posedirs_values[mouth, :] = 0.0
        output['gt_posedirs'] = gt_posedirs_values


        gt_shapedirs_values = torch.ones(bz, 3, 50).float().cuda()
        gt_shapedirs_values[surface_mask] = gt_shapedirs

        disable_shapedirs_for_mouth_and_cloth = False
        if disable_shapedirs_for_mouth_and_cloth:
            # I accidentally deleted these when cleaning the code...
            # So this is why I don't see teeth anymore...QAQ
            # Most of suppmat experiments used this code block, but it doesn't necessarily help in all cases.
            if self.gt_w_seg:
                # mouth interior and eye glasses doesn't deform
                mouth = semantics[:, :, 3].reshape(-1) == 1
                gt_shapedirs_values[mouth, :] = 0.0
            if ghostbone and self.gt_w_seg:
                # cloth doesn't deform with facial expressions
                cloth = semantics[:, :, 7].reshape(-1) == 1
                gt_shapedirs_values[cloth, :] = 0.
        output['gt_shapedirs'] = gt_shapedirs_values
        return output
    
    # def get_gt_blendshape(self, output, semantics, ghostbone):
    #     gt_skinning_values = output['gt_lbs_weight']
    #     if self.gt_w_seg:
    #         hair = semantics[:, :, 6].reshape(-1) == 1
    #         gt_skinning_values[hair, :] = 0.
    #         gt_skinning_values[hair, 2 if ghostbone else 1] = 1.
    #     if ghostbone and self.gt_w_seg:
    #         # cloth deforms with ghost bone (identity)
    #         cloth = semantics[:, :, 7].reshape(-1) == 1
    #         gt_skinning_values[cloth, :] = 0.
    #         gt_skinning_values[cloth, 0] = 1.
    #     output['gt_lbs_weight'] = gt_skinning_values
        
    #     gt_posedirs_values = output['gt_posedirs']
    #     if self.gt_w_seg:
    #         # mouth interior and eye glasses doesn't deform
    #         mouth = semantics[:, :, 3].reshape(-1) == 1
    #         gt_posedirs_values[mouth, :] = 0.0
    #     output['gt_posedirs'] = gt_posedirs_values
        
    #     gt_shapedirs_values = output['gt_shapedirs']
    #     disable_shapedirs_for_mouth_and_cloth = False
    #     if disable_shapedirs_for_mouth_and_cloth:
    #         # I accidentally deleted these when cleaning the code...
    #         # So this is why I don't see teeth anymore...QAQ
    #         # Most of suppmat experiments used this code block, but it doesn't necessarily help in all cases.
    #         if self.gt_w_seg:
    #             # mouth interior and eye glasses doesn't deform
    #             mouth = semantics[:, :, 3].reshape(-1) == 1
    #             gt_shapedirs_values[mouth, :] = 0.0
    #         if ghostbone and self.gt_w_seg:
    #             # cloth doesn't deform with facial expressions
    #             cloth = semantics[:, :, 7].reshape(-1) == 1
    #             gt_shapedirs_values[cloth, :] = 0.
    #     output['gt_shapedirs'] = gt_shapedirs_values
        
    #     return output

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_contact_reg_bbox(self, sdf_head, sdf_hand, nonrigid_deformation):
        if nonrigid_deformation is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        sdf_head = sdf_head.reshape(-1)
        sdf_hand = sdf_hand.reshape(-1)

        # 1. ~ (both sdf < 0);
        # mask = ~((sdf_head < 0) & (sdf_hand < 0))
        # margin = 1e-6
        margin = 0
        mask = ~((sdf_head < margin) & (sdf_hand < margin))

        if mask.sum() == 0: # which means all points are contact points
            # return ((nonrigid_deformation)** 2).mean()
            # return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
            return 0.1 * (nonrigid_deformation**2).mean()  # Weak regularization
        else:
            return ((nonrigid_deformation[mask])** 2).mean()
        
    def get_contact_reg_surface(self, nonrigid_deformation):
        return ((nonrigid_deformation)** 2).mean()
    
    def get_temporal_reg_nonrigid(self, nonrigid_deformation, prev_nonrigid_deformation):
        if prev_nonrigid_deformation is None or nonrigid_deformation is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        else:
            return ((nonrigid_deformation - prev_nonrigid_deformation)** 2).mean()

    def get_contact_direction_loss(self, sdf_head, sdf_hand, nonrigid_deformation, normal_hand):
        if nonrigid_deformation is None or normal_hand is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        sdf_head = sdf_head.reshape(-1)
        sdf_hand = sdf_hand.reshape(-1)

        mask = (sdf_head < 0) & (sdf_hand < 0)

        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        else:
            cos_sim = self.cosine_loss(nonrigid_deformation[mask], normal_hand[mask])
            return torch.mean(1 - cos_sim)  # Minimize 1 - cos(θ)
    
    def get_contact_loss_bbox_hingeloss(self, sdf_head, sdf_hand):
        if sdf_head is None or sdf_hand is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
            
        sdf_head = sdf_head.reshape(-1)
        sdf_hand = sdf_hand.reshape(-1)

        # 1. both sdf < 0;
        # mask = (sdf_head < 0) & (sdf_hand < 0)
        # margin = 1e-6
        margin = 0
        mask = ((sdf_head < margin) & (sdf_hand < margin))
        
        # breakpoint()

        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        # sdf --> 0
        else:
            # return ((sdf_head[mask])** 2).mean()
            # return torch.relu(-sdf_head[mask]).mean()
            # return ((sdf_head[mask])** 2).mean() + torch.relu(-sdf_hand[mask]).mean()
            # return torch.relu(-sdf_head[mask])
            return (torch.relu(-sdf_head[mask]).mean() + (sdf_head[mask]**2).mean()) #+ torch.relu(-sdf_hand[mask]).mean()
        
    def get_contact_loss_mesh(self, sdf_head):
        if sdf_head is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
            
        sdf_head = sdf_head.reshape(-1)
        # 1. both sdf < 0;
        # mask = (sdf_head < 0) & (sdf_hand < 0)
        # margin = 1e-6
        margin = 0
        mask = (sdf_head < margin)
        
        # breakpoint()

        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        # sdf --> 0
        else:
            # return (sdf_head[mask]**2).mean()
            return (torch.relu(-sdf_head[mask]).mean() + (sdf_head[mask]**2).mean())

    def get_rgb_loss_hand(self, pred_rgb_img, gt_rgb_img, pred_hand_mask_img, gt_hand_mask_img):
        pred_hand_mask_img = pred_hand_mask_img.bool()
        gt_hand_mask_img = gt_hand_mask_img.bool()
        
        mask = pred_hand_mask_img & gt_hand_mask_img 
        
        pred_rgb_img = pred_rgb_img * mask
        gt_rgb_img = gt_rgb_img * mask
        rgb_loss = self.l1_loss(pred_rgb_img, gt_rgb_img) / float(mask.sum())
        return rgb_loss
    
    def get_mask_loss_hand(self, pred_hand_mask_img, gt_hand_mask_img, gt_head_mask_img):
        pred_hand_mask_img = pred_hand_mask_img.bool()
        gt_hand_mask_img = gt_hand_mask_img.bool()
        gt_head_mask_img = gt_head_mask_img.bool()
        
        # mask = pred_hand_mask_img & gt_hand_mask_img & (~gt_head_mask_img)
        mask = (~gt_head_mask_img) & (pred_hand_mask_img | gt_hand_mask_img)
        
        pred_hand_mask_img = (pred_hand_mask_img * mask).float()
        gt_hand_mask_img = (gt_hand_mask_img * mask).float()
        mask_loss = self.l1_loss(pred_hand_mask_img, gt_hand_mask_img) / float(mask.sum())
        return mask_loss
    
    def get_depth_loss(self, predicted_depth, depth, network_object_mask, head_mask):
        mask = network_object_mask & head_mask      #     return loss
        # return
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        return self.depth_loss(predicted_depth[None, :, None], depth[:, :, None], mask[None, :, None])
    
    def get_depth_loss_hand(self, pred_depth_img, gt_depth_img, pred_hand_mask_img, gt_hand_mask_img):
        pred_hand_mask_img = pred_hand_mask_img.bool()
        gt_hand_mask_img = gt_hand_mask_img.bool()
        
        mask = pred_hand_mask_img & gt_hand_mask_img 
        
        pred_depth_img = pred_depth_img * mask
        gt_depth_img = gt_depth_img * mask
        depth_loss = self.depth_loss(pred_depth_img, gt_depth_img, mask)
        return depth_loss
    
    def get_contact_loss_hand(self, verts_hand_tips, points_head, points_hand, sdf_hand_in_head):
        if sdf_hand_in_head is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        verts_hand_tips = verts_hand_tips.reshape(1,-1,3)
        points_head = points_head.reshape(1,-1,3)
        points_hand = points_hand.reshape(1,-1,3)
        
        contact_loss = ((knn_points(verts_hand_tips, points_head, K=1, return_nn=False)[0])**2).mean()
        
        closest_idx_hand = knn_points(verts_hand_tips, points_hand, K=3, return_nn=False)[1]
        closest_idx_hand = closest_idx_hand.reshape(-1)
        sdf_hand_in_head = sdf_hand_in_head[closest_idx_hand]
        
        mask = (sdf_hand_in_head < 0)

        if mask.sum() == 0:
            return contact_loss

        contact_loss = contact_loss + ((sdf_hand_in_head[mask])** 2).mean()

        return contact_loss
    
    def get_depth_mask_loss(self, depth_mask_head, depth_mask_hand, mask_head, mask_hand):
        depth_mask = torch.zeros_like(depth_mask_head).float().to(depth_mask_head.device)
        depth_mask[depth_mask_head] = 1
        depth_mask[depth_mask_hand] = 2

        mask_head = mask_head.reshape(-1)
        mask_hand = mask_hand.reshape(-1)
        mask = torch.zeros_like(mask_head).float().to(mask_head.device)
        mask[mask_head.bool()] = 1
        mask[mask_hand.bool()] = 2

        return self.cross_entropy_loss(depth_mask, mask)
    
    def get_landmark_loss_hand(self, pred_hand_lmk, gt_hand_lmk):
        lmk_loss = self.l2_loss(pred_hand_lmk, gt_hand_lmk)
        return lmk_loss
    
    def get_flame_sdf_loss(self, sdf, flame_sdf):
        mask = (flame_sdf < 0) & (sdf > 0)
        # mask = (flame_sdf is not None) & (sdf is not None)
        # mask = torch.ones_like(flame_sdf).to(flame_sdf.device).bool()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        
        flame_sdf_loss = self.l1_loss(sdf[mask], flame_sdf[mask]) / float(mask.sum())
        return flame_sdf_loss
        
        # sdf_pred = -self.alpha * sdf_output[mask]
        # gt = object_mask[mask].float()
        # flame_sdf_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
    
    # def forward(self, model_outputs, ground_truth):
    #     loss = self.get_flame_sdf_loss(model_outputs['implicit_sdf'], model_outputs['flame_sdf'])
    #     out = {
    #         'loss': loss,
    #     }
        
    #     return out
    
    def forward(self, model_outputs, ground_truth, prev_nonrigid_deformation):
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        # rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], ground_truth['rgb'], network_object_mask, object_mask)
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], ground_truth['rgb'], network_object_mask, model_outputs['head_mask'], model_outputs['network_object_mask_hand'], model_outputs['depth_mask_head'])
        # mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask, model_outputs['valid_mask'])
        # mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask, model_outputs['hand_mask'])
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, model_outputs['head_mask'], model_outputs['hand_mask'])
        mask_loss = self.mask_weight * mask_loss

        # eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss + mask_loss #+ self.eikonal_weight * eikonal_loss
        

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
            # 'eikonal_loss': eikonal_loss,
        }

        if self.depth_weight > 0:
            depth_loss = self.get_depth_loss(model_outputs['depth_values'], ground_truth['depth'], network_object_mask, model_outputs['head_mask'])
            depth_loss = self.depth_weight * depth_loss
            loss += depth_loss
            out['depth_loss'] = depth_loss


        if self.lbs_weight != 0:
            ghostbone = model_outputs['lbs_weight'].shape[1] == 6
            # outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'], model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'], ground_truth['semantics'], model_outputs['network_object_mask'] & model_outputs['object_mask'], ghostbone)
            outputs = model_outputs
            num_points = model_outputs['lbs_weight'].shape[0]
            if self.gt_w_seg:
                # do not enforce nearest neighbor skinning weight for teeth, learn from data instead.
                # now it's also not enforcing nn skinning wieght for glasses, I'm too lazy to correct it but the skinning weight can still learn correctly for glasses.
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], network_object_mask, object_mask & (ground_truth['semantics'][:, :, 3].reshape(-1) != 1))
            else:
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], network_object_mask, object_mask)

            lbs_loss = lbs_loss * self.lbs_weight * 0.1
            out['loss'] += lbs_loss
            out['lbs_loss'] = lbs_loss

            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10, outputs['gt_posedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
            posedirs_loss = posedirs_loss * self.lbs_weight * 10.0
            out['loss'] += posedirs_loss
            out['posedirs_loss'] = posedirs_loss

            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1) * 10, outputs['gt_shapedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
            shapedirs_loss = shapedirs_loss * self.lbs_weight * 10.0
            out['loss'] += shapedirs_loss
            out['shapedirs_loss'] = shapedirs_loss

            # if self.gt_w_seg:
            #     # do not enforce nearest neighbor skinning weight for teeth, learn from data instead.
            #     # now it's also not enforcing nn skinning wieght for glasses, I'm too lazy to correct it but the skinning weight can still learn correctly for glasses.
            #     lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), network_object_mask, object_mask & (ground_truth['semantics'][:, :, 3].reshape(-1) != 1))
            # else:
            #     lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), network_object_mask, object_mask)

            # out['loss'] += lbs_loss * self.lbs_weight * 0.1
            # out['lbs_loss'] = lbs_loss

            # posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10, outputs['gt_posedirs'].reshape(num_points, -1) * 10, network_object_mask, object_mask)
            # out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            # out['posedirs_loss'] = posedirs_loss

            # shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1) * 10, outputs['gt_shapedirs'].reshape(num_points, -1) * 10, network_object_mask, object_mask)
            # out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            # out['shapedirs_loss'] = shapedirs_loss

        # if 'semantics' in ground_truth and self.flame_distance_weight > 0 and self.gt_w_seg:
        #     out['flame_distance_loss'] = self.get_flame_distance_loss(model_outputs['flame_distance'], ground_truth['semantics'], network_object_mask)
        #     out['loss'] += out['flame_distance_loss'] * self.flame_distance_weight

        # # outputs = self.get_gt_blendshape(model_outputs, ground_truth['semantics'], ghostbone)
        # outputs = model_outputs
        # # if 'lbs_weight' in model_outputs:
        # if 'gt_lbs_weight' in model_outputs:       
        #     ghostbone = model_outputs['lbs_weight'].shape[1] == 6
        #     num_points = model_outputs['lbs_weight'].shape[0]
        #     lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), model_outputs['gt_lbs_weight'].reshape(num_points, -1), network_object_mask, object_mask)#, model_outputs['flame_distance'])
        #     out['loss'] += lbs_loss * self.lbs_weight #* 10.0 #0.1
        #     out['lbs_loss'] = lbs_loss
        # if 'posedirs' in model_outputs:
        #     num_points = model_outputs['posedirs'].shape[0]
        #     posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10, model_outputs['gt_posedirs'].reshape(num_points, -1) * 10, network_object_mask, object_mask)#, model_outputs['flame_distance'])
        #     out['loss'] += posedirs_loss * self.lbs_weight * 10.0
        #     out['posedirs_loss'] = posedirs_loss
        # if 'shapedirs' in model_outputs:
        #     num_points = model_outputs['shapedirs'].shape[0]
        #     shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1) * 10, model_outputs['gt_shapedirs'].reshape(num_points, -1) * 10, network_object_mask, object_mask)#, model_outputs['flame_distance'])
        #     out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
        #     out['shapedirs_loss'] = shapedirs_loss
            
        if self.expression_reg_weight != 0 and 'expression' in ground_truth:
            out['expression_reg_loss'] = self.expression_reg_weight * self.get_expression_reg_weight(model_outputs['expression'][..., :50], ground_truth['expression'])
            out['loss'] += out['expression_reg_loss']

        if self.pose_reg_weight != 0 and 'flame_pose' in ground_truth:
            out['pose_reg_loss'] =  self.pose_reg_weight * self.get_expression_reg_weight(model_outputs['flame_pose'], ground_truth['flame_pose'])
            out['loss'] += out['pose_reg_loss']

        if self.cam_reg_weight != 0 and 'cam_pose' in ground_truth:
            out['cam_reg_loss'] = self.cam_reg_weight * self.get_expression_reg_weight(model_outputs['cam_pose'][:, :3, 3], ground_truth['cam_pose'][:, :3, 3])
            out['loss'] += out['cam_reg_loss']
        
        
        # ######################### HAND ########################
        
        # img_res = int(model_outputs['img_res'][0][0].item())
        # pred_rgb_img = model_outputs['rgb_image_hand'].reshape(img_res,img_res,3)
        # pred_hand_mask_img = model_outputs['mask_image_hand'].reshape(img_res,img_res,1)
        # pred_hand_lmk = model_outputs['hand_lmk']
        # gt_rgb_img = ground_truth['rgb_image_hand'].reshape(img_res,img_res,3)
        # gt_hand_mask_img = ground_truth['mask_image_hand'].reshape(img_res,img_res,1)
        # gt_hand_lmk = ground_truth['hand_lmk']
        # gt_head_mask_img = ground_truth['mask_image_head'].reshape(img_res,img_res,1)
        # pred_depth_img = model_outputs['depth_image_hand'].reshape(img_res,img_res,1)
        # gt_depth_img = ground_truth['depth_image_hand'].reshape(img_res,img_res,1)
        
        # hand_rgb_loss = self.get_rgb_loss_hand(pred_rgb_img, gt_rgb_img, pred_hand_mask_img, gt_hand_mask_img)
        # out['hand_rgb_loss'] = hand_rgb_loss
        # out['loss'] = out['loss'] + out['hand_rgb_loss']
        
        # hand_mask_loss = self.get_mask_loss_hand(pred_hand_mask_img, gt_hand_mask_img, gt_head_mask_img)
        # out['hand_mask_loss'] = hand_mask_loss
        # out['loss'] = out['loss'] + out['hand_mask_loss']
        
        # hand_landmark_loss = self.get_landmark_loss_hand(pred_hand_lmk[:,:,:2], gt_hand_lmk[:,:,:2])
        # out['hand_landmark_loss'] = hand_landmark_loss
        # out['loss'] = out['loss'] + out['hand_landmark_loss']
        
        # hand_depth_loss = self.get_depth_loss_hand(pred_depth_img, gt_depth_img, pred_hand_mask_img, gt_hand_mask_img)
        # out['hand_depth_loss'] = hand_depth_loss
        # out['loss'] = out['loss'] + out['hand_depth_loss']
        
        # depth_mask_loss = self.get_depth_mask_loss(model_outputs['depth_mask_head'], model_outputs['depth_mask_hand'], model_outputs['head_mask'], model_outputs['hand_mask'])
        # depth_mask_loss = depth_mask_loss * 1e-5
        # out['depth_mask_loss'] = depth_mask_loss
        # out['loss'] = out['loss'] + out['depth_mask_loss']
    
        
        optimize_contact = model_outputs['optimize_contact']
        if optimize_contact:
            out['contact_loss'] = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

            # contact_regularation_surface = self.get_contact_reg_bbox(model_outputs['sdf_output_handsurf_tohead'], model_outputs['sdf_output_hand'], model_outputs['nonrigid_deformation_handsurf_tohead'])
            # contact_regularation_surface = contact_regularation_surface + self.get_contact_reg_surface(model_outputs['nonrigid_deformation'])
            # contact_regularation_surface = contact_regularation_surface + self.get_contact_reg_surface(model_outputs['nonrigid_deformation_reg'])
            
            # contact_regularation_surface = contact_regularation_surface * 100000.0  #1000000 #10000
            # out['contact_regularation_surface'] = contact_regularation_surface
            # out['contact_loss'] = out['contact_loss'] + contact_regularation_surface
            
            # contact_loss_surf = self.get_contact_loss_mesh(model_outputs['sdf_output_handmesh_tohead'])
            # contact_loss_surf = contact_loss_surf * 50 #500.0 #* 1000.0 #1000 #3 #* 5 #8 #10
            # out['contact_loss_surf'] = contact_loss_surf
            # out['contact_loss'] = out['contact_loss'] + contact_loss_surf
            
            
            # Contact regularization for left hand surface
            contact_regularation_surface_left = self.get_contact_reg_bbox(
                model_outputs['sdf_output_leftsurf_tohead'],
                model_outputs['sdf_output_left_hand'],  # Assuming this exists; if not, generate it
                model_outputs['nonrigid_deformation_leftsurf_tohead']
            )
            
            contact_regularation_surface_left  = contact_regularation_surface_left  + self.get_contact_reg_bbox(model_outputs['sdf_output'], model_outputs['sdf_output_headsurf_toleft'], model_outputs['nonrigid_deformation'])
            contact_regularation_surface_left  = contact_regularation_surface_left  + self.get_contact_reg_bbox(model_outputs['sdf_output_reg'], model_outputs['sdf_output_reg_toleft'], model_outputs['nonrigid_deformation_reg'])

            # Contact regularization for right hand surface
            contact_regularation_surface_right = self.get_contact_reg_bbox(
                model_outputs['sdf_output_rightsurf_tohead'],
                model_outputs['sdf_output_right_hand'],  # Assuming this exists
                model_outputs['nonrigid_deformation_rightsurf_tohead']
            )
            
            contact_regularation_surface_right  = contact_regularation_surface_right  + self.get_contact_reg_bbox(model_outputs['sdf_output'], model_outputs['sdf_output_headsurf_toright'], model_outputs['nonrigid_deformation'])
            contact_regularation_surface_right  = contact_regularation_surface_right  + self.get_contact_reg_bbox(model_outputs['sdf_output_reg'], model_outputs['sdf_output_reg_toright'], model_outputs['nonrigid_deformation_reg'])

            # Combine or treat separately depending on your use case
            contact_regularation_surface = contact_regularation_surface_left + contact_regularation_surface_right

            # # Additional contact regularization from general non-rigid deformation
            # contact_regularation_surface += self.get_contact_reg_surface(model_outputs['nonrigid_deformation'])
            # contact_regularation_surface += self.get_contact_reg_surface(model_outputs['nonrigid_deformation_reg'])

            # Scale the total contact regularization
            contact_regularation_surface *= 30000 #25000 #50000 #10000 #50000 #10000 #15000.0 #20000.0 #10000.0
            out['contact_regularation_surface'] = contact_regularation_surface
            out['contact_loss'] = out['contact_loss'] + contact_regularation_surface

            # Contact loss from mesh-based SDF (for both hands)
            contact_loss_surf_left = self.get_contact_loss_mesh(model_outputs['sdf_output_leftmesh_tohead'])
            contact_loss_surf_right = self.get_contact_loss_mesh(model_outputs['sdf_output_rightmesh_tohead'])

            contact_loss_surf = (contact_loss_surf_left + contact_loss_surf_right) * 100 #150 #5000  # scale factor

            out['contact_loss_surf'] = contact_loss_surf
            out['contact_loss'] = out['contact_loss'] + contact_loss_surf
            
            if 'deform_dir' in model_outputs.keys():
                deform_dir_loss = self.get_lbs_loss(model_outputs['deform_dir'].reshape(num_points, -1) * 10, outputs['gt_deform_dir'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
                deform_dir_loss = deform_dir_loss * self.lbs_weight * 10 #100.0 #* 10.0 
                out['deform_dir_loss'] = deform_dir_loss 
                out['contact_loss'] = out['contact_loss'] + deform_dir_loss 
            
            out['nonrigid_deformation_min'] = model_outputs['nonrigid_deformation'].min()
            out['nonrigid_deformation_max'] = model_outputs['nonrigid_deformation'].max()
            

            # temporal_reg_nonrigid = self.get_temporal_reg_nonrigid(model_outputs['nonrigid_deformation_handsurf_tohead'], prev_nonrigid_deformation)
            # temporal_reg_nonrigid = temporal_reg_nonrigid *1000
            # out['temporal_reg_nonrigid'] = temporal_reg_nonrigid
            # out['contact_loss'] = out['contact_loss'] + temporal_reg_nonrigid
                

        flame_sdf_loss = self.get_flame_sdf_loss(model_outputs['implicit_sdf'], model_outputs['flame_sdf'])
        flame_sdf_loss = flame_sdf_loss * 0.01
        # flame_sdf_loss = flame_sdf_loss * 1e-6 
        out['flame_sdf_loss'] = flame_sdf_loss
        out['loss'] = out['loss'] + flame_sdf_loss 
        
        # if self.pose_reg_weight != 0 and 'hand_pose' in ground_truth:
        #     out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose'])
        #     # out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose']) \
        #     #                             + self.get_expression_reg_weight(model_outputs['hand_global_orient'].reshape(-1,9), ground_truth['hand_global_orient']) \
        #     #                             + self.get_expression_reg_weight(model_outputs['hand_transl'].reshape(-1,3), ground_truth['hand_transl'])
        #     out['loss'] = out['loss'] + out['hand_pose_reg_loss'] * self.pose_reg_weight
        
        # if self.pose_reg_weight != 0 and 'mano_transl' in ground_truth:
        #     out['mano_transl_reg_loss'] = self.get_expression_reg_weight(model_outputs['mano_transl'].reshape(-1,3), ground_truth['mano_transl'])
        #     # out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose']) \
        #     #                             + self.get_expression_reg_weight(model_outputs['hand_global_orient'].reshape(-1,9), ground_truth['hand_global_orient']) \
        #     #                             + self.get_expression_reg_weight(model_outputs['hand_transl'].reshape(-1,3), ground_truth['hand_transl'])
        #     out['loss'] = out['loss'] + out['mano_transl_reg_loss'] * self.pose_reg_weight
            
        return out
    
    # def cal_contact_loss(self, model_outputs, ground_truth):
    #     out = {}
    #     out['contact_loss'] = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

    #     contact_loss_weight = 1

    #     # contact_regularation_surface = self.get_contact_reg_bbox(model_outputs['sdf_output_detach'], model_outputs['sdf_output_headsurf_tohand'], model_outputs['nonrigid_deformation_detach'])
    #     contact_regularation_surface = self.get_contact_reg_bbox(model_outputs['surface_output'], model_outputs['sdf_output_headsurf_tohand'], model_outputs['nonrigid_deformation'])
    #     contact_regularation_surface = contact_regularation_surface * contact_loss_weight * 1000  #1000000 #10000
    #     out['contact_regularation_surface'] = contact_regularation_surface
    #     out['contact_loss'] = out['contact_loss'] + contact_regularation_surface

    #     # contact_loss_surf = self.get_contact_loss_bbox_hingeloss(model_outputs['sdf_output_detach'], model_outputs['sdf_output_headsurf_tohand'])
    #     contact_loss_surf = self.get_contact_loss_bbox_hingeloss(model_outputs['surface_output'], model_outputs['sdf_output_headsurf_tohand'])
    #     contact_loss_surf = contact_loss_surf * contact_loss_weight * 10 #* 1000.0 #1000 #3 #* 5 #8 #10
    #     out['contact_loss_surf'] = contact_loss_surf
    #     out['contact_loss'] = out['contact_loss'] + contact_loss_surf
        
    #     network_object_mask = model_outputs['network_object_mask']
    #     object_mask = model_outputs['object_mask']
    #     num_points = model_outputs['lbs_weight'].shape[0]
        
    #     # deform_dir_loss = self.get_lbs_loss(model_outputs['deform_dir_detach'].reshape(num_points, -1) * 10, outputs['gt_deform_dir'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
    #     deform_dir_loss = self.get_lbs_loss(model_outputs['deform_dir'].reshape(num_points, -1) * 10, model_outputs['gt_deform_dir'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
    #     out['deform_dir_loss'] = deform_dir_loss
    #     out['contact_loss'] = out['contact_loss'] + deform_dir_loss * self.lbs_weight * 10.0

    #     return out


