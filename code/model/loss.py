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
    def __init__(self, mask_weight, depth_weight, lbs_weight, flame_distance_weight, alpha, expression_reg_weight, pose_reg_weight, cam_reg_weight, eikonal_weight, contact_sdf_weight, contact_reg_weight, rgb_weight, gt_w_seg=False):
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
        self.contact_sdf_weight = contact_sdf_weight
        self.contact_reg_weight = contact_reg_weight
        self.rgb_weight = rgb_weight
        

    # def get_rgb_loss(self, rgb_values, rgb_gt, pred_mask, gt_mask, other_parts_mask=None, depth_occlusion_mask=None):
    #     if (pred_mask & gt_mask).sum() == 0:
    #         return torch.tensor(0.0).cuda().float()

    #     mask = pred_mask & gt_mask 
        
    #     if other_parts_mask is not None and depth_occlusion_mask is not None:
    #         mask = mask & (~other_parts_mask | depth_occlusion_mask)
            
    #     # mask includes the intersection of pred and gt mask, all seen parts (before other parts), and excludes mask of other parts.
    #     # for example, for head, the mask includes the intersection of pred and gt head masks, head before hand masks, and excludes hand masks.

    #     rgb_values = rgb_values[mask]
    #     rgb_gt = rgb_gt.reshape(-1, 3)[mask]
    #     rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(mask.shape[0])
    #     return rgb_loss
    
    def get_rgb_loss(self, rgb_values, rgb_gt, pred_mask, gt_mask, other_parts_mask=None):
        if (pred_mask & gt_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        mask = pred_mask & gt_mask 
        
        if other_parts_mask is not None:
            mask = mask & (~other_parts_mask)
            
        # mask includes the intersection of pred and gt mask, all seen parts (before other parts), and excludes mask of other parts.
        # for example, for head, the mask includes the intersection of pred and gt head masks, head before hand masks, and excludes hand masks.

        rgb_values = rgb_values[mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(mask.shape[0])
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
        # flame_distance_mask = flame_distance < 0.0001
        # flame_distance_mask = flame_distance < 0.01
        # flame_distance_mask = flame_distance < 0.1
        if (network_object_mask & object_mask & flame_distance_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        lbs_weight = lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(object_mask.shape[0])
        return lbs_loss
    
    def get_mask_loss(self, sdf_output, pred_mask, gt_mask, occ_mask=None):
        mask = (~(pred_mask & gt_mask))
        
        if occ_mask is not None:
            mask = mask & ~occ_mask
            
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = gt_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(mask.shape[0])
        return mask_loss

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
    

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss
    
    def get_contact_reg(self, nonrigid_deformation, sdf_head=None, sdf_hand=None):
        if sdf_head is None or sdf_hand is None:
            return (nonrigid_deformation**2).mean()
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
            return (nonrigid_deformation**2).mean()  # Weak regularization
        else:
            reg = ((nonrigid_deformation[mask])** 2).mean()
            return reg
    
    # def get_contact_reg(self, nonrigid_deformation, sdf_head=None, sdf_hand=None, beta=20.0):
    #     if sdf_head is None or sdf_hand is None or nonrigid_deformation is None:
    #         return (nonrigid_deformation**2).mean() if nonrigid_deformation is not None else 0.0

    #     sdf_head = sdf_head.reshape(-1)
    #     sdf_hand = sdf_hand.reshape(-1)
    #     deformation = nonrigid_deformation.reshape(len(sdf_head), -1)  # 假设一一对应

    #     # Soft contact mask
    #     inside_head = torch.sigmoid(-beta * sdf_head)
    #     inside_hand = torch.sigmoid(-beta * sdf_hand)
    #     contact_weight = inside_head * inside_hand  # [0,1]，越大越接触
    #     reg_weight = 1.0 - contact_weight  # 非接触权重

    #     # Weighted L2 regularization on deformation magnitude
    #     deformation_sq = (deformation ** 2).sum(dim=-1)  # per-point magnitude
    #     reg = (reg_weight * deformation_sq).mean()

    #     return reg
    
    # def get_contact_loss(self, sdf_head):
    #     if sdf_head is None:
    #         return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
            
    #     sdf_head = sdf_head.reshape(-1)
    #     # 1. both sdf < 0;
    #     # mask = (sdf_head < 0) & (sdf_hand < 0)
    #     # margin = 1e-6
    #     margin = 0
    #     mask = (sdf_head < margin)
        
    #     # breakpoint()

    #     if mask.sum() == 0:
    #         return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
    #     # sdf --> 0
    #     else:
    #         return (sdf_head[mask]**2).mean()
    #         # return (torch.relu(-sdf_head[mask]).mean() + (sdf_head[mask]**2).mean())
    
    def get_contact_loss(self, sdf_head, sdf_hand):
        if sdf_head is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
            
        sdf_head = sdf_head.reshape(-1)
        # 1. both sdf < 0;
        # mask = (sdf_head < 0) & (sdf_hand < 0)
        # margin = 1e-5
        margin = 0
        # mask = (sdf_head < margin) & (sdf_hand < margin)
        mask = (sdf_head < 0) & (sdf_hand < margin)
        
        # breakpoint()

        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        # sdf --> 0
        else:
            loss = torch.relu(-sdf_head[mask]).mean()
            return loss
            # return (sdf_head[mask]**2).mean()
            # return (torch.relu(-sdf_head[mask]).mean() + (sdf_head[mask]**2).mean())
    
    # def get_contact_loss(self, sdf_head, sdf_hand, beta=20.0, margin=0.0):
    #     if sdf_head is None or sdf_head.numel() == 0:
    #         return sdf_head.new_zeros([])

    #     sdf_head = sdf_head.reshape(-1)
    #     sdf_hand = sdf_hand.reshape(-1)

    #     # Soft mask: high weight when both are inside (sdf < 0)
    #     # Smooth approximation of (sdf_head < 0) & (sdf_hand < 0)
    #     inside_head = torch.sigmoid(-beta * sdf_head)      # near 1 if sdf_head < 0
    #     inside_hand = torch.sigmoid(-beta * sdf_hand)      # near 1 if sdf_hand < 0
    #     soft_mask = inside_head * inside_hand

    #     # Encourage sdf_head to be close to 0 where contact is likely
    #     # You can also use L2: (sdf_head ** 2)
    #     contact_loss_per_point = torch.abs(sdf_head)  # pull toward surface
    #     loss = (soft_mask * contact_loss_per_point).sum() / (soft_mask.sum().clamp(min=1e-6))

    #     return loss

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
    
    def get_l1_depth_loss(self, predicted_depth, depth, network_object_mask, head_mask):
        mask = network_object_mask & head_mask & (depth!=-1)      #     return loss
        # return
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        # return self.depth_loss(predicted_depth[None, :, None], depth[:, :, None], mask[None, :, None])
        # return l1_distance(predicted_depth.reshape(-1), depth.reshape(-1), mask.reshape(-1))
        mask = mask.reshape(-1)
        predicted_depth = predicted_depth.reshape(-1)[mask]
        depth = depth.reshape(-1)[mask]
        return self.l1_loss(predicted_depth, depth) / float(mask.shape[0])
    
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
    
    def get_flame_sdf_loss(self, head_sdf, flame_sdf):
        mask = (flame_sdf < 0) & (head_sdf > 0) # fill the hole in the back part of the head avatar
        # mask = (flame_sdf is not None) & (sdf is not None)
        # mask = torch.ones_like(flame_sdf).to(flame_sdf.device).bool()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        
        flame_sdf_loss = self.l1_loss(head_sdf[mask], flame_sdf[mask]) / float(mask.sum())
        return flame_sdf_loss
    
    def get_contact_mask_loss(self, sdf_output, pred_mask, gt_mask, occ_mask):
        
        mask = occ_mask
        
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = gt_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(mask.shape[0])
        return mask_loss
    
    def forward(self, model_outputs, ground_truth):
        
        ######################### HEAD ########################
        pred_head_mask = model_outputs['pred_head_mask']
        object_mask = model_outputs['object_mask'] # masks include hand and head

        # rgb_loss_head = self.get_rgb_loss(model_outputs['rgb_values_head'], ground_truth['rgb'], pred_head_mask, model_outputs['head_mask'], model_outputs['pred_hand_mask'], model_outputs['depth_head_before_hand_mask'])
        rgb_loss_head = self.get_rgb_loss(model_outputs['rgb_values_head'], ground_truth['rgb'], pred_head_mask, model_outputs['head_mask'], model_outputs['hand_mask']|model_outputs['pred_hand_mask'])
        rgb_loss_head = rgb_loss_head * self.rgb_weight
        
        mask_loss_head = self.get_mask_loss(model_outputs['sdf_output_head'], pred_head_mask, object_mask, model_outputs['hand_mask']|model_outputs['pred_hand_mask'])
        # mask_loss_head = self.get_mask_loss(model_outputs['sdf_output_head'], pred_head_mask, model_outputs['head_mask']|model_outputs['hand_mask'], model_outputs['hand_mask']|model_outputs['pred_hand_mask'])
        mask_loss_head = self.mask_weight * mask_loss_head

        # eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss_head + mask_loss_head #+ self.eikonal_weight * eikonal_loss
        
        
        out = {
            'loss': loss,
            'rgb_loss_head': rgb_loss_head,
            'mask_loss_head': mask_loss_head,
            # 'eikonal_loss': eikonal_loss,
        }
        
        out['rgb_loss'] = rgb_loss_head

        if self.depth_weight > 0:
            depth_loss_head = self.get_depth_loss(model_outputs['depth_values_head'], ground_truth['depth'], pred_head_mask, model_outputs['head_mask'])
            depth_loss_head = self.depth_weight * depth_loss_head
            out['loss'] += depth_loss_head
            out['depth_loss_head'] = depth_loss_head

        if self.lbs_weight != 0:
            ghostbone = model_outputs['lbs_weight'].shape[1] == 6
            # outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'], model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'], ground_truth['semantics'], model_outputs['pred_head_mask'] & model_outputs['object_mask'], ghostbone)
            outputs = model_outputs
            num_points = model_outputs['lbs_weight'].shape[0]
            if self.gt_w_seg:
                # do not enforce nearest neighbor skinning weight for teeth, learn from data instead.
                # now it's also not enforcing nn skinning wieght for glasses, I'm too lazy to correct it but the skinning weight can still learn correctly for glasses.
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], pred_head_mask, object_mask & (ground_truth['semantics'][:, :, 3].reshape(-1) != 1))
            else:
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], pred_head_mask, object_mask)

            lbs_loss = lbs_loss * self.lbs_weight * 0.1
            out['loss'] += lbs_loss
            out['lbs_loss'] = lbs_loss
            

            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10, outputs['gt_posedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], pred_head_mask, object_mask)
            posedirs_loss = posedirs_loss * self.lbs_weight * 10.0
            out['loss'] += posedirs_loss
            out['posedirs_loss'] = posedirs_loss

            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1) * 10, outputs['gt_shapedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], pred_head_mask, object_mask)
            shapedirs_loss = shapedirs_loss * self.lbs_weight * 10.0
            out['loss'] += shapedirs_loss
            out['shapedirs_loss'] = shapedirs_loss
            
            # posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1)*0.1, outputs['gt_posedirs'].reshape(num_points, -1)*0.1, model_outputs['flame_distance'], pred_head_mask, object_mask)
            # posedirs_loss = posedirs_loss * self.lbs_weight *0.1#* 10.0
            # out['loss'] += posedirs_loss
            # out['posedirs_loss'] = posedirs_loss

            # shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)*0.1, outputs['gt_shapedirs'].reshape(num_points, -1)*0.1, model_outputs['flame_distance'], pred_head_mask, object_mask)
            # shapedirs_loss = shapedirs_loss * self.lbs_weight *0.1#* 10.0
            # out['loss'] += shapedirs_loss
            # out['shapedirs_loss'] = shapedirs_loss

     
        if self.expression_reg_weight != 0 and 'expression' in ground_truth:
            out['expression_reg_loss'] = self.expression_reg_weight * self.get_expression_reg_weight(model_outputs['expression'][..., :50], ground_truth['expression'])
            out['loss'] += out['expression_reg_loss']

        if self.pose_reg_weight != 0 and 'flame_pose' in ground_truth:
            out['pose_reg_loss'] =  self.pose_reg_weight * self.get_expression_reg_weight(model_outputs['flame_pose'], ground_truth['flame_pose'])
            out['loss'] += out['pose_reg_loss']

        if self.cam_reg_weight != 0 and 'cam_pose' in ground_truth:
            out['cam_reg_loss'] = self.cam_reg_weight * self.get_expression_reg_weight(model_outputs['cam_pose'][:, :3, 3], ground_truth['cam_pose'][:, :3, 3])
            out['loss'] += out['cam_reg_loss']
        
        if 'semantics' in ground_truth and self.flame_distance_weight > 0 and self.gt_w_seg:
            flame_distance_loss = self.get_flame_distance_loss(model_outputs['flame_distance'], ground_truth['semantics'], pred_head_mask)
            flame_distance_loss = flame_distance_loss * self.flame_distance_weight
            out['flame_distance_loss'] = flame_distance_loss
            out['loss'] = out['loss'] + flame_distance_loss
            
        flame_sdf_loss = self.get_flame_sdf_loss(model_outputs['sdf_sampleflamebbox_tohead'], model_outputs['sdf_sampleflamebbox_toflame']) # this is to prevent holes in the back of head, which is not optimized well
        flame_sdf_loss = flame_sdf_loss * 0.01
        out['flame_sdf_loss'] = flame_sdf_loss
        out['loss'] = out['loss'] + flame_sdf_loss 
        
        ######################### HAND ########################
        img_res = int(model_outputs['img_res'][0][0].item())
        pred_rgb_img = model_outputs['rgb_image_hand'].reshape(img_res,img_res,3)
        pred_hand_mask_img = model_outputs['mask_image_hand'].reshape(img_res,img_res,1)
        pred_hand_lmk = model_outputs['hand_lmk']
        pred_depth_img = model_outputs['depth_image_hand'].reshape(img_res,img_res,1)
        
        gt_rgb_img = ground_truth['rgb_image_hand'].reshape(img_res,img_res,3)
        gt_hand_mask_img = ground_truth['mask_image_hand'].reshape(img_res,img_res,1)
        
        hand_rgb_loss = self.get_rgb_loss_hand(pred_rgb_img, gt_rgb_img, pred_hand_mask_img, gt_hand_mask_img)
        out['hand_rgb_loss'] = hand_rgb_loss
        out['loss'] = out['loss'] + out['hand_rgb_loss']
        
        if model_outputs['optimize_mano_pose']:
            gt_head_mask_img = ground_truth['mask_image_head'].reshape(img_res,img_res,1)
            gt_depth_img = ground_truth['depth_image_hand'].reshape(img_res,img_res,1)
            gt_hand_lmk = ground_truth['hand_lmk']
            
            hand_mask_loss = self.get_mask_loss_hand(pred_hand_mask_img, gt_hand_mask_img, gt_head_mask_img)
            out['hand_mask_loss'] = hand_mask_loss
            out['loss'] = out['loss'] + out['hand_mask_loss']
        
            hand_landmark_loss = self.get_landmark_loss_hand(pred_hand_lmk[:,:,:2], gt_hand_lmk[:,:,:2])
            out['hand_landmark_loss'] = hand_landmark_loss
            out['loss'] = out['loss'] + out['hand_landmark_loss']
            
            hand_depth_loss = self.get_depth_loss_hand(pred_depth_img, gt_depth_img, pred_hand_mask_img, gt_hand_mask_img)
            out['hand_depth_loss'] = hand_depth_loss
            out['loss'] = out['loss'] + out['hand_depth_loss']
            
            depth_mask_loss = self.get_depth_mask_loss(model_outputs['depth_mask_head'], model_outputs['depth_mask_hand'], model_outputs['head_mask'], model_outputs['hand_mask'])
            depth_mask_loss = depth_mask_loss * 1e-5
            out['depth_mask_loss'] = depth_mask_loss
            out['loss'] = out['loss'] + out['depth_mask_loss']
        
        ######################### CONTACT ########################
        if model_outputs['optimize_contact']:
            contact_out = self.cal_contact_loss(model_outputs)
            for k in contact_out.keys():
                out[k] = contact_out[k]
            
            # out['nonrigid_deformation_min'] = model_outputs['nonrigid_deformation_head'].min()
            # out['nonrigid_deformation_max'] = model_outputs['nonrigid_deformation_head'].max()
            
            # temporal_reg_nonrigid = self.get_temporal_reg_nonrigid(model_outputs['nonrigid_deformation_onhand_tohead'], prev_nonrigid_deformation)
            # temporal_reg_nonrigid = temporal_reg_nonrigid *1000
            # out['temporal_reg_nonrigid'] = temporal_reg_nonrigid
            # out['contact_loss'] = out['contact_loss'] + temporal_reg_nonrigid
                
            # contact_direction_loss = self.get_contact_direction_loss(model_outputs['contact_sdf_head'], model_outputs['contact_sdf_hand'], model_outputs['nonrigid_deformation_detach_bbox'], model_outputs['contact_points_normal_hand'])
            # contact_direction_loss = contact_direction_loss * contact_loss_weight * 1 #* 0.1
            # out['contact_direction_loss'] = contact_direction_loss
            # out['contact_loss'] = out['contact_loss'] + contact_direction_loss
        
        if self.pose_reg_weight != 0 and 'hand_pose' in ground_truth and model_outputs['optimize_mano_pose']:
            out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose'])
            out['loss'] = out['loss'] + out['hand_pose_reg_loss'] * self.pose_reg_weight
        
        if self.pose_reg_weight != 0 and 'mano_transl' in ground_truth and model_outputs['optimize_mano_pose']:
            out['mano_transl_reg_loss'] = self.get_expression_reg_weight(model_outputs['mano_transl'].reshape(-1,3), ground_truth['mano_transl'])
            # out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose']) \
            #                             + self.get_expression_reg_weight(model_outputs['hand_global_orient'].reshape(-1,9), ground_truth['hand_global_orient']) \
            #                             + self.get_expression_reg_weight(model_outputs['hand_transl'].reshape(-1,3), ground_truth['hand_transl'])
            out['loss'] = out['loss'] + out['mano_transl_reg_loss'] * self.pose_reg_weight
            
        return out

    def cal_contact_loss(self, model_outputs):
        out = {}
        out['contact_loss'] = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        # TODO: may not regularize points sampled on hand, which may cause contact loss to not work well
        # contact_regularation = self.get_contact_reg(model_outputs['nonrigid_deformation_onhand_tohead'], model_outputs['sdf_onhand_tohead'], model_outputs['sdf_output_hand']) # reg points on hand
        # contact_regularation = contact_regularation + self.get_contact_reg(model_outputs['nonrigid_deformation_sampleonhand_tohead'], model_outputs['sdf_sampleonhand_tohead'], model_outputs['sdf_sampleonhand']) # reg sampled points on hand
        
        contact_regularation = self.get_contact_reg(model_outputs['nonrigid_deformation_head'], model_outputs['sdf_onhead_tohand'], model_outputs['sdf_output_head']) # reg points on head
        # contact_regularation = contact_regularation + self.get_contact_reg(model_outputs['nonrigid_deformation_sampleonhead'], model_outputs['sdf_sampleonhead_tohand'], model_outputs['sdf_sampleonhead']) # reg sampled points on head
        
        # contact_regularation = (model_outputs['nonrigid_deformation_head']**2).mean() + (model_outputs['nonrigid_deformation_sampleonhead']**2).mean()
        
        contact_regularation = contact_regularation * self.contact_reg_weight
        out['contact_regularation'] = contact_regularation
        out['contact_loss'] = out['contact_loss'] + contact_regularation
        

        # contact_sdf_loss = self.get_contact_loss(model_outputs['sdf_onhand_tohead'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_sampleonhand_tohead'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_onhead_tohand'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_sampleonhead_tohand'])
        
        # contact_sdf_loss = self.get_contact_loss(model_outputs['sdf_onhand_tohead'], model_outputs['sdf_output_hand'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_sampleonhand_tohead'], model_outputs['sdf_sampleonhand'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_output_head'], model_outputs['sdf_onhead_tohand'])
        # contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_sampleonhead'], model_outputs['sdf_sampleonhead_tohand'])
        
        contact_sdf_loss = self.get_contact_loss(model_outputs['sdf_sampleonhand_tohead'], model_outputs['sdf_sampleonhand'])
        
        contact_sdf_loss = contact_sdf_loss * self.contact_sdf_weight 
        out['contact_sdf_loss'] = contact_sdf_loss
        out['contact_loss'] = out['contact_loss'] + contact_sdf_loss
        
        pred_head_mask = model_outputs['pred_head_mask']
        object_mask = model_outputs['object_mask']
        num_points = model_outputs['lbs_weight'].shape[0]
        if 'nonrigid_dir' in model_outputs.keys():
            nonrigid_dir_loss = self.get_lbs_loss(model_outputs['nonrigid_dir'].reshape(num_points, -1) * 10, model_outputs['gt_nonrigid_dir'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], pred_head_mask, object_mask)
            nonrigid_dir_loss = nonrigid_dir_loss * self.lbs_weight * 10 #100.0 #* 10.0 
            out['nonrigid_dir_loss'] = nonrigid_dir_loss 
            out['contact_loss'] = out['contact_loss'] + nonrigid_dir_loss 
            
        # contact_mask_loss = self.get_contact_mask_loss(model_outputs['sdf_output_head'], pred_head_mask, object_mask, model_outputs['depth_head_before_hand_mask']!=model_outputs['pred_head_mask'])
        # out['contact_mask_loss'] = contact_mask_loss * self.mask_weight
        # out['contact_loss'] = out['contact_loss'] + contact_mask_loss 
        
        # depth_mask_loss = self.get_depth_mask_loss(model_outputs['depth_head_before_hand_mask'], model_outputs['depth_hand_before_head_mask'], model_outputs['head_mask'], model_outputs['hand_mask'])
        # depth_mask_loss = depth_mask_loss * self.mask_weight
        # out['depth_mask_loss'] = depth_mask_loss
        # out['contact_loss'] = out['contact_loss'] + out['depth_mask_loss']
            
        return out