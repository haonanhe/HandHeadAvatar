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
        if (network_object_mask & object_mask & flame_distance_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        lbs_weight = lbs_weight[network_object_mask & flame_distance_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & flame_distance_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(network_object_mask.shape[0])
        
        return lbs_loss
    
    def get_nonrigid_lbs_loss(self, lbs_weight, gt_lbs_weight, flame_distance, network_object_mask, object_mask):
        flame_distance_mask = flame_distance < 0.1
        if (network_object_mask & object_mask & flame_distance_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        lbs_weight = lbs_weight[network_object_mask & flame_distance_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & flame_distance_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(network_object_mask.shape[0])
        
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
    

    def get_contact_reg(self, nonrigid_deformation, sdf_head=None, sdf_hand=None, mask=None):
        if sdf_head is None or sdf_hand is None:
            return (nonrigid_deformation**2).mean()
        if nonrigid_deformation is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        sdf_head = sdf_head.reshape(-1)
        sdf_hand = sdf_hand.reshape(-1)

        # Contact region definition
        contact_core_margin = 0.005  # 5mm - realistic contact threshold
        transition_margin = 0.02  # 20mm transition zone from core boundary

        contact_core_mask = (sdf_head < 0) & (sdf_hand.abs() < contact_core_margin)

        dist_head_condition = torch.relu(sdf_head)  # 0 if sdf_head < 0, else sdf_head
        dist_hand_condition = torch.relu(sdf_hand.abs() - contact_core_margin)
        
        dist_to_contact = torch.maximum(dist_head_condition, dist_hand_condition)

        sharpness = 6.0
        reg_weight = torch.sigmoid(sharpness * (dist_to_contact / transition_margin - 0.5))

        deform_magnitude = (nonrigid_deformation ** 2).mean(dim=-1)
        
        weighted_reg = deform_magnitude * reg_weight

        return weighted_reg.mean()

    def get_contact_loss(self, sdf_head, sdf_hand, surf_mask=None):
        if sdf_head is None:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        sdf_head = sdf_head.reshape(-1)
        sdf_hand = sdf_hand.reshape(-1)

        # Use same margin as contact_reg for consistency
        margin = 0.005  # 5mm - realistic contact threshold

        if surf_mask is None:
            # Surface contact: head near zero-level set AND hand inside/near surface
            # This encourages the head surface to align with hand surface in contact regions
            # surface_mask = (sdf_head.abs() < margin) & (sdf_hand < margin)
            surface_mask = (-margin < sdf_head) & (sdf_head < 0) & (sdf_hand < margin)
        else:
            surface_mask = surf_mask

        # Surface loss: minimize head SDF in contact regions (push head surface to zero-level)
        surface_loss = (sdf_head[surface_mask]**2).mean() if surface_mask.sum() > 0 else torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        # Inside loss: prevent head from going too far inside hand
        # When head is deeply inside hand (sdf_head < -margin) and hand surface is inside (sdf_hand < 0)
        inside_mask = (sdf_head < -margin) & (sdf_hand < 0)
        inside_loss = torch.relu(-sdf_head[inside_mask] - margin).mean() if inside_mask.sum() > 0 else torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        loss = surface_loss + inside_loss
        return loss
    

    def get_rgb_loss_hand(self, pred_rgb_img, gt_rgb_img, pred_hand_mask_img, gt_hand_mask_img):
        pred_hand_mask_img = pred_hand_mask_img.bool()
        gt_hand_mask_img = gt_hand_mask_img.bool()
        
        mask = pred_hand_mask_img & gt_hand_mask_img 
        
        pred_rgb_img = pred_rgb_img * mask
        gt_rgb_img = gt_rgb_img * mask
        rgb_loss = self.l1_loss(pred_rgb_img, gt_rgb_img) / float(mask.sum())
        return rgb_loss
    
    def get_depth_loss(self, predicted_depth, depth, network_object_mask, head_mask):
        mask = network_object_mask & head_mask      #     return loss
        # return
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        return self.depth_loss(predicted_depth[None, :, None], depth[:, :, None], mask[None, :, None])
    
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
    
    def get_flame_sdf_loss(self, head_sdf, flame_sdf):
        mask = (flame_sdf < 0) & (head_sdf > 0) # fill the hole in the back part of the head avatar
        # mask = (flame_sdf is not None) & (sdf is not None)
        # mask = torch.ones_like(flame_sdf).to(flame_sdf.device).bool()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)
        
        flame_sdf_loss = self.l1_loss(head_sdf[mask], flame_sdf[mask]) / float(mask.sum())
        return flame_sdf_loss
    
    def forward(self, model_outputs, ground_truth):
        
        ######################### HEAD ########################
        pred_head_mask = model_outputs['pred_head_mask']
        object_mask = model_outputs['head_mask']
        body_mask = model_outputs['object_mask'] & (~model_outputs['head_mask']) & pred_head_mask

        rgb_loss_head = self.get_rgb_loss(model_outputs['rgb_values_head'], ground_truth['rgb'], pred_head_mask, model_outputs['head_mask'], model_outputs['hand_mask']|model_outputs['pred_hand_mask']|body_mask)
        rgb_loss_head = rgb_loss_head * self.rgb_weight
        
        mask_loss_head = self.get_mask_loss(model_outputs['sdf_output_head'], pred_head_mask, object_mask, model_outputs['hand_mask']|model_outputs['pred_hand_mask']|body_mask)
        mask_loss_head = self.mask_weight * mask_loss_head

        loss = rgb_loss_head + mask_loss_head 
        
        
        out = {
            'loss': loss,
            'rgb_loss_head': rgb_loss_head,
            'mask_loss_head': mask_loss_head,
        }
        
        out['rgb_loss'] = rgb_loss_head

        if self.depth_weight > 0:
            depth_loss_head = self.get_depth_loss(model_outputs['depth_values_head'], ground_truth['depth'], pred_head_mask, model_outputs['head_mask'])
            depth_loss_head = self.depth_weight * depth_loss_head
            out['loss'] += depth_loss_head
            out['depth_loss_head'] = depth_loss_head

        if self.lbs_weight != 0:
            ghostbone = model_outputs['lbs_weight'].shape[1] == 6
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
        
        ######################### CONTACT ########################
        if model_outputs['optimize_contact']:
            contact_out = self.cal_contact_loss(model_outputs, ground_truth)
            for k in contact_out.keys():
                out[k] = contact_out[k]
            
        if self.pose_reg_weight != 0 and 'hand_pose' in ground_truth and model_outputs['optimize_mano_pose']:
            out['hand_pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['hand_pose'].reshape(-1,135), ground_truth['hand_pose'])
            out['loss'] = out['loss'] + out['hand_pose_reg_loss'] * self.pose_reg_weight
        
        if self.pose_reg_weight != 0 and 'mano_transl' in ground_truth and model_outputs['optimize_mano_pose']:
            out['mano_transl_reg_loss'] = self.get_expression_reg_weight(model_outputs['mano_transl'].reshape(-1,3), ground_truth['mano_transl'])
            out['loss'] = out['loss'] + out['mano_transl_reg_loss'] * self.pose_reg_weight
            
        return out

    
    def cal_contact_loss(self, model_outputs, ground_truth):
        out = {}
        out['contact_loss'] = torch.tensor(0.0, device="cuda", dtype=torch.float32, requires_grad=True)

        # Contact regularization: only regularize non-contact regions
        contact_regularation = self.get_contact_reg(model_outputs['nonrigid_deformation_head'], model_outputs['sdf_output_head'], model_outputs['sdf_onhead_tohand']) # reg points on head
        contact_regularation = contact_regularation + self.get_contact_reg(model_outputs['nonrigid_deformation_sampleonhead'], model_outputs['sdf_sampleonhead'], model_outputs['sdf_sampleonhead_tohand']) # reg sampled points on head

        contact_regularation = contact_regularation * self.contact_reg_weight
        out['contact_regularation'] = contact_regularation
        out['contact_loss'] = out['contact_loss'] + contact_regularation

        # Contact SDF loss: encourage contact deformation
        contact_sdf_loss = self.get_contact_loss(model_outputs['sdf_sampleonhand_tohead'], model_outputs['sdf_sampleonhand'])
        contact_sdf_loss = contact_sdf_loss + self.get_contact_loss(model_outputs['sdf_sampleonhead'], model_outputs['sdf_sampleonhead_tohand'])

        contact_sdf_loss = contact_sdf_loss * self.contact_sdf_weight
        out['contact_sdf_loss'] = contact_sdf_loss
        out['contact_loss'] = out['contact_loss'] + contact_sdf_loss

        # Nonrigid direction consistency loss
        pred_head_mask = model_outputs['pred_head_mask']
        object_mask = model_outputs['object_mask']
        num_points = model_outputs['nonrigid_dir'].shape[0]

        if 'nonrigid_dir' in model_outputs.keys():
            nonrigid_dir_loss = self.get_nonrigid_lbs_loss(model_outputs['nonrigid_dir'].reshape(num_points, -1) * 10, model_outputs['gt_nonrigid_dir'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], pred_head_mask, object_mask)
            nonrigid_dir_loss = nonrigid_dir_loss * self.lbs_weight * 0.1
            out['nonrigid_dir_loss'] = nonrigid_dir_loss
            out['contact_loss'] = out['contact_loss'] + nonrigid_dir_loss

        return out