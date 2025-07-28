import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import datetime
import json
import pickle
import open3d as o3d
from glob import glob
from PIL import Image
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.io import save_obj, load_obj
import shutil
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, face_normals, compute_vertex_normals

from pysdf import SDF

from pyhocon import ConfigFactory

import gc

from tqdm import tqdm
# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
GLOBAL_POSE = True
# GLOBAL_POSE = False

import cv2
import argparse

import sys
sys.path.append('../code')
from external.body_models import MANOLayer
from model.monosdf_loss import ScaleAndShiftInvariantLoss

sys.path.append('./submodules/DECA')
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

import wandb

np.random.seed(0)

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

def cal_bary_weights(points, verts, faces):
    B = verts.shape[0]
    
    verts = verts.reshape(B, -1, 3)
    # faces = faces.reshape(B, -1, 3)
    faces = faces[None,...].expand(B, -1, 3)

    triangles = face_vertices(verts, faces)

    bary_weights_list = []
    prev_bary_weights_list = []
    closest_triangles_list = []
    
    residues, pts_ind, _ = point_to_mesh_distance(points[0].view(1, -1, 3).contiguous(), triangles[:1].view(1, -1, 3, 3))
    closest_triangles = torch.gather(
        triangles[:1].view(1, -1, 3, 3), 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(
        points[0].view(-1, 3), closest_triangles)
    bary_weights_list.append(bary_weights)
    prev_bary_weights_list.append(bary_weights)
    closest_triangles_list.append(closest_triangles)
    
    for i in range(1, B):
        residues, pts_ind, _ = point_to_mesh_distance(points[i].view(1, -1, 3).contiguous(), triangles[i].view(1, -1, 3, 3))
        closest_triangles = torch.gather(
            triangles[i].view(1, -1, 3, 3), 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        closest_triangles_list.append(closest_triangles)
        bary_weights = barycentric_coordinates_of_projection(
            points[i].view(-1, 3), closest_triangles.view(-1, 3, 3))
        bary_weights_list.append(bary_weights)
        prev_bary_weights = barycentric_coordinates_of_projection(
            points[i].view(-1, 3), closest_triangles_list[i-1].view(-1, 3, 3))
        prev_bary_weights_list.append(prev_bary_weights)
    
    bary_weights = torch.stack(bary_weights_list)
    prev_bary_weights = torch.stack(prev_bary_weights_list)
    
    return bary_weights, prev_bary_weights

def barycentric_coordinates_of_projection_batch(points, vertices):
    """
    Compute barycentric coordinates for batched points and triangles.
    
    :param points: Points to project [B, N, 3]
    :param vertices: Triangle vertices [B, 3, 3]
    :returns: Barycentric coordinates [B, N, 3]
    """
    # Extract vertices
    v0 = vertices[:, :, 1, :]  # [B, 1, 3]
    v1 = vertices[:, :, 1, :]  # [B, 1, 3]
    v2 = vertices[:, :, 2, :]  # [B, 1, 3]
    
    # Reshape points to match vertices
    p = points  # [B, N, 3]
    
    # Compute triangle vectors and normal
    u = v1 - v0  # [B, 1, 3]
    v = v2 - v0  # [B, 1, 3]
    n = torch.cross(u, v, dim=-1)  # [B, 1, 3]
    
    # Compute triangle area
    s = torch.sum(n * n, dim=-1)  # [B, 1]
    
    # Handle degenerate triangles
    s = torch.where(s == 0, torch.full_like(s, 1e-6), s)
    oneOver4ASquared = 1.0 / s  # [B, 1]
    
    # Compute intermediate vectors
    w = p - v0  # [B, N, 3]
    
    # Compute barycentric coordinates
    b2 = torch.sum(torch.cross(u, w, dim=-1) * n, dim=-1) * oneOver4ASquared  # [B, N]
    b1 = torch.sum(torch.cross(w, v, dim=-1) * n, dim=-1) * oneOver4ASquared  # [B, N]
    
    # Combine weights
    weights = torch.stack([1 - b1 - b2, b1, b2], dim=-1)  # [B, N, 3]
    
    return weights

def cal_bary_weights_batch(points, verts, faces):
    B = verts.shape[0]
    
    verts = verts.reshape(B, -1, 3)
    # faces = faces.reshape(B, -1, 3)
    faces = faces[None,...].expand(B, -1, 3)

    Bsize = points.shape[0]

    triangles = face_vertices(verts, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points.contiguous(), triangles)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3))#.view(-1, 3, 3)

    prev_closest_triangles = torch.cat([closest_triangles[0][None,...], closest_triangles[:-1]], dim=0)
    # ()
    bary_weights = barycentric_coordinates_of_projection_batch(points.view(B, -1, 3), closest_triangles.view(B, -1, 3, 3))
    prev_bary_weights = barycentric_coordinates_of_projection_batch(points.view(B, -1, 3), prev_closest_triangles.view(B, -1, 3, 3))
    
    return bary_weights, prev_bary_weights
    
def save_point_cloud_to_ply(points, colors=None, filename="point_cloud.ply"):
    """
    Save point cloud to a PLY file.
    
    Args:
    points (numpy.ndarray): Nx3 array of point coordinates
    colors (numpy.ndarray, optional): Nx3 array of RGB colors (values 0-255)
    filename (str): Output filename (should end with .ply)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        if colors.dtype != np.uint8:
            colors = (colors * 255).astype(np.uint8)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def l2_distance(verts1, verts2, weight=None):
    # if verts1.shape[1] == 21:
    #     # finger_tips = [4,8,12,16,20]
    #     # index_finger = [5,6,7,8]
    #     index_finger = [8] # index finger tip
    #     weight = torch.ones_like(verts1[...,:1]).to(verts1.device)
    #     # weight[:,8:9,:] *= 10 # index finger tip
    #     # weight[:,finger_tips,:] *= 10
    #     weight[:,index_finger,:] *= 30
    #     # weights[:,2:5,:] *= 0.00001 # thumb finger
    if weight is not None:
        return torch.sqrt((((verts1 - verts2)**2)*weight).sum(2)).mean(1).mean()
    else:
        return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()
    
# def l2_distance_hand(verts1, verts2, weight=None):
#     if verts1.shape[1] == 21:
#         finger_tips = [4,8,12,16,20]
#         # index_finger = [5,6,7,8]
#         # index_finger = [8] # index finger tip
#         weight = torch.ones_like(verts1[...,:1]).to(verts1.device)
#         # weight[:,8:9,:] *= 10 # index finger tip
#         # weight[:,finger_tips,:] *= 10
#         weight[:,finger_tips,:] *= 2
#         # weights[:,2:5,:] *= 0.00001 # thumb finger
#     if weight is not None:
#         return torch.sqrt((((verts1 - verts2)**2)*weight).sum(2)).mean(1).mean() / verts1.shape[1]
#     else:
#         return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean() / verts1.shape[1]

def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d


def inverse_projection(points2d, K, c2w):
    i = points2d[:, :, 0]
    j = points2d[:, :, 1]
    dirs = torch.stack([(i - K[2]) / K[0], (j - K[3]) / K[1], torch.ones_like(i) * -1], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, :3, :3], -1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def _load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def depth_mse_loss(prediction, target, valid_mask=None) -> float:
    if valid_mask is not None:
        non_zero_mask = valid_mask.bool()
        non_zero_mask = non_zero_mask.reshape(prediction.shape[0],prediction.shape[1],prediction.shape[2])
        prediction = prediction[non_zero_mask]
        target = target[non_zero_mask]
    loss = torch.mean((prediction - target) ** 2)

    return loss

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def modify_batched_masks(A):
    """
    Modify batched binary masks.
    
    Args:
    A (torch.Tensor): Batched binary masks of shape (B, H, W)
    
    Returns:
    torch.Tensor: Modified masks of shape (B, H, W)
    """
    # Ensure A is a boolean tensor
    A = A.bool()
    
    # Create output tensor with the same shape and device as input
    B = torch.zeros_like(A)
    
    # Process each mask in the batch
    for i in range(A.shape[0]):
        # Current mask
        mask = A[i]
        
        # Find the indices where the mask is True
        rows, cols = torch.nonzero(mask, as_tuple=True)
        
        # Skip if no True values in the mask
        if len(rows) == 0:
            continue
        
        # Get the bounding box coordinates
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # scale bounding box
        scale = 1.5
        center_row = (min_row + max_row) / 2
        half_size_row = (max_row - min_row) / 2
        scaled_half_size_row = half_size_row * scale
        min_row = center_row - scaled_half_size_row
        max_row = center_row + scaled_half_size_row
        center_col = (min_col + max_col) / 2
        half_size_col = (max_col - min_col) / 2
        scaled_half_size_col = half_size_col * scale
        min_col = center_col - scaled_half_size_col
        max_col = center_col + scaled_half_size_col
        min_row, max_row = min_row.int(), max_row.int()
        min_col, max_col = min_col.int(), max_col.int()
        
        # Create a white mask for the bounding box
        B[i, min_row:max_row+1, min_col:max_col+1] = True
        
        # Restore the originally white pixels to black
        B[i][mask] = False
    
    return B
                             
class SmallDataset(Dataset):
    def __init__(self, depth_images_pths, shape, exp, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, translation_v, hand_mask_pths, mano_scale, head_mask_pths, translation_f, betas_ori, hand_poses_ori, global_orient_ori, weight_hand_lmks, scales_all, flame_scale, translation_p):
        """
        Args:
            image_paths (list): List of paths to the depth images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.depth_images_pths = depth_images_pths
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),       
        ])
        self.shape = shape
        self.exp = exp
        self.face_landmarks = face_landmarks
        self.face_poses = face_poses
        self.betas = betas
        self.global_orient = global_orient
        self.hand_landmarks = hand_landmarks
        self.hand_poses = hand_poses
        self.translation_p = translation_p 
        self.translation_v = translation_v
        self.mano_scale = mano_scale
        self.translation_f = translation_f

        self.hand_mask_pths = hand_mask_pths
        self.head_mask_pths = head_mask_pths
        
        self.betas_ori = betas_ori
        self.hand_poses_ori = hand_poses_ori
        self.global_orient_ori = global_orient_ori

        self.weight_hand_lmks = weight_hand_lmks

        # self.poses_all = poses_all
        self.depth_scale = 1000.
        self.scales_all = scales_all

        self.flame_scale = flame_scale

    def __len__(self):
        return len(self.depth_images_pths)

    def __getitem__(self, idx):
        depth_img_pth = self.depth_images_pths[idx]
        depth_img = Image.open(depth_img_pth)
        if self.transform:
            depth_img = self.transform(depth_img)
        depth_img = depth_img.convert('L')  
        depth_img = np.array(depth_img, dtype=np.float32)
        depth_img /= 255.0 
        depth_img = torch.tensor(depth_img, dtype=torch.float32)
        depth_img = 1 - depth_img

        hand_mask_pth = self.hand_mask_pths[idx]
        hand_mask = Image.open(hand_mask_pth)
        if self.transform:
            hand_mask = self.transform(hand_mask)
        hand_mask = hand_mask.convert('L')  
        hand_mask = np.array(hand_mask, dtype=np.float32)
        hand_mask /= 255.0 
        hand_mask = torch.tensor(hand_mask, dtype=torch.float32)

        head_mask_pth = self.head_mask_pths[idx]
        head_mask = Image.open(head_mask_pth)
        if self.transform:
            head_mask = self.transform(head_mask)
        head_mask = head_mask.convert('L')  
        head_mask = np.array(head_mask, dtype=np.float32)
        head_mask /= 255.0 
        head_mask = torch.tensor(head_mask, dtype=torch.float32)

        hand_landmarks = self.hand_landmarks[idx]
        face_landmarks = self.face_landmarks[idx]

        shape = self.shape#[idx]
        exp = self.exp[idx]
        
        face_poses = self.face_poses[idx]
        betas = self.betas[idx]
        global_orient = self.global_orient[idx]
        hand_poses = self.hand_poses[idx]

        translation_p = self.translation_p[idx]
        translation_v = self.translation_v[idx]
        translation_f = self.translation_f[idx]

        mano_scale = self.mano_scale[idx]
        
        betas_ori = self.betas_ori[idx]
        hand_poses_ori = self.hand_poses_ori[idx]
        global_orient_ori = self.global_orient_ori[idx]

        flame_scale = self.flame_scale[idx]

        out_dict = {
            'depth_img': depth_img,
            'shape': shape,
            'exp': exp,
            'face_landmarks': face_landmarks,
            'face_poses': face_poses,
            'betas': betas,
            'global_orient': global_orient,
            'hand_landmarks': hand_landmarks,
            'hand_poses': hand_poses,
            'translation_p': translation_p,
            'translation_v': translation_v,
            'hand_mask': hand_mask,
            'mano_scale': mano_scale,
            'head_mask': head_mask,
            'translation_f': translation_f,
            'betas_ori': betas_ori,
            'hand_poses_ori': hand_poses_ori,
            'global_orient_ori': global_orient_ori,
            # 'poses': poses_all,
            'idx': torch.from_numpy(np.array(idx)).long(),
            'flame_scale': flame_scale
            # 'weight_hand_lmks': weight_hand_lmks
        }

        return out_dict

class SlidingWindowDataset(Dataset):
    def __init__(self, original_dataset, window_size, stride):
        self.original_dataset = original_dataset  # 原始数据集
        self.window_size = window_size            # 窗口大小（batch size）
        self.stride = stride                      # 滑动步长

    def __len__(self):
        return (len(self.original_dataset) - self.window_size) // self.stride + 1
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        # 这里会多次调用 original_dataset.__getitem__(i)
        # return [self.original_dataset[i] for i in range(start, end)]
        
        # Get all samples in the sliding window
        samples = [self.original_dataset[i] for i in range(start, end)]
        
        # Stack each field in the dict
        stacked_dict = {
            key: torch.stack([sample[key] for sample in samples])
            for key in samples[0].keys()  # Assumes all samples have the same keys
        }
        
        return stacked_dict
    
class DepthRankingLoss(nn.Module):
    def __init__(self, margin=1e-6):
        """
        Initializes the depth ranking loss.

        Args:
            margin: The distance margin between the ranked depths.
        """
        super(DepthRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, pred_depth_hand, pred_depth_head, gt_depth, mask1, mask2):
        B, H, W = pred_depth_hand.shape
        # num_samples_per_batch = 10000 // B
        num_samples_per_batch = 1000

        # Get valid pixel indices for all batches at once
        batch_idx1 = []
        batch_idx2 = []
        y1_all = []
        x1_all = []
        y2_all = []
        x2_all = []

        # Prepare indices for all batches in parallel
        for b in range(B):
            valid_idx1 = torch.nonzero(mask1[b]).float()  # [N1, 2] with (y,x) coordinates
            valid_idx2 = torch.nonzero(mask2[b]).float()  # [N2, 2] with (y,x) coordinates
            
            if valid_idx1.size(0) == 0 or valid_idx2.size(0) == 0:
                continue

            # Random sampling for this batch
            rand_idx1 = torch.randint(0, valid_idx1.size(0), (num_samples_per_batch,))
            rand_idx2 = torch.randint(0, valid_idx2.size(0), (num_samples_per_batch,))
            
            sampled_points1 = valid_idx1[rand_idx1]  # [num_samples_per_batch, 2]
            sampled_points2 = valid_idx2[rand_idx2]  # [num_samples_per_batch, 2]
            
            # Store batch index and coordinates
            batch_idx1.append(torch.full((num_samples_per_batch,), b, device=pred_depth_hand.device))
            batch_idx2.append(torch.full((num_samples_per_batch,), b, device=pred_depth_hand.device))
            
            y1_all.append(sampled_points1[:, 0])
            x1_all.append(sampled_points1[:, 1])
            y2_all.append(sampled_points2[:, 0])
            x2_all.append(sampled_points2[:, 1])

        if not batch_idx1:  # If no valid samples were found
            return torch.tensor(0.0, device=pred_depth_hand.device, requires_grad=True)

        # Concatenate all indices
        batch_idx1 = torch.cat(batch_idx1)
        batch_idx2 = torch.cat(batch_idx2)
        y1_all = torch.cat(y1_all).long()
        x1_all = torch.cat(x1_all).long()
        y2_all = torch.cat(y2_all).long()
        x2_all = torch.cat(x2_all).long()

        # Gather all depths in parallel
        pred_depth_1 = pred_depth_hand[batch_idx1, y1_all, x1_all]
        pred_depth_2 = pred_depth_head[batch_idx2, y2_all, x2_all]
        gt_depth_1 = gt_depth[batch_idx1, y1_all, x1_all]
        gt_depth_2 = gt_depth[batch_idx2, y2_all, x2_all]

        # Compute loss in parallel
        gt_order = torch.sign(gt_depth_1 - gt_depth_2)
        pred_diff = pred_depth_1 - pred_depth_2
        loss = torch.clamp(self.margin - gt_order * pred_diff, min=0)

        return loss.mean()

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity
    
def depth_evaluation(predicted_depth_original, ground_truth_depth_original, mask=None, max_depth=80, custom_mask=None, post_clip_min=None, post_clip_max=None, pre_clip_min=None, pre_clip_max=None,
                     align_with_lstsq=False, align_with_lad=False, align_with_lad2=False, lr=1e-4, max_iters=1000, use_gpu=False, align_with_scale=False,
                     disp_input=True):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    ground_truth_depth_original = torch.clamp(ground_truth_depth_original, 0.1, 1)
    
    if mask is not None:
        predicted_depth = predicted_depth_original[mask]
        ground_truth_depth = ground_truth_depth_original[mask]
    else:
        predicted_depth = predicted_depth_original.clone()
        ground_truth_depth = ground_truth_depth_original.clone()
    
    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input: # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)
        
    # Convert to numpy for lstsq
    predicted_depth_np = predicted_depth.detach().cpu().numpy().reshape(-1, 1)
    ground_truth_depth_np = ground_truth_depth.detach().cpu().numpy().reshape(-1, 1)
    
    # Add a column of ones for the shift term
    A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
    
    # ()
    
    # Solve for scale (s) and shift (t) using least squares
    result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
    s, t = result[0][0], result[0][1]

    # convert to torch tensor
    s = torch.tensor(s, device=predicted_depth_original.device)
    t = torch.tensor(t, device=predicted_depth_original.device)
    
    # Apply scale and shift
    predicted_depth = s * predicted_depth + t

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    # Calculate the metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth)#.item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth)#.item()
    
    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2))#.item()
    
    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2))#.item()
    
    predicted_depth_original = predicted_depth_original * s + t
    if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)

    results = {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'Log RMSE': log_rmse,
    }

    return results, predicted_depth_original

def bidirectional_contact_loss(verts_p_hand, verts_p_face):
    # 手部到面部的最近距离
    dist_hand_to_face = torch.cdist(verts_p_hand, verts_p_face).min(dim=1)[0]
    
    # 面部到手部的最近距离
    dist_face_to_hand = torch.cdist(verts_p_face, verts_p_hand).min(dim=1)[0]
    
    # 组合两个方向的损失
    loss = (dist_hand_to_face.mean() + dist_face_to_hand.mean()) / 2
    
    return loss

class Optimizer(object):
    def __init__(self, device='cuda:0', save_folder=None, conf=None):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device, image_size=512, uv_size=512)
        self.deca_optim = DECA(config=deca_cfg, device=device, image_size=128, uv_size=128)

        self.MANOServer = MANOLayer(model_path="../code/mano_model/data/mano",
                                    is_rhand=True,
                                    batch_size=1,
                                    flat_hand_mean=False,
                                    dtype=torch.float32,
                                    use_pca=False,).cuda()

        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        self.depth_ranking_loss = DepthRankingLoss()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        os.environ['WANDB_DIR'] = os.path.join(save_folder)
        name = save_folder.split('/')[-2]
        wandb.init(project='HHMavatar_preprocess', name=name + '_preprocess')
        
        self.conf = conf
        self.conf_loss = self.conf.get_config('loss')
        self.optimize_depth_rank = self.conf_loss['optimize_depth_rank']
        self.contact_frame_idx = self.conf_loss['contact_frame_idx']

    def optimize(self, shape, exp, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name, depth_images_pths, hand_mask_pths, head_mask_pths, image_pths, weight_hand_lmks=None):
        num_img = face_poses.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()

        save_intrinsics = torch.tensor([-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
            intrinsics[3] / size]).float().cuda()

        K = torch.eye(3)
        K[0,0] = save_intrinsics[0] * 512
        K[1,1] = save_intrinsics[1] * 512
        K[0,2] = save_intrinsics[2] * 512
        K[1,2] = save_intrinsics[3] * 512

        with open("../code/mano_model/data/contact_zones.pkl", "rb") as f:
            contact_zones = pickle.load(f)
        contact_zones = contact_zones["contact_zones"]
        contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])
        # contact_idx = contact_idx[19:47] #index finger
        # contact_idx = contact_idx[-17:-1] # thumb
        # contact_idx = np.concatenate([contact_idx[19:47], contact_idx[-17:-1]], axis=0)
        # contact_idx =contact_idx[19:] #all fingers
        
        index_contact_idx = contact_idx[19:47] #index finger
        thumb_contact_idx = contact_idx[-17:-1]
        all_contact_idx = contact_idx[19:]

        with open("../preprocess/submodules/DECA/data/FLAME_masks.pkl", "rb") as f:
            flame_masks = pickle.load(f, encoding='latin1')
        flame_face_masks_full = flame_masks["face"]
        
        with open("../preprocess/submodules/DECA/data/face_idx.pkl", "rb") as f:
            flame_face_masks = pickle.load(f)
        flame_face_masks = np.array(flame_face_masks)
        # flame_face_masks = torch.from_numpy(flame_face_masks).cuda()
        # # flame_face_masks = flame_face_masks[:24]
        # # flame_face_masks = flame_face_masks[::2]
        
        with open("../preprocess/submodules/DECA/data/right_face_touch_region.pkl", "rb") as f:
            flame_face_masks_right_cheeck = pickle.load(f)
        flame_face_masks_right_cheeck = np.array(flame_face_masks_right_cheeck)
        
        flame_face_masks_left_cheeck = result = np.setdiff1d(flame_face_masks, flame_face_masks_right_cheeck)

        if GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -4]).float().cuda()
        else:
            translation_p = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
        
        translation_v = torch.tensor([0, 0, 0]).unsqueeze(0).expand(num_img, -1).float().cuda()
        translation_f = torch.tensor([0, 0, 0]).unsqueeze(0).expand(num_img, -1).float().cuda()
        mano_scale = torch.tensor([4]).unsqueeze(0).expand(num_img, -1).float().cuda()
        flame_scale = torch.tensor([4]).unsqueeze(0).expand(num_img, -1).float().cuda()

        if GLOBAL_POSE:
            face_poses = torch.cat([torch.zeros_like(face_poses[:, :3]), face_poses], dim=1)
        use_iris = False
        if face_landmarks.shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True
        if use_iris:
            face_poses = torch.cat([face_poses, torch.zeros_like(face_poses[:, :6])], dim=1)

        face_poses = nn.Parameter(face_poses)
        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)
        # flame_scale = nn.Parameter(flame_scale)
        # translation_f = nn.Parameter(translation_f)
        
        hand_poses = nn.Parameter(hand_poses)
        betas = nn.Parameter(betas)
        mano_scale = nn.Parameter(mano_scale)
        translation_v = nn.Parameter(translation_v)
        
        translation_p = nn.Parameter(translation_p)
        
        # Load all depth images and masks upfront
        depth_images = []
        hand_masks = []
        head_masks = []
        # transform = transforms.Compose([transforms.Resize((512, 512))])
        transform = transforms.Compose([transforms.Resize((128, 128))])
        
        for i in range(num_img):
            # Load depth image
            depth_img = Image.open(depth_images_pths[i])
            depth_img = transform(depth_img.convert('L'))
            depth_img = 1 - (torch.tensor(np.array(depth_img, dtype=np.float32)/255.0).float().cuda())
            depth_images.append(depth_img)
            
            # Load hand mask
            hand_mask = Image.open(hand_mask_pths[i])
            hand_mask = transform(hand_mask.convert('L'))
            hand_mask = torch.tensor(np.array(hand_mask, dtype=np.float32)/255.0).float().cuda()
            hand_masks.append(hand_mask)
            
            # Load head mask
            head_mask = Image.open(head_mask_pths[i])
            head_mask = transform(head_mask.convert('L'))
            head_mask = torch.tensor(np.array(head_mask, dtype=np.float32)/255.0).float().cuda()
            head_masks.append(head_mask)
        
        depth_images = torch.stack(depth_images)
        hand_masks = torch.stack(hand_masks)
        head_masks = torch.stack(head_masks)
        
       ################## Stage I ##################

        # opt_t = torch.optim.Adam(
        #     [translation_v, translation_f, face_poses, mano_scale, flame_scale], 
        #     lr=1e-2)
        opt_t = torch.optim.Adam(
            [translation_v, translation_p, face_poses, mano_scale], 
            lr=1e-2)
        opt_p = torch.optim.Adam(
            [exp, shape, betas, hand_poses, global_orient],
            lr=1e-4)

        # optimization steps
        len_landmark_face = face_landmarks.shape[1]
        len_landmark_hand = hand_landmarks.shape[1]
        hand_landmarks = (hand_landmarks[..., :2] - (size/2)) / (size/2)
        
        for k in tqdm(range(1501)):
        # for k in tqdm(range(1801)):
        # for k in range(2001):
        # for k in range(101):
            face_full_pose = face_poses
            if not use_iris:
                face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
            
            verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                   expression_params=exp,
                                                                                   full_pose=face_full_pose)
            
            verts_p_face *= flame_scale.unsqueeze(1)
            landmarks3d_p_face *= flame_scale.unsqueeze(1)
            landmarks2d_p_face *= flame_scale.unsqueeze(1)
            verts_p_face += translation_f.unsqueeze(1)
            landmarks3d_p_face += translation_f.unsqueeze(1)
            landmarks2d_p_face += translation_f.unsqueeze(1)

            pred_mano_params = {
                'global_orient': global_orient,
                'hand_pose': hand_poses,
                # 'betas': betas.expand(num_img, -1),
                'betas': betas,
                'transl': translation_v,
                'scale': mano_scale
            }
            mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
            verts_p_hand = mano_output.vertices.clone()
            landmarks3d_p_hand = mano_output.joints.clone()
            
            verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)

            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            if GLOBAL_POSE:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

            trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
            trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
            landmark = torch.cat([face_landmarks, hand_landmarks], dim=1)
            
            ## landmark loss
            trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
            trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
            trans_landmarks3d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p, no_intrinsics=True)
            trans_landmarks3d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
            landmark = torch.cat([hand_landmarks, face_landmarks], dim=1)
            
            ## landmark loss
            landmark_loss2 = l2_distance(trans_landmarks2d_face[:, :len_landmark_face, :2], face_landmarks[:, :len_landmark_face]) 
            landmark_loss2 += l2_distance(trans_landmarks2d_hand[:, :len_landmark_hand, :2], hand_landmarks[:, :len_landmark_hand])
            landmark_loss2 = landmark_loss2
            total_loss = landmark_loss2

            smooth_loss = 0
            smooth_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            smooth_loss += torch.mean(torch.square(face_poses[1:] - face_poses[:-1])) * 10
            # smooth_loss += torch.mean(torch.square(translation_f[1:] - translation_f[:-1])) * 10
            
            smooth_loss += torch.mean(torch.square(global_orient[1:] - global_orient[:-1])) * 10
            smooth_loss += torch.mean(torch.square(hand_poses[1:] - hand_poses[:-1])) * 10
            smooth_loss += torch.mean(torch.square(translation_v[1:] - translation_v[:-1])) * 10
            smooth_loss += torch.mean(torch.square(mano_scale[1:] - mano_scale[:-1])) * 10
            
            # smooth_loss += torch.mean(torch.square(flame_scale[1:] - flame_scale[:-1]))
            # smooth_loss += torch.mean(torch.square(trans_landmarks3d_face[1:] - trans_landmarks3d_face[:-1])) * 100
            # smooth_loss += torch.mean(torch.square(trans_landmarks3d_hand[1:] - trans_landmarks3d_hand[:-1])) * 100
            # smooth_loss += torch.mean(torch.square(verts_p_face[1:] - verts_p_face[:-1]))
            # smooth_loss += torch.mean(torch.square(verts_p_hand[1:] - verts_p_hand[:-1]))
            smooth_loss = smooth_loss
            total_loss += smooth_loss
            
            # # contact_loss = 0
            # # contact loss
            # # mano_vertices_tips = verts_p_hand
            # mano_vertices_tips = verts_p_hand[:,contact_idx,:]
            # knn_v = verts_p_face.detach()[:,flame_face_masks,:].clone()
            # contact_loss = ((knn_points(mano_vertices_tips, knn_v, K=6, return_nn=False)[0])**2).mean()
            # contact_loss = contact_loss * self.conf_loss['contact_weight']
            # total_loss += contact_loss 
            
            # mask_loss = 0
            # depth_ranking_loss = 0
            
            # # if k % 50 == 0:
            # # if k % 50 == 0 and k != 0 and k > 500:
            # if k > 1400 and self.optimize_depth_rank:
            #     trans_verts_hand = projection(verts_p_hand, cam_intrinsics, w2c_p)
            #     trans_verts_hand_cam = projection(verts_p_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
            #     trans_verts_head = projection(verts_p_face, cam_intrinsics, w2c_p)
            #     trans_verts_head_cam = projection(verts_p_face, cam_intrinsics, w2c_p, no_intrinsics=True)
            #     trans_verts = projection(verts_p, cam_intrinsics, w2c_p)
            #     trans_verts_cam = projection(verts_p, cam_intrinsics, w2c_p, no_intrinsics=True)
                
            #     pred_depth_images_hand_list, pred_mask_hand_list = [], []
            #     pred_depth_images_head_list, pred_mask_head_list = [], []
            #     pred_depth_images_list, pred_mask_list = [], []
                
            #     chunk_size = 1
            #     for i in range(0, verts_p.shape[0], chunk_size):
            #         pred_depth_images_hand, pred_mask_hand = self.deca_optim.render_hand.render_depth(trans_verts_hand[i:i+chunk_size], trans_verts_hand_cam[i:i+chunk_size], render_mano=True)
            #         pred_depth_images_hand = pred_depth_images_hand.reshape(-1,128,128)
            #         pred_mask_hand = pred_mask_hand.reshape(-1,128,128).bool()
                    
            #         pred_depth_images_head, pred_mask_head = self.deca_optim.render.render_depth(trans_verts_head[i:i+chunk_size], trans_verts_head_cam[i:i+chunk_size])
            #         pred_depth_images_head = pred_depth_images_head.reshape(-1,128,128)
            #         pred_mask_head = pred_mask_head.reshape(-1,128,128).bool()

            #         # pred_depth_images, pred_mask = self.deca_optim.render_hand_head.render_depth(trans_verts[i:i+chunk_size], trans_verts_cam[i:i+chunk_size])
            #         # pred_depth_images = pred_depth_images.reshape(-1,128,128)
            #         # pred_mask = pred_mask.reshape(-1,128,128).bool()
                    
            #         pred_depth_images_hand_list.append(pred_depth_images_hand)
            #         pred_mask_hand_list.append(pred_mask_hand)
            #         pred_depth_images_head_list.append(pred_depth_images_head)
            #         pred_mask_head_list.append(pred_mask_head)
            #         # pred_depth_images_list.append(pred_depth_images)
            #         # pred_mask_list.append(pred_mask)
                
            #     pred_depth_images_hand = torch.cat(pred_depth_images_hand_list, dim=0)
            #     pred_mask_hand = torch.cat(pred_mask_hand_list, dim=0)
            #     pred_depth_images_head = torch.cat(pred_depth_images_head_list, dim=0)
            #     pred_mask_head = torch.cat(pred_mask_head_list, dim=0)
            #     # pred_depth_images = torch.cat(pred_depth_images_list, dim=0)
            #     # pred_mask = torch.cat(pred_mask_list, dim=0)
                
            #     b_hand_mask = hand_masks.clone().bool().cuda()
            #     b_head_mask = head_masks.clone().bool().cuda()
            #     b_depth_img = depth_images.clone().cuda()
            
            #     bz = b_depth_img.shape[0]
            #     pred_mask_hand = pred_mask_hand.bool() #& b_hand_mask.bool() #& (~b_head_mask.bool())
            #     overlap_mask_hand = pred_mask_hand.bool() & pred_mask_head.bool() #& b_hand_mask.bool() 
            #     bbox_mask_head = modify_batched_masks(overlap_mask_hand.reshape(bz,128,128))
            #     bbox_mask_head = bbox_mask_head & pred_mask_head.bool() & (~pred_mask_hand) & (~b_hand_mask.bool())

            #     # # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128), (pred_mask_head & b_head_mask.bool()).reshape(bz,128,128)) # whole hand and whole head
            #     # # depth_ranking_loss = depth_ranking_loss + self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), bbox_mask_head.reshape(bz,128,128)) # address the overlap area
            #     # # depth_ranking_loss = depth_ranking_loss + self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_hand.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128)) # to make sure the depth order inside the hand is correct (global orient)
                
            #     # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), bbox_mask_head.reshape(bz,128,128)) # address the overlap area
            #     depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), (pred_mask_head & b_head_mask.bool()).reshape(bz,128,128)) # address the overlap area
                
            #     # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images.reshape(bz,128,128), pred_depth_images.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (b_hand_mask.bool()).reshape(bz,128,128), (b_head_mask.bool()).reshape(bz,128,128)) # whole hand and whole head
                
            #     depth_ranking_loss = depth_ranking_loss * self.conf_loss['depth_rank_weight']
            #     total_loss += depth_ranking_loss 
                
            #     # pred_mask_compose = torch.zeros_like(pred_mask_hand, dtype=torch.float, device=pred_mask_hand.device)
            #     # pred_mask_hand = pred_mask_hand.bool()
            #     # pred_mask_head = pred_mask_head.bool()
            #     # overlap = pred_mask_hand & pred_mask_head
            #     # hand_only = pred_mask_hand & ~overlap
            #     # head_only = pred_mask_head & ~overlap
            #     # pred_mask_compose[hand_only] = 1.
            #     # pred_mask_compose[head_only] = 2.
            #     # if overlap.any():
            #     #     hand_depth_overlap = pred_depth_images_hand[overlap]
            #     #     head_depth_overlap = pred_depth_images_head[overlap]
            #     #     closer_hand = hand_depth_overlap < head_depth_overlap
            #     #     pred_mask_compose[overlap] = torch.where(closer_hand, 1., 2.)
                    
            #     # gt_mask_compose = torch.zeros_like(b_hand_mask).float().to(b_hand_mask.device)
            #     # gt_mask_compose[b_hand_mask==True] = 1.
            #     # gt_mask_compose[b_head_mask==True] = 2.
            #     # mask_loss = self.cross_entropy_loss(pred_mask_compose[overlap], gt_mask_compose[overlap]) / float(overlap.sum()) 
            #     # mask_loss = mask_loss * self.conf_loss['mask_weight']
            #     # total_loss += mask_loss 
                
            opt_p.zero_grad()
            opt_t.zero_grad()
            total_loss.backward()
            opt_p.step()
            opt_t.step()

            # visualize
            if k % 100 == 0:
                # if k > 198:
                #     del trans_verts_hand, trans_verts_hand_cam, trans_verts_head, trans_verts_head_cam, trans_verts, trans_verts_cam
                #     del b_hand_mask, b_head_mask, b_depth_img, pred_depth_images_hand, pred_mask_hand, pred_depth_images_head, pred_mask_head, pred_depth_images, pred_mask, pred_mask_compose, gt_mask_compose, bbox_mask_head
                #     torch.cuda.empty_cache()
                # if k > 1400 and self.optimize_depth_rank:
                #     del trans_verts_hand, trans_verts_hand_cam, trans_verts_head, trans_verts_head_cam, trans_verts, trans_verts_cam
                #     del b_hand_mask, b_head_mask, b_depth_img
                #     # del pred_depth_images, pred_mask
                #     # del bbox_mask_head
                #     del pred_depth_images_hand, pred_mask_hand, pred_depth_images_head, pred_mask_head
                #     # del pred_mask_compose, gt_mask_compose, overlap, hand_only, head_only, hand_depth_overlap, head_depth_overlap, closer_hand
                #     torch.cuda.empty_cache()  # Frees up unused memory on GPU
                #     gc.collect()  # Forces garbage collection
                with torch.no_grad():
                    # loss_info = '----iter: {}, time: {}\n'.format(k,
                    #                                               datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    # loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                    
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    
                    acc_loss = {}
                    acc_loss['landmark_loss'] = landmark_loss2
                    acc_loss['smooth_loss'] = smooth_loss
                    # acc_loss['contact_loss'] = contact_loss
                    # acc_loss['mask_loss'] = mask_loss
                    # acc_loss['depth_ranking_loss'] = depth_ranking_loss
                    wandb.log(acc_loss)
                    
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}' \
                                          + f', smooth_loss: {smooth_loss}'\
                                        #   + f', contact_loss: {contact_loss}' \
                                        #   + f', mask_loss: {mask_loss}' \
                                        #   + f', depth_ranking_loss: {depth_ranking_loss}' \
                                       
                    print(loss_info)
                    
                    # verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)
                    # trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_verts_cam = projection(verts_p[::50], cam_intrinsics, w2c_p[::50], no_intrinsics=True)
                    # shape_images = self.deca.render_hand_head.render_shape(verts_p[::50], trans_verts)
                    # depth_images_vis, _ = self.deca.render_hand_head.render_depth(trans_verts, trans_verts_cam)
                    # visdict = {
                    #     # 'inputs': visualize_images,
                    #     'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50], isScale=True),
                    #     'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50], isScale=True),
                    #     'shape_images': shape_images,
                    #     'depth_images': depth_images_vis
                    # }
                    # cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))
        
        ################## Stage II ##################
        opt_t = torch.optim.Adam(
            [translation_v, mano_scale],
            lr=1e-2)
        # opt_p = torch.optim.Adam(
        #     [exp, shape, betas, hand_poses, global_orient],
        #     lr=1e-4)
        opt_p = torch.optim.Adam(
            [global_orient],
            lr=1e-4)

        # # optimization steps
        # len_landmark_face = face_landmarks.shape[1]
        # len_landmark_hand = hand_landmarks.shape[1]
        # hand_landmarks = (hand_landmarks[..., :2] - (size/2)) / (size/2)
        
        # for k in tqdm(range(0)):
        for k in tqdm(range(1501)):
        # for k in range(2001):
        # for k in range(101):
            face_full_pose = face_poses
            if not use_iris:
                face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
            
            verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                   expression_params=exp,
                                                                                   full_pose=face_full_pose)
            
            verts_p_face *= flame_scale.unsqueeze(1)
            landmarks3d_p_face *= flame_scale.unsqueeze(1)
            landmarks2d_p_face *= flame_scale.unsqueeze(1)
            verts_p_face += translation_f.unsqueeze(1)
            landmarks3d_p_face += translation_f.unsqueeze(1)
            landmarks2d_p_face += translation_f.unsqueeze(1)

            pred_mano_params = {
                'global_orient': global_orient,
                'hand_pose': hand_poses,
                # 'betas': betas.expand(num_img, -1),
                'betas': betas,
                'transl': translation_v,
                'scale': mano_scale
            }
            mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
            verts_p_hand = mano_output.vertices.clone()
            landmarks3d_p_hand = mano_output.joints.clone()
            
            verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)

            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            if GLOBAL_POSE:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

            ## landmark loss
            trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
            trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
            trans_landmarks3d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p, no_intrinsics=True)
            trans_landmarks3d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
            landmark = torch.cat([hand_landmarks, face_landmarks], dim=1)
            
            ## landmark loss
            # landmark_loss2 = l2_distance(trans_landmarks2d_face[:, :len_landmark_face, :2], face_landmarks[:, :len_landmark_face]) 
            landmark_loss2 = l2_distance(trans_landmarks2d_hand[:, :len_landmark_hand, :2], hand_landmarks[:, :len_landmark_hand])
            landmark_loss2 = landmark_loss2 * self.conf_loss['landmark_weight']
            total_loss = landmark_loss2

            smooth_loss = 0
            # smooth_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            # smooth_loss += torch.mean(torch.square(face_poses[1:] - face_poses[:-1]))
            smooth_loss += torch.mean(torch.square(global_orient[1:] - global_orient[:-1]))* 10 #* 1e-1
            smooth_loss += torch.mean(torch.square(hand_poses[1:] - hand_poses[:-1]))* 10
            # smooth_loss += torch.mean(torch.square(translation_f[1:] - translation_f[:-1])) #* 10
            smooth_loss += torch.mean(torch.square(translation_v[1:] - translation_v[:-1])) * 10#* 10
            # smooth_loss += torch.mean(torch.square(trans_landmarks3d_face[1:] - trans_landmarks3d_face[:-1])) * 100
            # smooth_loss += torch.mean(torch.square(trans_landmarks3d_hand[1:] - trans_landmarks3d_hand[:-1])) * 100
            smooth_loss += torch.mean(torch.square(mano_scale[1:] - mano_scale[:-1]))* 10
            # smooth_loss += torch.mean(torch.square(flame_scale[1:] - flame_scale[:-1]))
            # smooth_loss += torch.mean(torch.square(verts_p_face[1:] - verts_p_face[:-1]))
            # smooth_loss += torch.mean(torch.square(verts_p_hand[1:] - verts_p_hand[:-1]))
            smooth_loss = smooth_loss * self.conf_loss['smooth_weight']
            total_loss += smooth_loss
            
            # contact_loss = 0
            # contact loss
            # mano_vertices_tips = verts_p_hand
            # mano_vertices_tips = verts_p_hand[:,contact_idx,:]
            # knn_v = verts_p_face.detach()[:,flame_face_masks,:].clone()
            # contact_loss = ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()
            # contact_loss = contact_loss * self.conf_loss['contact_weight']
            # total_loss += contact_loss 
            
           
            mano_vertices_tips = verts_p_hand[:self.contact_frame_idx,:,:][:,index_contact_idx,:] 
            knn_v = verts_p_face.detach()[:self.contact_frame_idx,:,:][:,flame_face_masks_right_cheeck,:].clone()
            contact_loss = ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()
            
            mano_vertices_tips = verts_p_hand[self.contact_frame_idx:,:,:][:,thumb_contact_idx,:]
            knn_v = verts_p_face.detach()[self.contact_frame_idx:,:,:][:,flame_face_masks_right_cheeck,:].clone()
            contact_loss = contact_loss + ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()
            
            mano_vertices_tips = verts_p_hand[self.contact_frame_idx:,:,:][:,index_contact_idx,:]
            knn_v = verts_p_face.detach()[self.contact_frame_idx:,:,:][:,flame_face_masks_left_cheeck,:].clone()
            contact_loss = contact_loss + ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()
            
            contact_loss = contact_loss * self.conf_loss['contact_weight']
            total_loss += contact_loss 
            
            # ############
            # flame_face_masks_right_cheeks = torch.where(verts_p_face[0][flame_face_masks][:, 0] < 0)[0]
            # flame_face_masks_right_cheeks = flame_face_masks[flame_face_masks_right_cheeks]
            # with open('/home/haonan/data/HHAvatar/preprocess/submodules/DECA/data/right_face_touch_region.pkl', 'wb') as f:
            #     pickle.dump(flame_face_masks_right_cheeks.cpu().numpy(), f)
            
            # colors = torch.tensor([
            #     [0.5, 0.5, 0.5],  # Green
            # ], dtype=torch.float32).repeat(verts_p_face[0].shape[0],1).float().cuda()
            # red_color = torch.tensor([
            #     [1, 0, 0],  # Red
            # ], dtype=torch.float32).float().cuda()
            # colors[flame_face_masks_right_cheeks] = red_color
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(verts_p_face[0].detach().cpu().numpy())
            # point_cloud.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy()*255)
            # save_dir = '/home/haonan/data/HHAvatar/preprocess'
            # o3d.io.write_point_cloud(os.path.join(save_dir, "contact_points_colored_points_head.ply"), point_cloud)
            # breakpoint()
            
            # colors = torch.tensor([
            #     [0.5, 0.5, 0.5],  # Green
            # ], dtype=torch.float32).repeat(verts_p_hand[0].shape[0],1).float().cuda()
            # red_color = torch.tensor([
            #     [1, 0, 0],  # Red
            # ], dtype=torch.float32).float().cuda()
            # colors[contact_idx] = red_color
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(verts_p_hand[0].detach().cpu().numpy())
            # point_cloud.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy()*255)
            # save_dir = '/home/haonan/data/HHAvatar/preprocess'
            # o3d.io.write_point_cloud(os.path.join(save_dir, "contact_points_colored_points_hand.ply"), point_cloud)
            # breakpoint()
            ############
            
            mask_loss = 0
            depth_ranking_loss = 0
            
            # if k % 50 == 0:
            # if k % 50 == 0 and k != 0 and k > 500:
            if k > 1000 and self.optimize_depth_rank:
                trans_verts_hand = projection(verts_p_hand, cam_intrinsics, w2c_p)
                trans_verts_hand_cam = projection(verts_p_hand, cam_intrinsics, w2c_p, no_intrinsics=True)
                trans_verts_head = projection(verts_p_face, cam_intrinsics, w2c_p)
                trans_verts_head_cam = projection(verts_p_face, cam_intrinsics, w2c_p, no_intrinsics=True)
                trans_verts = projection(verts_p, cam_intrinsics, w2c_p)
                trans_verts_cam = projection(verts_p, cam_intrinsics, w2c_p, no_intrinsics=True)
                
                pred_depth_images_hand_list, pred_mask_hand_list = [], []
                pred_depth_images_head_list, pred_mask_head_list = [], []
                pred_depth_images_list, pred_mask_list = [], []
                
                chunk_size = 1
                for i in range(0, verts_p.shape[0], chunk_size):
                    pred_depth_images_hand, pred_mask_hand = self.deca_optim.render_hand.render_depth(trans_verts_hand[i:i+chunk_size], trans_verts_hand_cam[i:i+chunk_size], render_mano=True)
                    pred_depth_images_hand = pred_depth_images_hand.reshape(-1,128,128)
                    pred_mask_hand = pred_mask_hand.reshape(-1,128,128).bool()
                    
                    pred_depth_images_head, pred_mask_head = self.deca_optim.render.render_depth(trans_verts_head[i:i+chunk_size], trans_verts_head_cam[i:i+chunk_size])
                    pred_depth_images_head = pred_depth_images_head.reshape(-1,128,128)
                    pred_mask_head = pred_mask_head.reshape(-1,128,128).bool()

                    # pred_depth_images, pred_mask = self.deca_optim.render_hand_head.render_depth(trans_verts[i:i+chunk_size], trans_verts_cam[i:i+chunk_size])
                    # pred_depth_images = pred_depth_images.reshape(-1,128,128)
                    # pred_mask = pred_mask.reshape(-1,128,128).bool()
                    
                    pred_depth_images_hand_list.append(pred_depth_images_hand)
                    pred_mask_hand_list.append(pred_mask_hand)
                    pred_depth_images_head_list.append(pred_depth_images_head)
                    pred_mask_head_list.append(pred_mask_head)
                    # pred_depth_images_list.append(pred_depth_images)
                    # pred_mask_list.append(pred_mask)
                
                pred_depth_images_hand = torch.cat(pred_depth_images_hand_list, dim=0)
                pred_mask_hand = torch.cat(pred_mask_hand_list, dim=0)
                pred_depth_images_head = torch.cat(pred_depth_images_head_list, dim=0)
                pred_mask_head = torch.cat(pred_mask_head_list, dim=0)
                # pred_depth_images = torch.cat(pred_depth_images_list, dim=0)
                # pred_mask = torch.cat(pred_mask_list, dim=0)
                
                b_hand_mask = hand_masks.clone().bool().cuda()
                b_head_mask = head_masks.clone().bool().cuda()
                b_depth_img = depth_images.clone().cuda()
            
                bz = b_depth_img.shape[0]
                pred_mask_hand = pred_mask_hand.bool() #& b_hand_mask.bool() #& (~b_head_mask.bool())
                overlap_mask_hand = pred_mask_hand.bool() & pred_mask_head.bool() #& b_hand_mask.bool() 
                bbox_mask_head = modify_batched_masks(overlap_mask_hand.reshape(bz,128,128))
                bbox_mask_head = bbox_mask_head & pred_mask_head.bool() & (~pred_mask_hand) & (~b_hand_mask.bool())

                # # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128), (pred_mask_head & b_head_mask.bool()).reshape(bz,128,128)) # whole hand and whole head
                # # depth_ranking_loss = depth_ranking_loss + self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), bbox_mask_head.reshape(bz,128,128)) # address the overlap area
                # # depth_ranking_loss = depth_ranking_loss + self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_hand.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128), (pred_mask_hand & b_hand_mask.bool()).reshape(bz,128,128)) # to make sure the depth order inside the hand is correct (global orient)
                
                # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), bbox_mask_head.reshape(bz,128,128)) # address the overlap area
                depth_ranking_loss = self.depth_ranking_loss(pred_depth_images_hand.reshape(bz,128,128), pred_depth_images_head.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), overlap_mask_hand.reshape(bz,128,128), (pred_mask_head & b_head_mask.bool()).reshape(bz,128,128)) # address the overlap area
                
                # depth_ranking_loss = self.depth_ranking_loss(pred_depth_images.reshape(bz,128,128), pred_depth_images.reshape(bz,128,128), b_depth_img.reshape(bz,128,128), (b_hand_mask.bool()).reshape(bz,128,128), (b_head_mask.bool()).reshape(bz,128,128)) # whole hand and whole head
                
                depth_ranking_loss = depth_ranking_loss * self.conf_loss['depth_rank_weight']
                total_loss += depth_ranking_loss 
                
                # pred_mask_compose = torch.zeros_like(pred_mask_hand, dtype=torch.float, device=pred_mask_hand.device)
                # pred_mask_hand = pred_mask_hand.bool()
                # pred_mask_head = pred_mask_head.bool()
                # overlap = pred_mask_hand & pred_mask_head
                # hand_only = pred_mask_hand & ~overlap
                # head_only = pred_mask_head & ~overlap
                # pred_mask_compose[hand_only] = 1.
                # pred_mask_compose[head_only] = 2.
                # if overlap.any():
                #     hand_depth_overlap = pred_depth_images_hand[overlap]
                #     head_depth_overlap = pred_depth_images_head[overlap]
                #     closer_hand = hand_depth_overlap < head_depth_overlap
                #     pred_mask_compose[overlap] = torch.where(closer_hand, 1., 2.)
                    
                # gt_mask_compose = torch.zeros_like(b_hand_mask).float().to(b_hand_mask.device)
                # gt_mask_compose[b_hand_mask==True] = 1.
                # gt_mask_compose[b_head_mask==True] = 2.
                # mask_loss = self.cross_entropy_loss(pred_mask_compose[overlap], gt_mask_compose[overlap]) / float(overlap.sum()) 
                # mask_loss = mask_loss * self.conf_loss['mask_weight']
                # total_loss += mask_loss 
                
            opt_p.zero_grad()
            opt_t.zero_grad()
            total_loss.backward()
            opt_p.step()
            opt_t.step()

            # visualize
            if k % 100 == 0:
                # if k > 198:
                #     del trans_verts_hand, trans_verts_hand_cam, trans_verts_head, trans_verts_head_cam, trans_verts, trans_verts_cam
                #     del b_hand_mask, b_head_mask, b_depth_img, pred_depth_images_hand, pred_mask_hand, pred_depth_images_head, pred_mask_head, pred_depth_images, pred_mask, pred_mask_compose, gt_mask_compose, bbox_mask_head
                #     torch.cuda.empty_cache()
                if k > 1000 and self.optimize_depth_rank:
                    del trans_verts_hand, trans_verts_hand_cam, trans_verts_head, trans_verts_head_cam, trans_verts, trans_verts_cam
                    del b_hand_mask, b_head_mask, b_depth_img
                    # del pred_depth_images, pred_mask
                    # del bbox_mask_head
                    del pred_depth_images_hand, pred_mask_hand, pred_depth_images_head, pred_mask_head
                    # del pred_mask_compose, gt_mask_compose, overlap, hand_only, head_only, hand_depth_overlap, head_depth_overlap, closer_hand
                    torch.cuda.empty_cache()  # Frees up unused memory on GPU
                    gc.collect()  # Forces garbage collection
                with torch.no_grad():
                    # loss_info = '----iter: {}, time: {}\n'.format(k,
                    #                                               datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    # loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                    
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    
                    acc_loss = {}
                    acc_loss['landmark_loss'] = landmark_loss2
                    acc_loss['smooth_loss'] = smooth_loss
                    acc_loss['contact_loss'] = contact_loss
                    acc_loss['mask_loss'] = mask_loss
                    acc_loss['depth_ranking_loss'] = depth_ranking_loss
                    wandb.log(acc_loss)
                    
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}' \
                                          + f', smooth_loss: {smooth_loss}'\
                                          + f', contact_loss: {contact_loss}' \
                                          + f', mask_loss: {mask_loss}' \
                                          + f', depth_ranking_loss: {depth_ranking_loss}' \
                                       
                    print(loss_info)
                    
                    # verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)
                    # trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_verts_cam = projection(verts_p[::50], cam_intrinsics, w2c_p[::50], no_intrinsics=True)
                    # shape_images = self.deca.render_hand_head.render_shape(verts_p[::50], trans_verts)
                    # depth_images_vis, _ = self.deca.render_hand_head.render_depth(trans_verts, trans_verts_cam)
                    # visdict = {
                    #     # 'inputs': visualize_images,
                    #     'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50], isScale=True),
                    #     'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50], isScale=True),
                    #     'shape_images': shape_images,
                    #     'depth_images': depth_images_vis
                    # }
                    # cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))
                    
        # save vis video
        save_imgs_dir = os.path.join(savefolder, 'preprocess_shape_images')
        if not os.path.exists(save_imgs_dir):
            os.makedirs(save_imgs_dir)
        else:
            shutil.rmtree(save_imgs_dir)
            os.makedirs(save_imgs_dir)

        contact_bbox_list = []
        contact_bbox_2d_list = []
        contact_map_list = []
        head_contact_idx_list = []
        for i in range(num_img):
            ### contact_bbox
            f = SDF(verts_p_face[i].detach().cpu().numpy(), self.deca.flame.faces_tensor.detach().cpu().numpy())
            hand_verts = verts_p_hand[i].detach().cpu().numpy()
            head_verts = verts_p_face[i].detach().cpu().numpy()
            hand_insides = f.contains(hand_verts)
            head_contact_idx = f.nn(hand_verts[hand_insides])
            if head_contact_idx.shape[0] > 0:
                head_contact_idx = head_contact_idx[np.isin(head_contact_idx, flame_face_masks_full)]
            
            # contact_map = torch.zeros(verts_p_face[i].shape[0], 1)
            # contact_map[head_contact_idx] = 1
            # contact_bbox = None
            # contact_bbox_2d = None
            # if hand_insides.sum() > 0 and head_contact_idx.shape[0] > 0:
            #     hand_verts_insides = hand_verts[hand_insides]
            #     head_verts_contact = head_verts[head_contact_idx]
            #     contact_verts = np.concatenate([head_verts_contact, hand_verts_insides], axis=0)
            #     min_coords = np.min(contact_verts, axis=0).tolist()
            #     max_coords = np.max(contact_verts, axis=0).tolist()
            #     contact_bbox = min_coords + max_coords

            #     head_verts_contact_2d = torch.tensor(head_verts_contact, dtype=torch.float32).reshape(1,-1,3).to(verts_p_face.device)
            #     # 
            #     trans_head_verts_contact_2d = projection(head_verts_contact_2d, cam_intrinsics, w2c_p[i][None,...])
            #     trans_head_verts_contact_2d = trans_head_verts_contact_2d.detach().cpu().numpy()[...,:2]
            #     trans_head_verts_contact_2d = trans_head_verts_contact_2d.reshape(-1, 2)
            #     min_coords_2d = np.min(trans_head_verts_contact_2d, axis=0).tolist()
            #     max_coords_2d = np.max(trans_head_verts_contact_2d, axis=0).tolist()
            #     contact_bbox_2d = min_coords_2d + max_coords_2d
            #     contact_bbox_2d = contact_bbox_2d * 4

            # contact_bbox_list.append(contact_bbox)
            # contact_bbox_2d_list.append(contact_bbox_2d)
            # contact_map_list.append(contact_map)
            head_contact_idx_list.append(head_contact_idx)
            

        # del contact_map, contact_bbox
        del head_contact_idx
        torch.cuda.empty_cache()

        chunk_size = 10
        for i in range(0, verts_p.shape[0], chunk_size):
            # trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
            trans_landmarks2d_face_vis = projection(landmarks3d_p_face[i:i+chunk_size], cam_intrinsics, w2c_p[i:i+chunk_size])
            trans_landmarks2d_hand_vis = projection(landmarks3d_p_hand[i:i+chunk_size], cam_intrinsics, w2c_p[i:i+chunk_size])
            trans_landmarks2d_vis = torch.cat([trans_landmarks2d_face_vis, trans_landmarks2d_hand_vis], dim=1)
            gt_trans_landmarks2d_vis = torch.cat([face_landmarks[i:i+chunk_size], hand_landmarks[i:i+chunk_size]], dim=1)

            # if i % 4 == 0:
            trans_verts = projection(verts_p[i:i+chunk_size], cam_intrinsics, w2c_p[i:i+chunk_size])

            num_frames = verts_p[i:i+chunk_size].shape[0]
            colors = torch.tensor([180, 180, 180])[None, None, :].repeat(num_frames, self.deca.render_hand_head.faces.max()+1, 1).float()/255.
            head_contact_indices = head_contact_idx_list[i:i+chunk_size]
            for j, indices in enumerate(head_contact_indices):
                if len(indices) > 0:  # Check if the sub-array is non-empty
                    colors[j, indices] = torch.tensor([255, 0, 0]).float()/255.
            face_colors = face_vertices(colors, self.deca.render_hand_head.faces.repeat(num_frames,1,1).to(colors.device)).to(verts_p.device)

            shape_images = self.deca.render_hand_head.render_shape(verts_p[i:i+chunk_size], trans_verts, colors=face_colors)
            for j in range(shape_images.shape[0]):
                k = i + j
                image = cv2.imread(image_pths[k]).astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                image = torch.from_numpy(image[None, :, :, :]).cuda()
                
                visdict = {
                    'gt_landmarks2d': util.tensor_vis_landmarks(image, gt_trans_landmarks2d_vis.detach()[j][None,...], isScale=True),
                    'landmarks2d': util.tensor_vis_landmarks(image, trans_landmarks2d_vis.detach()[j][None,...], isScale=True),
                    'inputs': image,
                    'shape_images': shape_images[j][None,...],
                }
                shape_image = self.deca.visualize(visdict)
                cv2.imwrite(os.path.join(save_imgs_dir, '%07d.jpg'%k), shape_image)

        del shape_images, head_contact_idx_list#, contact_bbox_list

        cmd = f'ffmpeg -framerate 20 -pattern_type glob -i "{save_imgs_dir}/*.jpg" -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y {save_imgs_dir}/../preprocess_shape_images.mp4'
        os.system(cmd)
        
        save = True
        if save:
            dict = {}
            frames = []

            mesh_save_dir = os.path.join(savefolder, 'preprocess_posed_meshes')
            if not os.path.exists(mesh_save_dir):
                os.mkdir(mesh_save_dir)
            else:
                shutil.rmtree(mesh_save_dir)
                os.mkdir(mesh_save_dir)

            hand_head_obj = '../code/mano_model/data/hand_head_uv_template_closed.obj'
            verts, faces, aux = load_obj(hand_head_obj)
            hand_head_mesh_faces = faces.verts_idx

            for idx in range(num_img):
                if idx % 50 == 0:
                    save_pth = os.path.join(mesh_save_dir, '%07d.obj'%idx)
                    save_obj(save_pth, verts_p[idx].reshape(-1,3), hand_head_mesh_faces.reshape(-1,3).cuda())

                frames.append({'file_path': './image/' + name[idx],
                            'world_mat': w2c_p[idx].detach().cpu().numpy().tolist(),
                            'expression': exp[idx].detach().cpu().numpy().tolist(),
                            'scales_all': self.scales_all[idx].tolist(),
                            # 'head_pose': face_full_pose[idx].detach().cpu().numpy().tolist(),
                            'pose': face_full_pose[idx].detach().cpu().numpy().tolist(),
                            'head_transl': translation_f[idx].detach().cpu().numpy().tolist(),
                            'global_orient': global_orient[idx].detach().cpu().numpy().tolist(),
                            'hand_pose': hand_poses[idx].detach().cpu().numpy().tolist(),
                            'betas': betas[idx].detach().cpu().numpy().tolist(),
                            'mano_scale': mano_scale[idx].detach().cpu().numpy().tolist(),
                            'hand_transl': translation_v[idx].detach().cpu().numpy().tolist(),
                            'bbox': torch.stack(
                                [torch.min(landmark[idx, :, 0]), torch.min(landmark[idx, :, 1]),
                                    torch.max(landmark[idx, :, 0]), torch.max(landmark[idx, :, 1])],
                                dim=0).detach().cpu().numpy().tolist(),
                            'flame_keypoints': trans_landmarks2d_face[idx, :,
                                                :2].detach().cpu().numpy().tolist(),
                            'mano_keypoints': trans_landmarks2d_hand[idx, :,
                                                :2].detach().cpu().numpy().tolist(),
                            'gt_mano_keypoints': hand_landmarks[idx, :,
                                                :2].detach().cpu().numpy().tolist(),
                            # 'contact_bbox': contact_bbox_list[idx],
                            # 'contact_bbox_2d': contact_bbox_2d_list[idx],
                            # 'contact_map': contact_map_list[idx].detach().cpu().numpy().tolist(),
                            'flame_scale': flame_scale[idx].detach().cpu().numpy().tolist(),
                            })

            dict['frames'] = frames
            dict['intrinsics'] = save_intrinsics.detach().cpu().numpy().tolist()
            dict['cam_intrinsics'] = cam_intrinsics.detach().cpu().numpy().tolist()
            dict['shape_params'] = shape[0].detach().cpu().numpy().tolist()
            dict['shape_params_hand'] = betas.detach().cpu().numpy().tolist()
            with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                json.dump(dict, fp)


    def run(self, deca_code_file, face_kpts_file, iris_file, mano_file, savefolder, image_path, json_path, intrinsics, size,
            save_name, conf):
        
        deca_code = json.load(open(deca_code_file, 'r'))
        face_kpts = json.load(open(face_kpts_file, 'r'))
        iris_kpts = None
        print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        face_landmarks = []
        face_poses = []
        name = []
        
        with open(mano_file,'rb') as f:
            mano_dict = pickle.load(f)
            
        # sample_intervals = 2
        sample_intervals = 1
        # sample_intervals = 10
        keys_to_slice = [item for item in list(mano_dict)[::sample_intervals]]
        
        mano_dict_sliced = {k: mano_dict[k] for k in keys_to_slice}
        mano_dict = mano_dict_sliced
        names = [str(k).split('/')[-1][:-4] for k in mano_dict]
        
        deca_code_sliced = {k: deca_code[k] for k in names}
        deca_code = deca_code_sliced
        

        for k_str in deca_code.keys():
            k = int(k_str)
            shape.append(torch.tensor(deca_code[k_str]['shape']).float().cuda())
            exps.append(torch.tensor(deca_code[k_str]['exp']).float().cuda())
            face_poses.append(torch.tensor(deca_code[k_str]['pose']).float().cuda())
            name.append(k_str)
            try:
                landmark = np.array(face_kpts['{}.png'.format(k_str)]).astype(np.float32)
            except:
                landmark = np.array(face_kpts['{}.png'.format(str(f"{k-1:07d}"))]).astype(np.float32)
            if iris_kpts is not None:
                iris = np.array(iris_kpts['{}.png'.format(k_str)]).astype(np.float32).reshape(2, 2)
                landmark = np.concatenate([landmark, iris[[1,0], :]], 0)
            landmark = landmark / size * 2 - 1
            face_landmarks.append(torch.tensor(landmark).float().cuda())
            # if k % 50 == 1:
        for k_str in list(deca_code.keys())[::50]:
            image = cv2.imread(image_path + '/{}.png'.format(k_str)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0)
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params_face']).float().cuda().unsqueeze(0)
        exps = torch.cat(exps, dim=0)
        face_landmarks = torch.stack(face_landmarks, dim=0)
        face_poses = torch.cat(face_poses, dim=0)
        visualize_images = torch.cat(visualize_images, dim=0)

        # visualize_images = []
        global_orient = []
        hand_poses = []
        betas = []
        hand_landmarks = []
        weight_hand_lmks = []

        lmk_dir = os.path.join(savefolder, 'sapiens_lmk', 'sapiens_2b')
        lmk_files = sorted(glob(os.path.join(lmk_dir, '*.json')))
        lmk_file_list = list(lmk_files)
        lmk_files = [lmk_file_list[(int(name)-1)] for name in names]

        for idx, k in enumerate(mano_dict):
            global_orient.append(torch.tensor(mano_dict[k]['pred_mano_params']['global_orient']).float().cuda())
            hand_poses.append(torch.tensor(mano_dict[k]['pred_mano_params']['hand_pose']).float().cuda())
            betas.append(torch.tensor(mano_dict[k]['pred_mano_params']['betas']).float().cuda())
            # name.append(str(k).split('/')[-1][:-4])
            hamer_landmark = np.array(mano_dict[k]['pred_kp_2d']).astype(np.float32)
            hamer_landmark = torch.tensor(hamer_landmark).float()[0,:,:2]
            hamer_lmk_thumb = hamer_landmark[1:5]#torch.tensor(hamer_landmark).float()[0,1:5,:2]

            ### sapiens_lmks
            lmk_pth = lmk_files[idx]
            lmk_info = json.load(open(lmk_pth, 'r'))
            keypoints = lmk_info['instance_info'][0]['keypoints']#[21:42]
            keypoints = torch.FloatTensor(keypoints).float()#.cuda().unsqueeze(0)
            ### 133 kp
            # [113:134]
            # TODO: 133 kp lmk
            keypoints_reorder = keypoints[112:133, :]
            keypoints_reorder[1:5] = hamer_lmk_thumb
            hand_landmarks.append(keypoints_reorder.float().cuda().unsqueeze(0))

        depth_images_dir = os.path.join(savefolder, 'depth_imgs_v2')
        depth_images_pths = [os.path.join(depth_images_dir, '%07d.jpg'%(int(name))) for name in names]
        hand_mask_dir = os.path.join(savefolder, 'hand_mask')
        hand_mask_pths = [os.path.join(hand_mask_dir, '%07d.png'%(int(name)-1)) for name in names]
        head_mask_dir = os.path.join(savefolder, 'head_mask')
        head_mask_pths = [os.path.join(head_mask_dir, '%07d.png'%(int(name)-1)) for name in names]
        image_dir = os.path.join(savefolder, 'image')
        image_pths = [os.path.join(image_dir, name+'.png') for name in names]
        
        self.scales_all = []
        for i in range(len(depth_images_pths)):
            self.scales_all.append(np.array([1.0]))

        betas = torch.cat(betas, dim=0)
        global_orient = torch.cat(global_orient, dim=0)
        hand_poses = torch.cat(hand_poses, dim=0)
        hand_landmarks = torch.cat(hand_landmarks, dim=0)
        # optimize
        self.optimize(shape, exps, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name, depth_images_pths, hand_mask_pths, head_mask_pths, image_pths, weight_hand_lmks=None)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='mano_flame_params_noflamescaleshift', help='Name for json')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--conf', type=str, default='.', help='path of configs of loss weights')

    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    
    model = Optimizer(save_folder=args.path, conf=conf)
    
    

    image_path = os.path.join(args.path, 'image')
    if args.shape_from == '.':
        args.shape_from = None
        json_path = None
    else:
        json_path = os.path.join(args.shape_from, args.save_name + '.json')
    print("Optimizing: {}".format(args.path))
    intrinsics = [args.fx, args.fy, args.cx, args.cy]
    model.run(deca_code_file=os.path.join(args.path, 'code.json'),
              face_kpts_file=os.path.join(args.path, 'keypoint.json'),
              iris_file=os.path.join(args.path, 'iris.json'),
              mano_file=os.path.join(args.path, 'hamer_codes.npy'),
              savefolder=args.path, image_path=image_path,
              json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name, conf=conf)

