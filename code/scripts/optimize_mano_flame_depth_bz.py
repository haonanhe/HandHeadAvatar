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

from pysdf import SDF

# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
# GLOBAL_POSE = True
GLOBAL_POSE = False

import cv2
import argparse

import sys
sys.path.append('../code')
from external.body_models import MANOLayer
from diff_renderer.diff_renderer import Diff_Renderer

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/haonan/Codes/IMavatar/preprocess/submodules/DECA')
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

np.random.seed(0)

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

def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()

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

def silog_loss(prediction, target, variance_focus: float = 0.85, valid_mask=None) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    # non_zero_mask = (target > 0) & (prediction > 0) 
    # if valid_mask is not None:
    #     valid_mask = valid_mask.view(non_zero_mask.shape[0], non_zero_mask.shape[1], non_zero_mask.shape[2])
    #     non_zero_mask = non_zero_mask & valid_mask.bool()#[:,None]

    if valid_mask is not None:
        non_zero_mask = valid_mask.bool()
        prediction = prediction[non_zero_mask]
        target = target[non_zero_mask]

    # # SILog
    # d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    # loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2))

    loss = torch.mean((prediction - target) ** 2)
    
    return loss

class SmallDataset(Dataset):
    def __init__(self, depth_images_pths, shape, exp, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, translation_p, translation_v, hand_mask_pths, mano_scale, head_mask_pths, translation_f):
        """
        Args:
            image_paths (list): List of paths to the depth images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.depth_images_pths = depth_images_pths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       
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

        shape = self.shape#[idx]
        exp = self.exp[idx]
        face_landmarks = self.face_landmarks[idx]
        face_poses = self.face_poses[idx]
        betas = self.betas#[idx]
        global_orient = self.global_orient[idx]
        hand_landmarks = self.hand_landmarks[idx]
        hand_poses = self.hand_poses[idx]

        translation_p = self.translation_p[idx]
        translation_v = self.translation_v[idx]
        translation_f = self.translation_f[idx]

        mano_scale = self.mano_scale[idx]

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
            'translation_f': translation_f
        }

        return out_dict

class Optimizer(object):
    def __init__(self, device='cuda:0'):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device)

        self.MANOServer = MANOLayer(model_path="/home/haonan/Codes/IMavatar/code/mano_model/data/mano",
                                    is_rhand=True,
                                    batch_size=1,
                                    flat_hand_mean=False,
                                    dtype=torch.float32,
                                    use_pca=False,).cuda()

    def optimize(self, shape, exp, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name, depth_images_pths, hand_mask_pths, head_mask_pths):
        num_img = face_poses.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()

        save_intrinsics = torch.tensor([-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
            intrinsics[3] / size]).float().cuda()

        K = torch.eye(3)
        K[0,0] = save_intrinsics[0] * 224
        K[1,1] = save_intrinsics[1] * 224
        K[0,2] = save_intrinsics[2] * 224
        K[1,2] = save_intrinsics[3] * 224

        # diff_renderer = Diff_Renderer(K, device='cuda', obj_file = '/home/haonan/Codes/IMavatar/code/mano_model/data/hand_head_uv_template.obj')
        diff_renderer_hand = Diff_Renderer(K, device='cuda')
        diff_renderer_head = Diff_Renderer(K, device='cuda', obj_file='/home/haonan/Codes/IMavatar/preprocess/submodules/DECA/data/head_template.obj')


        with open("/home/haonan/Codes/IMavatar/code/mano_model/data/contact_zones.pkl", "rb") as f:
            contact_zones = pickle.load(f)
        contact_zones = contact_zones["contact_zones"]
        contact_idx = np.array([item for sublist in contact_zones.values() for item in sublist])

        # if GLOBAL_POSE:
        #     translation_p = torch.tensor([0, 0, -1]).float().cuda()
        # else:
        #     translation_p = torch.tensor([0, 0, -1]).unsqueeze(0).expand(num_img, -1).float().cuda()
        
        if GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -4]).float().cuda()
        else:
            translation_p = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
        
        translation_v = torch.tensor([0, 0, 0]).unsqueeze(0).expand(num_img, -1).float().cuda()
        translation_f = torch.tensor([0, 0, 0]).unsqueeze(0).expand(num_img, -1).float().cuda()
        mano_scale = torch.tensor([4]).unsqueeze(0).expand(num_img, -1).float().cuda()

        if GLOBAL_POSE:
            face_poses = torch.cat([torch.zeros_like(face_poses[:, :3]), face_poses], dim=1)
        if face_landmarks.shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True
        if use_iris:
            face_poses = torch.cat([face_poses, torch.zeros_like(face_poses[:, :6])], dim=1)

        # pose_ori = pose.clone()

        translation_p = nn.Parameter(translation_p)
        face_poses = nn.Parameter(face_poses)
        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)
        hand_poses = nn.Parameter(hand_poses)
        translation_v = nn.Parameter(translation_v)
        # mano_scale = nn.Parameter(mano_scale)
        # translation_f = nn.Parameter(translation_f)
        # betas = nn.Parameter(betas)
        # global_orient = nn.Parameter(global_orient)

        # set optimizer
        # if json_path is None:
        #     opt_p = torch.optim.Adam(
        #         [translation_p, pose, betas, global_orient],
        #         lr=1e-2)
        # else:
        #     opt_p = torch.optim.Adam(
        #         [translation_p, pose, global_orient],
        #         lr=1e-2)

        if json_path is None:
            opt_t = torch.optim.Adam(
                [translation_p, translation_v, face_poses],#, betas, global_orient],
                lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses, global_orient, exp, shape],#, betas, global_orient],
                lr=1e-4)
        else:
            opt_t = torch.optim.Adam(
                [translation_p, translation_v, face_poses],#, betas, global_orient],
                lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses, global_orient, face_poses, exp, shape],#, betas, global_orient],
                lr=1e-4)

        # scheduler_t = torch.optim.lr_scheduler.ExponentialLR(opt_t, gamma=0.99)

        # optimization steps
        len_landmark_face = face_landmarks.shape[1]
        len_landmark_hand = hand_landmarks.shape[1]
        hand_landmarks = (hand_landmarks[..., :2] - (size/2)) / (size/2)

        # for k in range(1):
        for k in range(501):
        # for k in range(1001):
        # for k in range(3001):
            face_full_pose = face_poses
            if not use_iris:
                face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
            
            verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                   expression_params=exp,
                                                                                   full_pose=face_full_pose)
            
            # verts_p_face *= 4
            # landmarks3d_p_face *= 4
            # landmarks2d_p_face *= 4

            pred_mano_params = {
                'global_orient': global_orient,
                'hand_pose': hand_poses,
                'betas': betas.expand(num_img, -1),
                'transl': translation_v,
                # 'scale': mano_scale
            }
            mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
            verts_p_hand = mano_output.vertices.clone()
            landmarks3d_p_hand = mano_output.joints.clone()

            # if k % 100 == 0:
            #     point_cloud_np = verts_p[0].detach().cpu().numpy()
            #     save_point_cloud_to_ply(point_cloud_np, filename=os.path.join(savefolder, "./output_point_cloud_tmp.ply"))

            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            if GLOBAL_POSE:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

            # trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
            trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
            trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
            trans_landmarks2d = torch.cat([trans_landmarks2d_face, trans_landmarks2d_hand], dim=1)
            landmark = torch.cat([face_landmarks, hand_landmarks], dim=1)
            ## landmark loss
            landmark_loss2 = l2_distance(trans_landmarks2d_face[:, :len_landmark_face, :2], face_landmarks[:, :len_landmark_face])
            landmark_loss2 += l2_distance(trans_landmarks2d_hand[:, :len_landmark_hand, :2], hand_landmarks[:, :len_landmark_hand])
            total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
            total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            total_loss += torch.mean(torch.square(face_poses[1:] - face_poses[:-1])) * 10
            total_loss += torch.mean(torch.square(global_orient[1:] - global_orient[:-1])) * 1e-1
            total_loss += torch.mean(torch.square(hand_poses[1:] - hand_poses[:-1])) * 10
            # total_loss += torch.mean(torch.square(pose - pose_ori)) * 10
            # total_loss += torch.mean(torch.square(betas - betas_ori)) * 10
            if not GLOBAL_POSE:
                total_loss += torch.mean(torch.square(translation_p[1:] - translation_p[:-1])) * 10
            total_loss += torch.mean(torch.square(translation_v[1:] - translation_v[:-1])) * 10
            # total_loss += torch.mean(torch.square(mano_scale[1:] - mano_scale[:-1])) * 10

            opt_p.zero_grad()
            opt_t.zero_grad()
            total_loss.backward()
            opt_p.step()
            opt_t.step()

            # scheduler_t.step()

            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                  datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                    print(loss_info)
                    verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)
                    trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    # shape_images = self.deca.render_mano.render_shape(verts_p[::50], trans_verts)
                    shape_images = self.deca.render_hand_head.render_shape(verts_p[::50], trans_verts)
                    depth_images = self.deca.render_hand_head.render_depth(trans_verts)
                    visdict = {
                        # 'inputs': visualize_images,
                        'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50], isScale=True),
                        'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50], isScale=True),
                        'shape_images': shape_images,
                        'depth_images': depth_images
                    }
                    cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

                    # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                    # print(shape_images.shape)

                    save = True
                    if save:
                        save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                           intrinsics[3] / size]
                        dict = {}
                        frames = []
                        for i in range(num_img):
                            frames.append({'file_path': './image/' + name[i],
                                           'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                           'expression': exp[i].detach().cpu().numpy().tolist(),
                                           'head_pose': face_full_pose[i].detach().cpu().numpy().tolist(),
                                            # 'head_transl': translation_f[i].detach().cpu().numpy().tolist(),
                                            'global_orient': global_orient[i].detach().cpu().numpy().tolist(),
                                            'hand_pose': hand_poses[i].detach().cpu().numpy().tolist(),
                                            'betas': betas.detach().cpu().numpy().tolist(),
                                            'hand_transl': translation_v[i].detach().cpu().numpy().tolist(),
                                           'bbox': torch.stack(
                                               [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                               dim=0).detach().cpu().numpy().tolist(),
                                           'flame_keypoints': trans_landmarks2d_face[i, :,
                                                              :2].detach().cpu().numpy().tolist(),
                                           'mano_keypoints': trans_landmarks2d_hand[i, :,
                                                              :2].detach().cpu().numpy().tolist()
                                           })

                        dict['frames'] = frames
                        dict['intrinsics'] = save_intrinsics
                        dict['cam_intrinsics'] = cam_intrinsics.detach().cpu().numpy().tolist()
                        dict['shape_params_face'] = shape[0].cpu().numpy().tolist()
                        dict['shape_params_hand'] = betas.cpu().numpy().tolist()
                        with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                            json.dump(dict, fp)

        if json_path is None:
            # opt_p = torch.optim.Adam(
            #     [face_poses],#, betas, global_orient],
            #     lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses, global_orient, face_poses],#, betas, global_orient],
                lr=1e-3)
            opt_v = torch.optim.Adam(
                [translation_v, translation_p],#, betas, global_orient],
                lr=1e-2)
        else:
            # opt_p = torch.optim.Adam(
            #     [face_poses],#, betas, global_orient],
            #     lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses, global_orient, face_poses],#, betas, global_orient],
                lr=1e-3)
            opt_v = torch.optim.Adam(
                [translation_v, translation_p],#, betas, global_orient],
                lr=1e-2)

        # scheduler_v = torch.optim.lr_scheduler.ExponentialLR(opt_v, gamma=0.99)

        # translation_v = translation_v + 0.1*torch.ones_like(translation_v).to(translation_v.device)

        # for param_group in opt_t.param_groups:
        #     param_group['lr'] = 5e-3

        ### batch optimization
        if GLOBAL_POSE:
            bz_translation_p = translation_p.unsqueeze(0).expand(num_img, -1)
        else:
            bz_translation_p = translation_p
        dataset = SmallDataset(depth_images_pths, shape[0], exp, face_landmarks, face_poses, betas[0], global_orient, hand_landmarks, hand_poses, bz_translation_p, translation_v, hand_mask_pths, mano_scale, head_mask_pths, translation_f)
        batch_size = 720
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        for k in range(15):
        # for k in range(51):
        # for k in range(1001):
        # for k in range(3001):
            for batch in data_loader:
                b_depth_img = batch['depth_img']
                b_shape = batch['shape']
                b_exp = batch['exp']
                b_face_landmarks = batch['face_landmarks']
                b_face_poses = batch['face_poses']
                b_betas = batch['betas']
                b_global_orient = batch['global_orient']
                b_hand_landmarks = batch['hand_landmarks']
                b_hand_poses = batch['hand_poses']
                b_translation_p = batch['translation_p']
                b_translation_v = batch['translation_v']
                b_hand_mask = batch['hand_mask']
                b_mano_scale = batch['mano_scale']
                b_head_mask = batch['head_mask']
                b_translation_f = batch['translation_f']

                bz = b_depth_img.shape[0]
                
                face_full_pose = b_face_poses
                if not use_iris:
                    face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
                if not GLOBAL_POSE:
                    face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
                
                verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=b_shape,
                                                                                    expression_params=b_exp,
                                                                                    full_pose=face_full_pose)
                
                # verts_p_face += b_translation_f.unsqueeze(1)
                # landmarks3d_p_face += b_translation_f.unsqueeze(1)
                # landmarks2d_p_face += b_translation_f.unsqueeze(1)
                # verts_p_face *= 4
                # landmarks3d_p_face *= 4
                # landmarks2d_p_face *= 4

                pred_mano_params = {
                    'global_orient': b_global_orient,
                    'hand_pose': b_hand_poses,
                    'betas': b_betas,
                    'transl': b_translation_v,
                    # 'scale': b_mano_scale
                }
                mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
                verts_p_hand = mano_output.vertices.clone()
                landmarks3d_p_hand = mano_output.joints.clone()

                # if k % 100 == 0:
                #     point_cloud_np = verts_p[0].detach().cpu().numpy()
                #     save_point_cloud_to_ply(point_cloud_np, filename=os.path.join(savefolder, "./output_point_cloud_tmp.ply"))

                # perspective projection
                # Global rotation is handled in FLAME, set camera rotation matrix to identity
                ident = torch.eye(3).float().cuda().unsqueeze(0).expand(bz, -1, -1)
                # if GLOBAL_POSE:
                #     w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
                # else:
                w2c_p = torch.cat([ident, b_translation_p.unsqueeze(2)], dim=2)

                # trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                b_trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
                b_trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
                b_trans_landmarks2d = torch.cat([b_trans_landmarks2d_face, b_trans_landmarks2d_hand], dim=1)
                b_landmark = torch.cat([b_face_landmarks, b_hand_landmarks], dim=1)
                ## landmark loss
                landmark_loss2 = l2_distance(b_trans_landmarks2d_face[:, :len_landmark_face, :2], b_face_landmarks[:, :len_landmark_face])
                landmark_loss2 += l2_distance(b_trans_landmarks2d_hand[:, :len_landmark_hand, :2], b_hand_landmarks[:, :len_landmark_hand])
                total_loss = landmark_loss2*10 #+ torch.mean(torch.square(b_shape)) * 1e-2 + torch.mean(torch.square(b_exp)) * 1e-2
                # total_loss += torch.mean(torch.square(b_exp[1:] - b_exp[:-1])) * 1e-1
                total_loss += torch.mean(torch.square(b_face_poses[1:] - b_face_poses[:-1]))# * 10
                total_loss += torch.mean(torch.square(b_global_orient[1:] - b_global_orient[:-1])) * 1e-1
                total_loss += torch.mean(torch.square(b_hand_poses[1:] - b_hand_poses[:-1])) #* 10
                # total_loss += torch.mean(torch.square(pose - pose_ori)) * 10
                # total_loss += torch.mean(torch.square(betas - betas_ori)) * 10
                if not GLOBAL_POSE:
                    total_loss += torch.mean(torch.square(b_translation_p[1:] - b_translation_p[:-1])) #* 10
                total_loss += torch.mean(torch.square(b_translation_v[1:] - b_translation_v[:-1])) #* 10
                # total_loss += torch.mean(torch.square(b_mano_scale[1:] - b_mano_scale[:-1])) #* 10
                # total_loss += torch.mean(torch.square(b_translation_f[1:] - b_translation_f[:-1])) #* 10

                verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)
                trans_verts = projection(verts_p, cam_intrinsics, w2c_p)
                # depth_images = self.deca.render_hand_head.render_depth(trans_verts)
                verts_cam = verts_p
                extrinsic = torch.eye(4)[None,...].repeat(bz,1,1)
                for i in range(bz):
                    world_mat = torch.tensor(_load_K_Rt_from_P(None, w2c_p[i].detach().cpu().numpy().astype(np.float32))[1], dtype=torch.float32)
                    # camera matrix to openGL format 
                    R = world_mat[:3, :3]
                    R *= -1 
                    t = world_mat[:3, 3]
                    extrinsic[i, :3, :3] = R
                    extrinsic[i, :3, 3] = t
                extrinsic = extrinsic.cuda()

                # pred_depth_images, pred_mask = diff_renderer.render(verts_p, extrinsic)
                # pred_depth_images = pred_depth_images.squeeze(-1)
                # pred_mask = pred_mask.bool().squeeze(-1) #& b_hand_mask.bool().to(depth_images.device)
                # depth_loss = silog_loss(pred_depth_images, b_depth_img.to(depth_images.device), valid_mask=pred_mask)
                # total_loss += depth_loss

                # min_z = verts_p[:, :, 2].min()
                # trans_verts_p = verts_p.clone()
                # trans_verts_p[:, :, 2] = verts_p[:, :, 2] - min_z
                # verts_p_face = trans_verts_p[:,:verts_p_face.shape[1],:]
                # verts_p_hand = trans_verts_p[:,verts_p_face.shape[1]:,:]

                # # pred_depth_images, pred_mask = diff_renderer.render(verts_p, extrinsic)
                pred_depth_images_hand, pred_mask_hand = diff_renderer_hand.render(verts_p_hand, extrinsic)
                pred_depth_images_hand = pred_depth_images_hand.squeeze(-1)
                pred_mask_hand = pred_mask_hand.bool().squeeze(-1) & b_hand_mask.bool().to(depth_images.device)
                depth_loss = silog_loss(pred_depth_images_hand, b_depth_img.to(depth_images.device), valid_mask=pred_mask_hand)
                
                pred_depth_images_head, pred_mask_head = diff_renderer_head.render(verts_p_face, extrinsic)
                pred_depth_images_head = pred_depth_images_head.squeeze(-1)
                pred_mask_head = pred_mask_head.bool().squeeze(-1) & b_head_mask.bool().to(depth_images.device)
                depth_loss = depth_loss + silog_loss(pred_depth_images_head, b_depth_img.to(depth_images.device), valid_mask=pred_mask_head)
                # depth_loss = 0.1 * depth_loss
                # depth_loss += silog_loss(pred_depth_images, b_depth_img.to(depth_images.device), valid_mask=b_hand_mask.to(depth_images.device))
                total_loss += depth_loss

                # contact
                mano_vertices_tips = verts_p_hand[:,contact_idx,:]
                knn_v = verts_p_face.clone()
                contact_loss = knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0].mean()
                total_loss += contact_loss

                # hand_mean_depth = torch.mean(verts_p_hand.view(bz, -1, 3), dim=(0,1))[-1]
                # head_mean_depth = torch.mean(verts_p_face.view(bz, -1, 3), dim=(0,1))[-1]
                # mean_z_loss = torch.relu((head_mean_depth - hand_mean_depth))
                # total_loss += mean_z_loss

                # collision_loss = 0.
                # for i in range(bz):
                #     f = SDF(verts_p_face[i].detach().cpu().numpy(), self.deca.render.faces.squeeze(0).detach().cpu().numpy())
                #     hand_verts = verts_p_hand[i].detach().cpu().numpy()
                #     insides = f.contains(hand_verts)
                #     if True in insides:
                #         mano_inside = hand_verts[insides]
                #         head_nearest_idx = f.nn(mano_inside)
                #         collision_loss += torch.mean(torch.sqrt(torch.sum((verts_p_hand[i][insides] - verts_p_face[i][head_nearest_idx]) ** 2, dim=0)))
                # collision_loss = collision_loss * 0.1
                # total_loss += collision_loss

                # ########################
                # # b_trans_landmarks2d_face_index = b_trans_landmarks2d_face[:,:,:2].round().long()
                # # b_trans_landmarks2d_hand_index = b_trans_landmarks2d_hand[:,:,:2].round().long()
                # # landmarks = torch.cat((b_trans_landmarks2d_face_index, b_trans_landmarks2d_hand_index), dim=1)
                # # pred_landmarks = b_trans_landmarks2d[:,:,:2].round().long()
                # pred_landmarks = b_trans_landmarks2d_hand[:,:,:2].clone().round().long()
                # pred_landmarks[...,0] = pred_landmarks[...,0]*224/2 + 224/2
                # pred_landmarks[...,1] = pred_landmarks[...,1]*224/2 + 224/2
                # # landmarks = b_trans_landmarks2d_hand_index
                # N, M, _ = pred_landmarks.shape
                # indices = torch.arange(N).view(N, 1).expand(-1, M).reshape(-1)
                # pred_point_indices = pred_landmarks.view(-1, 2)
                # pred_point_indices_clamp = torch.clamp(pred_point_indices, min=0, max=223)
                # depth_kp_values_pred_hand = (pred_mask_hand*pred_depth_images_hand)[indices, pred_point_indices_clamp[:, 1], pred_point_indices_clamp[:, 0]].view(N, M, 1)

                # pred_landmarks = b_trans_landmarks2d_face[:,:,:2].clone().round().long()
                # pred_landmarks[...,0] = pred_landmarks[...,0]*224/2 + 224/2
                # pred_landmarks[...,1] = pred_landmarks[...,1]*224/2 + 224/2
                # # landmarks = b_trans_landmarks2d_hand_index
                # N, M, _ = pred_landmarks.shape
                # indices = torch.arange(N).view(N, 1).expand(-1, M).reshape(-1)
                # pred_point_indices = pred_landmarks.view(-1, 2)
                # pred_point_indices_clamp = torch.clamp(pred_point_indices, min=0, max=223)
                # depth_kp_values_pred_head = (pred_mask_head*pred_depth_images_head)[indices, pred_point_indices_clamp[:, 1], pred_point_indices_clamp[:, 0]].view(N, M, 1)

                # depth_kp_values_pred = torch.cat([depth_kp_values_pred_hand, depth_kp_values_pred_head], dim=1)

                # # gt_landmarks = b_landmark[:,:,:2].round().long()
                # # gt_landmarks = b_hand_landmarks[:,:,:2].round().long()
                # gt_landmarks = torch.cat((b_hand_landmarks, b_face_landmarks), dim=1).round().long()
                # gt_landmarks[...,0] = gt_landmarks[...,0]*224/2 + 224/2
                # gt_landmarks[...,1] = gt_landmarks[...,1]*224/2 + 224/2
                # N, M, _ = gt_landmarks.shape
                # indices = torch.arange(N).view(N, 1).expand(-1, M).reshape(-1)
                # gt_point_indices = gt_landmarks.view(-1, 2)
                # gt_point_indices_clamp = torch.clamp(gt_point_indices, min=0, max=223)
                # pred_mask = pred_mask_head | pred_mask_hand
                # depth_kp_values_gt = (pred_mask*b_depth_img.to(depth_images.device))[indices, gt_point_indices_clamp[:, 1], gt_point_indices_clamp[:, 0]].view(N, M, 1)
                
                # # pred_depth_images = pred_depth_images[:,:,:,None]
                # # depth_kp_values_pred = (pred_mask*pred_depth_images)[indices, pred_point_indices_clamp[:, 1], pred_point_indices_clamp[:, 0]].view(N, M, 1)
                # # depth_kp_values_pred_hand = (pred_mask_hand*pred_depth_images_hand)[indices, pred_point_indices_clamp_hand[:, 1], pred_point_indices_clamp_hand[:, 0]].view(N, M, 1)
                # # depth_kp_values_pred_head = (pred_mask_head*pred_depth_images_head)[indices, pred_point_indices_clamp_head[:, 1], pred_point_indices_clamp_head[:, 0]].view(N, M, 1)
                # # depth_loss = silog_loss(pred_depth, gt_depth, valid_mask=mask_bool)
                # # losses['all_depth'] = depth_loss
                # # valid_mask = (pred_point_indices[..., 0] >= 0) & (pred_point_indices[..., 0] < 224) & (pred_point_indices[..., 1] >= 0) & (pred_point_indices[..., 1] < 224) \
                # #             & (gt_point_indices[..., 0] >= 0) & (gt_point_indices[..., 0] < 224) & (gt_point_indices[..., 1] >= 0) & (gt_point_indices[..., 1] < 224) #& pred_mask
                # depth_loss = silog_loss(depth_kp_values_pred, depth_kp_values_gt, variance_focus=0.5, valid_mask=None)
                # total_loss += depth_loss
                # ########################
                
                # visdict = {
                #         'gt_depth_images': b_depth_img[::20].unsqueeze(1),
                #         'depth_images': (pred_depth_images_hand*pred_mask_hand)[::20].unsqueeze(1),
                #         'gt_depth_images_hand': (b_depth_img.to(depth_images.device)*pred_mask_hand)[::20].unsqueeze(1)
                #         # 'head_depth_images': (pred_depth_images_head*pred_mask_head)[::20].unsqueeze(1)
                #     }
                # cv2.imwrite(os.path.join(savefolder, 'batch_depth_images.jpg'), self.deca.visualize(visdict))
                # # # cv2.imwrite(os.path.join(savefolder, 'depth_image.jpg'), pred_depth_images[0].detach().cpu().numpy()*255)

                opt_p.zero_grad()
                opt_v.zero_grad()
                total_loss.backward()
                opt_p.step()
                opt_v.step()

                # scheduler_t.step()

            if k % 25 == 0 and k != 0:
                for param_group in opt_v.param_groups:
                    param_group['lr'] *= 0.1
                for param_group in opt_p.param_groups:
                    param_group['lr'] *= 0.1

            # visualize
            if k % 1 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}' + f'depth_loss: {depth_loss}' #+ f'collision_loss: {collision_loss}'
                    print(loss_info)

                    face_full_pose = face_poses
                    if not use_iris:
                        face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
                    if not GLOBAL_POSE:
                        face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
                    
                    verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                        expression_params=exp,
                                                                                        full_pose=face_full_pose)
                    
                    # verts_p_face += translation_f.unsqueeze(1)
                    # landmarks3d_p_face += translation_f.unsqueeze(1)
                    # landmarks2d_p_face += translation_f.unsqueeze(1)
                    # verts_p_face *= 4
                    # landmarks3d_p_face *= 4
                    # landmarks2d_p_face *= 4

                    pred_mano_params = {
                        'global_orient': global_orient,
                        'hand_pose': hand_poses,
                        'betas': betas.expand(num_img, -1),
                        'transl': translation_v,
                        # 'scale': mano_scale
                    }
                    mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
                    verts_p_hand = mano_output.vertices.clone()
                    landmarks3d_p_hand = mano_output.joints.clone()

                    # if k % 100 == 0:
                    #     point_cloud_np = verts_p[0].detach().cpu().numpy()
                    #     save_point_cloud_to_ply(point_cloud_np, filename=os.path.join(savefolder, "./output_point_cloud_tmp.ply"))

                    # perspective projection
                    # Global rotation is handled in FLAME, set camera rotation matrix to identity
                    ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
                    if GLOBAL_POSE:
                        w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
                    else:
                        w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

                    # trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    trans_landmarks2d_face = projection(landmarks3d_p_face, cam_intrinsics, w2c_p)
                    trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
                    trans_landmarks2d = torch.cat([trans_landmarks2d_face, trans_landmarks2d_hand], dim=1)
                    landmark = torch.cat([face_landmarks, hand_landmarks], dim=1)
                    
                    verts_p = torch.cat([verts_p_face, verts_p_hand], dim=1)
                    trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    # shape_images = self.deca.render_mano.render_shape(verts_p[::50], trans_verts)
                    shape_images = self.deca.render_hand_head.render_shape(verts_p[::50], trans_verts)
                    depth_images = self.deca.render_hand_head.render_depth(trans_verts)
                    visdict = {
                        # 'inputs': visualize_images,
                        'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50], isScale=True),
                        'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50], isScale=True),
                        'shape_images': shape_images,
                        'depth_images': depth_images
                    }
                    cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

                    # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                    # print(shape_images.shape)

                    save = True
                    if save:
                        save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                        intrinsics[3] / size]
                        dict = {}
                        frames = []
                        for i in range(num_img):
                            frames.append({'file_path': './image/' + name[i],
                                        'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                        'expression': exp[i].detach().cpu().numpy().tolist(),
                                        'head_pose': face_full_pose[i].detach().cpu().numpy().tolist(),
                                        # 'head_transl': translation_f[i].detach().cpu().numpy().tolist(),
                                        'global_orient': global_orient[i].detach().cpu().numpy().tolist(),
                                        'hand_pose': hand_poses[i].detach().cpu().numpy().tolist(),
                                        'betas': betas.detach().cpu().numpy().tolist(),
                                        'hand_transl': translation_v[i].detach().cpu().numpy().tolist(),
                                        'bbox': torch.stack(
                                            [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                            dim=0).detach().cpu().numpy().tolist(),
                                        'flame_keypoints': trans_landmarks2d_face[i, :,
                                                            :2].detach().cpu().numpy().tolist(),
                                        'mano_keypoints': trans_landmarks2d_hand[i, :,
                                                            :2].detach().cpu().numpy().tolist()
                                        })

                        dict['frames'] = frames
                        dict['intrinsics'] = save_intrinsics
                        dict['cam_intrinsics'] = cam_intrinsics.detach().cpu().numpy().tolist()
                        dict['shape_params_face'] = shape[0].cpu().numpy().tolist()
                        dict['shape_params_hand'] = betas.cpu().numpy().tolist()
                        with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                            json.dump(dict, fp)

    def run(self, deca_code_file, face_kpts_file, iris_file, mano_file, savefolder, image_path, json_path, intrinsics, size,
            save_name):
        
        deca_code = json.load(open(deca_code_file, 'r'))
        face_kpts = json.load(open(face_kpts_file, 'r'))
        try:
            iris_kpts = json.load(open(iris_file, 'r'))
        except:
            iris_kpts = None
            print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        face_landmarks = []
        face_poses = []
        name = []
        num_img = len(deca_code)
        # ffmpeg extracted frames, index starts with 1
        for k in range(1, num_img + 1):
            k_str = str('%07d'%k)
            shape.append(torch.tensor(deca_code[k_str]['shape']).float().cuda())
            exps.append(torch.tensor(deca_code[k_str]['exp']).float().cuda())
            face_poses.append(torch.tensor(deca_code[k_str]['pose']).float().cuda())
            name.append(k_str)
            landmark = np.array(face_kpts['{}.png'.format(k_str)]).astype(np.float32)
            if iris_kpts is not None:
                iris = np.array(iris_kpts['{}.png'.format(k_str)]).astype(np.float32).reshape(2, 2)
                landmark = np.concatenate([landmark, iris[[1,0], :]], 0)
            landmark = landmark / size * 2 - 1
            face_landmarks.append(torch.tensor(landmark).float().cuda())
            if k % 50 == 1:
                image = cv2.imread(image_path + '/{}.png'.format(k_str)).astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0)
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params']).float().cuda().unsqueeze(0)
        exps = torch.cat(exps, dim=0)
        face_landmarks = torch.stack(face_landmarks, dim=0)
        face_poses = torch.cat(face_poses, dim=0)
        visualize_images = torch.cat(visualize_images, dim=0)

        with open(mano_file,'rb') as f:
            mano_dict = pickle.load(f)

        # visualize_images = []
        global_orient = []
        hand_poses = []
        betas = []
        hand_landmarks = []
        # num_img = len(deca_code)
        # ffmpeg extracted frames, index starts with 1
        # for k in mano_dict.keys():
        for idx, k in enumerate(mano_dict):
            global_orient.append(torch.tensor(mano_dict[k]['pred_mano_params']['global_orient']).float().cuda())
            hand_poses.append(torch.tensor(mano_dict[k]['pred_mano_params']['hand_pose']).float().cuda())
            betas.append(torch.tensor(mano_dict[k]['pred_mano_params']['betas']).float().cuda())
            # name.append(str(k).split('/')[-1][:-4])
            landmark = np.array(mano_dict[k]['pred_kp_2d']).astype(np.float32)
            # landmark = np.array(face_kpts['{}.png'.format(str(k))]).astype(np.float32)
            # landmark = landmark / size * 2 - 1
            hand_landmarks.append(torch.tensor(landmark).float().cuda())
            # if idx % 50 == 1:
            #     # image = cv2.imread(image_path + '/{}.png'.format(str(k))).astype(np.float32) / 255.
            #     image = cv2.imread(str(k)).astype(np.float32) / 255.
            #     image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #     visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        depth_images_dir = os.path.join(savefolder, 'depth_imgs_v2')
        depth_images_pths = sorted(glob(os.path.join(depth_images_dir, '*.jpg')))
        hand_mask_dir = os.path.join(savefolder, 'hand_mask')
        hand_mask_pths = sorted(glob(os.path.join(hand_mask_dir, '*.png')))
        head_mask_dir = os.path.join(savefolder, 'head_mask')
        head_mask_pths = sorted(glob(os.path.join(head_mask_dir, '*.png')))

        betas = torch.cat(betas, dim=0)
        betas = torch.mean(betas, dim=0).unsqueeze(0)
        global_orient = torch.cat(global_orient, dim=0)
        hand_poses = torch.cat(hand_poses, dim=0)
        # landmarks = torch.stack(landmarks, dim=0)
        hand_landmarks = torch.cat(hand_landmarks, dim=0)
        # visualize_images = torch.cat(visualize_images, dim=0)
        # optimize
        self.optimize(shape, exps, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name, depth_images_pths, hand_mask_pths, head_mask_pths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='mano_flame_params', help='Name for json')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)

    args = parser.parse_args()
    model = Optimizer()

    image_path = os.path.join(args.path, 'image')
    if args.shape_from == '.':
        args.shape_from = None
        json_path = None
    else:
        json_path = os.path.join(args.shape_from, args.save_name + '.json')
        # json_path = os.path.join(args.shape_from, args.save_name + '.npy')
    print("Optimizing: {}".format(args.path))
    intrinsics = [args.fx, args.fy, args.cx, args.cy]
    # model.run(deca_code_file=os.path.join(args.path, 'code.json'),
    #           face_kpts_file=os.path.join(args.path, 'keypoint.json'),
    #           iris_file=os.path.join(args.path, 'iris.json'), savefolder=args.path, image_path=image_path,
    #           json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name)
    model.run(deca_code_file=os.path.join(args.path, 'code.json'),
              face_kpts_file=os.path.join(args.path, 'keypoint.json'),
              iris_file=os.path.join(args.path, 'iris.json'),
              mano_file=os.path.join(args.path, 'hamer_codes.npy'),
              savefolder=args.path, image_path=image_path,
              json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name)

