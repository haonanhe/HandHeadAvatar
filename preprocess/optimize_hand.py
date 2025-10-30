import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import json
import pickle
import open3d as o3d
from glob import glob
import re
from PIL import Image
from pytorch3d.ops import knn_points
import shutil

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

def read_point_cloud_from_ply(filename):
    """
    Read mesh from a PLY file.
    
    Args:
    filename (str): Input filename (should end with .ply)
    
    Returns:
    tuple: vertices tensor of shape (N, 3) and faces tensor of shape (M, 3)
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertices = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces).long()
    return vertices, faces

def l2_distance(verts1, verts2, weight=None):
    if weight is not None:
        return torch.sqrt((((verts1 - verts2)**2)*weight).sum(2)).mean(1).mean()
    else:
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





class Optimizer(object):
    def __init__(self, device='cuda:0', save_folder=None, conf=None):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device, image_size=512, uv_size=512)
        self.render_size = 256
        self.deca_optim = DECA(config=deca_cfg, device=device, image_size=self.render_size, uv_size=self.render_size)

        self.MANOServer = MANOLayer(model_path="../code/mano_model/data/mano",
                                    is_rhand=True,
                                    batch_size=1,
                                    flat_hand_mean=False,
                                    dtype=torch.float32,
                                    use_pca=False,).cuda()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.conf = conf
        self.conf_loss = self.conf.get_config('loss')
        self.optimize_depth_rank = self.conf_loss['optimize_depth_rank']
        self.contact_frame_idx = self.conf_loss['contact_frame_idx']

    def optimize(self, shape, exp, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name, depth_images_pths, hand_mask_pths, head_mask_pths, image_pths, w2c_p, hand_transl, mano_scale, head_verts, sample_interval, save_name_refine, save_intrinsics, cam_intrinsics, flame_scale, flame_transl, weight_hand_lmks=None):
        num_img = head_verts.shape[0]
        K = torch.eye(3)
        K[0,0] = save_intrinsics[0] * 512
        K[1,1] = save_intrinsics[1] * 512
        K[0,2] = save_intrinsics[2] * 512
        K[1,2] = save_intrinsics[3] * 512

        with open("../code/mano_model/data/mano_contact_vertices.pkl", "rb") as f:
            contact_zones = pickle.load(f)

        video_name = savefolder.split('/')[-1]
        if "finger" in video_name:
            contact_idx = np.array(contact_zones['index'])
        elif "fist" in video_name:
            contact_idx = np.array(contact_zones['fist'])
            # contact_idx = np.concatenate((np.array(contact_zones['index']), np.array(contact_zones['middle']), np.array(contact_zones['ring'])), axis=0)
        elif "palm" in video_name:
            # contact_idx = np.concatenate((np.array(contact_zones['index']), np.array(contact_zones['middle']), np.array(contact_zones['ring'])), axis=0)
            contact_idx = np.concatenate((np.array(contact_zones['index']), np.array(contact_zones['middle'])), axis=0)
            # contact_idx = np.array(contact_zones['middle'])
        elif "pinch" in video_name:
            # contact_idx = np.concatenate((np.array(contact_zones['index']), np.array(contact_zones['middle']), np.array(contact_zones['ring']), np.array(contact_zones['thumb'])), axis=0)
            # contact_idx_1 = np.concatenate((np.array(contact_zones['index']), np.array(contact_zones['middle']), np.array(contact_zones['ring'])), axis=0)
            contact_idx_1 = np.array(contact_zones['index'])
            contact_idx_2 = np.array(contact_zones['thumb'])
               
        with open("../preprocess/submodules/DECA/data/FLAME_masks.pkl", "rb") as f:
            flame_masks = pickle.load(f, encoding='latin1')

        with open("../preprocess/submodules/DECA/data/face_idx.pkl", "rb") as f:
            flame_face_masks = pickle.load(f)
        flame_face_masks = np.array(flame_face_masks)

        with open("../preprocess/submodules/DECA/data/right_face_touch_region.pkl", "rb") as f:
            flame_face_masks_right_cheek = pickle.load(f)
        flame_face_masks_right_cheek = np.array(flame_face_masks_right_cheek)
        flame_face_masks_left_cheek = np.setdiff1d(flame_face_masks, flame_face_masks_right_cheek)

        verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                expression_params=exp,
                                                                                full_pose=face_poses)
        verts_p_face *= flame_scale.unsqueeze(1)
        verts_p_face += flame_transl.unsqueeze(1)
        verts_p_face_right_cheek = verts_p_face[:,flame_face_masks_right_cheek,:].clone()
        knn_v = head_verts.detach()[:,:,:].clone()
        head_verts_idx_right_cheek = knn_points(verts_p_face_right_cheek, knn_v, K=1, return_nn=False)[1].squeeze(-1)
        
        verts_p_face_left_cheek = verts_p_face[:,flame_face_masks_left_cheek,:].clone()
        head_verts_idx_left_cheek = knn_points(verts_p_face_left_cheek, knn_v, K=1, return_nn=False)[1].squeeze(-1)
        
        global_orient = nn.Parameter(global_orient)
        hand_poses = nn.Parameter(hand_poses)
        mano_scale = nn.Parameter(mano_scale)
        hand_transl = nn.Parameter(hand_transl)
        
        opt_t = torch.optim.Adam(
            [hand_transl, mano_scale],
            lr=1e-2)
        opt_p = torch.optim.Adam(
            [global_orient],
            lr=1e-4)

        # # optimization steps
        len_landmark_hand = hand_landmarks.shape[1]

        for k in tqdm(range(3001)):
            pred_mano_params = {
                'global_orient': global_orient,
                'hand_pose': hand_poses,
                # 'betas': betas.expand(num_img, -1),
                'betas': betas,
                'transl': hand_transl,
                'scale': mano_scale
            }
            mano_output = self.MANOServer(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
            verts_p_hand = mano_output.vertices.clone()
            landmarks3d_p_hand = mano_output.joints.clone()

            ## landmark loss
            trans_landmarks2d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p)
            trans_landmarks3d_hand = projection(landmarks3d_p_hand, cam_intrinsics, w2c_p, no_intrinsics=True)

            ## landmark loss
            # landmark_loss2 = l2_distance(trans_landmarks2d_face[:, :len_landmark_face, :2], face_landmarks[:, :len_landmark_face]) 
            landmark_loss2 = l2_distance(trans_landmarks2d_hand[:, :len_landmark_hand, :2], hand_landmarks[:, :len_landmark_hand])
            landmark_loss2 = landmark_loss2 * self.conf_loss['landmark_weight']
            total_loss = landmark_loss2

            smooth_loss = 0
            smooth_loss += torch.mean(torch.square(global_orient[1:] - global_orient[:-1])) #* 1e-1
            smooth_loss += torch.mean(torch.square(hand_poses[1:] - hand_poses[:-1]))
            smooth_loss += torch.mean(torch.square(hand_transl[1:] - hand_transl[:-1]))
            smooth_loss += torch.mean(torch.square(trans_landmarks2d_hand[1:] - trans_landmarks2d_hand[:-1])) 
            smooth_loss += torch.mean(torch.square(trans_landmarks3d_hand[1:] - trans_landmarks3d_hand[:-1]))
            smooth_loss += torch.mean(torch.square(mano_scale[1:] - mano_scale[:-1]))
            smooth_loss += torch.mean(torch.square(verts_p_hand[1:] - verts_p_hand[:-1]))
            smooth_loss = smooth_loss * self.conf_loss['smooth_weight']
            total_loss += smooth_loss
            
            if video_name == 'pinch':
                mano_vertices_tips = verts_p_hand[:,contact_idx_1,:] 
                knn_v = torch.gather(head_verts.detach(), 1, head_verts_idx_left_cheek.unsqueeze(-1).expand(-1, -1, 3)).clone()
                contact_loss = ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()/contact_idx_1.shape[0]
        
                mano_vertices_tips = verts_p_hand[:,contact_idx_2,:] 
                knn_v = torch.gather(head_verts.detach(), 1, head_verts_idx_right_cheek.unsqueeze(-1).expand(-1, -1, 3)).clone()
                contact_loss = contact_loss + ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()/contact_idx_2.shape[0]
            else:
                mano_vertices_tips = verts_p_hand[:,contact_idx,:] 
                knn_v = torch.gather(head_verts.detach(), 1, head_verts_idx_right_cheek.unsqueeze(-1).expand(-1, -1, 3)).clone()
                contact_loss = ((knn_points(mano_vertices_tips, knn_v, K=3, return_nn=False)[0])**2).mean()/contact_idx.shape[0]
                
            contact_loss = contact_loss * self.conf_loss['contact_weight']
            if 'palm' in video_name:
                contact_loss = contact_loss * 4
            total_loss += contact_loss
            
            opt_p.zero_grad()
            opt_t.zero_grad()
            total_loss.backward()
            opt_p.step()
            opt_t.step()

            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    
                    acc_loss = {}
                    acc_loss['landmark_loss'] = landmark_loss2
                    acc_loss['smooth_loss'] = smooth_loss
                    acc_loss['contact_loss'] = contact_loss

                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}' \
                                          + f', smooth_loss: {smooth_loss}'\
                                          + f', contact_loss: {contact_loss}'
                                       
                    print(loss_info)
                    
        # save vis video
        save_imgs_dir = os.path.join(savefolder, 'preprocess_shape_images')
        if not os.path.exists(save_imgs_dir):
            os.makedirs(save_imgs_dir)
        else:
            shutil.rmtree(save_imgs_dir)
            os.makedirs(save_imgs_dir)
            
        mesh_save_dir = os.path.join(savefolder, 'refine_meshes')
        if not os.path.exists(mesh_save_dir):
            os.mkdir(mesh_save_dir)
        else:
            shutil.rmtree(mesh_save_dir)
            os.mkdir(mesh_save_dir)

        save = True
        if save:
            with open(json_path, 'r') as f:
                camera_dict = json.load(f)
            
            for idx, frame in enumerate(camera_dict['frames'][::sample_interval]):
                frame['global_orient'] = global_orient[idx].detach().cpu().numpy().tolist()
                frame['hand_pose'] = hand_poses[idx].detach().cpu().numpy().tolist()
                frame['mano_scale'] = mano_scale[idx].detach().cpu().numpy().tolist()
                frame['hand_transl'] = hand_transl[idx].detach().cpu().numpy().tolist()
                frame['mano_keypoints'] = trans_landmarks2d_hand[idx, :, :2].detach().cpu().numpy().tolist()

            with open(os.path.join(savefolder, save_name_refine + '.json'), 'w') as fp:
                json.dump(camera_dict, fp)


    def run(self, deca_code_file, face_kpts_file, iris_file, mano_file, savefolder, image_path, json_path, intrinsics, size,
            save_name, conf, save_name_refine, head_mesh_folder, epoch, sample_intervals):
        
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
        w2c_p = []
        
        # visualize_images = []
        global_orient = []
        hand_poses = []
        betas = []
        hand_transl = []
        mano_scale = []
        hand_landmarks = []
        flame_scale = []
        flame_transl = []
        weight_hand_lmks = []
        
        with open(mano_file,'rb') as f:
            mano_dict = pickle.load(f)
            
        # sample_intervals = 4
        keys_to_slice = [item for item in list(mano_dict)[::sample_intervals]]
        
        mano_dict_sliced = {k: mano_dict[k] for k in keys_to_slice}
        mano_dict = mano_dict_sliced
        names = [str(k).split('/')[-1][:-4] for k in mano_dict]
        
        deca_code_sliced = {k: deca_code[k] for k in names}
        deca_code = deca_code_sliced

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                camera_dict = json.load(f)
        
        shape = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0).cuda()
        for frame in camera_dict['frames'][::sample_intervals]:
            # world to camera matrix
            world_mat = torch.tensor(np.array(frame['world_mat']).astype(np.float32)).float().cuda()
            w2c_p.append(world_mat)
            
            exps.append(torch.tensor(np.array(frame['expression']).astype(np.float32)).float().cuda())
            face_poses.append(torch.tensor(np.array(frame['pose']).astype(np.float32)).float().cuda())
            flame_scale.append(torch.tensor(np.array(frame['flame_scale']).astype(np.float32)).float().cuda())
            flame_transl.append(torch.tensor(np.array(frame['head_transl']).astype(np.float32)).float().cuda())
            
            global_orient.append(torch.tensor(np.array(frame['global_orient']).astype(np.float32)).float().cuda())
            hand_poses.append(torch.tensor(np.array(frame['hand_pose']).astype(np.float32)).float().cuda())
            betas.append(torch.tensor(np.array(frame['betas']).astype(np.float32)).float().cuda())
            hand_transl.append(torch.tensor(np.array(frame['hand_transl']).astype(np.float32)).float().cuda())
            mano_scale.append(torch.tensor(np.array(frame['mano_scale']).astype(np.float32)).float().cuda())
            hand_landmarks.append(torch.tensor(np.array(frame['gt_mano_keypoints']).astype(np.float32)).float().cuda())
            
        subject_name, sub_dir = savefolder.split('/')[-2:]
        instance_dir = os.path.join(head_mesh_folder, sub_dir, f'epoch_{epoch}')
        head_mesh_pths = sorted(glob(os.path.join(instance_dir, '*_head.ply')), key=lambda x: int(re.search(r'surface_(\d+)_head\.ply', x).group(1)))

        head_verts = []
        for head_mesh_pth in head_mesh_pths:
            vertices, faces = read_point_cloud_from_ply(head_mesh_pth)
            head_verts.append(vertices)
        min_num_verts = min(vertices.shape[0] for vertices in head_verts)
        head_verts = [vertices[:min_num_verts] for vertices in head_verts]         
        head_verts = torch.stack(head_verts, dim=0).cuda()

        w2c_p = torch.stack(w2c_p, dim=0)
        save_intrinsics = torch.tensor(camera_dict['intrinsics']).float().cuda()
        cam_intrinsics = torch.tensor(camera_dict['cam_intrinsics']).float().cuda()
        
        exps = torch.stack(exps, dim=0)
        face_poses = torch.stack(face_poses, dim=0)
        flame_scale = torch.stack(flame_scale, dim=0)
        flame_transl = torch.stack(flame_transl, dim=0)
        shape = torch.tensor(camera_dict['shape_params']).float().cuda().unsqueeze(0)

        for k_str in deca_code.keys():
            k = int(k_str)
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
        for k_str in list(deca_code.keys())[::50]:
            image = cv2.imread(image_path + '/{}.png'.format(k_str)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        face_landmarks = torch.stack(face_landmarks, dim=0)
        visualize_images = torch.cat(visualize_images, dim=0)

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

        betas = torch.stack(betas, dim=0)
        global_orient = torch.stack(global_orient, dim=0)
        hand_poses = torch.stack(hand_poses, dim=0)
        hand_transl = torch.stack(hand_transl, dim=0)
        mano_scale = torch.stack(mano_scale, dim=0)
        hand_landmarks = torch.stack(hand_landmarks, dim=0)
        # optimize
        self.optimize(shape, exps, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name, depth_images_pths, hand_mask_pths, head_mask_pths, image_pths, w2c_p, hand_transl, mano_scale, head_verts, sample_intervals, save_name_refine, save_intrinsics, cam_intrinsics, flame_scale, flame_transl, weight_hand_lmks=None)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='mano_flame_params_noflamescaleshift', help='Name for json')
    parser.add_argument('--save_name_refine', type=str, default='mano_flame_params_noflamescaleshift_refinehand', help='Name for json')
    parser.add_argument('--head_mesh_folder', type=str, help='folder saving head meshes')
    parser.add_argument('--epoch', type=int, help='load epoch')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--conf', type=str, default='.', help='path of configs of loss weights')
    parser.add_argument('--sample_intervals', type=int, default=1)

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
              json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name, conf=conf, save_name_refine=args.save_name_refine, head_mesh_folder=args.head_mesh_folder, epoch=args.epoch, sample_intervals=args.sample_intervals)
