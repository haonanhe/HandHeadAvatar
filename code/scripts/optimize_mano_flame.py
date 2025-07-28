import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import json
import pickle
import open3d as o3d

# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
GLOBAL_POSE = True
# GLOBAL_POSE = False

import cv2
import argparse

import sys
sys.path.append('../code')
from external.body_models import MANOLayer

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

    def optimize(self, shape, exps, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name):
        num_img = face_poses.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()

        # if GLOBAL_POSE:
        #     translation_p = torch.tensor([0, 0, -1]).float().cuda()
        # else:
        #     translation_p = torch.tensor([0, 0, -1]).unsqueeze(0).expand(num_img, -1).float().cuda()
        
        if GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -4]).float().cuda()
        else:
            translation_p = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
        
        translation_v = torch.tensor([0, 0, 0]).unsqueeze(0).expand(num_img, -1).float().cuda()

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
        exp = nn.Parameter(exps)
        shape = nn.Parameter(shape)
        hand_poses = nn.Parameter(hand_poses)
        translation_v = nn.Parameter(translation_v)
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
                [translation_p, translation_v, global_orient, face_poses, exp, shape],#, betas, global_orient],
                lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses],#, betas, global_orient],
                lr=1e-4)
        else:
            opt_t = torch.optim.Adam(
                [translation_p, translation_v, global_orient, face_poses, exp],#, betas, global_orient],
                lr=1e-2)
            opt_p = torch.optim.Adam(
                [hand_poses],#, betas, global_orient],
                lr=1e-4)

        scheduler_t = torch.optim.lr_scheduler.ExponentialLR(opt_t, gamma=0.99)

        # optimization steps
        len_landmark_face = face_landmarks.shape[1]
        len_landmark_hand = hand_landmarks.shape[1]
        hand_landmarks = (hand_landmarks[..., :2] - (size/2)) / (size/2)

        for k in range(1001):
        # for k in range(3001):
            face_full_pose = face_poses
            if not use_iris:
                face_full_pose = torch.cat([face_full_pose, torch.zeros_like(face_full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                face_full_pose = torch.cat([torch.zeros_like(face_full_pose[:, :3]), face_full_pose], dim=1)
            
            verts_p_face, landmarks2d_p_face, landmarks3d_p_face = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                                   expression_params=exp,
                                                                                   full_pose=face_full_pose)
            
            verts_p_face *= 4
            landmarks3d_p_face *= 4
            landmarks2d_p_face *= 4

            pred_mano_params = {
                'global_orient': global_orient,
                'hand_pose': hand_poses,
                'betas': betas.expand(num_img, -1),
                'transl': translation_v
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
                                           'face_pose': face_full_pose[i].detach().cpu().numpy().tolist(),
                                           'global_orient': global_orient[i].detach().cpu().numpy().tolist(),
                                           'hand_pose': hand_poses[i].detach().cpu().numpy().tolist(),
                                           'betas': betas.detach().cpu().numpy().tolist(),
                                           'transl': translation_v[i].detach().cpu().numpy().tolist(),
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

        betas = torch.cat(betas, dim=0)
        betas = torch.mean(betas, dim=0).unsqueeze(0)
        global_orient = torch.cat(global_orient, dim=0)
        hand_poses = torch.cat(hand_poses, dim=0)
        # landmarks = torch.stack(landmarks, dim=0)
        hand_landmarks = torch.cat(hand_landmarks, dim=0)
        # visualize_images = torch.cat(visualize_images, dim=0)
        # optimize
        self.optimize(shape, exps, face_landmarks, face_poses, betas, global_orient, hand_landmarks, hand_poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name)


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

