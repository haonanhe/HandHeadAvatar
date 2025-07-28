"""
# The code is based on https://github.com/lioryariv/idr
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import os
import torch
import numpy as np
import cv2
from utils import rend_util
import json

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 sample_size,
                 subsample=1,
                 use_semantics=False,
                 only_json=False,
                 ):
        """
        sub_dir: list of scripts/testing subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
        Data structure:
            RGB images in data_folder/subject_name/subject_name/sub_dir[i]/image
            foreground masks in data_folder/subject_name/subject_name/sub_dir[i]/mask
            optional semantic masks in data_folder/subject_name/subject_name/sub_dir[i]/semantic
            json files containing FLAME parameters in data_folder/subject_name/subject_name/sub_dir[i]/json_name
        json file structure:
            frames: list of dictionaries, which are structured like:
                file_path: relative path to image
                world_mat: camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                           since the camera is fixed during capture.
                           The FLAME head is centered at the origin, scaled by 4 times.
                expression: 50 dimension expression parameters
                pose: 15 dimension pose parameters
                flame_keypoints: 2D facial keypoints calculated from FLAME
            shape_params: 100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics: camera focal length fx, fy and the offsets of the principal point cx, cy

        img_res: a list containing height and width, e.g. [256, 256] or [512, 512]
        sample_size: number of pixels sampled for each scripts step, set to -1 to render full images.
        subsample: subsampling the images to reduce frame rate, mainly used for inference and evaluation
        use_semantics: whether to use semantic maps as scripts input
        only_json: used for testing, when there is no GT images, masks or semantics
        """
        sub_dir = [str(dir) for dir in sub_dir]
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.sample_size = sample_size
        self.use_semantics = use_semantics

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            "head_mask_paths": [],
            
            "depth_paths": [],
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
            "sub_dir": [],
            "bbox": [],
            "flame_scale": [],
            "flame_transl": [],
            "scales_all": [],
            
            "gt_left_hand_landmarks": [],
            "gt_right_hand_landmarks": [],

            "left_hand_mask_paths": [],
            "right_hand_mask_paths": [],

            "left_mano_global_orient": [],
            "left_mano_hand_pose": [],
            "left_mano_betas": [],
            "left_mano_transl": [],
            "left_mano_scale": [],

            "right_mano_global_orient": [],
            "right_mano_hand_pose": [],
            "right_mano_betas": [],
            "right_mano_transl": [],
            "right_mano_scale": [],
                        
            "w2c_p": [],
            "cam_intrinsics": [],
            
        }
        if self.use_semantics:
            # optionally using semantic maps
            self.data["semantic_paths"] = []

        for dir in sub_dir:
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory is empty"

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)
                
            frames = camera_dict['frames']
            # frames = frames[1229:1230]
            # frames = frames[269:270]
            # frames = frames[395:396]
            # frames = frames[451:452]
            # frames = frames[457:458]
            # frames = frames[453:454]
            # frames = frames[451:453]
            # frames = frames[474:475]
            # frames = frames[538:539]
            # frames = frames[590:591]
            # frames = frames[761:762]
            # frames = frames[800:801]
            # frames = frames[803:804]
            # frames = frames[807:808]
            frames = frames[815:816]
            # frames = frames[862:863]
            # frames = frames[925:926]
            # frames = frames[971:972]
            # frames = frames[975:976]
            # frames = frames[984:985]
            # frames = frames[1033:1034]
            # frames = frames[1086:1087]
            # frames = frames[1091:1092]
            # frames = frames[1270:1271]
            # frames = frames[1280:1281]
            # frames = frames[1335:1336]
            # frames = frames[1486:1487]

            for frame in frames:
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                self.data["w2c_p"].append(world_mat)
                ## camera to world matrix
                self.data["world_mats"].append(rend_util.load_K_Rt_from_P(None, world_mat[:3,:])[1])
                
                # cam = rend_util.load_K_Rt_from_P(None, world_mat[:3,:])[1]
                # extrinsic = np.eye(4)
                # R = cam[:3, :3].copy()
                # t = cam[:3, 3].copy()
                # # ######################################## for deca
                # # angle_radians = np.radians(180)
                # # R_Z = np.array([
                # #     [np.cos(angle_radians), -np.sin(angle_radians), 0],
                # #     [np.sin(angle_radians), np.cos(angle_radians), 0],
                # #     [0, 0, 1]
                # # ])
                # # R = np.dot(R_Z, R)
                # # t = np.dot(R_Z, t)

                # F_x = np.array([
                #     [-1, 0, 0],
                #     [0, 1, 0],
                #     [0, 0, 1]
                # ])
                # R = np.dot(F_x, R)
                # t = np.dot(F_x, t)
                # # ######################################## for deca
                # extrinsic[:3, :3] = R
                # extrinsic[:3, 3] = t #- (R @ head_transl)
                # self.data["world_mats"].append(extrinsic)
                
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                # self.data["flame_transl"].append(np.array(frame['head_transl']).astype(np.float32))
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(os.path.join(os.path.dirname(image_path.replace('image', 'mask')), '%07d.png'%(int(frame["file_path"].split('/')[-1]))))
                self.data["head_mask_paths"].append(os.path.join(os.path.dirname(image_path.replace('image', 'head_mask')), '%07d.png'%(int(frame["file_path"].split('/')[-1])-1)))
                # Append left and right hand mask paths separately
                self.data["left_hand_mask_paths"].append(
                    os.path.join(
                        os.path.dirname(image_path.replace('image', 'hand_mask_left')), 
                        '%07d.png' % (int(frame["file_path"].split('/')[-1]) - 1)
                    )
                )

                self.data["right_hand_mask_paths"].append(
                    os.path.join(
                        os.path.dirname(image_path.replace('image', 'hand_mask_right')), 
                        '%07d.png' % (int(frame["file_path"].split('/')[-1]) - 1)
                    )
                )
                self.data["depth_paths"].append(os.path.join(os.path.dirname(image_path.replace('image', 'depth_imgs_v2')), '%07d.jpg'%(int(frame["file_path"].split('/')[-1]))))
                # self.data["depth_paths"].append(os.path.join(os.path.dirname(image_path.replace('image', 'depth')), '%07d.tiff'%(int(frame["file_path"].split('/')[-1]))))
                if use_semantics:
                    self.data["semantic_paths"].append(image_path.replace('image', 'semantic'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))
                self.data["sub_dir"].append(dir)
                self.data["bbox"].append(((np.array(frame['bbox']) + 1.) * np.array([img_res[0],img_res[1],img_res[0],img_res[1]])/ 2).astype(int))
                
                if 'head_transl' in frame.keys():
                    head_transl = np.array(frame['head_transl']).astype(np.float32)
                    self.data["flame_transl"].append(head_transl)
                else:
                    self.data["flame_transl"].append(None)
                if 'flame_scale' in frame.keys():
                    flame_scale = np.array(frame['flame_scale']).astype(np.float32)
                    self.data["flame_scale"].append(flame_scale)
                else:
                    self.data["flame_transl"].append(None)
                    
                self.data["scales_all"].append(np.array(frame['scales_all']).astype(np.float32))
                
                if 'left_hand_pose' in frame.keys() and 'right_hand_pose' in frame.keys():
                    # Left hand MANO parameters
                    self.data["left_mano_global_orient"].append(np.array(frame['left_global_orient']).astype(np.float32))
                    self.data["left_mano_hand_pose"].append(np.array(frame['left_hand_pose']).astype(np.float32))
                    self.data["left_mano_betas"].append(np.array(frame['left_betas']).astype(np.float32))
                    left_transl = np.array(frame['left_hand_transl'])
                    self.data["left_mano_transl"].append(left_transl.astype(np.float32))
                    left_scale = np.array(frame['left_mano_scale'])
                    self.data["left_mano_scale"].append(left_scale.astype(np.float32))

                    # Right hand MANO parameters
                    self.data["right_mano_global_orient"].append(np.array(frame['right_global_orient']).astype(np.float32))
                    self.data["right_mano_hand_pose"].append(np.array(frame['right_hand_pose']).astype(np.float32))
                    self.data["right_mano_betas"].append(np.array(frame['right_betas']).astype(np.float32))
                    right_transl = np.array(frame['right_hand_transl'])
                    self.data["right_mano_transl"].append(right_transl.astype(np.float32))
                    right_scale = np.array(frame['right_mano_scale'])
                    self.data["right_mano_scale"].append(right_scale.astype(np.float32))

                    # Ground truth landmarks for both hands
                    self.data["gt_left_hand_landmarks"].append(np.array(frame['gt_left_mano_keypoints']).astype(np.float32))
                    self.data["gt_right_hand_landmarks"].append(np.array(frame['gt_right_mano_keypoints']).astype(np.float32))

                else:
                    # Left hand placeholders
                    self.data["left_mano_global_orient"].append(None)
                    self.data["left_mano_hand_pose"].append(None)
                    self.data["left_mano_betas"].append(None)
                    self.data["left_mano_transl"].append(None)
                    self.data["left_mano_scale"].append(None)

                    # Right hand placeholders
                    self.data["right_mano_global_orient"].append(None)
                    self.data["right_mano_hand_pose"].append(None)
                    self.data["right_mano_betas"].append(None)
                    self.data["right_mano_transl"].append(None)
                    self.data["right_mano_scale"].append(None)

                    # Landmark placeholders
                    self.data["gt_left_hand_landmarks"].append(None)
                    self.data["gt_right_hand_landmarks"].append(None)

        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']
        # construct intrinsic matrix in pixels
        intrinsics = np.eye(3)
        if focal_cxcy[3] > 1:
            # An old format of my datasets...
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0] / 512
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1] / 512
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0] / 512
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1] / 512
        else:
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0]
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1]
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0]
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1]
        self.intrinsics = intrinsics
        
        self.cam_intrinsics = torch.tensor(camera_dict['cam_intrinsics']).float()

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
        elif isinstance(subsample, list):
            if len(subsample) == 2:
                subsample = list(range(subsample[0], subsample[1]))
            for k, v in self.data.items():
                self.data[k] = [v[s] for s in subsample]

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["flame_scale"] = torch.from_numpy(np.stack(self.data["flame_scale"], 0))
        self.data["flame_transl"] = torch.from_numpy(np.stack(self.data["flame_transl"], 0))
        # self.data["flame_transl"] = torch.from_numpy(np.stack(self.data["flame_transl"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()
        self.data["w2c_p"] = torch.from_numpy(np.stack(self.data["w2c_p"], 0)).float()
        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json
        
        # Left hand MANO parameters
        if self.data["left_mano_global_orient"][0] is not None:
            self.data["left_mano_global_orient"] = torch.from_numpy(np.stack(self.data["left_mano_global_orient"], 0))
            self.data["left_mano_hand_pose"] = torch.from_numpy(np.stack(self.data["left_mano_hand_pose"], 0))
            self.data["left_mano_betas"] = torch.from_numpy(np.stack(self.data["left_mano_betas"], 0))
            self.data["left_mano_transl"] = torch.from_numpy(np.stack(self.data["left_mano_transl"], 0))
            self.data["left_mano_scale"] = torch.from_numpy(np.stack(self.data["left_mano_scale"], 0))
            self.data["gt_left_hand_landmarks"] = torch.from_numpy(np.stack(self.data["gt_left_hand_landmarks"], 0))

        # Right hand MANO parameters
        if self.data["right_mano_global_orient"][0] is not None:
            self.data["right_mano_global_orient"] = torch.from_numpy(np.stack(self.data["right_mano_global_orient"], 0))
            self.data["right_mano_hand_pose"] = torch.from_numpy(np.stack(self.data["right_mano_hand_pose"], 0))
            self.data["right_mano_betas"] = torch.from_numpy(np.stack(self.data["right_mano_betas"], 0))
            self.data["right_mano_transl"] = torch.from_numpy(np.stack(self.data["right_mano_transl"], 0))
            self.data["right_mano_scale"] = torch.from_numpy(np.stack(self.data["right_mano_scale"], 0))
            self.data["gt_right_hand_landmarks"] = torch.from_numpy(np.stack(self.data["gt_right_hand_landmarks"], 0))

    def __len__(self):
        return len(self.data["image_paths"])
        # return 1570

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        

        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "uv": uv,
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            # "flame_transl": self.data["flame_transl"][idx],
            "cam_pose": self.data["world_mats"][idx],
            "flame_scale": self.data["flame_scale"][idx],
            "flame_transl": self.data["flame_transl"][idx],
            
            "mano_left_global_orient": self.data["left_mano_global_orient"][idx],
            "mano_left_hand_pose": self.data["left_mano_hand_pose"][idx],
            "mano_left_betas": self.data["left_mano_betas"][idx],
            "mano_left_transl": self.data["left_mano_transl"][idx],
            "mano_left_scale": self.data["left_mano_scale"][idx],

            "mano_right_global_orient": self.data["right_mano_global_orient"][idx],
            "mano_right_hand_pose": self.data["right_mano_hand_pose"][idx],
            "mano_right_betas": self.data["right_mano_betas"][idx],
            "mano_right_transl": self.data["right_mano_transl"][idx],
            "mano_right_scale": self.data["right_mano_scale"][idx],
                        
            "w2c_p": self.data["w2c_p"][idx],
            "cam_intrinsics": self.cam_intrinsics,
            "img_res": torch.LongTensor([self.img_res][0]),
        }
        if not self.only_json:
            sample["object_mask"] = torch.from_numpy(rend_util.load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1)).bool()
            sample["head_mask"] = torch.from_numpy(rend_util.load_mask(self.data["head_mask_paths"][idx], self.img_res).reshape(-1)).bool()
            sample["left_hand_mask"] = torch.from_numpy(
                rend_util.load_mask(self.data["left_hand_mask_paths"][idx], self.img_res).reshape(-1)
            ).bool()

            sample["right_hand_mask"] = torch.from_numpy(
                rend_util.load_mask(self.data["right_hand_mask_paths"][idx], self.img_res).reshape(-1)
            ).bool()
        if not self.only_json:
            # ground_truth = {
            #     "rgb": torch.from_numpy(rend_util.load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1)
            #                             .transpose(1, 0)).float() * sample["object_mask"].unsqueeze(1).float()
            #                             + (1 - sample["object_mask"].unsqueeze(1).float())
            # }
            ground_truth = {
                "rgb": torch.from_numpy(rend_util.load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1)
                                        .transpose(1, 0)).float() * sample["object_mask"].unsqueeze(1).float()
                                        + (1 - sample["object_mask"].unsqueeze(1).float()),
                "mask": torch.from_numpy(rend_util.load_mask(self.data["mask_paths"][idx], self.img_res).reshape(1, -1)
                                        .transpose(1, 0)).float(),
                "mask_head": torch.from_numpy(rend_util.load_mask(self.data["head_mask_paths"][idx], self.img_res).reshape(1, -1)
                                        .transpose(1, 0)).float(),
                "mask_image_head": torch.from_numpy(rend_util.load_mask(self.data["head_mask_paths"][idx], self.img_res).reshape(1, -1)
                                        .transpose(1, 0)).float(),
                
                "mask_left_hand": torch.from_numpy(rend_util.load_mask(self.data["left_hand_mask_paths"][idx], self.img_res).reshape(1, -1)
                                        .transpose(1, 0)).float(),
                "rgb_image_left_hand": torch.from_numpy(rend_util.load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1)
                                                        .transpose(1, 0)).float() * sample["left_hand_mask"].unsqueeze(1).float()
                                                        + (1 - sample["left_hand_mask"].unsqueeze(1).float()),
                "mask_image_left_hand": torch.from_numpy(rend_util.load_mask(self.data["left_hand_mask_paths"][idx], self.img_res).reshape(1, -1)
                                                        .transpose(1, 0)).float(),
                                                        
                "depth_image_left_hand": torch.from_numpy(rend_util.load_depth(self.data["depth_paths"][idx], self.img_res).reshape(1, -1)
                                                        .transpose(1, 0)).float() * sample["left_hand_mask"].unsqueeze(1).float()
                                                        + (1 - sample["left_hand_mask"].unsqueeze(1).float()),

                "left_hand_lmk": self.data["gt_left_hand_landmarks"][idx],


                "mask_right_hand": torch.from_numpy(rend_util.load_mask(self.data["right_hand_mask_paths"][idx], self.img_res).reshape(1, -1)
                                                        .transpose(1, 0)).float(),
                "rgb_image_right_hand": torch.from_numpy(rend_util.load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1)
                                                        .transpose(1, 0)).float() * sample["right_hand_mask"].unsqueeze(1).float()
                                                        + (1 - sample["right_hand_mask"].unsqueeze(1).float()),
                "mask_image_right_hand": torch.from_numpy(rend_util.load_mask(self.data["right_hand_mask_paths"][idx], self.img_res).reshape(1, -1)
                                                        .transpose(1, 0)).float(),
                                                        
                "depth_image_right_hand": torch.from_numpy(rend_util.load_depth(self.data["depth_paths"][idx], self.img_res).reshape(1, -1)
                                                        .transpose(1, 0)).float() * sample["right_hand_mask"].unsqueeze(1).float()
                                                        + (1 - sample["right_hand_mask"].unsqueeze(1).float()),

                "right_hand_lmk": self.data["gt_right_hand_landmarks"][idx],
            }
        else:
            ground_truth = {}
        if self.use_semantics:
            if not self.only_json:
                ground_truth["semantics"] = torch.from_numpy(rend_util.load_semantic(self.data["semantic_paths"][idx], self.img_res).reshape(-1, self.img_res[0] * self.img_res[1]).transpose(1, 0)).float()
        ground_truth["depth"] = torch.from_numpy(rend_util.load_depth(self.data["depth_paths"][idx], self.img_res).reshape(-1))
        # ground_truth["depth"] = torch.from_numpy(rend_util.load_depth_metahuman(self.data["depth_paths"][idx], self.img_res, self.data["scales_all"][idx]).reshape(-1))

        if self.only_json:
            assert self.sample_size == -1  # for testing, we need to run all pixels
            
        if self.sample_size != -1:
            # sampling half of the pixels from entire image
            sampling_idx = torch.randperm(self.total_pixels)[:self.sample_size//2]
            rgb = ground_truth["rgb"]
            mask = sample["object_mask"]
            head_mask = sample["head_mask"]
            left_hand_mask = sample["left_hand_mask"]
            right_hand_mask = sample["right_hand_mask"]
            depth = ground_truth["depth"]
            if self.use_semantics:
                semantic = ground_truth["semantics"]

            ground_truth["rgb"] = rgb[sampling_idx, :]
            sample["object_mask"] = mask[sampling_idx]
            sample["head_mask"] = head_mask[sampling_idx]
            sample["left_hand_mask"] = left_hand_mask[sampling_idx]
            sample["right_hand_mask"] = right_hand_mask[sampling_idx]
            ground_truth["depth"] = depth[sampling_idx]
            if self.use_semantics:
                ground_truth["semantics"] = semantic[sampling_idx, :]
            sample["uv"] = uv[sampling_idx, :]

            # and half of the pixels from bbox
            bbox = self.data["bbox"][idx]
            bbox_inside = torch.logical_and(torch.logical_and(uv[:, 0] > bbox[0], uv[:, 1] > bbox[1]),
                                            torch.logical_and(uv[:, 0] < bbox[2], uv[:, 1] < bbox[3]))
            uv_bbox = uv[bbox_inside]
            rgb_bbox = rgb[bbox_inside]
            mask_bbox = mask[bbox_inside]
            head_mask_bbox = head_mask[bbox_inside]
            left_hand_mask_bbox = left_hand_mask[bbox_inside]
            right_hand_mask_bbox = right_hand_mask[bbox_inside]
            depth_bbox = depth[bbox_inside]
            if self.use_semantics:
                semantic_bbox = semantic[bbox_inside]

            sampling_idx_bbox = torch.randperm(len(uv_bbox))[:self.sample_size - self.sample_size // 2]
            ground_truth["rgb"] = torch.cat([ground_truth["rgb"], rgb_bbox[sampling_idx_bbox, :]], 0)
            sample["object_mask"] = torch.cat([sample["object_mask"], mask_bbox[sampling_idx_bbox]], 0)
            sample["head_mask"] = torch.cat([sample["head_mask"], head_mask_bbox[sampling_idx_bbox]], 0)
            sample["left_hand_mask"] = torch.cat([sample["left_hand_mask"], left_hand_mask_bbox[sampling_idx_bbox]], 0)
            sample["right_hand_mask"] = torch.cat([sample["right_hand_mask"], right_hand_mask_bbox[sampling_idx_bbox]], 0)
            ground_truth["depth"] = torch.cat([ground_truth["depth"], depth_bbox[sampling_idx_bbox]], 0)

            if self.use_semantics:
                ground_truth["semantics"] = torch.cat([ground_truth["semantics"], semantic_bbox[sampling_idx_bbox]], 0)

            sample["uv"] = torch.cat([sample["uv"], uv_bbox[sampling_idx_bbox, :]], 0)
            
            sample["sampling_idx"] = sampling_idx
            sample["bbox_inside"] = bbox_inside
            sample["sampling_idx_bbox"] = sampling_idx_bbox
        
        sample["depth"] = ground_truth["depth"]
        sample["depth_image_left_hand"] = ground_truth["depth_image_left_hand"]
        sample["depth_image_right_hand"] = ground_truth["depth_image_right_hand"]
        
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
