"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util
import os
import torch.nn as nn
from utils import mesh_util

from pytorch3d.io import load_obj
mano_arm_obj = './mano_model/data/cano_uv_closed.obj'
verts, faces, aux = load_obj(mano_arm_obj)
mano_faces = faces.verts_idx.numpy()

def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'yellow', #0
                'green', #1
                'blue', #2
                'red', #3
                'green', #4
                'blue', #5
                'red', #6
                'green', #7
                'blue', #8
                'red', #9
                'green', #10
                'blue', #11
                'red', #12
                'green', #13
                'blue', #14
                'red', #15
    ]

    color_mapping = {'cyan': cmap.colors[2],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[4],
                    'green':cmap.colors[3],
                    'yellow': cmap.colors[6],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None, None]# [1x24x3]
    verts_colors = weights * torch.from_numpy(colors).cuda()
    verts_colors = verts_colors.sum(2)
    return verts_colors

def plot(img_index, sdf_function, model_outputs, pose, ground_truth, path, epoch, img_res, plot_nimgs, min_depth, max_depth, res_init, res_up, is_eval=False):
    # arrange data to plot
    batch_size = pose.shape[0]
    num_samples = int(model_outputs['rgb_values_head'].shape[0] / batch_size)
    pred_head_mask = model_outputs['pred_head_mask']
    points_head = model_outputs['points_head'].reshape(batch_size, num_samples, 3)
    # plot rendered images
    
    # num_samples_hand = int(model_outputs['normal_values_hand'].shape[0] / batch_size)

    # depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    # depth[pred_head_mask] = rend_util.get_depth(points_head, pose).reshape(-1)[pred_head_mask]
    # depth = (depth.reshape(batch_size, num_samples, 1) - min_depth) / (max_depth - min_depth)

    # all the 1.0 and 1.0 here are just to make the depth image in range with the original min max depth parameters
    depth = torch.ones(batch_size * num_samples).cuda().float()
    depth_values = rend_util.get_depth(points_head, pose).reshape(-1)[pred_head_mask]
    # min_depth = depth_values.clone().min()
    # max_depth = depth_values.clone().max()
    depth_values = (depth_values - min_depth) / (max_depth - min_depth)
    depth[pred_head_mask] = depth_values
    
    # if (depth.min() < 0.) or (depth.max() > 1.):
    #     print("Depth out of range, min: {} and max: {}".format(depth.min(), depth.max()))
    #     depth = torch.clamp(depth, 0., 1.)
    
    pred_hand_mask = None
    
    depth_hand = None
    num_samples_hand = None
    points_hand = None
    
    if 'rgb_values_hand' in model_outputs:
        num_samples_hand = int(model_outputs['rgb_values_hand'].shape[0] / batch_size)
        pred_hand_mask = model_outputs['pred_hand_mask']
        
        points_hand = model_outputs['points_hand'].reshape(batch_size, num_samples_hand, 3)
        depth_hand = torch.ones(batch_size * num_samples_hand).cuda().float()
        depth_values_hand = rend_util.get_depth(points_hand, pose).reshape(-1)[pred_hand_mask]
        # min_depth_hand = depth_values_hand.clone().min()
        # max_depth_hand = depth_values_hand.clone().max()
        # depth_values_hand = (depth_values_hand - min_depth_hand) / (max_depth_hand - min_depth_hand)
        depth_values_hand = (depth_values_hand - min_depth) / (max_depth - min_depth)
        depth_hand[pred_hand_mask] = depth_values_hand
        
        # depth_values_hand = model_outputs['depth_values_hand']
        # depth_hand = torch.ones(batch_size * num_samples_hand).cuda().float()
        # depth_values_hand = (depth_values_hand - min_depth) / (max_depth - min_depth)
        # depth_hand[pred_hand_mask] = depth_values_hand[pred_hand_mask]

        # if (depth_hand.min() < 0.) or (depth_hand.max() > 1.):
        #     print("Depth out of range, min: {} and max: {}".format(depth_hand.min(), depth_hand.max()))
        #     depth_hand = torch.clamp(depth_hand, 0., 1.)

    depth_gt = None
    if 'depth' in ground_truth:
        # depth_gt = ((ground_truth['depth'].reshape(-1) + 1.0) * 10. - min_depth) / (max_depth - min_depth)
        depth_values_gt = ground_truth['depth'].reshape(-1)
        mask_depth = pred_head_mask | pred_hand_mask
        min_depth_gt = depth_values_gt[mask_depth].clone().min()
        max_depth_gt = depth_values_gt[mask_depth].clone().max()
        depth_values_gt = (depth_values_gt - min_depth_gt) / (max_depth_gt - min_depth_gt)
        depth_gt = torch.ones(batch_size * num_samples).cuda().float()
        depth_gt[mask_depth] = depth_values_gt[mask_depth]
        depth_gt = depth_gt.reshape(batch_size, num_samples, 1)
        if (depth_gt.min() < 0.) or (depth_gt.max() > 1.):
            print("Depth GT out of range, min: {} and max: {}".format(depth_gt.min(), depth_gt.max()))
            depth_gt = torch.clamp(depth_gt, 0., 1.)
            
    if 'depth' in model_outputs:
        # mask_depth = pred_head_mask | pred_hand_mask
        # depth_gt = torch.ones(batch_size * num_samples).cuda().float()
        # depth_gt[mask_depth] = model_outputs['depth'][mask_depth]
        # depth_gt = depth_gt.reshape(batch_size, num_samples, 1)
        # if (depth_gt.min() < 0.) or (depth_gt.max() > 1.):
        #     print("Depth GT out of range, min: {} and max: {}".format(depth_gt.min(), depth_gt.max()))
        #     depth_gt = torch.clamp(depth_gt, 0., 1.)
    
        depth_gt = model_outputs['depth'].reshape(batch_size, num_samples, 1)
        

    plot_images(model_outputs, depth, depth_hand, depth_gt, ground_truth, path, epoch, img_index, 1, img_res, batch_size, num_samples, is_eval)
    del depth, points_head, pred_head_mask
    # Generate mesh.
    if True: # is_eval:
        with torch.no_grad():
            import time
            start_time = time.time()
            meshexport = mesh_util.generate_mesh(sdf_function, verts=model_outputs['verts_head'].reshape(-1, 3), level_set=0, res_init=res_init, res_up=res_up)
            meshexport.export('{0}/surface_{1}_head.ply'.format(path, img_index), 'ply')
            print("Plot time per mesh:", time.time() - start_time)
            del meshexport
            
            # if sdf_function_hand is not None: 
                # meshexport = mesh_util.generate_mesh(sdf_function_hand, verts=model_outputs['verts_hand'].reshape(-1, 3), level_set=0, res_init=res_init, res_up=res_up)
            #     meshexport.export('{0}/surface_{1}_hand.ply'.format(path, img_index), 'ply')
            meshexport = trimesh.Trimesh(model_outputs['verts_hand'].detach().cpu().numpy().reshape(-1, 3), mano_faces.reshape(-1, 3))
            meshexport.export('{0}/surface_{1}_hand.ply'.format(path, img_index), 'ply')
            print("Plot time per mesh:", time.time() - start_time)
            del meshexport

def plot_depth_maps(depth_maps, path, epoch, img_index, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/depth'.format(path)):
        os.mkdir('{0}/depth'.format(path))
    img.save('{0}/depth/{1}.png'.format(path, img_index))


def plot_image(rgb, path, epoch, img_index, plot_nrow, img_res, type):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    # img.save('{0}/{2}/{1}.png'.format(path, img_index, type))
    img.save('{0}/{2}/{1:07d}.png'.format(path, img_index, type))


def plot_images(model_outputs, depth_image, depth_image_hand, depth_gt, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, num_samples, is_eval):
    if 'rgb' in ground_truth:
        rgb_gt = torch.ones(batch_size * num_samples, 3).cuda().float()
        # rgb_gt[model_outputs['object_mask']] = ground_truth['rgb'].squeeze(0)[model_outputs['object_mask']].reshape(-1, 3)
        rgb_gt_mask = model_outputs['head_mask'] | model_outputs['hand_mask']
        rgb_gt[rgb_gt_mask] = ground_truth['rgb'].squeeze(0)[rgb_gt_mask].reshape(-1, 3)
        rgb_gt = (rgb_gt.cuda() + 1.) / 2.
        rgb_gt = rgb_gt.reshape(batch_size, num_samples, 3)
    else:
        rgb_gt = None
    
    rgb_points_head = torch.ones(batch_size * num_samples, 3).cuda().float()
    rgb_points_head[model_outputs['pred_head_mask']] = model_outputs['rgb_values_head'][model_outputs['pred_head_mask']]
    rgb_points_head = rgb_points_head.reshape(batch_size, num_samples, 3)

    normal_points_head = torch.ones(batch_size * num_samples, 3).cuda().float()
    normal_points_head[model_outputs['pred_head_mask']] = model_outputs['normal_values_head'][model_outputs['pred_head_mask']]
    normal_points_head = normal_points_head.reshape(batch_size, num_samples, 3)

    rgb_points_head = (rgb_points_head + 1.) / 2.
    normal_points_head = (normal_points_head + 1.) / 2.
    
    output_vs_gt = rgb_points_head

    if 'normal_values_hand' in model_outputs:
        rgb_points_hand = torch.ones(batch_size * num_samples, 3).cuda().float()
        rgb_points_hand[model_outputs['pred_hand_mask']] = model_outputs['rgb_values_hand'][model_outputs['pred_hand_mask']]
        rgb_points_hand = rgb_points_hand.reshape(batch_size, -1, 3)
        rgb_points_hand = (rgb_points_hand + 1.) / 2.

        normal_points_hand = torch.ones(batch_size * num_samples, 3).cuda().float()
        normal_points_hand[model_outputs['pred_hand_mask']] = model_outputs['normal_values_hand'][model_outputs['pred_hand_mask']]
        normal_points_hand = normal_points_hand.reshape(batch_size, -1, 3)
        normal_points_hand = (normal_points_hand + 1.) / 2.

        depth_head = depth_image.clone().reshape(-1)
        depth_hand = depth_image_hand.clone().reshape(-1)
        depth_head_mask = (depth_head < depth_hand).bool()
        depth_hand_mask = (depth_head >= depth_hand).bool()
        
        num_samples_all = rgb_gt.shape[1]
        rgb_points_all = torch.ones_like(rgb_gt).float().cuda().reshape(-1,3)
        normal_points_all = torch.ones_like(rgb_gt).float().cuda().reshape(-1,3)
        rgb_points_all[depth_head_mask] = rgb_points_head.reshape(-1,3)[depth_head_mask]
        rgb_points_all[depth_hand_mask] = rgb_points_hand.reshape(-1,3)[depth_hand_mask]
        normal_points_all[depth_head_mask] = normal_points_head.reshape(-1,3)[depth_head_mask]
        normal_points_all[depth_hand_mask] = normal_points_hand.reshape(-1,3)[depth_hand_mask]
        rgb_points_all = rgb_points_all.reshape(batch_size, num_samples_all, 3)
        normal_points_all = normal_points_all.reshape(batch_size, num_samples_all, 3)
        
        output_vs_gt = torch.cat((rgb_points_hand, output_vs_gt, rgb_points_all, rgb_gt, depth_image[None,:,None].repeat(1, 1, 3), depth_gt.repeat(1, 1, 3), normal_points_hand, normal_points_head, normal_points_all), dim=0)

        if 'depth_values_flame' in model_outputs.keys():
            output_vs_gt = torch.cat((output_vs_gt, model_outputs['depth_values_flame'].reshape(-1)[None,:,None].repeat(1, 1, 3)), dim=0)
        
    else:
        output_vs_gt = torch.cat((output_vs_gt, normal_points_head), dim=0)

    if 'lbs_weight' in model_outputs:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('Paired')
        red = cmap.colors[5]
        cyan = cmap.colors[3]
        blue = cmap.colors[1]
        pink = [1, 1, 1]

        lbs_points = model_outputs['lbs_weight']
        lbs_points = lbs_points.reshape(batch_size, num_samples, -1)
        if lbs_points.shape[-1] == 5:
            colors = torch.from_numpy(np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()
        else:
            colors = torch.from_numpy(np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()

        lbs_points = (colors * lbs_points[:, :, :, None]).sum(2)
        mask = torch.logical_not(model_outputs['pred_head_mask'])
        lbs_points[mask[None, ..., None].expand(-1, -1, 3)] = 1.
        output_vs_gt = torch.cat((output_vs_gt, lbs_points), dim=0)


        lbs_points = model_outputs['gt_lbs_weight']
        lbs_points = lbs_points.reshape(batch_size, num_samples, -1)
        if lbs_points.shape[-1] == 5:
            colors = torch.from_numpy(np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()
        else:
            colors = torch.from_numpy(np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()

        lbs_points = (colors * lbs_points[:, :, :, None]).sum(2)
        mask = torch.logical_not(model_outputs['pred_head_mask'])
        lbs_points[mask[None, ..., None].expand(-1, -1, 3)] = 1.
        output_vs_gt = torch.cat((output_vs_gt, lbs_points), dim=0)

    if 'shapedirs' in model_outputs:
        shapedirs_points = model_outputs['shapedirs']
        shapedirs_points = shapedirs_points.reshape(batch_size, num_samples, 3, 50)[:, :, :, 0] * 50.

        shapedirs_points = (shapedirs_points + 1.) / 2.
        shapedirs_points = torch.clamp(shapedirs_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, shapedirs_points), dim=0)
        
        shapedirs_points = model_outputs['gt_shapedirs']
        shapedirs_points = shapedirs_points.reshape(batch_size, num_samples, 3, 50)[:, :, :, 0] * 50.

        shapedirs_points = (shapedirs_points + 1.) / 2.
        shapedirs_points = torch.clamp(shapedirs_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, shapedirs_points), dim=0)
    
    if 'nonrigid_dir' in model_outputs:
        deform_dir_points = model_outputs['nonrigid_dir']
        deform_dir_points = deform_dir_points.reshape(batch_size, num_samples, 3, 30)[:, :, :, 0] * 30.
        deform_dir_points = (deform_dir_points + 1.) / 2.
        deform_dir_points = torch.clamp(deform_dir_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, deform_dir_points), dim=0)
        
        gt_deform_dir_points = model_outputs['gt_nonrigid_dir'].clone()
        gt_deform_dir_points = gt_deform_dir_points.reshape(batch_size, num_samples, 3, 30)[:, :, :, 0] * 30.
        gt_deform_dir_points = (gt_deform_dir_points + 1.) / 2.
        gt_deform_dir_points = torch.clamp(gt_deform_dir_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, gt_deform_dir_points), dim=0)
        
    if 'deform_dir_detach' in model_outputs:
        deform_dir_points = model_outputs['deform_dir_detach']
        deform_dir_points = deform_dir_points.reshape(batch_size, num_samples, 3, 30)[:, :, :, 0] * 30.
        deform_dir_points = (deform_dir_points + 1.) / 2.
        deform_dir_points = torch.clamp(deform_dir_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, deform_dir_points), dim=0)
        
        gt_deform_dir_points = model_outputs['gt_nonrigid_dir'].clone()
        gt_deform_dir_points = gt_deform_dir_points.reshape(batch_size, num_samples, 3, 30)[:, :, :, 0] * 30.
        gt_deform_dir_points = (gt_deform_dir_points + 1.) / 2.
        gt_deform_dir_points = torch.clamp(gt_deform_dir_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, gt_deform_dir_points), dim=0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=output_vs_gt.shape[0]).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    wo_epoch_path = path.replace('/epoch_{}'.format(epoch), '')
    if not os.path.exists('{0}/rendering'.format(wo_epoch_path)):
        os.mkdir('{0}/rendering'.format(wo_epoch_path))
    img.save('{0}/rendering/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, img_index))
    if is_eval:
        plot_image(rgb_points_head, path, epoch, img_index, plot_nrow, img_res, 'rgb_head')
        plot_image(rgb_points_all, path, epoch, img_index, plot_nrow, img_res, 'rgb_all')
        plot_image(rgb_gt, path, epoch, img_index, plot_nrow, img_res, 'rgb_gt')
        plot_image(normal_points_head, path, epoch, img_index, plot_nrow, img_res, 'normal_head')
        # plot_image(lbs_points, path, epoch, img_index, plot_nrow, img_res, 'weights')
        if 'nonrigid_dir' in model_outputs:
            # plot_image(gt_deform_dir_points, path, epoch, img_index, plot_nrow, img_res, 'gt_pca_basis')
            plot_image(deform_dir_points, path, epoch, img_index, plot_nrow, img_res, 'pca_basis')
        if 'normal_values_hand' in model_outputs:
            plot_image(normal_points_all, path, epoch, img_index, plot_nrow, img_res, 'normal_all')
            # plot_image(normal_points_hand, path, epoch, img_index, plot_nrow, img_res, 'normal_hand')
    del output_vs_gt

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
