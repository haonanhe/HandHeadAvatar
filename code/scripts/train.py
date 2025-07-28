"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import time
import shutil

import utils.general as utils
import utils.plots as plt

import wandb
import pygit2
from functools import partial
print = partial(print, flush=True)
class TrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = self.conf.get_int('train.batch_size')
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')
        self.optimize_contact = self.conf.get_bool('train.optimize_contact')
        self.optimize_pca = self.conf.get_bool('train.optimize_pca')
        self.contact_only = self.conf.get_bool('train.contact_only')
        self.optimize_mano_pose = self.conf.get_bool('train.optimize_mano_pose')

        if not os.path.exists(os.path.join(self.exps_folder_name)):
            os.makedirs(os.path.join(self.exps_folder_name))
            
        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=kwargs['wandb_workspace'], name=self.subject + '_' + self.methodname, config=self.conf)

        self.optimize_inputs = self.optimize_latent_code or self.optimize_expression or self.optimize_pose or self.optimize_contact or self.optimize_pca or self.optimize_mano_pose
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))
        
        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['is_continue']:
            if kwargs['load_path'] != '':
                load_path = kwargs['load_path']
            else:
                load_path = self.train_dir
            if os.path.exists(load_path):
                is_continue = True
            else:
                is_continue = False
        else:
            is_continue = False

        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.train_dir, 'runconf.conf')))
        

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          sample_size=self.conf.get_int('train.num_pixels'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         sample_size=-1,
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'), shape_params=self.train_dataset.shape_params, gt_w_seg=self.conf.get_bool('loss.gt_w_seg'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        

        self.lr = self.conf.get_float('train.learning_rate')

        # exclue parameters in nonrigid_deformer
        param_ids = {id(p) for p in self.model.nonrigid_deformer.parameters()}
        params = [p for p in self.model.parameters() if id(p) not in param_ids]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
        self.optimizer_rgb = torch.optim.Adam(self.model.rendering_network.parameters(), lr=self.lr)
        
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        
        # settings for camera optimization
        if self.optimize_inputs:
            # num_training_frames = len(self.train_dataset)
            num_training_frames = self.train_dataset.full_len_dataset
            param = []
            if self.optimize_latent_code:
                self.latent_codes = torch.nn.Embedding(num_training_frames, 32, sparse=True).cuda()
                torch.nn.init.uniform_(
                    self.latent_codes.weight.data,
                    0.0,
                    1.0,
                )
                param += list(self.latent_codes.parameters())
            if self.optimize_expression:
                init_expression = torch.cat((self.train_dataset.data["expressions"], torch.zeros(self.train_dataset.data["expressions"].shape[0], self.model.deformer_network.num_exp - 50).float()), dim=1)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=True).cuda()
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            
            if self.optimize_mano_pose:
                self.mano_pose = torch.nn.Embedding(num_training_frames, 135, _weight=self.train_dataset.data["mano_hand_pose"].reshape(-1, 135), sparse=True).cuda()
                self.mano_transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["mano_transl"].reshape(-1, 3), sparse=True).cuda()
                param += list(self.mano_pose.parameters())
                param += list(self.mano_transl.parameters())
            
            self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

            num_training_frames = len(self.train_dataset)
            nonrigid_param = []
            self.nonrigid_params = torch.nn.Embedding(num_training_frames, 30, sparse=True).cuda()
            torch.nn.init.uniform_(
                self.nonrigid_params.weight.data,
                0.0,
                1.0,
            )
            nonrigid_param += list(self.nonrigid_params.parameters()) 
            self.optimizer_nonrigid_param = torch.optim.SparseAdam(nonrigid_param, self.conf.get_float('train.learning_rate_nonrigid'))
            self.optimizer_nonrigid = torch.optim.Adam(self.model.nonrigid_deformer.parameters(), lr=self.conf.get_float('train.learning_rate_nonrigid'))

        self.sched_contact_milestones = self.conf.get_list('train.contact_milestones', default=[])
        self.scheduler_nonrigid = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_nonrigid, self.sched_contact_milestones, gamma=self.sched_factor)
        self.scheduler_nonrigid_param = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_nonrigid_param, self.sched_contact_milestones, gamma=self.sched_factor)
        
        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']
            
            
            # # Filter Nonrigid Deformer
            # # 加载保存的模型状态
            # saved_model_state = torch.load(
            #     os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
            # )
            # # 获取模型的 state_dict
            # model_state_dict = saved_model_state["model_state_dict"]
            # # 创建一个新的 state_dict，排除 nonrigid_deformer 的参数
            # filtered_state_dict = {
            #     k: v for k, v in model_state_dict.items()
            #     if not k.startswith("nonrigid_deformer.")
            # }
            # # 加载过滤后的参数
            # self.model.load_state_dict(filtered_state_dict, strict=False)  # 注意：strict=False
            # # 恢复其他信息（如 epoch）
            # self.start_epoch = saved_model_state['epoch']
            

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            # self.optimizer_nonrigid.load_state_dict(data["optimizer_nonrigid_state_dict"])
            # self.optimizer_nonrigid_param.load_state_dict(data["optimizer_nonrigid_param_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            # self.scheduler.load_state_dict(data["scheduler_state_dict"])
            # self.scheduler_nonrigid.load_state_dict(data["scheduler_nonrigid_state_dict"])
            # self.scheduler_nonrigid_param.load_state_dict(data["scheduler_nonrigid_param_state_dict"])

            if self.optimize_inputs:

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    print("input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                    if self.optimize_pca:
                        self.nonrigid_params.load_state_dict(data["nonrigid_params_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")
                if self.optimize_latent_code:
                    self.latent_codes.load_state_dict(data["latent_codes_state_dict"])

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.0)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
                self.loss.flame_distance_weight = self.loss.flame_distance_weight * self.GT_lbs_factor
        if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
            self.loss.lbs_weight = 0.
            self.loss.flame_distance_weight = 0.
        
        # train_subsample = self.conf.get_string('dataset.train.subsample')
        train_subsample = self.conf.get_config('dataset.train')['subsample']
        if isinstance(train_subsample, int):
            train_subsample = int(train_subsample)
        elif isinstance(train_subsample, list):
            if len(train_subsample) == 2:
                train_subsample = int(train_subsample[1] - train_subsample[0])
            elif len(train_subsample) == 3:
                train_subsample = int(train_subsample[1] - train_subsample[0]) // train_subsample[2]
        # test_subsample = self.conf.get_string('dataset.test.subsample')
        test_subsample = self.conf.get_config('dataset.test')['subsample']
        if isinstance(test_subsample, int):
            test_subsample = int(test_subsample)
        elif isinstance(test_subsample, list):
            if len(test_subsample) == 2:
                test_subsample = int(test_subsample[1] - test_subsample[0])
            elif len(test_subsample) == 3:
                test_subsample = int(test_subsample[1] - test_subsample[0]) // test_subsample[2]
        self.eval_subsample = test_subsample // train_subsample 

    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_nonrigid_state_dict": self.scheduler_nonrigid.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_nonrigid_param_state_dict": self.scheduler_nonrigid_param.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_nonrigid_state_dict": self.optimizer_nonrigid.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_nonrigid_param_state_dict": self.optimizer_nonrigid_param.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_nonrigid_state_dict": self.scheduler_nonrigid.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_nonrigid_param_state_dict": self.scheduler_nonrigid_param.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            if not only_latest:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))\

            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_latent_code:
                dict_to_save["latent_codes_state_dict"] = self.latent_codes.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if self.optimize_pca:
                dict_to_save["nonrigid_params_state_dict"] = self.nonrigid_params.state_dict()
            if self.optimize_mano_pose:
                dict_to_save["mano_pose_state_dict"] = self.mano_pose.state_dict()
                dict_to_save["mano_transl_state_dict"] = self.mano_transl.state_dict()
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))

    def run(self):
        acc_loss = {}

        for epoch in range(self.start_epoch, self.nepochs + 1):
            # For geometry network annealing frequency band
            self.model.geometry_network.alpha = float(epoch)
            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
                self.loss.flame_distance_weight = self.loss.flame_distance_weight * self.GT_lbs_factor
            if len(self.GT_lbs_milestones) > 0 and epoch >= self.GT_lbs_milestones[-1]:
                self.loss.lbs_weight = 0.
                self.loss.flame_distance_weight = 0.

            if epoch % 5 == 0:
                self.save_checkpoints(epoch)
            else:
                self.save_checkpoints(epoch, only_latest=True)

            if (epoch % self.plot_freq == 0 and epoch < 5 and not self.optimize_contact) or (epoch % (self.plot_freq * 5) == 0 and not self.optimize_contact):
                self.model.eval()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.eval()
                    if self.optimize_latent_code:
                        self.latent_codes.eval()
                    if self.optimize_pose:
                        self.flame_pose.eval()
                        self.camera_pose.eval()
                    if self.optimize_pca and self.optimize_contact:
                        self.nonrigid_params.eval()
                    if self.optimize_mano_pose:
                        self.mano_pose.eval()
                        self.mano_transl.eval()
                eval_iterator = iter(self.plot_dataloader)
                for img_index in range(len(self.plot_dataset)):
                    start_time = time.time()
                    if img_index >= self.conf.get_int('plot.plot_nimgs'):
                        break
                    indices, model_input, ground_truth = next(eval_iterator)

                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.cuda()
                        except:
                            model_input[k] = v

                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.cuda()
                        except:
                            ground_truth[k] = v

                    if self.optimize_inputs:
                        if self.optimize_latent_code:
                            model_input['latent_code'] = self.latent_codes(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        # if self.optimize_contact:
                        #     model_input['latent_codes_nonrigid'] = self.latent_codes_nonrigid(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_expression:
                            model_input['expression'] = self.expression(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_pca:
                            model_input['nonrigid_params'] = self.nonrigid_params(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_pose:
                            model_input['flame_pose'] = self.flame_pose(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                            model_input['camera_pose'] = self.camera_pose(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_mano_pose:
                            model_input['mano_hand_pose'] = self.mano_pose(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                            model_input['mano_transl'] = self.mano_transl(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                    split = utils.split_input(model_input, self.total_pixels, n_pixels=min(33000, self.img_res[0] * self.img_res[1]))
                    # split = utils.split_input(model_input, self.total_pixels, n_pixels=self.img_res[0] * self.img_res[1])

                    res = []
                    scale = None
                    shift = None
                    for s in split:
                        s['depth_scale'] = scale
                        s['depth_shift'] = shift
                        out, sdf_function = self.model(s, return_sdf=True)
                        for k, v in out.items():
                            try:
                                out[k] = v.detach()
                            except:
                                out[k] = v
                        res.append(out)
                        scale = out['depth_scale']
                        shift = out['depth_shift']

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                    plot_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0], 'epoch_'+str(epoch))
                    img_name = model_input['img_name'][0,0].cpu().numpy()
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0], 'epoch_'+str(epoch)))
                    print("Saving image {} into {}".format(img_name, plot_dir))
                    plt.plot(img_name,
                             sdf_function,
                             model_outputs,
                             model_input['cam_pose'],
                             ground_truth,
                             plot_dir,
                             epoch,
                             self.img_res,
                             is_eval=False,
                             **self.plot_conf
                             )
                    print("Plot time per image: {}".format(time.time() - start_time))
                    del model_outputs, res, ground_truth
                self.model.train()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.train()
                    if self.optimize_latent_code:
                        self.latent_codes.train()
                    if self.optimize_pose:
                        self.flame_pose.train()
                        self.camera_pose.train()
                    if self.optimize_pca and self.optimize_contact:
                        self.nonrigid_params.train()
                    if self.optimize_mano_pose:
                        self.mano_pose.train()
                        self.mano_transl.train()
            start_time = time.time()
            

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                self.model.geometry_network.alpha = (float(epoch) + data_index/len(self.train_dataloader))
     
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                if self.optimize_inputs:
                    if self.optimize_expression:
                        ground_truth['expression'] = model_input['expression']
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pca:
                        model_input['nonrigid_params'] = self.nonrigid_params(model_input["idx"]).squeeze(1)
                    if self.optimize_latent_code:
                        model_input['latent_code'] = self.latent_codes(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        ground_truth['flame_pose'] = model_input['flame_pose']
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        ground_truth['cam_pose'] = model_input['cam_pose']
                        model_input['cam_pose'] = torch.eye(4).unsqueeze(0).repeat(ground_truth['cam_pose'].shape[0], 1, 1).cuda()
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
                    if self.optimize_mano_pose:
                        ground_truth['hand_pose'] = model_input['mano_hand_pose'].reshape(-1, 135)
                        model_input['mano_hand_pose'] = self.mano_pose(model_input["idx"]).squeeze(1)
                        ground_truth['mano_transl'] = model_input['mano_transl'].reshape(-1, 3)
                        model_input['mano_transl'] = self.mano_transl(model_input["idx"]).squeeze(1)
                        
                model_input['optimize_contact'] = self.optimize_contact
                model_input['optimize_mano_pose'] = self.optimize_mano_pose
                
                # model_outputs = self.model(model_input)
                # loss_output = self.loss(model_outputs, ground_truth)
                
                model_outputs = self.model(model_input)
                # if self.optimize_contact:
                #     prev_nonrigid_deformation = model_outputs['nonrigid_deformation_onhand_tohead'].detach()
                
                if self.contact_only and self.optimize_contact:
                    loss_output = self.loss.cal_contact_loss(model_outputs)
                    contact_loss = loss_output['contact_loss']

                    self.optimizer_nonrigid.zero_grad()
                    self.optimizer_nonrigid_param.zero_grad()

                    # contact_loss.backward(retain_graph=True)
                    contact_loss.backward()

                    self.optimizer_nonrigid.step()
                    self.optimizer_nonrigid_param.step()
                
                elif self.optimize_contact and epoch >= 30:
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss'] 
                    contact_loss = loss_output['contact_loss']
                    
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()
                    self.optimizer_nonrigid.zero_grad()
                    self.optimizer_nonrigid_param.zero_grad()
                    
                    contact_loss.backward(retain_graph=True)
                   
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()

                    loss.backward()
                    
                    self.optimizer.step()
                    if self.optimize_inputs:
                        self.optimizer_cam.step()  
                    self.optimizer_nonrigid.step()
                    self.optimizer_nonrigid_param.step()

                else:
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']
                    
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                    if self.optimize_inputs:
                        self.optimizer_cam.step()
                        
                # rgb_loss = loss_output['rgb_loss']   
                # self.optimizer_rgb.zero_grad()
                # rgb_loss.backward()
                # self.optimizer_rgb.step()
                
                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                if data_index % 50 == 0:
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                    for k, v in acc_loss.items():
                        print_str += '{}: {} '.format(k, v)
                    print(print_str)
                    acc_loss['lr_nonrigid'] = self.optimizer_nonrigid.param_groups[0]['lr']
                    wandb.log(acc_loss)
                    acc_loss = {}
                    
                del model_outputs, ground_truth, loss_output, model_input
            
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # self.scheduler.step()
            # self.scheduler_nonrigid.step()
            # self.scheduler_nonrigid_param.step()
            print("Epoch time: {}".format(time.time() - start_time))
            
            # del model_outputs, ground_truth

            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()

