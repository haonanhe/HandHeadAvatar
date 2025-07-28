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

import utils.general as utils
import utils.plots_2hands as plt

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
            os.mkdir(os.path.join(self.exps_folder_name))
            
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

        # with open(os.path.join(self.train_dir, 'runconf.conf'), 'a+') as f:
        #     f.write(str(pygit2.Repository('.').head.shorthand) + '\n'  # branch
        #             + str(pygit2.Repository('.').head.target) + '\n') # commit hash)


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

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # params = [p for p in self.model.parameters() if p not in self.model.nonrigid_deformer_network.parameters()]
        param_ids = {id(p) for p in self.model.nonrigid_deformer_network.parameters()}
        params = [p for p in self.model.parameters() if id(p) not in param_ids]
        # param_ids = {id(p) for p in self.model.nonrigid_deformer_network.parameters()}
        # param_ids_deformer = {id(p) for p in self.model.deformer_network.parameters()}
        # params = [p for p in self.model.parameters() if id(p) not in param_ids and id(p) not in param_ids_deformer]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            num_training_frames = 1570
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
            
            # if self.optimize_pca:
            #     self.deform_params = torch.nn.Embedding(num_training_frames, 30, sparse=True).cuda()
            #     torch.nn.init.uniform_(
            #         self.deform_params.weight.data,
            #         0.0,
            #         1.0,
            #     )
            #     param += list(self.deform_params.parameters())
            # self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

        # if self.optimize_contact:
            # num_training_frames = len(self.train_dataset)
            nonrigid_param = []
            self.deform_params = torch.nn.Embedding(num_training_frames, 30, sparse=True).cuda()
            torch.nn.init.uniform_(
                self.deform_params.weight.data,
                0.0,
                1.0,
            )
            nonrigid_param += list(self.deform_params.parameters()) 
            self.optimizer_nonrigid_code = torch.optim.SparseAdam(nonrigid_param, self.conf.get_float('train.learning_rate_nonrigid'))
            self.optimizer_nonrigid = torch.optim.Adam(self.model.nonrigid_deformer_network.parameters(), lr=self.conf.get_float('train.learning_rate_nonrigid'))
            
            self.latent_codes_nonrigid = torch.nn.Embedding(num_training_frames, 32, sparse=True).cuda()
            torch.nn.init.uniform_(
                self.latent_codes_nonrigid.weight.data,
                0.0,
                1.0,
            )

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

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
                        self.deform_params.load_state_dict(data["deform_params_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")
                if self.optimize_latent_code:
                    self.latent_codes.load_state_dict(data["latent_codes_state_dict"])
                # if self.optimize_contact:
                #     self.latent_codes_nonrigid.load_state_dict(data["latent_codes_nonrigid_state_dict"])

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
            
        train_subsample = int(self.conf.get_string('dataset.train.subsample'))
        test_subsample = int(self.conf.get_string('dataset.test.subsample'))
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
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
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
                dict_to_save["deform_params_state_dict"] = self.deform_params.state_dict()
            if self.optimize_contact:
                dict_to_save["latent_codes_nonrigid_state_dict"] = self.latent_codes_nonrigid.state_dict()
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

            if (epoch % self.plot_freq == 0 and epoch < 5) or (epoch % (self.plot_freq * 5) == 0):
            # if True:
                self.model.eval()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.eval()
                    if self.optimize_pca:
                        self.deform_params.eval()
                    if self.optimize_latent_code:
                        self.latent_codes.eval()
                    if self.optimize_pose:
                        self.flame_pose.eval()
                        self.camera_pose.eval()
                    if self.optimize_contact:
                        self.latent_codes_nonrigid.eval()
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
                            # use the latent code from the first scripts frame
                            # model_input['latent_code'] = self.latent_codes(torch.LongTensor([0]).cuda()).squeeze(1).detach()
                            model_input['latent_code'] = self.latent_codes(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_contact:
                            model_input['latent_codes_nonrigid'] = self.latent_codes_nonrigid(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_expression:
                            model_input['expression'] = self.expression(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
                        if self.optimize_pca:
                            model_input['deform_params'] = self.deform_params(model_input["idx"]*self.eval_subsample).squeeze(1).detach()
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
                        out, sdf_function, sdf_function_left_hand, sdf_function_right_hand = self.model(s, return_sdf=True)
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
                             sdf_function_left_hand,
                             sdf_function_right_hand,
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
                    if self.optimize_pca:
                        self.deform_params.train()
                    if self.optimize_latent_code:
                        self.latent_codes.train()
                    if self.optimize_pose:
                        self.flame_pose.train()
                        self.camera_pose.train()
                    if self.optimize_contact:
                        self.latent_codes_nonrigid.train()
                    if self.optimize_mano_pose:
                        self.mano_pose.train()
                        self.mano_transl.train()
            start_time = time.time()
            
            prev_nonrigid_deformation = None

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                self.model.geometry_network.alpha = (float(epoch) + data_index/len(self.train_dataloader))
     
                # if model_input['img_name'].item() != 1217:
                #     continue

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
                        model_input['deform_params'] = self.deform_params(model_input["idx"]).squeeze(1)
                    if self.optimize_latent_code:
                        model_input['latent_code'] = self.latent_codes(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        ground_truth['flame_pose'] = model_input['flame_pose']
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        ground_truth['cam_pose'] = model_input['cam_pose']
                        model_input['cam_pose'] = torch.eye(4).unsqueeze(0).repeat(ground_truth['cam_pose'].shape[0], 1, 1).cuda()
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
                    if self.optimize_contact:
                        model_input['latent_codes_nonrigid'] = self.latent_codes_nonrigid(model_input["idx"]).squeeze(1)
                    if self.optimize_mano_pose:
                        ground_truth['hand_pose'] = model_input['mano_hand_pose'].reshape(-1, 135)
                        model_input['mano_hand_pose'] = self.mano_pose(model_input["idx"]).squeeze(1)
                        ground_truth['mano_transl'] = model_input['mano_transl'].reshape(-1, 3)
                        model_input['mano_transl'] = self.mano_transl(model_input["idx"]).squeeze(1)
                        
                model_input['optimize_contact'] = self.optimize_contact
                
                # model_outputs = self.model(model_input)
                # loss_output = self.loss(model_outputs, ground_truth)
                
                model_outputs = self.model(model_input)
                prev_nonrigid_deformation = None
                loss_output = self.loss(model_outputs, ground_truth, prev_nonrigid_deformation)
                # prev_nonrigid_deformation = model_outputs['nonrigid_deformation_handsurf_tohead'].detach()
                
                if self.contact_only:
                    contact_loss = loss_output['contact_loss']

                    self.optimizer_nonrigid.zero_grad()
                    self.optimizer_nonrigid_code.zero_grad()

                    contact_loss.backward(retain_graph=True)

                    self.optimizer_nonrigid.step()
                    self.optimizer_nonrigid_code.step()
                
                # if self.contact_only:
                #     contact_loss = loss_output['contact_loss']
                #     # contact_loss = loss_output['deform_dir_loss'] 
                #     # contact_loss = loss_output['contact_regularation_surface'] 
                    
                #     self.optimizer_nonrigid.zero_grad()
                #     self.optimizer_nonrigid_code.zero_grad()

                #     contact_loss.backward(retain_graph=True)
                #     # deform_dir_loss.backward(retain_graph=True)

                #     self.optimizer_nonrigid.step()
                #     self.optimizer_nonrigid_code.step()
                        
                elif self.optimize_contact and epoch >= 10:
                    loss = loss_output['loss'] 
                    contact_loss = loss_output['contact_loss']
                    
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()
                    self.optimizer_nonrigid.zero_grad()
                    self.optimizer_nonrigid_code.zero_grad()
                    
                    contact_loss.backward(retain_graph=True)
                    
                    # grads = [p.grad for p in self.model.nonrigid_deformer_network.parameters()] 
                    # if grads[0].sum() != 0 or self.deform_params.weight.grad._values().sum() != 0:
                    #     breakpoint()
                    
                    # deform_dir_loss = loss_output['deform_dir_loss'] 
                    # deform_dir_loss.backward(retain_graph=True)

                    # contact_loss.backward()
                    
                    # grads = [p.grad for p in self.model.nonrigid_deformer_network.parameters()] 
                    # if grads[0].sum() != 0 or self.deform_params.weight.grad._values().sum() != 0:
                    #     breakpoint()
                    # # if self.deform_params.weight.grad._values().sum() != 0:
                    # #     breakpoint()
                    
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()

                    loss.backward()
                    
                    # grads = [p.grad for p in self.model.nonrigid_deformer_network.parameters()] 
                    # if grads[0].sum() != 0 or self.deform_params.weight.grad._values().sum() != 0:
                    #     breakpoint()
                    # # # if self.deform_params.weight.grad._values().sum() != 0:
                    # # #     breakpoint()
                    
                    self.optimizer.step()
                    if self.optimize_inputs:
                        self.optimizer_cam.step()  
                    self.optimizer_nonrigid.step()
                    self.optimizer_nonrigid_code.step()

                else:
                    loss = loss_output['loss']
                    
                    self.optimizer.zero_grad()
                    if self.optimize_inputs:
                        self.optimizer_cam.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                    if self.optimize_inputs:
                        self.optimizer_cam.step()
                
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
                    wandb.log(acc_loss)
                    acc_loss = {}

                del model_outputs, ground_truth, loss_output, model_input
                
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
            self.scheduler.step()
            print("Epoch time: {}".format(time.time() - start_time))
            
            # del model_outputs, ground_truth

            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()

