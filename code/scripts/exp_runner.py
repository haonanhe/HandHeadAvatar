"""
The code is based on https://github.com/lioryariv/idr
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import sys
sys.path.append('../code')
import argparse

from scripts.train import TrainRunner
from scripts.train_2hands import TrainRunner as TrainRunner_2hands
from scripts.test import TestRunner
from scripts.test_2hands import TestRunner as TestRunner_2hands

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--is_eval', default=False, action="store_true", help='If set, only render images')
    # Training flags
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--wandb_workspace', type=str)
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    # Testing flags
    parser.add_argument('--only_json', default=False, action="store_true", help='If set, do not load images during testing. ')
    # Checkpoints
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number in case of continuing from a previous run.')
    parser.add_argument('--load_path', type=str, default='', help='If set explicitly, then load from this path, instead of the continue-scripts path')
    opt = parser.parse_args()
    
    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(opt.conf)
    train_script = conf.get_string('train.train_script')
    try:
        test_script = conf.get_string('train.test_script')
    except:
        test_script = None

    if not opt.is_eval:
        if train_script == 'train_2hands':
            runner = TrainRunner_2hands(conf=opt.conf,
                             nepochs=opt.nepoch,
                             checkpoint=opt.checkpoint,
                             is_continue=opt.is_continue,
                             load_path=opt.load_path,
                             wandb_workspace=opt.wandb_workspace,
                             )
        else:
            runner = TrainRunner(conf=opt.conf,
                                nepochs=opt.nepoch,
                                checkpoint=opt.checkpoint,
                                is_continue=opt.is_continue,
                                load_path=opt.load_path,
                                wandb_workspace=opt.wandb_workspace,
                                )
    else:
        if test_script is not None and test_script == 'test_2hands':
            runner = TestRunner_2hands(conf=opt.conf,
                            checkpoint=opt.checkpoint,
                            load_path=opt.load_path,
                            only_json=opt.only_json,
                            )
        else:
            runner = TestRunner(conf=opt.conf,
                                checkpoint=opt.checkpoint,
                                load_path=opt.load_path,
                                only_json=opt.only_json,
                                )

    runner.run()
