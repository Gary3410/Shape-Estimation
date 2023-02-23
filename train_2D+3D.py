from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from layers.modules import AffinityLoss
from yolact import Yolact
import torch.nn.functional as F
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval_affinity as eval_script



import argparse
import datetime
import os.path as osp
import shutil
import time

import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_gpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
os.environ['RANK'] = '0'
os.environ['MASTER_PORT'] = '29555'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=20, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default="/home/potato/workplace/SoftGroup/work_dirs/weights-scratch/yolact_resnet50_198_21614_interrupt.pth", type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.') # -1
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default='yolact_resnet50_config',
                    help='The config object to use.')
parser.add_argument('--save_interval', default=1000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=1000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=5, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=10000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')


parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args_yolact = parser.parse_args()


if args_yolact.config is not None:
    set_cfg(args_yolact.config)

if args_yolact.dataset is not None:
    set_dataset(args_yolact.dataset)

"""
if args_yolact.autoscale and args_yolact.batch_size != 8:
    factor = args_yolact.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args_yolact.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]
"""
# Update training parameters from the config if necessary
def replace(name):
    if getattr(args_yolact, name) == None: setattr(args_yolact, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args_yolact.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args_yolact.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args_yolact.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args_yolact.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss, affinity_criterion:AffinityLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
        self.affinity_criterion = affinity_criterion


        # self.affinity_loss = None

    def forward(self, images, targets, masks, num_crowds, feature_list=None):
        #print("feature_list", feature_list.shape)
        preds, context_prior_map = self.net(images, feature_list=feature_list)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)

        seg_mask = []
        for i in range(len(masks)):
            number, h, w = masks[i].shape
            seg_mask_one = masks[i].view(number, -1)

            sum_mask = torch.sum(seg_mask_one, dim=0)
            sum_mask_id = torch.nonzero(torch.gt(sum_mask, 1))
            seg_mask_one[:, sum_mask_id] = 0
            obj_one = targets[i][:, 4:5]

            seg_mask_one = torch.mul(seg_mask_one, obj_one)
            seg_mask_one = seg_mask_one.view(number, h, -1)
            seg_mask_one = torch.sum(seg_mask_one, dim=0)
            seg_mask.append(seg_mask_one)

        seg_mask = torch.stack(seg_mask)

        ideal_affinity_matrix = self._construct_ideal_affinity_matrix(seg_mask, label_size=(69, 69))
        affinity_loss = self.affinity_criterion(context_prior_map, ideal_affinity_matrix)

        return losses, affinity_loss

    def _construct_ideal_affinity_matrix(self, label, label_size, num_classes=27):
        label = label.unsqueeze(0)
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")

        #print(scaled_labels.shape)
        scaled_labels = scaled_labels.squeeze(dim=0)
        scaled_labels = scaled_labels.squeeze_().long()
        #print(scaled_labels.shape)
        scaled_labels[scaled_labels >= 27] = num_classes

        one_hot_labels = F.one_hot(scaled_labels, num_classes + 1)
        #print(one_hot_labels.shape)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, num_classes + 1).float()

        #print(one_hot_labels.shape)
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
            one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        #print("before_prepare:{}".format(torch.cuda.memory_allocated(0)))
        splits = prepare_data(inputs[0], devices, allocation=args_yolact.batch_alloc)
        #print("after_prepare:{}".format(torch.cuda.memory_allocated(0)))
        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out

def get_args():
    # default='/home/potato/workplace/SoftGroup/configs/softgroup_s3dis_backbone_fold5.yaml'
    # default='/home/potato/workplace/SoftGroup/work_dirs'
    parser_softgroup = argparse.ArgumentParser('SoftGroup')
    parser_softgroup.add_argument('--config', type=str,default='/home/potato/workplace/SoftGroup/configs/softgroup_s3dis_backbone_fold5.yaml',  help='path to config file')
    parser_softgroup.add_argument('--dist', action='store_true', default=False, help='run with distributed parallel')
    parser_softgroup.add_argument('--resume', type=str, help='path to resume from')
    parser_softgroup.add_argument('--work_dir', type=str, help='working directory')
    parser_softgroup.add_argument('--skip_validate', action='store_true', default=False, help='skip validation')
    args = parser_softgroup.parse_args()
    return args


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups[:3]:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr

def set_lr_affinity(optimizer, new_lr):
    for param_group in optimizer.param_groups[3:]:
        param_group['lr'] = new_lr

def gradinator(x):
    x.requires_grad = False
    return x


def prepare_data(datum, devices: list = None, allocation: list = None, fusion_feature:list=None):
    with torch.no_grad():
        images, (targets, masks, num_crowds, path), fusion_feature = datum
        feature_list = []

        for batch_id in range(fusion_feature[0].shape[0]):
            feature_list.append(fusion_feature[0][batch_id, :, :, :])

       # print("feature_list", len(feature_list))


        path_id = list(range(args_yolact.batch_size))
        images_sample = []
        targets_sample = []
        masks_sample = []
        num_crowds_sampler = []

        for start_indx in range(0, args_yolact.batch_size, 5):
            sampler_indx = np.random.choice(np.arange(5).astype(int), size=1, replace=False)
            images_one = images[start_indx:start_indx + 5]
            targets_one = targets[start_indx:start_indx + 5]
            masks_one = masks[start_indx:start_indx + 5]
            num_crowds_one = num_crowds[start_indx:start_indx + 5]
            for sampler_indx_one in sampler_indx:
                images_sample.append(images_one[sampler_indx_one])
                targets_sample.append(targets_one[sampler_indx_one])
                masks_sample.append(masks_one[sampler_indx_one])
                num_crowds_sampler.append(num_crowds_one[sampler_indx_one])

        if devices is None:
            devices = ['cuda:0'] if args_yolact.cuda else ['cpu']
        if allocation is None:
            allocation = [len(images_sample) // len(devices)] * (len(devices) - 1)
            allocation.append(len(images_sample) - sum(allocation))  # The rest might need more/less

        cur_idx = 0


        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images_sample[cur_idx] = gradinator(images_sample[cur_idx].to(device))
                targets_sample[cur_idx] = gradinator(targets_sample[cur_idx].to(device))
                masks_sample[cur_idx] = gradinator(masks_sample[cur_idx].to(device))
                feature_list[cur_idx] = feature_list[cur_idx].to(device)
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images_sample[random.randint(0, len(images_sample) - 1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images_sample, targets_sample, masks_sample, num_crowds_sampler)):
                images_sample[idx], targets_sample[idx], masks_sample[idx], num_crowds_sampler[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds, split_feature \
            = [[None for alloc in allocation] for _ in range(5)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images_sample[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets_sample[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks_sample[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds_sampler[cur_idx:cur_idx + alloc]
            split_feature[device_idx] = torch.stack(feature_list, dim=0)
            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds, split_feature


def no_inf_mean(x: torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()


def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds, feature_list = prepare_data(datum)
            out = net(images, feature_list)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args_yolact.validation_size <= iterations * args_yolact.batch_size:
                break

        for k in losses:
            losses[k] /= iterations

        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)


def compute_validation_map(epoch, iteration, yolact_net, dataset, softgroup_net, point_val, cfg_softgroup, log: Log = None):
    with torch.no_grad():
        yolact_net.eval()
        softgroup_net.eval()
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, softgroup_net, point_val, cfg_softgroup, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()
        softgroup_net.train()


def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images=' + str(args_yolact.validation_size)])

def train_softgroup(epoch, model, optimizer, scaler, train_loader, cfg_softgroup, logger, writer,
                    yolact_dataloader=None, yolact_dataset=None, yolact_val=None, point_val=None):

    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args_yolact.log:
        log = Log(cfg.name, args_yolact.log_folder, dict(args_yolact._get_kwargs()),
            overwrite=(args_yolact.resume is None), log_gpu_stats=args_yolact.log_gpu)

    timer.disable_all()

    if args_yolact.resume == 'interrupt':
        args_yolact.resume = SavePath.get_interrupt(args_yolact.save_folder)
    elif args_yolact.resume == 'latest':
        args_yolact.resume = SavePath.get_latest(args_yolact.save_folder, cfg.name)



    if args_yolact.resume is not None:
        print('Resuming training, loading {}...'.format(args_yolact.resume))
        yolact_net.load_weights(args_yolact.resume)

        if args_yolact.start_iter == -1:
            args_yolact.start_iter = SavePath.from_str(args_yolact.resume).iteration

    else:

        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args_yolact.save_folder + cfg.backbone.path)


    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    pg_conv0, pg_conv1, pg_conv2 = [], [], []
    for k, v in yolact_net.named_modules():
        #if "affinity" in k or "proto_net" in k: #or "fpn" in k or "prediction_layers" in k:
        #if "affinity" in k:
        #if "affinity" in k or "proto_net" in k or "fpn" in k or "prediction_layers" in k \
        #        or "backbone" in k:
        if "affinity_res" in k:

            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg_conv2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg_conv0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg_conv1.append(v.weight)  # apply decay
        else:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
    optimizer_yolact = optim.SGD(pg1, lr=args_yolact.lr, momentum=args_yolact.momentum, weight_decay=args_yolact.decay)
    optimizer_yolact.add_param_group({'params': pg2})
    optimizer_yolact.add_param_group({'params': pg0})

    optimizer_yolact.add_param_group({'params': pg_conv2, 'lr': args_yolact.lr * 100})
    optimizer_yolact.add_param_group({'params': pg_conv0, 'lr': args_yolact.lr * 100})
    optimizer_yolact.add_param_group({'params': pg_conv1, 'lr': args_yolact.lr * 100})
    del pg0, pg1, pg2, pg_conv0, pg_conv1, pg_conv2
    #optimizer_yolact = optim.SGD(net.parameters(), lr=args_yolact.lr, momentum=args_yolact.momentum,
    #    weight_decay=args_yolact.decay)

    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
        pos_threshold=cfg.positive_iou_threshold,
        neg_threshold=cfg.negative_iou_threshold,
        negpos_ratio=cfg.ohem_negpos_ratio)

    affinity_criterion = AffinityLoss()

    if args_yolact.batch_alloc is not None:
        args_yolact.batch_alloc = [int(x) for x in args_yolact.batch_alloc.split(',')]
        if sum(args_yolact.batch_alloc) != args_yolact.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args_yolact.batch_alloc, args_yolact.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion, affinity_criterion))
    if args_yolact.cuda:
        net = net.cuda()

    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)


    iteration = max(args_yolact.start_iter, 0)
    last_time = time.time()

    epoch_size = len(yolact_dataset) // args_yolact.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    step_index = 0

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args_yolact.save_folder)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}


    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg_softgroup.dist:
        train_loader.sampler.set_epoch(epoch)



    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        """
        for i, datum in enumerate(yolact_dataloader):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                batch = next(dataloader_iterator)
        """
        for i, (batch, datum) in enumerate(zip(train_loader, yolact_dataloader), start=0):


            #print("-----------------------------------")
            #print("start:{}".format(torch.cuda.memory_allocated(0)))
            data_time.update(time.time() - end)
            cosine_lr_after_step(optimizer, cfg_softgroup.optimizer.lr, epoch - 1, cfg_softgroup.step_epoch, cfg_softgroup.epochs)
            with torch.cuda.amp.autocast(enabled=cfg_softgroup.fp16):
                loss, log_vars, feature_list = model(batch, return_loss=True)
                #print(len(feature_list)) # len(feature_list)=1
                #print(feature_list[0].shape)
                #exit()
            #print("after_3D:{}".format(torch.cuda.memory_allocated(0)))
            changed = False
            for change in cfg.delayed_settings:
                if iteration >= change[0]:
                    changed = True
                    cfg.replace(change[1])

                    # Reset the loss averages because things might have changed
                    for avg in loss_avgs:
                        avg.reset()

            # If a config setting was changed, remove it from the list so we don't keep checking
            if changed:
                cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

            # Warm up by linearly interpolating the learning rate from some smaller value
            if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:

                lr_one = (args_yolact.lr - cfg.lr_warmup_init) * (
                            iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init
                set_lr(optimizer_yolact, lr_one)
                set_lr_affinity(optimizer_yolact, lr_one*100)

            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                step_index += 1
                lr_one = args_yolact.lr * (args_yolact.gamma ** step_index)
                set_lr(optimizer_yolact, lr_one)
                set_lr_affinity(optimizer_yolact, lr_one*100)

            # Zero the grad to get ready to compute gradients
            optimizer_yolact.zero_grad()

            # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
            # print("begin_forward-----------------------")
            #print("before_append:{}".format(torch.cuda.memory_allocated(0)))
            datum = list(datum)
            datum.append([F.interpolate(torch.mean(feature_list.dense(), dim=4), size=(69,69),mode='bilinear', align_corners=True)])
            losses, affinity_loss = net(datum)
            #print("after_2D:{}".format(torch.cuda.memory_allocated(0)))
            #print("after_yolact:{}".format(torch.cuda.memory_allocated(0)))
            losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel
            loss_yolact = sum([losses[k] for k in losses])
            loss_yolact = loss_yolact + affinity_loss
            # meter_dict
            for k, v in log_vars.items():
                if k not in meter_dict.keys():
                    meter_dict[k] = AverageMeter()
                meter_dict[k].update(v)
            #print("after_loss_sum:{}".format(torch.cuda.memory_allocated(0)))
            # backward
            #print("before_zero_grad:{}".format(torch.cuda.memory_allocated(0)))
            optimizer.zero_grad()

            scaler.scale(loss).backward() # retain_graph=True
            scaler.step(optimizer)
            scaler.update()
            #print("after_3D_optimizer:{}".format(torch.cuda.memory_allocated(0)))
            # print("after_zero_grad:{}".format(torch.cuda.memory_allocated(0)))
            loss_yolact.backward()  # Do this to free up vram even if loss is not finite
            if torch.isfinite(loss_yolact).item():
                optimizer_yolact.step()

            #print("after_2D_optimizer:{}".format(torch.cuda.memory_allocated(0)))
            #print("after_optimizer:{}".format(torch.cuda.memory_allocated(0)))
            # Add the loss to the moving average for bookkeeping
            for k in losses:
                loss_avgs[k].add(losses[k].item())
            #print("after_LOSS_FOR:{}".format(torch.cuda.memory_allocated(0)))
            cur_time = time.time()
            elapsed = cur_time - last_time
            last_time = cur_time

            # Exclude graph setup from the timing information
            if iteration != args_yolact.start_iter:
                time_avg.add(elapsed)

            if iteration % 10 == 0:
                eta_str = str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                      % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

            if args_yolact.log:
                precision = 5
                loss_info = {k: round(losses[k].item(), precision) for k in losses}
                loss_info['T'] = round(loss_yolact.item(), precision)

                if args_yolact.log_gpu:
                    log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                    lr=round(cur_lr, 10), elapsed=elapsed)

                log.log_gpu_stats = args_yolact.log_gpu

            iteration += 1

            if iteration % args_yolact.save_interval == 0 and iteration != args_yolact.start_iter:
                if args_yolact.keep_latest:
                    latest = SavePath.get_latest(args_yolact.save_folder, cfg.name)

                #print('Saving state, iter:', iteration)
                #yolact_net.save_weights(save_path(epoch, iteration))

                if args_yolact.keep_latest and latest is not None:
                    if args_yolact.keep_latest_interval <= 0 or iteration % args_yolact.keep_latest_interval != args_yolact.save_interval:
                        print('Deleting old save...')
                        os.remove(latest)

            # time and print
            remain_iter = len(train_loader) * (cfg_softgroup.epochs - epoch + 1) - i
            iter_time.update(time.time() - end)
            end = time.time()
            remain_time = remain_iter * iter_time.avg
            remain_time = str(datetime.timedelta(seconds=int(remain_time)))
            lr = optimizer.param_groups[0]['lr']

            if is_multiple(i, 10):
                log_str = f'Epoch [{epoch}/{cfg_softgroup.epochs}][{i}/{len(train_loader)}]  '
                log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                    f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
                for k, v in meter_dict.items():
                    log_str += f', {k}: {v.val:.4f}'
                logger.info(log_str)
            #print("final:{}".format(torch.cuda.memory_allocated(0)))
        # This is done per epoch
        if args_yolact.validation_epoch > 0:
            if epoch % args_yolact.validation_epoch == 0 and epoch > 0:
                compute_validation_map(epoch, iteration, yolact_net, yolact_val,
                    model, point_val, cfg_softgroup, log if args_yolact.log else None)

        writer.add_scalar('train/learning_rate', lr, epoch)
        for k, v in meter_dict.items():
            writer.add_scalar(f'train/{k}', v.avg, epoch)

        if epoch % args_yolact.validation_epoch == 0 and epoch > 0:
            checkpoint_save(epoch, model, optimizer, cfg_softgroup.work_dir, cfg_softgroup.save_freq)
            yolact_net.save_weights(save_path(epoch, iteration))
        writer.flush()

def validate_softgroup(epoch, model, val_loader, cfg, logger, writer):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_gpu(results, len(val_set))
    if is_main_process():
        for res in results:
            all_sem_preds.append(res['semantic_preds'])
            all_sem_labels.append(res['semantic_labels'])
            all_offset_preds.append(res['offset_preds'])
            all_offset_labels.append(res['offset_labels'])
            all_inst_labels.append(res['instance_labels'])
            if not cfg.model.semantic_only:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(val_set.CLASSES)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
        logger.info('Evaluate semantic segmentation and offset MAE')
        miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
                                  cfg.model.ignore_label, logger)
        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/Acc', acc, epoch)
        writer.add_scalar('val/Offset MAE', mae, epoch)

def main():
    torch.multiprocessing.set_start_method('spawn')

    args_softgroup = get_args()
    cfg_txt = open(args_softgroup.config, 'r').read()
    cfg_softgroup = Munch.fromDict(yaml.safe_load(cfg_txt))
    print(args_softgroup.dist)
    if args_softgroup.dist:
        init_dist()
    cfg_softgroup.dist = args_softgroup.dist


    if args_softgroup.work_dir:
        cfg_softgroup.work_dir = args_softgroup.work_dir
    else:
       cfg_softgroup.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args_softgroup.config))[0])
       # cfg_softgroup.work_dir = osp.join('./work_dirs/', osp.splitext(osp.basename(args_softgroup.config))[0])


    os.makedirs(osp.abspath(cfg_softgroup.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())


    log_file = osp.join(cfg_softgroup.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args_softgroup.dist}')
    logger.info(f'Mix precision training: {cfg_softgroup.fp16}')
    shutil.copy(args_softgroup.config, osp.join(cfg_softgroup.work_dir, osp.basename(args_softgroup.config)))
    writer = SummaryWriter(cfg_softgroup.work_dir)


    model_softgroup = SoftGroup(**cfg_softgroup.model).cuda()


    if args_softgroup.dist:
        model_softgroup = DistributedDataParallel(model_softgroup, device_ids=[torch.cuda.current_device()])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg_softgroup.fp16)

    # data

    train_set_softgroup = build_dataset(cfg_softgroup.data.train, logger)
    val_set_softgroup = build_dataset(cfg_softgroup.data.test, logger)
    train_loader_softgroup = build_dataloader(train_set_softgroup, training=True, dist=args_softgroup.dist, **cfg_softgroup.dataloader.train)
    val_loader_softgroup = build_dataloader(val_set_softgroup, training=False, dist=args_softgroup.dist, **cfg_softgroup.dataloader.test)

    # optim

    optimizer_softgroup = build_optimizer(model_softgroup, cfg_softgroup.optimizer)
    # pretrain, resume
    start_epoch = 1
    if args_softgroup.resume:
        logger.info(f'Resume from {args_softgroup.resume}')
        start_epoch = load_checkpoint(args_softgroup.resume, logger, model_softgroup, optimizer=optimizer_softgroup)
    elif cfg_softgroup.pretrain:
        print("using pretrain")
        logger.info(f'Load pretrain from {cfg_softgroup.pretrain}')
        load_checkpoint(cfg_softgroup.pretrain, logger, model_softgroup)

    if not os.path.exists(args_yolact.save_folder):
        os.mkdir(args_yolact.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
        info_file=cfg.dataset.train_info,
        transform=SSDAugmentation(MEANS))

    if args_yolact.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
            info_file=cfg.dataset.valid_info,
            transform=BaseTransform(MEANS))

    data_loader = data.DataLoader(dataset, args_yolact.batch_size,
        num_workers=args_yolact.num_workers,
        shuffle=False, collate_fn=detection_collate,
        pin_memory=False)  # shuffle=True

    # train and val
    logger.info('Training')
    epoch = 0
    train_softgroup(epoch, model_softgroup, optimizer_softgroup, scaler, train_loader_softgroup, cfg_softgroup, logger, writer,
                    yolact_dataloader=data_loader, yolact_dataset=dataset, yolact_val=val_dataset,
                    point_val=val_loader_softgroup)
def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args_yolact.validation_size)])

if __name__ == '__main__':
    main()






