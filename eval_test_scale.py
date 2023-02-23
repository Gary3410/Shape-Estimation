from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
import torch.nn.functional as F
from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from Net_deploy import load_models, FS_Net_Test, FS_Net_Test_obj_size, load_models_my
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import trimesh
import _pickle as cPickle
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
# -----------------------------------------
# 上述是2D的文件导入

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

from my_utils import get_point_view, get_instance
from tqdm import tqdm
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
        default='/home/potato/workplace/SoftGroup/weights_2/yolact_resnet50_120_10527.pth', type=str,
        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=True, dest='no_sort', action='store_true',
        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
        benchmark=False, no_sort=True, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
        emulate_playback=False)

    global args_yolact
    args_yolact = parser.parse_args(argv)

    if args_yolact.output_web_json:
        args_yolact.output_coco_json = True

    if args_yolact.seed is not None:
        random.seed(args_yolact.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
#iou_thresholds = [0.5]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args_yolact.display_lincomb,
            crop_masks=args_yolact.crop,
            score_threshold=args_yolact.score_threshold)
        # 直接完成解码，只有一句话
        print("class", t[0].shape)
        print("scores", t[1].shape)
        print("boxes", t[2].shape)
        print("maskes", t[3].shape)
        print("entropy", t[4].shape)
        for i in range(t[3].shape[0]):
            print(np.unique(t[3][i, :, :].cpu()))
        exit()
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args_yolact.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args_yolact.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args_yolact.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args_yolact.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in
                            range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args_yolact.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args_yolact.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args_yolact.display_text or args_yolact.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args_yolact.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args_yolact.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args_yolact.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_numpy


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args_yolact.crop, score_threshold=args_yolact.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args_yolact.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()

    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]


def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, args_yolact.bbox_det_file),
            (self.mask_data, args_yolact.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                       'use_yolo_regressors', 'use_prediction_matching',
                       'train_masks']

        output = {
            'info': {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args_yolact.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections: Detections = None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args_yolact.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args_yolact.crop, score_threshold=args_yolact.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if args_yolact.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def evalimage(net: Yolact, path: str, save_path: str = None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


from multiprocessing.pool import ThreadPool
from queue import Queue


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalvideo(net: Yolact, path: str, out_path: str = None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()

    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True

    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args_yolact.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args_yolact.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        cv2.imshow(path, frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args_yolact.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args_yolact.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe

                if out_path is None or args_yolact.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()


    extract_frame = lambda x, i: (
    x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args_yolact.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)

                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args_yolact.video_multiframe / frame_times.get_avg()
            else:
                fps = 0

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
            fps, video_fps, frame_buffer.qsize())
            if not args_yolact.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')

    cleanup_and_exit()
def get_iou(box_a, box_b, eps=1e-10):
    """Computes IoU of two axis aligned bboxes.

    Args:
        box_a, box_b: xyzxyz
    Returns:
        iou
    """

    max_a = box_a[3:]
    max_b = box_b[3:]
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3]
    min_b = box_b[0:3]
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = (box_a[3:6] - box_a[:3]).prod()
    vol_b = (box_b[3:6] - box_b[:3]).prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union

def points2aabb(points):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    aabb_min = pcd.get_min_bound()
    aabb_max = pcd.get_max_bound()
    aabb = np.concatenate([aabb_min, aabb_max], axis=0)

    return aabb

def loss_recon(a, b):
    if torch.cuda.is_available():
        # chamferdist = ChamferDistance()
        chamferdist = ChamferDistance()
        dist1, dist2 = chamferdist(a, b)
        #dist1, dist2, idx1, idx2 = chamLoss(a, b)
        loss = torch.mean(dist1) + torch.mean(dist2)
    else:
        loss = torch.Tensor([100.0])
    return loss

def mesh2points(mesh, scale_one=None, tmp=False):
    if tmp:
        #points, _ = trimesh.sample.sample_surface_even(mesh, 2000)
        points = mesh.vertices
        ptsn = points.copy() * 1000
        max_point = np.max(points, axis=0) * 1000
        min_point = np.min(points, axis=0) * 1000
        old_scale = np.asarray(max_point - min_point).reshape([1, -1])
        # print(scale_one.shape)
        #scale_one = np.asarray([0, 0, 0])
        #print(scale_one)
        #print(old_scale[0])
        new_scale = old_scale[0] + scale_one
        #print("tmp", new_scale)

        ex, ey, ez = new_scale / old_scale[0]
        ptsn[:, 0] = ptsn[:, 0] * ex
        ptsn[:, 1] = ptsn[:, 1] * ey
        ptsn[:, 2] = ptsn[:, 2] * ez

        return ptsn
    else:
        #points, _ = trimesh.sample.sample_surface_even(mesh, 2000)
        points = mesh.vertices
        ptsn = points.copy() * 1000
        max_point = np.max(points, axis=0) * 1000
        min_point = np.min(points, axis=0) * 1000
        old_scale = np.asarray(max_point - min_point).reshape([1, -1])
        #print(scale_one.shape)
        #print("pred", scale_one[0])
        #exit()
        ex, ey, ez = scale_one[0] / old_scale[0]
        ptsn[:, 0] = ptsn[:, 0] * ex
        ptsn[:, 1] = ptsn[:, 1] * ey
        ptsn[:, 2] = ptsn[:, 2] * ez

        return ptsn

def get_3D_mAP(obj_pts_list_all, obj_class_all, position_label, pred_mesh_list, obj_size_list, mesh_base_path, mesh_tmp_scale):

    # 先获取label的各个点云块
    label_name = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]

    pts_label_all = []
    pts_label_class_all = []
    label_mesh_all = []
    number_one = np.unique(position_label[:, 4]).shape[0]
    for i in range(1, number_one+1):
        pts_one = position_label[position_label[:, 4] == i][:, :4]
        if pts_one.shape[0] <= 100:
            continue
        cls = int(pts_one[0, 3].item())
        pts_label_all.append(pts_one[:, :3])
        pts_label_class_all.append(cls)
        cate = label_name[cls-1]
        mesh = trimesh.load_mesh(mesh_base_path + '/%s/%s.ply' % (cate, cate))
        label_mesh_all.append(mesh)

    # 计算每个点云块的bbox
    obj_pts_bbox_list = []
    for i in range(len(obj_pts_list_all)):

        pcd = o3d.geometry.PointCloud()
        pts_one = obj_pts_list_all[i][:, :3]
        pts_one = pts_one[pts_one[:, 2] >= 0.002]
        pcd.points = o3d.utility.Vector3dVector(pts_one)
        aabb_min = pcd.get_min_bound()
        aabb_max = pcd.get_max_bound()
        aabb = np.concatenate([aabb_min, aabb_max], axis=0)
        obj_pts_bbox_list.append(np.asarray(aabb))

    obj_label_bbox_list = []
    for i in range(len(pts_label_all)):
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(pts_label_all[i])
        aabb_min = pcd.get_min_bound()
        aabb_max = pcd.get_max_bound()
        aabb = np.concatenate([aabb_min, aabb_max], axis=0)
        obj_label_bbox_list.append(aabb)

    # 生成混淆矩阵
    iou_mat = np.zeros([len(obj_pts_list_all), len(pts_label_all)])

    for i in range(len(obj_pts_list_all)):
        for j in range(len(pts_label_all)):
            iou_mat[i][j] = get_iou(obj_pts_bbox_list[i], obj_label_bbox_list[j])


    # 计算模板的混淆矩阵
    iou_mat_mesh = np.zeros([len(obj_pts_list_all), len(pts_label_all)])
    #cd_dis_mat = np.zeros([len(obj_pts_list_all), len(pts_label_all)])
    # 计算各个预测点云块与模板与预测模板的iou
    for i in range(len(obj_pts_list_all)):
        for j in range(len(pts_label_all)):
            pred_mesh_one = pred_mesh_list[i]

            scale_one = obj_size_list[i]
            if j >= mesh_tmp_scale.shape[0]:
                j = mesh_tmp_scale.shape[0] - 1
            scale_tmp_one = mesh_tmp_scale[j]  #
            label_mesh_one = label_mesh_all[j]
            mesh_tmp_points = mesh2points(label_mesh_one, scale_tmp_one, tmp=True)
            new_mesh_points = mesh2points(pred_mesh_one, scale_one)
            cd_loss = loss_recon(mesh_tmp_points, new_mesh_points)
            #cd_dis_mat[i][j] = cd_loss.cpu().numpy().item()
            tmp_bbox = points2aabb(mesh_tmp_points)
            new_bbox = points2aabb(new_mesh_points)
            iou_mat_mesh[i][j] = get_iou(tmp_bbox, new_bbox)
    # 设定阈值, 统计mAP
    #print(iou_mat)
    iou_thresholds_3D = [x / 100 for x in range(10, 60, 5)]
    iou_thresholds_3D.append(0.25)
    mAP = []
    mAR = []

    for iou_one in iou_thresholds_3D:
        result_one = np.argmax(iou_mat, axis=1)
        # result_one的结果就是预测的类别
        tp = 0

        for i in range(result_one.shape[0]):
            if iou_mat[i][result_one[i]] >= 0.5:
                if obj_class_all[i] == pts_label_class_all[result_one[i]]:
                    if iou_mat_mesh[i][result_one[i]] >= iou_one:
                        tp = tp + 1

        ap = tp / len(obj_pts_list_all)

        ar = tp / number_one
        mAP.append(ap)
        mAR.append(ar)

    #print(mAP)
    #exit()
    return mAP, mAR


def evaluate(net:Yolact, dataset, model=None, point_val=None, cfg_softgroup=None, train_mode=False):
    net.detect.use_fast_nms = args_yolact.fast_nms
    net.detect.use_cross_class_nms = args_yolact.cross_class_nms

    cfg.mask_proto_debug = args_yolact.mask_proto_debug
    if args_yolact.image is not None:
        if ':' in args_yolact.image:
            inp, out = args_yolact.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args_yolact.image)
        return

    elif args_yolact.images is not None:
        inp, out = args_yolact.images.split(':')
        evalimages(net, inp, out)
        return

    elif args_yolact.video is not None:
        if ':' in args_yolact.video:
            inp, out = args_yolact.video.split(':')
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args_yolact.video)
        return

    frame_times = MovingAverage()
    #dataset_size = len(dataset) if args_yolact.max_images < 0 else min(args_yolact.max_images, len(dataset))
    dataset_size = len(dataset)
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args_yolact.display and not args_yolact.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))
    print(len(dataset))
    """
    # shuffle与no_sort都不启动
    if args_yolact.shuffle:
        random.shuffle(dataset_indices)
    elif not args_yolact.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])
    """
    dataset_indices = dataset_indices[:dataset_size]

    # 处理3D数据集
    classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0 = load_models_my('banana', model_cat=[
        "banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"])
    label_name = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
    mean_scale_list = np.loadtxt("/home/potato/workplace/obj_model/mean_scale_cls.txt")

    dataloader_iterator = iter(point_val)
    try:
        # Main eval loop
        # 相当于一张一张的经过, 对我非常有利
        segment_heat_map_list = []
        position_list = []
        mask_list = []
        mask_tp_list = []
        class_tp_list = []
        view_result_list = []
        class_list = []
        proj_factor_list = []
        score_list = []
        result_list = []
        position_label_list = []
        AP50 = []
        AP25 = []
        mAP_list = []
        ar75 = []
        ar50 = []
        ar25 = []
        ap75 = []
        ap50 = []
        ap25 = []
        map = []
        mar = []
        viewMat_list = [
            [1.0, 0.0, -0.0, 0.0, -0.0, 1.0, -0.00017452123574912548, 0.0, 0.0, 0.00017452123574912548, 1.0, 0.0, 0.5,
             -7.417152664856985e-05, -0.4266100227832794, 1.0],
            [1.0, 0.0, -0.0, 0.0, -0.0, 0.8660240173339844, -0.5000023245811462, 0.0, 0.0, 0.5000023245811462,
             0.8660240173339844, 0.0, 0.5, 2.925097942352295e-05, -0.426557332277298, 1.0],
            [0.017451846972107887, 0.8658873438835144, -0.49993473291397095, 0.0, -0.9998477697372437,
             0.015113634057343006,
             -0.00872611254453659, 0.0, 0.0, 0.5000109076499939, 0.8660191893577576, 0.0, 0.012434440664947033,
             0.4329407513141632, -0.6765085458755493, 1.0],
            [-0.999847412109375, 0.015133202075958252, -0.008737090043723583, 0.0, -0.017474284395575523,
             -0.8658948540687561, 0.49992072582244873, 0.0, 0.0, 0.4999971091747284, 0.8660269975662231, 0.0,
             -0.4962104260921478, 0.007570326328277588, -0.4309096336364746, 1.0],
            [-0.01745261251926422, -0.8658953309059143, 0.4999208450317383, 0.0, 0.9998477697372437,
             -0.015114436857402325,
             0.008726253174245358, 0.0, 0.0, 0.49999701976776123, 0.8660271763801575, 0.0, -0.005017626099288464,
             -0.43294382095336914, -0.1765807718038559, 1.0]]
        projMat = [0.74999994, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.00002003, -1.0, 0.0, 0.0, -0.0200002,
                   0.0]
        data_base_path = "/home/potato/workplace/dataset"
        for it, image_idx in tqdm(enumerate(dataset_indices)):
            timer.reset()
            if image_idx % 5 == 0 and image_idx > 0:
                segment_heat_map_list = []
                position_list = []
                mask_list = []
                mask_tp_list = []
                class_tp_list = []
                view_result_list = []
                class_list = []
                proj_factor_list = []
                score_list = []
                result_list = []
                position_label_list = []
            if (it) % 5 == 0:

                batch_softgroup = next(dataloader_iterator)

            feature_list = model(batch_softgroup, only_feature=True)
            feature_list = F.interpolate(torch.mean(feature_list.dense(), dim=4), size=(69,69),mode='bilinear', align_corners=True)
            # 开始挨个读取图片
            # 数据处理工作
            if image_idx % 5 != 0 or image_idx == 0 or image_idx % 5 == 0:
                view_id = image_idx % 5
                viewMat = viewMat_list[view_id]
                with timer.env('Load Data'):
                    img, gt, gt_masks, h, w, num_crowd, path = dataset.pull_item(image_idx)
                    img_name = os.path.basename(path)[:-4]

                    # 获取模板信息
                    sense_id = img_name.split("_")[0]

                    depth_path = os.path.join(data_base_path, "depth", img_name + ".npy")
                    depthImg = np.load(depth_path)
                    ints_img_path = os.path.join(data_base_path, "ints_img", img_name + ".png")
                    label_img_path = os.path.join(data_base_path, "label_img", img_name + ".png")
                    ints_img = Image.open(ints_img_path)
                    label_img = Image.open(label_img_path)
                    ints_img = np.asarray(ints_img).reshape([-1, 1])
                    label_img = np.asarray(label_img).reshape([-1, 1])
                    # Test flag, do not upvote
                    if cfg.mask_proto_debug:
                        with open('scripts/info.txt', 'w') as f:
                            f.write(str(dataset.ids[image_idx]))
                        np.save('scripts/gt.npy', gt_masks)

                    batch = Variable(img.unsqueeze(0))
                    if args_yolact.cuda:
                        batch = batch.cuda()
                        device = batch.device
                with timer.env('Network Extra'):
                    preds = net(batch, feature_list=feature_list)
                with timer.env('Postprocess'):
                    classes, scores, boxes, masks = postprocess(preds, w, h, crop_masks=True, score_threshold=0.35)
                position, proj_factor = get_point_view(depthImg, viewMat, projMat)
                position_label = np.concatenate((position[:, :3], label_img, ints_img), axis=1)
                position_list.append(position)

                # 我们直接去掉背景点
                position_label = position_label[position_label[:, 3] > 0]
                position_label_list.append(position_label)

                mask_list.append(masks.cpu().numpy())
                class_list.append(classes.cpu().numpy())
                score_list.append(scores.cpu().numpy())
                if (image_idx + 1) % 5 != 0:
                    continue
            # 处理预测之后的结果
            # 整个场景的点云label, obj_pts_list_all 场景中每个ints的点云块

            data_base_path = "/home/potato/workplace/dataset/val_data_random_nodef"
            mesh_base_path = "/home/potato/workplace/dataset/data"
            label_path = os.path.join(data_base_path, "label", sense_id + ".pkl")
            label = cPickle.load(open(label_path, 'rb'))
            tmp_scale = label.get('scales')

            position_label = np.concatenate(position_label_list, axis=0)
            obj_pts_list_all, obj_class_all = get_instance(mask_list, position_list, class_list, device)
            obj_class_all = np.asarray(obj_class_all)

            # 分别处理各个点云块
            pred_mesh_list = []
            obj_size_list = []
            for ptd in range(len(obj_pts_list_all)):
                pts = obj_pts_list_all[ptd]
                cls_ptd = obj_class_all[ptd]
                cate = label_name[cls_ptd]
                mesh = trimesh.load_mesh(mesh_base_path + '/%s/%s.ply' % (cate, cate))
                pred_mesh_list.append(mesh)
                vpc = mesh.vertices
                pc = np.asarray(vpc).copy()
                pc = pc * 1000.0

                choice = np.random.choice(pts.shape[0], 2000, replace=True)
                pts_scale = pts[choice, :].copy()
                pts_scale = pts_scale * 1000
                model_size = mean_scale_list[cls_ptd]
                obj_size = FS_Net_Test_obj_size(pts_scale, pc, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red,
                    cate, model_size,
                    cate_id0,
                    num_cor=3, pts_rec=None)
                obj_size_list.append(obj_size)

            obj_class_all = obj_class_all + 1
            ap, ar = get_3D_mAP(obj_pts_list_all, obj_class_all, position_label, pred_mesh_list, obj_size_list, tmp_scale)
            mAP = np.asarray(ap[:-1])
            map.append(np.mean(mAP))
            mAR = np.asarray(np.mean(ar[:-1]))
            mar.append(np.mean(mAR))
            ap25.append(ap[-1])
            ap50.append(ap[0])
            ap75.append(ap[8])
            ar25.append(ar[-1])
            ar50.append(ar[0])
            ar75.append(ar[8])

        print("mAP", np.mean(map))
        print("AP25", np.mean(ap25))
        print("AP10", np.mean(ap50))
        print("AP50", np.mean(ap75))
        print("mAR", np.mean(mar))
        print("AR25", np.mean(ar25))
        print("AR10", np.mean(ar50))
        print("AR50", np.mean(ar75))


    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--config', type=str, default="/home/potato/workplace/SoftGroup/configs/softgroup_s3dis_backbone_fold5.yaml", help='path to config file')
    parser.add_argument('--checkpoint', type=str, default="/home/potato/workplace/SoftGroup/work_dirs_2/softgroup_s3dis_backbone_fold5/epoch_120.pth", help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', default=False, help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parse_args()

    # 主要是把模型与数据启动
    args_softgroup = get_args()
    cfg_txt = open(args_softgroup.config, 'r').read()
    cfg_softgroup = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args_softgroup.dist:
        init_dist()
    logger = get_root_logger()
    model = SoftGroup(**cfg_softgroup.model).cuda()
    if args_softgroup.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args_softgroup.checkpoint}')
    load_checkpoint(args_softgroup.checkpoint, logger, model)

    model.eval()
    # 传入的数据集是正确的
    dataset_softgroup = build_dataset(cfg_softgroup.data.test, logger)

    dataloader_softgroup = build_dataloader(dataset_softgroup, training=False, dist=args_softgroup.dist, **cfg_softgroup.dataloader.test)
    results = []

    if args_yolact.config is not None:
        set_cfg(args_yolact.config)

    if args_yolact.trained_model == 'interrupt':
        args_yolact.trained_model = SavePath.get_interrupt('weights/')
    elif args_yolact.trained_model == 'latest':
        args_yolact.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args_yolact.config is None:
        model_path = SavePath.from_str(args_yolact.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args_yolact.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args_yolact.config)
        print(args_yolact.config)
        set_cfg(args_yolact.config)

    if args_yolact.detect:
        cfg.eval_mask_branch = False

    if args_yolact.dataset is not None:
        set_dataset(args_yolact.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args_yolact.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args_yolact.resume and not args_yolact.display:
            with open(args_yolact.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args_yolact.image is None and args_yolact.video is None and args_yolact.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args_yolact.trained_model)
        net.eval()
        print(' Done.')

        if args_yolact.cuda:
            net = net.cuda()

        evaluate(net, dataset, model=model, point_val=dataloader_softgroup)