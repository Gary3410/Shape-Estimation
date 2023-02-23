from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

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

import matplotlib.pyplot as plt
import cv2

trained_model = "/home/zhenyu/yolact-master/weights/yolact_resnet50_1666_30000.pth"
config_name = "yolact_resnet50_config_my"
set_cfg(config_name)
with torch.no_grad():
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = None
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    frame = torch.from_numpy(cv2.imread("/home/zhenyu/yolact-master/data/coco/images/9.png")).cuda().float()
    net = net.cuda()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    # 获得image的大小
    h, w, _ = frame.shape
    t = postprocess(preds, w, h, visualize_lincomb=False,
                    crop_masks=True,
                    score_threshold=0.35)
    print("class", t[0].shape)
    print("scores", t[1].shape)
    print("boxes", t[2].shape)
    print("maskes", t[3].shape)
    for i in range(t[3].shape[0]):
        print(np.unique(t[3][i, :, :].cpu()))
    for i in range(t[4].shape[0]):
        print(t[4][i, :2, :3])
    print("entropy", t[4].shape)
