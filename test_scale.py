import argparse
import multiprocessing as mp
import os
import os.path as osp
from functools import partial

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_gpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import matplotlib.pyplot as plt
from color_map import deepglobe_color_map_27
from Net_deploy import load_models, FS_Net_Test, FS_Net_Test_obj_size, load_models_my
import open3d as o3d
import trimesh
import _pickle as cPickle
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--config', type=str, default="/home/potato/workplace/SoftGroup/configs/softgroup_s3dis_fold5.yaml",help='path to config file')
    parser.add_argument('--checkpoint', type=str, default="/home/potato/workplace/SoftGroup/work_dirs/softgroup_s3dis_fold5/epoch_20.pth", help='path to checkpoint')
    parser.add_argument('--dist', action='store_true',default=False,  help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args



def get_scale():

    return 0


def change_GT():
    return 0


def points2aabb(points):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    aabb_min = pcd.get_min_bound()
    aabb_max = pcd.get_max_bound()
    aabb = np.concatenate([aabb_min, aabb_max], axis=0)

    return aabb

def mesh2points(mesh, scale_one=None, tmp=False):
    if tmp:
        #points, _ = trimesh.sample.sample_surface_even(mesh, 500)

        points = mesh.vertices
        points_prt = points.copy()
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
        #points, _ = trimesh.sample.sample_surface_even(mesh, 500)

        points = mesh.vertices
        points_prt = points.copy()
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


def get_iou(box_a, box_b, eps=1e-10):
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


def get_3D_mAP(ptsn_list, pts_cls_list, obj_size_list, scan_id, position_label, mesh_list, tmp_scale):


    pts_label_all = []
    pts_label_class_all = []
    number_one = np.unique(position_label[:, 4]).shape[0]
    for i in np.unique(position_label[:, 4]):
        pts_one = position_label[position_label[:, 4] == i][:, :4]
        if pts_one.shape[0] <= 10:
            continue
        cls = int(pts_one[0, 3].item())
        pts_label_all.append(pts_one[:, :3])
        pts_label_class_all.append(cls)


    obj_pts_bbox_list = []
    for i in range(len(ptsn_list)):
        pcd = o3d.geometry.PointCloud()
        pts_one = ptsn_list[i][:, :3]
        #pts_one = pts_one[pts_one[:, 2] >= 0.002]
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

    iou_mat = np.zeros([len(ptsn_list), len(pts_label_all)])
    for i in range(len(ptsn_list)):
        for j in range(len(pts_label_all)):
            iou_mat[i][j] = get_iou(obj_pts_bbox_list[i], obj_label_bbox_list[j])

    iou_mat_mesh = np.zeros([len(ptsn_list), len(pts_label_all)])
    cd_dis_mat = np.zeros([len(ptsn_list), len(pts_label_all)])

    for i in range(len(ptsn_list)):
        for j in range(len(pts_label_all)):
            mesh_one = mesh_list[i]
            scale_one = obj_size_list[i]
            if j >= tmp_scale.shape[0]:
                j = tmp_scale.shape[0] - 1
            scale_tmp_one = tmp_scale[j] #
            mesh_tmp_points = mesh2points(mesh_one, scale_tmp_one, tmp=True)
            new_mesh_points = mesh2points(mesh_one, scale_one)
            a1 = mesh_tmp_points.copy()
            a2 = new_mesh_points.copy()
            a1 = torch.from_numpy(a1.astype(np.float32)).unsqueeze(0).cuda()
            a2 = torch.from_numpy(a2.astype(np.float32)).unsqueeze(0).cuda()
            loss = loss_recon(a2, a1)
            cd_dis_mat[i][j] = loss.cpu().numpy().item()
            tmp_bbox = points2aabb(mesh_tmp_points)
            new_bbox = points2aabb(new_mesh_points)
            iou_mat_mesh[i][j] = get_iou(tmp_bbox, new_bbox)

    iou_thresholds_3D = [x / 100 for x in range(10, 60, 5)]
    iou_thresholds_3D.append(0.25)
    mAP = []
    mAR = []
    loss_list = []
    for iou_one in iou_thresholds_3D:
        result_one = np.argmax(iou_mat, axis=1)
        # result_one的结果就是预测的类别
        tp = 0
        loss = 0
        for i in range(result_one.shape[0]):
            if iou_mat[i][result_one[i]] >= 0.5:
                if pts_cls_list[i] == pts_label_class_all[result_one[i]]:
                    if iou_mat_mesh[i][result_one[i]] >= iou_one+0.1:
                        tp = tp + 1
                        #loss = loss + cd_dis_mat[i][result_one[i]]
        ap = tp / len(ptsn_list)
        ar = tp / number_one
        mAP.append(ap)
        mAR.append(ar)
        loss_list.append(np.mean(cd_dis_mat)/(tp + number_one))

    return mAP, mAR, loss_list


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()

def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def save_single_instance(root, scan_id, insts):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt='%d')
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


def main():


    classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0 = load_models_my('banana', model_cat=["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"])
    label_name = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    print(args.checkpoint)
    load_checkpoint(args.checkpoint, logger, model)


    dataset = build_dataset(cfg.data.test, logger)


    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)
    results = []
    scan_ids, coords, sem_preds, sem_labels, offset_preds, offset_labels = [], [], [], [], [], []
    inst_labels, pred_insts, gt_insts = [], [], []
    _, world_size = get_dist_info()
    color_map = deepglobe_color_map_27()
    color_map = color_map[6:, :]
    color_map = color_map / 255.0
    # scan_id :'Area_5_office_11 其实是一个路径
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    with torch.no_grad():
        model.eval()
        mean_scale_list = np.loadtxt("/home/potato/workplace/obj_model/mean_scale_cls.txt")
        map = []
        mar = []
        ar75 = []
        ar50 = []
        ar25 = []
        ap75 = []
        ap50 = []
        ap25 = []
        cd_loss = []
        for i, batch in enumerate(dataloader):
            result = model(batch)
            points = result.get('coords_float')
            points_ints = result['pred_instances']
            scan_id = result['scan_id']


            points_label = result['instance_labels']
            seg_label = result.get('semantic_labels')
            cls_ints = np.concatenate((seg_label.reshape([-1, 1]), points_label.reshape([-1, 1])), axis=1)
            pts_label_all = np.concatenate((points, cls_ints), axis=1)


            sense_id = int(scan_id.split("_")[-1])
            data_base_path = "/home/potato/workplace/dataset/val_data_hard"
            mesh_base_path = "/home/potato/workplace/dataset/data"
            label_path = osp.join(data_base_path, "label", str(sense_id-1)+".pkl")
            label = cPickle.load(open(label_path, 'rb'))
            # 根据预测结果,得到该场景下的各个ints
            tmp_scale = label.get('scales')
            ptsn_list = []
            obj_size_list = []
            pts_cls_list = []
            mesh_list = []
            for obj_id in range(len(points_ints)):
                points_dict = points_ints[obj_id]
                points_mask = points_dict.get('pred_mask')
                points_mask = rle_decode(points_mask)
                pts = np.concatenate((points, points_mask.reshape([-1, 1])), axis=1)

                # pts便是单个的ints
                pts = pts[pts[:, 3] == 1][:, :3]
                cate = label_name[points_dict.get('label_id') - 1]
                mesh = trimesh.load_mesh(mesh_base_path + '/%s/%s.ply' % (cate, cate))
                mesh_list.append(mesh)
                vpc = mesh.vertices
                pc = np.asarray(vpc).copy()
                pc = pc * 1000.0

                choice = np.random.choice(pts.shape[0], 2000, replace=True)
                pts_scale = pts[choice, :].copy()
                pts_scale = pts_scale * 1000
                model_size = mean_scale_list[points_dict.get('label_id')-1]
                obj_size = FS_Net_Test_obj_size(pts_scale, pc, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red,
                    cate, model_size,
                    cate_id0,
                    num_cor=3, pts_rec=None)
                obj_size_list.append(obj_size)
                ptsn_list.append(pts)
                pts_cls_list.append(points_dict.get('label_id') - 1)

            ap, ar, loss = get_3D_mAP(ptsn_list, pts_cls_list, obj_size_list, scan_id, pts_label_all, mesh_list, tmp_scale)
            mAP = np.asarray(ap[:-2])
            map.append(np.mean(mAP))
            cd_loss.append(np.mean(loss))
            mAR = np.asarray(np.mean(ar[:-2]))
            mar.append(np.mean(mAR))
            ap25.append(ap[-1])
            ap50.append(ap[0])
            ap75.append(ap[8])
            ar25.append(ar[-1])
            ar50.append(ar[0])
            ar75.append(ar[8])
            results.append(result)
            progress_bar.update(world_size)
        print("mAP", np.mean(map))
        print("AP25", np.mean(ap25))
        print("AP10", np.mean(ap50))
        print("AP50", np.mean(ap75))
        print("mAR", np.mean(mar))
        print("AR25", np.mean(ar25))
        print("AR10", np.mean(ar50))
        print("AR50", np.mean(ar75))
        print("cd", np.mean(cd_loss))
        progress_bar.close()
        results = collect_results_gpu(results, len(dataset))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            coords.append(res['coords_float'])
            sem_preds.append(res['semantic_preds'])
            sem_labels.append(res['semantic_labels'])
            offset_preds.append(res['offset_preds'])
            offset_labels.append(res['offset_labels'])
            inst_labels.append(res['instance_labels'])
            if not cfg.model.semantic_only:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            print(scan_ids)
            print(len(scan_ids))
            scannet_eval = ScanNetEval(dataset.CLASSES)

            scannet_eval.evaluate(pred_insts, gt_insts)
        logger.info('Evaluate semantic segmentation and offset MAE')
        ignore_label = cfg.model.ignore_label
        evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
        evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
        evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)

        # save output
        if not args.out:
            return
        logger.info('Save results')
        save_npy(args.out, 'coords', scan_ids, coords)
        if cfg.save_cfg.semantic:
            save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
            save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
        if cfg.save_cfg.offset:
            save_npy(args.out, 'offset_pred', scan_ids, offset_preds)
            save_npy(args.out, 'offset_label', scan_ids, offset_labels)
        if cfg.save_cfg.instance:
            save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts)
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts)


if __name__ == '__main__':
    main()
