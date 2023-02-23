# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm


from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable

import torch
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_train_cate
import torch.nn as nn
import numpy as np
import time
from uti_tool import data_augment
from yolov3_fsnet.detect_fsnet_train import det

from pyTorchChamferDistance.chamfer_distance import ChamferDistance
#import chamfer3D.dist_chamfer_3D

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models', help='output folder')
parser.add_argument('--outclass', type=int, default=2, help='point class')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()

kc = opt.outclass
num_cor = 3
num_vec = 8
nw = 0  # number of cpu
localtime = (time.localtime(time.time()))
year = localtime.tm_year
month = localtime.tm_mon
day = localtime.tm_mday
hour = localtime.tm_hour

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

cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]

for cat in [["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]]:
    print(cat)
    classifier_seg3D = GCN3D_segR(class_num=2, vec_num=1, support_num=7, neighbor_num=10)
    classifier_ce = Point_center_res_cate()  ## translation estimation
    classifier_Rot_red = Rot_red(F=1296, k=6)  ## rotation red
    classifier_Rot_green = Rot_green(F=1296, k=6)  ### rotation green

    num_classes = opt.outclass

    Loss_seg3D = nn.CrossEntropyLoss()
    Loss_func_ce = nn.MSELoss()
    Loss_func_Rot1 = nn.MSELoss()
    Loss_func_Rot2 = nn.MSELoss()
    Loss_func_s = nn.MSELoss()

    classifier_seg3D = nn.DataParallel(classifier_seg3D)
    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_Rot_red = nn.DataParallel(classifier_Rot_red)
    classifier_Rot_green = nn.DataParallel(classifier_Rot_green)

    classifier_seg3D = classifier_seg3D.train()
    classifier_ce = classifier_ce.train()
    classifier_Rot_red = classifier_Rot_red.train()
    classifier_Rot_green = classifier_Rot_green.train()

    Loss_seg3D.cuda()
    Loss_func_ce.cuda()
    Loss_func_Rot1.cuda()
    Loss_func_Rot2.cuda()
    Loss_func_s.cuda()

    classifier_seg3D.cuda()
    classifier_ce.cuda()
    classifier_Rot_red.cuda()
    classifier_Rot_green.cuda()

    #chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    if len(cats) > 1:
        opt.outf = 'models/FS_Net_demo'
    else:
        opt.outf = 'models/FS_Net_%s' % (cat[0])
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    cat1 = 'mix_all'          # weight file name

    sepoch = 0

    batch_size = 10  # bathcsize

    lr = 0.001

    epochs = 100

    optimizer = optim.Adam([{'params': classifier_seg3D.parameters()}, {'params': classifier_ce.parameters()},
                            {'params': classifier_Rot_red.parameters()}, {'params': classifier_Rot_green.parameters()}],
                           lr=lr, betas=(0.9, 0.99))

    bbxs = 0
    K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    data_path = '/home/potato/workplace/dataset/data'        ##### use total dataset
    dataloader = load_pts_train_cate(data_path, batch_size, K, cat, lim=1, rad=300, shuf=True, drop=True, corners=0,
                                     nw=nw)

    log = open('%s/log.txt' % opt.outf, 'w')
    # eval_log = open('%s/eval_log.txt' % opt.outf, 'w')

    # iou50, iou75, d5cm5, d10cm5 = 0, 0, 0, 0

    for epoch in range(sepoch, epochs):
        seg, res, recon, size, rot_g, rot_r = [], [], [], [], [], []
        if epoch > 0 and epoch % (epochs // 5) == 0:
            lr = lr / 2

        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 10
        optimizer.param_groups[2]['lr'] = lr * 20
        optimizer.param_groups[3]['lr'] = lr * 20
        # p

        for i, data in enumerate(dataloader):

            points, target_, Rs, Ts, obj_id, S, imgp, model = data['points'], data['label'], data['R'], data['T'], data[
                'cate_id'], data['scale'], data['dep'], data['model']


            #ptsori = points.clone()
            ptsori = data['ptsori']
            target_seg = target_[:, :, 0]  ###seg_target

            points_ = points.numpy().copy()

            points, corners, centers, pts_recon = data_augment(points_[:, :, 0:3], Rs, Ts, num_cor, target_seg, a=15.0)

            points, target_seg, pts_recon = Variable(torch.Tensor(points)), Variable(target_seg), Variable(pts_recon)

            points, target_seg, pts_recon = points.cuda(), target_seg.cuda(), pts_recon.cuda()

            pointsf = points[:, :, 0:3].unsqueeze(2)

            optimizer.zero_grad()
            points = pointsf.transpose(3, 1)
            points_n = pointsf.squeeze(2)

            # obj_idh = torch.zeros((1, 1))
            #
            # if obj_idh.shape[0] == 1:
            #     obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
            # else:
            #     obj_idh = obj_idh.view(-1, 1)
            obj_idh = obj_id.view(-1, 1)
            target_seg = torch.cat((target_seg, target_seg[:, :500]), dim=1)

            one_hot = torch.zeros(points.shape[0], 16).scatter_(1, obj_idh.cpu().long(), 1)
            one_hot = one_hot.cuda()  ## the pre-defined category ID
            # print(one_hot)
            model=model.cuda()

            points_n1 = torch.cat([points_n, model], dim=1)

            pred_seg, box_pred_, feavecs = classifier_seg3D(points_n1, one_hot)


            pred_choice = pred_seg.data.max(2)[1]  ## B N

            # print(pred_choice[0])
            p = pred_choice  # [0].cpu().numpy() B N
            N_seg = 1000 #pred_seg.shape[1]
            pts_s = torch.zeros(points.shape[0], N_seg, 3)

            box_pred = torch.zeros(points.shape[0], N_seg, 3)

            pts_sv = torch.zeros(points.shape[0], N_seg, 3)

            feat = torch.zeros(points.shape[0], N_seg, feavecs.shape[2])

            corners0 = torch.zeros((points.shape[0], num_cor, 3))
            if torch.cuda.is_available():
                ptsori = ptsori.cuda()
            #box_pred_ = box_pred_.cpu()
            #ptsori = ptsori.cpu()
            #feavecs = feavecs.cpu()
            Tt = np.zeros((points.shape[0], 3))
            for ib in range(points.shape[0]):
                if len(p[ib, :].nonzero()) < 100: # 10
                    continue
                aa = p[ib, :].nonzero()[:, 0]
                pts_ = torch.index_select(ptsori[ib, :, 0:3], 0, p[ib, :].nonzero()[:, 0])  ##Nx3

                box_pred__ = torch.index_select(box_pred_[ib, :, 0:3], 0, p[ib, :].nonzero()[:, 0])
                feavec_ = torch.index_select(feavecs[ib, 0:2500, :], 0, p[ib, :].nonzero()[:, 0])

                choice = np.random.choice(len(pts_), N_seg, replace=True)
                pts_s[ib, :, :] = pts_[choice, :]

                box_pred[ib] = box_pred__[choice]

                choice1 = np.random.choice(len(pts_), 800, replace=True)
                choice2 = np.random.choice(500, 200, replace=True)
                feat_ori = feavec_[choice1, :]
                feat_model = feavecs[ib, 2000:2500,:][choice2,:]
                feavec_ = torch.cat([feat_ori, feat_model], dim=0)


                feat[ib, :, :] = feavec_[:, :]

                corners0[ib] = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]]))

            pts_s = pts_s.cuda()

            pts_s = pts_s.transpose(2, 1)

            # rec = box_pred
            # rec = rec.cuda()
            # rec = rec.transpose(2,1)

            cen_pred, obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), obj_id)
            # cen_pred, obj_size = classifier_ce(rec, obj_id)

            feavec = feat.transpose(1, 2)

            kp_m = classifier_Rot_green(feavec)

            centers = Variable(torch.Tensor((centers)))

            corners = Variable(torch.Tensor((corners)))

            if torch.cuda.is_available():
                box_pred = box_pred.cuda()
                centers = centers.cuda()
                S = S.cuda()
                corners = corners.cuda()
                feat = feat.cuda()
                corners0 = corners0.cuda()

            loss_seg = Loss_seg3D(pred_seg.reshape(-1, pred_seg.size(-1)), target_seg.view(-1, ).long())
            loss_res = Loss_func_ce(cen_pred, centers.float())

            loss_size = Loss_func_s(obj_size, S.float())

            loss_vec = loss_recon(box_pred, pts_recon)

            kp_m2 = classifier_Rot_red(feat.transpose(1, 2))  # .detach())

            green_v = corners[:, 0:6].float().clone()
            red_v = corners[:, (0, 1, 2, 6, 7, 8)].float().clone()
            target = torch.tensor([[1]], dtype=torch.float).cuda()

            loss_rot_g = Loss_func_Rot1(kp_m, green_v)
            loss_rot_r = Loss_func_Rot2(kp_m2, red_v)

            symme = 1
            # if cat in ['bottle','bowl','can']:
            #     symme=0.0

            # Loss = loss_seg * 20.0 + loss_res / 10.0 + loss_vec / 200.0 + loss_size / 20.0 + symme * loss_rot_r / 500.0 + loss_rot_g / 500.0
            Loss = loss_seg * 20.0 + loss_res / 20.0 + loss_vec / 200.0 + loss_size / 20.0 + symme * loss_rot_r / 100.0 + loss_rot_g / 100.0
            Loss.backward()
            optimizer.step()

            print(cats[obj_id[0] - 1])
            log.write(cats[obj_id[0] - 1] + '\n')
            print('[%d: %d] train loss_seg: %f, loss_res: %f, loss_recon: %f, loss_size: %f, loss_rot_g: %f, '
                  'loss_rot_r: %f' % (
                      epoch, i, loss_seg.item(), loss_res.item(), loss_vec.item(), loss_size.item(), loss_rot_g.item(),
                      loss_rot_r.item()))

            log.write('[%d: %d] train loss_seg: %f, loss_res: %f, loss_recon: %f, loss_size: %f, loss_rot_g: %f, '
                      'loss_rot_r: %f\n' % (
                          epoch, i, loss_seg.item(), loss_res.item(), loss_vec.item(), loss_size.item(),
                          loss_rot_g.item(),
                          loss_rot_r.item()))

            print()

            torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_last_obj%s.pth' % (opt.outf,
                                                                                   cat1))
            torch.save(classifier_ce.state_dict(), '%s/Tres_last_obj%s.pth' % (opt.outf, cat1))
            torch.save(classifier_Rot_green.state_dict(),
                       '%s/Rot_g_last_obj%s.pth' % (opt.outf, cat1))
            torch.save(classifier_Rot_red.state_dict(),
                       '%s/Rot_r_last_obj%s.pth' % (opt.outf, cat1))

            seg.append(loss_seg.item())
            res.append(loss_res.item())
            recon.append(loss_vec.item())
            size.append(loss_size.item())
            rot_g.append(loss_rot_g.item())
            rot_r.append(loss_rot_r.item())

            if epoch > 0 and epoch % 20 == 0:  ##save mid checkpoints

                torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_epoch%d_obj%s.pth' % (opt.outf,
                                                                                          epoch, cat))
                torch.save(classifier_ce.state_dict(), '%s/Tres_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_green.state_dict(),
                           '%s/Rot_g_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_red.state_dict(),
                           '%s/Rot_r_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
        eval_log = open('%s/eval_log.txt' % opt.outf, 'a')
        eval_log.write('[epoch %d] train loss_seg: %f, loss_res: %f, loss_recon: %f, loss_size: %f, loss_rot_g: %f, '
                  'loss_rot_r: %f\n' % (
                      epoch, np.mean(seg), np.mean(res), np.mean(recon), np.mean(size), np.mean(rot_g), np.mean(rot_r)))
        eval_log.close()
        #if epoch > 0 and epoch % 9 == 0:
        #    for ca in cat:
        #        det(ca,cat1,opt.outf,epoch)

    #for ca in cat:
    #    det(ca, cat1, opt.outf, epochs)
    log.close()
