from torch.utils.data import DataLoader

from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config

import torch
import time
import argparse

from scipy.integrate import simps
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from train_config import config as cfg

ds = FaceKeypointDataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)
ds = DataLoader(ds,
                1,
                num_workers=cfg.TRAIN.process_num,
                shuffle=False)

from lib.core.model.face_model import Net

def vis(model):
    ###build model
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    for images, labels in ds:

        img_show = images.numpy()
        print(img_show.shape)
        img_show = np.transpose(img_show[0], axes=[1, 2, 0])

        images = images.to('cpu').float()

        start = time.time()
        res = model(images)
        res = res.detach().numpy()
        print(res)
        print('xxxx', time.time() - start)
        # print(res)

        img_show = img_show.astype(np.uint8)

        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(res[0][0:136]).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            # print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 128),
                                         int(x_y[1] * 128)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp', img_show)
        cv2.waitKey(0)

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse

def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate

def eval(model):
    checkpoint_file = os.path.join(cfg.MODEL.model_path, cfg.MODEL.checkpoint)
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.module.state_dict().keys()}
    model.module.load_state_dict(pretrained_dict)
    model.eval()
    nme_list = []
    pred_landmarks = []
    gt_landmarks = []
    for images, labels in ds:
        img_show = images.numpy()
        # print(img_show.shape)
        img_show = np.transpose(img_show[0], axes=[1, 2, 0])

        images = images.to('cpu').float()

        start = time.time()
        res = model(images)
        res = res.detach().numpy()
        # print(res)
        # print('time cost', time.time() - start)
        # print(res)

        img_show = img_show.astype(np.uint8)

        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        landmark = np.array(res[0][0:136]).reshape([-1, 2])
        pred_landmarks.append(landmark)

        gt_label = labels.numpy()
        gt_landmarks.append(gt_label[:, 0:136].reshape([-1, 2]))

        # for _index in range(landmark.shape[0]):
        #     x_y = landmark[_index]
        #     # print(x_y)
        #     cv2.circle(img_show, center=(int(x_y[0] * config.MODEL.hin),
        #                                  int(x_y[1] * config.MODEL.hin)),
        #                color=(255, 0, 0), radius=1, thickness=1)
        #
        # cv2.imshow('tmp', img_show)
        # cv2.waitKey(0)
    pred_landmarks = np.array(pred_landmarks)
    gt_landmarks = np.array(gt_landmarks)
    print(pred_landmarks.shape, gt_landmarks.shape)
    nme_temp = compute_nme(pred_landmarks, gt_landmarks)
    for item in nme_temp:
        nme_list.append(item)
    # nme
    print('nme: {:.4f}'.format(np.mean(nme_list)))
    # auc and failure rate
    failureThreshold = 0.1
    auc, failure_rate = compute_auc(nme_list, failureThreshold)
    print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
        failureThreshold, auc))
    print('failure_rate: {:}'.format(failure_rate))

def load_checkpoint(net, checkpoint):
    net.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=True)


if __name__ == '__main__':
    model = Net(model_name=cfg.MODEL.model_name, pretrained=False, num_classes=cfg.MODEL.out_channel)
    model = torch.nn.DataParallel(model)
    eval(model)
