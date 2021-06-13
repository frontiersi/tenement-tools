#MIT License

#Copyright (c) 2018 XuanyiLi

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# Based on https://github.com/meteorshowers/hed-pytorch/

import torch
import warnings
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import models
import math
from fastai.callbacks.hooks import hook_outputs
from fastai.vision.learner import create_body
from fastai.callbacks.hooks import model_sizes
from ._arcgis_model import _get_backbone_meta
from fastprogress.fastprogress import progress_bar
from skimage.morphology import skeletonize, binary_dilation

class _HEDModel(nn.Module):
    def __init__(self, backbone_fn, chip_size=224):
        super().__init__()

        backbone_name = backbone_fn.__name__
        if "vgg" in backbone_name:
            self.backbone = create_body(backbone_fn, pretrained=True)[0]#[:-1]
        else:
            self.backbone = create_body(backbone_fn, pretrained=True)
        
        self.hookable_modules = list(self.backbone.children())

        for i, module in enumerate(self.hookable_modules):
            if isinstance(module, nn.MaxPool2d):
                module.ceil_mode = True
                module.kernel_size = 2
            elif isinstance(module, nn.Conv2d) and i==0:
                module.stride = (1, 1)

        if "vgg" in backbone_name:
            hooks = [self.hookable_modules[i-1] for i, module in enumerate(self.hookable_modules) if isinstance(module, nn.MaxPool2d)]

        else:
            hooks = [self.hookable_modules[i] for i in range(2,8) if i != 3]
        
        self.hook = hook_outputs(hooks)
        model_sizes(self.backbone, size=(chip_size, chip_size))
        layer_num_channels = [k.stored.shape[1] for k in self.hook]

        self.score_dsn1 = nn.Conv2d(layer_num_channels[0], 1, 1)
        self.score_dsn2 = nn.Conv2d(layer_num_channels[1], 1, 1)
        self.score_dsn3 = nn.Conv2d(layer_num_channels[2], 1, 1)
        self.score_dsn4 = nn.Conv2d(layer_num_channels[3], 1, 1)
        self.score_dsn5 = nn.Conv2d(layer_num_channels[4], 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)
        
    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        x = self.backbone(x)
        features = self.hook.stored
        
        so1 = self.score_dsn1(features[0])
        so2 = self.score_dsn2(features[1])
        so3 = self.score_dsn3(features[2])
        so4 = self.score_dsn4(features[3])
        so5 = self.score_dsn5(features[4])
        
        weight_deconv2 =  make_bilinear_weights(4, 1).to(x.device)
        weight_deconv3 =  make_bilinear_weights(8, 1).to(x.device)
        weight_deconv4 =  make_bilinear_weights(16, 1).to(x.device)
        weight_deconv5 =  make_bilinear_weights(32, 1).to(x.device)
        
        upsample2 = torch.nn.functional.conv_transpose2d(so2, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)
        
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        
        return results
    
def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]
    
def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def cross_entropy_loss(prediction, label):
    
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask)#, reduce=False)
    return torch.sum(cost)

def hed_loss(out, labels):
    loss = 0.
    for o in out:
        loss = loss + cross_entropy_loss(o, labels)
    return loss

def accuracy(input, target):
    if isinstance(input, tuple): # while training
        input = input[0]
    target = target.byte().squeeze(1)
    input = input[-1]
    input = (input>=0.5).byte().squeeze(1)
    return (input == target).float().mean()

def get_true_positive(mask1, mask2, buffer):
    tp = 0
    indices = np.where(mask1 == 1)
    for ind in range(len(indices[0])):
        tp += np.any(
            mask2[max(indices[0][ind]-buffer,0): indices[0][ind]+buffer+1,
              max(indices[1][ind]-buffer,0): indices[1][ind]+buffer+1]).astype(np.int)
    return tp

def get_confusion_metric(gt, pred, buffer):
    
    tp, predicted_tp, actual_tp = 0, 0, 0
    for i in range(gt.shape[0]):
        gt_mask = skeletonize(binary_dilation(gt[i]))
        pred_mask = skeletonize(binary_dilation(pred[i]))
        tp += get_true_positive(gt_mask, pred_mask, buffer)
        predicted_tp += len(np.where(pred_mask == 1)[0])
        actual_tp += len(np.where(gt_mask == 1)[0])
    
    return tp, predicted_tp, actual_tp

def f1_score(pred, gt):

    gt = gt.byte().squeeze(1).cpu().numpy()
    pred = (pred[-1]>=0.5).byte().squeeze(1).cpu().numpy()
    tp, predicted_tp, actual_tp = get_confusion_metric(gt, pred, 3)
    precision = tp/(predicted_tp + 1e-12)
    recall = tp/(actual_tp + 1e-12)
    f1score = 2*precision*recall/(precision + recall + 1e-12)
    return torch.tensor(f1score)

def accuracies(model, dl, detect_thresh=0.5, buffer=3, show_progress=True):

    precision, recall, f1score = [], [], []
    model.learn.model.eval()
    acc = {}
    with torch.no_grad():
        for input, gt in progress_bar(dl, display=show_progress):
            predictions = model.learn.model(input)
            gt = gt.byte().squeeze(1).cpu().numpy()
            pred = (predictions[-1]>=detect_thresh).byte().squeeze(1).cpu().numpy()
            tp, predicted_tp, actual_tp = get_confusion_metric(gt, pred, buffer)
            prec = tp/(predicted_tp + 1e-12)
            rec = tp/(actual_tp + 1e-12)
            precision.append(prec)
            recall.append(rec)
            f1score.append(2*prec*rec/(prec + rec + 1e-12))
    acc['Precision'] = np.mean(precision)
    acc['Recall'] = np.mean(recall)
    acc['F1 Score'] = np.mean(f1score)

    return acc
