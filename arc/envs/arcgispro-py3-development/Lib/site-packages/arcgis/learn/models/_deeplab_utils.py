# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Based on https://github.com/hszhao/semseg

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
from torchvision.models.segmentation.segmentation import _segm_resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCNHead
from ._arcgis_model import _get_backbone_meta
from fastprogress.fastprogress import progress_bar
from ._PointRend import PointRendSemSegHead
from fastai.vision import flatten_model

class Deeplab(nn.Module):
    def __init__(self, num_classes, backbone_fn, chip_size=224, pointrend=True, keep_dilation=False):
        super().__init__()
        self.pointrend = pointrend
        self.vgg = False
        if getattr(backbone_fn, '_is_multispectral', False):
            self.backbone = create_body(backbone_fn, pretrained=True, cut=_get_backbone_meta(backbone_fn.__name__)['cut'])
        else:
            self.backbone = create_body(backbone_fn, pretrained=True)
        
        backbone_name = backbone_fn.__name__

        ## Support for different backbones
        if "densenet" in backbone_name or "vgg" in backbone_name:
            hookable_modules = list(self.backbone.children())[0]
        else:
            hookable_modules = list(self.backbone.children())
        
        if "vgg" in backbone_name:
            modify_dilation_index = -5
            self.vgg = True
        else:
            if self.pointrend and not keep_dilation:
                modify_dilation_index = -1
            else:
                modify_dilation_index = -2
            
        if backbone_name == 'resnet18' or backbone_name == 'resnet34':
            module_to_check = 'conv' 
        else:
            module_to_check = 'conv2'
        
        if "vgg" in backbone_name:
            hooks = [hookable_modules[i-1] for i, module in enumerate(hookable_modules) if isinstance(module, nn.MaxPool2d)]

        else:
            hooks = [hookable_modules[-2], hookable_modules[-4]]

        custom_idx = 0
        for i, module in enumerate(hookable_modules[modify_dilation_index:]): 
            dilation = 2 * (i + 1)
            padding = 2 * (i + 1)
            for n, m in module.named_modules():
                if module_to_check in n:
                    m.dilation, m.padding, m.stride = (dilation, dilation), (padding, padding), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)                    
                    
            if "vgg" in backbone_fn.__name__:
                if isinstance(module, nn.Conv2d):
                    dilation = 2 * (custom_idx + 1)
                    padding = 2 * (custom_idx + 1)
                    module.dilation, module.padding, module.stride = (dilation, dilation), (padding, padding), (1, 1)
                    custom_idx += 1

        ## Hook at the index where we need to get the auxillary logits out along with Fine-grained features 
        self.hook = hook_outputs(hooks)
        
        ## returns the size of various activations
        feature_sizes = model_sizes(self.backbone, size=(chip_size, chip_size))
        
        if not self.vgg:
            ## Geting the number of channel persent in stored activation inside of the hook
            num_channels_aux_classifier = self.hook[0].stored.shape[1]
            ## Get number of channels in the last layer
            num_channels_classifier = feature_sizes[-1][1]
        else:
            num_channels_aux_classifier = self.hook[-2].stored.shape[1]
            num_channels_classifier = self.hook[-1].stored.shape[1]

        self.classifier = DeepLabHead(num_channels_classifier, num_classes)
        self.aux_classifier = FCNHead(num_channels_aux_classifier, num_classes)

        if self.pointrend:

            if self.vgg:
                num_channels = self.hook[-3].stored.shape[1] + self.hook[-4].stored.shape[1]
                stride = chip_size / self.hook[-1].stored.shape[2]  
            else:
                num_channels = self.hook[1].stored.shape[1]
                stride = chip_size / feature_sizes[-1][2]

            subdivision_steps = math.log(stride, 2)
            self.pointrend_head = PointRendSemSegHead(num_classes,
                                                    num_channels,
                                                    train_num_points=(chip_size/stride)**2,
                                                    subdivision_num_points=(chip_size/(stride/2))**2,
                                                    subdivision_steps=subdivision_steps)

    def forward(self, x):

        x_size = x.size()
        x = self.backbone(x)
        features = self.hook.stored
        
        if self.vgg:
            x = self.classifier(features[-1])
        else:
            x = self.classifier(x)

        if self.pointrend:
            
            if self.vgg:
                pointrend_out = self.pointrend_head(x, [features[-4], features[-3]])
            else:
                pointrend_out = self.pointrend_head(x, [features[1]])

        result = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

        if self.training:

            if self.vgg:
                x = self.aux_classifier(features[-2])
            else:
                x = self.aux_classifier(features[0])

            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

            if self.pointrend:
                return result, x, pointrend_out
            else:
                return result, x
        else:

            if self.pointrend:
                return pointrend_out
            else:
                return result

def mask_iou(mask1, mask2):
     
    mask1 = mask1.permute(0, 2, 3, 1)
    mask2 = mask2.permute(0, 2, 3, 1)
    mask1 = torch.reshape(mask1 > 0, (-1, mask1.shape[-1])).type(torch.float64)
    mask2 = torch.reshape(mask2 > 0, (-1, mask2.shape[-1])).type(torch.float64)
    area1 = torch.sum(mask1, dim=0)
    area2 = torch.sum(mask2, dim=0)
    intersection = torch.sum(mask1*mask2, dim=0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    return iou

def compute_miou(model, dl, mean, num_classes, show_progress, ignore_mapped_class=[]):

    ious=[]
    model.learn.model.eval()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=show_progress):
            pred = model.learn.model(input)
            target = target.squeeze(1)
            if ignore_mapped_class != []:
                for k in ignore_mapped_class:
                    pred[:, k] = -1000
                pred = pred.argmax(dim=1)
            else:
                pred = pred.argmax(dim=1)
            mask1 = []
            mask2 = []
            for i in range(pred.shape[0]):
                mask1.append(pred[i].to(model._device) == num_classes[:, None, None].to(model._device))
                mask2.append(target[i].to(model._device) == num_classes[:, None, None].to(model._device))
            mask1 = torch.stack(mask1)
            mask2 = torch.stack(mask2)
            iou = mask_iou(mask1, mask2)
            ious.append(iou.tolist())

    return np.mean(ious, 0)