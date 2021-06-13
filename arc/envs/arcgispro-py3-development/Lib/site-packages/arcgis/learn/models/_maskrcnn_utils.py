from fastai.vision import ImageSegment, Image
from fastai.vision.image import open_image, show_image, pil2tensor
from fastai.vision.data import SegmentationProcessor, ImageList
from fastai.layers import CrossEntropyFlat
from fastai.basic_train import LearnerCallback
import torch
import warnings
import PIL
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from torch import LongTensor
import os
from .._utils.common import ArcGISMSImage

from fastprogress.fastprogress import progress_bar


class ArcGISImageSegment(Image):
    "Support applying transforms to segmentation masks data in `px`."
    def __init__(self, x, cmap=None, norm=None):
        super(ArcGISImageSegment, self).__init__(x)
        self.cmap = cmap
        self.mplnorm = norm
        self.type = np.unique(x)

    def lighting(self, func, *args, **kwargs):
        return self

    def refresh(self):
        self.sample_kwargs['mode'] = 'nearest'
        return super().refresh()

    @property
    def data(self):
        "Return this image pixels as a `LongTensor`."
        return self.px.long()

    def show(self, ax = None, figsize = (3,3), title = None, hide_axis = True, cmap='tab20', alpha = 0.5, **kwargs):

        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        masks = self.data[0].numpy()
        for i in range(1, self.data.shape[0]):
            max_unique = np.max(np.unique(masks))
            mask = np.where(self.data[i]>0, self.data[i] + max_unique, self.data[i])
            masks += mask
        ax.imshow(masks, cmap=cmap, alpha=alpha, **kwargs)
        if hide_axis: ax.axis('off')
        if title: ax.set_title(title)


def is_no_color(color_mapping):
    if isinstance(color_mapping, dict):
        color_mapping = list(color_mapping.values())
    return (np.array(color_mapping) == [-1., -1., -1.]).any()

class ArcGISSegmentationLabelList(ImageList):
    "`ItemList` for segmentation masks."
    _processor = SegmentationProcessor
    def __init__(self, items, chip_size, classes=None, class_mapping=None, color_mapping=None, index_dir=None, **kwargs):
        super().__init__(items, **kwargs)
        self.class_mapping = class_mapping
        self.color_mapping = color_mapping
        self.copy_new.append('classes')
        self.classes, self.loss_func = classes, CrossEntropyFlat(axis=1)
        self.chip_size = chip_size
        self.inverse_class_mapping = {}
        self.index_dir = index_dir
        for k, v in self.class_mapping.items():
            self.inverse_class_mapping[v] = k
        if is_no_color(list(color_mapping.values())):
            self.cmap = 'tab20'  ## compute cmap from palette
            import matplotlib as mpl
            bounds = list(color_mapping.keys())
            if len(bounds) < 3: # Two handle two classes i am adding one number to the classes which is not already in bounds
                bounds = bounds + [max(bounds)+1]
            self.mplnorm = mpl.colors.BoundaryNorm(bounds, len(bounds))
        else:
            import matplotlib as mpl
            bounds = list(color_mapping.keys())
            if len(bounds) < 3: # Two handle two classes i am adding one number to the classes which is not already in bounds
                bounds = bounds + [max(bounds)+1]
            self.cmap = mpl.colors.ListedColormap(np.array(list(color_mapping.values()))/255)
            self.mplnorm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)

        if len(color_mapping.keys()) == 1:
            self.cmap = 'tab20'
            self.mplnorm = None

    def open(self, fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
            if len(fn) != 0:
                img_shape = ArcGISMSImage.read_image(fn[0]).shape
            else:
                labeled_mask = torch.zeros((len(self.class_mapping), self.chip_size, self.chip_size))
                return ArcGISImageSegment(labeled_mask, cmap=self.cmap, norm=self.mplnorm)
            k = 0

            labeled_mask = np.zeros((1, img_shape[0], img_shape[1]))

            for j in range(len(self.class_mapping)):

                if k < len(fn):
                    lbl_name = int(self.index_dir[self.inverse_class_mapping[fn[k].parent.name]])
                else:
                    lbl_name = len(self.class_mapping) + 2
                if lbl_name == j+1:                    
                    img = ArcGISMSImage.read_image(fn[k])
                    k = k + 1
                    if len(img.shape)==3:
                        img = img.transpose(2,0,1)
                        img_mask = img[0]
                        for i in range(1, img.shape[0]):
                            max_unique = np.max(np.unique(img_mask))
                            img_i = np.where(img[i]>0, img[i] + max_unique, img[i])
                            img_mask += img_i
                        img_mask = np.expand_dims(img_mask, axis = 0)
                    else:
                        img_mask = np.expand_dims(img, axis = 0)
                else:
                    img_mask = np.zeros((1, img_shape[0], img_shape[1]))
                labeled_mask = np.append(labeled_mask, img_mask, axis = 0)
            labeled_mask = labeled_mask[1:,:,:]
            labeled_mask = torch.Tensor(list(labeled_mask))
        return ArcGISImageSegment(labeled_mask, cmap=self.cmap, norm=self.mplnorm)

    def reconstruct(self, t): 
        return ArcGISImageSegment(t, cmap=self.cmap, norm=self.mplnorm)

class ArcGISInstanceSegmentationItemList(ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = ArcGISSegmentationLabelList, False
    _div = None
    _imagery_type = None
    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)

class ArcGISInstanceSegmentationMSItemList(ArcGISInstanceSegmentationItemList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = ArcGISSegmentationLabelList, False
    def open(self, fn):
        return ArcGISMSImage.open_gdal(fn)

def mask_rcnn_loss(loss_value, *args):

    final_loss = 0.
    for i in loss_value.values():
        i[torch.isnan(i)] = 0.
        i[torch.isinf(i)] = 0.
        final_loss += i
        
    return final_loss

def mask_to_dict(last_target, device):
    target_list = []
    
    for i in range(len(last_target)):

        boxes =  []
        masks = np.zeros((1, last_target[i].shape[1], last_target[i].shape[2]))
        labels = []
        for j in range(last_target[i].shape[0]):

            mask = np.array(last_target[i].data[j].cpu())
            obj_ids = np.unique(mask)

            if len(obj_ids)==1:
                continue

            obj_ids = obj_ids[1:]
            mask_j = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)

            for k in range(num_objs):
                pos = np.where(mask_j[k])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmax-xmin==0:
                    xmax += 1
                if ymax-ymin==0:
                    ymax += 1
                boxes.append([xmin, ymin, xmax, ymax])

            masks = np.append(masks, mask_j, axis = 0)
            labels_j = torch.ones((num_objs,), dtype=torch.int64)
            labels_j = labels_j*(j+1)
            labels.append(labels_j)
        
        if(masks.shape[0]==1): # if no object in image
            masks[0,50:51,50:51] = 1
            labels = torch.tensor([0])
            boxes = torch.tensor([[50.,50.,51.,51.]])
        else:
            labels = torch.cat(labels)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = masks[1:,:,:]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target = {}
        target["boxes"] = boxes.to(device)
        target["labels"] = labels.to(device)
        target["masks"] = masks.to(device)
        target_list.append(target)

    return target_list

class train_callback(LearnerCallback):

    def __init__(self, learn):
        super().__init__(learn)
   
    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Handle new batch `xb`,`yb` in `train` or validation."      
        target_list = mask_to_dict(last_target, self.c_device)
        self.learn.model.train()
        last_input = [list(last_input), target_list]
        last_target = [torch.tensor([1]) for i in last_target]
        return {'last_input':last_input, 'last_target':last_target}

def masks_iou(masks1, masks2):
    # Mask R-CNN

    # The MIT License (MIT)

    # Copyright (c) 2017 Matterport, Inc.

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.

    #Method is based on https://github.com/matterport/Mask_RCNN

    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return torch.zeros((masks1.shape[0], masks2.shape[0]))
    masks1 = masks1.permute(1,2,0)
    masks2 = masks2.permute(1,2,0)
    masks1 = torch.reshape(masks1 > .5, (-1, masks1.shape[-1])).type(torch.float64)
    masks2 = torch.reshape(masks2 > .5, (-1, masks2.shape[-1])).type(torch.float64)
    area1 = torch.sum(masks1, dim=0)
    area2 = torch.sum(masks2, dim=0)

    intersections = torch.mm(masks1.transpose(1,0), masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    return overlaps

def compute_matches(gt_class_ids, gt_masks,
                    pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, detect_threshold=0.5):

    #Method is based on https://github.com/matterport/Mask_RCNN
    indices = torch.argsort(pred_scores, descending=True)
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[indices]

    ious_mask = masks_iou(pred_masks, gt_masks)

    pred_match = -1 * np.ones([pred_masks.shape[0]])
    if 0 not in ious_mask.shape:
        max_iou, matches = ious_mask.max(1)
        detected = []
        for i in range(len(pred_class_ids)):
            if max_iou[i] >= iou_threshold and pred_scores[i] >= detect_threshold and matches[i] not in detected and gt_class_ids[matches[i]] == pred_class_ids[i]:
                detected.append(matches[i])
                pred_match[i] = pred_class_ids[i]

    return pred_match

def compute_ap(gt_class_ids, gt_masks,
               pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5, detect_threshold=0.5):

    #Method is based on https://github.com/matterport/Mask_RCNN
    pred_match = compute_matches(
        gt_class_ids, gt_masks,
        pred_class_ids, pred_scores, pred_masks,
        iou_threshold, detect_threshold)

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_class_ids)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    return mAP

def compute_class_AP(model, dl, n_classes, show_progress, detect_thresh=0.5, iou_thresh=0.5, mean=False):

    model.learn.model.eval()
    if mean:
        aps = []
    else:
        aps = [[] for _ in range(n_classes)]
    with torch.no_grad():
        for input,target in progress_bar(dl, display=show_progress):
            predictions = model.learn.model(list(input))
            ground_truth = mask_to_dict(target, model._device)
            for i in range(len(predictions)):

                predictions[i]["masks"] = predictions[i]["masks"].squeeze()
                if predictions[i]["masks"].shape[0] == 0:
                    continue
                if len(predictions[i]["masks"].shape) == 2:
                    predictions[i]["masks"] = predictions[i]["masks"][None]
                if mean:
                    ap = compute_ap(ground_truth[i]["labels"],
                                    ground_truth[i]["masks"],
                                    predictions[i]["labels"],
                                    predictions[i]["scores"],
                                    predictions[i]["masks"],
                                    iou_thresh,
                                    detect_thresh)
                    aps.append(ap)
                else:
                    for k in range(1, n_classes+1):
                        gt_labels_index = (ground_truth[i]["labels"]== k).nonzero().reshape(-1)
                        gt_labels = ground_truth[i]["labels"][gt_labels_index]
                        gt_masks = ground_truth[i]["masks"][gt_labels_index]
                        pred_labels_index = (predictions[i]["labels"]== k).nonzero().reshape(-1)
                        pred_labels = predictions[i]["labels"][pred_labels_index]
                        pred_masks = predictions[i]["masks"][pred_labels_index]
                        pred_scores = predictions[i]["scores"][pred_labels_index]
                        if len(gt_labels):
                            ap = compute_ap(gt_labels,
                                            gt_masks,
                                            pred_labels,
                                            pred_scores,
                                            pred_masks,
                                            iou_thresh,
                                            detect_thresh)
                            aps[k-1].append(ap)
    if mean:
        if aps != []:
            aps = np.mean(aps, axis=0)
        else:
            return 0.0
    else:
        for i in range(n_classes):
            if aps[i] != []:
                aps[i] = np.mean(aps[i])
            else:
                aps[i] = 0.0
    if model._device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return aps
