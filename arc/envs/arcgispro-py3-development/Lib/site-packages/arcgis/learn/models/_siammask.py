from ._arcgis_model import ArcGISModel
from pathlib import Path
import json
import random
import traceback
from ._arcgis_model import _EmptyData
from .._utils.env import raise_fastai_import_error

try:
    import torch
    import glob
    from torch import nn
    from .._utils.common import _get_emd_path
    from ._siammask_utils import train_callback, get_learner, \
        Custom, load_pretrain, siamese_init, siamese_track, calculate_iou, download_backbone
    import matplotlib.pyplot as plt
    from fastai.torch_core import split_model_idx
    from fastai.vision import flatten_model

    from fastai.basic_train import Learner
    from fastai.vision import ImageList
    from .._utils.pascal_voc_rectangles import ObjectDetectionCategoryList

    import numpy as np
    import cv2, os
    from PIL import Image
    from types import SimpleNamespace
    from os import makedirs
    from os.path import join, isdir, isfile


    from torch.autograd import Variable
    import torch.nn.functional as F

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False


class Track:
    """
    Creates a Track object, used to maintain the state of a track

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    id                      Required int. ID for each track initialized
    ---------------------   -------------------------------------------
    label                   Required String. label/class name of the track
    ---------------------   -------------------------------------------
    bbox                    Required list. Bounding box of the track
    ---------------------   -------------------------------------------
    mask                    Required numpy array. Mask for the tack
    =====================   ===========================================

    :returns: `Track` Object
    """
    def __init__(self, id, label, bbox, mask):
        self.id = id
        self.label = label
        self.bbox = bbox
        self.score = 1
        self.status = 16
        self.mask = mask
        self.location = None
        self.age = 0
        # self.color = color


class SiamMask(ArcGISModel):
    """
    Creates a SiamMask object.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Optional fastai Databunch. Returned data object from
                            `prepare_data` function. Default value is None.
    =====================   ===========================================

    :returns: `SiamMask` Object
    """

    def __init__(self, data=None, **kwargs):

        if not HAS_FASTAI:
            raise_fastai_import_error(import_exception=import_exception,
                                      message="",
                                      installation_steps=' ')

        self._is_multispectral = False

        pretrained_path = kwargs.get('pretrained_path')
        self.cfg = {}
        self.anchors = {
            "stride": 8,
            "ratios": [0.33, 0.5, 1, 2, 3],
            "scales": [8],
            "round_dight": 0
        }
        self.cfg["network"] = {"arch": "Custom"}
        self.cfg["hp"] = {
            "instance_size": 255,
            "base_size": 8,
            "out_size": 127,
            "seg_thr": 0.35,
            "penalty_k": 0.04,
            "window_influence": 0.4,
            "lr": 1.0
        }

        self.track_list = []
        self.state_list = {}
        self.num_tracks = 0

        if data is not None:
            super().__init__(data)
            self.learn = get_learner(data=data, anchors=self.anchors)
            self.learn.callbacks.append(train_callback(self.learn))
        else:
            data = create_siammask_data()

            file_path = download_backbone(url="http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth",
                                          file_name="SiamMask_DAVIS.pth")
            model = Custom(anchors=self.anchors, pretrain=False)
            model = load_pretrain(model, file_path)
            self.learn = Learner(data=data, model=model)
            self._backend = "pytorch"
            self._data = data
            self._learning_rate = None
            self._model_metrics_cache = None

            class Resnet50():
                def __init__(self):
                    self.name = "Resnet50"
            self._backbone = Resnet50


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.learn.model = self.learn.model.to(self._device)
        # self.learn.c_device = self._device

        self.freeze()
        self._arcgis_init_callback()

        self.load_model = pretrained_path



    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)

    @property
    def supported_backbones(self):
        """ Supported torchvision backbones for this model. """
        return ["resnet50"]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a SiamMask Object tracker from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from `prepare_data` function or None for
                                inferencing.
        =====================   ===========================================

        :returns: `SiamMask` Object
        """
        emd_path = _get_emd_path(emd_path)
        emd_path = Path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd['ModelFile'])
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd['ModelParameters']
        if data is None:
            data = _EmptyData(path=emd_path.parent,
                              loss_func=None,
                              c=2,
                              chip_size=224)

            data.emd_path = emd_path
            data.emd = emd
            data._dataset_type = "_emptydata"
            for key, value in emd['DataAttributes'].items():
                setattr(data, key, value)

        return cls(data,
                   **model_params,
                   pretrained_path=str(model_file), load_from_model=True)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {"DataAttributes": {}, "ModelParameters": {}}
        # chip size
        _emd_template["DataAttributes"]["chip_size"] = self._data.chip_size
        _emd_template["DataAttributes"]["_is_multispectral"] = self._data._is_multispectral

        norm_stats = []
        for k in self._data.infos:
            norm_stats.append(k)
        _emd_template["DataAttributes"]["norm_stats"] = list(norm_stats)
        _emd_template["DataAttributes"]["batch_size"] = self._data.batch_size
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        # object classifier config can be used.
        _emd_template["ModelConfiguration"] = "_siammask_inferencing"
        _emd_template["ModelParameters"]["hp"] =  self.cfg["hp"]
        _emd_template["ModelParameters"]["anchors"] =  self.anchors
        # Model Type
        _emd_template["ModelType"] = "ObjectTracking"

        return _emd_template


    def freeze(self):
        "Freezes the pretrained backbone."

        self.learn.layer_groups = split_model_idx(self.learn.model, [158])
        self.learn.create_opt(lr=0.000478630092322)


    def init(self, frame, detections, labels=None, reset=True):
        """
        Initializes the position of the object in the frame/Image using detections.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is used to initialize
                                the object to track.
        ---------------------   -------------------------------------------
        detections              Required list. A list of bounding box to intialize
                                the object.
        ---------------------   -------------------------------------------
        labels                  Optional list. A list of class labels to intialize
                                the object.
        =====================   ===========================================

        :returns: Track list
        """

        if reset:
            self.track_list = []
            self.state_list = {}
            self.num_tracks = 0

        if self.load_model is not None:

            pretrained_path = self.load_model
            siammask = Custom(anchors=self.anchors)
            siammask = load_pretrain(siammask, pretrained_path)
        else:
            siammask = self.learn.model

        siammask = siammask.eval().to(self.device)

        filtered_detections = []
        for i, detection in enumerate(detections):
            x, y = detection[0], detection[1]
            w, h = detection[2], detection[3]
            add = True
            for key, val in self.state_list.items():
                v = val[1]

                box1 = v[0], v[1], v[2], v[3]
                box2 = x, y, w, h

                iou = calculate_iou(box1, box2)
                if iou > 0:
                    add = False
                    break
            if add:
                filtered_detections.append(detection)
                x, y = detection[0], detection[1]
                w, h = detection[2], detection[3]
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(frame, target_pos,
                                     target_sz, siammask,
                                     self.cfg['hp'], device=self.device)
                # mask = state['mask'] > state['p'].seg_thr

                if labels is not None:
                    track = Track(self.num_tracks, labels[i], [x, y, w, h], None)
                else:
                    track = Track(self.num_tracks, "Object", [x, y, w, h], None)
                self.track_list.append(track)
                self.state_list[track.id] = [state, [x, y, w, h]]
                self.num_tracks += 1

        return self.track_list

    def update(self, frame):
        """
        Tracks the position of the object in the frame/Image

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is used to update
                                the object track.
        =====================   ===========================================

        :returns: Updated track list
        """
        for track in self.track_list:
            state = siamese_track(self.state_list[track.id][0], frame, mask_enable=True,
                                  refine_enable=True, device=self.device)
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            # frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            target_pos = state["target_pos"]
            w, h = state["target_sz"]
            x, y = target_pos[0] - (w / 2), target_pos[1] - (h / 2)
            track.bbox = [x, y, w, h]
            track.mask = mask
            track.location = location
            track.age += 1
            track.label = track.label
            track.score = state["score"]
            self.state_list[track.id][1] = [x, y, w, h]
        return self.track_list

    def remove(self, track_ids):
        """
        Removes the tracks from the track list using track_ids

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        track_ids               Required List. List of track ids to be removed
                                from the track list.
        =====================   ===========================================

        :returns: Updated track list
        """
        try:
            for id in track_ids:
                for i in self.track_list:
                    if i.id == id:
                        self.track_list.remove(i)
                try:
                    del self.state_list[id]
                except Exception as e:
                    print(e)

                if self.num_tracks > 0:
                    self.num_tracks -= 1
            # print("Tracks has been removed succesfully!")
        except Exception as e:
            print(e)

        return self.track_list

    def show_results(self, rows=5):
        """
        Displays the results of a trained model on a part of the validation set

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows to display.
        =====================   ===========================================

        """
        if self._data._dataset_type == "_ObjectTracking":
            self._check_requisites()
            return None
        self._check_requisites()
        self.track_list = []
        self.state_list = {}
        self.num_tracks = 0

        idx = random.randint(0, len(self.learn.data.valid_ds) - 1)
        data = self.learn.data.valid_ds.__getitem__(idx, debug=False, show_batch=False, show_results=True)
        base_folder = data[0].split("crop")[0]
        folder_name = data[0].split("\\")[-2]
        frame_id = data[0].split("\\")[-1].split(".")[1]
        json_data = self.learn.data.valid_ds.show_batch_data[folder_name][frame_id]
        bbox = json_data[list(json_data.keys())[0]]
        all_images = glob.glob(os.path.join(base_folder, "JPEGImages", folder_name, "*.jpg"))
        image_counter = 0
        track_id_set = set()
        if (len(all_images) //3 ) < rows:
            rows = len(all_images) //3

        fig, axes = plt.subplots(nrows=rows,
                                 ncols=3,
                                 squeeze=False,
                                 figsize=(20, rows * 3))

        for idx in range(0, rows):
            if len(all_images) == image_counter:
                break
            for j in range(0, 3):
                if image_counter == 0:
                    img_path = all_images[image_counter]
                    image = cv2.imread(img_path, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    x, y = int(bbox[0]), int(bbox[1])
                    x1, y1 = int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
                    w, h = x1 - x, y1 - y
                    detections = [[x, y, w, h]]
                    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]) - int(bbox[0]), int(bbox[3] - int(bbox[1]))), (255, 0, 0), 2)

                    self.init(image, detections)
                else:
                    img_path = all_images[image_counter]
                    image = cv2.imread(img_path, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tracks = self.update(image)
                    for track in tracks:
                        track_id_set.add(track.id)
                        mask = track.mask
                        image[:, :, 2] = (mask > 0) * 255 + (mask == 0) * image[:, :, 2]
                        cv2.polylines(image, [np.int0(track.location).reshape((-1, 1, 2))], True, (w, 255, h), 3)

                axes[idx][j].set_xticks([])
                axes[idx][j].set_yticks([])
                axes[idx][j].imshow(image)
                image_counter += 1

    @property
    def _model_metrics(self):
        if self._data._dataset_type == "_ObjectTracking":
            return {}
        self._check_requisites()
        return self.compute_metrics()

    def compute_metrics(self, iou_thres=0.2):
        """
        Computes mean IOU and f-measure on validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        iou_thresh              Optional float. The intersection over union
                                threshold with the ground truth mask, above
                                which a predicted mask will be
                                considered a true positive.
        =====================   ===========================================

        :returns: `dict` with mean IOU and F-Measure
        """
        self._check_requisites()
        if self._data._dataset_type == "_ObjectTracking":
            return {}
        tp = 0
        fp = 0
        fn = 0
        all_ious = []
        data = self.learn.data
        sep = os.sep
        for folder in data.val_folders.keys():
            all_images = glob.glob(os.path.join(data.path, "JPEGImages", folder, "*.jpg"))
            init = False
            for img in all_images:
                objects = data.val_folders[folder].keys()
                img_name = img.split(sep)[-1].split(".")[0]
                gt_bboxes = []
                for obj in objects:
                    img_list = [x for x in list(data.val_folders[folder][obj].keys())]
                    for i in img_list:
                        if int(img_name) == int(i):
                            gt_bbox = data.val_folders[folder][obj][i]
                            gt_bboxes.append(gt_bbox)
                            break

                image = cv2.imread(img, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                anno = img.replace("JPEGImages", "Annotations").replace("jpg", "png")
                anno_img = cv2.imread(anno, 0)
                if len(np.unique(anno_img)) < 2:
                    continue

                detections = []
                for bbox in gt_bboxes:
                    x, y = int(bbox[0]), int(bbox[1])
                    x1, y1 = int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
                    w, h = x1 - x, y1 - y
                    detections.append([x, y, w, h])

                if not init:
                    self.init(image, detections)
                    init = True
                    clr = np.unique(anno_img)
                else:
                    tracks = self.update(image)
                    fn_temp = 0
                    tp_temp = 0
                    fp_temp = 0
                    for ind, track in enumerate(tracks):
                        mask = track.mask
                        if mask.shape[0] == 0:
                            fn_temp += 1
                            continue

                        pred_mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                        pred_mask[:, :, 2] = (mask > 0) * 255 + (mask == 0) * pred_mask[:, :, 2]
                        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
                        pred_mask[np.where(pred_mask > 0)] = 255

                        anno_mask = (anno_img == clr[ind + 1]).astype(np.uint8)
                        contour, _ = cv2.findContours(anno_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        gt_mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                        gt_mask = cv2.drawContours(gt_mask, contour, -1, (255, 255, 255), -1)

                        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

                        iou = get_ious(pred_mask, gt_mask)

                        if iou >= iou_thres:
                            tp_temp += 1
                        else:
                            fp_temp += 1

                        all_ious.append(iou)
                    fn += fn_temp
                    tp += tp_temp
                    fp += fp_temp

        f_measure = get_f_measure(tp, fp, fn)
        mean_iou = np.mean(all_ious, axis=0)
        return {'mean_IOU': mean_iou, "f_measure": f_measure}


def create_siammask_data():
    """ Create an empty databunch for siammask dataset."""

    train_tfms = []
    val_tfms = []
    ds_tfms = (train_tfms, val_tfms)

    siammask_class_ids = [0]

    siammask_label_names = ('background')

    siammask_class_mapping = {k: v for k, v in enumerate(siammask_label_names)}
    class_mapping = {k: v for k, v in siammask_class_mapping.items() if k in siammask_class_ids}

    import tempfile
    sd = ImageList([], path=tempfile.NamedTemporaryFile().name, ignore_empty=True).split_none()
    data = sd.label_const(0, label_cls=ObjectDetectionCategoryList, classes=list(class_mapping.values())).transform(
        ds_tfms).databunch()

    data.class_mapping = class_mapping
    data.classes = list(class_mapping.values())
    data._is_empty = False
    data._is_siammask = True
    data.resize_to = 127
    data.chip_size = 127
    data._is_multispectral = False

    data.infos = {
                    'template': 127,
                    'search': 143
                  }
    data._dataset_type = '_ObjectTracking'
    # data._is_empty = True

    return data

def get_ious(pred_mask, gt_mask):
    mask_sum = (pred_mask > 0).astype(np.uint8) + (gt_mask > 0).astype(np.uint8)
    intxn = np.sum(mask_sum == 2)
    union = np.sum(mask_sum > 0)
    iou = intxn / (union + 1e-6)
    return iou

def get_f_measure(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision > 0:
        f_measure = (2 * precision * recall) / (precision + recall)
    else:
        f_measure = 0.0
    return f_measure