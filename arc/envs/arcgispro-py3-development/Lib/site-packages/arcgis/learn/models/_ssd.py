from ._arcgis_model import ArcGISModel, _get_device
from pathlib import Path
import json
from ._codetemplate import code
import warnings
import math      
from .._data import _raise_fastai_import_error  
import traceback    

import logging
logger = logging.getLogger()

HAS_OPENCV = True
HAS_FASTAI = True
HAS_ARCPY = True

try:
    import torch
    import numpy as np
    from fastai.vision.learner import cnn_learner
    from fastai.callbacks.hooks import model_sizes
    from fastai.vision.learner import create_body, cnn_config
    from fastai.vision.data import ImageDataBunch
    from fastai.vision import ImageList
    from fastai.vision import imagenet_stats, normalize
    from fastai.vision.image import open_image, bb2hw, image2np, Image, pil2tensor
    from torchvision.models import resnet34
    from torchvision.models import mobilenet_v2
    from torchvision import models
    from .._utils.pascal_voc_rectangles import ObjectDetectionCategoryList, show_results_multispectral
    from ._ssd_utils import SSDHead, BCE_Loss, FocalLoss, one_hot_embedding, nms, postprocess
    from ._ssd_utils import compute_class_AP, SSDHeadv2, kmeans, avg_iou
    from .._data import prepare_data
    from fastai.callbacks import EarlyStoppingCallback
    from ._arcgis_model import SaveModelCallback, _set_multigpu_callback, _resnet_family, _vgg_family, _densenet_family, _change_tail
    from ._unet_utils import is_no_color
    from torch.nn import Module as NnModule
    import PIL
    from .._image_utils import _get_image_chips, _get_transformed_predictions, _draw_predictions, _exclude_detection
    from .._video_utils import VideoUtils
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path, read_image
    from fastprogress.fastprogress import progress_bar
    from .._utils.env import _IS_ARCGISPRONOTEBOOK
    import matplotlib.pyplot as plt
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    class NnModule():
        pass
    HAS_FASTAI = False

try:
    import cv2
except Exception:
    HAS_OPENCV = False

try:
    import arcpy
except Exception:
    HAS_ARCPY = False


def _mobilenet_split(m:NnModule): return m[0][0][0], m[1]


class SingleShotDetector(ArcGISModel):

    """
    Creates a Single Shot Detector with the specified grid sizes, zoom scales
    and aspect ratios. Based on Fast.ai MOOC Version2 Lesson 9.
    
    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    grids                   Required list. Grid sizes used for creating anchor
                            boxes.
    ---------------------   -------------------------------------------
    zooms                   Optional list. Zooms of anchor boxes.
    ---------------------   -------------------------------------------
    ratios                  Optional list of tuples. Aspect ratios of anchor
                            boxes.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `SingleShotDetector`, which
                            is `resnet34` by default.
    ---------------------   -------------------------------------------
    dropout                 Optional float. Dropout probability. Increase it to
                            reduce overfitting.
    ---------------------   -------------------------------------------
    bias                    Optional float. Bias for SSD head.
    ---------------------   -------------------------------------------
    focal_loss              Optional boolean. Uses Focal Loss if True.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    location_loss_factor    Optional float. Sets the weight of the bounding box
                            loss. This should be strictly between 0 and 1. This 
                            is default `None` which gives equal weight to both 
                            location and classification loss. This factor
                            adjusts the focus of model on the location of 
                            bounding box.
    ---------------------   -------------------------------------------
    ssd_version             Optional int within [1,2]. Use version=1 for arcgis v1.6.2 or earlier
    ---------------------   -------------------------------------------
    backend                 Optional string. Controls the backend framework to be used
                            for this model, which is 'pytorch' by default.

                            valid options are 'pytorch', 'tensorflow'
    =====================   ===========================================
    
    :returns: `SingleShotDetector` Object
    """

    def __init__(self, data, grids=None, zooms=[1.], ratios=[[1., 1.]],
                 backbone=None, drop=0.3, bias=-4., focal_loss=False, 
                 pretrained_path=None, location_loss_factor=None, 
                 ssd_version=2, backend='pytorch', *args, **kwargs):

        super().__init__(data, backbone, **kwargs)

        self._backend = backend
        if self._backend == 'tensorflow':
            self._intialize_tensorflow(data, grids, zooms, ratios, backbone, drop, bias, pretrained_path, location_loss_factor)
        else:
            # assert (location_loss_factor is not None) or ((location_loss_factor > 0) and (location_loss_factor < 1)),
            if not ssd_version in [1, 2]:
                raise Exception("ssd_version can be only [1,2]")

            if location_loss_factor is not None:
                if not ((location_loss_factor > 0) and (location_loss_factor < 1)):
                    raise Exception('`location_loss_factor` should be greater than 0 and less than 1')
            self.location_loss_factor = location_loss_factor

            self._code = code
            self.ssd_version = ssd_version

            backbone_cut = None
            backbone_split = None

            self._check_dataset_support(self._data)
            if not (self._check_backbone_support(getattr(self, '_backbone', backbone))):
                raise Exception (f"Enter only compatible backbones from {', '.join(self.supported_backbones)}")
            backbone_name = self._backbone.__name__[:3]

            if self._backbone.__name__ == 'mobilenet_v2':
                backbone_cut = -1
                backbone_split = _mobilenet_split

            if ssd_version == 1:
                if grids == None:
                    grids =[4,2,1]
                    
                self._create_anchors(grids, zooms, ratios)

                feature_sizes = model_sizes(create_body(self._backbone, cut=backbone_cut), size=(data.chip_size, data.chip_size))
                num_features = feature_sizes[-1][-1]
                num_channels = feature_sizes[-1][1]

                ssd_head = SSDHead(grids, self._anchors_per_cell, data.c, num_features=num_features, drop=drop, bias=bias, num_channels=num_channels)
            elif ssd_version == 2:

                # find bounding boxes height and width
            
                if grids is None:
                    logger.info("Computing optimal grid size...")
                    hw = data.height_width
                    hw = np.array(hw)
                    
                    # find most suitable centroids for dataset
                    centroid = kmeans(hw , 1) 
                    avg = avg_iou(hw, centroid)

                    for num_anchor in range(2, 5):
                        new_centroid = kmeans(hw, num_anchor)
                        new_avg = avg_iou(hw, new_centroid)
                        if (new_avg - avg) < 0.05:
                            break
                        avg = new_avg
                        centroid = new_centroid.copy()

                    # find grid size

                    grids = list(map(int, map(round, data.chip_size/np.sort(np.max(centroid, axis=1)))))
                    grids = list(set(grids))
                    grids.sort(reverse = True)
                    if grids[-1] == 0:
                        grids[-1] = 1
                    grids = list(set(grids))
                
                self._create_anchors(grids, zooms, ratios)
                if hasattr(self, '_orig_backbone'):
                    feature_sizes = model_sizes(create_body(self._orig_backbone, cut=backbone_cut), size=(data.chip_size, data.chip_size))
                else:
                    feature_sizes = model_sizes(create_body(self._backbone, cut=backbone_cut), size=(data.chip_size, data.chip_size))
                num_features = feature_sizes[-1][-1]
                num_channels = feature_sizes[-1][1] 

                if grids[0] > 8 and abs(num_features - grids[0]) > 4 and backbone_name == 'res':
                    num_features = feature_sizes[-2][-1]
                    num_channels = feature_sizes[-2][1]
                    backbone_cut = -3
                ssd_head = SSDHeadv2(grids, self._anchors_per_cell, data.c, num_features=num_features, drop=drop, bias=bias, num_channels=num_channels)

            else:
                raise Exception('SSDVersion can only be 1 or 2')
            
            if hasattr(self, '_backbone_ms'):
                self._orig_backbone = self._backbone
                self._backbone = self._backbone_ms

            if hasattr(self, '_orig_backbone') and 'densenet' in self._orig_backbone.__name__:
                backbone_cut = cnn_config(self._orig_backbone)['cut']
                backbone_split = cnn_config(self._orig_backbone)['split']

            self.learn = cnn_learner(data=data, base_arch=self._backbone, cut=backbone_cut, split_on=backbone_split, custom_head=ssd_head)
            self._arcgis_init_callback() # make first conv weights learnable

            if focal_loss:
                self._loss_f = FocalLoss(data.c)
            else:
                self._loss_f = BCE_Loss(data.c)
            self.learn.loss_func = self._ssd_loss

            _set_multigpu_callback(self)
            if pretrained_path is not None:
                self.load(pretrained_path)        

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)

    @property
    def supported_backbones(self):
        """ Supported torchvision backbones for this model. """
        return SingleShotDetector._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family, *_densenet_family, *_vgg_family, models.mobilenet_v2.__name__]

    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return SingleShotDetector._supported_datasets()
    
    @staticmethod
    def _supported_datasets():
        return ['PASCAL_VOC_rectangles', 'KITTI_rectangles']

    @classmethod
    def from_model(cls, emd_path, data=None):

        """
        Creates a Single Shot Detector from an Esri Model Definition (EMD) file.
        
        Note: Only supported for Pytorch models.

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
        
        :returns: `SingleShotDetector` Object
        """
        return cls.from_emd(data, emd_path)

    @classmethod
    def from_emd(cls, data, emd_path):

        """
        Creates a Single Shot Detector from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from `prepare_data` function or None for
                                inferencing.
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        =====================   ===========================================
        
        :returns: `SingleShotDetector` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
            
        emd_path = _get_emd_path(emd_path)
        emd = json.load(open(emd_path))
        model_file = Path(emd['ModelFile'])
        backbone = emd.get('backbone', 'resnet34')
        ssd_version = int(emd.get('SSDVersion', 1))
        chip_size = emd["ImageWidth"]

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i['Value']: i['Name'] for i in emd['Classes']}
        
        resize_to = emd.get('resize_to')
        if isinstance(resize_to, list):
            resize_to = (resize_to[0], resize_to[1])

        # Tensorflow support        
        backend = emd.get("ModelParameters", {}).get('backend', 'pytorch')
        if backend == 'tensorflow':
            backbone = emd["ModelParameters"].get("backbone", "ResNet50")

        data_passed = True
        # Create an image databunch for when loading the model using emd (without training data)
        if data is None:
            data_passed = False
            train_tfms = []
            val_tfms = []
            ds_tfms = (train_tfms, val_tfms)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                
                sd = ImageList([], path=emd_path.parent.parent.parent).split_by_idx([])
                data = sd.label_const(0, label_cls=ObjectDetectionCategoryList, classes=list(class_mapping.values())).transform(ds_tfms).databunch(device=_get_device()).normalize(imagenet_stats)

            data.chip_size = chip_size
            data.class_mapping = class_mapping
            data.classes = ['background'] + list(class_mapping.values())
            data._is_empty = True
            # Add 1 for background class
            data.c += 1
            data.emd_path = emd_path
            data.emd = emd 
            data = get_multispectral_data_params_from_emd(data, emd)

        data.resize_to = resize_to

        ssd = cls(data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=str(model_file), backend=backend, backbone=backbone, ssd_version=ssd_version)

        if not data_passed:
            ssd.learn.data.single_ds.classes = ssd._data.classes
            ssd.learn.data.single_ds.y.classes = ssd._data.classes
        
        return ssd

    def _create_anchors(self, anc_grids, anc_zooms, anc_ratios):

        self.grids = anc_grids
        self.zooms = anc_zooms
        self.ratios = anc_ratios

        anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i,j) in anc_ratios]

        self._anchors_per_cell = len(anchor_scales)

        anc_offsets = [1/(o*2) for o in anc_grids]

        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), self._anchors_per_cell, axis=0)

        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])

        self._grid_sizes = torch.Tensor(np.concatenate([np.array([ 1/ag  for i in range(ag*ag) for o,p in anchor_scales])
                       for ag in anc_grids])).unsqueeze(1).to(self._device)

        self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self._device)

        self._anchor_cnr = self._hw2corners(self._anchors[:,:2], self._anchors[:,2:])

    def _hw2corners(self, ctr, hw):
        return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

    def _get_y(self, bbox, clas):
        try:
            bbox = bbox.view(-1, 4)  # /sz
        except Exception:
            bbox = torch.zeros(size=[0, 4])
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def _actn_to_bb(self, actn, anchors, grid_sizes):
        actn_bbs = torch.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
        return self._hw2corners(actn_centers, actn_hw)

    def _map_to_ground_truth(self, overlaps, print_it=False):
        prior_overlap, prior_idx = overlaps.max(1)
        if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap, gt_idx

    def _ssd_1_loss(self, b_c, b_bb, bbox, clas, print_it=False):
        bbox,clas = self._get_y(bbox,clas)
        bbox = self._normalize_bbox(bbox)

        a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
        overlaps = self._jaccard(bbox.data, self._anchor_cnr.data)
        try:
            gt_overlap,gt_idx = self._map_to_ground_truth(overlaps,print_it)
        except Exception as e:
            return torch.tensor(0., requires_grad=True).to(self._device), torch.tensor(0., requires_grad=True).to(self._device)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[~pos] = 0
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss = self._loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def _ssd_loss(self, pred, targ1, targ2, print_it=False):
        lcs, lls = 0., 0.
        for b_c,b_bb,bbox,clas in zip(*pred, targ1, targ2):
            loc_loss, clas_loss = self._ssd_1_loss(b_c, b_bb,bbox.to(self._device), clas.to(self._device), print_it)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print('loc: {lls}, clas: {lcs}'.format(lls=lls, lcs=lcs))
        if self.location_loss_factor is None:
            return lls + lcs
        else:
            return self.location_loss_factor * lls + (1 - self.location_loss_factor) * lcs

    def _intersect(self,box_a, box_b):
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _box_sz(self, b):
        return (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])

    def _jaccard(self, box_a, box_b):
        inter = self._intersect(box_a, box_b)
        union = self._box_sz(box_a).unsqueeze(1) + self._box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def _normalize_bbox(self, bbox):
        return (bbox+1.)/2.

    @property
    def _model_metrics(self):
        return {'average_precision_score': self.average_precision_score(show_progress=True)}

    def _analyze_pred(self, pred, thresh=0.5, nms_overlap=0.1, ret_scores=True, device=None):
        return postprocess(pred, model=self, thresh=thresh, nms_overlap=nms_overlap, ret_scores=ret_scores, device=device)

    def _get_emd_params(self, save_inference_file):
        import random
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISObjectDetector.py"
        else:
            _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectDetector.py"
        _emd_template["ModelConfiguration"] = "_DynamicSSD"
        _emd_template["ModelType"] = "ObjectDetection"
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template['backbone'] = self._backbone.__name__        
        if _emd_template['backbone'] == 'backbone_wrapper':
            _emd_template['backbone'] = self._orig_backbone.__name__
        _emd_template['Grids'] = self.grids
        _emd_template['Zooms'] = self.zooms
        _emd_template['Ratios'] = self.ratios
        _emd_template['SSDVersion'] = self.ssd_version
        _emd_template['Classes'] = []

        class_data = {}
        for i, class_name in enumerate(self._data.classes[1:]):  # 0th index is background
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template['Classes'].append(class_data.copy())

        return _emd_template

    def _get_tfonnx_emd_params(self):
        return {"ModelConfiguration": "_SSDTensorflow"}

    def show_results(self, rows=5, thresh=0.5, nms_overlap=0.1):

        """
        Displays the results of a trained model on a part of the validation set.
        """
        self._check_requisites()
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        self.learn.show_results(rows=rows, thresh=thresh, nms_overlap=nms_overlap, model=self)
        if _IS_ARCGISPRONOTEBOOK:
            plt.show()

    def _show_results_multispectral(self, rows=5, thresh=0.3, nms_overlap=0.1, alpha=1, **kwargs):
        return_fig = kwargs.get('return_fig', False)
        ret_val = show_results_multispectral(
            self, 
            nrows=rows, 
            thresh=thresh, 
            nms_overlap=nms_overlap, 
            alpha=alpha, 
            **kwargs
        )
        if return_fig:
            fig, ax = ret_val
            return fig

    def predict_video(
        self,
        input_video_path,
        metadata_file,
        threshold=0.5,
        nms_overlap=0.1,
        track=False,
        visualize=False,
        output_file_path=None,
        multiplex=False,
        multiplex_file_path=None,
        tracker_options={
            'assignment_iou_thrd': 0.3,
            'vanish_frames': 40,
            'detect_frames': 10
        },
        visual_options={
            'show_scores': True,
            'show_labels': True,
            'thickness': 2,
            'fontface': 0,
            'color': (255, 255, 255)
        },
        resize=False
    ):

        """
        Runs prediction on a video and appends the output VMTI predictions in the metadata file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        input_video_path        Required. Path to the video file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        metadata_file           Required. Path to the metadata csv file where
                                the predictions will be saved in VMTI format.
        ---------------------   -------------------------------------------
        threshold               Optional float. The probability above which
                                a detection will be considered.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        track                   Optional bool. Set this parameter as True to
                                enable object tracking. 
        ---------------------   -------------------------------------------
        visualize               Optional boolean. If True a video is saved
                                with prediction results.
        ---------------------   -------------------------------------------
        output_file_path        Optional path. Path of the final video to be saved.
                                If not supplied, video will be saved at path input_video_path
                                appended with _prediction.
        ---------------------   -------------------------------------------
        multiplex               Optional boolean. Runs Multiplex using the VMTI detections.
        ---------------------   -------------------------------------------
        multiplex_file_path     Optional path. Path of the multiplexed video to be saved.
                                By default a new file with _multiplex.MOV extension is saved
                                in the same folder.
        ---------------------   -------------------------------------------
        tracking_options        Optional dictionary. Set different parameters for
                                object tracking. assignment_iou_thrd parameter is used
                                to assign threshold for assignment of trackers, 
                                vanish_frames is the number of frames the object should
                                be absent to consider it as vanished, detect_frames 
                                is the number of frames an object should be detected
                                to track it.
        ---------------------   -------------------------------------------
        visual_options          Optional dictionary. Set different parameters for
                                visualization.
                                show_scores boolean, to view scores on predictions,
                                show_labels boolean, to view labels on predictions,
                                thickness integer, to set the thickness level of box,
                                fontface integer, fontface value from opencv values,
                                color tuple (B, G, R), tuple containing values between
                                0-255.
        ---------------------   -------------------------------------------
        resize                  Optional boolean. Resizes the video frames to the same size
                                (chip_size parameter in prepare_data) that the model was trained on,
                                before detecting objects.
                                Note that if resize_to parameter was used in prepare_data,
                                the video frames are resized to that size instead.

                                By default, this parameter is false and the detections are run
                                in a sliding window fashion by applying the model on cropped sections
                                of the frame (of the same size as the model was trained on).
        =====================   ===========================================
        
        """

        VideoUtils.predict_video(
            self,
            input_video_path,
            metadata_file,
            threshold,
            nms_overlap,
            track, visualize,
            output_file_path,
            multiplex,
            multiplex_file_path,
            tracker_options,
            visual_options,
            resize
        )

    def predict(
        self,
        image_path,
        threshold=0.5,
        nms_overlap=0.1,
        return_scores=False,
        visualize=False,
        resize=False
    ):

        """
        Runs prediction on an Image.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        image_path              Required. Path to the image file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        threshold               Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        return_scores           Optional boolean. Will return the probability
                                scores of the bounding box predictions if True.
        ---------------------   -------------------------------------------
        visualize               Optional boolean. Displays the image with
                                predicted bounding boxes if True.
        ---------------------   -------------------------------------------
        resize                  Optional boolean. Resizes the image to the same size
                                (chip_size parameter in prepare_data) that the model was trained on,
                                before detecting objects.
                                Note that if resize_to parameter was used in prepare_data,
                                the image is resized to that size instead.

                                By default, this parameter is false and the detections are run
                                in a sliding window fashion by applying the model on cropped sections
                                of the image (of the same size as the model was trained on).
        =====================   ===========================================
        
        :returns: 'List' of xmin, ymin, width, height of predicted bounding boxes on the given image
        """
        if not HAS_OPENCV:
            raise Exception("This function requires opencv 4.0.1.24. Install it using pip install opencv-python==4.0.1.24")


        if isinstance(image_path, str):
            #
            if self._data._is_multispectral:
                resize_to = None
                if resize:
                    if self._data.resize_to is not None:
                        resize_to = self._data.resize_to
                    elif self._data.chip_size is not None:
                        resize_to = self._data.chip_size
                image = read_image(image_path, resize_to)
            else:
                image = cv2.imread(image_path)
        else:
            image = image_path

        orig_height, orig_width, _ = image.shape
        orig_frame = image.copy()

        if not self._data._is_multispectral:
            if resize and self._data.resize_to is None\
                    and self._data.chip_size is not None:
                image = cv2.resize(image, (self._data.chip_size, self._data.chip_size))

            if self._data.resize_to is not None:
                if isinstance(self._data.resize_to, tuple):
                    image = cv2.resize(image, self._data.resize_to)
                else:
                    image = cv2.resize(image, (self._data.resize_to, self._data.resize_to))

        height, width, _ = image.shape

        if self._data.chip_size is not None:
            chips = _get_image_chips(image, self._data.chip_size)
        else:
            chips = [{'width': width, 'height': height, 'xmin': 0, 'ymin': 0, 'chip': image, 'predictions': []}]

        include_pad_detections = False
        if len(chips) == 1:
            include_pad_detections = True

        valid_tfms = self._data.valid_ds.tfms
        self._data.valid_ds.tfms = []

        for chip in chips:
            if self._data._is_multispectral:
                t = torch.tensor(np.rollaxis(chip['chip'], -1, 0).astype(np.float32), dtype=torch.float32)[None]
                scaled_t = self._data._min_max_scaler(t)[0]
                frame = Image(scaled_t[self._data._extract_bands])
            else:
                frame = Image(pil2tensor(PIL.Image.fromarray(cv2.cvtColor(chip['chip'], cv2.COLOR_BGR2RGB)), dtype=np.float32).div_(255))
            bbox = self.learn.predict(frame, thresh=threshold, nms_overlap=nms_overlap, ret_scores=True, model=self)[0]
            if bbox:
                scores = bbox.scores
                bboxes, lbls = bbox._compute_boxes()
                bboxes.add_(1).mul_(torch.tensor([chip['height'] / 2, chip['width'] / 2, chip['height'] / 2, chip['width'] / 2])).long()
                for index, bbox in enumerate(bboxes):
                    if lbls is not None:
                        label = lbls[index]
                    else:
                        label = 'Default'

                    data = bb2hw(bbox)
                    if include_pad_detections or not _exclude_detection((data[0], data[1], data[2], data[3]), chip['width'], chip['height']):
                        chip['predictions'].append({
                            'xmin': data[0],
                            'ymin': data[1],
                            'width': data[2],
                            'height': data[3],
                            'score': float(scores[index]),
                            'label': label
                        })

        self._data.valid_ds.tfms = valid_tfms

        predictions, labels, scores = _get_transformed_predictions(chips)

        y_ratio = orig_height/height
        x_ratio = orig_width/width

        for index, prediction in enumerate(predictions):
            prediction[0] = prediction[0]*x_ratio
            prediction[1] = prediction[1]*y_ratio
            prediction[2] = prediction[2]*x_ratio
            prediction[3] = prediction[3]*y_ratio

            # Clip xmin
            if prediction[0] < 0: 
                prediction[2] = prediction[2] + prediction[0]
                prediction[0] = 1

            # Clip width when xmax greater than original width
            if prediction[0] + prediction[2] > orig_width:
                prediction[2] = (prediction[0] + prediction[2]) - orig_width

            # Clip ymin
            if prediction[1] < 0:
                prediction[3] = prediction[3] + prediction[1]
                prediction[1] = 1

            # Clip height when ymax greater than original height
            if prediction[1] + prediction[3] > orig_height:
                prediction[3] = (prediction[1] + prediction[3]) - orig_height

            predictions[index] = [
                prediction[0],
                prediction[1],
                prediction[2],
                prediction[3]
            ]      

        if visualize:
            if self._data._is_multispectral:
                t = torch.tensor(np.rollaxis(orig_frame, -1, 0).astype(np.float32), dtype=torch.float32)[None]
                # im = PIL.Image.fromarray(orig_frame)
                # t = pil2tensor(im, dtype=np.float32)[None]
                scaled_t = self._data._min_max_scaler(t)[0]
                orig_frame = (scaled_t*255).round().numpy().astype(np.uint8)[self._data._symbology_rgb_bands]
                orig_frame = np.rollaxis(orig_frame, 0, 3)
                a = np.zeros(orig_frame.shape, dtype=np.uint8)
                a[:] = orig_frame[:]
                if len(labels) > 0:
                    image = _draw_predictions(a, predictions, labels)
                else:
                    image = orig_frame
            else:
                image = _draw_predictions(orig_frame, predictions, labels)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            import matplotlib.pyplot as plt
            plt.xticks([])
            plt.yticks([])
            plt.imshow(PIL.Image.fromarray(image))

        if return_scores:
            return predictions, labels, scores
        else:
            return predictions, labels

    def average_precision_score(self, detect_thresh=0.2, iou_thresh=0.1, mean=False, show_progress=True):

        """
        Computes average precision on the validation set for each class.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered for computing
                                average precision.
        ---------------------   -------------------------------------------
        iou_thresh              Optional float. The intersection over union
                                threshold with the ground truth labels, above
                                which a predicted bounding box will be
                                considered a true positive.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.                        
        =====================   ===========================================
        
        :returns: `dict` if mean is False otherwise `float`
        """
        self._check_requisites()

        aps = compute_class_AP(self, self._data.valid_dl, self._data.c - 1, show_progress, detect_thresh=detect_thresh, iou_thresh=iou_thresh)
        if mean:
            import statistics
            return statistics.mean(aps)
        else:
            return dict(zip(self._data.classes[1:], aps))
    
    ## Tensorflow specific functions start ##
    def _intialize_tensorflow(self, data, grids, zooms, ratios, backbone, drop, bias, pretrained_path, location_loss_factor):
        self._check_tf()
        
        from .._utils.fastai_tf_fit import TfLearner
        import tensorflow as tf
        from tensorflow.keras.losses import BinaryCrossentropy
        from tensorflow.keras.models import Model
        from tensorflow.keras import applications
        from tensorflow.keras.optimizers import Adam
        from fastai.basics import defaults
        from .._utils.object_detection import get_ssd_head_output

        if data._is_multispectral:
            raise Exception('Multispectral data is not supported with backend="tensorflow"')

        # prepare color array
        alpha = 0.7
        color_mapping = getattr(data, 'color_mapping', None)
        if color_mapping is None:
            color_array = torch.tensor([[1., 1., 1.]]).float()
        else:
            color_array = torch.tensor( list(color_mapping.values()) ).float() / 255
        alpha_tensor = torch.tensor( [alpha]*len(color_array) ).view(-1, 1).float()
        color_array = torch.cat( [ color_array, alpha_tensor ], dim=-1)
        background_color = torch.tensor( [[0, 0, 0, 0]] ).float()
        data._multispectral_color_array = torch.cat( [background_color, color_array] )

        self.ssd_version = 1#ssd_version
        if backbone is None:
            backbone = 'ResNet50'
        
        if type(backbone) == str:
            backbone = getattr(applications, backbone)
            
        self._backbone = backbone 
        
        x, y = next(iter(data.train_dl))
        if tf.keras.backend.image_data_format() == 'channels_last':
            in_shape = [x.shape[-1], x.shape[-1], 3]
        else:
            in_shape = [3, x.shape[-1],x.shape[-1]]

        self._backbone_initalized = self._backbone(
            input_shape=in_shape, 
            include_top=False, 
            #weights='imagenet'
        )
        self._backbone_initalized.trainable = False

        self._device = torch.device('cpu')
        self._data = data

        #self._loss_function_classification = BinaryCrossentropy(from_logits=True, reduction=Reduction.SUM) #2.0.0
        self._loss_function_classification = BinaryCrossentropy(from_logits=True, reduction='sum')
        self.location_loss_factor = location_loss_factor
        
        if grids is None:
            # find most suitable centroids for dataset
            height_width = np.array(data.height_width)
            centroid = kmeans(height_width , 1) 
            avg = avg_iou(height_width, centroid)

            for num_anchor in range(2, 5):
                new_centroid = kmeans(height_width, num_anchor)
                new_avg = avg_iou(height_width, new_centroid)
                if (new_avg - avg) < 0.05:
                    break
                avg = new_avg
                centroid = new_centroid.copy()

            # find grid size
            grids = list(map(int, map(round, data.chip_size/np.sort(np.max(centroid, axis=1)))))

            grids = list(set(grids))
            grids.sort(reverse = True)
            if grids[-1] == 0:
                grids[-1] = 1
            grids = list(set(grids))

        self.grids = grids
        self.zooms = zooms
        self.ratios = ratios

        self._create_anchors(grids, zooms, ratios)

        output_layer = get_ssd_head_output(self)

        model = Model(inputs=self._backbone_initalized.input, outputs=output_layer)

        self.learn = TfLearner(
            data, 
            model,
            opt_func=Adam,
            loss_func=self._loss_func_tf,
            true_wd=True, 
            bn_wd=True, 
            wd=defaults.wd, 
            train_bn=True
        )
        
        self.learn.unfreeze()
        self.learn.freeze_to(len(self._backbone_initalized.layers))

        self.show_results = self._show_results_multispectral

        self._code = code
        if pretrained_path is not None:
            self.load(pretrained_path)


    def _loss_func_tf(self, y_bboxes, y_classes, predictions):
        from .._utils.object_detection import tf_loss_function_single_image
        import tensorflow as tf
        predicted_classes = predictions[0]
        predicted_bboxes = predictions[1]
        
        localization_loss, classification_loss = tf.constant(0.), tf.constant(0.)
        # Now we will iterate over each image and calculate loss for a single image
        for image_y_bboxes, image_y_calsses, image_p_bboxes, image_p_classes in zip(y_bboxes, y_classes, predicted_bboxes, predicted_classes):
            _localization_loss, _classification_loss = tf_loss_function_single_image(self, image_y_bboxes, image_y_calsses, image_p_bboxes, image_p_classes )
            classification_loss += _classification_loss
            localization_loss += _localization_loss

        if self.location_loss_factor is None:
            return localization_loss + classification_loss
        else:
            return self.location_loss_factor * localization_loss + (1 - self.location_loss_factor) * classification_loss
    
    ## Tensorflow specific functions end ##