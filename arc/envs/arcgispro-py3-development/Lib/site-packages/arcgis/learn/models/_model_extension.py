from pathlib import Path
import json
from ._arcgis_model import _EmptyData, _change_tail, ArcGISModel, _get_device
from ._codetemplate import code, image_classifier_prf
import warnings
import arcgis
import sys, os, importlib
from functools import partial
import logging
logger = logging.getLogger()

HAS_OPENCV = True
HAS_FASTAI = True
HAS_ARCPY = True

try:
    import torch
    from torch import nn
    import numpy as np
    from fastai.basic_train import Learner, LearnerCallback
    from fastai.torch_core import split_model_idx
    from fastai.vision import ImageList
    from fastai.vision import imagenet_stats, normalize
    from fastai.core import has_arg, split_kwargs_by_func
    from fastai.basic_data import DatasetType
    from fastai.callback import Callback
    from fastai.torch_core import to_cpu, grab_idx
    from fastai.basic_train import loss_batch
    from fastai.vision.image import open_image, bb2hw, image2np, Image, pil2tensor
    import PIL
    from ._ssd_utils import compute_class_AP
    from .._utils.pascal_voc_rectangles import show_results_multispectral, ObjectDetectionCategoryList
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from ._arcgis_model import _set_ddp_multigpu, _isnotebook
    from ._hed_utils import accuracies
    from .._image_utils import _get_image_chips, _get_transformed_predictions, _draw_predictions, _exclude_detection
    from .._video_utils import VideoUtils
    import inspect
    from .._utils.env import _IS_ARCGISPRONOTEBOOK
    from matplotlib import pyplot as plt
    HAS_FASTAI = True
    
except Exception as e:
    HAS_FASTAI = False

try:
    import cv2
except Exception:
    HAS_OPENCV = False

try:
    import arcpy
except Exception:
    HAS_ARCPY = False    

class ModelExtension(ArcGISModel):
    """
    Creates a ModelExtension object, to train the model for object detection, semantic segmentation, and edge detection.

    =====================   ============================================================
    **Argument**            **Description**
    ---------------------   ------------------------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            ``prepare_data`` function.
    ---------------------   ------------------------------------------------------------
    model_conf              A class definition contains the following methods:

                                * ``get_model(self, data, backbone=None)``: for model definition,
                                
                                * ``on_batch_begin(self, learn, model_input_batch, model_target_batch)``: for 
                                  feeding input to the model during training, 

                                * ``transform_input(self, xb)``: for feeding input to the model during
                                  inferencing/validation,

                                * ``transform_input_multispectral(self, xb)``: for feeding input to the
                                  model during inferencing/validation in case of multispectral data,

                                * ``loss(self, model_output, *model_target)``: to return loss value of the model, and 

                                * ``post_process(self, pred, nms_overlap, thres, chip_size, device)``: to post-process
                                  the output of the object-detection model.

                                * ``post_process(self, pred, thres)``: to post-process the output of the segmentation model.
                                  
    ---------------------   ------------------------------------------------------------
    backbone                Optional function. If custom model requires any backbone.
    ---------------------   ------------------------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ============================================================

    :return: ``ModelExtension`` Object
    """

    def __init__(self, data, model_conf, backbone=None, pretrained_path=None, **kwargs):

        super().__init__(data, backbone, **kwargs)
        self.model_conf = model_conf()
        self.model_conf_class  = model_conf
        self._backend = 'pytorch'
        self._kwargs = kwargs
        model = self.model_conf.get_model(data, backbone, **kwargs)
        if self._is_multispectral:
            model = _change_tail(model, data)
        if not _isnotebook() and os.name=='posix':
            _set_ddp_multigpu(self)
            if self._multigpu_training:
                self.learn = Learner(data, model, loss_func=self.model_conf.loss).to_distributed(self._rank_distributed)
            else:
                self.learn = Learner(data, model, loss_func=self.model_conf.loss)
        else:
            self.learn = Learner(data, model, loss_func=self.model_conf.loss)
        self.learn.callbacks.append(self._train_callback(self.learn, self.model_conf.on_batch_begin))
        if self._data.dataset_type == 'Classified_Tiles':
            if getattr(self, "_is_edge_detection", False):
                from ._hed_utils import accuracy, f1_score
                self.learn.metrics = [accuracy, f1_score]
            else:
                from ._psp_utils import accuracy
                self.learn.metrics = [accuracy]
            self._code = image_classifier_prf
        else:
            self._code = code
        self._arcgis_init_callback() # make first conv weights learnable
        self._bind_dataset_methods()
        if pretrained_path is not None:
            self.load(pretrained_path)

    if HAS_FASTAI:
        class _train_callback(LearnerCallback):

            def __init__(self, learn, on_batch_begin_fn):
                super().__init__(learn)
                self.on_batch_begin_fn = on_batch_begin_fn

            def on_batch_begin(self, last_input, last_target, train, **kwargs):

                last_input, last_target = self.on_batch_begin_fn(self.learn, last_input, last_target)

                return {'last_input':last_input, 'last_target':last_target}

    def _analyze_pred(self, pred, thresh=0.5, nms_overlap=0.1, ret_scores=True, device=None):
        return self.model_conf.post_process(pred, nms_overlap, thresh, self.learn.data.chip_size, device)
       
    def _get_emd_params(self, save_inference_file):
        import random
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        if self._data.dataset_type == 'Classified_Tiles':
            _emd_template["ModelType"] = "ImageClassification"
            if save_inference_file:
                _emd_template["InferenceFunction"] = "ArcGISImageClassifier.py"
            else:
                _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISImageClassifier.py"
            _emd_template['IsEdgeDetection'] = getattr(self, "_is_edge_detection", False)
        else:
            _emd_template["ModelType"] = "ObjectDetection"
            if save_inference_file:
                _emd_template["InferenceFunction"] = "ArcGISObjectDetector.py"
            else:
                _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectDetector.py"
        _emd_template["ModelConfiguration"] = "_model_extension_inferencing"
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template['Classes'] = []
        _emd_template['ModelConfigurationFile'] = "ModelConfiguration.py"
        _emd_template['ModelFileConfigurationClass'] = type(self.model_conf).__name__
        _emd_template['DatasetType'] = self._data.dataset_type
        _emd_template['Kwargs'] = self._kwargs

        class_data = {}
        for i, class_name in enumerate(self._data.classes[1:]):  # 0th index is background
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)]
            class_data["Color"] = color
            _emd_template['Classes'].append(class_data.copy())

        return _emd_template

    @property
    def _is_model_extension(self):
        return True

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a ``ModelExtension`` object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from ``prepare_data`` function or None for
                                inferencing.

        =====================   ===========================================

        :returns: `ModelExtension` Object
        """

        emd_path = _get_emd_path(emd_path)

        with open(emd_path) as f:
            emd = json.load(f)
            
        model_file = Path(emd['ModelFile'])
        
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file
        
        modelconf = Path(emd['ModelConfigurationFile'])

        if not modelconf.is_absolute():
            modelconf = emd_path.parent / modelconf

        modelconfclass = emd['ModelFileConfigurationClass']

        sys.path.append(os.path.dirname(modelconf))
        model_configuration = getattr(importlib.import_module('{}'.format(modelconf.name[0:-3])), modelconfclass)

        backbone = emd['ModelParameters']['backbone']
        dataset_type = emd.get('DatasetType', 'PASCAL_VOC_rectangles')
        chip_size = emd["ImageWidth"]
        resize_to = emd.get('resize_to', None)
        kwargs = emd.get('Kwargs', {})
        if isinstance(resize_to, list):
            resize_to = (resize_to[0], resize_to[1])

        try:
            class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
            color_mapping = {i['Value'] : i['Color'] for i in emd['Classes']}
        except KeyError:
            class_mapping = {i['ClassValue'] : i['ClassName'] for i in emd['Classes']} 
            color_mapping = {i['ClassValue'] : i['Color'] for i in emd['Classes']}                

        data_passed = True
        if data is None:           

            data_passed = False
            if dataset_type == 'PASCAL_VOC_rectangles':
                train_tfms = []
                val_tfms = []
                ds_tfms = (train_tfms, val_tfms)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    sd = ImageList([], path=emd_path.parent.parent).split_by_idx([])
                    data = sd.label_const(0, label_cls=ObjectDetectionCategoryList, classes=list(class_mapping.values())).transform(ds_tfms).databunch(device=_get_device()).normalize(imagenet_stats)
                # Add 1 for background class
                data.c += 1
            else:
                data = _EmptyData(path=emd_path.parent.parent, loss_func=None, c=len(class_mapping) + 1, chip_size=emd['ImageHeight'])
            
            data.chip_size = chip_size
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data.classes = ['background'] + list(class_mapping.values())
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data = get_multispectral_data_params_from_emd(data, emd)
            data.dataset_type = dataset_type

        data.resize_to = resize_to
        mextnsn = cls(data, model_configuration, backbone, pretrained_path=str(model_file), **kwargs)

        if not data_passed and dataset_type == 'PASCAL_VOC_rectangles':
            mextnsn.learn.data.single_ds.classes = mextnsn._data.classes
            mextnsn.learn.data.single_ds.y.classes = mextnsn._data.classes

        return mextnsn

    @property
    def _model_metrics(self):
        if self._data.dataset_type == 'Classified_Tiles':
            return {'accuracy': '{0:1.4e}'.format(self._get_model_metrics())}
        else:
            return {'average_precision_score': self.average_precision_score(show_progress=False)}

    def _get_model_metrics(self, **kwargs):
        checkpoint = kwargs.get('checkpoint', True)
        if not hasattr(self.learn, 'recorder'):
            return 0.0
        model_accuracy = self.learn.recorder.metrics[-1][0]
        if checkpoint:
            model_accuracy = np.max(self.learn.recorder.metrics)
        return float(model_accuracy)

    def _get_y(self, bbox, clas):
        try:
            bbox = bbox.view(-1, 4)
        except Exception:
            bbox = torch.zeros(size=[0, 4])
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

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

    def _bind_dataset_methods(self):

        if self._data.dataset_type == 'Classified_Tiles':
            if getattr(self, "_is_edge_detection", False):
                self.show_results = self._show_results_edge_detection
                self.compute_precision_recall = self._edge_detection_accuracies
            else:
                self.show_results = self._show_results_segmentation
        else:
            if self._is_multispectral:
                self.show_results = self._show_results_multispectral
            else:
                self.show_results = self._show_results_object_detection
            self.average_precision_score = self._average_precision_score
            self.predict = self._predict
            self.predict_video = self._predict_video

    def _show_results_object_detection(self, rows=5, thresh=0.5, nms_overlap=0.1):

        """
        Displays the results of a trained model on a part of the validation set.
        """
        self._check_requisites()
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        self._show_results_modified(rows=rows, thresh=thresh, nms_overlap=nms_overlap, model=self)

    def _show_results_segmentation(self, rows=5, thresh=0.5, **kwargs):

        """
        Displays the results of a trained model on a part of the validation set.
        """
        self._check_requisites()
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        self._show_results_modified(rows=rows, thresh=thresh, model=self, **kwargs)

    def _show_results_edge_detection(self, rows=5, thresh=0.5, thinning=True,**kwargs):

        """
        Displays the results of a trained model on a part of the validation set.
        """
        self._check_requisites()
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        self._show_results_modified(rows=rows, thresh=thresh, model=self, thinning=thinning, **kwargs)

    def _show_results_multispectral(self, rows=5, thresh=0.3, nms_overlap=0.1, alpha=1, **kwargs):
        return_fig = kwargs.get("return_fig", False)
        ax = show_results_multispectral(
            self,
            nrows=rows,
            thresh=thresh,
            nms_overlap=nms_overlap,
            alpha=alpha,
            **kwargs
        )
        if return_fig:
            fig,axs=ax
            return fig

    def _show_results_modified(self, rows=5, **kwargs):

        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        ds_type = DatasetType.Valid
        n_items = rows ** 2 if self.learn.data.train_ds.x._square_show_res else rows
        if self.learn.dl(ds_type).batch_size < n_items: n_items = self.learn.dl(ds_type).batch_size
        ds = self.learn.dl(ds_type).dataset
        xb,yb = self.learn.data.one_batch(ds_type, detach=False, denorm=False)
        self.learn.model.eval()
        transform_kwargs, kwargs = split_kwargs_by_func(kwargs, self.model_conf.transform_input)
        try:
            preds = self.learn.model(self.model_conf.transform_input(xb, **transform_kwargs))
        except Exception as e:

            if getattr(self, "_is_fasterrcnn", False):
                preds = []
                for _ in range(xb.shape[0]):
                    res={}
                    res['boxes'] = torch.empty(0,4)
                    res['scores'] = torch.tensor([])
                    res['labels'] = torch.tensor([])
                    preds.append(res)
            else:
                raise e

        x,y = to_cpu(xb),to_cpu(yb)
        norm = getattr(self.learn.data,'norm',False)
        if norm:
            x = self.learn.data.denorm(x)
            if norm.keywords.get('do_y',False):
                y     = self.learn.data.denorm(y, do_x=True)
                preds = self.learn.data.denorm(preds, do_x=True)
        analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        preds = ds.y.analyze_pred(preds,**analyze_kwargs)
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        if has_arg(ds.y.reconstruct, 'x'):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
        else:
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
            zs = [ds.y.reconstruct(z) for z in preds]
        ds.x.show_xyzs(xs, ys, zs, **kwargs)
        if _IS_ARCGISPRONOTEBOOK:
            plt.show()

    def _predict_learn_modified(self, item, **kwargs):
        "Return predicted class, label and probabilities for `item`."
        batch = self.learn.data.one_item(item)
        transform_kwargs, kwargs = split_kwargs_by_func(kwargs, self.model_conf.transform_input)
        self.learn.model.eval()
        try:
            pred = self.learn.model(self.model_conf.transform_input(batch[0], **transform_kwargs))
        except Exception as e:

            if getattr(self, "_is_fasterrcnn", False):
                pred = []
                for _ in range(batch[0].shape[0]):
                    res={}
                    res['boxes'] = torch.empty(0,4)
                    res['scores'] = torch.tensor([])
                    res['labels'] = torch.tensor([])
                    pred.append(res)
            else:
                raise e
        ds = self.learn.data.single_ds
        analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        pred = ds.y.analyze_pred(pred, **analyze_kwargs)
        x = batch[0]
        norm = getattr(self.learn.data,'norm',False)
        if norm:
            x = self.learn.data.denorm(x)
        x = ds.x.reconstruct(grab_idx(x, 0))
        y = ds.y.reconstruct(pred[0], x) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred[0])
        return y

    def _average_precision_score(self, detect_thresh=0.2, iou_thresh=0.1, mean=False, show_progress=True):

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

        aps = compute_class_AP(self,
                               self._data.valid_dl, 
                               self._data.c - 1, 
                               show_progress, 
                               iou_thresh=iou_thresh, 
                               detect_thresh=detect_thresh, 
                               thresh=detect_thresh, 
                               nms_overlap=iou_thresh)
        if mean:
            import statistics
            return statistics.mean(aps)
        else:
            return dict(zip(self._data.classes[1:], aps))

    def _edge_detection_accuracies(self, thresh=0.5, buffer=3, show_progress=True):
        """
        Computes precision, recall and f1 score on validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered edge pixel.
        ---------------------   -------------------------------------------
        buffer                  Optional int. pixels in neighborhood to
                                consider true detection.
        =====================   ===========================================

        :returns: `dict` 
        """
        self._check_requisites()
        acc = accuracies(self, self._data.valid_dl, detect_thresh=thresh, buffer=buffer, show_progress=show_progress)
        return acc

    def _predict(
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
        
        :returns:  Returns a tuple with predictions, labels and optionally confidence scores
                   if return_scores=True. The predicted bounding boxes are returned as a list
                   of lists containing the  xmin, ymin, width and height of each predicted object
                   in each image. The labels are returned as a list of class values and the
                   confidence scores are returned as a list of floats indicating the confidence
                   of each prediction.
        """
        if not HAS_OPENCV:
            raise Exception("This function requires opencv 4.0.1.24. Install it using pip install opencv-python==4.0.1.24")

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        orig_height, orig_width, _ = image.shape
        orig_frame = image.copy()

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
            frame = Image(pil2tensor(PIL.Image.fromarray(cv2.cvtColor(chip['chip'], cv2.COLOR_BGR2RGB)), dtype=np.float32).div_(255))
            bbox = self._predict_learn_modified(frame, thresh=threshold, nms_overlap=nms_overlap, ret_scores=True, model=self)
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
            image = _draw_predictions(orig_frame, predictions, labels)
            import matplotlib.pyplot as plt
            plt.xticks([])
            plt.yticks([])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(PIL.Image.fromarray(image))

        if return_scores:
            return predictions, labels, scores
        else:
            return predictions, labels


    def _predict_video(
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