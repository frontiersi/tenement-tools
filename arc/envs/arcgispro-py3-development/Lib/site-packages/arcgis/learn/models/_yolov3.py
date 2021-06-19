# necessary imports
import os
import json
import random
import warnings
import traceback
import statistics
from pathlib import Path
from ._codetemplate import code
from .._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel, _get_device

HAS_OPENCV = True
HAS_FASTAI = True
HAS_PIL = True

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from fastai.vision import ImageList
    from fastai.basic_train import Learner
    from fastai.vision import imagenet_stats, normalize
    from fastai.vision.image import bb2hw, Image, pil2tensor
    from .._utils.pascal_voc_rectangles import ObjectDetectionCategoryList, show_results_multispectral
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from .._utils.utils import extract_zipfile
    from ._yolov3_utils import YOLOv3_Model, YOLOv3_Loss, AppendLabelsCallback, generate_anchors, compute_class_AP
    from ._yolov3_utils import download_yolo_weights, parse_yolo_weights, postprocess, coco_config, coco_class_mapping
    from .._image_utils import _get_image_chips, _get_transformed_predictions, _draw_predictions, _exclude_detection
    from .._video_utils import VideoUtils
    from .._utils.env import _IS_ARCGISPRONOTEBOOK
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False

try:
    import cv2
except ImportError:
    HAS_OPENCV = False

try:
    import PIL
except ImportError:
    HAS_PIL = False

# Yolov3 model
class YOLOv3(ArcGISModel):
    """
    Creates a YOLOv3 object detector.
    
    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================
    
    :returns: `YOLOv3` Object
    """

    def __init__(self, data=None, pretrained_path=None, **kwargs):

        self._check_dataset_support(data)

        if data is None:
            data = create_coco_data()
        else:
            #Removing normalization because YOLO ingests images with values in range 0-1
            data.remove_tfm(data.norm)
            data.norm, data.denorm = None, None

        super().__init__(data)

        #Creating a dummy class for the backbone because this model does not use a torchvision backbone
        class DarkNet53():
            def __init__(self):
                self.name = "DarkNet53"
        self._backbone = DarkNet53

        self._code = code
        self._data = data

        self.config_model = {}
        if getattr(data, "_is_coco", "") == True:
            self.config_model = coco_config()
        else:
            anchors = kwargs.get('anchors', None)
            self.config_model['ANCHORS'] = anchors if anchors is not None else generate_anchors(num_anchor=9, hw=data.height_width)
            self.config_model['ANCH_MASK'] = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            self.config_model['N_CLASSES'] = data.c - 1 # Subtract 1 for the background class
            n_bands = kwargs.get('n_bands', None)
            self.config_model['N_BANDS'] = n_bands if n_bands is not None else data.x[0].data.shape[0]

        self._model = YOLOv3_Model(self.config_model)

        pretrained = kwargs.get('pretrained_backbone', True)
        if pretrained:
            # Download (if required) and load YOLOv3 weights pretrained on COCO dataset
            weights_path = os.path.join(Path.home(), '.cache', 'weights')
            if not os.path.exists(weights_path): os.makedirs(weights_path)
            weights_file = os.path.join(weights_path, 'yolov3.weights')
            if not os.path.exists(weights_file):
                try:
                    download_yolo_weights(weights_path)
                    extract_zipfile(weights_path, 'yolov3.zip', remove=True)
                except Exception as e:
                    print (e)
                    print("[INFO] Can't download and extract COCO pretrained weights for YOLOv3.\nProceeding without pretrained weights.")
            if os.path.exists(weights_file):
                parse_yolo_weights(self._model, weights_file)
                from IPython.display import clear_output
                clear_output()

        self._loss_f = YOLOv3_Loss()
        self.learn = Learner(data, self._model, loss_func=self._loss_f)
        self.learn.split([self._model.module_list[11]]) #Splitting the model at Darknet53 backbone
        self.learn.freeze()

        if pretrained_path is not None:
            self.load(str(pretrained_path))

        # make first conv weights learnable and use _show_results_multispectral when using multispectral data
        self._arcgis_init_callback()

        # Set a default flag to toggle appending labels with images before passing images through the model
        self.learn.predicting = False
        self.learn.callbacks.append(AppendLabelsCallback(self.learn))
        

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)
    
    @property
    def supported_backbones(self):
        """ Supported backbones for this model. """
        return YOLOv3._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return ['DarkNet53']
    
    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return YOLOv3._supported_datasets()
    
    @staticmethod
    def _supported_datasets():
        return ['PASCAL_VOC_rectangles', 'KITTI_rectangles'] 

    @property
    def _model_metrics(self):
        if getattr(self._data, "_is_coco", "") == True:
            return {'accuracy': {'IoU': 0.50, 'AP': 0.558}}
        return {'accuracy': self.average_precision_score(show_progress=True)}

    def _analyze_pred(self, pred, thresh=0.1, nms_overlap=0.1, ret_scores=True, device=None):
        """        """
        return postprocess(pred, chip_size=self.learn.data.chip_size, conf_thre=thresh, nms_thre=nms_overlap)
    
    def show_results(self, rows=5, thresh=0.1, nms_overlap=0.1):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered valid. 
                                Defaults to 0.1. To be modified according 
                                to the dataset and training.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding 
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        =====================   ===========================================
        
        """
        self._check_requisites()

        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)

        self.learn.predicting = True
        self.learn.show_results(rows=rows, thresh=thresh, nms_overlap=nms_overlap, model=self)
        if _IS_ARCGISPRONOTEBOOK:
            import matplotlib.pyplot as plt
            plt.show()
    
    def _show_results_multispectral(self, rows=5, thresh=0.1, nms_overlap=0.1, alpha=1, **kwargs):
        return_fig = kwargs.get('return_fig', False)
        self.learn.predicting = True
        fig,ax = show_results_multispectral(
            self, 
            nrows=rows, 
            thresh=thresh, 
            nms_overlap=nms_overlap, 
            alpha=alpha, 
            **kwargs
        )
        self.learn.predicting = False # toggling the flag here because show_results_multispectral doesn't invoke callbacks

    def predict(self, image_path, threshold=0.1, nms_overlap=0.1, return_scores=True, visualize=False, resize=False):
        """
        Predicts and displays the results of a trained model on a single image.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        image_path              Required. Path to the image file to make the
                                predictions on.
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability above which
                                a detection will be considered valid. 
                                Defaults to 0.1. To be modified according 
                                to the dataset and training.
        ---------------------   -------------------------------------------
        nms_overlap             Optional float. The intersection over union
                                threshold with other predicted bounding 
                                boxes, above which the box with the highest
                                score will be considered a true positive.
        ---------------------   -------------------------------------------
        return_scores           Optional boolean.
                                Will return the probability scores of the 
                                bounding box predictions if True.
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

        if not HAS_PIL:
            raise Exception("This function requires PIL. Please install it via pip or conda")

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

        valid_tfms = self._data.valid_ds.tfms
        self._data.valid_ds.tfms = []

        include_pad_detections = False
        if len(chips) == 1:
            include_pad_detections = True

        for chip in chips:
            frame = Image(pil2tensor(PIL.Image.fromarray(cv2.cvtColor(chip['chip'], cv2.COLOR_BGR2RGB)), dtype=np.float32).div_(255))
            self.learn.predicting = True
            bbox = self.learn.predict(frame, thresh=threshold, nms_overlap=nms_overlap, ret_scores=True, model=self)[0]
            if bbox:
                scores = bbox.scores
                bboxes, lbls = bbox._compute_boxes()
                bboxes.add_(1).mul_(
                    torch.tensor([chip['height'] / 2, chip['width'] / 2, chip['height'] / 2, chip['width'] / 2])).long()
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

        # Scale the predictions to original image and clip the predictions to image dims
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
            image = _draw_predictions(orig_frame, predictions, labels, color=(255, 0, 0), fontface=2, thickness=1)
            import matplotlib.pyplot as plt
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if getattr(self._data, "_is_coco", "") == True: 
                figsize = (20,20)
            else:
                figsize = (4,4)
            fig, ax = plt.subplots(1,1, figsize=figsize)
            ax.imshow(image)

        if return_scores:
            return predictions, labels, scores
        else:
            return predictions, labels


    def predict_video(
        self,
        input_video_path,
        metadata_file,
        threshold=0.1,
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
                                a detection will be considered. Defaults to
                                0.1. To be modified according to the dataset
                                and training.
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

    def average_precision_score(self, detect_thresh=0.1, iou_thresh=0.1, mean=False, show_progress=True):
        """
        Computes average precision on the validation set for each class.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        detect_thresh           Optional float. The probability above which
                                a detection will be considered for computing
                                average precision. Defaults to 0.1. To be 
                                modified according to the dataset and training.
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
            return statistics.mean(aps)
        else:
            return dict(zip(self._data.classes[1:], aps))


    def _get_emd_params(self, save_inference_file):
        
        class_data = {}
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISObjectDetector.py"
        else:
            _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISObjectDetector.py"
        _emd_template["ModelConfiguration"] = "_yolov3_inference"
        _emd_template["ModelType"] = "ObjectDetection"
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template['ModelParameters'] = {}
        _emd_template['ModelParameters']['anchors'] = self.config_model['ANCHORS']
        _emd_template['ModelParameters']['n_bands'] = self.config_model['N_BANDS']
        _emd_template['Classes'] = []

        if self._data is not None:
            for i, class_name in enumerate(self._data.classes[1:]): # 0th index is background
                inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
                class_data["Value"] = inverse_class_mapping[class_name]
                class_data["Name"] = class_name
                color = [random.choice(range(256)) for i in range(3)]
                class_data["Color"] = color
                _emd_template['Classes'].append(class_data.copy())

        else:
            for k, i in coco_class_mapping().items():
                class_data['Value'] = k
                class_data['Name'] = i
                color = [random.choice(range(256)) for i in range(3)]
                class_data["Color"] = color
                _emd_template['Classes'].append(class_data.copy())

        return _emd_template


    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a YOLOv3 Object Detector from an Esri Model Definition (EMD) file.

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
        
        :returns: `YOLOv3` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
            
        emd_path = _get_emd_path(emd_path)
        emd = json.load(open(emd_path))
        model_file = Path(emd['ModelFile'])
        chip_size = emd["ImageWidth"]

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
        
        resize_to = emd.get('resize_to')
        if isinstance(resize_to, list):
            resize_to = (resize_to[0], resize_to[1])

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
            data = get_multispectral_data_params_from_emd(data, emd)
            # Add 1 for background class
            data.c += 1
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd

        data.resize_to = resize_to
        ret = cls(data, **emd['ModelParameters'], pretrained_path=model_file)
        
        if not data_passed:
            ret.learn.data.single_ds.classes = ret._data.classes
            ret.learn.data.single_ds.y.classes = ret._data.classes
        
        return ret

def create_coco_data():
    """ Create an empty databunch for COCO dataset."""

    train_tfms = []
    val_tfms = []
    ds_tfms = (train_tfms, val_tfms)

    class_mapping = coco_class_mapping()

    import tempfile
    sd = ImageList([], path=tempfile.NamedTemporaryFile().name, ignore_empty=True).split_none()
    data = sd.label_const(0, label_cls=ObjectDetectionCategoryList, classes=list(class_mapping.values())).transform(ds_tfms).databunch()

    data.class_mapping = class_mapping
    data.classes = list(class_mapping.values())
    data._is_empty = False
    data._is_coco = True
    data.resize_to = 416
    data.chip_size = 416

    return data
