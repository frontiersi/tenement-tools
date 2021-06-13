from ._arcgis_model import ArcGISModel
from pathlib import Path
import json
from ._arcgis_model import _EmptyData, _change_tail
from ._codetemplate import instance_detector_prf
import math

try:
    import torch
    from fastai.vision.learner import cnn_learner
    from fastai.callbacks.hooks import model_sizes
    from fastai.vision.learner import create_body
    from fastai.vision.image import open_image
    from fastai.vision import flatten_model
    from fastai.core import has_arg, split_kwargs_by_func
    from torchvision.models import resnet34
    from torchvision import models
    import numpy as np
    from .._data import prepare_data, _raise_fastai_import_error
    from fastai.callbacks import EarlyStoppingCallback
    from ._arcgis_model import SaveModelCallback, _set_multigpu_callback, _get_backbone_meta, _resnet_family, _set_ddp_multigpu, _isnotebook
    import torchvision
    from torchvision import models
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from fastai.basic_train import Learner
    from ._maskrcnn_utils import is_no_color, mask_rcnn_loss, train_callback, compute_class_AP
    from ._MaskRCNN_PointRend import create_pointrend
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from fastai.torch_core import split_model_idx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib
    from fastai.basic_data import DatasetType
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    import os as arcgis_os
    from .._utils.common import get_nbatches, image_batch_stretcher
    from .._utils.env import _IS_ARCGISPRONOTEBOOK

    HAS_FASTAI = True
except Exception as e:
    #raise Exception(e)
    HAS_FASTAI = False


class MaskRCNN(ArcGISModel):
    """
    Model architecture from https://arxiv.org/abs/1703.06870.
    Creates a ``MaskRCNN`` Instance segmentation model,
    based on https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            ``prepare_data`` function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `MaskRCNN`, which
                            is `resnet50` by default. 
                            Compatible backbones: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    ---------------------   -------------------------------------------
    pointrend               Optional boolean. If True, it will use PointRend
                            architecture on top of the segmentation head.
                            Default: False. PointRend architecture from
                            https://arxiv.org/pdf/1912.08193.pdf.      
    =====================   ===========================================
    
    **kwargs**

    =============================   =============================================
    **Argument**                    **Description**
    -----------------------------   ---------------------------------------------
    rpn_pre_nms_top_n_train         Optional int. Number of proposals to keep before
                                    applying NMS during training.
                                    Default: 2000
    -----------------------------   ---------------------------------------------
    rpn_pre_nms_top_n_test          Optional int. Number of proposals to keep before
                                    applying NMS during testing.
                                    Default: 1000
    -----------------------------   ---------------------------------------------
    rpn_post_nms_top_n_train        Optional int. Number of proposals to keep after
                                    applying NMS during training.
                                    Default: 2000
    -----------------------------   ---------------------------------------------
    rpn_post_nms_top_n_test         Optional int. Number of proposals to keep after
                                    applying NMS during testing.
                                    Default: 1000
    -----------------------------   ---------------------------------------------
    rpn_nms_thresh                  Optional float. NMS threshold used for postprocessing
                                    the RPN proposals.
                                    Default: 0.7
    -----------------------------   ---------------------------------------------
    rpn_fg_iou_thresh               Optional float. Minimum IoU between the anchor
                                    and the GT box so that they can be considered
                                    as positive during training of the RPN.
                                    Default: 0.7
    -----------------------------   ---------------------------------------------
    rpn_bg_iou_thresh               Optional float. Maximum IoU between the anchor and
                                    the GT box so that they can be considered as negative
                                    during training of the RPN.
                                    Default: 0.3
    -----------------------------   ---------------------------------------------
    rpn_batch_size_per_image        Optional int. Number of anchors that are sampled
                                    during training of the RPN for computing the loss.
                                    Default: 256
    -----------------------------   ---------------------------------------------
    rpn_positive_fraction           Optional float. Proportion of positive anchors in a
                                    mini-batch during training of the RPN.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_score_thresh                Optional float. During inference, only return proposals
                                    with a classification score greater than box_score_thresh
                                    Default: 0.05
    -----------------------------   ---------------------------------------------
    box_nms_thresh                  Optional float. NMS threshold for the prediction head.
                                    Used during inference.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_detections_per_img          Optional int. Maximum number of detections per
                                    image, for all classes.
                                    Default: 100
    -----------------------------   ---------------------------------------------
    box_fg_iou_thresh               Optional float. Minimum IoU between the proposals and
                                    the GT box so that they can be considered as positive
                                    during training of the classification head.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_bg_iou_thresh               Optional float. Maximum IoU between the proposals and 
                                    the GT box so that they can be considered as negative 
                                    during training of the classification head.
                                    Default: 0.5
    -----------------------------   ---------------------------------------------
    box_batch_size_per_image        Optional int. Number of proposals that are sampled during
                                    training of the classification head.
                                    Default: 512
    -----------------------------   ---------------------------------------------
    box_positive_fraction           Optional float. Proportion of positive proposals in a
                                    mini-batch during training of the classification head.
                                    Default: 0.25
    =============================   =============================================

    :returns: ``MaskRCNN`` Object
    """
    def __init__(self, data, backbone=None, pretrained_path=None, pointrend=False, *args, **kwargs):

        # Set default backbone to be 'resnet50'
        if backbone is None:
            backbone = models.resnet50

        self._check_dataset_support(data)
        if not (self._check_backbone_support(backbone)):
            raise Exception (f"Enter only compatible backbones from {', '.join(self.supported_backbones)}")

        super().__init__(data, backbone, **kwargs)
        if self._is_multispectral:
            self._backbone_ms = self._backbone
            self._backbone = self._orig_backbone
            scaled_mean_values = data._scaled_mean_values[data._extract_bands].tolist()
            scaled_std_values = data._scaled_std_values[data._extract_bands].tolist()

        self._code = instance_detector_prf

        self._pointrend = pointrend

        self.maskrcnn_kwargs, kwargs = split_kwargs_by_func(kwargs, models.detection.MaskRCNN.__init__)

        if self._backbone.__name__ is 'resnet50':
            model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                min_size = 1.5*data.chip_size,
                max_size = 2*data.chip_size,
                **self.maskrcnn_kwargs
            )

            if self._is_multispectral:
                model.backbone = _change_tail(model.backbone, data)
                model.transform.image_mean = scaled_mean_values
                model.transform.image_std = scaled_std_values
        elif self._backbone.__name__ in ['resnet18','resnet34'] and not pointrend:
            if self._is_multispectral:
                backbone_small = create_body(self._backbone_ms, cut=_get_backbone_meta(self._backbone.__name__)['cut'])
                backbone_small.out_channels = 512
                model = models.detection.MaskRCNN(
                    backbone_small, 
                    91, 
                    min_size = 1.5*data.chip_size, 
                    max_size = 2*data.chip_size, 
                    image_mean = scaled_mean_values, 
                    image_std = scaled_std_values,
                    **self.maskrcnn_kwargs
                )
            else:
                backbone_small = create_body(self._backbone)
                backbone_small.out_channels = 512
                model = models.detection.MaskRCNN(
                    backbone_small,
                    91,
                    min_size = 1.5*data.chip_size,
                    max_size = 2*data.chip_size,
                    **self.maskrcnn_kwargs
                )
        else:
            backbone_fpn = resnet_fpn_backbone(self._backbone.__name__, True)
            if self._is_multispectral:
                backbone_fpn = _change_tail(backbone_fpn, data)
                model = models.detection.MaskRCNN(
                    backbone_fpn, 
                    91, 
                    min_size = 1.5*data.chip_size, 
                    max_size = 2*data.chip_size, 
                    image_mean = scaled_mean_values, 
                    image_std = scaled_std_values,
                    **self.maskrcnn_kwargs
                )
            else:
                model = models.detection.MaskRCNN(
                    backbone_fpn,
                    91,
                    min_size = 1.5*data.chip_size,
                    max_size = 2*data.chip_size,
                    **self.maskrcnn_kwargs
                )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, data.c)

        if pointrend:
            model = create_pointrend(model, data.c)
        else:
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        data.c)

        if not _isnotebook() and arcgis_os.name=='posix':
            _set_ddp_multigpu(self)
            if self._multigpu_training:
                self.learn = Learner(data, model, loss_func=mask_rcnn_loss).to_distributed(self._rank_distributed)
            else:
                self.learn = Learner(data, model, loss_func=mask_rcnn_loss)
        else:
            self.learn = Learner(data, model, loss_func=mask_rcnn_loss)
        self.learn.callbacks.append(train_callback(self.learn))
        self.learn.model = self.learn.model.to(self._device)
        self.learn.c_device = self._device

        # fixes for zero division error when slice is passed
        idx = 27
        if self._backbone.__name__ in ['resnet18','resnet34']:
            idx = self._freeze()
        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=3e-3)

        # make first conv weights learnable
        self._arcgis_init_callback()

        if pretrained_path is not None:
            self.load(pretrained_path)
                
        if self._is_multispectral:
            self._orig_backbone = self._backbone
            self._backbone = self._backbone_ms
             
    def unfreeze(self):
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def _freeze(self):
        "Freezes the pretrained backbone."
        for idx, i in enumerate(flatten_model(self.learn.model.backbone)):
            if isinstance(i, (torch.nn.BatchNorm2d)):
                continue
            for p in i.parameters():
                p.requires_grad = False
        return idx

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)

    @property
    def supported_backbones(self):
        """ Supported torchvision backbones for this model. """        
        return MaskRCNN._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family]

    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return MaskRCNN._supported_datasets()
    
    @staticmethod
    def _supported_datasets():
        return ['RCNN_Masks'] 
    
    @classmethod
    def from_model(cls, emd_path, data=None, **kwargs):
        """
        Creates a ``MaskRCNN`` Instance segmentation object from an Esri Model Definition (EMD) file.

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

        :returns: `MaskRCNN` Object
        """

        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)
            
        model_file = Path(emd['ModelFile'])
        
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file
            
        model_params = emd['ModelParameters']
        maskrcnn_kwargs = emd.get("MaskRCNNkwargs", {})

        try:
            class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
            color_mapping = {i['Value'] : i['Color'] for i in emd['Classes']}
        except KeyError:
            class_mapping = {i['ClassValue'] : i['ClassName'] for i in emd['Classes']} 
            color_mapping = {i['ClassValue'] : i['Color'] for i in emd['Classes']}                

        if data is None:
            data = _EmptyData(path=emd_path.parent.parent, loss_func=None, c=len(class_mapping) + 1, chip_size=kwargs.get('chip_size', emd['ImageHeight']))
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data.emd_path = emd_path
            data.emd = emd
            data = get_multispectral_data_params_from_emd(data, emd)

        return cls(data, **model_params, pretrained_path=str(model_file), **maskrcnn_kwargs)

    def _get_emd_params(self, save_inference_file):
        import random

        _emd_template = {"ModelParameters" : {}}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_maskrcnn_inferencing"
        if save_inference_file:
            _emd_template["InferenceFunction"] = "ArcGISInstanceDetector.py"
        else:
            _emd_template["InferenceFunction"] = "[Functions]System\\DeepLearning\\ArcGISLearn\\ArcGISInstanceDetector.py"
        _emd_template["ModelType"] = "InstanceDetection"
        _emd_template["MaskRCNNkwargs"] = self.maskrcnn_kwargs
        _emd_template["ModelParameters"]["pointrend"] = self._pointrend
        _emd_template["ExtractBands"] = [0, 1, 2]
        _emd_template["SupportsVariableTileSize"] = True
        _emd_template['Classes'] = []
        class_data = {}
        for i, class_name in enumerate(self._data.classes[1:]):  # 0th index is background
            inverse_class_mapping = {v: k for k, v in self._data.class_mapping.items()}
            class_data["Value"] = inverse_class_mapping[class_name]
            class_data["Name"] = class_name
            color = [random.choice(range(256)) for i in range(3)] if is_no_color(self._data.color_mapping) else \
            self._data.color_mapping[inverse_class_mapping[class_name]]
            class_data["Color"] = color
            _emd_template['Classes'].append(class_data.copy())

        return _emd_template

    @property
    def _model_metrics(self):
        return {'average_precision_score': self.average_precision_score(show_progress=True)}

    def _predict_results(self, xb):

        self.learn.model.eval()
        xb_l = xb.to(self._device)
        predictions = self.learn.model(list(xb_l))
        xb_l = xb_l.detach().cpu()
        del xb_l
        predictionsf = []
        for i in range(len(predictions)):
            predictionsf.append({})
            predictionsf[i]['masks'] = predictions[i]['masks'].detach().cpu().numpy()
            predictionsf[i]['boxes'] = predictions[i]['boxes'].detach().cpu().numpy()
            predictionsf[i]['labels'] = predictions[i]['labels'].detach().cpu().numpy()
            predictionsf[i]['scores'] = predictions[i]['scores'].detach().cpu().numpy()
            del predictions[i]['masks']
            del predictions[i]['boxes']
            del predictions[i]['labels']
            del predictions[i]['scores']
        if self._device == torch.device('cuda'):
            torch.cuda.empty_cache()
        return predictionsf

    def _predict_postprocess(self, predictions, threshold=0.5, box_threshold = 0.5):

        pred_mask = []
        pred_box = []

        for i in range(len(predictions)):
            out = predictions[i]['masks'].squeeze()
            pred_box.append([])

            if out.shape[0] != 0:  # handle for prediction with n masks
                if len(out.shape) == 2: # for out dimension hxw (in case of only one predicted mask)
                    out = out[None]
                ymask = np.where(out[0]> threshold, 1, 0)
                if predictions[i]['scores'][0] > box_threshold:
                    pred_box[i].append(predictions[i]['boxes'][0])
                for j in range(1,out.shape[0]):
                    ym1 = np.where(out[j]> threshold, j+1, 0)
                    ymask += ym1
                    if predictions[i]['scores'][j] > box_threshold:
                        pred_box[i].append(predictions[i]['boxes'][j])
            else:
                ymask = np.zeros((self._data.chip_size, self._data.chip_size)) # handle for not predicted masks
            pred_mask.append(ymask)
        return pred_mask, pred_box

    def show_results(self, rows=4, mode='mask', mask_threshold=0.5, box_threshold=0.7, imsize=5, index=0, alpha=0.5, cmap='tab20', **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        mode                    Required arguments within ['bbox', 'mask', 'bbox_mask'].
                                    * ``bbox`` - For visualizing only bounding boxes.
                                    * ``mask`` - For visualizing only mask
                                    * ``bbox_mask`` - For visualizing both mask and bounding boxes.
        ---------------------   -------------------------------------------
        mask_threshold          Optional float. The probability above which
                                a pixel will be considered mask.
        ---------------------   -------------------------------------------
        box_threshold           Optional float. The probability above which
                                a detection will be considered valid.
        ---------------------   -------------------------------------------
        nrows                   Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================
        """
        self._check_requisites()
        if mode not in ['bbox', 'mask', 'bbox_mask']:
            raise Exception("mode can be only ['bbox', 'mask', 'bbox_mask']")

        # Get Number of items
        nrows = rows
        ncols=2


        type_data_loader = kwargs.get('data_loader', 'validation') # options : traininig, validation, testing
        if type_data_loader == 'training':
            data_loader = self._data.train_dl
        elif type_data_loader == 'validation':
            data_loader = self._data.valid_dl
        elif type_data_loader == 'testing':
            data_loader = self._data.test_dl
        else:
            e = Exception(f'could not find {type_data_loader} in data. Please ensure that the data loader type is traininig, validation or testing ')
            raise(e)

        statistics_type = kwargs.get('statistics_type', 'dataset') # Accepted Values `dataset`, `DRA`
        stretch_type = kwargs.get('stretch_type', 'minmax') # Accepted Values `minmax`, `percentclip`

        cmap_fn = getattr(matplotlib.cm, cmap)
        return_fig = kwargs.get('return_fig', False)


        x_batch, y_batch = get_nbatches(data_loader, nrows)
        x_batch = torch.cat(x_batch)
        y_batch = torch.cat(y_batch)

        nrows = min(nrows, len(x_batch))
        
        title_font_size = 16
        if kwargs.get('top', None) is not None:
            top = kwargs.get('top')
        else:
            top = 1 - (math.sqrt(title_font_size)/math.sqrt(100*nrows*imsize))



        # Get Predictions
        prediction_store = []
        for i in range(0, x_batch.shape[0], self._data.batch_size):
            prediction_store.extend(self._predict_results(x_batch[i:i+self._data.batch_size]))
        pred_mask, pred_box = self._predict_postprocess(prediction_store, mask_threshold, box_threshold)

        if self._is_multispectral:
            rgb_bands = kwargs.get('rgb_bands', self._data._symbology_rgb_bands)

            e = Exception('`rgb_bands` should be a valid band_order, list or tuple of length 3 or 1.')
            symbology_bands = []
            if not ( len(rgb_bands) == 3 or len(rgb_bands) == 1 ):
                raise(e)
            for b in rgb_bands:
                if type(b) == str:
                    b_index = self._bands.index(b)
                elif type(b) == int:
                    self._bands[b] # To check if the band index specified by the user really exists.
                    b_index = b
                else:
                    raise(e)
                b_index = self._data._extract_bands.index(b_index)
                symbology_bands.append(b_index)

             # Denormalize X
            if self._data._do_normalize:
                x_batch = (self._data._scaled_std_values[self._data._extract_bands].view(1, -1, 1, 1).to(x_batch) * x_batch ) + self._data._scaled_mean_values[self._data._extract_bands].view(1, -1, 1, 1).to(x_batch)

            # Extract RGB Bands
            symbology_x_batch = x_batch[:, symbology_bands]
            if stretch_type is not None:
                symbology_x_batch = image_batch_stretcher(symbology_x_batch, stretch_type, statistics_type)

            # Channel first to channel last for plotting
            symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
            # Clamp float values to range 0 - 1
            if symbology_x_batch.mean() < 1:
                symbology_x_batch = symbology_x_batch.clamp(0, 1)
        else:
            symbology_x_batch = x_batch.permute(0, 2, 3, 1)

        # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
        if symbology_x_batch.shape[-1] == 1:
            symbology_x_batch = symbology_x_batch.squeeze()

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*imsize, nrows*imsize))
        fig.suptitle('Ground Truth / Predictions', fontsize=title_font_size)
        for i in range(nrows):
            if nrows == 1:
                ax_i = ax
            else:
                ax_i = ax[i]

            # Ground Truth
            ax_i[0].imshow(symbology_x_batch[i].cpu())
            ax_i[0].axis('off')
            if mode in ['mask', 'bbox_mask']:
                n_instance = y_batch[i].unique().shape[0]
                y_merged = y_batch[i].max(dim=0)[0].cpu().numpy()
                y_rgba = cmap_fn._resample(n_instance)(y_merged)
                y_rgba[y_merged == 0] = 0
                y_rgba[:, :, -1] = alpha
                ax_i[0].imshow(y_rgba)
            ax_i[0].axis('off')

            # Predictions
            ax_i[1].imshow(symbology_x_batch[i].cpu())
            ax_i[1].axis('off')
            if mode in ['mask', 'bbox_mask']:
                n_instance = np.unique(pred_mask[i]).shape[0]
                p_rgba = cmap_fn._resample(n_instance)(pred_mask[i])
                p_rgba[pred_mask[i] == 0] = 0
                p_rgba[:, :, -1] = alpha
                ax_i[1].imshow(p_rgba)
            if mode in ['bbox_mask','bbox']:
                if pred_box[i] != []:
                    for num_boxes in pred_box[i]:
                        rect = patches.Rectangle((num_boxes[0], num_boxes[1]), num_boxes[2]-num_boxes[0], num_boxes[3]-num_boxes[1], linewidth=1, edgecolor='r', facecolor='none')
                        ax_i[1].add_patch(rect)
            ax_i[1].axis('off')
        plt.subplots_adjust(top=top)
        if self._device == torch.device('cuda'):
            torch.cuda.empty_cache()

        if _IS_ARCGISPRONOTEBOOK:
            plt.show()
        if return_fig:
            return fig

    def average_precision_score(self, detect_thresh=0.5, iou_thresh=0.5, mean=False, show_progress=True):

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
                                threshold with the ground truth mask, above
                                which a predicted mask will be
                                considered a true positive.
        ---------------------   -------------------------------------------
        mean                    Optional bool. If False returns class-wise
                                average precision otherwise returns mean
                                average precision.
        =====================   ===========================================
        :returns: `dict` if mean is False otherwise `float`
        """
        self._check_requisites()
        if mean:
            aps = compute_class_AP(self, self._data.valid_dl, 1, show_progress, detect_thresh, iou_thresh, mean)
            return aps
        else:
            aps = compute_class_AP(self, self._data.valid_dl, self._data.c - 1, show_progress, detect_thresh, iou_thresh)
            return dict(zip(self._data.classes[1:], aps))
