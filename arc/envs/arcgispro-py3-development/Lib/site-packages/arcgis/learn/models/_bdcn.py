from pathlib import Path
import json
from ._model_extension import ModelExtension
from ._arcgis_model import _EmptyData

try:
    from fastai.vision import flatten_model
    import torch
    from fastai.torch_core import split_model_idx
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from ._arcgis_model import _resnet_family, _vgg_family

    HAS_FASTAI = True

except Exception as e:
    HAS_FASTAI = False

class CustomBDCN():
    """
    Create class with following fixed function names and the number of arguents to train your model from external source
    """

    try:
        import torch
        from torchvision import models
        import pathlib
        import os
        import fastai
        from arcgis.learn.models import _bdcn_utils as bdcn
    except:
        pass
    
    def get_model(self, data, backbone=None):
        """
        In this fuction you have to define your model with following two arguments!
        
        """
        if backbone is None:
            self._backbone = self.models.vgg19
        elif type(backbone) is str:
            if hasattr(self.models, backbone):
                self._backbone = getattr(self.models, backbone)
            elif hasattr(self.models.detection, backbone):
                self._backbone = getattr(self.models.detection, backbone)
        else:
            self._backbone = backbone

        model = self.bdcn._BDCNModel(self._backbone, data.chip_size)
        
        return model
    
    def on_batch_begin(self, learn, model_input_batch, model_target_batch):
        
        return model_input_batch, model_target_batch
    
    def transform_input(self, xb):
        
        return xb
    
    def transform_input_multispectral(self, xb):

        return xb

    def loss(self, model_output, *model_target):

        final_loss = self.bdcn.bdcn_loss(model_output, *model_target)
        
        return final_loss
    
    def post_process(self, pred, thres=0.5, thinning=True):
        """
        In this function you have to return list with appended output for each image in the batch with shape [C=1,H,W]!
        
        """

        from skimage.morphology import skeletonize, binary_dilation
        import numpy as np

        post_processed_pred = []
        pred = pred[-1]
        if thinning:
            for p in pred:
                p = self.torch.unsqueeze(self.torch.tensor(skeletonize(binary_dilation(np.squeeze((p>=thres).byte().cpu().numpy())))), dim=0)
                post_processed_pred.append(p)
        else:
            return (pred>=thres).byte()
        return post_processed_pred

class BDCNEdgeDetector(ModelExtension):
    """
    Model architecture from https://arxiv.org/pdf/1902.10903.pdf.
    Creates a ``Bi-Directional Cascade Network for Perceptual Edge Detection`` model

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            ``prepare_data`` function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `Bi-Directional Cascade Network
                            for Perceptual Edge Detection`, which
                            is `vgg19` by default. 
                            Compatible backbones: resnet and VGG
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    :returns: ``Bi-Directional Cascade Network for Perceptual Edge Detection`` Object
    """
    def __init__(self, data, backbone='vgg19', pretrained_path=None):

        self._check_dataset_support(data)
        backbone_name = backbone if type(backbone) is str else backbone.__name__
        if backbone_name not in self.supported_backbones:
            raise Exception (f"Enter only compatible backbones from {', '.join(self.supported_backbones)}")

        super().__init__(data, CustomBDCN, backbone, pretrained_path)
        self._freeze()

    def unfreeze(self):
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def _freeze(self):
        "Freezes the pretrained backbone."
        count = 0
        count_strided_conv = 0
        for idx, i in enumerate(flatten_model(self.learn.model.backbone)):
            if isinstance(i, (torch.nn.BatchNorm2d)):
                continue

            for p in i.parameters():
                p.requires_grad = False
                
            if isinstance(i, torch.nn.MaxPool2d):
                count += 1
                if count == 3:
                    break
            if isinstance(i, torch.nn.Conv2d):
                if i.stride[0] == 2:
                    count_strided_conv += 1
                    if count_strided_conv == 4:
                        break

        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=3e-3)

    @property
    def _is_edge_detection(self):
        return True

    @property
    def supported_backbones(self):
        """ Supported torchvision backbones for this model. """
        return BDCNEdgeDetector._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family, *_vgg_family]

    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return BDCNEdgeDetector._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ['Classified_Tiles']

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a ``Bi-Directional Cascade Network for Perceptual Edge Detection`` object from an Esri Model Definition (EMD) file.

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

        :returns: `Bi-Directional Cascade Network for Perceptual Edge Detection` Object
        """
        emd_path = _get_emd_path(emd_path)

        with open(emd_path) as f:
            emd = json.load(f)
            
        model_file = Path(emd['ModelFile'])
        
        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file
        
        backbone = emd['ModelParameters']['backbone']

        try:
            class_mapping = {i['Value'] : i['Name'] for i in emd['Classes']}
            color_mapping = {i['Value'] : i['Color'] for i in emd['Classes']}
        except KeyError:
            class_mapping = {i['ClassValue'] : i['ClassName'] for i in emd['Classes']} 
            color_mapping = {i['ClassValue'] : i['Color'] for i in emd['Classes']}                

        if data is None:
            data = _EmptyData(path=emd_path.parent.parent, loss_func=None, c=len(class_mapping) + 1, chip_size=emd['ImageHeight'])
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data.emd_path = emd_path
            data.emd = emd
            data.classes = ['background']
            for k, v in class_mapping.items():
                data.classes.append(v)
            data = get_multispectral_data_params_from_emd(data, emd)
            data.dataset_type = emd['DatasetType']
        
        return cls(data, backbone, pretrained_path=str(model_file))

    def compute_precision_recall(self, thresh=0.5, buffer=3, show_progress=True):

        """
        Computes precision, recall and f1 score on validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        thresh                  Optional float. The probability on which
                                the detection will be considered edge pixel.
        ---------------------   -------------------------------------------
        buffer                  Optional int. pixels in neighborhood to
                                consider true detection.
        =====================   ===========================================

        :returns: `dict` 
        """

    def show_results(self, rows=5, thresh=0.5, thinning=True,**kwargs):

        """
        Displays the results of a trained model on a part of the validation set.
        """