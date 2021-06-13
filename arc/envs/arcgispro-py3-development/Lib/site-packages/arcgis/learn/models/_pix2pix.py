from ._codetemplate import image_translation_prf
import json
import traceback
from .._data import _raise_fastai_import_error
from ._arcgis_model import ArcGISModel
try:
    from ._pix2pix_utils import pix2pixLoss, pix2pixTrainer, optim, compute_metrics, compute_fid_metric
    from ._pix2pix_utils import  pix2pix as pix2pix_model
    from .._utils.pix2pix import ImageTuple, ImageTupleList2, ImageTupleListMS2
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path
    from torchvision import transforms
    from pathlib import Path
    from fastai.vision import *
    from fastai.vision import DatasetType, Learner, partial, open_image
    import torch

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False

class Pix2Pix(ArcGISModel):

    """
    Creates a model object which generates fake images of type B from type A.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================
                                             
    :returns: `Pix2Pix` Object
    """
    
    def __init__(self, data, pretrained_path=None, *args, **kwargs):
        super().__init__(data)
        self._check_dataset_support(data)
        pix2pix_gan = pix2pix_model(self._data.n_channel,self._data.n_channel)
        self.learn = Learner(data, 
                             pix2pix_gan, 
                             loss_func=pix2pixLoss(pix2pix_gan), 
                             opt_func=partial(optim.Adam,betas=(0.5,0.99)),
                             callback_fns=[pix2pixTrainer])

        self.learn.model = self.learn.model.to(self._device)
        self._slice_lr = False
        if pretrained_path is not None:
            self.load(pretrained_path)
        self._code = image_translation_prf
        def __str__(self):
            return self.__repr__()
        def __repr__(self):
            return '<%s>' % (type(self).__name__)
        
    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a Pix2Pix object from an Esri Model Definition (EMD) file.

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
        
        :returns: `Pix2Pix` Object
        """
        
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)
            
        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd['ModelFile'])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        model_params = emd['ModelParameters']
        resize_to = emd.get('resize_to')
        chip_size = emd['ImageHeight']
        if data is None:
             if emd.get('IsMultispectral', False):
                data = ImageTupleListMS2.from_folders(emd_path.parent, emd_path.parent, emd_path.parent, batch_stats_a=None, batch_stats_b=None).split_none().label_empty().databunch(bs=2,no_check = True)
                data.n_channel = emd['n_channel']
                data = get_multispectral_data_params_from_emd(data, emd)
                data._is_multispectral = emd.get('IsMultispectral', False)
                normalization_stats_b = dict(emd.get("NormalizationStats_b"))
                for _stat in normalization_stats_b:
                    if normalization_stats_b[_stat] is not None:
                        normalization_stats_b[_stat] = torch.tensor(normalization_stats_b[_stat])
                    setattr(data, ('_'+_stat), normalization_stats_b[_stat])

             else:
                 data = ImageTupleList2.from_folders(emd_path.parent, emd_path.parent, emd_path.parent)\
                     .split_none()\
                     .label_empty()\
                     .transform(size=(chip_size, chip_size))\
                     .databunch(bs=2, no_check = True)
        data.n_channel = emd['n_channel']
        data._is_empty = True
        data.emd_path = emd_path
        data.emd = emd
        data.resize_to = chip_size
        
        return cls(data, **model_params, pretrained_path=str(model_file))
        
    @property
    def _model_metrics(self):
        return self.compute_metrics(show_progress=True)

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_pix2pix"
        _emd_template["InferenceFunction"] = "ArcGISImageTranslation.py"
        _emd_template["ModelType"] = "Pix2Pix"
        _emd_template["n_channel"] = self._data.n_channel
        if self._data._is_multispectral:
            _emd_template["NormalizationStats_b"] = {
                    "band_min_values": self._data._band_min_values_b,
                    "band_max_values": self._data._band_max_values_b,
                    "band_mean_values": self._data._band_mean_values_b,
                    "band_std_values": self._data._band_std_values_b,
                    "scaled_min_values": self._data._scaled_min_values_b,
                    "scaled_max_values": self._data._scaled_max_values_b,
                    "scaled_mean_values": self._data._scaled_mean_values_b,
                    "scaled_std_values": self._data._scaled_std_values_b
        }
            for _stat in _emd_template["NormalizationStats_b"]:
                    if _emd_template["NormalizationStats_b"][_stat] is not None:
                        _emd_template["NormalizationStats_b"][_stat] = _emd_template["NormalizationStats_b"][_stat].tolist()
        return _emd_template

    def show_results(self,rows=5):
        """
        Displays the results of a trained model on a part of the validation set.

        """

        self.learn.model.arcgis_results = True
        self.learn.show_results()
        self.learn.model.arcgis_results = False

    def predict(self, img_path):
        """
        Predicts and display the image.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        img_path                Required path of an image.
        =====================   ===========================================

        """
        self.learn.model.arcgis_results = True
        img_path = Path(img_path)
        raw_img = open_image(img_path)
        n_band = self._data.n_channel
        if n_band > raw_img.shape[0]:
            cont = []
            last_tile = np.expand_dims(raw_img.data[raw_img.shape[0]-1,:,:], 0)
            res = abs(n_band - raw_img.shape[0])
            for i in range(res):
                raw_img = Image(torch.tensor(np.concatenate((raw_img.data, last_tile), axis=0)))
        raw_img_tuple = ImageTuple(raw_img, raw_img)
        pred_tuple = self.learn.predict(raw_img_tuple)
        pred_img = pred_tuple[1][0]/2+0.5
        
        pred_img = transforms.ToPILImage()(pred_img).convert("RGB")
        self.learn.model.arcgis_results = False
        return pred_img

    def compute_metrics(self, accuracy=True, show_progress=True):
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) and 
        Structural Similarity Index Measure (SSIM) on validation set.

        """
        psnr, ssim = compute_metrics(self, self._data.valid_dl, show_progress)
        if self._data._imagery_type_b == 'RGB' and self._data.n_channel == 3:
            fid = compute_fid_metric(self, self._data)
            return {"PSNR":'{0:1.4e}'.format(psnr), 
                    "SSIM":'{0:1.4e}'.format(ssim), 
                    "FID":'{0:1.4e}'.format(fid)}
        else:
            fid = None
            return {"PSNR":'{0:1.4e}'.format(psnr),
                    "SSIM":'{0:1.4e}'.format(ssim)}

    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return Pix2Pix._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ['Pix2Pix'] 