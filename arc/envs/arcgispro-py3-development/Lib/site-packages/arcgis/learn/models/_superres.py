from ._codetemplate import super_resolution
import json
import traceback
from ._arcgis_model import _EmptyData
from .._data import _raise_fastai_import_error  
try:
    from ._arcgis_model import ArcGISModel, _resnet_family
    from ._superres_utils import FeatureLoss, gram_matrix, compute_metrics, get_resize, create_loss
    from fastai.vision.learner import unet_learner
    from fastai.vision import nn, ImageImageList, get_transforms, imagenet_stats, NormType, open_image
    from fastai.callbacks import LossMetrics
    from fastai.utils.mem import Path
    from .._utils.common import _get_emd_path
    from .._utils.env import _IS_ARCGISPRONOTEBOOK

    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False




class SuperResolution(ArcGISModel):

    """
    Creates a model object which increases the resolution and improves the quality of images.
    Based on Fast.ai MOOC Lesson 7.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            `prepare_data` function.
    ---------------------   -------------------------------------------
    backbone                Optional function. Backbone CNN model to be used for
                            creating the base of the `UnetClassifier`, which
                            is `resnet34` by default.
                            Compatible backbones: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================
                                             
    :returns: `SuperResolution` Object
    """
    def __init__(self, data, backbone=None, pretrained_path=None, *args, **kwargs):
        super().__init__(data, backbone, **kwargs)
        self._check_dataset_support(data)
        feat_loss = create_loss(self._device.type)
        data.c = 3
        self.learn = unet_learner(data, arch=self._backbone, wd=1e-3, loss_func=feat_loss, callback_fns=LossMetrics, blur=True, norm_type=NormType.Weight)
        self.learn.model = self.learn.model.to(self._device)
        if pretrained_path is not None:
            self.load(pretrained_path)
        
        self._code = super_resolution

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)

    @property
    def supported_backbones(self):
        """
        Supported torchvision backbones for this model.
        """        
        return SuperResolution._supported_backbones()

    @staticmethod
    def _supported_backbones():
        return [*_resnet_family]

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a SuperResolution object from an Esri Model Definition (EMD) file.

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
        
        :returns: `SuperResolution` Object
        """
        return cls.from_emd(data, emd_path)

    @classmethod
    def from_emd(cls, data, emd_path):
        """
        Creates a SuperResolution object from an Esri Model Definition (EMD) file.

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
        
        :returns: `SuperResolution` Object
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
        downsample_factor = emd.get('downsample_factor')
        resize_to = emd.get('resize_to')
        chip_size = emd['ImageHeight']
        feat_loss = create_loss()
        if data is None:
            data = (ImageImageList.from_folder(emd_path.parent.parent).split_none().label_from_func(lambda x: x).transform(get_transforms(do_flip=False),size=(chip_size, chip_size), tfm_y=True).databunch(bs=2, no_check=True).normalize(imagenet_stats, do_y=True))
            data._is_empty = True
            data.emd_path = emd_path
            data.downsample_factor = downsample_factor
            data.emd = emd
        data.resize_to = resize_to
        
        return cls(data, **model_params, pretrained_path=str(model_file))
    
    @property
    def _model_metrics(self):
        return self.compute_metrics(show_progress=True)
         

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["Framework"] = "arcgis.learn.models._inferencing"
        _emd_template["ModelConfiguration"] = "_superres"
        _emd_template["InferenceFunction"] = "ArcGISSuperResolution.py"
        _emd_template["ModelType"] = "SuperResolution"
        _emd_template["downsample_factor"] = self._data.downsample_factor
        return _emd_template

    def compute_metrics(self, accuracy=True, show_progress=True):
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) and 
        Structural Similarity Index Measure (SSIM) on validation set.

        """
        self._check_requisites()
        psnr, ssim = compute_metrics(self, self._data.valid_dl, show_progress)
        return {'PSNR': '{0:1.4e}'.format(psnr),
                'SSIM': '{0:1.4e}'.format(ssim)}

    
    def show_results(self, rows=5):
        """
        Displays the results of a trained model on a part of the validation set.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional int. Number of rows of results
                                to be displayed.
        =====================   ===========================================

        """
        if rows > len(self._data.valid_ds):
            rows = len(self._data.valid_ds)
        
        self._check_requisites()
        self.learn.show_results(rows=rows)
        if _IS_ARCGISPRONOTEBOOK:
            from matplotlib import pyplot as plt
            plt.show()

    def predict(self, img_path, width=None, height=None):
        """
        Predicts and display the image.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        img_path                Required path of an image.
        ---------------------   -------------------------------------------
        width                   Optional int. Width of the predicted 
                                output image.
        ---------------------   -------------------------------------------
        height                  Optional int. Height of the predicted
                                output image.
        =====================   ===========================================

        """
        img_path = Path(img_path)
        img = open_image(img_path)        
        temp_databunch = self.learn.data
        if width is not None or height is not None:
            if width is None:
                width = height
            elif height is None:
                height = width
        elif width is None and height is None:
            _,width,height = img.shape
        
        y_new, z_new = width, height
        pred_databunch = (ImageImageList.from_folder(img_path.parent).split_none()\
        .label_from_func(lambda x: x)\
        .transform(get_transforms(do_flip=False), size=(height,width), tfm_y=True)\
        .databunch(bs=2, no_check=True).normalize(imagenet_stats, do_y=True))
            
        self.learn.data = pred_databunch
        
        pred_img = self.learn.predict(img)[0]
        self.learn.data = temp_databunch
        return pred_img

    @property
    def  supported_datasets(self):
        """ Supported dataset types for this model. """
        return SuperResolution._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ['Export_Tiles', 'superres'] 



