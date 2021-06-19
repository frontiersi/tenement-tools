import os
import math
import numpy as np
import torch
from fastai.vision.data import ObjectCategoryList, ObjectItemList
from fastai.vision.image import ImageBBox
from fastai.core import split_kwargs_by_func
from .common import ArcGISMSImage, get_nbatches, denorm_x, dynamic_range_adjustment, image_batch_stretcher
from matplotlib import pyplot as plt
from matplotlib import patheffects
from .._utils.env import _IS_ARCGISPRONOTEBOOK


class ObjectDetectionCategoryList(ObjectCategoryList):
    "`ItemList` for labelled bounding boxes detected using YOLOv3."
    def analyze_pred(self, pred, model, thresh=0.5, nms_overlap=0.1, ret_scores=True, device=torch.device('cpu')):
        return model._analyze_pred(pred, thresh=thresh, nms_overlap=nms_overlap, ret_scores=ret_scores, device=device)
        
    def reconstruct(self, t, x):
        return _reconstruct(t, x, self.pad_idx, self.classes)

    
class ObjectDetectionItemList(ObjectItemList):
    "`ItemList` suitable for object detection."
    _label_cls,_square_show_res = ObjectDetectionCategoryList, False
    _div = None
    _imagery_type = None
    def open(self, fn):
        return ArcGISMSImage.open(fn, div=self._div, imagery_type=self._imagery_type)


def _reconstruct(t, x, pad_idx, classes):
    """Function to take post-processed output of model and return ImageBBox."""

    if t is None: return None

    t = list(t)
    if len(t[0]) == 0:
        return None

    if len(t) == 3:
        bboxes, labels, scores = t
        if len((labels - pad_idx).nonzero()) == 0: 
            ret = ImageBBox.create(*x.size, bboxes, labels=labels, classes=classes, scale=False)
            ret.scores = t[2]
            return ret
        i = (labels - pad_idx).nonzero().min()
        bboxes,labels,scores = bboxes[i:],labels[i:], scores[i:]
        ret = ImageBBox.create(*x.size, bboxes, labels=labels, classes=classes, scale=False)
        ret.scores = t[2]
        return ret
    else:
        bboxes, labels = t
        if len((labels - pad_idx).nonzero()) == 0: return ImageBBox.create(*x.size, bboxes, labels=labels, classes=classes, scale=False)
        i = (labels - pad_idx).nonzero().min()
        bboxes,labels = bboxes[i:],labels[i:]
        return ImageBBox.create(*x.size, bboxes, labels=labels, classes=classes, scale=False)


class ObjectMSItemList(ObjectItemList):
    "`ItemList` suitable for object detection on Multispectral Data."
    _label_cls,_square_show_res = ObjectDetectionCategoryList, False
    def open(self, fn):
        return ArcGISMSImage.open_gdal(fn)


def show_batch_pascal_voc_rectangles(self, rows=3, alpha=1, **kwargs): # parameters adjusted in kwargs
    nrows = rows
    ncols = kwargs.get('ncols', nrows)
    #start_index = kwargs.get('start_index', 0) # Does not work with dataloader
    
    n_items = kwargs.get('n_items', nrows*ncols)
    n_items = min(n_items, len(self.x))
    nrows = math.ceil(n_items/ncols)

    type_data_loader = kwargs.get('data_loader', 'training') # options : traininig, validation, testing
    if type_data_loader == 'training':
        data_loader = self.train_dl
    elif type_data_loader == 'validation':
        data_loader = self.valid_dl
    elif type_data_loader == 'testing':
        data_loader = self.test_dl
    else:
        e = Exception(f'could not find {type_data_loader} in data.')
        raise(e)

    rgb_bands = kwargs.get('rgb_bands', self._symbology_rgb_bands)
    nodata = kwargs.get('nodata', 0)
    imsize = kwargs.get('imsize', 5)
    statistics_type = kwargs.get('statistics_type', 'dataset') # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get('stretch_type', 'minmax') # Accepted Values `minmax`, `percentclip`
    label_font_size = kwargs.get('label_font_size', 16)

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
        b_index = self._extract_bands.index(b_index)
        symbology_bands.append(b_index)

    # Get Batch
    x_batch, y_batch = get_nbatches(data_loader, math.ceil(n_items/self.batch_size))
    x_batch = torch.cat(x_batch)
    # Denormalize X
    x_batch = (self._scaled_std_values[self._extract_bands].view(1, -1, 1, 1).to(x_batch) * x_batch ) + self._scaled_mean_values[self._extract_bands].view(1, -1, 1, 1).to(x_batch)
    y_bboxes = []
    y_classes = []
    for yb in y_batch:
        y_bboxes.extend(yb[0])
        y_classes.extend(yb[1])
    #return y_bboxes, y_classes, x_batch

    # Extract N Items and RGB Bands
    symbology_x_batch = x_batch[:(nrows*ncols), symbology_bands]
    if stretch_type is not None:
        symbology_x_batch = image_batch_stretcher(symbology_x_batch, stretch_type, statistics_type)
        # symbology_x_batch = dynamic_range_adjustment(symbology_x_batch)

    # Channel first to channel last and clamp float values to range 0 - 1 for plotting
    symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
    # Clamp float values to range 0 - 1
    if symbology_x_batch.mean() < 1:
        symbology_x_batch = symbology_x_batch.clamp(0, 1)

    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Get color Array
    color_array = self._multispectral_color_array
    color_array[1:, 3] = alpha

    # Size for plotting
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*imsize, nrows*imsize))
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            axi = axs
            if nrows == 1:
                axi = axi
            else:
                axi = axi[r]
            if ncols == 1:
                axi = axi
            else:
                axi = axi[c]
            if idx < symbology_x_batch.shape[0]:
                axi.imshow(symbology_x_batch[idx].cpu().numpy())
                classes = y_classes[idx][y_classes[idx] > 0]
                bboxes = y_bboxes[idx][y_classes[idx] > 0]
                bboxes = (bboxes+1)*.5
                bboxes = bboxes.clamp(0, 1)*(x_batch.shape[-1]-1)
                for i, bbox in enumerate(bboxes):
                    xs = bbox[[1, 1, 3, 3, 1]]
                    ys = bbox[[0, 2, 2, 0, 0]]
                    color = self._multispectral_color_array[classes[i]].tolist()
                    axi.plot(xs.cpu().numpy(), ys.cpu().numpy(), color=color, linewidth=2)
                    class_value = classes[i].item()
                    lbl = self.classes[class_value]
                    if lbl.strip()=='':
                        lbl=str(class_value)
                    axi.text(xs[0]+1, ys[0]+1+(label_font_size*(x_batch.shape[-1]-1)/256), lbl, size=label_font_size, color=color, path_effects=[patheffects.Stroke(linewidth=.5, foreground='gray')])
                axi.axis('off')
            else:
                axi.axis('off')
            idx+=1
    if _IS_ARCGISPRONOTEBOOK:
        plt.show()

def show_results_multispectral(self, nrows=5, alpha=1, **kwargs): # parameters adjusted in kwargs
    from matplotlib import pyplot as plt
    from matplotlib import patheffects

    # Get Number of items
    ncols = 2

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

    nodata = kwargs.get('nodata', 0)
    return_fig = kwargs.get('return_fig', False)

    index = kwargs.get('start_index', 0)

    imsize = kwargs.get('imsize', 4)

    thresh = kwargs.get('thresh', 0.3)

    nms_overlap = kwargs.get('nms_overlap', 0.1)

    if getattr(self, "_is_model_extension", False):
        transform_kwargs,kwargs = split_kwargs_by_func(kwargs, self.model_conf.transform_input_multispectral)

    title_font_size = 16
    _top = 1 - (math.sqrt(title_font_size)/math.sqrt(100*nrows*imsize))
    top = kwargs.get('top', _top)

    statistics_type = kwargs.get('statistics_type', 'dataset') # Accepted Values `dataset`, `DRA`
    stretch_type = kwargs.get('stretch_type', 'minmax') # Accepted Values `minmax`, `percentclip`
    label_font_size = kwargs.get('label_font_size', 16)
    
    # Get Batch
    x_batch, y_batch = get_nbatches(data_loader, math.ceil(nrows/self._data.batch_size))
    x_batch = torch.cat(x_batch)
    y_bboxes = []
    y_classes = []
    for yb in y_batch:
        y_bboxes.extend(yb[0])
        y_classes.extend(yb[1])

    nrows = min(nrows, len(x_batch))
    
    predictions_store = []
    pred_model_external = []

    for i in range(0, x_batch.shape[0], self._data.batch_size):
        
        if self._backend == 'pytorch':
            if getattr(self, "_is_model_extension", False):
                xb = self.model_conf.transform_input_multispectral(x_batch[i:i+self._data.batch_size], **transform_kwargs)
                try:
                    _pred_ext = self.learn.model.eval()(xb)
                except Exception as e:

                    if getattr(self, "_is_fasterrcnn", False):
                        _pred_ext = []
                        for _ in range(self._data.batch_size):
                            res={}
                            res['boxes'] = torch.empty(0,4)
                            res['scores'] = torch.tensor([])
                            res['labels'] = torch.tensor([])
                            _pred_ext.append(res)
                    else:
                        raise e
                analyzed_pred_ext = self._analyze_pred(_pred_ext, 
                                                    thresh=thresh, 
                                                    nms_overlap=nms_overlap, 
                                                    ret_scores=True,
                                                    device=self._device,
                                                    )
            else:
                predictions = self.learn.model.eval()(x_batch[i:i+self._data.batch_size])
        elif self._backend == 'tensorflow':
            from .fastai_tf_fit import _pytorch_to_tf_batch
            _classes_sparse, _activations = self.learn.model(_pytorch_to_tf_batch(x_batch[i:i+self._data.batch_size]))
            _classes_sparse, _activations = _classes_sparse.detach().numpy(), _activations.detach().numpy()
            predictions = (torch.tensor(_classes_sparse), torch.tensor(_activations))
        
        if getattr(self, "_is_model_extension", False):
            pred_model_external.extend(analyzed_pred_ext)
        else:
            predictions_store.append(predictions)

    if not getattr(self, "_is_model_extension", False):
        if self.__class__.__name__ == 'YOLOv3':
            predictions_store = torch.cat(predictions_store)
        else:
            __predictions_store = []
            for __batch in predictions_store:
                for __chip in zip(*__batch):
                    __predictions_store.append((__chip[0], __chip[1]))
            predictions_store = __predictions_store

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
        x_batch = (self._data._scaled_std_values[self._data._extract_bands].view(1, -1, 1, 1).to(x_batch) * x_batch ) + self._data._scaled_mean_values[self._data._extract_bands].view(1, -1, 1, 1).to(x_batch)

        # Extract RGB Bands
        symbology_x_batch = x_batch[:(nrows*ncols), symbology_bands]
        if stretch_type is not None:
            symbology_x_batch = image_batch_stretcher(symbology_x_batch, stretch_type, statistics_type)
    else:
        # normalization stats
        symbology_x_batch = denorm_x(x_batch)

    # Channel first to channel last for plotting
    symbology_x_batch = symbology_x_batch.permute(0, 2, 3, 1)
    # Clamp float values to range 0 - 1
    if symbology_x_batch.mean() < 1:
        symbology_x_batch = symbology_x_batch.clamp(0, 1)
    
    # Squeeze channels if single channel (1, 224, 224) -> (224, 224)
    if symbology_x_batch.shape[-1] == 1:
        symbology_x_batch = symbology_x_batch.squeeze()

    # Get color Array
    color_array = self._data._multispectral_color_array
    color_array[:, 3] = alpha

    # Size for plotting
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*imsize, nrows*imsize))
    fig.suptitle('Ground Truth / Predictions', fontsize=title_font_size)
    plt.subplots_adjust(top=top)
    idx=0
    for r in range(0, nrows):
        if nrows == 1:
            ax_i = axs
        else:
            ax_i = axs[r]

        # Plot Ground Truth
        ax_ground_truth = ax_i[0]
        ax_ground_truth.axis('off')
        ax_ground_truth.imshow(symbology_x_batch[idx].cpu().numpy())
        gt_classes = y_classes[idx][y_classes[idx] > 0]
        gt_bboxes = y_bboxes[idx][y_classes[idx] > 0]
        gt_bboxes = (gt_bboxes+1)*.5
        gt_bboxes = gt_bboxes.clamp(0, 1)*(x_batch.shape[-1]-1)
        for i, bbox in enumerate(gt_bboxes):
            xs = bbox[[1, 1, 3, 3, 1]]
            ys = bbox[[0, 2, 2, 0, 0]]
            color = self._data._multispectral_color_array[gt_classes[i]].tolist()
            ax_ground_truth.plot(xs.cpu().numpy(), ys.cpu().numpy(), color=color, linewidth=2, path_effects=[patheffects.Stroke(linewidth=3, foreground='black'), patheffects.Normal()])
            class_value = gt_classes[i].item()
            lbl = self._data.classes[class_value]
            if lbl.strip() == '':
                lbl = str(class_value)
            ax_ground_truth.text(xs[0]+1, ys[0]+1+(label_font_size*(x_batch.shape[-1]-1)/256), lbl, size=label_font_size, color=color, path_effects=[patheffects.Stroke(linewidth=1, foreground='black'), patheffects.Normal()])

        # Plot Predictions
        ax_prediction  = ax_i[1]
        ax_prediction.axis('off')
        ax_prediction.imshow(symbology_x_batch[idx].cpu().numpy())

        if getattr(self, "_is_model_extension", False):
            analyzed_prediction = pred_model_external[idx]
        else:
            analyzed_prediction = self._analyze_pred(predictions_store[idx], 
                                                    thresh=thresh, 
                                                    nms_overlap=nms_overlap, 
                                                    ret_scores=True, 
                                                    device=self._device
                                                    )

        if analyzed_prediction is not None:
            predicted_bboxes, predicted_classes, predicted_confidences = analyzed_prediction
            predicted_bboxes = (predicted_bboxes+1)*.5
            predicted_bboxes = predicted_bboxes.clamp(0, 1)*(x_batch.shape[-1]-1)
            if len(predicted_bboxes) > 0:
                for i, bbox in enumerate(predicted_bboxes):
                    xs = bbox[[1, 1, 3, 3, 1]]
                    ys = bbox[[0, 2, 2, 0, 0]]
                    color = self._data._multispectral_color_array[predicted_classes[i]].tolist()
                    ax_prediction.plot(xs.detach().cpu().numpy(), ys.detach().cpu().numpy(), color=color, linewidth=2, path_effects=[patheffects.Stroke(linewidth=3, foreground='black'), patheffects.Normal()])
                    class_value = predicted_classes[i].item()
                    lblp = self._data.classes[class_value]
                    if lblp.strip() == '':
                        lblp = str(class_value)
                    ax_prediction.text(xs[0]+1, ys[0]+1+(label_font_size*(x_batch.shape[-1]-1)/256), lblp, size=label_font_size, color=color, path_effects=[patheffects.Stroke(linewidth=1, foreground='black'), patheffects.Normal()])
            
        idx+=1
    if _IS_ARCGISPRONOTEBOOK:
        plt.show()
    return fig,axs
