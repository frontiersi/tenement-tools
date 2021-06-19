from logging import exception
from fastai.vision import ItemBase, ItemList, Tensor, ImageList, Tuple, Path, get_transforms, random, open_image, Image, math, plt, torch, Learner, partial, optim, ifnone
from .._utils.common import ArcGISImageList, ArcGISMSImage
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from .._utils.cyclegan import get_activations, InceptionV3
import os
import json

class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj,self.data = (img1,img2),[-1+2*img1.data,-1+2*img2.data]
        self.data2 = [-1+2*img2.data,-1+2*img1.data]
        self.shape = img1.shape
    
    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, **kwargs)
        return self
    
    def to_one(self):
        return ArcGISMSImage(0.5+torch.cat(self.data,2)/2)
    def to_one_pred(self): 
        return ArcGISMSImage(0.5+(self.data2[0])/2)
    
    def __repr__(self):
         return f'{self.__class__.__name__}{(self.img1.shape, self.img2.shape)}'

class TargetTupleList(ItemList):
    def reconstruct(self, t:Tensor): 
        if len(t.size()) == 0: return t
        return ImageTuple(ArcGISMSImage(t[0]/2+0.5),ArcGISMSImage(t[1]/2+0.5))

class ImageTupleList2(ImageList):
    _label_cls=TargetTupleList
    def __init__(self, items, itemsB=None, itemsB_valid=None, **kwargs):
        self.itemsB = itemsB
        self.itemsB_valid = itemsB_valid
        super().__init__(items, **kwargs)
    
    def new(self, items, **kwargs):
        return super().new(items, itemsB=self.itemsB, itemsB_valid=self.itemsB_valid, **kwargs)
    
    def get(self, i):
        
        if len(self.items) == len(self.itemsB):
            img1 = super().get(i)
            fn = self.itemsB[i]
        else:
            img1 = super().get(i)
            fn = self.itemsB_valid[i]
        return ImageTuple(img1, open_image(fn))
    
    def reconstruct(self, t:Tensor): 
        return ImageTuple(Image(t[0]/2+0.5),Image(t[1]/2+0.5))
    
    @classmethod
    def from_folders(cls, path, folderA, folderB, **kwargs):
        itemsB = ImageList.from_folder(folderB).items
        res = super().from_folder(folderA, itemsB=itemsB, itemsB_valid =itemsB, **kwargs)
        # The parent dir of the path below (i.e. 'path') is the working dir for saving the model
        if is_old_format_pix2pix(path):
            res.path = path/'Images'
        return res
    
    def split_by_idxs(self, train_idx, valid_idx):
        "Split the data between `train_idx` and `valid_idx`."
        self.itemsB_valid = self.itemsB_valid[valid_idx]
        self.itemsB = self.itemsB[train_idx]
        return self.split_by_list(self[train_idx], self[valid_idx])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()
    
    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        
        figsize = ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,z) in enumerate(zip(xs,zs)):
            x.to_one().show(ax=axs[i,0], **kwargs)
            z.to_one_pred().show(ax=axs[i,1], **kwargs)
    
_batch_stats_a = None
_batch_stats_b = None

class ImageTupleListMS2(ArcGISImageList):
    _label_cls=TargetTupleList
    def __init__(self, items, itemsB=None, itemsB_valid=None, **kwargs):
        self.itemsB = itemsB
        self.itemsB_valid = itemsB_valid
        super().__init__(items, **kwargs)

    def new(self, items, **kwargs):
        return super().new(items, itemsB=self.itemsB, itemsB_valid=self.itemsB_valid, **kwargs)

    def get(self, i):
        
        if len(self.items) == len(self.itemsB):
            fn1 = self.items[i]
            img1 = ArcGISMSImage.open(fn1)
            fn = self.itemsB[i]
            img2 = ArcGISMSImage.open(fn)
            if img1.shape[0] < img2.shape[0]:
                cont = []
                last_tile = np.expand_dims(img1.data[img1.shape[0]-1,:,:], 0)
                res = abs(img2.shape[0] - img1.shape[0])
                for i in range(res):
                    img1 = ArcGISMSImage(torch.tensor(np.concatenate((img1.data, last_tile), axis=0)))
            if img2.shape[0] < img1.shape[0]:
                cont = []
                last_tile = np.expand_dims(img2.data[img2.shape[0]-1,:,:], 0)
                res = abs(img1.shape[0] - img2.shape[0])
                for i in range(res):
                    img2 = ArcGISMSImage(torch.tensor(np.concatenate((img2.data, last_tile), axis=0)))
        else:
            fn1 = self.items[i]
            img1 = ArcGISMSImage.open(fn1)
            fn = self.itemsB_valid[i]
            img2 = ArcGISMSImage.open(fn)
            if img1.shape[0] < img2.shape[0]:
                cont = []
                last_tile = np.expand_dims(img1.data[img1.shape[0]-1,:,:], 0)
                res = abs(img2.shape[0] - img1.shape[0])
                for i in range(res):
                    img1 = ArcGISMSImage(torch.tensor(np.concatenate((img1.data, last_tile), axis=0)))
            if img2.shape[0] < img1.shape[0]:
                cont = []
                last_tile = np.expand_dims(img2.data[img2.shape[0]-1,:,:], 0)
                res = abs(img1.shape[0] - img2.shape[0])
                for i in range(res):
                    img2 = ArcGISMSImage(torch.tensor(np.concatenate((img2.data, last_tile), axis=0)))
        global _batch_stats_a
        global _batch_stats_b
        img1_scaled = _tensor_scaler_tfm(img1.data, min_values=_batch_stats_a['band_min_values'], max_values=_batch_stats_a['band_max_values'], mode='minmax')
        img2_scaled = _tensor_scaler_tfm(img2.data, min_values=_batch_stats_b['band_min_values'], max_values=_batch_stats_b['band_max_values'], mode='minmax')
        img1_scaled = ArcGISMSImage(img1_scaled)
        img2_scaled = ArcGISMSImage(img2_scaled)
        return ImageTuple(img1_scaled, img2_scaled)
    
    def reconstruct(self, t:Tensor): 
        return ImageTuple(ArcGISMSImage(t[0]/2+0.5),ArcGISMSImage(t[1]/2+0.5))
    
    @classmethod
    def from_folders(cls, path, folderA, folderB, batch_stats_a, batch_stats_b, **kwargs):
        itemsB = ImageList.from_folder(folderB).items
        res = super().from_folder(folderA, itemsB=itemsB, itemsB_valid=itemsB, **kwargs)
        # The path below (i.e. 'path') is the working dir for saving the model
        res.path = path
        global _batch_stats_a
        global _batch_stats_b
        _batch_stats_a = batch_stats_a
        _batch_stats_b = batch_stats_b
        return res
    
    def split_by_idxs(self, train_idx, valid_idx):
        "Split the data between `train_idx` and `valid_idx`."
        self.itemsB_valid = self.itemsB_valid[valid_idx]
        self.itemsB = self.itemsB[train_idx]
        return self.split_by_list(self[train_idx], self[valid_idx])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        
        figsize = ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,z) in enumerate(zip(xs,zs)):
            x.to_one().show(ax=axs[i,0], **kwargs)
            z.to_one_pred().show(ax=axs[i,1], **kwargs)

def calculate_activation_statistics(batch_size, data_len, batch_list):
    act = get_activations(batch_size, data_len, batch_list)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def _tensor_scaler_tfm(tensor_batch, min_values, max_values, mode='minmax'):
    from .._data import _tensor_scaler
    x = tensor_batch
    if x.shape[0] > min_values.shape[0]:
        res = x.shape[0] - min_values.shape[0]
        last_val = torch.tensor([min_values[min_values.shape[0]-1]])
        for i in range(res):
            min_values = torch.tensor(np.concatenate((min_values, last_val), axis=0))
    if x.shape[0] > max_values.shape[0]:
        res = x.shape[0] - max_values.shape[0]
        last_val = torch.tensor([max_values[max_values.shape[0]-1]])
        for i in range(res):
            max_values = torch.tensor(np.concatenate((max_values, last_val), axis=0))
    max_values = max_values.view(-1, 1, 1).to(x.device)
    min_values = min_values.view(-1, 1, 1).to(x.device)
    x = _tensor_scaler(x, min_values, max_values, mode, create_view=False)
    return x

def _batch_stats_json(path, img_list, norm_pct, stats_file_name="esri_normalization_stats.json"):
    from .._data import _get_batch_stats
    if len(img_list) < 300:
        norm_pct = 1

    dummy_stats = {
                "batch_stats_for_norm_pct_0" : {
                    "band_min_values":None, 
                    "band_max_values":None, 
                    "band_mean_values":None, 
                    "band_std_values":None, 
                    "scaled_min_values":None, 
                    "scaled_max_values":None, 
                    "scaled_mean_values":None, 
                    "scaled_std_values":None}}

    normstats_json_path = os.path.abspath(path / '..' / stats_file_name)

    if not os.path.exists(normstats_json_path):       
        normstats = dummy_stats
        with open(normstats_json_path, 'w', encoding='utf-8') as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)
    else:
        with open(normstats_json_path) as f:
                normstats = json.load(f)

    norm_pct_search = f"batch_stats_for_norm_pct_{round(norm_pct*100)}"
    if norm_pct_search in normstats:
        batch_stats = normstats[norm_pct_search]
        for s in batch_stats:
            if batch_stats[s] is not None:
                batch_stats[s] = torch.tensor(batch_stats[s])
    else:
        batch_stats = _get_batch_stats(img_list, norm_pct)
        normstats[norm_pct_search] = dict(batch_stats)
        for s in normstats[norm_pct_search]:
            if normstats[norm_pct_search][s] is not None:
                normstats[norm_pct_search][s] = normstats[norm_pct_search][s].tolist()
        with open(normstats_json_path, 'w', encoding='utf-8') as f:
            json.dump(normstats, f, ensure_ascii=False, indent=4)

    return batch_stats

def prepare_data_ms_pix2pix(path, norm_pct, val_split_pct, seed, databunch_kwargs):
    path_a, path_b = pix2pix_paths(path)

    img_list_a = ArcGISImageList.from_folder(path_a)
    img_list_b = ArcGISImageList.from_folder(path_b)
    
    batch_stats_a = _batch_stats_json(path_a, img_list_a, norm_pct, stats_file_name="esri_normalization_stats_a.json")
    batch_stats_b = _batch_stats_json(path_b, img_list_b, norm_pct, stats_file_name="esri_normalization_stats_b.json")

    data = ImageTupleListMS2.from_folders(path, path_a, path_b, batch_stats_a, batch_stats_b)\
            .split_by_rand_pct(val_split_pct, seed=seed)\
            .label_empty()\
            .databunch(**databunch_kwargs)

    data._band_min_values = batch_stats_a['band_min_values']
    data._band_max_values = batch_stats_a['band_max_values']
    data._band_mean_values = batch_stats_a['band_mean_values']
    data._band_std_values = batch_stats_a['band_std_values']
    data._scaled_min_values = batch_stats_a['scaled_min_values']
    data._scaled_max_values = batch_stats_a['scaled_max_values']
    data._scaled_mean_values = batch_stats_a['scaled_mean_values']
    data._scaled_std_values = batch_stats_a['scaled_std_values']

    data._band_min_values_b = batch_stats_b['band_min_values']
    data._band_max_values_b = batch_stats_b['band_max_values']
    data._band_mean_values_b = batch_stats_b['band_mean_values']
    data._band_std_values_b = batch_stats_b['band_std_values']
    data._scaled_min_values_b = batch_stats_b['scaled_min_values']
    data._scaled_max_values_b = batch_stats_b['scaled_max_values']
    data._scaled_mean_values_b = batch_stats_b['scaled_mean_values']
    data._scaled_std_values_b = batch_stats_b['scaled_std_values']
    
    # add dataset_type
    data._dataset_type = 'Pix2Pix'

    return data

def is_old_format_pix2pix(path):
    """
    Function that returns 'True' if the manually created dir structure is being used.
    """
    return os.path.exists(path/'Images'/'train_a') and os.path.exists(path/'Images'/'train_b')

def pix2pix_paths(path):
    """
    Function to return the appropriate image dir paths in the provided dataset dir.
    """
    if is_old_format_pix2pix(path):
        return (path/'Images'/'train_a', path/'Images'/'train_b')
    else:
        return (path/'images', path/'images2')

def folder_check_pix2pix(path):
    """
    Function to check if the correct dir structure is provided.
    """
    img_folder1 = os.path.exists(path/'Images'/'train_a') or os.path.exists(path/'images')
    img_folder2 = os.path.exists(path/'Images'/'train_b') or os.path.exists(path/'images2')
    if not all([img_folder1, img_folder2]):
        raise Exception(f"""You might be using an incorrect format to train your model. \nPlease ensure your training data has the following folder structure:
                ├─dataset folder name
                    ├─images
                    ├─images2   """)

def rgb_or_ms(im_path):
    """
    Function that returns the imagery type (RGB or ms) of an image.
    """
    try:
        import gdal
        ds = gdal.Open(im_path)
        if ds.RasterCount!=3 or ds.GetRasterBand(1).DataType != gdal.GDT_Byte:
            return 'ms'
        else:
            return 'RGB'
    except:
        return None