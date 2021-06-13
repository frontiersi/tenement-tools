import os
import re
from pathlib import Path
from functools import partial
import xml.etree.ElementTree as ET
import math
import sys
import json 
import logging
import tempfile
import types
import traceback
import copy

from ._utils.env import ARCGIS_ENABLE_TF_BACKEND

import_exception = None
try:
    import arcgis
    import numpy as np
    from fastai.vision.data import imagenet_stats, ImageList, bb_pad_collate
    from fastai.vision.transform import crop, rotate, dihedral_affine, brightness, contrast, skew, rand_zoom, get_transforms, flip_lr, ResizeMethod
    from fastai.vision import ImageDataBunch, parallel, ifnone
    from fastai.core import Category
    import fastai.vision
    from fastai.torch_core import data_collate
    import torch
    from .models._unet_utils import ArcGISSegmentationItemList, is_no_color
    from .models._maskrcnn_utils import ArcGISInstanceSegmentationItemList, ArcGISInstanceSegmentationMSItemList
    from ._utils._ner_utils import _NERData
    from ._utils.pascal_voc_rectangles import ObjectDetectionItemList
    from .models._superres_utils import resize_one
    from ._utils.common import ArcGISMSImage, ArcGISImageList
    from ._utils.env import HAS_GDAL, raise_gdal_import_error
    from ._utils.classified_tiles import show_batch_classified_tiles
    from ._utils.labeled_tiles import show_batch_labeled_tiles
    from ._utils.rcnn_masks import show_batch_rcnn_masks
    from ._utils.pascal_voc_rectangles import ObjectMSItemList, show_batch_pascal_voc_rectangles
    from ._utils.pointcloud_data import pointcloud_prepare_data
    from ._utils.superres import ImageImageListSR
    from fastai.tabular import TabularDataBunch
    from fastai.tabular.transform import FillMissing, Categorify, Normalize
    from fastai.tabular import cont_cat_split, add_datepart
    from ._utils.tabular_data import TabularDataObject
    from ._utils.text_data import TextDataObject
    from ._utils.cyclegan import ImageTupleList, prepare_data_ms_cyclegan
    from ._utils.pix2pix import ImageTupleList2, prepare_data_ms_pix2pix
    import random
    import PIL
    HAS_FASTAI = True
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False


band_abrevation_lib = {
    'b': 'BLUE',
    'c': 'CIRRUS',
    'ca': 'COASTAL AEROSOL',
    'g': 'GREEN',
    'nir': 'NEAR INFRARED',
    'nnir': 'NARROW NEAR INFRARED',
    'p': 'PANCHROMATIC',
    'r': 'RED',
    'swir': 'SHORT WAVELENGTH INFRARED',
    'swirc': 'SHORT WAVELENGTH INFRARED – Cirrus',
    'tir': 'THERMAL INFRARED',
    'vre': 'Vegetation red edge',
    'wv': 'WATER VAPOUR'
}

imagery_type_lib = {
    'landsat8': {
        "bands": ['ca', 'b', 'g', 'r', 'nir', 'swir', 'swir', 'c', 'qa', 'tir', 'tir'],
        "bands_info": { # incomplete
        }
    },
    "naip": {
        "bands": ['r', 'g', 'b', 'nir'],
        "bands_info": { # incomplete
        }
    },
    'sentinel2': { 
        "bands": ['ca', 'b', 'g', 'r', 'vre', 'vre', 'vre', 'nir', 'nnir', 'wv', 'swirc', 'swir', 'swir'],
        "bands_info": { # incomplete
            "b1": {
                "Name": "costal",
                "max": 10000,
                "min": 10000
            },            
            "b2": {
                "Name": "blue",
                "max": 10000,
                "min": 10000
            }
        }
    }
}


def get_installation_command():
    installation_steps = ('\nPlease install all required dependencies by following the instructions at: \nhttps://developers.arcgis.com/python/guide/install-and-set-up/#Install-deep-learning-dependencies')
    return installation_steps 


def _raise_fastai_import_error(import_exception=import_exception):
    installation_steps = get_installation_command()
    raise Exception(f"{import_exception} \n\nDeep learning dependencies are missing. This module requires fastai, PyTorch, torchvision "
                    f"as its dependencies.\n{installation_steps}")

def _raise_conda_import_error(import_exception=import_exception):
    installation_steps = ('\nAdditionally, please ensure all required deep learning dependencies are installed by following the instructions at: \nhttps://developers.arcgis.com/python/guide/install-and-set-up/#Install-deep-learning-dependencies\n')
    raise Exception(f"""{import_exception} \n\nThis module requires conda, python 3.7 and currently supported on Windows.\n{installation_steps}""")

class _ImagenetCollater():
    def __init__(self, chip_size):
        self.chip_size = chip_size
    def __call__(self, batch):
        _xb = []
        for sample in batch:
            data = sample[0].data
            if data.shape[1] < self.chip_size or data.shape[2] < self.chip_size:
                data = sample[0].resize(self.chip_size).data
            _xb.append(data)
        _xb = torch.stack(_xb)
        if isinstance(sample[1],Category):
            _yb = torch.stack([torch.tensor(sample[1].data) for sample in batch])
        else:
            _yb = torch.stack([torch.tensor(sample[1]) for sample in batch])
        return _xb, _yb

def _bb_pad_collate(samples, pad_idx=0):
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    if isinstance(samples[0][1], int):
        return data_collate(samples)
    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data

        if not (bbs.nelement() == 0) or list(bbs) == [[0,0,0,0]]:
            bboxes[i,-len(lbls):] = bbs
            labels[i,-len(lbls):] = torch.tensor(lbls, device=bbs.device).long()
    return torch.cat(imgs,0), (bboxes,labels)    


def _get_bbox_classes(label_file, class_mapping , height_width=[], **kwargs):
    dataset_type = kwargs.get('dataset_type', None)

    if dataset_type == 'KITTI_rectangles':
        classes = []
        truncated = []
        occluded = []
        obs_angle = []
        bboxes = []
        hwl = []
        d_xyz = []
        occluded = []
        rot_yaxis = []     
        start_space = re.compile('^\s+') #pattern to capture leading spaces
        spaces_to_be_replaced = re.compile('(?<=\d)(\s+)(?=\d)') #pattern to capture spaces between numeric values

        with open(label_file) as f: #reading the bbox and class labels
            lines = f.readlines()
        
        for line in lines:
            line = re.sub(start_space, '', line)  
            lst = re.sub(spaces_to_be_replaced, ',', line).split(',') #reading kitti labels from string to a list
            xmin, ymin, xmax, ymax = [float(n) for n in lst[4:8]] #reading bbox coordinates
            hieght, width, length = [float(n) for n in lst[8:11]] 
            x, y, z = [float(n) for n in lst[11:14]]

            data_class_text = str(lst[0])
            if (not data_class_text.isnumeric() and not class_mapping.get(data_class_text))\
                or (data_class_text.isnumeric() and not(class_mapping.get(data_class_text) or class_mapping.get(int(data_class_text)))):
                continue
            if data_class_text.isnumeric():
                data_class_mapping = class_mapping[data_class_text] if class_mapping.get(data_class_text) else class_mapping[int(data_class_text)]
            else:
                data_class_mapping = class_mapping[data_class_text]
            
            classes.append(data_class_mapping) #object class
            truncated.append(float(lst[1]))    #if the object is truncated
            obs_angle.append(float(lst[2]))    #onservation angle
            occluded.append(float(lst[3]))     #if the object is occluded
            bboxes.append([ymin, xmin, ymax, xmax])
            height_width.append(((xmax - xmin)*1.25, (ymax - ymin)*1.25))
            hwl.append([hieght, width, length])
            d_xyz.append([x, y, z])
            rot_yaxis.append(float(lst[14]))   #angle of rotation along y axis
    else:
        tree = ET.parse(label_file)
        xmlroot = tree.getroot()
        bboxes = []
        classes = []
        for tag_obj in xmlroot.findall('object'):
            bnd_box = tag_obj.find('bndbox')
            xmin, ymin, xmax, ymax = float(bnd_box.find('xmin').text), \
                                    float(bnd_box.find('ymin').text), \
                                    float(bnd_box.find('xmax').text), \
                                    float(bnd_box.find('ymax').text)
            data_class_text = tag_obj.find('name').text

            if (not data_class_text.isnumeric() and not class_mapping.get(data_class_text))\
                or (data_class_text.isnumeric() and not(class_mapping.get(data_class_text) or class_mapping.get(int(data_class_text)))):
                continue

            if data_class_text.isnumeric():
                data_class_mapping = class_mapping[data_class_text] if class_mapping.get(data_class_text) else class_mapping[int(data_class_text)]
            else:
                data_class_mapping = class_mapping[data_class_text]

            classes.append(data_class_mapping)
            bboxes.append([ymin, xmin, ymax, xmax])
            height_width.append(((xmax - xmin)*1.25, (ymax - ymin)*1.25))

    if len(bboxes) == 0:
        return [[[0., 0., 0., 0.]], [list(class_mapping.values())[0]]]
    return [bboxes, classes]


def _get_bbox_lbls(imagefile, class_mapping, height_width, **kwargs):
    dataset_type = kwargs.get('dataset_type', None)
    if dataset_type == 'KITTI_rectangles':
        label_suffix = '.txt'
    else:
        label_suffix = '.xml'
    label_file = imagefile.parents[1] / 'labels' / imagefile.name.replace('{ims}'.format(ims=imagefile.suffix), label_suffix)
    return _get_bbox_classes(label_file, class_mapping, height_width, dataset_type=dataset_type)


def _get_lbls(imagefile, class_mapping):
    xmlfile = imagefile.parents[1] / 'labels' / imagefile.name.replace('{ims}'.format(ims=imagefile.suffix), '.xml')
    return _get_bbox_classes(xmlfile, class_mapping)[1][0]


def _get_multi_lbls(imagefile):
    """
    Function that returns class labels for an image for multilabel classification.
    input: imagefile (Path)
    returns: labels (List[str])
    """
    xmlfile = imagefile.parents[1] / 'labels' / imagefile.name.replace('{ims}'.format(ims=imagefile.suffix), '.xml')
    labels = ET.parse(xmlfile).getroot().find('object').find('name').text
    labels = labels.split(',')
    return labels


def _check_esri_files(path):
    if os.path.exists(path / 'esri_model_definition.emd') \
        and os.path.exists(path / 'map.txt') \
            and os.path.exists(path / 'esri_accumulated_stats.json'):
        return True

    return False


def _get_class_mapping(path, **kwargs):
    '''getting class mapping from labels, incase esri definition files and class mapping is not provided'''
    dataset_type = kwargs.get('dataset_type', None)
    class_mapping = {}
    if dataset_type == 'KITTI_rectangles':
        start_space = re.compile('^\s+') #pattern to capture leading spaces
        spaces_to_be_replaced = re.compile('(?<=\d)(\s+)(?=\d)') #pattern to capture spaces between numeric values

        for txtfile in os.listdir(path): 
            if not txtfile.endswith('.txt'):
                continue
            with open(os.path.join(path , txtfile)) as f:
                lines = f.readlines()
            for line in lines:
                line = re.sub(start_space, '', line)
                lst = re.sub(spaces_to_be_replaced, ',', line).split(',')
                class_mapping[str(lst[0])] = str(lst[0])
    else:
        for xmlfile in os.listdir(path):
            if not xmlfile.endswith('.xml'):
                continue
            tree = ET.parse(os.path.join(path, xmlfile))
            xmlroot = tree.getroot()
            for tag_obj in xmlroot.findall('object'):
                class_mapping[tag_obj.find('name').text] = tag_obj.find('name').text

    return class_mapping


def _get_batch_stats(image_list, 
                     norm_pct=1, 
                     _band_std_values=False, 
                     scaled_std=True, 
                     reshape=True):
    n_normalization_samples = round(len(image_list)*norm_pct)
    #n_normalization_samples = max(256, n_normalization_samples)
    random_indexes = np.random.randint(0, len(image_list), size=min(n_normalization_samples, len(image_list)))

    # Original Band Stats
    min_values_store = []
    max_values_store = []
    mean_values_store = []

    data_shape = image_list[0].data.shape
    n_bands = data_shape[0]
    feasible_chunk = round(512*4*400/(n_bands*data_shape[1])) # ~3gb footprint
    chunk = min(feasible_chunk, n_normalization_samples)
    i = 0
    n_c = image_list[0].data.shape[0]
    for i in range(0, n_normalization_samples, chunk):
        if reshape:
            x_tensor_chunk = torch.cat([x.data.view(n_c, -1) for x in image_list[random_indexes[i:i+chunk]]], dim=1)
            min_values = x_tensor_chunk.min(dim=1).values
            max_values = x_tensor_chunk.max(dim=1).values
            mean_values = x_tensor_chunk.mean(dim=1)
        else:
            """
            min_values = torch.zeros(n_bands)
            max_values = torch.zeros(n_bands)
            mean_values = torch.zeros(n_bands)
            for bi in range(n_bands):
                min_values[bi] = x_tensor_chunk[:, bi].min()
                max_values[bi] = x_tensor_chunk[:, bi].max()
                mean_values[bi] = x_tensor_chunk[:, bi].mean()
            """          
            x_tensor_chunk = torch.stack([x.data for x in image_list[random_indexes[i:i+chunk]]])
            min_values = x_tensor_chunk.min(dim=0)[0].min(dim=1)[0].min(dim=1)[0]
            max_values = x_tensor_chunk.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
            mean_values = x_tensor_chunk.mean((0, 2, 3))
        min_values_store.append(min_values)
        max_values_store.append(max_values)
        mean_values_store.append(mean_values)

    band_max_values = torch.stack(max_values_store).max(dim=0)[0]
    band_min_values = torch.stack(min_values_store).min(dim=0)[0]
    band_mean_values = torch.stack(mean_values_store).mean(dim=0)
    
    view_shape = _get_view_shape(image_list[0].data, band_mean_values)

    if _band_std_values:
        std_values_store = []
        for i in range(0, n_normalization_samples, chunk):
            x_tensor_chunk = torch.stack([ x.data for x in image_list[random_indexes[i:i+chunk]] ] )
            std_values = (x_tensor_chunk - band_mean_values.view(view_shape)).pow(2).sum((0, 2, 3))
            std_values_store.append(std_values)
        band_std_values = (torch.stack(std_values_store).sum(dim=0) / ((n_normalization_samples * data_shape[1] * data_shape[2])-1)).sqrt()
    else:
        band_std_values = None

    # Scaled Stats
    scaled_min_values = torch.tensor([0 for i in range(n_bands)], dtype=torch.float32)
    scaled_max_values = torch.tensor([1 for i in range(n_bands)], dtype=torch.float32)
    scaled_mean_values = _tensor_scaler(band_mean_values, band_min_values, band_max_values, mode='minmax')
    
    if scaled_std:
        scaled_std_values_store = []
        for i in range(0, n_normalization_samples, chunk):
            x_tensor_chunk = torch.stack([ x.data for x in image_list[random_indexes[i:i+chunk]] ] )
            x_tensor_chunk = _tensor_scaler(x_tensor_chunk, band_min_values, band_max_values, mode='minmax')
            std_values = (x_tensor_chunk - scaled_mean_values.view(view_shape)).pow(2).sum((0, 2, 3))
            scaled_std_values_store.append(std_values)
        scaled_std_values = (torch.stack(scaled_std_values_store).sum(dim=0) / ((n_normalization_samples * data_shape[1] * data_shape[2])-1)).sqrt()
    else:
        scaled_std_values = None

    #return band_min_values, band_max_values, band_mean_values, band_std_values, scaled_min_values, scaled_max_values, scaled_mean_values, scaled_std_values
    return {
        "band_min_values": band_min_values, 
        "band_max_values": band_max_values, 
        "band_mean_values": band_mean_values, 
        "band_std_values": band_std_values, 
        "scaled_min_values": scaled_min_values, 
        "scaled_max_values": scaled_max_values, 
        "scaled_mean_values": scaled_mean_values, 
        "scaled_std_values": scaled_std_values
    }

def sniff_rgb_bands(band_names):
    band_mapping_reverse = {k.lower(): i for i, k in enumerate(band_names)}
    rgb_bands = []
    for b in [
        'red',
        'green',
        'blue'
    ]:
        bi = band_mapping_reverse.get(b, None)
        if bi is None:
            return
        rgb_bands.append(bi)
    return rgb_bands

def data_is_multispectral(emd_info: dict) -> bool:
    """
    :param emd_info: Dictionary containing EMD info
    :return: Boolean value denoting whether data is multispectral or not.
    """
    is_multispectral = False
    if not 'InputRastersProps' in emd_info:
        return is_multispectral
    if len(emd_info['InputRastersProps']["BandNames"]) !=3:
        is_multispectral = True
    return is_multispectral

def _get_view_shape(tensor_batch, band_factors):
    view_shape = [1 for i in range(len(tensor_batch.shape))]
    view_shape[tensor_batch.shape.index(band_factors.shape[0])] = band_factors.shape[0]
    return tuple(view_shape)


def _tensor_scaler(tensor_batch, min_values, max_values, mode='minmax', create_view=True):
    if create_view:
        view_shape = _get_view_shape(tensor_batch, min_values)
        max_values = max_values.view(view_shape)
        min_values = min_values.view(view_shape)
    if mode=='minmax':
        # new_value = (((old_value - old_min) / (old_max-old_min))*(new_max-new_min))+new_min
        scaled_tensor_batch = (tensor_batch - min_values) / ( (max_values - min_values) + 1e-05)
    return scaled_tensor_batch


def _tensor_scaler_tfm(tensor_batch, min_values, max_values, mode='minmax'):
    x = tensor_batch[0]
    y = tensor_batch[1]
    max_values = max_values.view(1, -1, 1, 1).to(x.device)
    min_values = min_values.view(1, -1, 1, 1).to(x.device)
    x = _tensor_scaler(x, min_values, max_values, mode, create_view=False)
    return (x, y)

def _extract_bands_tfm(tensor_batch, band_indices):
    x_batch = tensor_batch[0][:, band_indices]
    y_batch = tensor_batch[1]
    return (x_batch, y_batch)

def _make_folder(path):
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        _make_folder(os.path.dirname(path))
    if not os.path.exists(path):
        os.mkdir(path)

_models_dir = 'models'
def _prepare_working_dir(path):
    path = os.path.abspath(path)
    _make_folder(os.path.join(path, _models_dir))
    temp_folder = tempfile.TemporaryDirectory(prefix=os.path.join(path, 'arcgisTemp_'))
    return temp_folder

def merge_emd_and_stats(data_folders):
    emd_store = {}
    stats_store = {}
    for i, data_folder in enumerate(data_folders):
        json_file = data_folder / 'esri_model_definition.emd'
        if json_file.exists():
            with open(json_file) as f:
                emd = json.load(f)
                emd_store[i] = emd
            with open(data_folder / 'esri_accumulated_stats.json') as f:
                stats_store[i] = json.load(f)
    emd_keys = list(emd_store.keys())
    #
    for i, k in enumerate(emd_keys[:-1]):
        if 'BandNames' in emd_store[k].get('InputRastersProps', {}):
            props_matched = emd_store[k]['InputRastersProps']['BandNames'] == \
                            emd_store[emd_keys[i + 1]]['InputRastersProps'][
                                'BandNames']
            if not props_matched:
                logger = logging.getLogger()
                logger.warning(
                    f'"InputRastersProps" does not match between {data_folders[emd_keys[i]]} and {data_folders[emd_keys[i + 1]]}')
    # Create master EMD and esri_accumulated_stats
    emd = emd_store[emd_keys[0]]
    eas = stats_store[emd_keys[0]]
    _class_hash = {x['Value']: x for x in emd['Classes']}
    for k in emd_keys[1:]:
        _emd = emd_store[k]
        for class_entry in _emd['Classes']:
            if class_entry['Value'] not in _class_hash:
                _class_hash[class_entry['Value']] = class_entry
        #
        _eas = stats_store[k]
        for i in range(len(eas.get('BandStatsState', []))):
            eas['BandStatsState'][i]['Min'] = min(eas['BandStatsState'][i]['Min'], _eas['BandStatsState'][i]['Min'])
            eas['BandStatsState'][i]['Max'] = max(eas['BandStatsState'][i]['Max'], _eas['BandStatsState'][i]['Max'])
            eas['BandStatsState'][i]['M1'] = ((eas['BandStatsState'][i]['M1'] * eas['BandStatsState'][i]['Num']) + (
                    _eas['BandStatsState'][i]['M1'] * _eas['BandStatsState'][i]['Num'])) / (
                                                     eas['BandStatsState'][i]['Num'] +
                                                     _eas['BandStatsState'][i]['Num'])  # Mean
            eas['BandStatsState'][i]['M2'] = eas['BandStatsState'][i]['M2'] + _eas['BandStatsState'][i][
                'M2']  # Variance
            eas['BandStatsState'][i]['Num'] = eas['BandStatsState'][i]['Num'] + _eas['BandStatsState'][i][
                'Num']  # Number of pixels
        eas['NumClasses'] = max(eas['NumClasses'], _eas['NumClasses'])
        eas['NumTiles'] += _eas['NumTiles']
        #
        stats_key1 = None
        stats_key1_1 = None
        stats_key2 = None
        stats_key2_1 = None
        if 'ClassPixelStats' in eas:
            stats_key1 = 'ClassPixelStats'
            stats_key1_1 = 'NumPixelsPerClass'
        elif 'FeatureStats' in eas:
            stats_key1 = 'FeatureStats'
            stats_key1_1 = 'NumFeaturesPerClass'
        if 'ClassPixelStats' in _eas:
            stats_key2 = 'ClassPixelStats'
            stats_key2_1 = 'NumPixelsPerClass'
        elif 'FeatureStats' in _eas:
            stats_key2 = 'FeatureStats'
            stats_key2_1 = 'NumFeaturesPerClass'
        if stats_key1 is not None and stats_key2 is not None:
            eas[stats_key1]["NumImagesTotal"] += _eas[stats_key2]["NumImagesTotal"]
            for i in range(eas[stats_key1].get('NumClasses', 0)):
                eas[stats_key1]["NumImagesPerClass"][i] += _eas[stats_key2]["NumImagesPerClass"][i]
                eas[stats_key1][stats_key1_1][i] += _eas[stats_key2][stats_key2_1][i]
    #
    for i in range(len(eas.get('BandStatsState', []))):
        emd['AllTilesStats'][i]['Min'] = eas['BandStatsState'][i]['Min']
        emd['AllTilesStats'][i]['Max'] = eas['BandStatsState'][i]['Max']
        emd['AllTilesStats'][i]['Mean'] = eas['BandStatsState'][i]['M1']
        emd['AllTilesStats'][i]['StdDev'] = (eas['BandStatsState'][i]['M2'] / eas['BandStatsState'][i]['Num']) ** .5
    emd['Classes'] = [_class_hash[x] for x in sorted(_class_hash)]
    path = Path(data_folders[emd_keys[0]])  # First folder that has esri files
    return emd, eas, path

def prepare_textdata(
        path,
        task,
        text_columns,
        label_columns,
        train_file="train.csv",
        valid_file=None,
        val_split_pct=0.1,
        seed=42,
        batch_size=8,
        process_labels=False,
        remove_html_tags=False,
        remove_urls=False,
        working_dir=None
    ):
    """
    Prepares a text data object from the files present at data folder

    =====================   =================================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------------
    path                    Required directory path. The directory path where
                            the training and validation files are present.
    ---------------------   -------------------------------------------------
    task                    Required string. The task for which the dataset is
                            prepared. Available choice at this point is "classification"
                            and "sequence_translation".
    ---------------------   -------------------------------------------------
    text_columns            Required string. The column that will be used as
                            feature.
    ---------------------   -------------------------------------------------
    label_columns           Required list. The list of columns denoting the
                            class label/translated text to predict. Provide a list of columns
                            in case of multi-label classification problem
    ---------------------   -------------------------------------------------
    train_file              Optional string. The file name containing the
                            training data. Supported file formats/extensions are
                            .csv and .tsv
                            Default value is `train.csv`
    ---------------------   -------------------------------------------------
    valid_file              Optional string. The file name containing the
                            validation data. Supported file formats/extensions
                            are .csv and .tsv.
                            Default value is `None`. If None then some portion
                            of the training data will be kept for validation
                            (based on the value of `val_split_pct` parameter)
    ---------------------   -------------------------------------------------
    val_split_pct           Optional float. Percentage of training data to keep
                            as validation.
                            By default 10% data is kept for validation.
    ---------------------   -------------------------------------------------
    seed                    Optional integer. Random seed for reproducible
                            train-validation split.
                            Default value is 42.
    ---------------------   -------------------------------------------------
    batch_size              Optional integer. Batch size for mini batch gradient
                            descent (Reduce it if getting CUDA Out of Memory
                            Errors).
                            Default value is 16.
    ---------------------   -------------------------------------------------
    process_labels          Optional boolean. If true, default processing functions
                            will be called on label columns as well.
                            Default value is False.
    ---------------------   -------------------------------------------------
    remove_html_tags        Optional boolean. If true, remove html tags from text.
                            Default value is False.
    ---------------------   -------------------------------------------------
    remove_urls             Optional boolean. If true, remove urls from text.
                            Default value is False.
    ---------------------   -------------------------------------------
    working_dir             Optional string. Sets the default path to be used as
                            a prefix for saving trained models and checkpoints.
    =====================   =================================================

    :returns: `TextData` object

    """
    # allowed_tasks = ["classification", "summarization", "translation",
    #                  "question-answering", "ner", "text-generation"]

    if not HAS_FASTAI:
        _raise_fastai_import_error(import_exception)

    # if task not in allowed_tasks:
    #     raise Exception(f"Wrong task choosen. Allowed tasks are {allowed_tasks}")

    if isinstance(label_columns, (str, bytes)):
        label_columns = [label_columns]

    force_cpu = arcgis.learn.models._arcgis_model._device_check()

    if hasattr(arcgis, "env") and force_cpu == 1:
        arcgis.env._processorType = "CPU"

    if task == "classification":
        data = TextDataObject.prepare_data_for_classification(
            path,
            text_columns,
            label_columns,
            train_file=train_file,
            valid_file=valid_file,
            val_split_pct=val_split_pct,
            seed=seed,
            batch_size=batch_size,
            process_labels=process_labels,
            remove_html_tags=remove_html_tags,
            remove_urls=remove_urls
        )
    
    elif task.lower() == "sequence_translation":
        data = TextDataObject.prepare_data_for_seq2seq(
            path,
            text_columns,
            label_columns,
            train_file=train_file,
            val_split_pct=val_split_pct,
            seed=seed,
            batch_size=batch_size,
            process_labels=process_labels,
            remove_html_tags=remove_html_tags,
            remove_urls=remove_urls
        )

    else:
        logger = logging.getLogger()
        logger.error(f"Wrong task - {task} provided. This function can handle only `classification` and 'sequence_translation' task currently")
        raise Exception(f"Wrong task - {task} provided. This function can handle only `classification` and 'sequence_translation' task currently")

    if working_dir is None:
        working_dir = ''
    temp_folder = _prepare_working_dir(working_dir)
    data.path = Path(os.path.abspath(working_dir))
    data._temp_folder = temp_folder
    return data

def prepare_tabulardata(
        input_features=None,
        variable_predict=None,
        explanatory_variables=None,
        explanatory_rasters=None,
        date_field=None,
        distance_features=None,
        preprocessors=None,
        val_split_pct=0.1,
        seed=42,
        batch_size=64,
        index_field=None,
        working_dir=None
    ):

    """
    Prepares a tabular data object from input_features and optionally rasters.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    input_features          Optional Feature Layer Object or spatially enabled dataframe.
                            This contains features denoting the value of the dependent variable.
                            Leave empty for using rasters with MLModel.
    ---------------------   -------------------------------------------
    variable_predict        Optional String, denoting the field_name of
                            the variable to predict.
                            Keep none for unsupervised training using MLModel.
    ---------------------   -------------------------------------------
    explanatory_variables   Optional list containing field names from input_features
                            By default the field type is continuous.
                            To override field type to categorical, pass
                            a 2-sized tuple in the list containing:
                                1. field to be taken as input from the input_features.
                                2. True/False denoting Categorical/Continuous variable.
                            For example:
                                ["Field_1", ("Field_2", True)]
                            Here Field_1 is treated as continuous and
                            Field_2 as categorical.
    ---------------------   -------------------------------------------
    explanatory_rasters     Optional list containing Raster objects.
                            By default the rasters are continuous.
                            To mark a raster categorical, pass a 2-sized tuple containing:
                                1. Raster object.
                                2. True/False denoting Categorical/Continuous variable.
                            For example:
                                [raster_1, (raster_2, True)]
                            Here raster_1 is treated as continuous and
                            raster_2 as categorical.
    ---------------------   -------------------------------------------
    date_field              Optional field_name.
                            This field contains the date in the input_features.
                            The field type can be a string or date time field.
                            If specified, the field will be split into
                            Year, month, week, day, dayofweek, dayofyear,
                            is_month_end, is_month_start, is_quarter_end,
                            is_quarter_start, is_year_end, is_year_start,
                            hour, minute, second, elapsed and these will be added
                            to the prepared data as columns.
                            All fields other than elapsed and dayofyear are treated
                            as categorical.
    ---------------------   -------------------------------------------
    distance_features       Optional list of Feature Layer objects.
                            Distance is calculated from features in these layers
                            to features in input_features.
                            Nearest distance to each feature is added in the prepared
                            data.
                            Field names in the prepared data added are
                            "NEAR_DIST_1", "NEAR_DIST_2" etc.
    ---------------------   -------------------------------------------
    preprocessors           For Fastai: Optional transforms list.
                            For Scikit-learn:
                            1. Supply a column transformer object.
                            2. Supply a list of tuple,
                            For example:
                            [('Col_1', 'Col_2', Transform1()), ('Col_3', Transform2())]
                            Categorical data is by default encoded.
                            If nothing is specified, default transforms are applied
                            to fill missing values and normalize categorical data.
                            For Raster use raster.name for the the first band,
                            raster.name_1 for 2nd band, raster.name_2 for 3rd
                            and so on.
    ---------------------   -------------------------------------------
    val_split_pct           Optional float. Percentage of training data to keep
                            as validation.
                            By default 10% data is kept for validation.
    ---------------------   -------------------------------------------
    seed                    Optional integer. Random seed for reproducible
                            train-validation split.
                            Default value is 42.
    ---------------------   -------------------------------------------
    batch_size              Optional integer. Batch size for mini batch gradient
                            descent (Reduce it if getting CUDA Out of Memory
                            Errors).
                            Default value is 64.
    ---------------------   -------------------------------------------
    index_field             Optional string. Field Name in the input features
                            which will be used as index field for the data.
                            Used for Time Series, to visualize values on the
                            x-axis.
    ---------------------   -------------------------------------------
    working_dir             Optional string. Sets the default path to be used as
                            a prefix for saving trained models and checkpoints.
    =====================   ===========================================

    :returns: `TabularData` object

    """
    if input_features is None and (explanatory_rasters is None or len(explanatory_rasters) == 0):
        raise Exception("No Features or Rasters found")

    import warnings
    if not HAS_FASTAI:
        _raise_fastai_import_error(import_exception)

    force_cpu = arcgis.learn.models._arcgis_model._device_check()

    if hasattr(arcgis, "env") and force_cpu == 1:
        arcgis.env._processorType = "CPU"

    HAS_COLUMN_TRANSFORMS = False

    column_transforms_mapping = {}
    if preprocessors and isinstance(preprocessors, list):
        for transform in preprocessors:
            if isinstance(transform, tuple):
                HAS_COLUMN_TRANSFORMS = True
                break

        if HAS_COLUMN_TRANSFORMS:
            column_transforms = []
            for transform in preprocessors:
                if not isinstance(transform, tuple):
                    warnings.warn("Please pass (Field_Name, transform) in the list of preprocessors")
                    return
                column_transforms.append(
                    (
                        transform[-1],
                        list(transform[0:-1])
                    )
                )

            from sklearn.compose import make_column_transformer
            preprocessors = make_column_transformer(*column_transforms)

    if preprocessors:
        for transform in preprocessors.transformers:
            for column in transform[2]:
                if not column_transforms_mapping.get(column):
                    column_transforms_mapping[column] = []
                if 'pipeline' in transform[0]:
                    for step in transform[1].steps:
                        column_transforms_mapping[column].append(step[1])
                else:
                    column_transforms_mapping[column].append(transform[1])

    data = TabularDataObject.prepare_data_for_layer_learner(
        input_features,
        variable_predict,
        feature_variables=explanatory_variables,
        raster_variables=explanatory_rasters,
        date_field=date_field,
        distance_feature_layers=distance_features,
        procs=preprocessors,
        val_split_pct=val_split_pct,
        seed=seed,
        batch_size=batch_size,
        index_field=index_field,
        column_transforms_mapping=column_transforms_mapping
    )

    if working_dir is None:
        working_dir = ''
    temp_folder = _prepare_working_dir(working_dir)
    data.path = Path(os.path.abspath(working_dir))
    data._temp_folder = temp_folder

    return data


def prepare_data(path,
                 class_mapping=None, 
                 chip_size=224, 
                 val_split_pct=0.1, 
                 batch_size=64, 
                 transforms=None, 
                 collate_fn=_bb_pad_collate, 
                 seed=42, 
                 dataset_type=None, 
                 resize_to=None,
                 working_dir=None,
                 **kwargs):
    """
    Prepares a data object from training sample exported by the 
    Export Training Data tool in ArcGIS Pro or Image Server, or training 
    samples in the supported dataset formats. This data object consists of 
    training and validation data sets with the specified transformations, 
    chip size, batch size, split percentage, etc. 
    -For object detection, use Pascal_VOC_rectangles or KITTI_rectangles format.
    -For feature categorization use Labelled Tiles or ImageNet format.
    -For pixel classification, use Classified Tiles format.
    -For entity extraction from text, use IOB, BILUO or ner_json formats. 

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    path                    Required string. Path to data directory.
                            Provide list of paths for Multi-folder training. 
                            Note: list of paths is currently supported by for dataset types: 
                            Classified_Tiles, Labeled_Tiles, MultiLabel_Tiles
                            PASCAL_VOC_rectangles, RCNN_Masks
    ---------------------   -------------------------------------------
    class_mapping           Optional dictionary. Mapping from id to
                            its string label.
                            For dataset_type=IOB, BILUO or ner_json:
                                Provide address field as class mapping
                                in below format:
                                class_mapping={'address_tag':'address_field'}.
                                Field defined as 'address_tag' will be treated
                                as a location. In cases where trained model extracts
                                multiple locations from a single document, that 
                                document will be replicated for each location.

    ---------------------   -------------------------------------------
    chip_size               Optional integer, default 224. Size of the image to train the
                            model. Images are cropped to the specified chip_size.
                            If image size is less than chip_size, the image size is
                             used as chip_size. Not supported for superres and siammask.
    ---------------------   -------------------------------------------
    val_split_pct           Optional float. Percentage of training data to keep
                            as validation.
    ---------------------   -------------------------------------------
    batch_size              Optional integer. Batch size for mini batch gradient
                            descent (Reduce it if getting CUDA Out of Memory
                            Errors). Batch size is required to be greater than 1.
                            For dataset_type = 'ObjectTracking', optimal batch_size
                            value is 64.
    ---------------------   -------------------------------------------
    transforms              Optional tuple. Fast.ai transforms for data
                            augmentation of training and validation datasets
                            respectively (We have set good defaults which work
                            for satellite imagery well). If transforms is set
                            to `False` no transformation will take place and 
                            `chip_size` parameter will also not take effect.
                            If the dataset_type is 'PointCloud', use 
                            `Transform3d` class from `arcgis.learn`.
    ---------------------   -------------------------------------------
    collate_fn              Optional function. Passed to PyTorch to collate data
                            into batches(usually default works).
    ---------------------   -------------------------------------------
    seed                    Optional integer. Random seed for reproducible
                            train-validation split.
    ---------------------   -------------------------------------------
    dataset_type            Optional string. `prepare_data` function will infer 
                            the `dataset_type` on its own if it contains a 
                            map.txt file. If the path does not contain the 
                            map.txt file pass either of 'PASCAL_VOC_rectangles', 
                            'KITTI_rectangles', 'RCNN_Masks', 'Classified_Tiles', 
                            'Labeled_Tiles', 'MultiLabeled_Tiles', 'Imagenet',  
                            'PointCloud', 'ImageCaptioning', 'ChangeDetection',
                            'superres', 'CycleGAN' and 'Pix2Pix'.
                            This parameter is mandatory for data which are not
                            exported by ArcGIS Pro / Enterprise which includes
                            'PointCloud', 'ImageCaptioning', 'ChangeDetection',
                            'CycleGAN', 'Pix2Pix' and 'ObjectTracking'.
                            This parameter is also mandatory while preparing data
                            for 'EntityRecognizer' model. Accepted data format
                            for this model are - ['ner_json','BIO', 'LBIOU'].
    ---------------------   -------------------------------------------
    resize_to               Optional integer. Resize the images to a given size.
                            Works only for "PASCAL_VOC_rectangles" and "superres".
                            First resizes the image to the given size and
                            then crops images of size equal to chip_size.
                            Note: Keep chip_size < resize_to
    ---------------------   -------------------------------------------
    working_dir             Optional string. Sets the default path to be used as
                            a prefix for saving trained models and checkpoints.
    =====================   ===========================================

    **Keyword Arguments**

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    imagery_type            Optional string. Type of imagery used to export 
                            the training data, valid values are:
                                - 'naip'
                                - 'sentinel2'
                                - 'landsat8'
                                - 'ms' - any other type of imagery
    ---------------------   -------------------------------------------
    bands                   Optional list. Bands of the imagery used to export 
                            training data. 
                            For example ['r', 'g', 'b', 'nir', 'u'] 
                            where 'nir' is near infrared band and 'u' is a miscellaneous band.
    ---------------------   -------------------------------------------
    rgb_bands               Optional list. Indices of red, green and blue bands 
                            in the imagery used to export the training data.
                            for example: [2, 1, 0] 
    ---------------------   -------------------------------------------
    extract_bands           Optional list. Indices of bands to be used for
                            training the model, same as in the imagery used to 
                            export the training data.
                            for example: [3, 1, 0] where we will not be using 
                            the band at index 2 to train our model. 
    ---------------------   -------------------------------------------
    norm_pct                Optional float. Percentage of training data to be 
                            used for calculating imagery statistics for  
                            normalizing the data. 
                            Default is 0.3 (30%) of data.
    ---------------------   -------------------------------------------
    downsample_factor       Optional integer. Factor to downsample the images 
                            for image SuperResolution. 
                            for example: if value is 2 and image size 256x256,
                            it will create label images of size 128x128.
                            Default is 4
    ---------------------   -------------------------------------------
    encoding                Optional string.
                            Applicable only when dataset_type=IOB, BILUO or ner_json:
                            The encoding to read the csv/json file.
                            Default is 'UTF-8'
    ---------------------   -------------------------------------------
    min_points              Optional int. Filtering based on minimum number
                            of points in a block.
                            Set `min_points=1000` to filter out blocks with less
                            than 1000 points. Applicable only for
                            dataset_type='PointCloud'
    ---------------------   -------------------------------------------
    classes_of_interest     Optional string. List of classes of interest. 
                            This will filter blocks based on `classes_of_interest`.
                            If we have classes [1, 3, 5, 7] in our dataset,
                            but we are mainly interested in 1 and 3,
                            Set `classes_of_interest=[1,3]`. Only those blocks
                            will be considered for training which either have
                            class 1 or 3 in them, rest of the blocks will
                            be filtered out.
                            If remapping of rest of the classes is required
                            set `background_classcode` to some value.
                            Applicable only for dataset_type='PointCloud'
    ---------------------   -------------------------------------------
    extra_features          Optional List. Contains a list of strings
                            which tells which extra features to use to
                            train PointCNN. By default only x,y and z
                            are considered for training irrespective
                            of what features were exported.
                            Set this to be a subset of 
                            ['intensity', 'numberOfReturns', 'returnNumber', 
                            'red', 'green', 'blue', 'nearInfrared'].
                            For data exported from `export_point_dataset` set
                            this to ['intensity', 'num_returns', 'return_num', 
                            'red', 'green', 'blue', 'nir'].
    ---------------------   -------------------------------------------
    remap_classes           Optional dictionary {int:int}. Mapping from  
                            class values to user defined values. 
                            If we have [1, 3, 5, 7] in our dataset
                            and we want to map class 5 to 3. Set this 
                            parameter to `remap_classes={5:3}`.
                            In training then 5 will also be considered as 3.
                            Applicable only for dataset_type='PointCloud' 
    ---------------------   -------------------------------------------
    background_classcode    Optional int. Default None.
                            If this is defined it will remap other 
                            class except `classes_of_interest` to 
                            `background_classcode` value. Only applicable
                            when specifying `classes_of_interest`.
                            Applicable only for dataset_type='PointCloud'.                             
    =====================   ===========================================

    :returns: data object

    """
    emd = {}
    height_width = []
    not_label_count = [0]

    if not HAS_FASTAI:
        _raise_fastai_import_error()

    (fastai.vision.data.image_extensions).add('.mrf')

    if isinstance(path, str) and not os.path.exists(path):
        message = f"Invalid input path. \nCould not find the path specified \n'{path}' \nPlease ensure that the input path is correct."
        if '\\' in path:
            message+=f"""\n\nif you are using windows style paths please ensure you have specified paths with raw modifier. for example {"path=r'{path}'"}"""
        raise Exception(message)

    if type(path) is str:
        path = Path(path)

    databunch_kwargs = {'num_workers':0} if sys.platform == 'win32' else {}
    databunch_kwargs['bs'] = batch_size

    force_cpu = arcgis.learn.models._arcgis_model._device_check()

    if hasattr(arcgis, "env") and force_cpu == 1:
        arcgis.env._processorType = "CPU"

    if getattr(arcgis.env, "_processorType", "") == "CPU":
        databunch_kwargs["device"] = torch.device('cpu')

    if ARCGIS_ENABLE_TF_BACKEND:
        databunch_kwargs["device"] = torch.device('cpu')
        databunch_kwargs["pin_memory"] = False

    kwargs_transforms = {}
    if resize_to:
        kwargs_transforms['size'] = resize_to
        # Applying SQUISH ResizeMethod to avoid reflection padding
        kwargs_transforms['resize_method'] = ResizeMethod.SQUISH

    # Multi Folder training support
    data_folders = None
    emd_in = kwargs.get('emd', None)
    eas_in = kwargs.get('eas', None)
    eas = None
    images_df = kwargs.get('images_df', None)
    if isinstance(path, (list, tuple)):
        data_folders = [Path(x) for x in path]
        emd, eas, path = merge_emd_and_stats(data_folders)
        if working_dir is None:
            working_dir = os.getcwd()
    if emd_in is not None:
        emd = copy.deepcopy(emd_in)
    if eas_in is not None:
        eas = copy.deepcopy(eas_in)

    has_esri_files = _check_esri_files(path)
            
    alter_class_mapping = False
    color_mapping = None

    # Multispectral Kwargs init
    _bands = None
    _imagery_type = None
    _is_multispectral = False
    _show_batch_multispectral = None
    stats_file = path / 'esri_accumulated_stats.json'

    if dataset_type is None:
        if has_esri_files:
            if data_folders is None:
                with open(stats_file) as f:
                    stats = json.load(f)
            else:
                stats = eas
            dataset_type = stats['MetaDataMode']
        # elif os.path.exists(path/'images_before') and os.path.exists(path/'images_after'):
        #     dataset_type = 'ChangeDetection'
        elif _check_esri_files(path/'A') and _check_esri_files(path/'B'):
            has_esri_files = True
            dataset_type = 'CycleGAN'
        elif not has_esri_files:
            raise Exception("Could not infer dataset type. Please specify a supported dataset type or ensure that the path contains valid esri files")
    
    # Pix2Pix data is exported as Export_Tiles with 'images' and 'images2' folders
    if dataset_type == 'Export_Tiles' and os.path.exists(path/'images2'):
        dataset_type = 'Pix2Pix'

    # Change Detection data is exported as Classified_Tiles with 'images', 'images2' and 'labels' folders
    if dataset_type == 'Classified_Tiles' and os.path.exists(path/'images2'):
        dataset_type = 'ChangeDetection'

    if dataset_type == 'ChangeDetection':
        from ._utils.change_detection_data import folder_check, is_old_format_change_detection
        folder_check(path)
        if is_old_format_change_detection(path):
            json_file = path / 'images_before' / 'esri_model_definition.emd'
        else:
            json_file = path / 'esri_model_definition.emd'
        if json_file.exists():
            with open(json_file) as f:
                emd = json.load(f)
        else:
            from ._utils.change_detection_data import get_files, image_extensions, folder_names
            images_before_folder, images_after_folder = folder_names(path)
            files_list = get_files(path  / images_before_folder,
                                  extensions=image_extensions,
                                  recurse=True)
            msimage_list = ArcGISImageList(files_list)
            if msimage_list[0].shape[0] != 3:
                kwargs['imagery_type'] = 'ms'

    elif dataset_type == "CycleGAN" or dataset_type == "Pix2Pix":
        from ._utils.cyclegan import get_files, image_extensions, cyclegan_paths, folder_check_cyclegan
        from ._utils.pix2pix import pix2pix_paths, folder_check_pix2pix, rgb_or_ms
        if dataset_type == "CycleGAN":
            folder_check_cyclegan(path)
            if _check_esri_files(path/'A') and _check_esri_files(path/'B'):
                has_esri_files = True
            path_a, path_b = cyclegan_paths(path)
            stats_path = path_a.parent
        else:
            folder_check_pix2pix(path)
            path_a, path_b = pix2pix_paths(path)
            stats_path = path

        if has_esri_files:
            stats_file = stats_path / 'esri_accumulated_stats.json'
            if data_folders is None:
                with open(stats_file) as f:
                    stats = json.load(f)
            else:
                stats = eas

        json_file = path_a.parent / 'esri_model_definition.emd'
        if json_file.exists():
            with open(json_file) as f:
                emd = json.load(f)

        files_list_a = get_files(path_a, extensions=image_extensions, recurse=True)
        files_list_b = get_files(path_b, extensions=image_extensions, recurse=True)
        msimage_list_a = ArcGISImageList(files_list_a)
        msimage_list_b = ArcGISImageList(files_list_b)
        img_type = 'RGB'
        if msimage_list_a[0].shape[0] != 3 or msimage_list_b[0].shape[0] != 3:
            img_type = kwargs['imagery_type'] = 'ms'
        imagery_type_a = ifnone(rgb_or_ms(str(files_list_a[0])), img_type)
        imagery_type_b = ifnone(rgb_or_ms(str(files_list_b[0])), img_type)

    if dataset_type not in ["Imagenet", "superres", "Export_Tiles", 'CycleGAN', 'Pix2Pix', 'ChangeDetection'] and has_esri_files:
        with open(stats_file) as f:
            stats = json.load(f)
            dataset_type = stats['MetaDataMode']

        with open(path / 'map.txt') as f:
            while True:
                line = f.readline()
                if len(line.split())==2:
                    break
        try:
            img_size = ArcGISMSImage.open_gdal((path/(line.split()[0]).replace('\\', os.sep))).shape[-1]
        except:            
            img_size = PIL.Image.open((path/(line.split()[0]).replace('\\', os.sep))).size[-1]
        if chip_size > img_size:
            chip_size = img_size
        right = line.split()[1].split('.')[-1].lower()

        json_file = path / 'esri_model_definition.emd'
        if data_folders is None:
            with open(json_file) as f:
                emd = json.load(f)
        else:
            pass  # emd already in memory

        # Create Class Mapping from EMD if not specified by user
        ## Validate user defined class_mapping keys with emd (issue #3064)
        # Get classmapping from emd file.
        try:
            emd_class_mapping = {i['Value']: i['Name'] for i in emd['Classes']}
        except KeyError:
            emd_class_mapping = {i['ClassValue']: i['ClassName'] for i in emd['Classes']}

        ## Change all keys to int.
        if class_mapping is not None:
            class_mapping = {int(key):value for key, value in class_mapping.items()}
        else:
            class_mapping = {}

        ## Map values from user defined classmapping to emd classmapping.
        for key, _ in emd_class_mapping.items():
            if class_mapping.get(key) is not None:
                emd_class_mapping[key] = class_mapping[key]
            
        class_mapping = emd_class_mapping

        color_mapping = {(i.get('Value', 0) or i.get('ClassValue', 0)): i['Color'] for i in emd.get('Classes', [])}

        if color_mapping.get(None):
            del color_mapping[None]

        if class_mapping.get(None):
            del class_mapping[None]

        # Multispectral support from EMD 
        # Not Implemented Yet
        if emd.get('bands', None) is not None: 
            _bands = emd.get['bands'] # Not Implemented
            
        if emd.get('imagery_type', None) is not None: 
            _imagery_type = emd.get['imagery_type'] # Not Implemented        

    elif dataset_type in ['PASCAL_VOC_rectangles', 'KITTI_rectangles'] and not has_esri_files:
        if class_mapping is None:
            class_mapping = _get_class_mapping(path / 'labels', dataset_type=dataset_type)
            alter_class_mapping = True

    _map_space = "MAP_SPACE"
    _pixel_space = "PIXEL_SPACE"
    if has_esri_files:
        _image_space_used = emd.get('ImageSpaceUsed', _map_space)
    else:
        _image_space_used = _pixel_space

    # Multispectral check
    # With Python API for ArcGIS 1.9 multispectral workflow will automatically kick in with the following conditions
    # 1. If the imagery source is not having exactly three bands
    # 2. If there is any band other than RGB
    # 3. If None among all three bands in the imagery is unknown
    #
    imagery_type = 'ASSUMED_RGB'
    if kwargs.get('imagery_type', None) is not None:
        imagery_type = kwargs.get('imagery_type')
    elif _imagery_type is not None:
        imagery_type = _imagery_type
    if has_esri_files and "InputRastersProps" in emd and kwargs.get('imagery_type', None) is None:
        sensor_name = emd["InputRastersProps"].get("SensorName", None)
        tile_stats = emd.get("AllTilesStats", None)
        band_names = emd["InputRastersProps"].get("BandNames", None)
        if tile_stats is not None and len(tile_stats)!=3:
            if not (len(tile_stats) == 4 and band_names is not None and len(band_names)>3 and band_names[3].lower() == 'alpha'):
                imagery_type = sensor_name
        else:
            # Check by band names
            if band_names is not None:
                band_mapping = {i:b.lower() for i, b in enumerate(band_names)}
                for b in emd["WellKnownBandNames (FYI, these band names can be used in ExtractBands)"]:
                    if b.lower() in [
                        "red",
                        "green",
                        "blue"
                    ]:
                        continue
                    if b.lower() in band_mapping:
                        imagery_type = sensor_name
                        break
                        
            # Check by values
            if tile_stats is not None:
                for stat in tile_stats:
                    if stat["Min"] < 0 or stat["Max"] > 255:
                        imagery_type = sensor_name
                        break

    if (not imagery_type in ('ASSUMED_RGB', 'RGB')) and not HAS_GDAL:
        raise_gdal_import_error()

    bands = None
    if kwargs.get('bands', None) is not None:
        bands = kwargs.get('bands')
        for i, b in enumerate(bands):
            if type(b) == str:
                bands[i] = b.lower()
    elif imagery_type_lib.get(imagery_type, None) is not None:
        bands = imagery_type_lib.get(imagery_type)['bands']
    elif _bands is not None:
        bands = _bands

    rgb_bands = None
    if kwargs.get('rgb_bands', None) is not None:
        rgb_bands = kwargs.get('rgb_bands')
    elif bands is not None:
        rgb_bands = [ bands.index(b) for b in ['r', 'g', 'b'] if b in bands ]
    
    if (bands is not None) or (rgb_bands is not None) or (not imagery_type in ['RGB', 'ASSUMED_RGB']):
        if imagery_type in ['RGB', 'ASSUMED_RGB']:
            imagery_type = 'MULTISPECTRAL'
        _is_multispectral = True
    
    if kwargs.get('norm_pct', None) is not None:
        norm_pct= kwargs.get('norm_pct')
        norm_pct = min(max(0, norm_pct), 1)
    else:
        norm_pct = .3

    lighting_transforms = kwargs.get('lighting_transforms', True)

    if dataset_type == 'RCNN_Masks':

        def get_labels(x, label_dirs, ext=right):
            label_path = []
            for lbl in label_dirs:
                if os.path.exists(Path(lbl) / (x.stem + '.{}'.format(ext))):
                    label_path.append(Path(lbl) / (x.stem + '.{}'.format(ext)))
            return label_path

        label_dirs = []
        index_dir = {} #for handling calss value with any number
        for i, k in enumerate(sorted(class_mapping.keys())):
            label_dirs.append(class_mapping[k])
            index_dir[k] = i+1
        label_dir = [os.path.join(path/'labels', lbl) for lbl in label_dirs if os.path.isdir(os.path.join(path/'labels', lbl))]
        get_y_func = partial(get_labels, label_dirs= label_dir)

        def image_without_label(imagefile, not_label_count=[0]):
            label_mask = get_y_func(imagefile)
            if label_mask == []:
                not_label_count[0] += 1
                return False
            return True
        
        remove_image_without_label = partial(image_without_label, not_label_count=not_label_count)

        if class_mapping.get(0):
            del class_mapping[0]

        if color_mapping.get(0):
            del color_mapping[0]

        if data_folders is None and images_df is None:
            data = (ArcGISInstanceSegmentationItemList.from_folder(path / 'images')
                    .filter_by_func(remove_image_without_label)
                    .split_by_rand_pct(val_split_pct, seed=seed)
                    .label_from_func(get_y_func, chip_size=chip_size, classes=['NoData'] + list(class_mapping.values()),
                                     class_mapping=class_mapping, color_mapping=color_mapping, index_dir=index_dir))
        else:
            if images_df is not None:
                # images_df should have two columns 0, 1
                ## column 0 --> images
                ## column 1 --> labels
                ## Note: Please supply absolute Paths
                ##
                src = ArcGISInstanceSegmentationItemList.from_df(images_df, 'images')
                src.items = images_df[images_df.columns[0]].values
                src = src.split_by_rand_pct(val_split_pct, seed=seed)
                if len(images_df.columns) > 1:
                    src = src.label_from_df(
                        chip_size=chip_size,
                        classes=(['NoData'] + list(class_mapping.values())),
                        class_mapping=class_mapping,
                        color_mapping=color_mapping,
                        index_dir=index_dir
                    )
                else:
                    src = src.label_from_func(
                        get_y_func,
                        chip_size=chip_size,
                        classes=(['NoData'] + list(class_mapping.values())),
                        class_mapping=class_mapping,
                        color_mapping=color_mapping,
                        index_dir=index_dir
                    )
            else:
                # MultiFolder Training
                imageslist = []
                for data_folder in data_folders:
                    imageslist.append(ArcGISInstanceSegmentationItemList.from_folder(data_folder / 'images').items)
                src = ArcGISInstanceSegmentationItemList(np.concatenate(imageslist)) \
                    .filter_by_func(remove_image_without_label) \
                    .split_by_rand_pct(val_split_pct, seed=seed) \
                    .label_from_func(
                    get_y_func,
                    chip_size=chip_size,
                    classes=(['NoData'] + list(class_mapping.values())),
                    class_mapping=class_mapping,
                    color_mapping=color_mapping,
                    index_dir=index_dir
                )
                src.path = os.path.abspath('images')
            data = src
        #
        _show_batch_multispectral = show_batch_rcnn_masks
        
        if transforms is None:
            ranges = (0, 1)
            if _image_space_used == _map_space:
                train_tfms = [
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    dihedral_affine(),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(1.0, 1.5)),
                    rand_zoom(scale=(1.0, 1.2))
                ]
            else:
                train_tfms = [
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(1.0, 1.5)),
                    rand_zoom(scale=(1.0, 1.2))
                ]
            val_tfms = [crop(size=chip_size, p=1., row_pct=0.5, col_pct=0.5)]
            transforms = (train_tfms, val_tfms)
            kwargs_transforms['size'] = chip_size

        kwargs_transforms['tfm_y'] = True

    elif dataset_type == 'Classified_Tiles':

        def get_y_func(x, ext=right):
            return x.parents[1] / 'labels' / (x.stem + '.{}'.format(ext))

        def image_without_label(imagefile, not_label_count=[0], ext=right):
            xmlfile = imagefile.parents[1] / 'labels' / (imagefile.stem + '.{}'.format(ext))
            if not os.path.exists(xmlfile):
                not_label_count[0] += 1
                return False
            return True
        
        remove_image_without_label = partial(image_without_label, not_label_count=not_label_count)

        if class_mapping.get(0):
            del class_mapping[0]

        if color_mapping.get(0):
            del color_mapping[0]

        if is_no_color(color_mapping):
            color_mapping = {j:[random.choice(range(256)) for i in range(3)] for j in class_mapping.keys()}
            
        # TODO : Handle NoData case

        if data_folders is None and images_df is None:
            data = ArcGISSegmentationItemList.from_folder(path / 'images') \
                .filter_by_func(remove_image_without_label) \
                .split_by_rand_pct(val_split_pct, seed=seed) \
                .label_from_func(
                get_y_func,
                classes=(['NoData'] + list(class_mapping.values())),
                class_mapping=class_mapping,
                color_mapping=color_mapping
            )
        else:
            if images_df is not None:
                # imagesdf should have two columns 0, 1
                ## column 0 --> images
                ## column 1 --> labels
                ## Note: Please supply absolute Paths
                ##
                src = ArcGISSegmentationItemList.from_df(images_df, 'images')
                src.items = images_df[images_df.columns[0]].values
                src = src.split_by_rand_pct(val_split_pct, seed=seed)
                if len(images_df.columns) > 1:
                    src = src.label_from_df(
                        class_mapping=class_mapping,
                        color_mapping=color_mapping,
                        classes=(['NoData'] + list(class_mapping.values()))
                    )
                else:
                    src = src.label_from_func(
                        get_y_func,
                        classes=(['NoData'] + list(class_mapping.values())),
                        class_mapping=class_mapping,
                        color_mapping=color_mapping
                    )
            else:
                # MultiFolder Training
                imageslist = []
                for data_folder in data_folders:
                    imageslist.append(ArcGISSegmentationItemList.from_folder(data_folder / 'images').items)
                src = ArcGISSegmentationItemList(np.concatenate(imageslist)) \
                    .filter_by_func(remove_image_without_label) \
                    .split_by_rand_pct(val_split_pct, seed=seed) \
                    .label_from_func(
                    get_y_func,
                    classes=(['NoData'] + list(class_mapping.values())),
                    class_mapping=class_mapping,
                    color_mapping=color_mapping
                )
            data = src
        #
        _show_batch_multispectral = show_batch_classified_tiles

        def classified_tiles_collate_fn(samples): # The default fastai collate_fn was causing memory leak on tensors
            r = ( torch.stack([x[0].data for x in samples]), torch.stack([x[1].data for x in samples]) )
            return r
        databunch_kwargs['collate_fn'] = classified_tiles_collate_fn

        if transforms is None:
            if _image_space_used == _map_space:
                transforms = get_transforms(
                    flip_vert=True,
                    max_rotate=90.,
                    max_zoom=3.0,
                    max_lighting=0.5
                )
            else:
                transforms = get_transforms(
                    max_zoom=3.0,
                    max_lighting=0.5
                )

        kwargs_transforms['tfm_y'] = True
        kwargs_transforms['size'] = chip_size
    elif dataset_type in ['PASCAL_VOC_rectangles', 'KITTI_rectangles']:

        def image_without_label(imagefile, dataset_type, not_label_count=[0]):
            imagefile = Path(imagefile)
            if dataset_type == 'KITTI_rectangles':
                label_suffix='.txt'
            else:
                label_suffix='.xml'
            label_file = imagefile.parents[1] / 'labels' / imagefile.name.replace('{ims}'.format(ims=imagefile.suffix), label_suffix)
            if not os.path.exists(label_file):
                not_label_count[0] += 1
                return False
            return True
        remove_image_without_label = partial(image_without_label, not_label_count=not_label_count, dataset_type=dataset_type)
        get_y_func = partial(
            _get_bbox_lbls,
            class_mapping=class_mapping,
            height_width=height_width,
            dataset_type=dataset_type
        )

        if data_folders is None and images_df is None:
            data = ObjectDetectionItemList.from_folder(path / 'images') \
                .filter_by_func(remove_image_without_label) \
                .split_by_rand_pct(val_split_pct, seed=seed) \
                .label_from_func(get_y_func)
        else:
            if images_df is not None:
                src = ObjectDetectionItemList.from_df(images_df, 'images')
                src.items = images_df[images_df.columns[0]].values
                src = src.split_by_rand_pct(val_split_pct, seed=seed) \
                        .label_from_func(get_y_func)
            else:
                # MultiFolder Training
                imageslist = []
                for data_folder in data_folders:
                    imageslist.append(ObjectDetectionItemList.from_folder(data_folder / 'images').items)
                src = ObjectDetectionItemList(np.concatenate(imageslist)) \
                    .filter_by_func(remove_image_without_label) \
                    .split_by_rand_pct(val_split_pct, seed=seed) \
                    .label_from_func(get_y_func)
            data = src
        #
        _show_batch_multispectral = show_batch_pascal_voc_rectangles


        if transforms is None:
            ranges = (0, 1)
            if _image_space_used == _map_space:
                train_tfms = [
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    dihedral_affine(),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(0.75, 1.5)),
                    rand_zoom(scale=(1.0, 1.5))
                ]
            else:
                train_tfms = [
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(0.75, 1.5)),
                    rand_zoom(scale=(1.0, 1.5))
                ]
            val_tfms = [crop(size=chip_size, p=1., row_pct=0.5, col_pct=0.5)]
            transforms = (train_tfms, val_tfms)

        kwargs_transforms['tfm_y'] = True
        databunch_kwargs['collate_fn'] = collate_fn
    elif dataset_type in ['Labeled_Tiles', 'MultiLabeled_Tiles', 'Imagenet']:
        if dataset_type == 'Labeled_Tiles':
            get_y_func = partial(_get_lbls, class_mapping=class_mapping)
        elif dataset_type == 'MultiLabeled_Tiles':
            get_y_func = _get_multi_lbls
        else:
            # Imagenet
            def get_y_func(x):
                return x.parent.stem
            if collate_fn is not _bb_pad_collate:
                databunch_kwargs['collate_fn'] = collate_fn
            else:
                databunch_kwargs['collate_fn'] = _ImagenetCollater(chip_size)
            _images_folder = os.path.join(os.path.abspath(path), 'images')
            if not os.path.exists(_images_folder):
                raise Exception(f"""Could not find a folder "images" in "{os.path.abspath(path)}",
                \na folder "images" should be present in the supplied path to work with "Imagenet" data_type. """
                )

        if data_folders is None and images_df is None:
            data = ArcGISImageList.from_folder(path / 'images') \
                .split_by_rand_pct(val_split_pct, seed=seed) \
                .label_from_func(get_y_func)
        else:
            if images_df is not None:
                src = ArcGISImageList.from_df(images_df, 'images')
                src.items = images_df[images_df.columns[0]].values
                src = src.split_by_rand_pct(val_split_pct, seed=seed) \
                        .label_from_func(get_y_func)
            else:
                # MultiFolder Training
                imageslist = []
                for data_folder in data_folders:
                    imageslist.append(ArcGISImageList.from_folder(data_folder / 'images').items)
                src = ArcGISImageList(np.concatenate(imageslist)) \
                    .split_by_rand_pct(val_split_pct, seed=seed) \
                    .label_from_func(get_y_func)
            data = src
        #
        _show_batch_multispectral = show_batch_labeled_tiles

        if dataset_type == 'Imagenet':
            if class_mapping is None:
                class_mapping = {}
                index = 1
                for class_name in data.classes:
                    class_mapping[index] = class_name
                    index = index + 1

        if transforms is None:
            ranges = (0, 1)
            if _image_space_used == _map_space:
                train_tfms = [
                    rotate(degrees=30, p=0.5),
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    dihedral_affine(),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(0.75, 1.5))
                ]
            else:
                train_tfms = [
                    rotate(degrees=30, p=0.5),
                    crop(size=chip_size, p=1., row_pct=ranges, col_pct=ranges),
                    brightness(change=(0.4, 0.6)),
                    contrast(scale=(0.75, 1.5))
                ]
            val_tfms = [crop(size=chip_size, p=1.0, row_pct=0.5, col_pct=0.5)]
            transforms = (train_tfms, val_tfms)
    elif dataset_type == "superres" or dataset_type == "Export_Tiles":
        path_hr = path/'images'
        path_lr = path/'labels'
        il = ImageList.from_folder(path_hr)
        hr_suffix = il.items[0].suffix
        img_size = il[0].shape[1]
        downsample_factor = kwargs.get('downsample_factor', None)
        if downsample_factor is None:
            downsample_factor = 4
        path_lr_check = path/f'esri_superres_labels_downsample_factor.txt'
        prepare_label = False
        if path_lr_check.exists():
            with open(path_lr_check) as f:
                label_downsample_ratio = float(f.read())
            if label_downsample_ratio != downsample_factor:
                prepare_label = True
        else:
            prepare_label = True
        if prepare_label:
            parallel(partial(resize_one, path_lr=path_lr, size=img_size/downsample_factor, path_hr=path_hr, img_size=img_size), il.items, max_workers=databunch_kwargs.get('num_workers'))
            with open(path_lr_check, 'w') as f:
                f.write(str(downsample_factor))

        data = ImageImageListSR.from_folder(path_lr)\
            .split_by_rand_pct(val_split_pct, seed=seed)\
            .label_from_func(lambda x: path_hr/x.with_suffix(hr_suffix).name)
        if resize_to is None:
            kwargs_transforms['size'] = img_size
        kwargs_transforms['tfm_y'] = True

        if has_esri_files:
            json_file = path / 'esri_model_definition.emd'
            with open(json_file) as f:
                emd = json.load(f)
        
    elif dataset_type in ['ner_json','BIO','IOB','LBIOU','BILUO']:
        if batch_size == 64:
            batch_size = 8
        encoding = kwargs.get("encoding", "UTF-8")
        ner_architecture = kwargs.get("ner_architecture", "spacy")
        data = _NERData(dataset_type=dataset_type, path=path, class_mapping=class_mapping, seed=seed,
                                val_split_pct=val_split_pct, batch_size=batch_size, encoding=encoding)
        if working_dir is not None:
            data.working_dir = path = Path(os.path.abspath(working_dir))
        else:
            path = os.path.abspath(data.path)
            data.working_dir = None
        if os.path.isfile(path):
            path = os.path.dirname(path)
        data._temp_folder = _prepare_working_dir(path)
        return data

    elif dataset_type == "PointCloud":
        from ._utils.pointcloud_data import Transform3d
        if transforms is None:
            transform_fn = Transform3d()
        elif transforms is False:
            transform_fn = None
        else:
            transform_fn = transforms
        data = pointcloud_prepare_data(path, class_mapping, batch_size, val_split_pct, dataset_type, transform_fn, **kwargs)
        data._data_path = data.path
        if working_dir is not None:
            data.path = Path(os.path.abspath(working_dir))
        data._temp_folder = _prepare_working_dir(data.path)
        return data

    elif dataset_type == "ImageCaptioning":
        from ._utils.image_captioning_data import prepare_captioning_dataset
        data = prepare_captioning_dataset(path,
                                          chip_size,
                                          batch_size,
                                          val_split_pct,
                                          transforms,
                                          resize_to,
                                          **kwargs)
        if working_dir is not None:
            data.path = Path(os.path.abspath(working_dir))
        data._temp_folder = _prepare_working_dir(data.path)
        return data
    elif dataset_type == "ChangeDetection":
        from ._utils.change_detection_data import prepare_change_detection_data
        kwargs.pop('rgb_bands', None)
        kwargs.pop('bands', None)
        kwargs.pop('norm_pct', None)
        return prepare_change_detection_data(path,
                                             chip_size,
                                             batch_size,
                                             val_split_pct,
                                             transforms,
                                             _is_multispectral=_is_multispectral,
                                             rgb_bands=rgb_bands,
                                             bands=bands,
                                             extract_bands=kwargs.pop('extract_bands', None),
                                             norm_pct=norm_pct,
                                             working_dir = working_dir,
                                             **kwargs)

    elif dataset_type == "CycleGAN":
        if _is_multispectral:
            data = prepare_data_ms_cyclegan(path, norm_pct, val_split_pct, seed, databunch_kwargs)
            data.n_channel = data.x[0].data[0].shape[0]
            data._is_multispectral = _is_multispectral
            data._imagery_type = _imagery_type
            data._imagery_type_a = imagery_type_a
            data._imagery_type_b = imagery_type_b
            data._bands = _bands
            data._norm_pct = norm_pct
            data._extract_bands = None
            data._do_normalize = False
            data._image_space_used = _image_space_used
            x_shape = data.train_ds[0][0].shape
            data.chip_size = x_shape[-1]
            if working_dir is not None:
                data.path = Path(os.path.abspath(working_dir))
            data._temp_folder = _prepare_working_dir(data.path)
            return data
        data = ImageTupleList.from_folders(path, path_a, path_b)\
                .split_by_rand_pct(val_split_pct, seed=seed)\
                .label_empty()
        img_size = data.x[0].shape[-1]
        if resize_to is None:
            kwargs_transforms['size'] = img_size
    elif dataset_type == "Pix2Pix":
        if _is_multispectral:
            data = prepare_data_ms_pix2pix(path, norm_pct, val_split_pct, seed, databunch_kwargs)
            data.n_channel = data.x[0].data[0].shape[0]
            data._is_multispectral = _is_multispectral
            data._imagery_type = _imagery_type
            data._imagery_type_a = imagery_type_a
            data._imagery_type_b = imagery_type_b
            data._bands = _bands
            data._norm_pct = norm_pct
            data._extract_bands = None
            data._do_normalize = False
            data._image_space_used = _image_space_used
            x_shape = data.train_ds[0][0].shape
            data.chip_size = x_shape[-1]
            if working_dir is not None:
                data.path = Path(os.path.abspath(working_dir))
            data._temp_folder = _prepare_working_dir(data.path)
            return data
        data = (ImageTupleList2.from_folders(path, path_a, path_b)
                      .split_by_rand_pct(val_split_pct, seed=seed)
                      .label_empty())
        img_size = data.x[0].shape[-1]
        if resize_to is None:
            kwargs_transforms['size'] = img_size
    elif dataset_type == "ObjectTracking":
        from ._utils.object_tracking_data import prepare_object_tracking_data
        data = prepare_object_tracking_data(path, batch_size, val_split_pct)
        data._is_multispectral = False
        data._extract_bands = None
        data._do_normalize = False
        data.chip_size = 127
        data._temp_folder = _prepare_working_dir(path)

        return data

    else:
        raise NotImplementedError('Unknown dataset_type="{}".'.format(dataset_type))
    
    if _is_multispectral:
        # Normalize multispectral imagery by calculating stats
        if dataset_type == 'RCNN_Masks':
            kwargs['do_normalize'] = False

        data = (data.transform(transforms, **kwargs_transforms)
                    .databunch(**databunch_kwargs))

        if "InputRastersProps" in emd:
            # Starting with ArcGIS Pro 2.7 and Python API for ArcGIS 1.9, the following multispectral kwargs have been
            # deprecated. This is done in favour of the newly added support for Imagery statistics and metadata in the
            # IA > Export Training data for Deep Learining GP Tool.
            #
            #   bands, rgb_bands, norm_pct,
            #
            data._emd = emd
            data._sensor_name = emd['InputRastersProps']['SensorName']
            bands = data._band_names = emd['InputRastersProps']['BandNames']
            # data._band_mapping = {i: k for i, k in enumerate(bands)}
            # data._band_mapping_reverse = {k: i for i, k in data._band_mapping.items()}
            data._nbands = len(data._band_names)
            band_min_values = []
            band_max_values = []
            band_mean_values = []
            band_std_values = []
            for band_stats in  emd['AllTilesStats']:
                band_min_values.append(band_stats['Min'])
                band_max_values.append(band_stats['Max'])
                band_mean_values.append(band_stats['Mean'])
                band_std_values.append(band_stats['StdDev'])

            data._rgb_bands = rgb_bands
            data._symbology_rgb_bands = rgb_bands

            data._band_min_values = torch.tensor(band_min_values, dtype=torch.float32)
            data._band_max_values = torch.tensor(band_max_values, dtype=torch.float32)
            data._band_mean_values = torch.tensor(band_mean_values, dtype=torch.float32)
            data._band_std_values = torch.tensor(band_std_values, dtype=torch.float32)
            data._scaled_min_values = torch.zeros((data._nbands,), dtype=torch.float32)
            data._scaled_max_values = torch.ones((data._nbands,), dtype=torch.float32)
            data._scaled_mean_values = _tensor_scaler(data._band_mean_values, min_values=data._band_min_values, max_values=data._band_max_values, mode='minmax')
            data._scaled_std_values = data._band_std_values*(data._scaled_mean_values/data._band_mean_values)

            # Handover to next section
            norm_pct = 1
            bands = data._band_names
            rgb_bands = symbology_rgb_bands = sniff_rgb_bands(data._band_names)
            if rgb_bands is None:
                rgb_bands = []
                if len(data._band_names) < 3:
                    symbology_rgb_bands = [0] # Panchromatic
                else:
                    symbology_rgb_bands = [0, 1, 2] # Case where could not find RGB in multiband imagery
        else:
            symbology_rgb_bands = rgb_bands
            if len(data.x) < 300:
                norm_pct = 1

            # Statistics
            dummy_stats = {
                "batch_stats_for_norm_pct_0" : {
                    "band_min_values":None,
                    "band_max_values":None,
                    "band_mean_values":None,
                    "band_std_values":None,
                    "scaled_min_values":None,
                    "scaled_max_values":None,
                    "scaled_mean_values":None,
                    "scaled_std_values":None
                }
            }
            normstats_json_path = os.path.abspath(data.path / '..' / 'esri_normalization_stats.json')
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
                batch_stats = _get_batch_stats(data.x, norm_pct)
                normstats[norm_pct_search] = dict(batch_stats)
                for s in normstats[norm_pct_search]:
                    if normstats[norm_pct_search][s] is not None:
                        normstats[norm_pct_search][s] = normstats[norm_pct_search][s].tolist()
                with open(normstats_json_path, 'w', encoding='utf-8') as f:
                    json.dump(normstats, f, ensure_ascii=False, indent=4)

            # batch_stats -> [band_min_values, band_max_values, band_mean_values, band_std_values, scaled_min_values, scaled_max_values, scaled_mean_values, scaled_std_values]
            data._band_min_values = batch_stats['band_min_values']
            data._band_max_values = batch_stats['band_max_values']
            data._band_mean_values = batch_stats['band_mean_values']
            data._band_std_values = batch_stats['band_std_values']
            data._scaled_min_values = batch_stats['scaled_min_values']
            data._scaled_max_values = batch_stats['scaled_max_values']
            data._scaled_mean_values = batch_stats['scaled_mean_values']
            data._scaled_std_values = batch_stats['scaled_std_values']
        #

        # Prevent Divide by zeros
        data._band_max_values[data._band_min_values == data._band_max_values]+=1
        data._scaled_std_values[data._scaled_std_values == 0]+=1e-02
        
        # Scaling
        data._min_max_scaler = partial(_tensor_scaler, min_values=data._band_min_values, max_values=data._band_max_values, mode='minmax')
        data.valid_ds.x._div = (data._band_min_values, data._band_max_values)
        data.train_ds.x._div = (data._band_min_values, data._band_max_values)

        # Normalize
        data._do_normalize = True
        if kwargs.get('do_normalize', None) is not None:
            data._do_normalize = kwargs.get('do_normalize', True)
        if data._do_normalize:
            data = data.normalize(stats=(data._scaled_mean_values, data._scaled_std_values), do_x=True, do_y=False)
        
    elif dataset_type == 'RCNN_Masks':
        data = (data.transform(transforms, **kwargs_transforms)
                .databunch(**databunch_kwargs))
        data.show_batch = types.MethodType( show_batch_rcnn_masks, data)
        # Exceptional case
        # We are dividing image pixel values by 255, at the time of opening it for rcnn masks
        # Not normalizing imagery here because model will normalize image internally
        data.train_ds.x._div = 255.
        data.valid_ds.x._div = 255.
    elif dataset_type == "Pix2Pix":
        data = (data.transform(get_transforms(), **kwargs_transforms)
            .databunch(**databunch_kwargs)) 
        data.n_channel = data.x[0].data[0].shape[0]
        data._imagery_type_a = imagery_type_a
        data._imagery_type_b = imagery_type_b
    
    elif dataset_type == "superres" or dataset_type == "Export_Tiles":
        data = (data.transform(get_transforms(), **kwargs_transforms)
            .databunch(**databunch_kwargs)
            .normalize(imagenet_stats, do_y=True))
    elif dataset_type == "CycleGAN":
        data = (data.transform(get_transforms(), **kwargs_transforms)
            .databunch(**databunch_kwargs))
        data.n_channel = data.x[0].data[0].shape[0] 
        data._imagery_type_a = imagery_type_a
        data._imagery_type_b = imagery_type_b      
    else:
        data = (data.transform(transforms, **kwargs_transforms)
            .databunch(**databunch_kwargs)
            .normalize(imagenet_stats))
        # RGB Image    if has_esri_files:
        #         data._emd = emd
        # We need to divide image pixel values by 255. at the time of opening it
        # because same method is used to open multispectral imagery as well
        # and that workflow depends on the imagery specific stats
        # Inflating imagenet_stats by 255x should have also worked
        # But fastai transforms clip image value to 1 and
        # in fastai 1.0.60 transforms are applied before normalization
        data.train_ds.x._div = 255.
        data.valid_ds.x._div = 255.

    # Imagery type used while opening image chips
    data._imagery_type = imagery_type
    data.train_ds.x._imagery_type = data._imagery_type
    data.valid_ds.x._imagery_type = data._imagery_type

    # Assigning chip size from training dataset and not data.x 
    # to consider transforms and resizing
    x_shape = data.train_ds[0][0].shape
    data.chip_size = x_shape[-1]
    data._val_split_pct = val_split_pct

    # Alpha channel check with GDAL
    if HAS_GDAL and x_shape[0] == 4:
        if data._imagery_type == 'ASSUMED_RGB':
            message = f"""
            Could not infer Imagery Type, Found 4 Bands in input imagery. Please set the optional parameter 'imagery_type' to an appropriate value.
            \nIf the imagery used to export the training data is a RGB imagery, please continue training by specifying `imagery_type='RGB'`.
            \nIf the imagery used to export the training data is a multispectral imagery containing information in the 4th band, please check the documentation for parameter 'imagery_type' to find a suitable value. 
            """
            raise Exception(message)

    if has_esri_files and dataset_type not in ['CycleGAN', 'Pix2Pix', 'ChangeDetection', 'superres']:
        data._dataset_type = stats['MetaDataMode']
    else:
        data._dataset_type = dataset_type
    
    if dataset_type == "superres" or dataset_type == "Export_Tiles":
        data._dataset_type = "SuperResolution"

    if alter_class_mapping:
        new_mapping = {}
        for i, class_name in enumerate(class_mapping.keys()):
            new_mapping[i+1] = class_name
        class_mapping = new_mapping

    ## For calculating loss from inverse of frquency.
    if dataset_type == 'Classified_Tiles':
        pixel_stats = stats.get('ClassPixelStats', stats.get('FeatureStats', None))
        if pixel_stats is not None:
            data.num_pixels_per_class = pixel_stats.get('NumPixelsPerClass',
                                                        pixel_stats.get('NumFeaturesPerClass', None))
        else:
            data.num_pixels_per_class = None

        ## Might want to change the variable name
        if data.num_pixels_per_class is not None:
            num_pixels_per_class = np.array(data.num_pixels_per_class, dtype=np.int64)
            if num_pixels_per_class.sum() < 0:
                data.overflow_encountered = True
                data.class_weight = None
            else:
                _num_pixels_per_class = np.copy(num_pixels_per_class)
                _num_pixels_per_class[_num_pixels_per_class==0]=1
                data.class_weight = num_pixels_per_class.sum() / _num_pixels_per_class
                data.class_weight[num_pixels_per_class==0]=0
        else:
            data.class_weight = None

    data.class_mapping = class_mapping
    data.color_mapping = color_mapping
    data.show_batch = types.MethodType(
        types.FunctionType(
            data.show_batch.__code__, data.show_batch.__globals__, data.show_batch.__name__,
            (min(int(math.sqrt(data.batch_size)), 5), *data.show_batch.__defaults__[1:]), data.show_batch.__closure__
        ),
        data
    )
    data.orig_path = path
    data.resize_to = kwargs_transforms.get('size', None)
    data.height_width = height_width
    data.downsample_factor = kwargs.get("downsample_factor")
    data.dataset_type = dataset_type
    
    data._is_multispectral = _is_multispectral
    if data._is_multispectral:
        data._bands = bands
        data._norm_pct = norm_pct
        data._rgb_bands = rgb_bands
        data._symbology_rgb_bands = symbology_rgb_bands

        # Handle invalid color mapping 
        data._multispectral_color_mapping = color_mapping
        if any( -1 in x for x in data._multispectral_color_mapping.values() ):
            random_color_list = np.random.randint(low=0, high=255, size=(len(data._multispectral_color_mapping), 3)).tolist()
            for i, c in enumerate(data._multispectral_color_mapping):
                if -1 in data._multispectral_color_mapping[c]:
                    data._multispectral_color_mapping[c] = random_color_list[i]

        # prepare color array
        alpha = kwargs.get('alpha', 0.7)
        color_array = torch.tensor( list(data.color_mapping.values()) ).float() / 255
        alpha_tensor = torch.tensor( [alpha]*len(color_array) ).view(-1, 1).float()
        color_array = torch.cat( [ color_array, alpha_tensor ], dim=-1)
        background_color = torch.tensor( [[0, 0, 0, 0]] ).float()
        data._multispectral_color_array = torch.cat( [background_color, color_array] )

        # Prepare unknown bands list if bands data is missing
        if data._bands is None:
            n_bands = data.x[0].data.shape[0]
            if n_bands == 1:# Handle Pancromatic case
                data._bands = ['p']
                data._symbology_rgb_bands = [0]
            else:
                data._bands = ['u' for i in range(n_bands)]
                if n_bands == 2:# Handle Data with two channels
                    data._symbology_rgb_bands = [0]
        
        # 
        if data._rgb_bands is None:
            data._rgb_bands = []

        # 
        if data._symbology_rgb_bands is None:
            data._symbology_rgb_bands = [0, 1, 2][:min(n_bands, 3)]

        # Complete symbology rgb bands
        if len(data._bands) > 2 and len(data._symbology_rgb_bands) < 3:
            data._symbology_rgb_bands += [ min(max(data._symbology_rgb_bands)+1, len(data._bands)-1)  for i in range(3 - len(data._symbology_rgb_bands)) ]
        
        # Overwrite band values at r g b indexes with 'r' 'g' 'b'
        for i, band_idx in enumerate(data._rgb_bands):
            if band_idx is not None:
                if data._bands[band_idx] == 'u':
                    data._bands[band_idx] = ['r', 'g', 'b'][i]

        # Attach custom show batch
        if _show_batch_multispectral is not None:
            data.show_batch = types.MethodType( _show_batch_multispectral, data )

        # Apply filter band transformation if user has specified extract_bands otherwise add a generic extract_bands
        """
        extract_bands : List containing band indices of the bands from imagery on which the model would be trained. 
                        Useful for benchmarking and applied training, for reference see examples below.
                        
                        4 band naip ['r, 'g', 'b', 'nir'] + extract_bands=[0, 1, 2] -> 3 band naip with bands ['r', 'g', 'b'] 

        """
        data._extract_bands = kwargs.get('extract_bands', None) 
        if data._extract_bands is None:
            data._extract_bands = list(range(len(data._bands)))
        else:
            data._extract_bands_tfm = partial(_extract_bands_tfm, band_indices=data._extract_bands)
            data.add_tfm(data._extract_bands_tfm)   

        # Tail Training Override
        _train_tail = True
        if [data._bands[i] for i in data._extract_bands] == ['r', 'g', 'b']:
            _train_tail = False
        data._train_tail = kwargs.get('train_tail', _train_tail)

    if not_label_count[0]:
            logger = logging.getLogger()
            logger.warning("Please check your dataset. " + str(not_label_count[0]) + " images dont have the corresponding label files.")

    data._image_space_used = _image_space_used

    if has_esri_files:
        data._emd = emd
        
    if working_dir is not None:
        data.path = Path(os.path.abspath(working_dir))
    else:
        data.path = Path(os.path.dirname(os.path.abspath(data.path)))
    data._temp_folder = _prepare_working_dir(data.path)

    from ._utils.env import _IS_ARCGISPRONOTEBOOK
    if _IS_ARCGISPRONOTEBOOK:
        from functools import wraps
        from matplotlib import pyplot as plt
        data._show_batch_orig = data.show_batch
        @wraps(data.show_batch)
        def show_batch_wrapper(rows=2, *args, **kwargs):
            res = data._show_batch_orig(rows, *args, **kwargs)
            plt.show()
            return res
        data.show_batch = show_batch_wrapper

    return data