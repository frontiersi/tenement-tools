# ensemble
'''
This script contains functions for performing Dempster-Shafer belief modelling.
The intention is to take 2 or more raster / netcdf file inputs, designate each
as either belief or disbelief, rescale each to 0-1 range using fuzzy sigmoidals
from the canopy module, then output belief, disbelief, plausability maps.
See https://www.sciencedirect.com/topics/computer-science/dempster-shafer-theory
for a good overview of Dempster-Shafer theory. Use of Dempster-Shafer allows us
to combine multiple evidence layers showing potential groundwater dependent 
vegetation into one, potentially improving statistical robustness of the model.
The plausability map also provides an assessment of areas that may be under sampled
and need attention.

See associated Jupyter Notebook ensemble.ipynb for a basic tutorial on the
main functions and order of execution.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('../../modules')
import canopy

sys.path.append('../../shared')
import satfetcher, tools


def prepare_data(file_list, nodataval):
    """
    Takes a list of filepath strings in which to open into xarray
    datasets. Calls satfetcher module functions for loading.

    Parameters
    ----------
    file_list: list
        A list of strings representing geotiff or netcdf filepaths.
    nodataval : numeric or numpy nan
        A value ion which to standardise all nodata values to in
        each provided input file.

    Returns
    ----------
    da_list : list of xarray dataset or arrays.
    """
    
    # check if file list is list
    if not isinstance(file_list, list):
        file_list = [file_list]
    
    # iter files and load depending on file type
    da_list = []
    for lyr_path in file_list:

        # get filename and extension
        lyr_fn = os.path.basename(lyr_path)
        lyr_name, lyr_ext = os.path.splitext(lyr_fn)

        # lazy load dataset depending on tif or nc
        if lyr_ext == '.tif':
            ds = satfetcher.load_local_rasters(rast_path_list=lyr_path, 
                                               use_dask=True, 
                                               conform_nodata_to=nodataval)    
        elif lyr_ext == '.nc':
            ds = satfetcher.load_local_nc(nc_path=lyr_path, 
                                          use_dask=True, 
                                          conform_nodata_to=nodataval)
        else:
            print('File {} was not a tif or netcdf - skipping.'.format(lyr_fn))
            continue
            
        # check if shape correct, convert array
        if ds.to_array().shape[0] != 1:
            print('More than one variable detected - skipping.')
            continue

        # append 
        da_list.append(ds)
        
    # if only one item, remove from list
    if len(da_list) == 1:
        da_list = da_list[0]
        
    # return
    return da_list


def apply_auto_sigmoids(items):
    """
    Takes a list of arrays with elements as [path, a, bc, d, ds] 
    and using this information, rescales each dataset 
    using fuzzy sigmoidal function. Auto-detects which fuzzy
    sigmoidal to apply based on values in a, bc, d. Output
    is an updated dataset for each array in list, rescaled to
    0 to 1.

    Parameters
    ----------
    items: list
        A list of arrays with elements as [path, a, bc, d, ds] .
        Path represents path of raster/netcdf, a, bc, d are values
        for inflection points on sigmoidal (e.g., a is low inflection, 
        bc is mid-point or max inflection, and ds represents
        the raw, un-scaled xarray dataset to be rescaled.

    Returns
    ----------
    items : list of rescaled array(s).
    """

    def get_min_max(ds, x):
        if x == 'Min':
            x = float(np.min(ds.to_array()))
        elif x == 'Max':
            x = float(np.max(ds.to_array()))
        return x
    
    for item in items:
        
        # fetch elements
        path = item[0]
        a, bc, d = item[1], item[2], item[3]
        ds = item[4]
        
        # expect 5 items and xr dataset...
        if len(item) != 5:
            continue
        elif ds is None:
            continue

        # get ds attributes
        attrs = ds.attrs

        # convert min or max type inflect points
        a = get_min_max(ds, a)
        bc = get_min_max(ds, bc)
        d = get_min_max(ds, d)

        # inc sigmoidal
        if a is not None and bc is not None and d is None:
            print('Applying increasing sigmoidal to {}'.format(path))
            ds = canopy.inc_sigmoid(ds, a=a, b=bc)

        # dec sigmoidal
        elif a is None and bc is not None and d is not None:
            print('Applying decreasing sigmoidal to {}'.format(path))
            ds = canopy.dec_sigmoid(ds, c=bc, d=d)

        # bell sigomoidal
        elif a is not None and bc is not None and d is not None:
            print('Applying bell sigmoidal to {}'.format(path))
            ds = canopy.bell_sigmoid(ds, a=a, bc=bc, d=d)

        # reapply attributes and update original list item
        ds.attrs = attrs
        item[4] = ds
        
    #return
    return items


def export_sigmoids(items, out_path):
    """
    Simple netcdf exporter. Exports fuzzy sigmoidal 
    versions of raster/netcdf file to netcdf file post-
    rescaling. Calls tools script for export code.

    Parameters
    ----------
    items: list
        A list of arrays with elements as [path, a, bc, d, ds] .
        Path represents path of raster/netcdf, a, bc, d are values
        for inflection points on sigmoidal (e.g., a is low inflection, 
        bc is mid-point or max inflection, and ds represents
        the raw, un-scaled xarray dataset to be rescaled. Only
        the ds element is considered.
        
    out_path : str
        A string for output file location and name.
    """
    
    # checks
    
    for item in items:
        
        # get elements
        in_path = item[0]
        ds = item[4]
        
        # get filename with netcdf ext
        fn = os.path.basename(in_path)
        fn = os.path.splitext(fn)[0]
        fn = fn + '_sigmoid' + '.nc'
        
        # notify
        print('Exporting sigmoidal {}'.format(fn))
        tools.export_xr_as_nc(ds, fn)
    

def append_dempster_attr(ds_list, dempster_label='belief'):
    """
    Helper functiont to append the dempster output type label
    to existing dataset. Just an xarray update function.

    Parameters
    ----------
    ds_list: list
        A list of xarray datasets.

    Returns
    ----------
    out_list : list of xarray datasets with appended attributes.
    """
    
    # check if list
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    
    # loop xr datasets and append dempster label
    out_list = []
    for ds in ds_list:
        ds.attrs.update({'dempster_type': dempster_label})
        out_list.append(ds)
        
    # return
    return out_list


def resample_datasets(ds_list, resample_to='lowest', resampling='nearest'):
    """
    Dumb but effective way of resampling one dataset to others. Takes
    a list of xarray datasets, finds lowest/highest resolution dataset
    within list, then resamples all other datasets to that resolution.
    Required for Dempster-Schafer ensemble modelling.

    Parameters
    ----------
    ds_list: list
        A list of xarray datasets.
        
    resample_to : str
        Either lowest or highest allowed. If highest, the dataset 
        with the highest resolution (smallest pixels) is used as the 
        resample template. If lowest, the opposite occurs.
    
    resampling: str
        Type of resampling method. Nearest neighjbour is default. See
        xarray resample method for more options.

    Returns
    ----------
    out_list : list of datasets outputs.
    """
    
    # notify
    print('Resampling datasets to {} resolution.'.format(resample_to))
    
    # get list of var/res/size dicts
    res_list = []
    for ds in ds_list:
        var_name = list(ds.data_vars)[0]
        res = tools.get_xr_resolution(ds)
        res_list.append({
            'name': var_name, 
            'res': res,
        })
        
    # get min/max res value
    if resample_to == 'lowest':
        val = max([r.get('res') for r in res_list])
    elif resample_to == 'highest':
        val = min([r.get('res') for r in res_list])
    else:
        raise ValueError('No resolution attribute available.')

    # get layer(s) where matches target res, pick first if many
    lyr_name_list = [r['name'] for r in res_list if r['res'] == val]
    if len(lyr_name_list) >= 1:
        resample_to_lyr = lyr_name_list[0]
    else:
        raise ValueError('No layers returned.')
        
    # get resample target xr dataset
    target_ds = [ds for ds in ds_list if list(ds.data_vars)[0] == resample_to_lyr]
    target_ds = target_ds[0] if isinstance(target_ds, list) else target_ds
        
    # iterate each layer and resample to target layer
    out_list = []
    for ds in ds_list:
        if list(ds.data_vars)[0] == resample_to_lyr:
            out_list.append(ds)
        else:
            out_list.append(tools.resample_xr(ds_from=ds, 
                                              ds_to=target_ds, 
                                              resampling=resampling))

    # return
    return out_list 


def perform_dempster(ds_list):
    """
    Performs Dempster-Schafer ensemble modelling. Creates site vs non-site 
    evidence layers and combines in to belief, disbelief and plausability 
    maps. Two or more layers in ds_list inout required.

    Parameters
    ----------
    ds_list: list
        A list of xarray datasets.
        
    resample_to : str
        Either lowest or highest allowed. If highest, the dataset 
        with the highest resolution (smallest pixels) is used as the 
        resample template. If lowest, the opposite occurs.
    
    resampling: str
        Type of resampling method. Nearest neighjbour is default. See
        xarray resample method for more options.

    Returns
    ----------
    ds : xarray dataset
        An xarray dataset containing belief, disbelief and plausability 
        variables.
    """

    # set up bpa's
    m_s, m_ns = None, None
    
    # split ds list into belief and disbelief
    da_s = [ds.to_array().squeeze(drop=True) for ds in ds_list if ds.dempster_type == 'belief']
    da_ns = [ds.to_array().squeeze(drop=True) for ds in ds_list if ds.dempster_type == 'disbelief']
    
    # generate site layer
    for idx, da in enumerate(da_s):
        if idx == 0:
            m_s = da
        else:
            m_s = (m_s * da) + ((1 - da) * m_s) + ((1 - m_s) * da)
            
    # generate non-site layer
    for idx, da in enumerate(da_ns):
        if idx == 0:
            m_ns = da
        else:
            m_ns = (m_ns * da) + ((1 - da) * m_ns) + ((1 - m_ns) * da)

    # generate final belief (site) layer
    da_belief = (m_s * (1 - m_ns)) / (1 - (m_ns * m_s))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_ns * (1 - m_s)) / (1 - (m_ns * m_s))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])

    return ds