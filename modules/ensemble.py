# ensemble
"""
"""

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('../../modules')
import canopy

sys.path.append('../../shared')
import satfetcher, tools

# check, meta
def prepare_data(file_list, nodataval):
    """
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

def __get_min_max__(ds, x):
    """
    Worker function for apply_auto_sigmoids()
    """
    if x == 'Min':
        x = float(np.min(ds.to_array()))
    elif x == 'Max':
        x = float(np.max(ds.to_array()))
    return x

# checks, meta
def apply_auto_sigmoids(items):
    """
    takes a list of items with elements in order [path, a, bc, d, ds].
    From that, will work out which sigmoidal to apply.
    """
    
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
        a = __get_min_max__(ds, a)
        bc = __get_min_max__(ds, bc)
        d = __get_min_max__(ds, d)

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

# checks, meta
def export_sigmoids(items, out_path):
    """
    Takes list of sigmoided datasets and exports them
    to desired folder with _sigmoid appended to end.
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
    
# checks, meta
def append_dempster_attr(ds_list, dempster_label='belief'):
    """
    Small helper function to append dempster label
    to xr dataset/array.
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

# check, meta
def resample_datasets(ds_list, resample_to='lowest', resampling='nearest'):
    """
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

# meta, checks, split is a bit weak
def perform_dempster(ds_list):
    """
    Performs dempster-schafer ensemble modelling. 
    Creates site vs non-site evidence layers and combines
    in to belief, disbelief and plausability maps.
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