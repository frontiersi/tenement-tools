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

from modules import canopy

from shared import satfetcher, tools

def smooth_xr_dataset(ds, win_size=3):
    """
    Basic moving window smoother via mean.

    Parameters
    ----------
    ds: xarray dataset
        An xarray datasets.
        
    win_size : int
        Size of the moving window. Must be odd number and 
        >= 3.

    Returns
    ----------
    ds : smoothed dataset.
    """

    # check xr
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset must have x and y dimensions.')
    if 'x' not in ds or 'y' not in ds:
        raise ValueError('Dataset must have x and y dimensions.')
    elif 'time' in ds:
        raise ValueError('Dataset must not have time dimension.')

    # check window size
    if win_size is None:
        raise ValueError('Window size not provided.')
    if not isinstance(win_size, int):
        raise ValueError('Window size must be integer.')
    elif win_size % 2 == 0:
        raise ValueError('Window size must be odd number.')
        
    try:
        # smoothing can extrapolate, so create flattened mask of nans
        ds_mask = xr.where(~ds.isnull(), 1, 0)
        ds_mask = ds_mask.to_array().sum('variable')
    except Exception as e:
        raise ValueError(e)

    try:
        # apply rolling window of user size
        ds = ds.rolling(x=win_size, 
                        y=win_size, 
                        center=True,
                        min_periods=1).mean()
                        
        # mask nan values back in 
        ds = ds.where(ds_mask != 0)
    except Exception as e:
        raise ValueError(e)
        
    # do one final check to see if any non-nans exist
    if ds.to_array().isnull().all() == True:
        raise ValueError('Smoothing resulted in empty dataset.')

    return ds


def perform_modelling(belief, disbelief):
    """
    Perform Dempster Shafer belief modelling using at least one 
    belief layer and at least one disbelief layer. Can have as many 
    belief and disbelief layers. Layers are expected to be either xarray 
    datasets

    Parameters
    ----------
    belief: list of xarray dataset
        List of xarray datasets holding belief layers.
        
    disbelief: list of xarray dataset
        List of xarray datasets holding disbelief layers.

    Returns
    ----------
    ds : dempster shafer result as dataset.
    """
    
    # check data type and size is right
    if belief is None or not isinstance(belief, list):
        raise TypeError('Belief must be a list of one or more datasets.')
    elif disbelief is None or not isinstance(disbelief, list):
        raise TypeError('Disbelief must be a list of one or more datasets.')
    
    # check belief, disbelief datasets are adequate
    for ds in belief + disbelief:
        if not isinstance(ds, (xr.DataArray, xr.Dataset)):
            raise TypeError('Belief/disbelief must be an xarray dataset or array.')
        elif 'x' not in ds or 'y' not in ds:
            raise ValueError('Belief/disbelief must have x and y dimensions.')
        elif 'time' in ds or 'time' in ds:
            raise ValueError('Belief/disbelief must not have time dimension.')

    # generate site layers (one or multi), force arrays
    m_site = None
    for idx, ds in enumerate(belief):
    
        # force to array
        if isinstance(ds, xr.Dataset):
            ds = ds.to_array()

        try:
            # if first ds just use ds, else site formula
            if idx == 0:
                m_site = ds
            else:
                m_site = (m_site * ds) + ((1 - ds) * m_site) + ((1 - m_site) * ds)
        except Exception as e:
            raise ValueError(e)
    
    # now generate non-site layers (one or multi), force arrays
    m_nonsite = None
    for idx, ds in enumerate(disbelief):
    
        # force to array
        if isinstance(ds, xr.Dataset):
            ds = ds.to_array()
            
        try:
            # if first ds just use ds, else non-site formula
            if idx == 0:
                m_nonsite = ds
            else:
                m_nonsite = (m_nonsite * ds) + ((1 - ds) * m_nonsite) + ((1 - m_nonsite) * ds)   
        except Exception as e:
            raise ValueError(e)
            
    try:
        # generate final belief (site) and disbelief (nonsite) layers
        da_belief = (m_site * (1 - m_nonsite)) / (1 - (m_nonsite * m_site))
        da_disbelief = (m_nonsite * (1 - m_site)) / (1 - (m_nonsite * m_site))

        # generate plausability, belief interval layers
        da_plauability = (1 - da_disbelief)
        da_interval = (da_plauability - da_belief)
    except Exception as e:
        raise ValueError(e)
       
    try:
        # combine into a single dataset with four vars
        ds = xr.merge([
            da_belief.to_dataset(name='belief'), 
            da_disbelief.to_dataset(name='disbelief'), 
            da_plauability.to_dataset(name='plausability'),
            da_interval.to_dataset(name='interval')])
            
        # remove residual variables if exist
        ds = ds.squeeze(drop=True)
    except Exception as e:
        raise ValueError(e)
    
    try:
        # create nan mask across all variables and apply
        ds_mask = xr.where(~ds.isnull(), 1, 0)
        ds_mask = ds_mask.to_array().sum('variable')
        ds = ds.where(ds_mask != 0, drop=True)
    except Exception as e:
        raise ValueError(e)
    
    # do one final check to see if any non-nans exist
    if ds.to_array().isnull().all() == True:
        raise ValueError('No non-nan values exist after modelling, returning None.')
    
    return ds




# deprecated!
def check_belief_disbelief_exist(in_lyrs):
    """
    Given a list of lists of dempster-shafer parameter
    values, check if the type element has at least one
    belief and disbelief type. If not, invalid is flagged.
    """
    
    belief_disbelief_list = []
    for lyr in in_lyrs:
        belief_disbelief_list.append(lyr[2])

    # check belief layers
    invalid = False
    if 'Belief' not in np.unique(belief_disbelief_list):
        invalid = True
    elif 'Disbelief' not in np.unique(belief_disbelief_list):
        invalid = True
        
    return invalid

# deprecated!
def prepare_data(file_list, var=None, nodataval=-999):
    """
    Takes a list of filepath strings in which to open into xarray
    datasets. Calls satfetcher module functions for loading.

    Parameters
    ----------
    file_list: list
        A list of strings representing geotiff or netcdf filepaths.
    var : string
        The name of a particular variable to extract from
        dataset with multiple variables.
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
        try:
            if lyr_ext == '.tif':
                ds = satfetcher.load_local_rasters(rast_path_list=lyr_path, 
                                                   use_dask=True, 
                                                   conform_nodata_to=nodataval)   
            elif lyr_ext == '.nc':
                ds = satfetcher.load_local_nc(nc_path=lyr_path, 
                                              use_dask=True, 
                                              conform_nodata_to=nodataval)
        except:
            raise ValueError('Could not load input: {}'.format(lyr_path))

        # check if crs is albers, else skipping
        try:
            crs = tools.get_xr_crs(ds)
        except:
            raise ValueError('Input: {} has no crs infomration.'.format(lyr_path))
            
        # check crs is albers
        if crs != 3577:
            raise ValueError('Input: {} crs is not projected in GDA94 Albers.'.format(lyr_path))

        # convert to dataset if need be
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(dim='variable')            
            
        # if variable given for nc, slice
        if lyr_ext == '.nc' and var is not None:
            ds = ds[var].to_dataset()
            
        # check if shape correct, convert array
        if ds.to_array().shape[0] != 1:
            raise ValueError('More than one band in input: {}. Only support one.'.format(lyr_path))

        # append 
        da_list.append(ds)
        
    # if only one item, remove from list
    if len(da_list) == 1:
        da_list = da_list[0]
        
    # return
    return da_list

# deprecated
def __get_min_max__(ds, x):
    """
    Worker function for apply_auto_sigmoids()
    """
    if x == 'Min':
        x = float(np.min(ds.to_array()))
    elif x == 'Max':
        x = float(np.max(ds.to_array()))
    return x

# deprecated! checks, meta
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
        A list of arrays with elements as [path, var, type, a, bc, d, ds] .
        Path represents path of raster/netcdf, a, bc, d are values
        for inflection points on sigmoidal (e.g., a is low inflection, 
        bc is mid-point or max inflection, and ds represents
        the raw, un-scaled xarray dataset to be rescaled.

    Returns
    ----------
    items : list of rescaled array(s).
    """
    
    for item in items:
        
        # fetch elements
        path = item[0]
        var = item[1]
        typ = item[2]
        a, bc, d = item[3], item[4], item[5]
        ds = item[6]
        
        # expect 7 items and xr dataset...
        if len(item) != 7:
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
        item[6] = ds
        
    #return
    return items

# deprecated
def seperate_ds_belief_disbelief(in_lyrs):
    """
    Takes a list of prepared dempster shafer list
    elements and seperates into two seperate lists,
    one for belief, one for disbelief.
    """
    
    belief_list, disbelief_list = [], []
    for lyr in in_lyrs:
        typ = lyr[2]
        
        # seperate
        if typ == 'Belief':
            belief_list.append(lyr[6])
        else:
            disbelief_list.append(lyr[6])
        
    # gimme
    return belief_list, disbelief_list

# deprecated
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
    
# deprecated
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

# DEPRECATED
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
    
    # generate belief interval
    da_interval = (da_plauability - da_belief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability'),
        da_interval.to_dataset(name='interval')
    ])

    return ds

# deprecated
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
