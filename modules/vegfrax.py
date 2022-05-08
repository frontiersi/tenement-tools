# vegfrax
'''
VegFrax extracts pixel values in a high resolution classified image 
that fall within randomlly selected low- to moderate-resolution pixels 
from another raster (e.g. Landsat or Sentinel) and determines the 
proportion of each of the different pixel classes within. Simply 
speaking, if a high-resolution raster has a pixel size of 1m and 
the moderate image has a pixel size of 10m, each random sample 
will have a window that captures 100 1m class pixels within - the 
percentage of each class in that window is calculated. For example, 
25 forest pixels would = 25% forest.

See associated Jupyter Notebook vegfrax.ipynb for a basic tutorial on 
the main functions and order of execution.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os
import sys
import random
import xarray as xr
import numpy as np
import pandas as pd

sys.path.append('../../shared')
import tools

from osgeo import ogr

import dask.array as dask_array
from dask_ml.wrappers import ParallelPostFit

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
    
import joblib

def prepare_classified_xr(ds, dtype='int8'):
    """
    Does basic checks and corrects on a classified
    raster opened from a local raster. Converts to 
    integers, renames class var, etc.
    
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset which will be prepared.

    Returns
    ----------
    ds : xarray dataset/array with prepared info.
    """
    
    # notify
    print('Preparing classified dataset.')
    
    # check xr type, dims in ds a
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x and/or y coordinate dimension in dataset.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if more than one var - reject if so
    if len(ds) != 1:
        raise ValueError('More than one variable detected. Ensure raster has one band.')   
        
    # rename classified var, if exists
    temp_vars = [var for var in ds]
    if len(temp_vars) == 1:
        ds = ds.rename({temp_vars[0]: 'classes'})
    else:
        raise ValueError('No variables in dataset.')
    
    # convert to int16
    ds = ds.astype(dtype)
    
    if was_da:
        ds = ds.to_array()

    # notify
    print('Prepared classified dataset successfully.')
    return ds


def prepare_raw_xr(ds, dtype='float32', conform_nodata_to=-128):
    """
    Does basic checks and corrects on a raw
    raster opened from a load ard. Converts to 
    median all-time and masks nodata value.
    
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset which will be prepared.
    dtype: string
        Data type of output dataset.
    conform_nodata_to : int or float
        Convert all detected nodata values
        to this value.

    Returns
    ----------
    ds : xarray dataset/array with prepared info.
    """
    
    # notify
    print('Preparing raw dataset.')
    
    # check xr type, dims in ds a
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x and/or y coordinate dimension in dataset.')
                
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # convert to requested dtype
    ds = ds.astype(dtype)
        
    # check if no data val attributes exist, replace with nan
    if hasattr(ds, 'nodatavals') and ds.nodatavals is not None:

        # check if nodata values a iterable, if not force it
        nds = ds.nodatavals
        if not isinstance(nds, (list, tuple)):
            nds = [nds]

        # mask nan for nodata values
        for nd in nds:
            ds = ds.where(ds != nd, conform_nodata_to)

        # update xr attributes to new nodata val
        if hasattr(ds, 'attrs'):
            ds.attrs.update({'nodatavals': conform_nodata_to})

        # convert from float64 to float32 if nan is nodata
        if conform_nodata_to is np.nan:
            ds = ds.astype(np.float32)

    else:
        # mask via provided nodata
        print('No NoData values found in raster.')
        ds.attrs.update({'nodatavals': 'unknown'})

    if was_da:
        ds = ds.to_array()

    # notify
    print('Prepared raw dataset successfully.')
    return ds


def reclassify_xr(ds, req_class, merge_classes=None, inplace=True):
    """
    Reclassify classes in dataset so requested classes are kept but
    all others are transformed to a value of 0. Nodata is also
    kept as requested.

    Parameters
    ----------
    ds : xarray dataset/array
        A dataset holding the low resolution, raw raster bands.
    req_class : int, list
        A list of requested classes in dataset. Could be a single class,
        or could be multiple.
    merge_classes: bool
        Merge all classes provided into 1, everything else 0. In other
        words, make a binary class.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.    

    Returns
    ----------
    ds: xarray dataset/array
        A reclassified xr dataset/array.
    """
    
    # notify
    print('Reclassifying classes.')
    
    # check xr type, dims in ds a
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset a not an xarray type.')
    
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)

    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if xr has nodatavals
    if not hasattr(ds, 'nodatavals'):
        raise ValueError('Dataset does not have nodata value attribute.')
    elif ds.nodatavals == 'unknown':
        raise ValueError('Dataset nodata value is unknown.')
        
    # check if req classes is list or int, convert to list
    req_class = req_class if req_class is not None else []
    req_classes = req_class if isinstance(req_class, list) else [req_class]
                    
    # add nodata to required classes
    classes = req_classes + [ds.nodatavals]
        
    # remove unrequested classes
    ds = ds.where(ds.isin(classes), 0)
    
    # if merge, merge
    if merge_classes:

        # merge classes other than 0 and nan
        ds = ds.where(ds.isin([0, ds.nodatavals]), 1)
    
    if was_da:
        ds = ds.to_array()

    # notify
    print('Reclassified dataset successfully.')
    return ds


def get_xr_classes(ds):
    """
    Takes an xarray dataset/array and extracts unique class
    values from dataset values. Returns a list of class values.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.

    Returns
    ----------
    classes : list
        List of unique classes.
    """
    
    # notify user
    print('Getting unique classes from dataset.')

    # check if dataset
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset is not an xarray dataset or array type.')  
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimension in dataset.')
    elif len(ds) != 1:
        raise ValueError('Dataset can only take one variable (i.e. 1 band).')
        
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if xr has nodatavals
    if not hasattr(ds, 'nodatavals'):
        raise ValueError('Dataset does not have nodata value attribute.')
    elif ds.nodatavals == 'unknown':
        raise ValueError('Dataset nodata value is unknown.')

    # get all unique classes in dataset and remove nodata
    np_classes = np.unique(ds.to_array())
    np_classes = np_classes[np_classes != ds.nodatavals]

    # check if something came back or too much came back
    if len(np_classes) <= 0:
        raise ValueError('No classes detected in dataset.')
    elif len(np_classes) > 100:
        print('Warning: >= 100 classes detected in dataset. Proceed with caution.')
        
    # notify and return
    str_classes = ', '.join([str(c) for c in np_classes])
    print('Detected classes in dataset: {}'.format(str_classes))
    
    # convert to list
    return np_classes.tolist()


def generate_random_samples(ds_raw, ds_class, num_samples=1000, snap=True, res_factor=3):
    """
    Generates random point locations within dataset mask. These points are used to
    train the random forest classifier. A pandas dataframe of x and y coordinates 
    extracted from the dataset are returned. This is a custom func specifically for 
    the gdv fractional cover module.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    num_samples: int
        A int indicating how many points to generated.
    snap : bool
        If true, pixel centroids will replace our random sample point x and y values
        to reduce chance of assigning a random sample point right on the border of 
        multiple pixels. Default is True.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_samples: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """
    
    # notify user
    print('Generating {0} randomised sample points.'.format(num_samples))

    # check if raw dataset is xarray dataeset type
    if not isinstance(ds_raw, xr.Dataset):
        raise ValueError('> Raw dataset is not an xarray dataset.')
        
    # check if class dataset is xarray dataeset type
    if not isinstance(ds_class, xr.Dataset):
        raise ValueError('> Classified dataset is not an xarray dataset.')
    
    # check if number of absence points is an int
    if not isinstance(num_samples, int):
        raise ValueError('> Num of points value is not an integer. Please check the entered value.')
        
    # check if number of absence points is an int
    if not isinstance(snap, bool):
        raise ValueError('> Snap must be a boolean (True or False.')
        
    # check if number of absence points is an int
    if not isinstance(res_factor, int):
        raise ValueError('> Resolution factor must be an integer.')
        
    # get cell resolution for both datasets    
    res_raw = tools.get_xr_resolution(ds_raw)
    res_class = tools.get_xr_resolution(ds_class)
    
    # check if class res greater than raw - 
    if res_class >= res_raw:
        raise ValueError('> Classified raster is lower resolution than raw raster(s) - must be other way around.')
                
    # create a dummy grid based on a single slice of raw data
    if 'variable' in ds_raw.to_array().dims:
        raw_dummy = xr.ones_like(ds_raw.to_array().isel(variable=0))
        raw_dummy = raw_dummy.drop('variable', errors='ignore')
    else:
        raise AttributeError('> Raw dataset does not contain a variable dim.')
        
    # get extent of class raster
    class_extent = tools.get_xr_extent(ds=ds_class)
        
    # 'clip' raw dataset to class image bounds incase it is small subsection
    raw_dummy = raw_dummy.sel(x=slice(class_extent.get('l'), class_extent.get('r')), 
                              y=slice(class_extent.get('t'), class_extent.get('b')))   

    # get 10% of x, y size, use to trim 10% off extent bounds, subset dummy with it
    x_slice, y_slice = round(raw_dummy['x'].size * 0.1), round(raw_dummy['y'].size * 0.1)
    raw_dummy = raw_dummy.isel(x=slice(x_slice, -x_slice), y=slice(y_slice, -y_slice))
    
    # check if raw dummy has pixels still
    if raw_dummy.size <= 0:
        raise ValueError('> No pixels exist when raw and classified rasters clipped. Do they overlap?')
            
    # ensure enough pixels in final dummy to handle num of random samples
    num_cells = raw_dummy['x'].size * raw_dummy['y'].size
    if num_samples > num_cells:
        print('Too many random samples requested - reducing to: {0}.'.format(num_cells))
        num_samples = num_cells

    # notify
    print('> Randomising points within mask area.')

    # get bounds of final dummy
    dummy_extent = tools.get_xr_extent(ds=ds_class)
        
    # create random points and fill a list with x and y
    counter = 0
    coords = []
    for i in range(num_samples):
        while counter < num_samples:

            # get random x and y coord
            rand_x = random.uniform(dummy_extent.get('l'), dummy_extent.get('r'))
            rand_y = random.uniform(dummy_extent.get('b'), dummy_extent.get('t'))

            # create point and add x and y to it
            pnt = ogr.Geometry(ogr.wkbPoint)
            pnt.AddPoint(rand_x, rand_y)
            
            try:
                # get pixel
                pixel = raw_dummy.sel(x=rand_x, y=rand_y, method='nearest', tolerance=res_raw * res_factor)

                # add if pixel is valid value
                if int(pixel) == 1:
                    
                    # get x and y based on snap (where point was generated or nearest pixel centroid)
                    if snap:
                        coords.append([pixel['x'].item(), pixel['y'].item()])
                    else:
                        coords.append([pnt.GetX(), pnt.GetY()])

                    # inc counter
                    counter += 1
           
            except:
                continue
                
    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_samples = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    raw_dummy, buff_geom = None, None

    # notify and return
    print('> Generated random sample points successfully.')
    return df_samples


def generate_strat_random_samples(ds_raw, ds_class, req_class=None, num_samples=500, 
                                  snap=True, res_factor=3):
    """
    Generates stratified random point locations within dataset classes. These points 
    are used to train the random forest classifier. A pandas dataframe of x and y coordinates 
    extracted from the dataset are returned. This is a custom func specifically for 
    the gdv fractional cover module.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    req_class : int, list
        A integer or list of class numbers in which to generate
        random stratified samples within.
    num_samples: int
        A int indicating how many points to generated.
    snap : bool
        If true, pixel centroids will replace our random sample point x and y values
        to reduce chance of assigning a random sample point right on the border of 
        multiple pixels. Default is True.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_samples: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """

    # notify user
    print('Generating {} stratified randomised sample points.'.format(num_samples))

    # check if raw dataset is xarray dataeset type
    if not isinstance(ds_raw, xr.Dataset):
        raise ValueError('Raw dataset is not an xarray dataset.')
        
    # check if class dataset is xarray dataeset type
    if not isinstance(ds_class, xr.Dataset):
        raise ValueError('Classified dataset is not an xarray dataset.')
    
    # check if number of absence points is an int
    if not isinstance(num_samples, int):
        raise ValueError('Number of points value is not an integer.')
    elif not isinstance(snap, bool):
        raise ValueError('Snap must be a boolean (True or False).')
    elif not isinstance(res_factor, int):
        raise ValueError('Resolution factor must be an integer.')
        
    # check if req classes is list or int, convert to list
    req_class = req_class if req_class is not None else []
    req_classes = req_class if isinstance(req_class, list) else [req_class]
        
    # get cell resolution for both datasets    
    res_raw = tools.get_xr_resolution(ds_raw)
    res_class = tools.get_xr_resolution(ds_class)
    
    # check if class res greater than raw
    if res_class >= res_raw:
        raise ValueError('Classified raster must be higher resolution than raw raster(s).')
                
    # create a dummy grid based on a single slice of raw data
    if 'variable' in ds_raw.to_array().dims:
        raw_dummy = xr.ones_like(ds_raw.to_array().isel(variable=0))
        raw_dummy = raw_dummy.drop('variable', errors='ignore')
    else:
        raise AttributeError('> Raw dataset does not contain a variable dim.')
                       
    # get extent of class raster
    class_extent = tools.get_xr_extent(ds=ds_class)
        
    # 'clip' raw dataset to class image bounds incase it is small subsection
    raw_dummy = raw_dummy.sel(x=slice(class_extent.get('l'), class_extent.get('r')), 
                              y=slice(class_extent.get('t'), class_extent.get('b')))   

    # get 5% of x, y size, use to trim 5% off extent bounds, subset dummy with it
    x_slice, y_slice = round(raw_dummy['x'].size * 0.05), round(raw_dummy['y'].size * 0.05)
    raw_dummy = raw_dummy.isel(x=slice(x_slice, -x_slice), y=slice(y_slice, -y_slice))
    
    # check if raw dummy has pixels still
    if raw_dummy.size <= 0:
        raise ValueError('No pixels exist when raw and classified rasters clipped. Do they overlap?')
            
    # ensure enough pixels in final dummy to handle num of random samples
    num_cells = raw_dummy['x'].size * raw_dummy['y'].size
    if num_samples > num_cells:
        print('Too many random samples requested - reducing to: {}.'.format(num_cells))
        num_samples = num_cells
        
    # get bounds of final dummy
    dummy_extent = tools.get_xr_extent(ds=raw_dummy)  

    # notify
    print('Generating stratified random points.')
    
    coords = []
    for req_class in req_classes:
        
        # notify
        print('Preparing samples for class: {}'.format(req_class))

        # convert to pandas
        df_mask = xr.where(ds_class == req_class, True, False).to_dataframe()
        df_mask = df_mask.loc[df_mask['classes'] == True].sample(5000)
        df_mask = df_mask.drop('classes', axis='columns')

        # create random points and fill a list with x and y
        counter = 0
        while counter < num_samples:

            # get a single, random row
            xy = df_mask.sample(1)
            xy = xy.index.to_list()[0]
            x, y = xy[0], xy[1]

            try:
                # get pixel value from low res ds
                pixel = raw_dummy.sel(x=x, y=y, 
                                      method='nearest', 
                                      tolerance=res_raw * res_factor)
                
                # removed check for valid (i.e., val = 1)

                if snap:
                    coords.append([float(pixel['x']), 
                                   float(pixel['y'])])
                else:
                    coords.append([float(x), 
                                   float(y)])
                
                # up the counter
                counter += 1

            except:
                continue
      
    # check if list is populated
    if not coords:
        raise ValueError('No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_samples = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    raw_dummy = None

    # notify and return
    print('Generated stratified random sample points successfully.')
    return df_samples


def create_frequency_windows(ds_raw, ds_class, df_records):
    """
    Generates "windows" of pixels captured from the classified
    dataset that fall within each raw dataset pixels that intersect 
    the random sample point coordinates. These windows represent this
    data as a 1d array of classes that fall within each window.
    These windows should then be passed to the create frequency
    windows function to convert into actual fractions.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    df_records : pandas dataframe
        A integer or list of class numbers in which random
        sample coordinates exist.

    Returns
    ----------
    df_windows: pandas dataframe
        A pandas dataframe containing class values captured
        within "windows" of low resolution pixels.
    """
    
    # notify user
    print('Creating frequency focal windows from random sample points.')

    # check if xarray dataset
    if not isinstance(ds_raw, xr.Dataset):
        raise TypeError('Raw dataset not a xarray dataset type.')
    elif not isinstance(ds_class, xr.Dataset):
        raise TypeError('Classified dataset not a xarray dataset type.')
    elif not isinstance(df_records, pd.DataFrame):
        raise TypeError('Not a pandas dataframe type.')

    # get cell resolution of raw rasters
    res_raw = tools.get_xr_resolution(ds=ds_raw)
    res_class = tools.get_xr_resolution(ds=ds_class)

    # create a copy of extraction samples with only x and y cols, add empty col for win vals
    df_windows = df_records[['x', 'y']].copy()
    df_windows['win_vals'] = ''

    # build window extents for each random point
    df_windows['l'] = df_windows['x'] - res_raw / 2
    df_windows['r'] = df_windows['x'] + res_raw / 2
    df_windows['b'] = df_windows['y'] - res_raw / 2
    df_windows['t'] = df_windows['y'] + res_raw / 2

    # if class dataset not chunked, chunk it now
    if not bool(ds_class.chunks):
        ds_class = ds_class.chunk({'x': 'auto', 'y': 'auto'})

    # convert to data array    
    da_class = ds_class.to_array()

    # calc expected window size
    expected_win_size = (res_raw / res_class) ** 2

    # loop each window, extract class pixels, count skipped
    window_list = []
    skip_count = 0
    for i, r in df_windows.iterrows():
        # generate x and y coords steps wrt class raster interval, extract values via nn
        x_values = np.arange(r.get('l'), r.get('r'), step=res_class)
        y_values = np.arange(r.get('b'), r.get('t'), step=res_class)
        window = da_class.interp(x=x_values, y=y_values, method='nearest')

        # if win is of adequate size convert to dask array and append, else skip
        if window.size == expected_win_size:
            window = dask_array.array(window)
            window_list.append(window)
        else:
            df_windows = df_windows.drop(i)
            skip_count += 1

    # check if anything returned
    if len(window_list) == 0:
        raise ValueError('No windows were returned. Cannot proceed. Please check your data.')

    # notify user
    print('Computing windows into memory, this can take awhile. Please wait.')

    # get original data type and compute, reset index incase rows removed
    np_windows = dask_array.concatenate(window_list).compute()
    df_windows = df_windows.reset_index(drop=True)

    # check if data sizes match - they must
    if len(df_windows) != len(np_windows):
        raise ValueError('Number of points and window arrays do not match.')
          
    # iter rows and add 1d window values to each point
    for i, r in df_windows.iterrows():
        df_windows.at[i, 'win_vals'] = np_windows[i].flatten().astype(da_class.dtype)

    # drop unneeded boundary cols
    df_windows = df_windows.drop(columns=['l', 'r', 't', 'b'])

    # notify user
    if skip_count == 0:
        print('{} windows generated successfully.'.format(len(df_records)))
    else:
        print('{0} windows generated successfully. {} omitted.'.format(len(df_records) - skip_count, skip_count))

    # return
    return df_windows


def convert_window_counts_to_freqs(df_windows, nodata_value=-9999):
    """
    Convert a pandas dataframe of raw class counts within
    sample windows into actual frequencies (0-100%). Currently
    transforms data to float to handle potential nan. 

    Parameters
    ----------
    df_windows : pandas dataframe
        A pandas dataframe holding raw class counts
        within each sample window.
    nodata_value : int or float
        A value representing the nodata values in dataset.

    Returns
    ----------
    df_freq: pandas dataframe
        A pandas dataframe containing class frequencies
        captured within "windows" of low resolution pixels.
    """
    
    # notify user
    print('Generating class frequencies from window pixels.')
    
    # copy dataframe
    df_freq = df_windows.copy(deep=True)
    
    # get datatype of arrays in win vals
    win_dtype = df_freq['win_vals'][0].dtype
    
    # create empty class label and freqs columns, update dtypes
    df_freq['class_lbls'], df_freq['class_frqs'] = '', ''

    # iter each row and calc unique classes, counts and freqs
    for i, r in df_freq.iterrows():
        # if valid...
        if isinstance(r['win_vals'], np.ndarray) and len(r['win_vals']) > 0:    
            # get dtype of classes (labels)
            lbl_dtype = r['win_vals'].dtype
            if lbl_dtype not in ['float16', 'float32', 'float64']:
                lbl_dtype = 'float16' # handle nan
                
            # get size of window
            win_size = len(r['win_vals'])

            # get unique classes, counts, frequencies
            lbls, cnts = np.unique(r['win_vals'], return_counts=True)
            frqs = cnts / win_size
            
            # update lbls datatype
            lbls = lbls.astype(lbl_dtype)

            # append nodata class and freq if missing
            if nodata_value not in lbls:
                lbls = np.insert(lbls, 0, nodata_value)
                frqs = np.insert(frqs, 0, 0.0)

            # add each numpy array to current row
            df_freq.at[i, 'class_lbls'] = lbls.astype(lbl_dtype)
            df_freq.at[i, 'class_frqs'] = frqs.astype(lbl_dtype)

        else:
            # reset values to nan for easy drop later on
            df_freq.at[i, 'class_lbls'] = np.nan
            df_freq.at[i, 'class_frqs'] = np.nan
            
    # notify user
    print('Checking for empty rows and dropping if exist.')
            
    # count nan values for labels and freqs
    lbls_nan_count = df_freq['class_lbls'].isna().sum()
    frqs_nan_count = df_freq['class_frqs'].isna().sum()

    # if nan, tell user and remove rows
    if lbls_nan_count > 0 or frqs_nan_count > 0:
        print('Empty rows detected. Dropping {0} rows.'.format(np.max(lbls_nan_count, frqs_nan_count)))
        df_freq.dropna(subset=['class_lbls', 'class_frqs'])
        
    # drop unneeded boundary cols
    df_freq = df_freq.drop(columns=['win_vals'])
    
    # reset index in case rows removed
    df_freq = df_freq.reset_index(drop=True)
    
    # notify user
    if len(df_freq) > 0:
        print('{0} windows transformed to frequencies successfully.'.format(len(df_freq)))
    else:
        raise ValueError('No frequencies were calculated. Aborting.')
           
    # return
    return df_freq


def prepare_freqs_for_analysis(ds_raw, ds_class, df_freqs, override_classes=[]):
    """
    Takes a dataframe with class frequencuies obtained from
    within classified windows and prepares for analysis. This mostly
    involves checking nodata frequency and removing from window
    if exists. This flags rows to include or exclude from final
    analysis, essentially.

    Parameters
    ----------
    ds_raw : xarray dataset
        A xr dataset with raw bands or indices as vars.
    ds_class : xarray dataset
        A xr dataset with a classified variable.
    df_freqs : pandas dataframe
        A pandas dataframe holding frequencies of class
        within each sample window.
    override_classes : list
        List of 1 or more classes to perform analysis
        on. Useful to limit classes to several instead
        of all.

    Returns
    ----------
    df_data: pandas dataframe
        A pandas dataframe containing analysis ready frequencies
        captured within "windows" of low resolution pixels.
    """

    # notify user
    print('Converting dataset to sklearn analysis-ready format.')

    # get class datatype
    class_dtype = ds_class.to_array().dtype
    
    # set up override int to list etc
    # 
    
    
    # check if xr has nodatavals
    if not hasattr(ds_class, 'nodatavals'):
        raise ValueError('Dataset does not have nodata value attribute.')
    elif ds_class.nodatavals == 'unknown':
        raise ValueError('Dataset nodata value is unknown.')
    

    # prepare classes we want for analysis dataframe - override or from dataset
    if len(override_classes) > 0:
        np_dataset_classes = np.unique((np.array(override_classes, dtype=class_dtype)))    
    else:
        np_dataset_classes = np.unique(ds_class.to_array())

    # check if no data col exists, if not, add it
    if ds_class.nodatavals not in np_dataset_classes:
        np_dataset_classes = np.insert(np_dataset_classes, 0, ds_class.nodatavals)

    # get unique class values sampled from windows
    np_sampled_classes = np.unique(np.concatenate(df_freqs['class_lbls']))

    # get missing classes - convert to strings to be safe
    np_missing = np.setdiff1d(np_dataset_classes, np_sampled_classes)

    # check if anything exists, throw error if so
    if len(np_missing) > 0:
        raise ValueError('One or more requested classes not captured within windows.')

    # notify 
    print('Converting classes and frequencies into analysis-ready format.')

    # create new dataframe where columns are class labels
    df_coords = pd.DataFrame(columns=['x', 'y'], dtype=np.dtype('float64'))
    df_data = pd.DataFrame(columns=np_dataset_classes, dtype=np.dtype('float32'))

    # create row template dict
    row_dict = {}
    for c in np_dataset_classes:
        row_dict[c] = 0.0

    # loop frequency rows, update black values to frequency for selected classes
    for i, r in df_freqs.iterrows():
        row = row_dict.copy()
        
        # get coords
        coords = {'x': r['x'], 'y': r['y']}
        
        # update row values with existing values
        lbls, freqs = r[['class_lbls', 'class_frqs']] 
        for lbl, frq in zip(lbls, freqs):
                        
            # include only classes we want
            if lbl in np_dataset_classes:
                row[lbl] = round(frq, 4)       

        # append to coords and analysis dataframes
        df_coords = df_coords.append(coords, ignore_index=True)
        df_data = df_data.append(row, ignore_index=True)

    # notify
    print('Normalising rows with regards to subset classes and NoData values.')

    # normalise frequencies within row
    for i, r in df_data.iterrows():
        df_data.at[i] = r / (1 - r[ds_class.nodatavals])

    # notify
    print('Creating NoData mask.')

    # we create an include column and base it on value in nodata value col
    df_data['include'] = df_data[ds_class.nodatavals] <= 0

    # drop nodata col
    df_data = df_data.drop(columns=[ds_class.nodatavals])
    
    # ensure dataframes not empty
    if df_data.isnull().values.all() or df_coords.isnull().values.all():
        raise ValueError('> Dataframe(s) empty or all rows null.')

    # add x and y columns on to data frame
    df_data = pd.concat([df_coords, df_data], axis=1)

    # notify and return
    print('Data successfully prepared for analysis.')
    return df_data

# set up function for random forest prediction
def __predict__(ds_input, estimator):
    """
    Helper function for perform_predictions(), should not be called directly.
    """

    # get x, y dims
    x, y, = ds_input['x'], ds_input['y']

    # get crs if exists
    try:
        attributes = ds_input.attrs
    except:
        print('No attributes available. Skipping.')
        attributes = None

    # seperate each var (image bands) and store in list
    input_data_list = []
    for var_name in ds_input.data_vars:
        input_data_list.append(ds_input[var_name])

    # flatten and chunk each dim array and add to flatten list
    input_data_flat = []
    for da in input_data_list:
        data = da.data.flatten().rechunk(chunk_size)  #!?!
        input_data_flat.append(data)

    # reshape for prediction via dask array type (dda)
    input_data_flat = dask_array.array(input_data_flat).transpose()

    # perform the prediction
    preds = estimator.predict(input_data_flat)     
            
    # reshape for output
    preds = preds.reshape(len(y), len(x))

    # recreate dataset
    ds_out = xr.DataArray(preds, coords={'x': x, 'y': y}, dims=['y', 'x'])
    ds_out = ds_out.to_dataset(name='result')

    # add attributes back on
    if attributes:
        ds_out.attrs.update(attributes)

    return ds_out

def perform_prediction(ds_input, estimator):  
    """
    Uses dask (if available) to run sklearn predict in parallel.
    Useful for quickly performing analysis.

    Parameters
    ----------
    ds_input : xarray dataset or array. 
             Dataset containing independent variables(i.e. low res image). Must 
             have dimensions 'x' and 'y'.
    estimator : sklearn estimator object
             A pre-defined RandomForestRegressor scikit-learn estimator model. 

    Returns
    ----------
    ds_out : xarray dataset
             An xarray dataset containing the probabilities of the random forest model.
    """
    
    # check ds in dataset or dataarray
    if not isinstance(ds_input, (xr.Dataset, xr.DataArray)):
        raise TypeError('> Input dataset is not xarray dataset or data array type.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds_input, xr.DataArray):
        try:
            was_da = True
            ds_input = ds_input.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if x and y dims exist
    if 'x' not in list(ds_input.dims) and 'y' not in list(ds_input.dims):
        raise ValueError('> No x and/or y coordinate dimension in dataset.')
    
    # if input_xr isn't dask, coerce it
    is_dask = True
    if not bool(ds_input.chunks):
        is_dask = False
        ds_input = ds_input.chunk({'x': len(ds_input.x), 'y': len(ds_input.y)})

    #get chunk size
    chunk_size = int(ds_input.chunks['x'][0]) * int(ds_input.chunks['y'][0])

    # predict via parallel, or if missing, regular compute
    if is_dask == True:
        estimator = ParallelPostFit(estimator)
        with joblib.parallel_backend('dask'):
            ds_out = __predict__(ds_input, estimator)
    else:
        ds_out = __predict__(ds_input, estimator).compute()

    # return
    return ds_out


def perform_optimised_validation(X, y, n_estimators=100, n_validations=10, split_ratio=0.10):
    """
    Perform model fit with a set number of estimators and validations. Model
    is fit using randomlly selected training and testing set. Provides
    mean metric of MAE, MSE, RMSE, R-squared. Produces an random
    forest estimator.

    Parameters
    ----------
    X : numpy ndarray
        A numpy array of dependent values.
    y : numpy ndarray
        A numpy array of response values.
    n_estimators : int
        Number of random forest estimators.
    n_validations : int
        Number of cross-validations for accuracy metrics.
    split_ratio : float
        Percentage of which ti split data into training and
        testing sets.

    Returns
    ----------
    estimator : xarray dataset
             An xarray dataset containing the probabilities of the random forest model.
    """
    
    # check for tyoe
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('X and/or y inputs must be numpy arrays.')
    elif not len(X.shape) == 2 or not len(y.shape) == 1:
        raise ValueError('X and/or y inputs are of incorrect size.')
        
    # check validations, split
    if n_estimators < 1:
        raise ValueError('Number of estimators must be > 0.')
    elif n_validations < 0:
        raise ValueError('Number of validations must be betwen 0 and 1.')
    elif split_ratio < 0 or split_ratio > 1:
        raise ValueError('Split ratio must be between 0 and 1.')
        
    # create a new random forest regressor
    estimator = RandomForestRegressor(n_estimators=n_estimators,
                                      max_features='sqrt',
                                      oob_score=True)

    result_list = []
    for i in range(n_validations):
        # split train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=split_ratio, 
                                                            random_state=None, 
                                                            shuffle=True)     

        # fit, predict model
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)

        # get accuracy measurements
        result_dict = {}
        result_dict['oob'] = estimator.oob_score_
        result_dict['mae'] = metrics.mean_absolute_error(y_test, pred)
        result_dict['rmse'] = metrics.mean_squared_error(y_test, pred, squared=False)
        result_dict['r2'] = metrics.r2_score(y_test, pred)
            
        # add to list
        result_list.append(result_dict)
        
    # unpack mean measures
    oob =    round(np.mean([r['oob'] for r in result_list]),  2)
    mae =    round(np.mean([r['mae'] for r in result_list]),  2)
    rmse =   round(np.mean([r['rmse'] for r in result_list]), 2)
    r2 =     round(np.mean([r['r2'] for r in result_list]),   2)
    max_r2 = round(np.max([r['r2'] for r in result_list]),  2)
    
    # notify and return
    msg = 'Mean OOB: {}. Mean MAE: {}. Mean RMSE: {}. Mean R-squared: {}. Best R-squared: {}'
    print(msg.format(oob, mae, rmse, r2, max_r2))
    return estimator
 
    
def perform_fca(ds_raw, ds_class, df_data, df_extract_clean, n_estimators=100, n_validations=10, split_ratio=0.10):
    """
    Perform model fit with a set number of estimators and validations. Model
    is fit using randomlly selected training and testing set. Provides
    mean metric of MAE, MSE, RMSE, R-squared. Produces an random
    forest estimator.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    df_data : pandas dataframe
        A pandas dataframe holding analysis ready class frequencies
        within each sample window. i.e., dependent variable.
    df_extract_clean : pandas dataframe
        A pandas dataframe holding analysis ready low resolution
        independent variables (i.e. spectral bands).
    n_estimators : int
        Number of random forest estimators.
    n_validations : int
        Number of cross-validations for accuracy metrics.
    split_ratio : float
        Percentage of which ti split data into training and
        testing sets.

    Returns
    ----------
    ds_preds : xarray dataset
        An xarray dataset containing the probabilities of the random forest model.
    """
        
    # notify
    print('Beginning fractional cover analysis (FCA).')
    
    # do checks

    # get original class raster dtype
    class_dtype = ds_class.to_array().dtype

    # remove x, y and include columns and get numpy array
    df_cols = df_data.drop(columns=['x', 'y', 'include']).columns

    # convert classes from dataframe to numpy
    np_classes = np.array(df_cols, dtype=class_dtype)

    # loop each class, fit model, predict onto raster, give accuraccy assessment
    pred_list = []
    for c in np_classes:
        # notify
        print('Fitting and predicting model for class: {0}'.format(c))

        # exclude flagged rows from analysis
        df_data_sub = df_data.loc[df_data['include'] == True]

        # combine dataframes to align samples
        df_merged = pd.merge(df_data_sub, df_extract_clean, on=['x', 'y'])

        # get independent vars out, excluding x and y, convert to numpy
        indep_cols = df_extract_clean.drop(columns=['x', 'y']).columns
        X = df_merged[indep_cols].to_numpy()

        # get dependent var (class col), convert to flatten 1d numpy
        y = df_merged[[c]].to_numpy().flatten()

        # optimise validation
        estimator = perform_optimised_validation(X=X, y=y, 
                                                 n_estimators=n_estimators, 
                                                 n_validations=n_validations, 
                                                 split_ratio=0.10)
            

        # predict onto raw dataset, rename and append
        ds_pred = perform_prediction(ds_raw, estimator)
        ds_pred = ds_pred.rename({'result': str(c)})
        pred_list.append(ds_pred)

    # merge result together
    ds_preds = xr.merge(pred_list)   

    # notify and return
    print('Fractional cover analysis (FCA) completed successfully.')
    return ds_preds
