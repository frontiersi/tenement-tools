# gdvspectra
'''
This script contains functions for calculating groundwater-dependent 
vegetation (GDV) from landsat or sentinel data. This model has been 
validated for three key Pilbara, Western Australia species Euc. victrix, Euc. 
camaldulenesis and Mel. argentea. It offers a SMCE approach to detecting
this vegetation. GDV is detected using a time series of vegetation indices,
moisture indices and seasonal stability in an AHP process, resulting in
a GDV likelihood (probability) map. Thresholding can be implemented via
standard deviation or groundtruthed point locations. GDV health
trends can be determined using Mann-Kendall trend analysis, Theil-sen slopes, 
or Change Vector Analysis (CVA) functions. Finally, significant breaks 
in vegetation can be detected using change point detection.

See associated Jupyter Notebook gdvspectra.ipynb for a basic tutorial on the
main functions and order of execution.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau, theilslopes

sys.path.append('../../shared')
import tools

def subset_months(ds, month=None, inplace=True):
    """
    Takes a xarray dataset/array and a list of months in which to
    subset the xr dataset down to. Used mostly when subsetting data
    down to wet and dry season (e.g. list of 1, 2, 3 for wet and list
    of 9, 10, 11 for dry months). Time dimension required, or error 
    occurs.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    month : int or list
        An int or a list representing the month(s) that represent
        the months to subset data to. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Subsetting down to specified months.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
                
    # check wet, dry month if none given
    if month is None:
        raise ValueError('Must provide at least one month.')       
                
    # check wet dry list, convert if not
    months = month if isinstance(month, list) else [month]
    
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
    
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # reduce to wet, dry months
    try:
        print('Reducing dataset into months: {0}.'.format(months))
        ds = ds.sel(time=ds['time.month'].isin(months))
    
    except:
        raise ValueError('Could not subset to requested months.')
    
    # notify and return
    print('Subset to requested months successfully.')
    if was_da:
        ds = ds.to_array()
    
    return ds


def resample_to_wet_dry_medians(ds, wet_month=None, dry_month=None, inplace=True):
    """
    Takes a xarray dataset/array and a list of wet, dry months which 
    to resample to. An annualised wet and dry season median image for 
    given wet, dry months will be created. For example: one wet, one 
    dry image for 2018, one wet, one dry image for 2019, etc. Time 
    dimension required, or error occurs.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. 
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Resampling dataset to annual wet and dry medians.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
                
    # check wet, dry month if none given
    if wet_month is None or dry_month is None:
        raise ValueError('Must provide at least one wet and dry month.')    

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

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

    # if dask, must compute for resample median
    # note, there seems to be a dask-resample bug where nan is returned
    # randomly when dask resampled. leaving this here in case bug occurs 
    #was_dask = False
    #if bool(ds.chunks):
        #print('Dask detected, not supported here. Computing, please wait.')
        #was_dask = True
        #ds = ds.compute()

    # split into wet, dry
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)
    
    # check if any data remains 
    if len(ds_wet['time']) == 0 or len(ds_dry['time']) == 0:
        raise ValueError('Not enough data available within wet/dry season. Add more years/months.')

    # create month map
    month_map = {
        1:  'JAN',
        2:  'FEB',
        3:  'MAR',
        4:  'APR',
        5:  'MAY',
        6:  'JUN',
        7:  'JUL',
        8:  'AUG',
        9:  'SEP',
        10: 'OCT',
        11: 'NOV',
        12: 'DEC'
    }

    # get wet, dry start month as string
    wet_start_month = month_map.get(wet_month[0])
    dry_start_month = month_map.get(dry_month[0])

    # resample wet, dry into annual wet, dry medians
    ds_wet = ds_wet.resample(time='AS-' + wet_start_month).median(keep_attrs=True)
    ds_dry = ds_dry.resample(time='AS-' + dry_start_month).median(keep_attrs=True)

    # concat wet, dry datasets back together
    ds = xr.concat([ds_wet, ds_dry], dim='time').sortby('time')
    
    # if was dask, make dask again
    # related to above comment on dask-resample issue
    #if was_dask:
        #ds = ds.chunk({'time': 1})
    
    # notify and return
    print('Resampled dataset to annual wet and dry medians successfully.')
    if was_da:
        ds = ds.to_array()

    return ds


def resample_to_freq_medians(ds, freq='YS', inplace=True):
    """
    Takes a xarray dataset/array and a frequency value in which to 
    resample to. Based on pandas resample frequencies. This function
    is basically just an xarray resample wrapped up with some error
    handling.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    freq : str
        A frequency value for resampler. Example: 1MS, YS.
        Read pandas frequency docs for help.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Resampling dataset down to annual medians.')

    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')

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
    
    # if dask, must compute for resample median
    # note, there seems to be a dask-resample bug where nan is returned
    # randomly when dask resampled. leaving this here in case bug occurs 
    #was_dask = False
    #if bool(ds.chunks):
        #print('Dask detected, not supported here. Computing, please wait.')
        #was_dask = True
        #ds = ds.compute()
        
    # resample wet, dry into annual wet, dry medians
    ds = ds.resample(time=freq).median(keep_attrs=True)
    
    # if was dask, make dask again
    # related to above comment on dask-resample issue
    #if was_dask:
        #ds = ds.chunk({'time': 1})
    
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Resampled down to annual medians successfully.')
    return ds


def nullify_wet_dry_outliers(ds, wet_month=None, dry_month=None, p_value=0.01, inplace=True):
    """
    Takes a xarray dataset/array and a list of wet, dry months which 
    to check for outliers. Wet and dry is seperated and a z-score
    test is done along time dimensions. Users can modify the 'strictness'
    of the outlier removal via the p-value input. A p-value of 0.01 will 
    only remove the most significant outliers, 0.05 less so, and 0.1 will
    remove many. Value is not actually removed, just 'nulled' to np.nan.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. 
    p_value : float
        A float of 0.01, 0.05 or 0.1. Default is 0.01.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Nullifying wet, dry season outliers usign Z-Score test.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')

    # check wet, dry month if none given
    if wet_month is None or dry_month is None:
        raise ValueError('Must provide at least one wet and dry month.')    

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
    
    if p_value not in (0.10, 0.05, 0.01):
        print('P-value not supported. Setting to 0.01.')
        p_value = 0.01

    # set z_value based on user significance (p_value)
    if p_value == 0.10:
        z_value = 1.65
    elif p_value == 0.05:
        z_value = 1.96
    elif p_value == 0.01:
        z_value = 2.58
    else:
        p_value = 0.01
        z_value = 2.58

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
            
    # split into wet, dry
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

    # get mean value of each time and var for wet, dry
    ds_wet = ds_wet.groupby('time').mean(['x', 'y'])
    ds_dry = ds_dry.groupby('time').mean(['x', 'y'])

    # perform zscore on wet, dry
    z_wet = (ds_wet - ds_wet.mean()) / ds_wet.std()
    z_dry = (ds_dry - ds_dry.mean()) / ds_dry.std()

    # get dates where outliers (zscore > zvalue) for wet, dry
    z_wet = z_wet.where(abs(z_wet) > z_value, drop=True)
    z_dry = z_dry.where(abs(z_dry) > z_value, drop=True)
    
    # combine wet, dry times of outliers and mask original ds
    da_times = xr.concat([z_wet['time'], z_dry['time']], dim='time')
    
    # create message and notify
    outlier_times = []
    for t in da_times.sortby('time'):
        txt = '{0}-{1}-{2}'.format(int(t.dt.year), int(t.dt.month), int(t.dt.day))
        outlier_times.append(txt)
        
    # notify user of outliers
    if outlier_times:
        print('Outlier dates detected: {0}'.format(', '.join([t for t in outlier_times])))
    else:
        print('No outlier dates detected.')

    # set all values to nan for outlier times
    ds = ds.where(~ds['time'].isin(da_times))
        
    # notify and return
    print('Nullified wet, dry season outliers successfully.')

    if was_da:
        ds = ds.to_array()

    return ds


def drop_incomplete_wet_dry_years(ds, inplace=True):
    """
    Takes a xarray dataset/array and counts the number of
    wet, dry images per year. As we are expecting exactly 2,
    if this is not matched the year will be dropped. If a 
    resample was performed beforehand, the only two possible
    years that can be dropped is the first and last.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Dropping years where both wet, dry do not exist.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
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
    
    # get annual groups list, get first and last year info
    groups = list(ds.groupby('time.year').groups.items())
    
    # loop each year, check seasons, drop year
    removed_years = []
    for group in groups:
        if len(group[1]) != 2:
            ds = ds.where(ds['time.year'] != group[0], drop=True)
            removed_years.append(group[0])
            
    # notify
    if removed_years:
        removed_years = ', '.join(str(y) for y in removed_years)
        print('Warning: years {0} were dropped.'.format(removed_years))
    else:
        print('No uneven years detected, no data lost.')

    # return
    if was_da:
        ds = ds.to_array()

    return ds

# deprecated: helper func to fill edges via bfill, ffill
def __fill_edge__(ds, edge=None):
    """
    Helper function for fill_empty_wet_dry_edges()

    Parameters
    ----------
    ds: xarray dataset
        A dataset with x, y and time dims. NOTE: ONLY accepts dataset as it's a helper function.
    edge: string
        Variable to pass in edge value, accepts either 'first' or 'last'

    Returns
    ----------
    ds : xarray dataset

    """
    if not isinstance(ds, (xr.Dataset)):
        raise TypeError('__fill_edge__() can ONLY accept a Dataset.')

    # check edge
    if edge not in ['first', 'last']:
        raise ValueError('Edge must be first or last.')

    # create sort order
    asc = True
    if edge == 'last':
        asc = False

    # loop each da in ds
    for i, dt in enumerate(ds['time'].sortby('time', asc)):
        
        # get current datetime
        da = ds.sel(time=dt)
        
        # check if vars are all nan depending on xr type
        if isinstance(da, xr.Dataset):
            da_has_nans = da.to_array().isnull().all()
        elif isinstance(da, xr.DataArray):
            da_has_nans = da.isnull().all()
            
        # if edge empty, get next time with vals, fill
        if i == 0 and da_has_nans:
            print('{0} time is empty. Processing to fill.'.format(edge.title()))

        elif i == 0 and not da_has_nans:
            print('{0} time has values. No need to fill.'.format(edge.title()))
            break

        elif i > 0 and not da_has_nans:
            print('Performing backfill.')
            if edge == 'first':
                ds = xr.where(ds['time'] <= ds['time'].sel(time=dt), 
                                ds.bfill('time'), ds)
            elif edge == 'last':
                ds = xr.where(ds['time'] >= ds['time'].sel(time=dt), 
                                ds.ffill('time'), ds)
            break

    return ds


def __manual_bfill__(ds, gap_size=25):
    """
    The xarray backfill is not great, here is a more manual
    and quicker implementation.
    """
    
    # get original dataset type
    ds_type = ds.to_array().dtype
    
    # get a 1d array of all times of all pixels == nan, sum Trues for each var
    ds_all_nans = ds.isnull().all(['x', 'y']).to_array()
    ds_all_nans = ds_all_nans.sum('variable') > 0

    # backfill only if first index on left (earliest) is all nan
    if ds_all_nans.isel(time=0) == True:
        print('First index in dataset is all nan. Checking data.')

        # get first non-empty index and raw data array
        first_full_idx = np.where(ds_all_nans == False)[0][0]
        first_full_data = ds.isel(time=first_full_idx).to_array()

        # if first non nan index is <= 3 times away, fill, else dont
        if first_full_idx <= gap_size:
            print('Full index exists close to missing. Backfilling.')

            # set main ds to array temporarily
            ds = ds.to_array()

            # back fill data in first index
            ds[:, 0] = first_full_data
            
            # note: to fill all indices before first full, use this code
            #for i in range(0, first_full_idx):
                #ds[:, i] = first_full_data   

            # convert back to dataset original dtype
            ds = ds.to_dataset(dim='variable').astype(ds_type)
            
        else:
            print('Gap too large, not backfilling.')
        
    return ds


def __manual_ffill__(ds, gap_size=25):
    """
    The xarray forwardfill is not great, here is a more manual
    and quicker implementation.
    """
    
    # get original dataset type
    ds_type = ds.to_array().dtype
    
    # reverse order of ds by time
    ds = ds.sortby('time', ascending=False)
    
    # get a 1d array of all times of all pixels == nan, sum Trues for each var
    ds_all_nans = ds.isnull().all(['x', 'y']).to_array()
    ds_all_nans = ds_all_nans.sum('variable') > 0

    # backfill only if first index on left (earliest) is all nan
    if ds_all_nans.isel(time=0) == True:
        print('First index in dataset is all nan. Checking data.')

        # get first non-empty index and raw data array
        first_full_idx = np.where(ds_all_nans == False)[0][0]
        first_full_data = ds.isel(time=first_full_idx).to_array()

        # if first non nan index is <= 3 times away, fill, else dont
        if first_full_idx <= gap_size:
            print('Full index exists close to missing. Backfilling.')

            # set main ds to array temporarily
            ds = ds.to_array()

            # back fill data in first index
            ds[:, 0] = first_full_data
            
            # note: to fill all indices before first full, use this code
            #for i in range(0, first_full_idx):
                #ds[:, i] = first_full_data   

            # convert back to dataset original dtype
            ds = ds.to_dataset(dim='variable').astype(ds_type)
            
        else:
            print('Gap too large, not backfilling.')
        
    # sort back
    ds = ds.sortby('time', ascending=True)
    
    return ds


def fill_empty_wet_dry_edges(ds, wet_month=None, dry_month=None, inplace=True):
    """
    Takes a xarray dataset/array and a list of wet, dry months as lists.
    For wet and dry times, first and last time is checked if all nan. If
    all nan, a forward or back fill is performed to fill these missing 
    values in.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. 
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Filling empty wet and dry edges in dataset.')        

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check wet, dry month if none given
    if wet_month is None or dry_month is None:
        raise ValueError('Must provide at least one wet and dry month.')   
        
    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
        
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

    # split into wet, dry - we dont want to fill wet with dry, vice versa
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

    # fill edges for wet first, last
    try:
        print('Filling wet season edges.')
        ds_wet = __manual_bfill__(ds_wet, gap_size=25)
        ds_wet = __manual_ffill__(ds_wet, gap_size=25)
        #ds_wet = __fill_edge__(ds_wet, edge='first')
        #ds_wet = __fill_edge__(ds_wet, edge='last')
    except:
        print('Could not fill missing wet season edges. Skipping.')
        return ds

    # fill edges for wet first, last
    try:
        print('Filling dry season edges.')
        ds_dry = __manual_bfill__(ds_dry, gap_size=25)
        ds_dry = __manual_ffill__(ds_dry, gap_size=25)
        #ds_dry = __fill_edge__(ds_dry, edge='first')
        #ds_dry = __fill_edge__(ds_dry, edge='last')
    except:
        print('Could not fill missing dry season edges. Skipping.')
        return ds        
    
    # concat wet, dry datasets back together
    ds = xr.concat([ds_wet, ds_dry], dim='time').sortby('time')
    
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Filled empty wet and dry edges successfully.')
    return ds


def interp_empty_wet_dry(ds, wet_month=None, dry_month=None, method='full', inplace=True):
    """
    Takes a xarray dataset/array and performs linear interpolation across
    wet, dry time dimension seperatly. A concatenated xr dataset is returned.
    This is a wrapper for perform_interp function. The method can be set to full 
    or half. Full will use the built in xr interpolate_na method, which is robust 
    and dask friendly but very slow. The quicker alternative is half, which only
    interpolates times that are all nan. Despite being slower, full method 
    recommended.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. 
    method : str
        Set the interpolation method: full or half. Full will use 
        the built in interpolate_na method, which is robust and dask 
        friendly but very slow. The quicker alternative is half, which 
        only interpolates times that are all nan.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Interpolating empty values in dataset.')

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check if method is valid
    if method not in ['full', 'half']:
        raise ValueError('Method must be full or half.')
        
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

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # split into wet, dry - we dont want to fill wet with dry, vice versa
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)
    
    # interpolate for wet, then dry
    ds_wet = tools.perform_interp(ds=ds_wet, method=method)
    ds_dry = tools.perform_interp(ds=ds_dry, method=method)

    # concat wet, dry datasets back together
    ds = xr.concat([ds_wet, ds_dry], dim='time').sortby('time')

    if was_da:
         ds = ds.to_array()

    # notify and return
    print('Interpolated empty values successfully.')
    return ds      


def interp_empty(ds, method='full', inplace=True):
    """
    Takes a xarray dataset/array and performs linear interpolation across
    all times in dataset. This is a wrapper for perform_interp function. 
    The method can be set to full or half. Full will use the built in xr 
    interpolate_na method, which is robust and dask friendly but very slow. 
    The quicker alternative is half, which only interpolates times that are 
    all nan. Despite being slower, full method recommended.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    method : str
        Set the interpolation method: full or half. Full will use 
        the built in interpolate_na method, which is robust and dask 
        friendly but very slow. The quicker alternative is half, which 
        only interpolates times that are all nan.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Interpolating empty values in dataset.')

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check if method is valid
    if method not in ['full', 'half']:
        raise ValueError('Method must be full or half.')
        
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
    
    # interpolate for ds
    ds = tools.perform_interp(ds=ds, method=method)

    # notify and return
    print('Interpolated empty values successfully.')

    if was_da:
        ds = ds.to_array()
   

    return ds   


def interp_empty_months(ds, method='full', inplace=True):
    """
    Takes a xarray dataset/array and performs linear interpolation across
    each month seperatly. A concatenated xr dataset is returned. This is a 
    wrapper for perform_interp function. The method can be set to full 
    or half. Full will use the built in xr interpolate_na method, which is 
    robust and dask friendly but very slow. The quicker alternative is half, 
    which only interpolates times that are all nan. Despite being slower, 
    full method recommended.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    method : str
        Set the interpolation method: full or half. Full will use 
        the built in interpolate_na method, which is robust and dask 
        friendly but very slow. The quicker alternative is half, which 
        only interpolates times that are all nan.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Interpolating empty values along months in dataset.')

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check if method is valid
    if method not in ['full', 'half']:
        raise ValueError('Method must be full or half.')

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

    # get list of months within dataset
    months_list = list(ds.groupby('time.month').groups.keys())

    # loop each month in months list
    da_list = []
    for month in months_list:
        print('Interpolating along month: {0}'.format(month))

        # get subset for month
        da = ds.where(ds['time.month'] == month, drop=True)
        da = da.copy(deep=True)
        
        # interpolate for current month time series
        da = tools.perform_interp(ds=da, method=method)
            
        # append to list
        da_list.append(da)

    # concat back together
    ds = xr.concat(da_list, dim='time').sortby('time')
    
    # notify and return
    print('Interpolating empty values along months successfully.')

    if was_da:
        ds = ds.to_array()

    return ds


def calc_ivts(ds, ds_med, q_upper=0.99, q_lower=0.05):
    """
    Takes a xarray dataset/array and an all-time median of same data and calculates
    invariant targets. Invariant targets are pixels that are greenest/moistest
    across all time + most stable across all time. Greenest and moistest values
    use upper percentile to detect them, where as stable pixels generated via
    orthogonal polynomial coefficients and quantile closest to 0, or lowest
    percentile.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    ds_med : xarray dataset/array
        The all-time median version of above dataset. Generate before
        entering here.
    q_upper : float
        Set the upper percentile of vegetation/moisture values. We
        need the highest values to standardise to, but don't want to
        just take max. Default is 0.99 and typically produces optimal
        results.
    q_lower : float
        Set the lowest percentile of stability values. We need to find
        the most 'stable' pixels across time to ensure standardisation
        works. Default is 0.05 and typically produces optimal results.

    Returns
    ----------
    ds_targets : xarray dataset or array.
    """
    
    # notify
    print('Calculating invariant target sites.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check if all time median is valid
    if not isinstance(ds_med, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds_med.dims) or 'y' not in list(ds_med.dims):
        raise ValueError('No x or y dimensions in dataset.')
    
    # check q_lower < q_upper
    if q_upper <= q_lower:
        raise ValueError('Upper quantile value must be larger than lower quantile.')
    # check q_value 0-1
    elif q_upper < 0 or q_upper > 1:
        raise ValueError('Upper quantile value must be between 0 and 1.')
    # check q_value 0-1
    elif q_lower < 0 or q_lower > 1:
        raise ValueError('Lower quantile value must be between 0 and 1.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')
    
    if isinstance(ds_med, xr.DataArray):
        try:
            ds_med = ds_med.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # get upper n quantile (i.e., percentiles) of dry vege, moist
    ds_quants = ds_med.quantile(q=q_upper, skipna=True)
    ds_quants = xr.where(ds_med > ds_quants, True, False)

    # get num times
    num_times = len(ds['time'])

    # get linear ortho poly coeffs, sum squares, constant, reshape 1d to 3d
    coeffs, ss, const = tools.get_linear_orpol_contrasts(num_times)
    coeffs = np.reshape(coeffs, (ds['time'].size, 1, 1)) # todo dynamic?

    # calculate dry linear slope, mask to greenest/moistest
    ds_poly = abs((ds * coeffs).sum('time') / ss * const)
    ds_poly = ds_poly.where(ds_quants)

    # get lowest stability areas in stable, most veg, moist
    ds_targets = xr.where(ds_poly < ds_poly.quantile(q=q_lower, skipna=True), True, False)

    # determine if targets exist based on xr type
    if isinstance(ds_targets, xr.Dataset):
        da_all_nans = ds_targets.to_array().isnull().all()
    elif isinstance(ds_targets, xr.DataArray):
        da_all_nans = ds_targets.isnull().all()
        
    # check if targets exist
    if da_all_nans:
        raise ValueError('No invariant targets were created. Increase lower quantile.')
            
    # notify and return
    print('Created invariant target sites successfully.')

    if was_da:
        ds_targets = ds_targets.to_array()


    return ds_targets


def standardise_to_dry_targets(ds, dry_month=None, q_upper=0.99, q_lower=0.05, inplace=True):
    """
    Takes a xarray dataset/array and calculates invariant targets. Invariant 
    targets are pixels that are greenest/moistest across all time + most stable 
    across all time. Greenest and moistest values use upper percentile to detect 
    them, where as stable pixels generated via orthogonal polynomial coefficients 
    and quantile closest to 0, or lowest percentile. These sites are then used to
    standardise all images to each other. Images are finally rescaled 0 to 1 via
    a fuzzy increasing sigmoidal.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. Standardisation is done to dry season values.
    q_upper : float
        Set the upper percentile of vegetation/moisture values. We
        need the highest values to standardise to, but don't want to
        just take max. Default is 0.99 and typically produces optimal
        results.
    q_lower : float
        Set the lowest percentile of stability values. We need to find
        the most 'stable' pixels across time to ensure standardisation
        works. Default is 0.05 and typically produces optimal results.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Standardising data using invariant targets.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check q_lower < q_upper
    if q_upper <= q_lower:
        raise ValueError('Upper quantile value must be larger than lower quantile.')
    # check q_value 0-1
    elif q_upper < 0 or q_upper > 1:
        raise ValueError('Upper quantile value must be between 0 and 1.')
    # check q_value 0-1
    elif q_lower < 0 or q_lower > 1:
        raise ValueError('Lower quantile value must be between 0 and 1.')
        
    # check wet, dry month if none given
    if dry_month is None:
        raise ValueError('Must provide at least one dry month.')   
        
    # check wet dry list, convert if not
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
        
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
        
    # get attributes - we lose them 
    attrs = ds.attrs
        
    # split into dry and get all time dry season median
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)
    ds_dry_med = ds_dry.median('time')

    # generate invariant target sites
    ds_targets = calc_ivts(ds=ds_dry, 
                           ds_med=ds_dry_med, 
                           q_upper=q_upper, 
                           q_lower=q_lower)

    # notify
    print('Standardising to invariant targets, rescaling via fuzzy sigmoidal.')
    
    # get low, high inflection point via hardcoded percentile
    li = ds.median('time').quantile(q=0.001, skipna=True)
    hi = ds.where(ds_targets).quantile(dim=['x', 'y'], q=0.99, skipna=True)
    
    # create masks for values outside requested range. inefficient...
    mask_a = xr.where(ds > li, True, False)
    mask_b = xr.where(ds < hi, True, False)
        
    # do inc sigmoidal
    ds = np.square(np.cos((1 - ((ds - li) / (hi - li))) * (np.pi / 2)))
    
    # replace out of range values
    ds = ds.where(mask_a, 0)
    ds = ds.where(mask_b, 1)
    
    # drop quantile tag the method adds, if exists
    ds = ds.drop('quantile', errors='ignore')
    
    # add attributes back on
    ds.attrs.update(attrs)
    
    # notify and return
    print('Standardised using invariant targets successfully.')

    if was_da:
        ds = ds.to_array()

    return ds


def standardise_to_targets(ds, q_upper=0.99, q_lower=0.05, inplace=True):
    """
    Takes a xarray dataset/array and calculates invariant targets. Invariant 
    targets are pixels that are greenest/moistest across all time + most stable 
    across all time. Greenest and moistest values use upper percentile to detect 
    them, where as stable pixels generated via orthogonal polynomial coefficients 
    and quantile closest to 0, or lowest percentile. These sites are then used to
    standardise all images to each other. Images are finally rescaled 0 to 1 via
    a fuzzy increasing sigmoidal.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    q_upper : float
        Set the upper percentile of vegetation/moisture values. We
        need the highest values to standardise to, but don't want to
        just take max. Default is 0.99 and typically produces optimal
        results.
    q_lower : float
        Set the lowest percentile of stability values. We need to find
        the most 'stable' pixels across time to ensure standardisation
        works. Default is 0.05 and typically produces optimal results.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Standardising data using invariant targets.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check q_lower < q_upper
    if q_upper <= q_lower:
        raise ValueError('Upper quantile value must be larger than lower quantile.')
    # check q_value 0-1
    elif q_upper < 0 or q_upper > 1:
        raise ValueError('Upper quantile value must be between 0 and 1.')
    # check q_value 0-1
    elif q_lower < 0 or q_lower > 1:
        raise ValueError('Lower quantile value must be between 0 and 1.')
        
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
        
    # get attributes - we lose them 
    attrs = ds.attrs
        
    # get all time median
    ds_med = ds.median('time')

    # generate invariant target sites
    ds_targets = calc_ivts(ds=ds, 
                           ds_med=ds_med, 
                           q_upper=q_upper, 
                           q_lower=q_lower)

    # notify
    print('Standardising to invariant targets, rescaling via fuzzy sigmoidal.')

    # get low, high inflection point via hardcoded percentile
    li = ds.median('time').quantile(q=0.001, skipna=True)
    hi = ds.where(ds_targets).quantile(dim=['x', 'y'], q=0.99, skipna=True)
    
    # create masks for values outside requested range. inefficient...
    mask_a = xr.where(ds > li, True, False)
    mask_b = xr.where(ds < hi, True, False)
    
    # do inc sigmoidal
    ds = np.square(np.cos((1 - ((ds - li) / (hi - li))) * (np.pi / 2)))
    
    # replace out of range values
    ds = ds.where(mask_a, 0)
    ds = ds.where(mask_b, 1)
    
    # drop quantile tag the method adds, if exists
    ds = ds.drop('quantile', errors='ignore')
    
    # add attributes back on
    ds.attrs.update(attrs)
    
    # notify and return
    print('Standardised using invariant targets successfully.')

    if was_da:
        ds = ds.to_array()

    return ds


def calc_seasonal_similarity(ds, wet_month=None, dry_month=None, q_mask=0.9, inplace=True):
    """
    Takes a xarray dataset/array and a list of wet, dry months which 
    to generate similarity between wet, dry seasons per year. Lower
    similarity values are more unchanging pixels between seasons. The 
    q mask accepts a value between 0 to 1 in which to select only pixels
    that are high greenness. The default is recommended.


    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov. 
    q_mask : float
        A float between 0 to 1. Set the percentile in which to 
        mask high vegetation from the rest. Optional - keep as None
        to not mask similarity. Default recommended.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Calculating seasonal similarity.')

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')
        
    # check q_mask 0-1
    if q_mask < 0 or q_mask > 1:
        raise ValueError('Mask quantile value must be between 0 and 1.')

    # check wet, dry month if none given
    if wet_month is None or dry_month is None:
        raise ValueError('Must provide at least one wet and dry month.')    

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # take attrs
    attrs = ds.attrs
    
    # generate similarity (i.e. diff between wet and dry, per year)
    similarity_list = []
    for year, idxs in ds.groupby('time.year').groups.items():
        da = ds.isel(time=idxs).copy(deep=True)
        
        # split into wet, dry
        da_wet = da.where(da['time.month'].isin(wet_months), drop=True)
        da_dry = da.where(da['time.month'].isin(dry_months), drop=True)

        # make dummy date for each year - easier to compare
        np_date = '{0}-01-01'.format(year)
        da_wet['time'] = np.array([np_date], dtype='datetime64')
        da_dry['time'] = np.array([np_date], dtype='datetime64')

        # calc similarity and add to list
        da_similarity = da_wet - da_dry
        similarity_list.append(da_similarity)

    # combine similarity back together
    ds_similarity = xr.concat(similarity_list, dim='time').sortby('time')

    # rescale from (-1 to 1) to (0 to 2)
    ds_similarity = ds_similarity + 1
    
    # notify
    print('Rescaling via increasing-decreasing sigmoidal.')

    # generate left and right sigmoidals
    left = np.square(np.cos((1 - ((ds_similarity - 0) / (1 - 0))) * (np.pi / 2))) 
    right = np.square(np.cos(((ds_similarity - 1) / (2 - 1)) * (np.pi / 2)))

    # set values to 0 depending on left or right side
    left = left.where(ds_similarity <= 1, 0.0)
    right = right.where(ds_similarity > 1, 0.0)

    # combine left and right
    ds_similarity = left + right

    # mask to high veg/moist, if requested
    if q_mask:
        
        # notify
        print('Masking similarity areas to higher vege and moist areas.')   

        # get mask where similarity greater than percentile
        mask = xr.where(ds > ds.quantile(dim=['x', 'y'], q=q_mask), True, False)

        # reduce seasons to a year, get max (true if both, true if one), rename time
        mask = mask.groupby('time.year').max('time')
        mask = mask.rename({'year': 'time'})

        # if lengths match, set mask times to match similairty, then mask
        if len(ds_similarity['time']) == len(mask['time']):
            mask['time'] = ds_similarity['time'].values
            ds_similarity = ds_similarity.where(mask, 0.0)
        
        else:
            print('Could not mask similarity areas. Returning full similarity.')

    # drop quantile tag the method adds, if exists
    ds_similarity = ds_similarity.drop('quantile', errors='ignore')
            
    # add attributes back on
    ds_similarity.attrs.update(attrs)
    
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Calculated seasonal similarity successfully.')
    return ds_similarity


def calc_likelihood(ds, ds_similarity, wet_month=None, dry_month=None):
    """
    Takes a xarray dataset/array of veg, moisture and another xr of
    similarity between wet, dry seasons. A list of wet, dry months also 
    required. For each year, a GDV likelihood map is generated using
    weights created from the AHP process.

    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with veg and moisture vars.
    ds_similarity : xarray dataset/array
        A dataset with x, y and time dims with similarity var.
    wet_month : int or list
        An int or a list representing the month(s) that represent
        the wet season months. Example [1, 2, 3] for Jan, Feb, 
        Mar. 
    dry_month : int or list
        An int or a list representing the month(s) that represent
        the dry season months. Example [9, 10, 11] for Sep, Oct, 
        Nov.

    Returns
    ----------
    ds : xarray dataset or array.
    """
        
    # notify
    print('Generating groundwater-dependent vegetation (GDV) model.')
    
    # we need a dataset, try and convert to array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years in dataset.')

    # check wet, dry month if none given
    if wet_month is None or dry_month is None:
        raise ValueError('Must provide at least one wet and dry month.')    

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
    
    # check if var names have veg and mst idx
    for var in list(ds.data_vars):
        if var not in ['veg_idx', 'mst_idx']:
            raise ValueError('Vegetation and/or moisture index variable missing.')

    # take attrs
    attrs = ds.attrs    

    # generate gdv likelihood using analytic hierarchy process (ahp) weights
    likelihood_list = []
    for year, idxs in ds.groupby('time.year').groups.items():

        # get wet, dry, similarity veg, moist for year
        da = ds.isel(time=idxs).copy(deep=True)
        da_sim = ds_similarity.where(ds_similarity['time.year'] == year, drop=True)

        # split each var for weighting
        da_wet_veg = da['veg_idx'].where(da['time.month'].isin(wet_months), drop=True)
        da_wet_mst = da['mst_idx'].where(da['time.month'].isin(wet_months), drop=True)
        da_dry_veg = da['veg_idx'].where(da['time.month'].isin(dry_months), drop=True)
        da_dry_mst = da['mst_idx'].where(da['time.month'].isin(dry_months), drop=True)
        da_sim_veg = da_sim['veg_idx']
        da_sim_mst = da_sim['mst_idx']

        # squeeze time dimension,d rop it too
        da_wet_veg = da_wet_veg.squeeze(drop=True)
        da_wet_mst = da_wet_mst.squeeze(drop=True)
        da_dry_veg = da_dry_veg.squeeze(drop=True)
        da_dry_mst = da_dry_mst.squeeze(drop=True)
        da_sim_veg = da_sim_veg.squeeze(drop=True)
        da_sim_mst = da_sim_mst.squeeze(drop=True)

        # apply weights. note: see old gdv tool for other options
        da_wet_veg = da_wet_veg * 0.112613224
        da_wet_mst = da_wet_mst * 0.054621642
        da_dry_veg = da_dry_veg * 0.462481346
        da_dry_mst = da_dry_mst * 0.184883442
        da_sim_veg = da_sim_veg * 0.157825868
        da_sim_mst = da_sim_mst * 0.027574478   

        # sum of weights
        da_like = (da_wet_veg + da_wet_mst + 
                   da_dry_veg + da_dry_mst + 
                   da_sim_veg + da_sim_mst)

        # expand dim to new time
        np_date = '{0}-01-01'.format(year)
        np_date = np.array([np_date], dtype='datetime64')
        da_like = da_like.expand_dims({'time': np_date})

        # convert to dataset
        da_like = da_like.to_dataset(name='like')

        # append to list
        likelihood_list.append(da_like)

    # combine similarity back together
    ds_likelihood = xr.concat(likelihood_list, dim='time').sortby('time')

    # add attributes back on
    ds_likelihood.attrs.update(attrs)
    
    # convert back to datarray
    if was_da:
        ds_likelihood = ds_likelihood.to_array()

    # notify and return
    print('Generated groundwater-dependent vegetation model successfully')
    return ds_likelihood


def threshold_xr_via_auc(ds, df, res_factor=3, if_nodata='any'):
    """
    Takes a xarray dataset/array of gdv likelihood values and thresholds them
    according to a pandas dataframe (df) of field occurrence points. Scipy
    roc curve and auc is generated to perform thresholding. Pandas dataframe
    must include absences along with presences or the roc curve cannot be
    performed.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    df : pandas dataframe
        A dataframe of field occurrences with x, y values and 
        presence, absence column.
    res_factors : int
        Controls the tolerance of occurence points intersection with
        nearest pixels. In other words, number of pixels that a occurrence
        point can be 'out'.
    if_nodata : str
        Whether to exclude a point from the auc threshold method if any
        or all values are nan. Default is any.
        
    Returns
    ----------
    ds_thresh : xarray dataset or array.
    """
    
    # imports check
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
    except:
        raise ImportError('Could not import sklearn.')

    # notify
    print('Thresholding dataset via occurrence records and AUC.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if pandas type, columns, actual field
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Occurrence records is not a pandas type.')
    elif 'x' not in df or 'y' not in df:
        raise ValueError('No x, y fields in occurrence records.')
    elif 'actual' not in df:
        raise ValueError('No actual field in occurrence records.')     
            
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')
        
    # check if res factor and if_nodata valid
    if not isinstance(res_factor, int) and res_factor < 1:
        raise TypeError('Resolution factor must be an integer of 1 or greater.')
    elif if_nodata not in ['any', 'all']:
        raise TypeError('If nodata policy must be either any or all.')
        
    # split ds into arrays depending on dims
    da_list = [ds]
    if 'time' in ds.dims:
        da_list = [ds.sel(time=dt) for dt in ds['time']]

    # loop each slice, threshold to auc
    thresh_list = []
    for da in da_list:

        # take a copy
        da = da.copy(deep=True)

        # intersect points with current da
        df_data = df[['x', 'y', 'actual']].copy()
        df_data = tools.intersect_records_with_xr(ds=da, 
                                                  df_records=df_data, 
                                                  extract=True,
                                                  res_factor=res_factor,
                                                  if_nodata=if_nodata)
        
        # remove no data
        df_data = tools.remove_nodata_records(df_data, nodata_value=ds.nodatavals)
        
        # check if dataframe has 1s and 0s only
        unq = df_data['actual'].unique()
        if not np.any(unq == 1) or not np.any(unq == 0):
            raise ValueError('Occurrence records do not contain 1s and/or 0s.')
        elif len(unq) != 2:
            raise ValueError('Occurrence records contain more than just 1s and/or 0s.')  

        # rename column, add column of actuals (1s)
        df_data = df_data.rename(columns={'like': 'predicted'})

        # get fpr, tpr, thresh, auc and optimal threshold
        fpr, tpr, thresholds = roc_curve(df_data['actual'], df_data['predicted'])
        auc = roc_auc_score(df_data['actual'], df_data['predicted'])
        cut_off = thresholds[np.argmax(tpr - fpr)]

        # threshold da to cutoff and append
        da = da.where(da > cut_off)
        thresh_list.append(da)

        # notify 
        if 'time' in ds.dims:
            print('AUC: {0} for time: {1}.'.format(round(auc, 3), da['time'].values))
        else:
            print('AUC: {0} for whole dataset.'.format(round(auc, 3)))
           
        # show (non-arcgis only, disabled)
        #print('- ' * 30)
        #plt.show()
        #print('- ' * 30)
        #print('')

    # concat array back together
    if len(thresh_list) > 1:
        ds_thresh = xr.concat(thresh_list, dim='time').sortby('time')
    else:
        ds_thresh = thresh_list[0]
    
    if was_da:
        ds_thresh = ds_thresh.to_array()

    # notify and return
    print('Thresholded dataset successfully.')
    return ds_thresh


def threshold_xr_via_std(ds, num_stdevs=3, inplace=True):
    """
    Takes a xarray dataset/array of gdv likelihood values and thresholds them
    according to a automated standard deviation approach. Users can set number 
    of standard devs to threshold by, with ~1 bringing more pixels back and ~3 
    bring less. An all-time median of likelihood is recommended as the ds input. 
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    num_stdevs : int
        The standard deviation factor in which to threshold likelihood
        values. Lower values return more, higher values return less.
    inplace : bool
        Copy new xarray into memory or modify inplace.
        
    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Thresholding dataset via standard deviation.')
    
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')  
    
    # check num_stdv > 0 and <= 10
    if num_stdevs < 0 or num_stdevs > 10:
        raise ValueError('Number of standard devs must be >= 0 and <= 10.')
        
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

    # calculate n stand devs and apply threshold
    ds_thresh = ds.mean(['x', 'y']) + (ds.std(['x', 'y']) * num_stdevs)
    ds = ds.where(ds > ds_thresh)
    
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Thresholded dataset successfully.')
    return ds


def threshold_likelihood(ds, df=None, num_stdevs=3, res_factor=3, if_nodata='any'):
    """
    Takes a xarray dataset/array of gdv likelihood values and thresholds them
    according to either a pandas dataframe (df) of field occurrence points or
    via a more automated standard deviation approach. Leave df as None to use
    standard deviation. Users can set number of standard devs to threshold by,
    with ~1 bringing more pixels back and ~3 bring less. An all-time median of
    likelihood is recommended as the ds input. Calls two functions 
    threshold_xr_via_std or threshold_xr_via_auc, depending.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    df : pandas dataframe
        A dataframe of field occurrences with x, y values and 
        presence, absence column.
    num_stdevs : int
        The standard deviation factor in which to threshold likelihood
        values. Lower values return more, higher values return less.
    res_factors : int
        Controls the tolerance of occurence points intersection with
        nearest pixels. In other words, number of pixels that a occurrence
        point can be 'out'.
    if_nodata : str
        Whether to exclude a point from the auc threshold method if any
        or all values are nan. Default is any.
        
    Returns
    ----------
    ds_thresh : xarray dataset or array.
    """

    # notify
    print('Thresholding groundwater-dependent vegeation likelihood.')

    # we need a dataset, try and convert to array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')
            
    # check xr type, dims, num time
    if not isinstance(ds, (xr.Dataset)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions in dataset.')  
        
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')
        
    # check if var names have veg and mst idx
    for var in list(ds.data_vars):
        if var not in ['like']:
            raise ValueError('Likelihood variable missing.')

    # check num_stdevs > 0 and < 10
    if num_stdevs < 0 or num_stdevs > 10:
        raise ValueError('Number of standard deviations must be >= 0 and <= 10.')

    # if records given, thresh via auc, else standard dev
    if df is not None:
        try:
            # attempt roc auc thresholding
            ds_thresh = threshold_xr_via_auc(ds=ds,
                                             df=df, 
                                             res_factor=res_factor, 
                                             if_nodata=if_nodata)
            
        except Exception as e:
            print('Could not threshold via occurrence records. Trying standard dev.')
            ds_thresh = threshold_xr_via_std(ds, num_stdevs=num_stdevs)

    else:
        # attempt roc standard dev thresholding
        ds_thresh = threshold_xr_via_std(ds, num_stdevs=num_stdevs)
    

    if was_da:
        ds_thresh = ds_thresh.to_array()

    # notify
    print('Thresholded likelihood succuessfully.')
    return ds_thresh


def __mk__(x, y, p, d, nd):
    """
    Helper function, should only be called by perform_mk_original()
    """

    result = None

    # check nans
    nans = np.isin(x, nd) | np.isnan(x)
    if np.all(nans):
        result = nd
    else:
        # remove nans
        x, y = x[~nans], y[~nans]

        # count finite values, abort if 3 or less
        num_fin = np.count_nonzero(x)
        if num_fin <= 3:
            result = nd
        else:
            # perform original mk
            tau, pvalue = kendalltau(x=x, y=y, nan_policy='omit')

            # if p given and its not sig, bail
            if p and pvalue >= p:
                result = nd
            
            # check direction
            elif d == 'both':
                result = tau
            elif d == 'inc' and tau > 0:
                result = tau
            elif d == 'dec' and tau < 0:
                result = tau
            else: 
                result = nd
    
    return result


def perform_mk_original(ds, pvalue=None, direction='both'):
    """
    Takes a xarray dataset/array of gdv likelihood values (thresholded or not)
    and performs a mann-kendall trend analysis on each pixel time series as 
    1d vector. Users can control whether singificant trends only are returned
    (pvalue) and whether the direction is inc or dec trend (direction). Leave
    pvalue empty for any trend and/or set direction to both for any direction.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    pvalue : float
        Significant of trend to return. Default is 0.05. Leave blank
        for any trend, significant or not.
    direction : str
        Direction of trend to return. If inc, only upward trends. If
        dec, only downward. Enter both for inc and dec returned.
        
    Returns
    ----------
    ds_mk : xarray dataset or array.
    """
    
    # imports check
    try:
        from scipy.stats import kendalltau
    except:
        raise ImportError('Could not import scipy.')
    
    # notify user
    print('Performing Mann-Kendall test (original).')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check xr type, dims
    if not isinstance(ds, (xr.Dataset)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif len(ds['time']) < 3:
        raise ValueError('More than 2 years required for analysis.')
    
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')
        
    # check if p is valid, if provided
    if pvalue and (pvalue < 0 or pvalue > 1):
        raise ValueError('P-value must be between 0 and 1.')
        
    # check distance value
    if direction not in ['inc', 'dec', 'both']:
        raise ValueError('Direction value must be inc, dec or both.')

    # generate mk
    ds_mk = xr.apply_ufunc(
        __mk__, ds,
        input_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={
            'y': ds['time'], 
            'p': pvalue, 
            'd': direction,
            'nd': ds.nodatavals
        })
    
    # rename like var if exists, else ignore
    try:
        ds_mk = ds_mk.rename({'like': 'tau'})
    except:
        pass
    
    if was_da:
        ds_mk = ds_mk.to_array()

    return ds_mk


def __ts__(y, x, a, nd):
    """
    Helper function, should only be called by perform_theilsen_slope
    """

    # check nans
    nans = np.isin(y, nd) | np.isnan(y)
    if np.all(nans):
        return nd

    # remove nans
    y, x = y[~nans], x[~nans]

    # count finite values, abort if 3 or less
    num_fin = np.count_nonzero(y)
    if num_fin <= 3:
        return nd
    
    # perform theil-sen
    medslope, medint, lo_slope, up_slope = theilslopes(y=y, x=x, alpha=a)
    
    # return
    return medslope


def perform_theilsen_slope(ds, alpha):
    """
    Takes a xarray dataset/array of gdv likelihood values (thresholded or not)
    and calculates theil-sen slopes on each pixel time series as 1d vector. 
    Users can control whether singificant slopes only are returned
    (alpha). Leave alpha empty for any slope.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    alpha : float
        Significance of slope to return. Default is 0.95. Leave blank
        for any trend, significant or not.
        
    Returns
    ----------
    ds_ts : xarray dataset or array.
    """
    
    #imports check
    try:
        from scipy.stats import theilslopes
    except:
        raise ImportError('Could not import scipy.')
    
    # notify user
    print('Performing Theil-Sen slope (original).')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # check if 3 or more times
    if len(ds['time']) < 3:
        raise ValueError('More than 2 years required for analysis.')
        
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')
        
    # check if p is valid, if provided
    if alpha and (alpha < 0 or alpha > 1):
        raise ValueError('Alpha must be between 0 and 1.')   

    # create ufunc to wrap mk in
    ds_ts = xr.apply_ufunc(
        __ts__, ds,
        input_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={
            'x': ds['time'], 
            'a': alpha, 
            'nd': ds.nodatavals
        })

    # rename like var if exists, else ignore
    try:
        ds_ts = ds_ts.rename({'like': 'theilsen'})
    except:
        pass
    
    if was_da:
        ds_ts = ds_ts.to_array()

    return ds_ts


def __cva__(ds_base, ds_comp, vege_var='tcg', soil_var='tcb', tmf=2):
    """
    Helper function, should only be called by perform_cva()
    """

    # check each dataset for issues
    for ds_temp in [ds_base, ds_comp]:

        # get vars in ds
        temp_vars = []
        if isinstance(ds_temp, xr.Dataset):
            temp_vars = list(ds_temp.data_vars)
        elif isinstance(ds_temp, xr.DataArray):
            temp_vars = list(ds_temp['variable'])

        # check if vege var and soil var given
        for v in [vege_var, soil_var]:
            if v not in temp_vars:
                raise ValueError('Vege and/or soil var name not in dataset.')

    # get difference between comp and base, calc magnitude
    ds_diff = ds_comp - ds_base
    ds_magnitude = xr.ufuncs.sqrt(xr.ufuncs.square(ds_diff['tcb']) + 
                                    xr.ufuncs.square(ds_diff['tcg']))

    # get threshold value and make mask where mag > threshold
    threshold = ds_magnitude.where(ds_magnitude > 0.0).mean() * tmf
    ds_target = xr.where(ds_magnitude > threshold, True, False)

    # calculate magnitude (as percentage) and angle values 
    ds_magnitude = ds_magnitude.where(ds_target) * 100
    ds_angle = xr.ufuncs.arctan2(ds_diff['tcb'].where(ds_target), 
                                    ds_diff['tcg'].where(ds_target)) / np.pi * 180

    # convert angles to 0-360 degrees
    ds_angle = xr.where(ds_angle < 0, ds_angle + 360, ds_angle)
    ds_angle = ds_angle.where(ds_target)

    # rename datasets and merge
    ds_angle = ds_angle.rename('angle')
    ds_magnitude = ds_magnitude.rename('magnitude')

    # merge arrays together into dataset
    ds_cva = xr.merge([ds_angle, ds_magnitude])

    # check and notify if no values returned
    if ds_angle.isnull().all() or ds_magnitude.isnull().all():
        print('Warning: No angles or magnitudes returned.')

    # return
    return ds_cva

def perform_cva(ds, base_times=None, comp_times=None, reduce_comp=False, 
                vege_var='tcg', soil_var='tcb', tmf=2):
    """
    Takes a xarray dataset/array of tasselled cap greeness and brightness
    values, as well as a a tuple of base years (e.g. 1990, 2010) that are
    averaged together to form a base image, and a comparison year tuple 
    (e.g. 2010, 2020). Each year in range comp_times is compared to the base_time
    average and a change vector analysis is performed. This results in a dataset
    of angles (change type) and magnitudes (intensity of change) for each 
    comparison to that base. users must tell func which vars are veg and soil,
    and provide a magnitude threshold factor (default 2) to remove noise.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    base_times : tuple
        A tuple of years (e.g. from, to) that will form the baseline
        vegetation/soil image. A range of years can be provided (e.g.
        1990, 2000), or if a single specific year wanted, provide that
        year twice e.g. (1990, 1990). A median is generated for these
        years.
    comp_times : tuple
        As above, except these years form the range of dates compared 
        to the baseline. For example, if (2010, 2020) was provided,
        each year would be compared to the baseline seperatly (unless
        reduce_comp set to True, in which a median is calculated). 
    veg_var : str
        Name of vegetation variable. Tasselled cap greeness is good.
    soil_var : str
        Name of soil variable. Tasselled cap brightness is good.
    tmf : int, float
        Threshold magnitude factor. Magnitude can be reduced to most 
        intense magnitude of change by providing a higher tmf value.
        Default is 2 in literature.
    
    Returns
    ----------
    ds_cva : xarray dataset or array.

    """
    
    # notify
    print('Performing CVA.')
    
    # check if xr types
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y dimension in dataset.')
    elif 'time' not in ds:
        raise ValueError('No time dimension in dataset.')   

    # check base, comp times
    if not isinstance(base_times, (list, tuple)):
        raise ValueError('Base times must be a tuple or list.')
    elif not isinstance(comp_times, (list, tuple)):
        raise ValueError('Comparison times must be a tuple or list.')
        
    # check if vege and soil vars provided
    if not vege_var or not soil_var:
        raise ValueError('Did not provide vege/soil variable name.')        

    # check if threshold factor provided
    if tmf < 0:
        raise ValueError('Threshold value must be provided and >= 0.')
    
    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')
    
    # generate year ranges
    base_range = np.arange(base_times[0], base_times[1] + 1, 1)
    comp_range = np.arange(comp_times[0], comp_times[1] + 1, 1)
       
    # check ranges for empty
    if len(base_range) == 0 or len(comp_range) == 0:
        raise ValueError('Requested times incompatible.')
    
    # subset into two datasets
    ds_base = ds.where(ds['time.year'].isin(base_range), drop=True)
    ds_comp = ds.where(ds['time.year'].isin(comp_range), drop=True)
    
    # get all-time median for base, comp if requested
    ds_base = ds_base.median('time', keep_attrs=True)
    if reduce_comp:
        ds_comp = ds_comp.median('time', keep_attrs=True)
            
    # prepare da lists depending on reduce
    ds_comp_list = []
    if not reduce_comp:
        for dt in ds_comp['time']:
            ds_comp_list.append(ds_comp.sel(time=dt))
    else:
        ds_comp_list.append(ds_comp)
                        
    # loop each ds in list and do cva
    ds_cva_list = []
    for i, da_comp in enumerate(ds_comp_list):
        
        # notify
        print('Doing CVA: {0}.'.format(i + 1))
        
        # do cva!
        ds_cva = __cva__(ds_base=ds_base, 
                     ds_comp=da_comp, 
                     vege_var=vege_var, 
                     soil_var=soil_var, 
                     tmf=tmf)
        
        # add to list
        ds_cva_list.append(ds_cva)
        
    # check if list, concat
    if ds_cva_list:
        ds_cva = xr.concat(ds_cva_list, dim='time')
    
    if was_da:
        ds_cva = ds_cva.to_array()

    # notify and return
    print('Performed CVA successfully.')
    return ds_cva


def isolate_cva_change(ds, angle_min=90, angle_max=180, inplace=True):
    """
    Takes a xarray dataset/array of cva angle/magnitude vars and 
    isolates specific angles. Different angle ranges represent different
    change types: 90 to 180 degrees represents soil increase, or veg
    decline. On the other hand, 270 to 360 degrees represents veg increase.
    
    Parameters
    ----------
    ds : xarray dataset/array
        A dataset with x, y and time dims with likelihood values.
    angle_min : float
        A value representing lowest angle, or start of angle range.
    angle_max : float
        A value representing highest angle, or end of angle range.
    
    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Isolating CVA angles from {0}-{1} degrees.'.format(angle_min, angle_max))
    
    # check if xr types
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')

    # we need a dataset, try and convert from array
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            was_da = True
            ds = ds.to_dataset(dim='variable')
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset.')

    # get vars
    if isinstance(ds, xr.Dataset):
        data_vars = list(ds.data_vars)
    elif isinstance(ds, xr.DataArray):
        data_vars = list(ds['variable'].values)
        
    # check if angle is a var
    if 'angle' not in data_vars:
        raise ValueError('No variable called angle provided.')
            
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
    
    # check if angles are supported
    if angle_min < 0 or angle_max < 0:
        raise ValueError('Angles cannot be less than 0.')
    elif angle_min > 360 or angle_max > 360:
        raise ValueError('Angles cannot be greater than 360.')
    elif angle_min >= angle_max:
        raise ValueError('Max angle cannot be greater than min angle.')
        
    # restrict ds to requested angles
    ds = ds.where((ds['angle'] > angle_min) & (ds['angle'] < angle_max))
    
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Isolated CVA angles successfully.')
    return ds


def detect_breaks(values, times, pen=3, fill_nan=True, quick_plot=False):
    """
    Takes a numpy of values and times and performs change point
    detection across 1d array. Uses ruptures library. Users can
    set penalty for pelt method. Lower penalty will return more
    change breaks, higher will be more strict. The idea with this
    function is to provide array of veg values and corresponding
    times in order at a single pixel. To be used in ArcGIS graphing.
    
    Parameters
    ----------
    values : numpy ndarray
        A numpy array 1d with time series values, i.e. veg.
    times : numpy ndarray
        A numpy array 1d with time series values, i.e. times.
    pen : int
        A value representing pelt change point method penalty.
        A Lower penalty will return more change breaks, higher 
        will be more strict. Default is 3.
    fill_nan : bool
        If true, any nan values will be filled for array. 
        Recommended, as change points will detected at nan
        values.
    quick_plot : bool
        If true, a quick plot will be shown to provide a
        quick idea of the time series values and where
        breaks are detected.
    
    Returns
    ----------
    brk_dates : numpy of times where break occured.
    """

    # import check
    try:
        import ruptures as rpt
    except:
        raise ImportError('Could not import ruptures.')

    # check if values, times numpy
    if not isinstance(values, np.ndarray):
        raise TypeError('Values must be numpy array.')
    elif not isinstance (times, np.ndarray):
        raise TypeError('Times must be numpy array.')

    # check penalty
    if pen <= 0:
        raise ValueError('Penalty must be greater than 0.')
        
    # if fill nan, fill nan!
    if fill_nan:
        nans, f = np.isnan(values), lambda z: z.nonzero()[0]
        values[nans]= np.interp(f(nans), f(~nans), values[~nans])

    # detect breaks using pelt
    try:
        brk_idxs = rpt.Pelt(model='rbf').fit(values).predict(pen=3)
    except:
        brk_idxs = np.array([])
        
    # if quick polt, quick plot!
    if quick_plot:
        rpt.display(values, brk_idxs)

    # rbt adds max index to array - remove
    if len(times) in brk_idxs:
        brk_idxs.remove(len(times))

    # if breaks, get date(s)
    brk_dates = np.array([])
    if len(brk_idxs) > 0:
        brk_dates = times[brk_idxs]

    # return
    return brk_dates