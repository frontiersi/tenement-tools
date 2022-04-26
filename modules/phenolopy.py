# phenolopy
'''
This script contains functions for calculating per-pixel phenology metrics (phenometrics)
on a timeseries of vegetation index values (e.g. NDVI) stored in a xarray DataArray. The
methodology is based on the TIMESAT 3.3 software. Some elements of Phenolopy were also
inspired by the great work of Chad Burton (chad.burton@ga.gov.au).

Links:
TIMESAT 3.3: http://web.nateko.lu.se/timesat/timesat.asp
GitHub: https://github.com/lewistrotter/phenolopy

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries todo fix 
import os, sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append('../../shared')
import tools

#from scipy.stats import zscore
from scipy import interpolate as sci_interp
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

def enforce_edge_dates(ds, years=None):
    """
    We need very accurate date start and ends for
    consistent interpolation and resampling, so
    this function checks if the 1st of jan for
    start year and 31st of dec for end year exist
    in the input xarray dataset. If they do not,
    the function will take the closest index to it
    and copy it (essentially a ffill and bfill). 
    If s_year or e_year is none, first and last
    time index year will be used.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional xr data type.
    years : int or list 
        A int of a specific year or a list of years.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        dummy 1st jan and 31st dec times (if needed).
    
    """

    # check dataset
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset is not xarray type.')
    elif 'time' not in ds:
        raise ValueError('No time dimension in dataset.')
        
    # prepare years
    if years is None or years == []:
        s_year = int(ds['time.year'].isel(time=0))
        e_year = int(ds['time.year'].isel(time=-1))
    elif years is int:
        s_year, e_year = years, years
    else: 
        s_year, e_year = years[0], years[-1]
        
    # check if years in dataset
    if s_year not in ds['time.year']:
        raise ValueError('Start year not in dataset.')
    elif e_year not in ds['time.year']:
        raise ValueError('End year not in dataset.')
        
    # prepare start date
    s_dates = ds['time'].where(ds['time.year'] == s_year, drop=True)
    s_date = s_dates.isel(time=0).dt.strftime('%Y-%m-%d')

    # if start year not 1st jan..
    if s_date != '{}-01-01'.format(s_year):

        # copy first start year date, replace date, concat, sort
        tmp = ds.sel(time=s_dates.isel(time=0)).copy()
        tmp['time'] = np.datetime64('{}-01-01'.format(s_year))
        ds = xr.concat([ds, tmp], dim='time').sortby('time')
        
    # likewise, prepare end date
    e_dates = ds['time'].where(ds['time.year'] == e_year, drop=True)
    e_date = e_dates.isel(time=-1).dt.strftime('%Y-%m-%d')

    # if end year not 31st dec..
    if e_date != '{}-12-31'.format(e_year):

        # copy last end year date, replace date, concat, sort
        tmp = ds.sel(time=e_dates.isel(time=-1)).copy()
        tmp['time'] = np.datetime64('{}-12-31'.format(e_year))
        ds = xr.concat([ds, tmp], dim='time').sortby('time')
    
    return ds


def remove_spikes(ds, user_factor=2, win_size=3):
    """
    Takes an xarray dataset containing vegetation index variable and removes 
    outliers within the timeseries on a per-pixel basis. The resulting dataset 
    contains the timeseries with outliers set to nan. Can work on datasets with
    or without existing nan values.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. 
        A higher factor value results in few outliers (i.e. only the biggest outliers). 
        Default factor is 2.
    win_size : int
        Controls the size of the rolling window. Increase to capture more times
        in the outlier assessment.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with all detected outliers in the
        veg_index variable set to nan.
    """
    
    # notify user
    print('Removing spike outliers.')
                
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')
    elif 'veg_idx' not in ds:
        raise ValueError('No veg_idx variable in dataset.')

    # check if user factor provided
    if user_factor <= 0:
        user_factor = 1

    # check win_size not less than 3 and odd num
    if win_size < 3:
        win_size == 3
    elif win_size % 2 == 0:
        win_size += 1

    # calc cutoff val per pixel i.e. stdv of pixel multiply by user-factor 
    cutoff = ds.std('time') * user_factor

    # calc rolling median for whole dataset
    ds_med = ds.rolling(time=win_size, center=True).median()        
        
    # calc abs diff of orig and med vals
    ds_dif = abs(ds - ds_med)
    
    # calc mask
    ds_mask = ds_dif > cutoff
    
    # shift vals left, right one time index, get mean and fmax per center
    l = ds.shift(time=1).where(ds_mask)
    r = ds.shift(time=-1).where(ds_mask)
    ds_mean = (l + r) / 2
    ds_fmax = xr.ufuncs.fmax(l, r)
    
    # flag only if mid val < mean of l, r - cutoff or mid val > max val + cutoff
    ds_spikes = xr.where((ds.where(ds_mask) < (ds_mean - cutoff)) | 
                         (ds.where(ds_mask) > (ds_fmax + cutoff)), True, False)    
    
    # set spikes to nan
    ds = ds.where(~ds_spikes)    

    # notify user and return
    print('Outlier removal successful.')
    return ds


def resample(ds, interval='1M'):
    """
    Takes an xarray dataset containing vegetation index variable and resamples
    to a new temporal resolution. The resulting dataset contains the new resampled 
    veg_idx variable.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    interval: str
        The new temporal interval which to resample the dataset to. Available
        intervals include 1W (weekly), 1SM (bi-month) and 1M (monthly).

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        newly resampled 'veg_index' variable.
    """
    
    # notify user
    print('Resampling dataset.')
    
    # check dataset
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')
    elif 'veg_idx' not in ds:
        raise ValueError('No veg_idx variable in dataset.')
        
    # check if interval supported
    if interval is None:
        raise ValueError('Did not provide a interval.')
    elif interval not in ['1W', '2W', 'SM', 'SMS', '1M', '1D']:
        raise ValueError('Interval is not supported.')
        
    # resample using median
    ds = ds.resample(time=interval).median('time')
    
    # notify and return
    return ds


def interpolate(ds):
    """
    Takes a xarray dataset and performs linear interpolation across
    all times in dataset. This is just a helper function. 

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y and time dims.

    Returns
    ----------
    ds : xarray dataset or array.
    """

    # notify
    print('Interpolating empty values in dataset.')

    # check xr type, dims, num time
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimensions in dataset.')
    elif 'veg_idx' not in ds:
        raise ValueError('No veg_idx variable in dataset.')
        
    # check chunks, ensure chunked -1 if so else error occurs
    if bool(ds.chunks):
        ds = ds.chunk({'time': -1})

    # interpolate all nan pixels linearly
    ds = ds.interpolate_na(dim='time', method='linear')

    # notify and return
    print('Interpolated empty values successfully.')
    return ds  


def drop_overshoot_dates(ds, min_dates=3):
    """
    Takes an xarray dataset containing datetime index (time) 
    and drops all data for any start and end year that has 
    <= the max dates set. This is required due to resampling
    frequencies often adding an extra date to the end of the
    dataset (e.g., 2020-12-30 might become 2021-01-04). We
    dont want these extra dates. Typically only run this
    after resampling has been performed, don't use on raw
    data.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array.
    min_dates : int
        The minimum number of dates in a year allowed.
        If there are <= to this number, the year will be
        dropped (only start and end years).

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with any 
        times removed that occured in non-dominant year (if exists).
    """    
    
    # notify user
    print('Removing overshoot date data.')
    
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in ds or 'y' not in ds or 'time' not in ds.dims:
        raise ValueError('No x, y and/or time dimension in dataset.')
        
    # check min dates is valid
    if min_dates is None or min_dates <= 0:
        raise ValueError('Minimum dates must be > 0.')
    
    # get the first and last years in dataset
    start_year = int(ds['time.year'].isel(time=0))
    end_year = int(ds['time.year'].isel(time=-1))
    
    # check start year <= min allowed dates, drop if violated
    start_count = len(ds['time'].where(ds['time.year'] == start_year, drop=True))
    if start_count <= min_dates:
        ds = ds.where(ds['time.year'] != start_year, drop=True)
        print('Removed overshoot start year: {}.'.format(start_year))
    
    # check end year <= min allowed dates, drop if violated
    end_count = len(ds['time'].where(ds['time.year'] == end_year, drop=True))
    if end_count <= min_dates:
        ds = ds.where(ds['time.year'] != end_year, drop=True)
        print('Removed overshoot end year: {}.'.format(end_year))
        
    # notify and return
    print('Removed overshoot years successfully.')
    return ds


def smooth(ds, var='veg_idx', window_length=3, polyorder=1):  
    """
    Takes an xarray dataset containing vegetation index variable 
    and smoothes timeseries on a per-pixel basis. The resulting 
    dataset contains a smoother timeseries. Recommended that no nan 
    values present in dataset. Uses savitsky-golay filtering method.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    var : string
        A variable name which will be smoothed. This var is replaced with smoothed
        values of same array size within this 
    window_length: int
        The length of the filter window (i.e., the number of coefficients). Value must 
        be a positive odd integer. The larger the window length, the smoother the dataset.
        Default value is 3 (as per TIMESAT).
    polyorder: int
        The order of the polynomial used to fit the samples. Must be a odd number (int) and
        less than window_length.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset as input into the function, with smoothed data in the
        veg_index variable.
    """
    
    # notify user
    print('Smoothing data.')
    
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')
        
    # check if var is in dataset
    if var is None:
        raise ValueError('No variable provided.')
    elif var not in ds:
        raise ValueError('No {} variable in dataset.'.format(var))
    
    # check params
    if window_length <= 0 :
        raise ValueError('Window_length must be greater than 0.')
    elif polyorder <= 0:
        raise ValueError('Polyorder must be greater than 0.')
    elif polyorder > window_length:
        raise ValueError('Polyorder is > than window length, must be less than.')
    
    # get axis of time dimension
    for idx, dim in enumerate(list(ds.dims)):
        if dim == 'time':
            axis = idx
    
    # set up kwargs
    kwargs = {
        'window_length': window_length, 
        'polyorder':polyorder, 
        'mode': 'nearest', 
        'axis': axis}

    # perform smoothing based on savitsky golay along time dim
    ds[var] = xr.apply_ufunc(savgol_filter, 
                             ds[var],
                             dask='parallelized',
                             kwargs=kwargs)

    # notify user and return
    print('Smoothing successful.')
    return ds


def subset_via_years(ds, years):
    """
    Takes an xarray Dataset and a year or list of 
    years and subsets the input dataset to these
    years. All other years in the dataset will be
    dropped.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array.
    
    years: int, list
        A year (int) or a list of years (list of ints)
        that represent the years user wants to subset
        the dataset down to. All other years outside
        of this list will be dropped.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with 
        all years in the years input.
    """
    
    # notify user
    print('Subsetting data.')
    
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')
        
    # check if var is in dataset
    if years is None or years == []:
        return ds
    elif not isinstance(years, (int, list)):
        raise TypeError('Years must be integer or list of integers.')
        
    # prepare years
    if isinstance(years, int):
        years = [years]

    # subset using years
    ds = ds.where(ds['time.year'].isin(years), drop=True) 
    
    # check if dataset is empty
    if len(ds['time']) == 0:
        raise ValueError('No data left after subset.')
        
    # notify and return
    print('Subset years successfully.')
    return ds


def subset_via_years_with_buffers(ds, years=None):
    """
    Takes an xarray Dataset and a year or list of 
    years and subsets the input dataset to these
    years. All other years in the dataset will be
    dropped. This method takes a month of data
    either side of the lowest and highest year in 
    the input years array (if exists) for interp 
    purposes.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array.
    
    years: int, list
        A year (int) or a list of years (list of ints)
        that represent the years user wants to subset
        the dataset down to. All other years outside
        of this list will be dropped.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with 
        all years in the years input.
    """
    
    # notify user
    print('Subsetting data with buffers.')
    
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')
        
    # check if var is in dataset
    if years is None or years == []:
        return ds
    elif not isinstance(years, (int, list)):
        raise TypeError('Years must be integer or list of integers.')
        
    # prepare years
    if isinstance(years, int):
        years = [years]
        
    # get range of dates with a month buffer either side   
    s_date = '{}-12-01'.format(years[0] - 1)
    e_date = '{}-01-31'.format(years[-1] + 1)

    # subset to only dates between buffer
    ds = ds.sel(time=slice(s_date, e_date))
    
    # check if dataset is empty
    if len(ds['time']) == 0:
        raise ValueError('No data left after subset with buffers.')
        
    # notify and return
    print('Subset years with buffers successfully.')
    return ds


def group(ds):
    """
    Takes an xarray dataset and groups it by strings
    of 'm-d'. This is best suited for when using resampled
    data using fortnight (SMS) interval, which always produces
    month-day 1-1, 1-15, 2-1, 2-15 etc. Don't use when
    resample done via week. If only one year provided,
    groupby will fail, so only groups if years > 1. 
    Either way, new labels from 1970-01-01 to 1970-12-31
    will be returned.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing 
        a vegetation index variable (i.e. 'veg_idx').

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        newly resampled 'veg_index' variable.
    """
    
    
    # notify
    print('Grouping dataset.')
    
    # check xr type, dims
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y and/or time dimension in dataset.')   
    
    # group only if > 1 year, else just re-label times
    if len(ds.groupby('time.year')) > 1:
    
        # convert time to labels for consistent time grouping
        ds['time'] = ds['time'].dt.strftime('%m-%d')
        ds = ds.groupby('time').median('time')

    # get the num of periods from grouping (will always be 24)
    periods = len(ds['time'])

    # check number of periods
    if periods == 0:
        raise ValueError('No date periods were generated.')

    # generate new date range, use arbitrary year
    dates = pd.date_range(start='1970-01-01', 
                          end='1970-12-31', 
                          periods=periods)

    # check if num periods same as dataset times
    if len(ds['time']) != periods:
        raise ValueError('Length of times and periods do not match.')

    # replace groupby values with times of same size
    ds['time'] = dates
    
    return ds


def interpolate_2d(arr, mask, method='nearest', fill_value=0):
    """
    Takes a 2d xarray data array and a mask of 
    pixels to interpolate into (True values will
    be interpolated). Method supports nearest,
    linear, cubic. Fill value is used to fill up
    data outside the convex hull of known pixel
    values. Default is 0, and it has no effect
    for the nearest method.
    
    Parameters
    ----------
    arr: numpy array 
        A numpy array or xarray data array of 
        values to be interpolated.
    mask: numpy array
        An array of booleans. True are pixels
        that will be targeted for interpolate,
        False will be ignored.
    method : str 
        Interpolation method. Nearest neighbour,
        cubic.
    fill_value: int 
        Used only during cubic, value to use fill 
        extrapolation areas.

    Returns
    -------
    arr : numpy array
        Interpolated version of input arr.
    """

    # get height and width of arr
    h, w = arr.shape[:2]
    
    # convert to equal grid 
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # mask window
    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = arr[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    # perform the interp on grid data
    interp_values = sci_interp.griddata(
        (known_x, known_y), 
         known_v, 
        (missing_x, missing_y),
         method=method, fill_value=fill_value
    )

    # generate copy and fill with interpolated values
    interp_image = arr.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def clean_edges(arr, doy, position):
    """
    Small helper to correct use alt peak/trough 
    if first or last doy is peak/trough.
    
    Parameters
    ----------
    arr: numpy array 
        A numpy array or xarray data array of 
        values to be cleaned.
    doy: numpy array
        An associated array of doy values for
        each arr value. Expected to have 365.
    position : str 
        Whether to remove peaks (max) or troughs
        (min).

    Returns
    -------
    arr : numpy array
        Cleaned version of input arr.
    """
    
    if position == 'max':
        
        # if first/last is peak, mask a month
        if doy[np.nanargmax(arr)] == 1:
            arr = np.where(doy <= 31, np.nan, arr)
        elif doy[np.nanargmax(arr)] == 365:
            arr = np.where(doy >= 334, np.nan, arr) 
            
    elif position == 'min':
        
        # if first/last is trough, mask a month
        if doy[np.nanargmin(arr)] == 1:
            arr = np.where(doy <= 31, np.nan, arr)
        elif doy[np.nanargmin(arr)] == 365:
            arr = np.where(doy >= 334, np.nan, arr)         
        
    return arr


def get_pos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate peak of season (pos) values, times (doys).
    Takes an xarray datatset with x, y, time and veg_idx
    variable. Set fill_nan to True for nan filling. If no
    nans present, will skip.
    """   
    
    def _pos(arr, doy, fix_edges):

        # mask extreme edges
        if fix_edges:
            arr = clean_edges(arr, doy, 'max')

        # get max value index
        idx = np.nanargmax(arr)

        # set values
        v, t = arr[idx], doy[idx]

        return v, t
    
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_pos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_vos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate valley of season (vos) values, times (doys).
    Takes an xarray datatset with x, y, time and veg_idx
    variable. Set fill_nan to True for nan filling. If no
    nans present, will skip.
    """   
    
    def _vos(arr, doy, fix_edges):
        
        # mask extreme edges
        if fix_edges:
            arr = clean_edges(arr, doy, 'min')

        # get min value index
        idx = np.nanargmin(arr)

        # set values
        v, t = arr[idx], doy[idx]

        return v, t   
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_vos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_mos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate middle of season (mos) values, (no times).
    Takes an xarray datatset with x, y, time and veg_idx
    variable. Set fill_nan to True for nan filling. If no
    nans present, will skip.
    """   
    
    def _mos(arr, doy, fix_edges):
            
        # mask extreme edges
        if fix_edges:
            arr = clean_edges(arr, doy, 'max')
        
        # get pos time (doy)
        pos_t = doy[np.nanargmax(arr)]

        # get slope values left, right of pos time (doy)
        s_l, s_r = arr[doy <= pos_t], arr[doy >= pos_t]

        # get upper 80% values on left, right slopes
        s_l = s_l[s_l >= np.nanpercentile(s_l, 80)]
        s_r = s_r[s_r >= np.nanpercentile(s_r, 80)]

        # get mean of left, right slope
        s_l, s_r = np.nanmean(s_l), np.nanmean(s_r)

        # get mean of both
        v = np.nanmean([s_l, s_r])

        return v
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_mos,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'],
                           kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_bse(ds, fix_edges=True, fill_nan=False):
    """
    Calculate base (bse) values, (no times). Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip.
    """   
    
    def _bse(arr, doy, fix_edges):
        
        # mask extreme edges
        if fix_edges:
            arr = clean_edges(arr, doy, 'min')
        
        # get pos time (doy)
        pos_t = doy[np.nanargmax(arr)]

        # get slope values left, right of pos time (doy)
        s_l, s_r = arr[doy <= pos_t], arr[doy >= pos_t]
        
        # get min value on left, right of slope
        s_l, s_r = np.nanmin(s_l), np.nanmin(s_r)

        # get mean of both
        v = np.nanmean([s_l, s_r])

        return v
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_bse,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'],
                           kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_aos(da_upper, da_lower, fill_nan=False):
    """
    Calculate amplitude of season (aos) values, (no times). 
    Takes arrays of pos/mos and vos/bse values. Upper could
    be either pos/vos or mos/bse, for example. Set fill_nan 
    to True for nan filling. If no nans present, will skip.
    """   
    
    def _aos(arr_upper, arr_lower):
        
        # get aos values
        v = np.abs(arr_upper - arr_lower)

        return v
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_aos,
                           da_upper,
                           da_lower,
                           dask='allowed',
                           output_dtypes=['float32', 'float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_sos_fos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate start of season (sos) values, times. Uses the 
    'first of slope' (fos) technique. Takes an xarray datatset 
    with x, y, time and veg_idx variable. Set fill_nan to True 
    for nan filling. If no nans present, will skip.    
    """
    
    def _sos_fos(arr, doy, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')
            
            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values left of pos and where slope +
            diff = np.where(np.gradient(arr) > 0, True, False)
            s_l = np.where((doy <= pos_t) & diff, arr, np.nan)

            # build non-nan start and end index ranges for slope +
            clusts = np.ma.clump_unmasked(np.ma.masked_invalid(s_l))

            # if cluster(s) exist, get last element of slope, else last element
            idx = 0
            if len(clusts) != 0:
                idx = clusts[0].start

            # set values
            v, t = arr[idx], doy[idx]
        
        except:
            v, t = arr_tmp[0], doy[0]

        return v, t  
    
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_sos_fos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_sos_mos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate start of season (sos) values, times. Uses the 
    'mean of slope' (mos) technique. Takes an xarray datatset 
    with x, y, time and veg_idx variable. Set fill_nan to True 
    for nan filling. If no nans present, will skip.    
    """
    
    def _sos_mos(arr, doy, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')
            
            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values left of pos and where slope +
            diff = np.where(np.gradient(arr) > 0, True, False)
            
            s_l = np.where((doy <= pos_t) & diff, arr, np.nan)
            
            # get mean of all values in slope + areas
            mean = np.nanmean(s_l)
            
            # calc abs distances of each slope + val and mean
            dists = np.abs(s_l - mean)
            
            # get the smallest idx
            idx = np.nanargmin(dists)
            
            # set values
            v = arr[idx]
            t = doy[idx]
        
        except:
            v, t = arr_tmp[0], doy[0]

        return v, t  
    
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_sos_mos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_sos_seaamp(ds, da_aos_v, da_bse_v, factor=0.75, fix_edges=True, fill_nan=False):
    """
    Calculate start of season (sos) values, times. Uses the 
    'seasonal amplitude' (seaamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires an xarray data array of aos and bse generated
    prior. Also requires a factor value, else sets to 
    default (0.75)
    """
    
    def _sos_seaamp(arr, doy, arr_aos_v, arr_bse_v, factor, fix_edges):   
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values left of pos and where slope +
            diff = np.where(np.gradient(arr) > 0, True, False)       
            s_l = np.where((doy <= pos_t) & diff, arr, np.nan)

            # add base to amplitude and threshold it via factor
            samp = float((arr_aos_v * factor) + arr_bse_v)

            # calc abs distances of each slope + val to seas amp
            dists = np.abs(s_l - samp)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]
            
        except:
            v, t = arr_tmp[0], doy[0]

        return v, t 


    # basic checks
    if factor is None or factor < 0:
        factor = 0.75
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_sos_seaamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              da_aos_v,
                              da_bse_v,
                              input_core_dims=[['time'], ['time'], [], []],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'factor': factor, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)


    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_sos_absamp(ds, abs_value=0.3, fix_edges=True, fill_nan=False):
    """
    Calculate start of season (sos) values, times. Uses the 
    'absolute amplitude' (absamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires a absolute value is set (in units of veg index, else 
    sets to default (0.3)
    """
    
    def _sos_absamp(arr, doy, abs_value, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values left of pos and where slope +
            diff = np.where(np.gradient(arr) > 0, True, False)       
            s_l = np.where((doy <= pos_t) & diff, arr, np.nan)

            # calc abs distances of each slope + val to absolute val
            dists = np.abs(s_l - abs_value)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]
            
        except:
            v, t = arr_tmp[0], doy[0]

        return v, t  


    # basic checks
    if abs_value is None:
        abs_value = 0.3
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_sos_absamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'abs_value': abs_value, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)

    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_sos_relamp(ds, factor=0.75, fix_edges=True, fill_nan=False):
    """
    Calculate start of season (sos) values, times. Uses the 
    'relative amplitude' (relamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires an xarray data array of aos and bse generated
    prior. Also requires a factor value, else sets to 
    default (0.75)
    """
    
    def _sos_relamp(arr, doy, factor, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()

            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values left of pos and where slope +
            diff = np.where(np.gradient(arr) > 0, True, False)       
            s_l = np.where((doy <= pos_t) & diff, arr, np.nan)

            # get relative amp via robust max and base (10% cut off)
            q_min = np.nanpercentile(arr, q=10)
            q_max = np.nanpercentile(arr, q=90)
            ramp = float(q_max - q_min)

            # add max to relative amp and threshold it via factor
            ramp = (ramp * factor) + q_min
                   
            # calc abs distances of each slope + val to relative value
            dists = np.abs(s_l - ramp)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]

        except:
            v, t = arr_tmp[0], doy[0]

        return v, t  


    # basic checks
    if factor is None or factor < 0:
        factor = 0.75
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_sos_relamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'factor': factor, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)

    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_eos_fos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate end of season (eos) values, times. Uses the 
    'first of slope' (fos) technique. Takes an xarray datatset 
    with x, y, time and veg_idx variable. Set fill_nan to True 
    for nan filling. If no nans present, will skip.    
    """
    
    def _eos_fos(arr, doy, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')
            
            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values right of pos and where slope -
            diff = np.where(np.gradient(arr) < 0, True, False)
            s_r = np.where((doy >= pos_t) & diff, arr, np.nan)

            # build non-nan start and end index ranges for slope -
            clusts = np.ma.clump_unmasked(np.ma.masked_invalid(s_r))

            # if cluster(s) exist, get last of slope
            idx = -1
            if len(clusts) != 0:
                idx = clusts[0].stop - 1
            
            # set values
            v, t = arr[idx], doy[idx]
        
        except:
            v, t = arr_tmp[-1], doy[-1]

        return v, t  
    
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_eos_fos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_eos_mos(ds, fix_edges=True, fill_nan=False):
    """
    Calculate end of season (eos) values, times. Uses the 
    'mean of slope' (mos) technique. Takes an xarray datatset 
    with x, y, time and veg_idx variable. Set fill_nan to True 
    for nan filling. If no nans present, will skip.    
    """
    
    def _eos_mos(arr, doy, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values right of pos and where slope -
            diff = np.where(np.gradient(arr) < 0, True, False)
            s_r = np.where((doy >= pos_t) & diff, arr, np.nan)
            
            # get mean of all values in slope - areas
            mean = np.nanmean(s_r)
            
            # calc abs distances of each slope - val and mean
            dists = np.abs(s_r - mean)
            
            # get the smallest idx
            idx = np.nanargmin(dists)
            
            # set values
            v = arr[idx]
            t = doy[idx]

        except:
            v, t = arr_tmp[-1], doy[-1]
            
        return v, t
    
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_eos_mos,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_eos_seaamp(ds, da_aos_v, da_bse_v, factor=0.75, fix_edges=True, fill_nan=False):
    """
    Calculate end of season (eos) values, times. Uses the 
    'seasonal amplitude' (seaamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires an xarray data array of aos and bse generated
    prior. Also requires a factor value, else sets to 
    default (0.75)
    """
    
    def _eos_seaamp(arr, doy, arr_aos_v, arr_bse_v, factor, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values right of pos and where slope -
            diff = np.where(np.gradient(arr) < 0, True, False)       
            s_r = np.where((doy >= pos_t) & diff, arr, np.nan)

            # add base to amplitude and threshold it via factor
            samp = float((arr_aos_v * factor) + arr_bse_v)

            # calc abs distances of each slope - val to seas amp
            dists = np.abs(s_r - samp)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]
            
        except:
            v, t = arr_tmp[-1], doy[-1]

        return v, t  


    # basic checks
    if factor is None or factor < 0:
        factor = 0.75
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_eos_seaamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              da_aos_v,
                              da_bse_v,
                              input_core_dims=[['time'], ['time'], [], []],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'factor': factor, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)

    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_eos_absamp(ds, abs_value=0.3, fix_edges=True, fill_nan=False):
    """
    Calculate end of season (eos) values, times. Uses the 
    'absolute amplitude' (absamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires a absolute value is set (in units of veg index, else 
    sets to default (0.3)
    """
    
    def _eos_absamp(arr, doy, abs_value, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values right of pos and where slope -
            diff = np.where(np.gradient(arr) < 0, True, False)       
            s_r = np.where((doy >= pos_t) & diff, arr, np.nan)

            # calc abs distances of each slope - val to absolute val
            dists = np.abs(s_r - abs_value)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]
            
        except:
            v, t = arr_tmp[-1], doy[-1]

        return v, t  

    
    # basic checks
    if abs_value is None:
        abs_value = 0.3
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_eos_absamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'abs_value': abs_value, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)

    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_eos_relamp(ds, factor=0.75, fix_edges=True, fill_nan=False):
    """
    Calculate end of season (eos) values, times. Uses the 
    'relative amplitude' (relamp) technique. Takes an xarray 
    datatset with x, y, time and veg_idx variable. Set fill_nan 
    to True for nan filling. If no nans present, will skip. 
    Requires an xarray data array of aos and bse generated
    prior. Also requires a factor value, else sets to 
    default (0.75)
    """
    
    def _eos_relamp(arr, doy, factor, fix_edges):
        try:
            # make copy of arr for error handling
            arr_tmp = arr.copy()
            
            # mask extreme edges
            if fix_edges:
                arr = clean_edges(arr, doy, 'max')

            # get pos time (doy)
            pos_t = doy[np.nanargmax(arr)]

            # get all values right of pos and where slope +
            diff = np.where(np.gradient(arr) < 0, True, False)       
            s_r = np.where((doy >= pos_t) & diff, arr, np.nan)

            # get relative amp via robust max and base (10% cut off)
            q_min = np.nanpercentile(arr, q=10)
            q_max = np.nanpercentile(arr, q=90)
            ramp = float(q_max - q_min)

            # add max to relative amp and threshold it via factor
            ramp = (ramp * factor) + q_min

            # calc abs distances of each slope + val to relative value
            dists = np.abs(s_r - ramp)

            # get the smallest idx
            idx = np.nanargmin(dists)

            # set values
            v, t = arr[idx], doy[idx]
            
        except:
            v, t = arr_tmp[-1], doy[-1]

        return v, t     


    # basic checks
    if factor is None or factor < 0:
        factor = 0.75
    
    try:
        # wrap above in apply function
        v, t = xr.apply_ufunc(_eos_relamp,
                              ds['veg_idx'],
                              ds['time.dayofyear'],
                              input_core_dims=[['time'], ['time']],
                              output_core_dims=[[], []],
                              vectorize=True,
                              dask='allowed',
                              output_dtypes=['float32', 'float32'],
                              kwargs={'factor': factor, 
                                      'fix_edges': fix_edges})
    except Exception as e:
        raise ValueError(e)

    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
            
            # interp time array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v, t


def get_los(da_sos_t, da_eos_t, fill_nan=False):
    """
    Calculate length of season (los) times, (no values). 
    Takes arrays of sos times and eos times (doys). Set 
    fill_nan to True for nan filling. If no nans present, 
    will skip.
    """   
    
    def _los(arr_sos_t, arr_eos_t):

        # get los values
        v = np.abs(arr_eos_t - arr_sos_t)    

        return v
    
    
    try:
        # wrap above in apply function
        t = xr.apply_ufunc(_los,
                           da_sos_t,
                           da_eos_t,
                           dask='allowed',
                           output_dtypes=['float32', 'float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if t.isnull().any() is True:
                mask = xr.where(t.isnull(), True, False)
                t = xr.apply_ufunc(interpolate_2d,
                                   t, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return t


def get_roi(da_pos_v, da_pos_t, da_sos_v, da_sos_t, fill_nan=False):
    """
    Calculate rate of increase (roi) values, (no times). 
    Takes arrays of pos values and times and sos values
    and times. Set fill_nan to True for nan filling. If no 
    nans present, will skip.
    """   
    
    def _roi(arr_pos_v, arr_pos_t, arr_sos_v, arr_sos_t):
        try:
            # get roi values   
            v = ((arr_pos_v - arr_sos_v) / 
                 (arr_pos_t - arr_sos_t) * 10000)

            # get absolute if negatives
            v = np.abs(v)
        
        except:
            v = np.nan
        
        return v
    
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_roi,
                           da_pos_v,
                           da_pos_t,
                           da_sos_v,
                           da_sos_t,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_rod(da_pos_v, da_pos_t, da_eos_v, da_eos_t, fill_nan=False):
    """
    Calculate rate of decrease (rod) values, (no times). 
    Takes arrays of pos values and times and eos values
    and times. Set fill_nan to True for nan filling. If no 
    nans present, will skip.
    """   
    
    def _rod(arr_pos_v, arr_pos_t, arr_eos_v, arr_eos_t):
        try:
            # get rod values   
            v = ((arr_eos_v - arr_pos_v) / 
                 (arr_eos_t - arr_pos_t) * 10000)
            
            # get absolute if negatives
            v = np.abs(v)
            
        except:
            v = np.nan
        
        return v
    
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_rod,
                           da_pos_v,
                           da_pos_t,
                           da_eos_v,
                           da_eos_t,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_sios(ds, da_sos_t, da_eos_t, da_bse_v, fill_nan=False):
    """
    Calculate short integral of season (sios) values, 
    (no times). Takes arrays of sos times and eos times 
    and bse values. Set fill_nan to True for nan filling. 
    If no nans present, will skip.
    """   
    
    def _sios(arr, doy, arr_sos_t, arr_eos_t, arr_bse_v):
        try:        
            # subtract the base from the original array
            arr = arr - arr_bse_v 
            
            # get all idxs between start and end of season
            season_idxs = np.where((doy >= arr_sos_t) & 
                                   (doy <= arr_eos_t))

            # subset arr and doy whereever in range
            arr = arr[season_idxs]
            doy = doy[season_idxs]
            
            # get short integral of all values via trapz   
            v = np.trapz(y=arr, x=doy)
            
            # rescale if negatives
            #if np.min(v) < 0:
                #v += np.abs(np.nanmin(v))
        
        except:
            v = np.nan
        
        return v
    
    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_sios,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           da_sos_t.rolling(x=3, y=3, center=True, min_periods=1).mean(),
                           da_eos_t.rolling(x=3, y=3, center=True, min_periods=1).mean(),
                           da_bse_v,
                           input_core_dims=[['time'], ['time'], [], [], []],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    # rescale if need be
    if v.min() < 0:
        v = v + np.abs(v.min())
    
    return v


def get_lios(ds, da_sos_t, da_eos_t, fill_nan=False):
    """
    Calculate long integral of season (lios) values, 
    (no times). Takes arrays of sos times and eos times. 
    Set fill_nan to True for nan filling. If no nans 
    present, will skip.
    """   
    
    def _lios(arr, doy, arr_sos_t, arr_eos_t):
        try:        
            # get all idxs between start and end of season
            season_idxs = np.where((doy >= arr_sos_t) &
                                   (doy <= arr_eos_t))

            # subset arr and doy whereever in range
            arr = arr[season_idxs]
            doy = doy[season_idxs]
            
            # get long integral of all values via trapz   
            v = np.trapz(y=arr, x=doy)
            
            # rescale if negatives
            #if np.min(v) < 0:
                #v += np.abs(np.nanmin(v))
        
        except:
            v = np.nan
        
        return v


    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_lios,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           da_sos_t.rolling(x=3, y=3, center=True, min_periods=1).mean(),
                           da_eos_t.rolling(x=3, y=3, center=True, min_periods=1).mean(),
                           input_core_dims=[['time'], ['time'], [], []],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    # rescale if need be
    if v.min() < 0:
        v = v + np.abs(v.min())
    
    return v


def get_siot(ds, da_bse_v, fill_nan=False):
    """
    Calculate short integral of total (siot) values, 
    (no times). Takes array of base values, (no times).
    Set fill_nan to True for nan filling. If no nans 
    present, will skip.
    """   
    
    def _siot(arr, doy, arr_bse_v):
        try:        
            # subtract the base from the orig array
            arr = arr - arr_bse_v
                    
            # get long integral of all values via trapz first
            v = np.trapz(y=arr, x=doy) 
            
            # rescale if negatives
            #if np.min(v) < 0:
                #v += np.abs(np.nanmin(v))

        except:
            v = np.nan
        
        return v


    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_siot,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           da_bse_v,
                           input_core_dims=[['time'], ['time'], []],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    # rescale if need be
    if v.min() < 0:
        v = v + np.abs(v.min())
    
    return v


def get_liot(ds, fill_nan=False):
    """
    Calculate long integral of total (liot) values, 
    (no times). Set fill_nan to True for nan filling. 
    If no nans present, will skip.
    """   
    
    def _liot(arr, doy):
        try:
            # get long integral of all values via trapz   
            v = np.trapz(y=arr, x=doy)
            
            # rescale if negatives
            #if np.min(v) < 0:
                #v += np.abs(np.nanmin(v))
        
        except:
            v = np.nan
        
        return v

    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_liot,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'])
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    # rescale if need be
    if v.min() < 0:
        v = v + np.abs(v.min())
    
    return v


def get_nos(ds, peak_spacing=12, fill_nan=False):
    """
    Calculate number of season (nos) values, (no times). 
    Set fill_nan to True for nan filling. If no nans present, 
    will skip. Set the peak spacing to divide the year of time 
    series data up into equal sizes. For example, 365 days divided 
    by 12 gives us month equal spaces.
    """   
    
    def _nos(arr, doy, peak_spacing):
        try:
            # calculate peak prominence, divide by months
            t = ((np.nanmax(arr) - np.nanmin(arr)) / 
                 (len(doy) / peak_spacing))

            # generate peak information
            peaks, _ = find_peaks(arr, prominence=t)

            # set value to 1 season, else > 1
            v = len(peaks) if len(peaks) > 0 else 1

        except:
            v = np.nan
        
        return v

    
    try:
        # wrap above in apply function
        v = xr.apply_ufunc(_nos,
                           ds['veg_idx'],
                           ds['time.dayofyear'],
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           vectorize=True,
                           dask='allowed',
                           output_dtypes=['float32'],
                           kwargs={'peak_spacing': peak_spacing})
    except Exception as e:
        raise ValueError(e)
        
    # fill nan pixels
    if fill_nan is True:
        try:
            # interp value array
            if v.isnull().any() is True:
                mask = xr.where(v.isnull(), True, False)
                v = xr.apply_ufunc(interpolate_2d,
                                   v, mask,
                                   dask='allowed',
                                   output_dtypes=['float32'],
                                   kwargs={'method': 'nearest'})
        except:
            print('Could not interpolate pixels, skipping.')
        
    return v


def get_phenometrics(ds, metrics=None, method='relative_amplitude', factor=0.75, abs_value=0.3, peak_spacing=12, fix_edges=True, fill_nan=False):
    """
    Generate phenometrics from an xr dataset. Users can choose 
    the seasonal derivation method for deriving season start and 
    end. Some of these methods require parameters factor, abs_value
    be set. Users can also set whether missing pixels are filled in.
    Much of this method is based on the TIMESAT 3.3 software.
    
    Parameters
    ----------
    ds : xarray Dataset
        A three-dimensional or multi-dimensional array containing 
        a variable of veg_index, x, y, and time dimensions. 
    metric : list or str
        A string or list of metrics to return. Leave empty to fetch 
        all.
    method: str
        A string indicating which start of season detection method 
        to use. The available options include:
        1. first_of_slope: lowest vege value of slope is sos.
        2. mean_of_slope: mean vege value of slope is sos.
        3. seasonal_amplitude: uses a % of the amplitude between 
           base and middle of season to find sos.
        4. absolute_amplitude: users defined absolute value in vege 
           index units.
        5. relative_amplitude: robust mean peak and base, and a 
           factor of that area.
    factor: float
        A float value between 0 and 1 which is used to increase or 
        decrease the amplitude threshold for the seasonal_amplitude 
        and relative_amplitude method. A factor closer to 0 results 
        in start of season nearer to min value, a factor closer to 1 
        results in start of season closer to peak of season.
    abs_value: float
        For absolute_value method only. Defines the absolute value in 
        units of the vege index to which sos and eos is defined. The 
        part of the vege slope that the absolute value hits will be the
        sos/eos value and time.
    peak_spacing : int
        When deriving number of seasons, set the number divisions to
        divide the time dimension by. Default is recommended.
    fix_edges : bool
        Some vegetation has a peak that may occur exactly on day of year
        1 or 365. This can cause issues when deriving start and end of 
        season. To skip these years and use the next closest peak or 
        trough value in the pixel time series, turn this to True. In 
        practical terms, this fixes some apparent noise, but slightly
        reduces accuracy of the result.
    fill_nan : bool
        Some methods may return nan values, set this to True to
        enforce a filling method to fill these pixels in using a
        nearest neighbourhood interpolation technique.
        
    Returns
    -------
    ds : xarray Dataset
        An xarray DataArray type with 1 or more phenometrics.
    """

    # notify user
    print('Preparing phenological metrics analysis...')

    # set approved metrics
    allowed_metrics = [
        'pos', 
        'vos', 
        'mos', 
        'bse',
        'aos', 
        'sos', 
        'eos', 
        'los', 
        'roi', 
        'rod', 
        'lios', 
        'sios', 
        'liot', 
        'siot',
        'nos']

    # check xarray
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Did not provide an xarray dataset.')
    elif 'time' not in ds or 'x' not in ds or 'y' not in ds:
        raise ValueError('No x, y or time dimensions in dataset.')
    elif len(ds.groupby('time.year')) != 1:
        raise ValueError('More than one year in dataset.')
    elif 'veg_idx' not in ds:
        raise ValueError('No veg_idx variable in dataset.')
    elif bool(ds.chunks):
        ds = ds.compute()

    # check metric type
    if metrics is None:
        metrics = allowed_metrics
    elif isinstance(metrics, str):
        metric = [metric]

    # check if somethign in list
    if len(metrics) == 0:
        raise ValueError('Did not request a metric.')

    # check if all metrics supported
    for metric in metrics:
        if metric not in allowed_metrics:
            raise ValueError('Metric {} not supported'.format(metric))

    allowed_methods = [
        'first_of_slope',
        'mean_of_slope',
        'seasonal_amplitude',
        'absolute_amplitude',
        'relative_amplitude']

    # check method is valid
    if method not in allowed_methods:
        raise ValueError('Method {} not supported'.format(method))

    # check parameters given with method
    if method == 'seasonal_amplitude' and factor is None:
        raise ValueError('Did not provide factor for seasonal amplitude.')
    elif method == 'absolute_amplitude' and abs_value is None:
        raise ValueError('Did not provide absolute value for absolute amplitude.')
    elif method == 'relative_amplitude' and factor is None:
        raise ValueError('Did not provide factor for relative amplitude.')

    # check peak spacing
    if peak_spacing is None:
        peak_spacing = 12

    # check fill nan
    if fill_nan is None:
        fill_nan = False

    # notify user
    print('Generating phenological metrics...')

    try:
        # generate both pos abd vos v, t
        print('Generating pos, vos...')
        da_pos_v, da_pos_t = get_pos(ds=ds, 
                                     fix_edges=fix_edges,
                                     fill_nan=fill_nan)
        da_vos_v, da_vos_t = get_vos(ds=ds, 
                                     fix_edges=fix_edges,
                                     fill_nan=fill_nan)

        # generate both mos and bse v (no t)
        print('Generating mos, bse...')
        da_mos_v = get_mos(ds=ds, 
                           fix_edges=fix_edges,
                           fill_nan=fill_nan)    
        da_bse_v = get_bse(ds=ds, 
                           fix_edges=fix_edges,
                           fill_nan=fill_nan)    

        # generate aos v (no t)
        print('Generating aos...')
        da_aos_v = get_aos(da_upper=da_mos_v, 
                           da_lower=da_bse_v, 
                           fill_nan=fill_nan)      

        # generate both sos and eos v, t
        print('Generating sos, eos...')
        if method == 'first_of_slope':
            da_sos_v, da_sos_t = get_sos_fos(ds=ds, 
                                             fix_edges=fix_edges,
                                             fill_nan=fill_nan)
            da_eos_v, da_eos_t = get_eos_fos(ds=ds, 
                                             fix_edges=fix_edges,
                                             fill_nan=fill_nan)

        elif method == 'mean_of_slope':
            da_sos_v, da_sos_t = get_sos_mos(ds=ds, 
                                             fix_edges=fix_edges,
                                             fill_nan=fill_nan)
            da_eos_v, da_eos_t = get_eos_mos(ds=ds, 
                                             fix_edges=fix_edges,
                                             fill_nan=fill_nan)

        elif method == 'seasonal_amplitude':
            da_sos_v, da_sos_t = get_sos_seaamp(ds=ds, 
                                                da_aos_v=da_aos_v,
                                                da_bse_v=da_bse_v,
                                                factor=factor,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)
            da_eos_v, da_eos_t = get_eos_seaamp(ds=ds, 
                                                da_aos_v=da_aos_v,
                                                da_bse_v=da_bse_v,
                                                factor=factor,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)

        elif method == 'absolute_amplitude':     
            da_sos_v, da_sos_t = get_sos_absamp(ds=ds, 
                                                abs_value=abs_value,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)
            da_eos_v, da_eos_t = get_eos_absamp(ds=ds, 
                                                abs_value=abs_value,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)

        elif method == 'relative_amplitude':     
            da_sos_v, da_sos_t = get_sos_relamp(ds=ds, 
                                                factor=factor,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)
            da_eos_v, da_eos_t = get_eos_relamp(ds=ds, 
                                                factor=factor,
                                                fix_edges=fix_edges,
                                                fill_nan=fill_nan)

        # generate los t (no v)
        print('Generating los...')
        da_los_t = get_los(da_sos_t=da_sos_t,
                           da_eos_t=da_eos_t,
                           fill_nan=fill_nan)

        # generate roi v (no t)
        print('Generating roi...')
        da_roi_v = get_roi(da_pos_v=da_pos_v, 
                           da_pos_t=da_pos_t, 
                           da_sos_v=da_sos_v, 
                           da_sos_t=da_sos_t, 
                           fill_nan=fill_nan)

        # generate rod v (no t)
        print('Generating rod...')
        da_rod_v = get_rod(da_pos_v=da_pos_v, 
                           da_pos_t=da_pos_t, 
                           da_eos_v=da_eos_v, 
                           da_eos_t=da_eos_t, 
                           fill_nan=fill_nan)

        # generate sios v (no t)
        print('Generating sios...')
        da_sios_v = get_sios(ds=ds,
                             da_sos_t=da_sos_t,
                             da_eos_t=da_eos_t,
                             da_bse_v=da_bse_v,
                             fill_nan=fill_nan)

        # generate lios v (no t)
        print('Generating lios...')
        da_lios_v = get_lios(ds=ds,
                             da_sos_t=da_sos_t,
                             da_eos_t=da_eos_t,
                             fill_nan=fill_nan)

        # generate siot v (no t)
        print('Generating siot...')
        da_siot_v = get_siot(ds=ds,
                             da_bse_v=da_bse_v,
                             fill_nan=fill_nan)

        # generate liot v (no t)
        print('Generating liot...')
        da_liot_v = get_liot(ds=ds, fill_nan=fill_nan)

        # generate nos v (no t)
        print('Generating nos...')
        da_nos_v = get_nos(ds=ds, 
                           peak_spacing=peak_spacing,
                           fill_nan=fill_nan)

    except Exception as e:
        raise ValueError(e)


        # notify user
        print('Success! Cleaning up...')

    try:
        # build dataset from var names and arrays
        ds = xr.merge([{
            'pos_values':  da_pos_v,
            'pos_times':   da_pos_t,
            'vos_values':  da_vos_v,
            'vos_times':   da_vos_t,
            'mos_values':  da_mos_v,
            'bse_values':  da_bse_v,
            'aos_values':  da_aos_v,
            'sos_values':  da_sos_v,
            'sos_times':   da_sos_t,
            'eos_values':  da_eos_v,
            'eos_times':   da_eos_t,
            'los_times':   da_los_t,
            'roi_values':  da_roi_v,
            'rod_values':  da_rod_v,
            'sios_values': da_sios_v,
            'lios_values': da_lios_v,
            'siot_values': da_siot_v,
            'liot_values': da_liot_v,
            'nos_values':  da_nos_v}])
        
        # update names of requested metrics
        metrics = (['{}_values'.format(m) for m in metrics] +
                   ['{}_times'.format(m) for m in metrics])

        # remove any metrics user did not request
        for var in ds:
            if var not in metrics:
                ds = ds.drop_vars(var)

    except Exception as e:
        raise ValueError(e)

    # notify user and return
    print('Generated phenological metrics successfully.')
    return ds