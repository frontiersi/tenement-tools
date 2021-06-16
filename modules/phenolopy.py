# phenolopy
'''
This script contains functions for calculating per-pixel phenology metrics (phenometrics)
on a timeseries of vegetation index values (e.g. NDVI) stored in a xarray DataArray. The
methodology is based on the TIMESAT 3.3 software. Some elements of Phenolopy were also
inspired by the great work of Chad Burton (chad.burton@ga.gov.au).

The script contains the following primary functions:
    1. calc_vege_index: calculate one of several vegetation indices;
    2. resample: resamples data to a defined time interval (e.g. bi-monthly);
    3. removal_outliers: detects and removes timeseries outliers;
    4. interpolate: fill in missing nan values;
    5. smooth: applies one of various smoothers to raw vegetation data;
    6. correct_upper_envelope: shift timeseries upwards closer to upper envelope;
    7. detect_seasons: count num of seasons (i.e. peaks) in timeseries;
    8. calc_phenometrics: generate phenological metrics on timeseries.

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


def remove_outliers(ds, method='median', user_factor=2, z_pval=0.05, inplace=True):
    """
    Takes an xarray dataset containing vegetation index variable and removes outliers within 
    the timeseries on a per-pixel basis. The resulting dataset contains the timeseries 
    with outliers set to nan. Can work on datasets with or without existing nan values. Note:
    Zscore method will compute memory.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    method: str
        The outlier detection method to apply to the dataset. The median method detects 
        outliers by calculating if values in pixel timeseries deviate more than a maximum 
        deviation (cutoff) from the median in a moving window (half window width = number 
        of values per year / 7) and it is lower than the mean value of its immediate neighbors 
        minus the cutoff or it is larger than the highest value of its immediate neighbor plus 
        The cutoff is the standard deviation of the entire time-series times a factor given by 
        the user. The second method, zscore, is similar but uses zscore to detect whether outlier
        is signficicantly (i.e. p-value) outside the population.
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. A higher factor 
        value results in few outliers (i.e. only the biggest outliers). Default factor is 2.
    z_pval: float
        The p-value for zscore method. A more significant p-value (i.e. 0.01) results in fewer
        outliers, a less significant p-value (i.e 0.1) results in more. Default is 0.05.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a all detected outliers in the
        veg_index variable set to nan.
    """
    
    # imports
    from scipy.stats import zscore
    
    # notify user
    print('Removing outliers via method: {}'.format(method))
            
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimension in dataset.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
        
    # check if user factor provided
    if user_factor <= 0:
        raise TypeError('User factor is less than 0, must be above 0.')
        
    # check if pval provided if method is zscore
    if method == 'zscore':
        if p_value == 0.10:
            z_value = 1.65
        elif p_value == 0.05:
            z_value = 1.96
        elif p_value == 0.01:
            z_value = 2.58
        else:
            print('P-value not supported. Setting to 0.01.')
            z_value = 2.58
            
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
        
    # remove outliers based on user selected method
    if method in ['median', 'zscore']:
        
        # calc cutoff val per pixel i.e. stdv of pixel multiply by user-factor 
        cutoffs = ds.std('time') * user_factor

        # generate outlier mask via median or zscore method
        if method == 'median':

            # calc mask of existing nan values (nan = True) in orig ds
            ds_mask = xr.where(ds.isnull(), True, False)

            # calc win size via num of dates in dataset
            win_size = int(len(ds['time']) / 7)
            win_size = int(win_size / int(len(ds.resample(time='1Y'))))

            if win_size < 3:
                win_size = 3
                print('Generated roll window size less than 3, setting to default (3).')
            elif win_size % 2 == 0:
                win_size = win_size + 1
                print('Generated roll window size is an even number, added 1 to make it odd ({0}).'.format(win_size))
            else:
                print('Generated roll window size is: {0}'.format(win_size))

            # calc rolling median for whole dataset
            ds_med = ds.rolling(time=win_size, center=True, keep_attrs=True).median()
            
            # calc nan mask of start/end nans from roll, replace them with orig vals
            med_mask = xr.where(ds_med.isnull(), True, False)
            med_mask = xr.where(ds_mask != med_mask, True, False)
            ds_med = xr.where(med_mask, ds, ds_med)

            # calc abs diff between orig ds and med ds vals at each pixel
            ds_diffs = abs(ds - ds_med)

            # calc mask of outliers (outlier = True) where absolute diffs exceed cutoff
            outlier_mask = xr.where(ds_diffs > cutoffs, True, False)

        elif method == 'zscore':

            # generate critical val from user provided p-value
            if z_pval == 0.01:
                crit_val = 2.3263
            elif z_pval == 0.05:
                crit_val = 1.6449
            elif z_pval == 0.1:
                crit_val = 1.2816
            else:
                raise ValueError('Zscore p-value not supported. Please use 0.1, 0.05 or 0.01.')

            # calc zscore, ignore nans in timeseries vectors
            zscores = ds.apply(zscore, nan_policy='omit', axis=0)

            # calc mask of outliers (outlier = True) where zscore exceeds critical value
            outlier_mask = xr.where(abs(zscores) > crit_val, True, False)

        # shift values left and right one time index and combine, get mean and max for each window
        lefts = ds.shift(time=1).where(outlier_mask)
        rights = ds.shift(time=-1).where(outlier_mask)
        nbr_means = (lefts + rights) / 2
        nbr_maxs = xr.ufuncs.fmax(lefts, rights)

        # keep nan only if middle val < mean of neighbours - cutoff or middle val > max val + cutoffs
        outlier_mask = xr.where((ds.where(outlier_mask) < (nbr_means - cutoffs)) | 
                                (ds.where(outlier_mask) > (nbr_maxs + cutoffs)), True, False)

        # flag outliers as nan in original da
        #ds = xr.where(outlier_mask, np.nan, ds)
        ds = ds.where(~outlier_mask)
        
    else:
        raise ValueError('Provided method not supported. Please use median or zscore.')
        
    # notify user and return
    print('Outlier removal successful.')
    return ds


def remove_overshoot_times(ds, max_times=3):
    """
    Takes an xarray dataset containing datetime index (time) and removes any 
    times that are the non-dominant year This function exists due to resampling 
    often adding an datetime at the end of the dataset that often stretches into 
    the next year.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array.
    max_times : int
        The maximum number of times in an overshoot before the
        function will abort. For example, if 3 times are in the
        non-dominant year and the max_times is set to 2, no times
        will be removed.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with any 
        times removed that occured in non-dominant year (if exists).
    """
    
    # imports
    from collections import Counter
    
    # notify user
    print('Removing times that occur in overshoot years.')

    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')

    # get unique years in dataset as dict pairs
    counts = Counter(list(ds['time.year'].values))
    unq_years = np.array(list(counts.keys()))
    unq_counts = np.array(list(counts.values()))
    
    if len(unq_years) > 1:
        print('Detected 2 or more years in dataset. Removing overshoot times.')
        
        # get indexes of counts sorted lowest to highest
        sort_count_idxs = unq_counts.argsort()
        
        # loop each and see if under max num times allowed
        for idx in sort_count_idxs:
            if unq_counts[idx] <= 3:
                year, num = unq_years[idx], unq_counts[idx]
                 
                # notify
                print('Dropped {} times for year {}.'.format(num, year))
                
                # remove all times that are not dominant year
                ds = ds.where(ds['time.year'] != year, drop=True)
                ds = ds.sortby('time')
                
    else:
        print('Only 1 year detected. No data removed.')

    # notify and return
    print('Removed times that occur in overshoot years successfully.')
    return ds


def conform_edge_dates(ds):
    """
    Takes an xarray dataset or array and checks if first and last dates in
    dataset are jan 1st and december 31st, respectively. If not, function 
    will create dummy scenes with a correct dates. It is essentially a bfill 
    and ffill with new dates.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional xr data type.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        dummy 1st jan and 31st dec times if needed.
    """
    
    # notify user
    print('Conforming edge dates.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    
    # get first, last datetime object
    f_dt, l_dt = ds['time'].isel(time=0), ds['time'].isel(time=-1)

    # convert to pandas timestamp
    f_dt = pd.Timestamp(f_dt.values).to_pydatetime()
    l_dt = pd.Timestamp(l_dt.values).to_pydatetime()

    # copy and update first time if not 1st day
    if f_dt.day != 1:
        print('First date was not Jan 1st. Prepending dummy.')
        f_da = ds.isel(time=0).copy(deep=True)
        f_dt = f_dt.replace(month=1, day=1)
        f_da['time'] = np.datetime64(f_dt)
        ds = xr.concat([f_da, ds], dim='time')

    # do the same for last time
    if l_dt.day != 31:
        print('Last date was not Dec 31st. Appending dummy.')
        l_da = ds.isel(time=-1).copy(deep=True)
        l_dt = l_dt.replace(month=12, day=31)
        l_da['time'] = np.datetime64(l_dt)
        ds = xr.concat([ds, l_da], dim='time')
        
    # notify and return
    print('Conformed edge dates successfully.')
    return ds


def resample(ds, interval='1M', inplace=True):
    """
    Takes an xarray dataset containing vegetation index variable and resamples
    to a new temporal resolution. The available time intervals are 1W (weekly),
    SM (bi-monthly) and 1M (monthly) resample intervals. The resulting dataset
    contains the new resampled veg_idx variable.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    interval: str
        The new temporal interval which to resample the dataset to. Available
        intervals include 1W (weekly), 1SM (bi-month) and 1M (monthly).
    inplace : bool
        Copy new xarray into memory or modify inplace.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        newly resampled 'veg_index' variable.
    """
    
    # notify user
    print('Resampling dataset.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x, y dimension in dataset.')
        
    # check if at least one year of data
    if len(ds.groupby('time.year').groups) < 1:
        raise ValueError('Need at least one year in dataset.')
        
    # get vars in ds
    temp_vars = []
    if isinstance(ds, xr.Dataset):
        temp_vars = list(ds.data_vars)
    elif isinstance(ds, xr.DataArray):
        temp_vars = list(ds['variable'])

    # check if vege var and soil var given
    if 'veg_idx' not in temp_vars:
        raise ValueError('Vege var name not in dataset.')
        
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
        
    # resample based on user selected interval and reducer
    if interval in ['1W', '1SM', '1M']:
        ds = ds.resample(time=interval).median('time', keep_attrs=True)
    else:
        raise ValueError('Provided resample interval not supported.')
                            
    # notify user and return
    print('Resampled dataset successful.\n')
    return ds


def group(ds, interval='month', inplace=True):
    """
    Takes an xarray dataset containing a vegetation index variable, groups and 
    reduces values based on a specified temporal group. The available group 
    options are by month or week only. The resulting dataset contains the new 
    grouped veg_index variable as a single year of weeks or months.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_idx').
    interval: str
        The groups which to reduce the dataset to. Available intervals only
        include month and week at this stage.
    inplace : bool
        Copy new xarray into memory or modify inplace.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        newly resampled 'veg_index' variable.
    """
    
    # notify user
    print('Grouping dataset.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x, y dimension in dataset.')
        
    # check if at least one year of data
    if len(ds.groupby('time.year').groups) < 1:
        raise ValueError('Need at least one year in dataset.')
        
    # get vars in ds
    temp_vars = []
    if isinstance(ds, xr.Dataset):
        temp_vars = list(ds.data_vars)
    elif isinstance(ds, xr.DataArray):
        temp_vars = list(ds['variable'])

    # check if vege var and soil var given
    if 'veg_idx' not in temp_vars:
        raise ValueError('Vege var name not in dataset.')
        
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
        
    # get all years in dataset, choose middle year in array for labels
    years = np.array([year for year in ds.groupby('time.year').groups])
    year = np.take(years, years.size // 2)
    
    # notify user
    print('Selecting year: {0} to re-label times after grouping.'.format(year))
          
    # group based on user selected interval and reducer
    if interval in ['week', 'month']:
        ds = ds.groupby('time' + '.' + interval).median('time', keep_attrs=True)
        ds = ds.rename({interval: 'time'})
    else:
        raise ValueError('Provided interval not supported.')
        
    # correct time index following group by
    if interval == 'month':
        times = [datetime(year, int(dt), 1) for dt in ds['time']]
    elif interval == 'week':
        times = []
        for dt in ds['time']:
            dt_string = '{} {} {}'.format(year, int(dt), 1)
            times.append(datetime.strptime(dt_string, '%Y %W %w'))
    else:
        raise ValueError('Interval was not found in dataset dimension. Aborting.')
        
    # check if someting was returned
    if not times:
        raise ValueError('No times returned.')
        
    # append array of dts to dataset
    ds['time'] = [np.datetime64(dt) for dt in times]
            
    # notify user and return
    print('Grouped dataset successful.')
    return ds

    
def interpolate(ds, method='full', inplace=True):
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
        
    # check if method is valid
    if method not in ['full', 'half']:
        raise ValueError('Method must be full or half.')
        
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
    
    # interpolate for ds
    ds = tools.perform_interp(ds=ds, method=method)

    # notify and return
    print('Interpolated empty values successfully.')
    return ds   


def smooth(ds, method='savitsky', window_length=3, polyorder=1, sigma=1):  
    """
    Takes an xarray dataset containing vegetation index variable and smoothes timeseries
    timeseries on a per-pixel basis. The resulting dataset contains a smoother timeseries. 
    Recommended that no nan values present in dataset.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    method: str
        The smoothing algorithm to apply to the dataset. The savitsky method uses the robust
        savitsky-golay smooting technique, as per TIMESAT. Symmetrical gaussian applies a simple 
        symmetrical gaussian. Default is savitsky.
    window_length: int
        The length of the filter window (i.e., the number of coefficients). Value must 
        be a positive odd integer. The larger the window length, the smoother the dataset.
        Default value is 3 (as per TIMESAT).
    polyorder: int
        The order of the polynomial used to fit the samples. Must be a odd number (int) and
        less than window_length.
    sigma: int
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter 
        must be provided as a single number between 1-9.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset as input into the function, with smoothed data in the
        veg_index variable.
    """
    
    # imports
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter
    
    # notify user
    print('Smoothing data via method: {0}.'.format(method))
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x, y dimension in dataset.')
        
    # check params
    if window_length <= 0 :
        raise ValueError('Window_length is <= 0. Must be greater than 0.')
    elif polyorder <= 0:
        raise ValueError('Polyorder is <= 0. Must be greater than 0.')
    elif polyorder > window_length:
        raise ValueError('Polyorder is > than window length. Must be smaller value.')
    elif sigma < 1 or sigma > 9:
        raise ValueError('Sigma is < 1 or > 9. Must be between 1 - 9.')
        
    # perform smoothing based on user selected method     
    if method in ['savitsky', 'symm_gaussian']:
        if method == 'savitsky':
            
            # create savitsky smoother func
            def smoother(da, window_length, polyorder):
                return da.apply(savgol_filter, window_length=window_length, polyorder=polyorder)
            
            # create kwargs dict
            kwargs = {'window_length': window_length, 
                      'polyorder': polyorder}

        elif method == 'symm_gaussian':
            
            # create gaussian smoother func
            def smoother(da, sigma):
                return da.apply(gaussian_filter, sigma=sigma)
            
            # create kwargs dict
            kwargs = {'sigma': sigma}
               
        # map func to dask chunks
        #temp = xr.full_like(ds, fill_value=np.nan)
        ds = xr.map_blocks(smoother, ds, template=ds, kwargs=kwargs)
        
    else:
        raise ValueError('Provided method not supported. Please use savtisky.')
        
    # notify user and return
    print('Smoothing successful.\n')
    return ds


def calc_num_seasons(ds):
    """
    Takes an xarray Dataset containing vege values and calculates the number of
    seasons for each timeseries pixel. The number of seasons provides a count of 
    number of significant peaks in each pixel timeseries. Note: this function will
    rechunk the dataset.

    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing an Dataset of veg_index 
        and time values. 

    Returns
    -------
    da_num_seasons : xarray DataArray
        An xarray DataSet type with an x and y dimension (no time). Each pixel is the 
        number of seasons value detected across the timeseries at each pixel.
    """
    
    # imports
    from scipy.signal import find_peaks
    
    # notify user
    print('Beginning calculation of number of seasons.')
    
    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x, y dimension in dataset.')
                
    # get vars in ds
    temp_vars = []
    if isinstance(ds, xr.Dataset):
        temp_vars = list(ds.data_vars)
    elif isinstance(ds, xr.DataArray):
        temp_vars = list(ds['variable'])

    # check if vege var and soil var given
    if 'veg_idx' not in temp_vars:
        raise ValueError('Vege var name not in dataset.')
        
    # if dask, need to rechunk to -1
    if bool(ds.chunks):
        print('Warning: had to rechunk dataset.')
        ds = ds.chunk({'time': -1})

    # set up calc peaks functions
    def calc_peaks(x, t):

        # calc num peaks
        peaks, _ = find_peaks(x, prominence=t)
        
        # check and correct to 1 if nothing
        if len(peaks) > 0:
            num_peaks = len(peaks)
        else:
            num_peaks = 1

        # return
        return num_peaks
    
    # prepare prominence threshold dataset
    ds_thresh = (ds.max('time') - ds.min('time')) / 8
        
    # calculate nos using calc_funcs func
    print('Calculating number of seasons.')
    ds_nos = xr.apply_ufunc(calc_peaks, 
                            ds['veg_idx'], 
                            ds_thresh['veg_idx'],
                            input_core_dims=[['time'], []],
                            vectorize=True, 
                            dask='parallelized', 
                            output_dtypes=[np.int8])
    
    # convert type
    #da_nos = da_nos.astype('int16')
    
    # rename
    ds_nos = ds_nos.rename('nos_values')
    
    # convert to dataset
    ds_nos = ds_nos.to_dataset()
    
    # notify user
    print('Success!')
        
    return ds_nos
 

def add_crs(ds, crs):
    """
    Takes an xarray Dataset adds previously extracted crs metadata, if exists. 
    Returns None if not found.
    
    Parameters
    ----------
    ds: xarray Dataset
        A single- or multi-dimensional array with/without crs metadata.

    Returns
    -------
    ds: xarray Dataset
        A Dataset with a new crs.
    """
    
    # notify user
    print('Beginning addition of CRS metadata.')
    try:
        # notify user
        print('> Adding CRS metadata.')
        
        # assign crs via odc utils
        ds = assign_crs(ds, str(crs))
        
        # notify user
        print('> Success!\n')
        
    except:
        # notify user
        print('> Could not add CRS metadata to data. Aborting.\n')
        pass
        
    return ds
    

def get_pos(da):
    """
    Takes an xarray DataArray containing veg_index values and calculates the vegetation 
    value and time (day of year) at peak of season (pos) for each timeseries per-pixel. 
    The peak of season is the maximum value in the timeseries, per-pixel.
    
    Parameters
    ----------
    ds: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 

    Returns
    -------
    da_pos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at the peak of season (pos).
    da_pos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at the peak of season (pos).
    """
    
    # notify user
    print('Beginning calculation of peak of season (pos) values and times.')   

    # get pos values (max val in each pixel timeseries)
    print('Calculating peak of season (pos) values.')
    da_pos_values = da.max('time', keep_attrs=True)
        
    # get pos times (day of year) at max val in each pixel timeseries)
    print('Calculating peak of season (pos) times.')
    i = da.argmax('time', skipna=True)
    da_pos_times = da['time.dayofyear'].isel(time=i, drop=True)
    
    # convert type
    da_pos_values = da_pos_values.astype('float32')
    da_pos_times = da_pos_times.astype('int16')
    
    # rename vars
    da_pos_values = da_pos_values.rename('pos_values')
    da_pos_times = da_pos_times.rename('pos_times')

    # notify user
    print('Success!')
    return da_pos_values, da_pos_times


def get_vos(da):
    """
    Takes an xarray DataArray containing veg_index values and calculates the vegetation 
    value and time (day of year) at valley of season (vos) for each timeseries per-pixel. 
    The valley of season is the minimum value in the timeseries, per-pixel.
    
    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 

    Returns
    -------
    da_vos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at the valley of season (vos).
    da_vos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at the valley of season (vos).
    """
    
    # notify user
    print('Beginning calculation of valley of season (vos) values and times.')

    # get vos values (min val in each pixel timeseries)
    print('Calculating valley of season (vos) values.')
    da_vos_values = da.min('time', keep_attrs=True)
    
    # get vos times (day of year) at min val in each pixel timeseries)
    print('Calculating valley of season (vos) times.')
    i = da.argmin('time', skipna=True)
    da_vos_times = da['time.dayofyear'].isel(time=i, drop=True)
    
    # convert type
    da_vos_values = da_vos_values.astype('float32')
    da_vos_times = da_vos_times.astype('int16')
    
    # rename vars
    da_vos_values = da_vos_values.rename('vos_values')
    da_vos_times = da_vos_times.rename('vos_times')

    # notify user
    print('Success!')
    return da_vos_values, da_vos_times


def get_mos(da, da_peak_times):
    """
    Takes an xarray DataArray containing veg_index values and calculates the vegetation 
    values (time not available) at middle of season (mos) for each timeseries per-pixel. 
    The middle of season is the mean vege value and time (day of year) in the timeseries
    at 80% to left and right of the peak of season (pos) per-pixel.
    
    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel must be
        the time (day of year) value calculated at peak of season (pos) prior.

    Returns
    -------
    da_mos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at the peak of season (pos).
    """
    
    # notify user
    print('Beginning calculation of middle of season (mos) values (times not possible).')  

    # get left and right slopes values
    print('Calculating middle of season (mos) values.')
    slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
    slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        
    # getupper 80% values in positive slope on left and right
    slope_l_upper = slope_l.where(slope_l >= (slope_l.max('time', keep_attrs=True) * 0.8))
    slope_r_upper = slope_r.where(slope_r >= (slope_r.max('time', keep_attrs=True) * 0.8))

    # get means of slope left and right
    slope_l_means = slope_l_upper.mean('time', keep_attrs=True)
    slope_r_means = slope_r_upper.mean('time', keep_attrs=True)
    
    # get attrs
    attrs = da.attrs

    # combine left and right veg_index means
    da_mos_values = (slope_l_means + slope_r_means) / 2
    
    # convert type, rename
    da_mos_values = da_mos_values.astype('float32')
    da_mos_values = da_mos_values.rename('mos_values')
    
    # add attrs back on
    da_mos_values.attrs = attrs

    # notify user
    print('Success!')
    return da_mos_values
    

def get_bse(da, da_peak_times):
    """
    Takes an xarray DataArray containing veg_index values and calculates the vegetation 
    value base (bse) for each timeseries per-pixel. The base is calculated as the mean 
    value of two minimum values; the min of the slope to the left of peak of season, and
    the min of the slope to the right of the peak of season. Users must provide an existing
    peak of season (pos) data array, which can either be the max of the timeseries, or the
    middle of season (mos) values.
    
    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional DataArray containing an array of veg_index
        values.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel must be
        the time (day of year) value calculated at either at peak of season (pos) or middle 
        of season (mos) prior.

    Returns
    -------
    da_bse_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        base (bse) veg_index value detected at across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of base (bse) values (times not possible).')

    # get vos values (min val in each pixel timeseries)
    print('Calculating base (bse) values.')
    
    # split timeseries into left and right slopes via provided peak/middle values
    slope_l = da.where(da['time.dayofyear'] <= da_peak_times).min('time')
    slope_r = da.where(da['time.dayofyear'] >= da_peak_times).min('time')
    
    # get attrs
    attrs = da.attrs
    
    # get per pixel mean of both left and right slope min values 
    da_bse_values = (slope_l + slope_r) / 2
    
    # convert type, rename
    da_bse_values = da_bse_values.astype('float32')
    da_bse_values = da_bse_values.rename('bse_values')
    
    # add attrs back on
    da_bse_values.attrs = attrs

    # notify user
    print('Success!')
    return da_bse_values


def get_aos(da_peak_values, da_base_values):
    """
    Takes two xarray DataArrays containing the highest vege values (pos or mos) and the
    lowest vege values (bse or vos) and calculates the amplitude of season (aos). 
    The amplitude is calculated as the highest values minus the lowest values per pixel.

    Parameters
    ----------
    da_peak_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the peak (pos) or middle (mos) of season.
    da_base_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the base (bse) or valley (vos) of season.

    Returns
    -------
    da_aos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        amplitude of season (aos) value detected between the peak and base vege values
        across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of amplitude of season (aos) values (times not possible).')
        
    # get attrs
    attrs = da_peak_values.attrs

    # get aos values (peak - base in each pixel timeseries)
    print('Calculating amplitude of season (aos) values.')
    da_aos_values = da_peak_values - da_base_values
    
    # convert type, rename
    da_aos_values = da_aos_values.astype('float32')
    da_aos_values = da_aos_values.rename('aos_values')
    
    # add attrs back on
    da_aos_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_aos_values


def get_sos(da, da_peak_times, da_base_values, da_aos_values, method, factor, thresh_sides, abs_value):
    """
    Takes several xarray DataArrays containing the highest vege values and times (pos or mos), 
    the lowest vege values (bse or vos), and the amplitude (aos) values and calculates the 
    vegetation values and times at the start of season (sos). Several methods can be used to
    detect the start of season; most are based on TIMESAT 3.3 methodology.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at either the peak (pos) or middle (mos) of 
        season.
    da_base_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the base (bse) or valley (vos) of season.
    da_aos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        amplitude of season (aos) value detected between the peak and base vege values
        across the timeseries at each pixel.
    method: str
        A string indicating which start of season detection method to use. Default is
        same as TIMESAT: seasonal_amplitude. The available options include:
        1. first_of_slope: lowest vege value of slope is sos (i.e. first lowest value).
        2. median_of_slope: middle vege value of slope is sos (i.e. median value).
        3. seasonal_amplitude: uses a percentage of the amplitude from base to find sos.
        4. absolute_value: users defined absolute value in vege index units is used to find sos.
        5. relative_amplitude: robust mean peak and base, and a factor of that area, used to find sos.
        6. stl_trend: robust but slow - uses seasonal decomp LOESS method to find trend line and sos.
    factor: float
        A float value between 0 and 1 which is used to increase or decrease the amplitude
        threshold for the seasonal_amplitude method. A factor closer to 0 results in start 
        of season nearer to min value, a factor closer to 1 results in start of season
        closer to peak of season.
    thresh_sides: str
        A string indicating whether the sos value threshold calculation should be the min 
        value of left slope (one_sided) only, or use the bse/vos value (two_sided) calculated
        earlier. Default is two_sided, as per TIMESAT 3.3. That said, one_sided is potentially
        more robust.
    abs_value: float
        For absolute_value method only. Defines the absolute value in units of the vege index to
        which sos is defined. The part of the vege slope that the absolute value hits will be the
        sos value and time.

    Returns
    -------
    da_sos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at the start of season (sos).
    da_sos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at the start of season (sos).
    """
    
    # imports
    try:
        from statsmodels.tsa.seasonal import STL as stl
    except:
        print('Could not import statsmodel. Using seasonal amplitude method instead of stl.')
        method = 'seasonal_amplitude'
    
    # notify user
    print('Beginning calculation of start of season (sos) values and times.')
    
    # check factor
    if factor < 0 or factor > 1:
        raise ValueError('Provided factor value is not between 0 and 1.')
            
    # check thresh_sides
    if thresh_sides not in ['one_sided', 'two_sided']:
        raise ValueError('Provided thresh_sides value is not one_sided or two_sided.')
                    
    if method == 'first_of_slope':
        
        # notify user
        print('Calculating start of season (sos) values via method: first_of_slope.')
          
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
                
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)
        
        # get median of vege on pos left slope, calc vege dists from median
        slope_l_med = slope_l_pos.median('time', keep_attrs=True)
        dists_from_median = slope_l_pos - slope_l_med 
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_median.isnull().all('time')
        dists_from_median = xr.where(mask, 0.0, dists_from_median)
        
        # get time index where min dist from median (first on slope)
        i = dists_from_median.argmin('time', skipna=True)
        
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)
        
        # notify user
        print('Calculating start of season (sos) times via method: first_of_slope.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)

    elif method == 'median_of_slope':
        
        # notify user
        print('Calculating start of season (sos) values via method: median_of_slope.')
          
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
                
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)
        
        # get median of vege on pos left slope, calc absolute vege dists from median
        slope_l_med = slope_l_pos.median('time')
        dists_from_median = slope_l_pos - slope_l_med
        dists_from_median = abs(dists_from_median)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_median.isnull().all('time')
        dists_from_median = xr.where(mask, 0.0, dists_from_median)
        
        # get time index where min absolute dist from median (median on slope)
        i = dists_from_median.argmin('time', skipna=True)
        
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)
        
        # notify user
        print('> Calculating start of season (sos) times via method: median_of_slope.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'seasonal_amplitude':
        
        # notify user
        print('Calculating start of season (sos) values via method: seasonal_amplitude.')
        
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
                
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)
                   
        # use just the left slope min val (one), or use the bse/vos calc earlier (two) for sos
        if thresh_sides == 'one_sided':
            da_sos_values = (da_aos_values * factor) + slope_l.min('time')
        elif thresh_sides == 'two_sided':
            da_sos_values = (da_aos_values * factor) + da_base_values
                        
        # calc distance of pos vege from calculated sos value
        dists_from_sos_values = abs(slope_l_pos - da_sos_values)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_sos_values.isnull().all('time')
        dists_from_sos_values = xr.where(mask, 0.0, dists_from_sos_values)
            
        # get time index where min absolute dist from sos
        i = dists_from_sos_values.argmin('time', skipna=True)
                 
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)
                
        # notify user
        print('Calculating start of season (sos) times via method: seasonal_amplitude.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)
    
    elif method == 'absolute_value':
        
        # notify user
        print('Calculating start of season (sos) values via method: absolute_value.')
        
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
        
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)
        
        # calc abs distance of positive slope from absolute value
        dists_from_abs_value = abs(slope_l_pos - abs_value)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_abs_value.isnull().all('time')
        dists_from_abs_value = xr.where(mask, 0.0, dists_from_abs_value)
        
        # get time index where min absolute dist from sos (absolute value)
        i = dists_from_abs_value.argmin('time', skipna=True)
        
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)
        
        # notify user
        print('Calculating start of season (sos) times via method: absolute_value.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'relative_value':

        # notify user
        print('Calculating start of season (sos) values via method: relative_value.')
        print('Warning: this can take a long time.')
        
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
        
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)

        # get relative amplitude via robust max and base (10% cut off either side)
        relative_amplitude = da.quantile(dim='time', q=0.90) - da.quantile(dim='time', q=0.10)
        
        # get sos value with user factor and robust mean base
        da_sos_values = (relative_amplitude * factor) + da.quantile(dim='time', q=0.10)
        
        # drop annoying quantile attribute from sos, ignore errors
        da_sos_values = da_sos_values.drop('quantile', errors='ignore')
           
        # calc abs distance of positive slope from sos values
        dists_from_sos_values = abs(slope_l_pos - da_sos_values)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_sos_values.isnull().all('time')
        dists_from_sos_values = xr.where(mask, 0.0, dists_from_sos_values)

        # get time index where min absolute dist from sos
        i = dists_from_sos_values.argmin('time', skipna=True)
                
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)

        # notify user
        print('Calculating start of season (sos) times via method: relative_value.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'stl_trend':
        
        # notify user
        print('Calculating start of season (sos) values via method: stl_trend.')
        
        # check if num seasons for stl is odd, +1 if not
        num_periods = len(da['time'])
        if num_periods % 2 == 0:
            num_periods = num_periods + 1
            print('Number of stl periods is even number, added 1 to make it odd.')
        
        # prepare stl params
        stl_params = {
            'period': num_periods,
            'seasonal': 7,
            'trend': None,
            'low_pass': None,
            'robust': False
        }
        
        # prepare stl func
        def func_stl(v, period, seasonal, trend, low_pass, robust):
            model = stl(v, period=period, seasonal=seasonal, trend=trend, low_pass=low_pass, robust=robust)
            return model.fit().trend
        
        # notify user
        print('Performing seasonal decomposition via LOESS. Warning: this can take a long time.')
        da_stl = xr.apply_ufunc(func_stl, da, 
                                input_core_dims=[['time']], 
                                output_core_dims=[['time']], 
                                vectorize=True, 
                                dask='parallelized', 
                                output_dtypes=[np.float32],
                                kwargs=stl_params)
        
        # get left slopes values, calc differentials, subset to positive differentials
        slope_l = da.where(da['time.dayofyear'] <= da_peak_times)
        slope_l_diffs = slope_l.differentiate('time')
        slope_l_pos_diffs = xr.where(slope_l_diffs > 0, True, False)
        
        # select vege values where positive on left slope
        slope_l_pos = slope_l.where(slope_l_pos_diffs)
        
        # get min value left known pos date
        stl_l = da_stl.where(da_stl['time.dayofyear'] <= da_peak_times)
        
        # calc abs distance of positive slope from stl values
        dists_from_stl_values = abs(slope_l_pos - stl_l)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_stl_values.isnull().all('time')
        dists_from_stl_values = xr.where(mask, 0.0, dists_from_stl_values)

        # get time index where min absolute dist from sos
        i = dists_from_stl_values.argmin('time', skipna=True)
                
        # get vege start of season values and times (day of year)
        da_sos_values = slope_l_pos.isel(time=i, drop=True)
        
        # notify user
        print('Calculating start of season (sos) times via method: stl_trend.')
        
        # get vege start of season times (day of year)
        da_sos_times = slope_l_pos['time.dayofyear'].isel(time=i, drop=True)
    
    else:
        raise ValueError('Provided method not supported.')
        
    # replace any all nan slices with first val and time in each pixel
    da_sos_values = da_sos_values.where(~mask, np.nan)
    da_sos_times = da_sos_times.where(~mask, np.nan)
    
    # convert type
    da_sos_values = da_sos_values.astype('float32')
    da_sos_times = da_sos_times.astype('int16')
    
    # rename
    da_sos_values = da_sos_values.rename('sos_values')
    da_sos_times = da_sos_times.rename('sos_times')
            
    # notify user
    print('Success!')
    return da_sos_values, da_sos_times


def get_eos(da, da_peak_times, da_base_values, da_aos_values, method, factor, thresh_sides, abs_value):
    """
    Takes several xarray DataArrays containing the highest vege values and times (pos or mos), 
    the lowest vege values (bse or vos), and the amplitude (aos) values and calculates the 
    vegetation values and times at the end of season (eos). Several methods can be used to
    detect the start of season; most are based on TIMESAT 3.3 methodology.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at either the peak (pos) or middle (mos) of 
        season.
    da_base_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the base (bse) or valley (vos) of season.
    da_aos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        amplitude of season (aos) value detected between the peak and base vege values
        across the timeseries at each pixel.
    method: str
        A string indicating which start of season detection method to use. Default is
        same as TIMESAT: seasonal_amplitude. The available options include:
        1. first_of_slope: lowest vege value of slope is eos (i.e. first lowest value).
        2. median_of_slope: middle vege value of slope is eos (i.e. median value).
        3. seasonal_amplitude: uses a percentage of the amplitude from base to find eos.
        4. absolute_value: users defined absolute value in vege index units is used to find eos.
        5. relative_amplitude: robust mean peak and base, and a factor of that area, used to find eos.
        6. stl_trend: robust but slow - uses seasonal decomp LOESS method to find trend line and eos.
    factor: float
        A float value between 0 and 1 which is used to increase or decrease the amplitude
        threshold for the seasonal_amplitude method. A factor closer to 0 results in end 
        of season nearer to min value, a factor closer to 1 results in end of season
        closer to peak of season.
    thresh_sides: str
        A string indicating whether the sos value threshold calculation should be the min 
        value of left slope (one_sided) only, or use the bse/vos value (two_sided) calculated
        earlier. Default is two_sided, as per TIMESAT 3.3. That said, one_sided is potentially
        more robust.
    abs_value: float
        For absolute_value method only. Defines the absolute value in units of the vege index to
        which sos is defined. The part of the vege slope that the absolute value hits will be the
        sos value and time.

    Returns
    -------
    da_eos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at the end of season (eos).
    da_eos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at the end of season (eos).
    """

    # imports
    try:
        from statsmodels.tsa.seasonal import STL as stl
    except:
        print('Could not import statsmodel. Using seasonal amplitude method instead of stl.')
        method = 'seasonal_amplitude'
    
    # notify user
    print('Beginning calculation of end of season (eos) values and times.')
    
    # check factor
    if factor < 0 or factor > 1:
        raise ValueError('Provided factor value is not between 0 and 1. Aborting.')
            
    # check thresh_sides
    if thresh_sides not in ['one_sided', 'two_sided']:
        raise ValueError('Provided thresh_sides value is not one_sided or two_sided. Aborting.')
                    
    if method == 'first_of_slope':
        
        # notify user
        print('Calculating end of season (eos) values via method: first_of_slope.')
          
        # get right slopes values, calc differentials, subset to negative differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
                
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)
        
        # get median of vege on neg right slope, calc vege dists from median
        slope_r_med = slope_r_neg.median('time')
        dists_from_median = slope_r_neg - slope_r_med 
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_median.isnull().all('time')
        dists_from_median = xr.where(mask, 0.0, dists_from_median)
        
        # get time index where min dist from median (first on slope)
        i = dists_from_median.argmin('time', skipna=True)
        
        # get vege end of season values and times (day of year)
        da_eos_values = slope_r_neg.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: first_of_slope.')
        
        # get vege start of season times (day of year)
        da_eos_times = slope_r_neg['time.dayofyear'].isel(time=i, drop=True)

    elif method == 'median_of_slope':
        
        # notify user
        print('Calculating end of season (eos) values via method: median_of_slope.')
          
        # get right slopes values, calc differentials, subset to positive differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
                
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)
        
        # get median of vege on neg right slope, calc absolute vege dists from median
        slope_r_med = slope_r_neg.median('time')
        dists_from_median = slope_r_neg - slope_r_med
        dists_from_median = abs(dists_from_median)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_median.isnull().all('time')
        dists_from_median = xr.where(mask, 0.0, dists_from_median)
        
        # get time index where min absolute dist from median (median on slope)
        i = dists_from_median.argmin('time', skipna=True)
        
        # get vege start of season values and times (day of year)
        da_eos_values = slope_r_neg.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: median_of_slope.')
        
        # get vege end of season times (day of year)
        da_eos_times = slope_r_neg['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'seasonal_amplitude':
        
        # notify user
        print('Calculating end of season (eos) values via method: seasonal_amplitude.')
        
        # get right slopes values, calc differentials, subset to negative differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
        
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)       
               
        # use just the right slope min val (one), or use the bse/vos calc earlier (two) for sos
        if thresh_sides == 'one_sided':
            da_eos_values = (da_aos_values * factor) + slope_r.min('time')
        elif thresh_sides == 'two_sided':
            da_eos_values = (da_aos_values * factor) + da_base_values
            
        # calc distance of neg vege from calculated eos value
        dists_from_eos_values = abs(slope_r_neg - da_eos_values)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_eos_values.isnull().all('time')
        dists_from_eos_values = xr.where(mask, 0.0, dists_from_eos_values)
        
        # get time index where min absolute dist from eos
        i = dists_from_eos_values.argmin('time', skipna=True)
                
        # get vege end of season values and times (day of year)
        da_eos_values = slope_r_neg.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: seasonal_amplitude.')
        
        # get vege end of season times (day of year)
        da_eos_times = slope_r_neg['time.dayofyear'].isel(time=i, drop=True)
    
    elif method == 'absolute_value':
        
        # notify user
        print('Calculating end of season (eos) values via method: absolute_value.')
        
        # get right slopes values, calc differentials, subset to negative differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
        
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)
        
        # calc abs distance of negative slope from absolute value
        dists_from_abs_value = abs(slope_r_neg - abs_value)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_abs_value.isnull().all('time')
        dists_from_abs_value = xr.where(mask, 0.0, dists_from_abs_value)
        
        # get time index where min absolute dist from eos (absolute value)
        i = dists_from_abs_value.argmin('time', skipna=True)
        
        # get vege end of season values and times (day of year)
        da_eos_values = slope_r_neg.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: absolute_value.')
        
        # get vege end of season times (day of year)
        da_eos_times = slope_r_neg['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'relative_value':

        # notify user
        print('Calculating end of season (eos) values via method: relative_value.')
        print('Warning: this can take a long time.')
        
        # get right slopes values, calc differentials, subset to negative differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
        
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)

        # get relative amplitude via robust max and base (10% cut off either side)
        relative_amplitude = da.quantile(dim='time', q=0.90) - da.quantile(dim='time', q=0.10)
        
        # get eos value with user factor and robust mean base
        da_eos_values = (relative_amplitude * factor) + da.quantile(dim='time', q=0.10)
        
        # drop annoying quantile attribute from eos, ignore errors
        da_eos_values = da_eos_values.drop('quantile', errors='ignore')
           
        # calc abs distance of negative slope from eos values
        dists_from_eos_values = abs(slope_r_neg - da_eos_values)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_eos_values.isnull().all('time')
        dists_from_eos_values = xr.where(mask, 0.0, dists_from_eos_values)

        # get time index where min absolute dist from eos
        i = dists_from_eos_values.argmin('time', skipna=True)
                
        # get vege end of season values and times (day of year)
        da_eos_values = slope_r_neg.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: relative_value.')
        
        # get vege end of season times (day of year)
        da_eos_times = slope_r_neg['time.dayofyear'].isel(time=i, drop=True)
        
    elif method == 'stl_trend':
        
        # notify user
        print('Calculating end of season (eos) values via method: stl_trend.')
        
        # check if num seasons for stl is odd, +1 if not
        num_periods = len(da['time'])
        if num_periods % 2 == 0:
            num_periods = num_periods + 1
            print('> Number of stl periods is even number, added 1 to make it odd.')
        
        # prepare stl params
        stl_params = {
            'period': num_periods,
            'seasonal': 7,
            'trend': None,
            'low_pass': None,
            'robust': False
        }
        
        # prepare stl func
        def func_stl(v, period, seasonal, trend, low_pass, robust):
            model = stl(v, period=period, seasonal=seasonal, trend=trend, low_pass=low_pass, robust=robust)
            return model.fit().trend
        
        # notify user
        print('Performing seasonal decomposition via LOESS. Warning: this can take a long time.')
        da_stl = xr.apply_ufunc(func_stl, da, 
                                input_core_dims=[['time']], 
                                output_core_dims=[['time']], 
                                vectorize=True, 
                                dask='parallelized', 
                                output_dtypes=[np.float32],
                                kwargs=stl_params)
        
        # get right slopes values, calc differentials, subset to negative differentials
        slope_r = da.where(da['time.dayofyear'] >= da_peak_times)
        slope_r_diffs = slope_r.differentiate('time')
        slope_r_neg_diffs = xr.where(slope_r_diffs < 0, True, False)
        
        # select vege values where negative on right slope
        slope_r_neg = slope_r.where(slope_r_neg_diffs)
        
        # get min value right known pos date
        stl_r = da_stl.where(da_stl['time.dayofyear'] >= da_peak_times)
        
        # calc abs distance of negative slope from stl values
        dists_from_stl_values = abs(slope_r_neg - stl_r)
        
        # make mask for all nan pixels and fill with 0.0 (needs to be float)
        mask = dists_from_stl_values.isnull().all('time')
        dists_from_stl_values = xr.where(mask, 0.0, dists_from_stl_values)

        # get time index where min absolute dist from eos
        i = dists_from_stl_values.argmin('time', skipna=True)
                
        # get vege end of season values and times (day of year)
        da_eos_values = slope_r_eos.isel(time=i, drop=True)
        
        # notify user
        print('Calculating end of season (eos) times via method: stl_trend.')
        
        # get vege end of season times (day of year)
        da_eos_times = slope_r_eos['time.dayofyear'].isel(time=i, drop=True)
        
    else:
        raise ValueError('Provided method not supported.')

    # replace any all nan slices with last val and time in each pixel
    da_eos_values = da_eos_values.where(~mask, np.nan)
    da_eos_times = da_eos_times.where(~mask, np.nan)
    
    # convert type
    da_eos_values = da_eos_values.astype('float32')
    da_eos_times = da_eos_times.astype('int16')
    
    # rename
    da_eos_values = da_eos_values.rename('eos_values')
    da_eos_times = da_eos_times.rename('eos_times')    
        
    # notify user
    print('Success!')
    return da_eos_values, da_eos_times    


def get_los(da, da_sos_times, da_eos_times):
    """
    Takes two xarray DataArrays containing the start of season (sos) times (day of year) 
    and end of season (eos) times (day of year) and calculates the length of season (los). 
    This is calculated as eos day of year minus sos day of year per pixel.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 
    da_sos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at start of season (sos).
    da_eos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at end of season (eos).

    Returns
    -------
    da_los_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        length of season (los) value detected between the sos and eos day of year values
        across the timeseries at each pixel. The values in los represents number of days.
    """
    
    # notify user
    print('Beginning calculation of length of season (los) values (times not possible).')
    
    # get attrs
    attrs = da.attrs

    # get los values (eos day of year - sos day of year)
    print('Calculating length of season (los) values.')
    da_los_values = da_eos_times - da_sos_times
    
    # correct los if negative values exist
    if xr.where(da_los_values < 0, True, False).any():
        
        # get max time (day of year) and negatives into data arrays
        da_max_times = da['time.dayofyear'].isel(time=-1)
        da_neg_values = da_eos_times.where(da_los_values < 0) - da_sos_times.where(da_los_values < 0)
        
        # replace negative values with max time 
        da_los_values = xr.where(da_los_values >= 0, da_los_values, da_max_times + da_neg_values)
        
        # drop time dim if exists
        da_los_values = da_los_values.drop({'time'}, errors='ignore')

    # convert type, rename
    da_los_values = da_los_values.astype('int16')
    da_los_values = da_los_values.rename('los_values')
    
    # add attrs back on
    da_los_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_los_values    


def get_roi(da_peak_values, da_peak_times, da_sos_values, da_sos_times):
    """
    Takes four xarray DataArrays containing the peak season values (either pos or 
    mos) and times (day of year), and start of season (sos) values and times (day 
    of year). The rate of increase (roi) is calculated as the ratio of the difference 
    in peak and sos for vege values and time (day of year) per pixel. 

    Parameters
    ----------
    da_peak_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        vege value detected at either the peak (pos) or middle (mos) of season.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at either the peak (pos) or middle (mos) of 
        season.
    da_sos_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        vege value detected at start of season (sos).
    da_sos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at start of season (sos).

    Returns
    -------
    da_roi_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        rate of increase value detected between the sos and peak values/times across the 
        timeseries at each pixel. The values in roi represents rate of vege growth.
    """
    
    # notify user
    print('Beginning calculation of rate of increase (roi) values (times not possible).')
    
    # get attrs
    attrs = da_peak_values.attrs

    # get ratio between the difference in peak and sos values and times
    print('Calculating rate of increase (roi) values.')
    da_roi_values = (da_peak_values - da_sos_values) / (da_peak_times - da_sos_times)    

    # convert type, rename
    da_roi_values = da_roi_values.astype('float32')
    da_roi_values = da_roi_values.rename('roi_values')
    
    # add attrs back on
    da_roi_values.attrs = attrs

    # notify user
    print('Success!')
    return da_roi_values


def get_rod(da_peak_values, da_peak_times, da_eos_values, da_eos_times):
    """
    Takes four xarray DataArrays containing the peak season values (either pos or 
    mos) and times (day of year), and end of season (eos) values and times (day 
    of year). The rate of decrease (rod) is calculated as the ratio of the difference 
    in peak and eos for vege values and time (day of year) per pixel. 

    Parameters
    ----------
    da_peak_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        vege value detected at either the peak (pos) or middle (mos) of season.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) value detected at either the peak (pos) or middle (mos) of 
        season.
    da_eos_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        vege value detected at end of season (eos).
    da_eos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at end of season (eos).

    Returns
    -------
    da_roi_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        rate of decrease value detected between the eos and peak values/times across the 
        timeseries at each pixel. The values in rod represents rate of vege decline.
    """
    
    # notify user
    print('Beginning calculation of rate of decrease (rod) values (times not possible).')   
    
    # get attrs
    attrs = da_peak_values.attrs

    # get abs ratio between the difference in peak and eos values and times
    print('Calculating rate of decrease (rod) values.')
    da_rod_values = abs((da_eos_values - da_peak_values) / (da_eos_times - da_peak_times))
    
    # convert type, rename
    da_rod_values = da_rod_values.astype('float32')
    da_rod_values = da_rod_values.rename('rod_values')
    
    # add attrs back on
    da_rod_values.attrs = attrs

    # notify user
    print('Success!')
    return da_rod_values


def get_lios(da, da_sos_times, da_eos_times):
    """
    Takes three xarray DataArrays containing vege values and sos/eos times (day of year) to
    calculate the long integral of season (lios) for each timeseries pixel. The lios is
    considered to act as a surrogate of vegetation productivity during growing season. The 
    long integral is calculated via the traperzoidal rule.

    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values.
    da_sos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at start of season (sos).
    da_eos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at end of season (eos).

    Returns
    -------
    da_lios_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        long integral of season (lios) value detected across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of long integral of season (lios) values (times not possible).')
    
    # get attrs
    attrs = da.attrs

    # get vals between sos and eos times, replace any outside vals with 0
    print('Calculating long integral of season (lios) values.')
    da_lios_values = da.where((da['time.dayofyear'] >= da_sos_times) &
                              (da['time.dayofyear'] <= da_eos_times), 0)
    
    # calculate lios using trapz (note: more sophisticated than integrate)
    da_lios_values = xr.apply_ufunc(np.trapz, 
                                    da_lios_values, 
                                    input_core_dims=[['time']],
                                    dask='parallelized', 
                                    output_dtypes=[np.float32],
                                    kwargs={'dx': 1})
    
    # convert type, rename
    da_lios_values = da_lios_values.astype('float32')
    da_lios_values = da_lios_values.rename('lios_values')
    
    # add attrs back on
    da_lios_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_lios_values


def get_sios(da, da_sos_times, da_eos_times, da_base_values):
    """
    Takes four xarray DataArrays containing vege values,  sos and eos times and base
    values (vos or bse) and calculates the short integral of season (sios) for each 
    timeseries pixel. The sios is considered to act as a surrogate of vegetation productivity 
    minus the understorey vegetation during growing season. The short integral is calculated 
    via integrating the array with the traperzoidal rule minus the trapezoidal of the base.

    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values.
    da_sos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at start of season (sos).
    da_eos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        time (day of year) detected at end of season (eos).
    da_base_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the base (bse) or valley (vos) of season.

    Returns
    -------
    da_sios_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        short integral of season (sios) value detected across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of short integral of season (sios) values (times not possible).')  
    
    # get attrs
    attrs = da.attrs

    # get veg vals between sos and eos times, replace any outside vals with 0
    print('Calculating short integral of season (sios) values.')
    da_sios_values = da.where((da['time.dayofyear'] >= da_sos_times) &
                              (da['time.dayofyear'] <= da_eos_times), 0)
    
    # calculate sios using trapz (note: more sophisticated than integrate)
    da_sios_values = xr.apply_ufunc(np.trapz, 
                                    da_sios_values, 
                                    input_core_dims=[['time']],
                                    dask='parallelized', 
                                    output_dtypes=[np.float32],
                                    kwargs={'dx': 1})
    
    # combine 2d base vals with 3d da, projecting const base val to pixel timeseries (i.e. a rectangle)
    da_sios_bse_values = da_base_values.combine_first(da)
    
    # get base vals between sos and eos times, replace any outside vals with 0
    da_sios_bse_values = da_sios_bse_values.where((da_sios_bse_values['time.dayofyear'] >= da_sos_times) &
                                                  (da_sios_bse_values['time.dayofyear'] <= da_eos_times), 0)
    
    # calculate trapz of base (note: more sophisticated than integrate)
    da_sios_bse_values = xr.apply_ufunc(np.trapz, 
                                        da_sios_bse_values, 
                                        input_core_dims=[['time']],
                                        dask='parallelized', 
                                        output_dtypes=[np.float32],
                                        kwargs={'dx': 1})
    
    # remove base trapz from sios values
    da_sios_values = da_sios_values - da_sios_bse_values
    
    # convert type, rename
    da_sios_values = da_sios_values.astype('float32')
    da_sios_values = da_sios_values.rename('sios_values')
    
    # add attrs back on
    da_sios_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_sios_values


def get_liot(da):
    """
    Takes an xarray DataArray containing vege values and calculates the long integral of
    total (liot) for each timeseries pixel. The liot is considered to act as a surrogate of 
    vegetation productivty. The long integral is calculated via the traperzoidal rule.

    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values. 

    Returns
    -------
    da_liot_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        long integral of total (liot) value detected across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of long integral of total (liot) values (times not possible).')
    
    # get attrs
    attrs = da.attrs

    # calculate liot using trapz (note: more sophisticated than integrate)
    print('Calculating long integral of total (liot) values.')
    da_liot_values = xr.apply_ufunc(np.trapz, 
                                    da, 
                                    input_core_dims=[['time']],
                                    dask='parallelized', 
                                    output_dtypes=[np.float32],
                                    kwargs={'dx': 1})
    
    # convert type, rename
    da_liot_values = da_liot_values.astype('float32')
    da_liot_values = da_liot_values.rename('liot_values')
    
    # add attrs back on
    da_liot_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_liot_values


def get_siot(da, da_base_values):
    """
    Takes an xarray DataArray containing vege values and calculates the short integral of
    total (siot) for each timeseries pixel. The siot is considered to act as a surrogate of 
    vegetation productivity minus the understorey vegetation. The short integral is calculated 
    via integrating the array with the traperzoidal rule minus the trapezoidal of the base.

    Parameters
    ----------
    da: xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index 
        and time values.
    da_base_values: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        veg_index value detected at either the base (bse) or valley (vos) of season.

    Returns
    -------
    da_siot_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the 
        short integral of total (siot) value detected across the timeseries at each pixel.
    """
    
    # notify user
    print('Beginning calculation of short integral of total (siot) values (times not possible).') 
    
    # get attrs
    attrs = da.attrs

    # calculate siot using trapz (note: more sophisticated than integrate)
    print('Calculating short integral of total (siot) values.')
    da_siot_values = xr.apply_ufunc(np.trapz, 
                                    da, 
                                    input_core_dims=[['time']],
                                    dask='parallelized', 
                                    output_dtypes=[np.float32],
                                    kwargs={'dx': 1})
    
    # combine 2d base vals with 3d da, projecting const base val to pixel timeseries (i.e. a rectangle)
    da_siot_bse_values = da_base_values.combine_first(da)
    
    # calculate trapz of base (note: more sophisticated than integrate)
    da_siot_bse_values = xr.apply_ufunc(np.trapz, 
                                        da_siot_bse_values, 
                                        input_core_dims=[['time']],
                                        dask='parallelized', 
                                        output_dtypes=[np.float32],
                                        kwargs={'dx': 1})
    
    # remove base trapz from siot values
    da_siot_values = da_siot_values - da_siot_bse_values
    
    # convert type, rename
    da_siot_values = da_siot_values.astype('float32')
    da_siot_values = da_siot_values.rename('siot_values')
    
    # add attrs back on
    da_siot_values.attrs = attrs
    
    # notify user
    print('Success!')
    return da_siot_values
    


def calc_phenometrics(ds, metrics, peak_metric='pos', base_metric='bse', method='first_of_slope', factor=0.5,
                      thresh_sides='two_sided', abs_value=0, inplace=True):
    """
    """
    
    # check metrics
    #metric = ['pos']

    # check xr type, dims
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in list(ds.dims):
        raise ValueError('No time dimension in dataset.')
    elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x, y dimension in dataset.')

    # check if at least one year of data
    if len(ds.groupby('time.year').groups) < 1:
        raise ValueError('Need at least one year in dataset.')
        
    # convert to data array if dataset
    if isinstance(ds, xr.Dataset):
        ds = ds.to_array()
        
    # get vars in ds
    temp_vars = list(ds['variable'])
    if 'veg_idx' not in temp_vars:
        raise ValueError('Vege var name not in dataset.')

    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
    
    # check if dask - not yet supported
    if bool(ds.chunks):
        print('Dask arrays not yet supported. Computing data.')    
        
    # check if max, min metric parameters supported
    if peak_metric not in ['pos', 'mos']:
        raise ValueError('> The peak_metric parameter must be either pos or mos.')
    elif base_metric not in ['bse', 'vos']:
        raise ValueError('> The base_metric parameter must be either bse or vos.')
        
    # get crs info before work
    crs = tools.get_xr_crs(ds=ds)

    # take a mask of all-nan slices for clean up at end and set all-nan to 0s
    ds_all_nan_mask = ds.isnull().all('time')
    ds = ds.where(~ds_all_nan_mask, 0.0)
        
    # notify user
    print('Beginning calculation of phenometrics. Please wait.')
    
    # calc peak of season (pos) values and times
    da_pos_values, da_pos_times = get_pos(da=ds)
            
    # calc valley of season (vos) values and times
    da_vos_values, da_vos_times = get_vos(da=ds)

    # calc middle of season (mos) value (time not possible)
    da_mos_values = get_mos(da=ds, 
                            da_peak_times=da_pos_times)
    
    # calc base (bse) values (time not possible).
    da_bse_values = get_bse(da=ds, 
                            da_peak_times=da_pos_times)

    # calc amplitude of season (aos) values (time not possible)
    da_peak = da_pos_values if peak_metric == 'pos' else da_mos_values
    da_base = da_bse_values if base_metric == 'bse' else da_vos_values
    da_aos_values = get_aos(da_peak_values=da_peak, 
                            da_base_values=da_base)   
     
    # calc start of season (sos) values and times
    da_sos_values, da_sos_times = get_sos(da=ds,
                                          da_peak_times=da_pos_times, 
                                          da_base_values=da_base,
                                          da_aos_values=da_aos_values, 
                                          method=method, 
                                          factor=factor,
                                          thresh_sides=thresh_sides, 
                                          abs_value=abs_value)

    # calc end of season (eos) values and times
    da_eos_values, da_eos_times = get_eos(da=ds, 
                                          da_peak_times=da_pos_times, 
                                          da_base_values=da_base,
                                          da_aos_values=da_aos_values, 
                                          method=method, 
                                          factor=factor,
                                          thresh_sides=thresh_sides, 
                                          abs_value=abs_value)     
        
    # calc length of season (los) values (time not possible)
    da_los_values = get_los(da=ds, 
                            da_sos_times=da_sos_times, 
                            da_eos_times=da_eos_times)
    
    # calc rate of icnrease (roi) values (time not possible)
    da_roi_values = get_roi(da_peak_values=da_pos_values, 
                            da_peak_times=da_pos_times,
                            da_sos_values=da_sos_values, 
                            da_sos_times=da_sos_times)
        
    # calc rate of decrease (rod) values (time not possible)
    da_rod_values = get_rod(da_peak_values=da_pos_values, 
                            da_peak_times=da_pos_times,
                            da_eos_values=da_eos_values, 
                            da_eos_times=da_eos_times)
    
    # calc long integral of season (lios) values (time not possible)
    da_lios_values = get_lios(da=ds, 
                              da_sos_times=da_sos_times, 
                              da_eos_times=da_eos_times)
 
    # calc short integral of season (sios) values (time not possible)
    da_sios_values = get_sios(da=ds, 
                              da_sos_times=da_sos_times, 
                              da_eos_times=da_eos_times,
                              da_base_values=da_base)

    # calc long integral of total (liot) values (time not possible)
    da_liot_values = get_liot(da=ds)
        
    # calc short integral of total (siot) values (time not possible)
    da_siot_values = get_siot(da=ds, 
                              da_base_values=da_base)

    return da_lios_values, da_liot_values
    
    # create data array list
    da_list = [
        da_pos_values, 
        da_pos_times,
        da_mos_values, 
        da_vos_values, 
        da_vos_times,
        da_bse_values,
        da_aos_values,
        da_sos_values, 
        da_sos_times,
        da_eos_values, 
        da_eos_times,
        da_los_values,
        da_roi_values,
        da_rod_values,
        da_lios_values,
        da_sios_values,
        da_liot_values,
        da_siot_values
    ]
  
    # combine data arrays into one dataset
    ds_phenos = xr.merge(da_list)
    
    # set original all nan pixels back to nan
    ds_phenos = ds_phenos.where(~da_all_nan_mask)
    
    # add crs metadata back onto dataset
    ds_phenos = add_crs(ds=ds_phenos, crs=crs)
    
    # notify user
    print('Phenometrics calculated successfully!')
    return ds_phenos
