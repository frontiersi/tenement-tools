# gdvspectra
"""
"""

# todo
# check which funcs need ds to da and back (was_da flag)

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import kendalltau, theilslopes

sys.path.append('../../shared')
import satfetcher, tools

# meta, checks
def get_wet_dry_months(ds, wet_month=None, dry_month=None):
    """
    wet_month = single month or list of months
    """
    
    # notify
    print('Getting wet (JFM) and dry (SON) season months.')
    
    # check type
    if not isinstance(ds, (xr.DataArray, xr.Dataset)):
        raise TypeError('Must be a xarray DataArray or Dataset.')
        
    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
                        
    # tell user and reduce dataset to months in wet (JFM) or dry (SON) seasons only
    try:
        print('Reducing dataset into wet ({0}) and dry ({1}) months.'.format(wet_months, dry_months))
        ds_wet_dry = ds.sel(time=ds['time.month'].isin(wet_months + dry_months))
    
    except:
        raise ValueError('Could not reduce dataset into wet and dry months. Check requested months.')
    
    # notify and return
    print('Got wet and dry seasons months successfully.')
    return ds_wet_dry

# meta
def resample_to_wet_dry_medians(ds, wet_month=None, dry_month=None):
    """
    """
    
    # notify
    print('Resampling dataset down to annual, seasonal medians.')

    # check data type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Did not provide an xarray Dataset or DataArray.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # create copy ds
    ds = ds.copy(deep=True)

    # split into wet, dry
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

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
    
    # notify and return
    print('Resampled down to annual seasonal medians successfully.')
    return ds


# meta, checks, do one solid once over
def nullify_wet_dry_outliers(ds, wet_month=None, dry_month=None, p_value=0.01):
    """
    takes a ds or da with >= 1 var, splits into wet dry, gets mean 
    and std of whole image per date, puts into a zscore, creates a 
    mask of dates outside of z_value, sets those dates to all nan in
    full ds.
    this method will have to load memory and thus take some time
    """

    # notify
    print('Nullifying wet and dry season outliers usign z-score.')

    # check data type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Did not provide an xarray Dataset or DataArray.')
    
    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')
        
    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')
        
    # check if num years less than 3
    if len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years worth of data in dataset. Include more.')
        
    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # set z_value based on user significance (p_value)
    if p_value == 0.10:
        z_value = 1.65
    elif p_value == 0.05:
        z_value = 1.96
    elif p_value == 0.01:
        z_value = 2.58
    else:
        print('P-value not supported. Setting to 0.01.')
        z_value = 2.58
        
    # create copy ds
    ds = ds.copy(deep=True)
    
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
    z_wet = z_wet.where(z_wet > z_value, drop=True)
    z_dry = z_dry.where(z_dry > z_value, drop=True)
    
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
    print('Nullified wet and dry season outliers successfully.')
    return ds


# meta
def drop_incomplete_wet_dry_years(ds):
    """
    Takes an xarray dataset or dataarray, looks at number of months
    per year, and drops any where not equal to two months per year (wet 
    and dry seasons). If user resampled prior, only first and last years 
    will be potentially missing a season or two.
    """
    # notify
    print('Dropping years where num seasons not equal to 2 (wet and dry).')
    
    # check data type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Did not provide an xarray Dataset or DataArray.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')
        
    # create copy ds
    ds = ds.copy(deep=True)
    
    # get annual groups list, get first and last year info
    groups = list(ds.groupby('time.year').groups.items())
    
    # loop each year, check seasons, drop year
    removed_years = []
    for group in groups:
        if len(group[1]) != 2:
            ds = ds.where(ds['time.year'] != group[0], drop=True)
            print('Dropped year: {0}, not enough seasons.'.format(y[0]))
            removed_years.append(group[0])
            
    # notify
    if removed_years:
        removed_years = ', '.join(str(y) for y in removed_years)
        print('Warning: years: {0} were dropped.'.format(removed_years))
    else:
        print('No incomplete years detected. No data was dropped.')

    # return
    return ds


# meta
def fill_empty_wet_dry_edges(ds, wet_month=None, dry_month=None):
    """
    """

    # notify
    print('Filling empty wet and dry edges in dataset.')

    # check if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')

    # check if num years less than 3
    if len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years worth of data in dataset. Add more.')

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # create copy ds
    ds = ds.copy(deep=True)

    # split into wet, dry - we dont want to fill wet with dry, vice versa
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

    # helper func to fill edges via bfill, ffill
    def fill_edge(ds, edge=None):
        
        # check edge
        if edge not in ['first', 'last']:
            raise ValueError('Edge must be first or last.')

        # create sort order
        asc = True
        if edge == 'last':
            asc = False

        # loop each da in ds
        for i, dt in enumerate(ds['time'].sortby('time', asc)):
            da = ds.sel(time=dt)

            # is da is all nan or not
            is_all_nan = da.isnull().all().to_array().any()

            # if edge empty, get next time with vals, fill
            if i == 0 and is_all_nan:
                print('{0} time is empty. Processing to fill.'.format(edge.title()))

            elif i == 0 and not is_all_nan:
                print('{0} time has values. No need to fill.'.format(edge.title()))
                break

            elif i > 0 and not is_all_nan:
                print('Performing backfill.')
                if edge == 'first':
                    ds = xr.where(ds['time'] <= ds['time'].sel(time=dt), ds.bfill('time'), ds)
                elif edge == 'last':
                    ds = xr.where(ds['time'] >= ds['time'].sel(time=dt), ds.ffill('time'), ds)
                break

        return ds

    # fill edges for wet first, last
    print('Filling wet season edges.')
    ds_wet = fill_edge(ds_wet, edge='first')
    ds_wet = fill_edge(ds_wet, edge='last')

    # fill edges for wet first, last
    print('Filling dry season edges.')
    ds_dry = fill_edge(ds_dry, edge='first')
    ds_dry = fill_edge(ds_dry, edge='last')

    # concat wet, dry datasets back together
    ds = xr.concat([ds_wet, ds_dry], dim='time').sortby('time')

    # convert back to datarray
    if was_da:
        ds = ds.to_array()
    
    # notify and return
    print('Filled empty wet and dry edges successfully.')
    return ds


# meta, dont like the chunking stuff here
def interpolate_empty_wet_dry(ds, wet_month=None, dry_month=None, method='full'):
    """
    method = full gives us interpolate_na. it can use dask, but can take awhile
    to compute. method = half only interpolates times where all nan, and no
    other pixels will be interpolated in other images. 
    """

    # notify
    print('Interpolating empty values in dataset.')

    # check if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')

    # check if num years less than 3
    if len(ds['time.year']) < 3:
        raise ValueError('Less than 3 years worth of data in dataset. Add more.')

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # create copy ds
    ds = ds.copy(deep=True)

    # split into wet, dry - we dont want to fill wet with dry, vice versa
    ds_wet = ds.where(ds['time.month'].isin(wet_months), drop=True)
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

    if method == 'full':
        print('Interpolating using full method. This can take awhile. Please wait.')
        
        # if wet dask, rechunk to avoid core dimension error
        if bool(ds_wet.chunks):
            wet_chunks = ds_wet.chunks
            ds_wet = ds_wet.chunk({'time': -1}) 
            
        # if dry dask, rechunk to avoid core dimension error
        if bool(ds_dry.chunks):
            dry_chunks = ds_dry.chunks
            ds_dry = ds_dry.chunk({'time': -1}) 
 
        # interpolate all nan pixels linearly
        ds_wet = ds_wet.interpolate_na(dim='time', method='linear')
        ds_dry = ds_dry.interpolate_na(dim='time', method='linear')
        
        # chunk back to orginal size
        if bool(ds_wet.chunks):
            ds_wet = ds_wet.chunk(wet_chunks) 
            
        if bool(ds_dry.chunks):
            ds_dry = ds_dry.chunk(dry_chunks)

    elif method == 'half':
        print('Interpolating using half method. Please wait.')

        # get times where all nan
        nan_dates_wet = ds_wet.dropna(dim='time', how='all').time
        nan_dates_dry = ds_dry.dropna(dim='time', how='all').time

        # get wet, dry time differences with original ds
        nan_dates_wet = np.setdiff1d(ds_wet['time'], nan_dates_wet)
        nan_dates_dry = np.setdiff1d(ds_dry['time'], nan_dates_dry)
        
        # drop all nan in ds wet, dry
        ds_wet = ds_wet.dropna(dim='time', how='all')
        ds_dry = ds_dry.dropna(dim='time', how='all')

        # interpolate wet all nan pixels linearly
        if len(nan_dates_wet):
            ds_wet_interp = ds_wet.interp(time=nan_dates_wet, method='linear')
            ds_wet = xr.concat([ds, ds_wet_interp], dim='time').sortby('time')

        # interpolate wet all nan pixels linearly
        if len(nan_dates_dry):
            ds_dry_interp = ds_dry.interp(time=nan_dates_dry, method='linear')
            ds_dry = xr.concat([ds, ds_dry_interp], dim='time').sortby('time')

    # concat wet, dry datasets back together
    ds = xr.concat([ds_wet, ds_dry], dim='time').sortby('time')


    # convert back to datarray
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Interpolated empty values successfully.')
    return ds   


# meta, does it need da to ds check?
def standardise_to_targets(ds, dry_month=None, q_upper=0.99, q_lower=0.05):
    """
    standardises all times to dry season invariant targets
    q_upper is used to set percentile of greennest/moistest pixels
    q_lower is used to set percentile of lowest stability pixels (most stable)
    """

    # notify
    print('Standardising data using invariant targets.')
    
    # check if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')

    # check q_value 0-1
    if q_upper < 0 or q_upper > 1:
        raise ValueError('Upper quantile value must be between 0 and 1.')

    # check q_value 0-1
    if q_lower < 0 or q_lower > 1:
        raise ValueError('Lower quantile value must be between 0 and 1.')
        
    # check wet dry list, convert if not
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

    # create copy ds and take attrs
    ds = ds.copy(deep=True)
    attrs = ds.attrs

    # split into wet, dry - we dont want to fill wet with dry, vice versa
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)

    # get median all time for wet, dry
    ds_dry_med = ds_dry.median('time', keep_attrs=True)

    # notify
    print('Getting Generating invariant targets.')

    # get upper n quantile (i.e., percentiles) of dry vege, moist
    ds_quants = ds_dry_med.quantile(q=q_upper, skipna=True)
    ds_quants = xr.where(ds_dry_med > ds_quants, True, False)

    # get num times
    num_times = len(ds_dry['time'])

    # get linear ortho poly coeffs, sum squares, constant, reshape 1d to 3d
    coeffs, ss, const = tools.get_linear_orpol_contrasts(num_times)
    coeffs = np.reshape(coeffs, (ds_dry['time'].size, 1, 1)) # todo dynamic?

    # calculate dry linear slope, mask to greenest/moistest
    ds_poly = abs((ds_dry * coeffs).sum('time') / ss * const)
    ds_poly = ds_poly.where(ds_quants)

    # get lowest stability areas in stable, most veg, moist
    ds_targets = xr.where(ds_poly < ds_poly.quantile(q=q_lower, skipna=True), True, False)

    # check if any targets exist
    for var in ds_targets:
        is_empty = ds_targets[var].where(ds_targets[var]).isnull().all()
        if is_empty:
            raise ValueError('No invariant targets created: increase lower quantile.')

    # notify
    print('Standardising to invariant targets and rescaling via increasing sigmoidal.')

    # get low, high inflections via hardcoded percentile, do inc sigmoidal
    li = ds.median('time').quantile(q=0.001, skipna=True)
    hi = ds.where(ds_targets).quantile(dim=['x', 'y'], q=0.99, skipna=True)
    ds = np.square(np.cos((1 - ((ds - li) / (hi - li))) * (np.pi / 2)))
    
    # drop quantile tag the method adds, if exists
    ds = ds.drop('quantile', errors='ignore')
    
    # add attributes back on
    ds.attrs.update(attrs)
    
    # convert back to datarray
    if was_da:
        ds = ds.to_array()

    # notify and return
    print('Standardised using invariant targets successfully.')
    return ds


# meta
def calc_seasonal_similarity(ds, wet_month=None, dry_month=None, q_mask=0.9):
    """
    """
    
    # notify
    print('Calculating seasonal similarity.')

    # check if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

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

    # rescale from -1 to 1 to 0 to 2
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
    
    # convert back to datarray
    if was_da:
        ds = ds.to_array()
    
    # notify and return
    print('Calculated seasonal similarity successfully.')
    return ds_similarity


#meta, check ds sim
def calc_likelihood(ds, ds_similarity, wet_month=None, dry_month=None):
    """
    ds holds veg and moist
    ds sim holds similarity
    """
    
    # notify
    print('Generating groundwater-dependent vegetation (GDV) model.')

    # check if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')

    # check for time dim
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimension detected.')

    # check for x and y dims
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')

    # check vars are veg, mst idx exist
    for var in ds:
        if var not in ['veg_idx', 'mst_idx']:
            raise ValueError('Vegetation and/or moisture variable missing.')

    # check wet dry list, convert if not
    wet_months = wet_month if isinstance(wet_month, list) else [wet_month]
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]

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
        ds = ds.to_array()

    # notify and return
    print('Generated groundwater-dependent vegetation model successfully')
    return ds_likelihood



# meta
def threshold_xr_via_auc(ds, df, res_factor=3, if_nodata='any'):
    """
    """
    
    # notify
    print('Thresholding dataset via occurrence records and AUC.')
    
    # check if dataframe is pandas
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Occurrence records is not a pandas type.')
        
    #  check if x, y, actual fields in df
    if 'x' not in df or 'y' not in df:
        raise ValueError('No x, y fields in occurrence records.')
    elif 'actual' not in df:
        raise ValueError('No actual field in occurrence records.')     
    
    # check if dataset is xarray
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset is not an xarray type.')    
    
    # check if x, y, like dims/vars in ds
    if 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dims in dataset.')
    elif 'like' not in list(ds.data_vars) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dims in dataset.')
        
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

    # concat array back together
    if len(thresh_list) > 1:
        ds_thresh = xr.concat(thresh_list, dim='time').sortby('time')
    else:
        ds_thresh = thresh_list[0]
        
    # notify and return
    print('Thresholded dataset successfully.')
    return ds_thresh



# meta
def threshold_xr_via_std(ds, num_stdevs=3):
    """
    """
    
    # notify
    print('Thresholding dataset via standard deviation.')
    
    # check if xr
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset is not an xarray type.')
        
    # check for x and y dims
    if 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
        raise ValueError('No x or y dimensions detected.')
    
    # check num_stdv > 0 and <= 5
    if num_stdevs <= 0 or num_stdevs > 5:
        raise ValueError('Number of standard devs must be > 0 and <= 5.')
        
    # copy ds
    ds = ds.copy(deep=True)
          
    # calculate n stand devs and apply threshold
    ds_thresh = ds.mean(['x', 'y']) + (ds.std(['x', 'y']) * num_stdevs)
    ds = ds.where(ds > ds_thresh)
    
    # notify and return
    print('Thresholded dataset successfully.')
    return ds



# meta
def threshold_likelihood(ds, df=None, num_stdevs=3, res_factor=3, if_nodata='any'):
    """
    """

    # notify
    print('Thresholding groundwater-dependent vegeation likelihood.')

    # check if array provided, if so, convert to dataset
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')
    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')
        
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')

    # check num_stdevs > 0 and <= 5
    if num_stdevs <= 0 or num_stdevs > 5:
        raise ValueError('Number of standard deviations must be > 0 and <= 5.')

    # if records given, thresh via auc, else standard dev
    if df is not None:
        try:
            # attempt roc auc thresholding
            ds_thresh = threshold_xr_via_auc(ds=ds,
                                             df=df, 
                                             res_factor=res_factor, 
                                             if_nodata=if_nodata)
        except Exception as e:
            # notify and attempt thresholding via stdv
            print('Could not threshold via occurrence records. Trying standard dev.')
            print(e)
            ds_thresh = threshold_xr_via_std(ds, num_stdevs=num_stdevs)

    else:
        # attempt roc standard dev thresholding
        ds_thresh = threshold_xr_via_std(ds, num_stdevs=num_stdevs)

    # convert back to datarray
    if was_da:
        ds = ds.to_array()

    # notify
    print('Thresholded likelihood succuessfully.')
    return ds_thresh



# meta
def perform_mk_original(ds, pvalue, direction):
    """
    ds : xarray dataset or array
    pvalue = set none for all
    direction = trend direction. (inc, dec or both)
    """
    
    # notify user
    print('Performing Mann-Kendall test (original).')
    
    # check if xarray dataset type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Must provided an xarray dataset or array.')
                
    # check if time dim exists
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimensions in dataset.')
        
    # check if 3 or more times
    if len(ds['time']) < 3:
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
    
    # define original mk function
    def mk(x, y, p, d, nd):

        # check nans
        nans = np.isin(x, nd) | np.isnan(x)
        if np.all(nans):
            return nd

        # remove nans
        x, y = x[~nans], y[~nans]

        # count finite values, abort if 3 or less
        num_fin = np.count_nonzero(x)
        if num_fin <= 3:
            return nd

        # perform original mk
        tau, pvalue = kendalltau(x=x, y=y, nan_policy='omit')

        # if p given and its not sig, bail
        if p and pvalue >= p:
            return nd

        # check direction
        if d == 'both':
            return tau
        elif d == 'inc' and tau > 0:
            return tau
        elif d == 'dec' and tau < 0:
            return tau
        else: 
            return nd

    # generate mk
    ds_mk = xr.apply_ufunc(
        mk, ds,
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
    
    return ds_mk



# meta
def perform_theilsen_slope(ds, alpha):
    """
    ds : xarray dataset or array
    alpha = confidence
    """
    
    # notify user
    print('Performing Theil-Sen slope (original).')
    
    # check if xarray dataset type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Must provided an xarray dataset or array.')
                
    # check if time dim exists
    if 'time' not in list(ds.dims):
        raise ValueError('No time dimensions in dataset.')
        
    # check if 3 or more times
    if len(ds['time']) < 3:
        raise ValueError('More than 2 years required for analysis.')
        
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')
        
    # check if p is valid, if provided
    if alpha and (alpha < 0 or alpha > 1):
        raise ValueError('Alpha must be between 0 and 1.')   
        
    # define ts function (note: y, x different to mk)
    def ts(y, x, a, nd):

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


    # create ufunc to wrap mk in
    ds_ts = xr.apply_ufunc(
        ts, ds,
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
    
    return ds_ts


# meta, might need to code a persist
def perform_cva(ds, base_times=None, comp_times=None, reduce_comp=False, 
                vege_var='tcg', soil_var='tcb', tmf=2):
    """
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
    
    # define cva here
    def cva(ds_base, ds_comp, vege_var='tcg', soil_var='tcb', tmf=2):

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
        
    # may need persist to prevent multiple computes # todo
                
    # loop each ds in list and do cva
    ds_cva_list = []
    for i, da_comp in enumerate(ds_comp_list):
        
        # notify
        print('Doing CVA: {0}.'.format(i + 1))
        
        # do cva!
        ds_cva = cva(ds_base=ds_base, 
                     ds_comp=da_comp, 
                     vege_var=vege_var, 
                     soil_var=soil_var, 
                     tmf=tmf)
        
        # add to list
        ds_cva_list.append(ds_cva)
        
    # check if list, concat
    if ds_cva_list:
        ds_cva = xr.concat(ds_cva_list, dim='time')
        
    # notify and return
    print('Performed CVA successfully.')
    return ds_cva


# meta, just do double check of output
def isolate_cva_change(ds, angle_min=90, angle_max=180):
    """
    """
    
    # notify
    print('Isolating CVA angles from {0}-{1} degrees.'.format(angle_min, angle_max))
    
    # check if xr types
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset not an xarray type.')

    # get vars in ds
    ds_vars = []
    if isinstance(ds, xr.Dataset):
        ds_vars = list(ds.data_vars)
    elif isinstance(ds, xr.DataArray):
        ds_vars = list(ds['variable'])
        
    # check if angle is a var
    if 'angle' not in ds_vars:
        raise ValueError('No variable called angle provided.')
            
    # create copy
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
    
    # notify and return
    print('Isolated CVA angles successfully.')
    return ds