# contaisn functionality to set columns to return. never used.
def subset_records_data(df_records, p_a_column=None, keep_fields=['x', 'y']):
    """
    takes a pandas dataframe with all columns and limits it down 
    to occurrence field set by user, plus x, y columns (optional).
    if no occurrence column name set, all points considered to be
    presence (1).

    keep_fields : string, list
        Specify names of fields in dataframe to retain on output. Leave empty 
        to remove all dataframe fields.
    """

    # notify
    print('Subsetting records from dataframe.')

    # check if pandas dataframe
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Records is not a pandas dataframe type.')

    # check if any records exist
    if df_records.shape[0] == 0:
        raise ValueError('No records in pandas dataframe.')

    # check if x, y in dataframe
    if 'x' not in df_records or 'y' not in df_records:
        raise ValueError('No x, y columns in dataframe.')

    # get dataframe columns (lower case)
    columns = [c.lower() for c in list(df_records)]

    # prepare and check keep fields request
    if keep_fields:
        keep_fields = keep_fields if isinstance(keep_fields, list) else [keep_fields]
        for field in keep_fields:
            if field.lower() not in columns:
                raise ValueError('Requested field: {0} not in dataframe'.format(field))
    else:
        keep_fields = []

    # check if pres abse column exists
    if p_a_column and p_a_column not in df_records:
        raise ValueError('Presence/absence column: {0} not in dataframe.'.format(p_a_column))

    # create copy of dataframe
    df_records = df_records.copy()

    # subset dataframe depending on user requests
    if p_a_column:
        if keep_fields:
            df_records = df_records[keep_fields + [p_a_column]]
        else:
            df_records = df_records[[p_a_column]]
    else:
        df_records = df_records[['x', 'y']]

    # rename requested occurrence column, if exists
    if p_a_column in df_records:
        df_records = df_records.rename(columns={p_a_column: 'actual'})

    # remove non 1, 0 values
    if 'actual' in df_records:

        # get original record count
        old_num = df_records.shape[0]

        # remove non-binary values
        df_records = df_records[df_records['actual'].isin([1, 0])]

        # compare old to new
        if old_num > df_records.shape[0]:
            num_diff = old_num - df_records.shape[0]
            print('Warning: {0} records were not 1 or 0 and removed.'.format(num_diff))

    # check if final dataframe isnt empty
    if df_records.shape[0] == 0:
        raise ValueError('No occurrence records obtained after cleaning.')

    # notify and return
    print('Subset records successfully.')
    return df_records

# contaisn functionality to set columns to return. never used.
def extract_xr_to_records(ds, df_records, keep_fields=['x', 'y'], res_factor=3, nodata_value=-9999):
    """
    Takes a pandas dataframe of occurrence records and clips them
    to a xarray dataset or array. Existing fields in the dataframe can
    be saved in output if field name is placed in keep_fields parameter.

    Parameters
    ----------
    ds: xarray dataset
        A dataset with data variables.
    df_records : pandas dataframe
        A pandas dataframe containing at least an x and y columns 
        with records.
    keep_fields : string, list
        Specify names of fields in dataframe to retain on output. Leave empty 
        to remove all dataframe fields.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.
    nodata_value : int or float
        A int or float indicating the no dat avalue expected. Default is -9999.

    Returns
    ----------
    df_records : pandas dataframe
    """

    # notify user
    print('Clipping pandas dataframe records to xarray dataset.')

    # check if coords is a pandas dataframe
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Records is not a pandas dataframe type.')
        
    # get dataframe columns (lower case)
    columns = [c.lower() for c in list(df_records)]
    
    # check if x, y exists in df
    if 'x' not in columns or 'y' not in columns:
        raise ValueError('No x, y values in records.')
        
    # check if any records exist
    if df_records.shape[0] == 0:
        raise ValueError('No records in pandas dataframe.')
        
    # prepare and check keep fields request
    if keep_fields:
        keep_fields = keep_fields if isinstance(keep_fields, list) else [keep_fields]
        for field in keep_fields:
            if field.lower() not in columns:
                raise ValueError('Requested field: {0} not in dataframe.'.format(field))
    else:
        keep_fields = []   
        
    # check if dataset, if array, convert
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')

    # check if x, y dims exist in ds
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x, y coordinate dimension in dataset.') 

    # check if ds is 2d
    if len(ds.dims) != 2:
        raise ValueError('Number of dataset dimensions not equal to 2.') 

    # check if res factor is int type
    if not isinstance(res_factor, int) and res_factor < 1:
        raise TypeError('Resolution factor must be an integer of 1 or greater.')

    # get cell resolution from dataset
    res = get_xr_resolution(ds)

    # check res exists
    if not res:
        raise ValueError('No resolution extracted from dataset.')

    # multiply res by res factor
    res = res * res_factor

    # create copies of ds and df
    df_records = df_records.copy()
    ds = ds.copy(deep=True)

    # loop through data var and extract values at coords
    values = []
    for i, row in df_records.iterrows():
        try:            
            # get values from vars at current pixel
            pixel = ds.sel(x=row.get('x'), y=row.get('y'), method='nearest', tolerance=res)
            pixel = pixel.to_array().values
            pixel = list(pixel)

        except:
            # fill with list of nan equal to data var size
            pixel = [nodata_value] * len(ds.data_vars)

        # add current point x and y columns if wanted
        if keep_fields:
            pixel = row[keep_fields].to_list() + pixel

        # append to list
        values.append(pixel)

    # get original df field dtypes as dict
    df_dtypes_dict = {}
    for var in keep_fields:
        df_dtypes_dict[var] = np.dtype(df_records[var].dtype)

    # get original ds var dtypes as dict
    ds_dtypes_dict = {}
    for var in ds.data_vars:
        ds_dtypes_dict[var] = np.dtype(ds[var].dtype)
        
    try:
        # get combined list of columns, dtypes
        col_names = keep_fields + list(ds.data_vars)
        col_dtypes = {**df_dtypes_dict, **ds_dtypes_dict}    

        # convert values list into pandas dataframe, ensure types match dataset
        df_records = pd.DataFrame(values, columns=col_names)
        df_records = df_records.astype(col_dtypes)

    except:
        raise ValueError('Could not create clipped data to pandas dataframe.')

    # convert back to datarray
    if was_da:
        ds = ds.to_array()

    # notify user and return
    print('Extracted xarray dataset values successfully.')
    return df_records

# old interpolation method for gdvspectra
def interpolate_empty_wet_dry(ds, wet_month=None, dry_month=None, method='full'):
    """
    Takes a xarray dataset/array and wet, dry months as lists. For wet and 
    dry times, nan values are interpolated along season time dimension, i.e. 
    from each wet season time to wet season time. This is to prevent interpolation 
    from wet to dry or vice versa. The method can be set to full or half. Full 
    will use the built in interpolate_na method, which is robust and dask friendly 
    but very slow. The quicker alternative is half, which only interpolates times 
    that are all nan. Despite being slower, full method recommended.

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

# old standardiser for gdvspectra
def standardise_to_dry_targets(ds, dry_month=None, q_upper=0.99, q_lower=0.05, inplace=True):
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
        
    # check q_value 0-1
    if q_upper < 0 or q_upper > 1:
        raise ValueError('Upper quantile value must be between 0 and 1.')

    # check q_value 0-1
    if q_lower < 0 or q_lower > 1:
        raise ValueError('Lower quantile value must be between 0 and 1.')
        
    # check wet, dry month if none given
    if dry_month is None:
        raise ValueError('Must provide at least one dry month.')   
        
    # check wet dry list, convert if not
    dry_months = dry_month if isinstance(dry_month, list) else [dry_month]
        
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
        
    # get attributes - we lose them 
    attrs = ds.attrs
        
    # split into dry and get all time dry season median
    ds_dry = ds.where(ds['time.month'].isin(dry_months), drop=True)
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
    
    # notify and return
    print('Standardised using invariant targets successfully.')
    return ds

# old standardise targets func
def standardise_to_targets(ds, q_upper=0.99, q_lower=0.05):
    """
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

    # create copy ds and take attrs
    ds = ds.copy(deep=True)
    attrs = ds.attrs

    # get median all time
    ds_med = ds.median('time', keep_attrs=True)

    # notify
    print('Generating invariant targets.')

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