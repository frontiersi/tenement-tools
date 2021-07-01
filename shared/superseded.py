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

# old validate rasters method. now implemented in load_nc
def validate_rasters(rast_path_list):
    """
    Compare all input rasters and ensure geo transformations, coordinate systems,
    size of dimensions (x and y) number of features, and nodata values all match. 
    Takes a list of paths to raster layers to be used in analysis.

    Parameters
    ----------
    rast_path_list : string
        A single list of strings with full path and filename of input rasters.
    """

    # notify user
    print('Checking raster spatial information to check for inconsistencies.')

    # check if shapefile and raster paths exists
    if not rast_path_list:
        raise ValueError('> No raster path list provided.')

    # check if list types, if not bail
    if not isinstance(rast_path_list, list):
        raise ValueError('> Raster path list must be a list.')

    # ensure raster paths in list exist and are strings
    for path in rast_path_list:

        # check if string, if not bail
        if not isinstance(path, str):
            raise ValueError('> Raster paths must be a string.')

        # check if shapefile or raster exists
        if not os.path.exists(path):
            raise OSError('> Unable to read raster, file not found.')

    # notify
    print('> Extracting raster spatial information.')

    # loop through each rast, extract info, store in list
    rast_dict_list = []
    for rast_path in rast_path_list:
        rast_dict = extract_rast_info(rast_path)
        rast_dict_list.append(rast_dict)

    # check if anything in output lists
    if not rast_dict_list:
        raise ValueError('> No Raster spatial information in outputs.')

    # epsg - get values and invalid layers
    epsg_list, no_epsg_list = [], []
    for info_dict in rast_dict_list:
        epsg_list.append(info_dict.get('epsg'))
        if not info_dict.get('epsg'):
            no_epsg_list.append(info_dict.get('layer'))

    # is_projected - get values and invalid layers
    is_proj_list, no_proj_list = [], []
    for info_dict in rast_dict_list:
        is_proj_list.append(info_dict.get('is_projected'))
        if not info_dict.get('is_projected'):
            no_proj_list.append(info_dict.get('layer'))

    # x_dim - get values and invalid layers
    x_dim_list, no_x_dim = [], []
    for info_dict in rast_dict_list:
        x_dim_list.append(info_dict.get('x_dim'))
        if not info_dict.get('x_dim') > 0:
            no_x_dim.append(info_dict.get('layer'))        

    # y_dim - get values and invalid layers
    y_dim_list, no_y_dim = [], []
    for info_dict in rast_dict_list:
        y_dim_list.append(info_dict.get('y_dim'))
        if not info_dict.get('y_dim') > 0:
            no_y_dim.append(info_dict.get('layer'))

    # nodata_val - get values and invalid layers
    nodata_list = []
    for info_dict in rast_dict_list:
        nodata_list.append(info_dict.get('nodata_val'))

    # notify - layers where epsg code missing 
    if no_epsg_list:
        print('> These layers have an unknown coordinate system: {0}.'.format(', '.join(no_epsg_list)))
    if not all([e == epsg_list[0] for e in epsg_list]):
        print('> Inconsistent coordinate systems between layers. Could cause errors.')

    # notify - layers where not projected
    if no_proj_list:
        print('> These layers are not projected: {0}. Must be projected.'.format(', '.join(no_proj_list)))
    if not all([e == is_proj_list[0] for e in is_proj_list]):
        print('> Not all layers projected. All layers must have a projection system.')

    # notify - layers where not x_dim
    if no_x_dim:
        print('> These layers have no x dimension: {0}. Must have.'.format(', '.join(no_x_dim)))
    if not all([e == x_dim_list[0] for e in x_dim_list]):
        print('> Inconsistent x dimensions between layers. Must be consistent.')    

    # notify - layers where not y_dim
    if no_y_dim:
        print('> These layers have no y dimension: {0}. Must have.'.format(', '.join(no_y_dim)))
    if not all([e == y_dim_list[0] for e in y_dim_list]):
        print('> Inconsistent y dimensions between layers. Must be consistent.')      

    # notify - layers where nodata
    if not all([e == nodata_list[0] for e in nodata_list]):
        print('> Inconsistent NoData values between layers. Could cause errors.'.format(', '.join(no_feat_count_list)))

    # raise error if any vital errors detected
    if no_epsg_list or no_proj_list or no_x_dim or no_y_dim:
        raise ValueError('> Errors found in layers (read above). Please fix and re-run the tool.')

# retired
def perform_optimised_fit(estimator, X, y, parameters, cv=10):
    """
    Takes an estimator, values (independent vars) with response
    (y), as well as gridcv parameters. Output is a fit, optimal
    model.

    Parameters
    ----------
    estimator : sklearn estimastor object
        A sklearn estimator.
    X : numpy ndarray
        A numpy array of dependent values.
    y : numpy ndarray
        A numpy array of response values.
    parameters : list
        Parameters for use in GridSearchCV.
    cv : int
        The number of cross-validations.

    Returns
    ----------
    gsc_result: numpy ndarray
        Output from GridSearchCV of fit values.
    """
    
    # imports
    from sklearn.model_selection import GridSearchCV
    
    # check for tyoe
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('X and/or y inputs must be numpy arrays.')
    elif not len(X.shape) == 2 or not len(y.shape) == 1:
        raise ValueError('X and/or y inputs are of incorrect size.')
                
    # check parameters
    if not isinstance(parameters, dict):
        raise TypeError('Parameters must be in a dictionary type.')
        
    # check entered parameters
    allowed_parameters = ['max_features', 'max_depth', 'n_estimators']
    for k, p in parameters.items():
        if k not in allowed_parameters:
            raise ValueError('Parameter: {0} not supported.'.format(k))
            
    # check cv
    if cv <= 0:
        raise ValueError('> CV (cross-validation) must be > 0.')
        
    # create grid search cv and fit it
    gsc = GridSearchCV(estimator, parameters, cv=cv, n_jobs=-1, scoring='max_error')
    gsc_result = gsc.fit(X, y)
        
    return gsc_result

# retired
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
    
    # imports
    import dask.array as dask_array
    from sklearn.ensemble import RandomForestRegressor
    from dask_ml.wrappers import ParallelPostFit
    import joblib
    
    # check ds in dataset or dataarray
    if not isinstance(ds_input, (xr.Dataset, xr.DataArray)):
        raise TypeError('> Input dataset is not xarray dataset or data array type.')
    
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

    # set up function for random forest prediction
    def predict(ds_input, estimator):

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
            data = da.data.flatten().rechunk(chunk_size)
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

    # predict via parallel, or if missing, regular compute
    if is_dask == True:
        estimator = ParallelPostFit(estimator)
        with joblib.parallel_backend('dask'):
            ds_out = predict(ds_input, estimator)
    else:
        ds_out = predict(ds_input, estimator).compute()

    # return
    return ds_out

# retired
def perform_optimised_validation(estimator, X, y, n_validations=50, split_ratio=0.10):
    """
    
    """
    
    # imports
    from sklearn.model_selection import train_test_split
    
    # check for tyoe
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('X and/or y inputs must be numpy arrays.')
    elif not len(X.shape) == 2 or not len(y.shape) == 1:
        raise ValueError('X and/or y inputs are of incorrect size.')
        
    # check validations, split
    if n_validations < 1:
        raise ValueError('Number of validations must be > 0.')
    elif split_ratio < 0 or split_ratio > 1:
        raise ValueError('SPlit ratio must be between 0 and 1.')
        
    # create array to hold validation results
    r2_list = []
    
    # iterate n validations
    for i in range(0, n_validations):
        
        # split X and y data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=split_ratio, 
                                                            random_state=100, 
                                                            shuffle=True)
        
        # fit model using pre-optimised estimator
        model = estimator.fit(X_train, y_train)
        
        # calc score and append 
        score = model.score(X_test, y_test)
        r2_list.append(score)
        
        
    # calc mean r2
    r2 = np.array(r2_list).mean()
    r2 = abs(round(r2, 2)) 
    
    # reset if model outside 0 - 1 (highly inaccurate)
    if r2 > 1:
        r2 = 0.0
    
    # return
    return r2
    
# retired
def perform_fca(ds_raw, ds_class, df_data, df_extract_clean, grid_params, validation_iters, nodata_value=-9999):
    """
    """
    
    # imports
    from sklearn.ensemble import RandomForestRegressor
    
    # notify
    print('Beginning fractional cover analysis (FCA). ')
    
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
        print('> Fitting and predicting model for class: {0}'.format(c))

        # exclude flagged rows from analysis
        df_data_sub = df_data.loc[df_data['include'] == True]

        # combine dataframes to align samples
        df_merged = pd.merge(df_data_sub, df_extract_clean, on=['x', 'y'])

        # get independent vars out, excluding x and y, convert to numpy
        indep_cols = df_extract_clean.drop(columns=['x', 'y']).columns
        X = df_merged[indep_cols].to_numpy()

        # get dependent var (class col), convert to flatten 1d numpy
        y = df_merged[[c]].to_numpy().flatten()

        # create a new random forest regressor
        estimator = RandomForestRegressor()

        # do optimised fit, get best estimator, make unfitted copy for validation
        grid = perform_optimised_fit(estimator, X, y, grid_params)
        estimator = grid.best_estimator_
        
        # predict onto raw dataset, rename and append
        ds_pred = perform_prediction(ds_raw, estimator)
        ds_pred = ds_pred.rename({'result': str(c)})
        pred_list.append(ds_pred)

        # cross validate model for mean r2
        r2 = perform_optimised_validation(estimator, X, y, validation_iters)
        print('> Mean r-squared for model: {0}.'.format(r2))

    # merge result together
    ds_preds = xr.merge(pred_list)   

    # notify and return
    print('Fractional cover analysis (FCA) completed successfully.')
    return ds_preds

# retired
def validate_input_data(shp_path_list, rast_path_list):
    """
    Compare all input shapefiles and rasters and ensure geo transformations, coordinate systems,
    size of dimensions (x and y) number of features, and nodata values all match. Takes two lists of
    paths to shapefile and raster layers to be used in analysis.

    Parameters
    ----------
    shp_path_list : string
        A single list of strings with full path and filename of input shapefiles.
    rast_path_list : string
        A single list of strings with full path and filename of input rasters.
    """

    # notify user
    print('Comparing shapefile and raster spatial information to check for inconsistencies.')

    # check if shapefile and raster paths exists
    if not shp_path_list or not rast_path_list:
        raise ValueError('> No shapefile or raster path list provided.')

    # check if list types, if not bail
    if not isinstance(shp_path_list, list) or not isinstance(rast_path_list, list):
        raise ValueError('> Shapefile or raster path list must be a list.')

    # ensure shapefile and raster paths in list exist and are strings
    for path in shp_path_list + rast_path_list:

        # check if string, if not bail
        if not isinstance(path, str):
            raise ValueError('> Shapefile and raster paths must be a string.')

        # check if shapefile or raster exists
        if not os.path.exists(path):
            raise OSError('> Unable to read shapefile or raster, file not found.')

    # notify
    print('> Extracting shapefile spatial information.')

    # loop through each rast, extract info, store in list
    shp_dict_list = []
    for shp_path in shp_path_list:
        shp_dict = extract_shp_info(shp_path)
        shp_dict_list.append(shp_dict)

    # notify
    print('> Extracting raster spatial information.')

    # loop through each rast, extract info, store in list
    rast_dict_list = []
    for rast_path in rast_path_list:
        rast_dict = extract_rast_info(rast_path)
        rast_dict_list.append(rast_dict)

    # check if anything in output lists
    if not shp_dict_list or not rast_dict_list:
        raise ValueError('> No shapefile or raster spatial information in outputs.')

    # combine dict lists
    info_dict_list = shp_dict_list + rast_dict_list

    # epsg - get values and invalid layers
    epsg_list, no_epsg_list = [], []
    for info_dict in info_dict_list:
        epsg_list.append(info_dict.get('epsg'))
        if not info_dict.get('epsg'):
            no_epsg_list.append(info_dict.get('layer'))

    # is_projected - get values and invalid layers
    is_proj_list, no_proj_list = [], []
    for info_dict in info_dict_list:
        is_proj_list.append(info_dict.get('is_projected'))
        if not info_dict.get('is_projected'):
            no_proj_list.append(info_dict.get('layer'))

    # feat count - get values and invalid layers (vector only)
    no_feat_count_list = []
    for info_dict in shp_dict_list:
        if not info_dict.get('feat_count') > 0:
            no_feat_count_list.append(info_dict.get('layer'))

    # x_dim - get values and invalid layers
    x_dim_list, no_x_dim = [], []
    for info_dict in rast_dict_list:
        x_dim_list.append(info_dict.get('x_dim'))
        if not info_dict.get('x_dim') > 0:
            no_x_dim.append(info_dict.get('layer'))        

    # y_dim - get values and invalid layers
    y_dim_list, no_y_dim = [], []
    for info_dict in rast_dict_list:
        y_dim_list.append(info_dict.get('y_dim'))
        if not info_dict.get('y_dim') > 0:
            no_y_dim.append(info_dict.get('layer'))

    # nodata_val - get values and invalid layers
    nodata_list = []
    for info_dict in rast_dict_list:
        nodata_list.append(info_dict.get('nodata_val'))

    # notify - layers where epsg code missing 
    if no_epsg_list:
        print('> These layers have an unknown coordinate system: {0}.'.format(', '.join(no_epsg_list)))
    if not all([e == epsg_list[0] for e in epsg_list]):
        print('> Inconsistent coordinate systems between layers. Could cause errors.')

    # notify - layers where not projected
    if no_proj_list:
        print('> These layers are not projected: {0}. Must be projected.'.format(', '.join(no_proj_list)))
    if not all([e == is_proj_list[0] for e in is_proj_list]):
        print('> Not all layers projected. All layers must have a projection system.')

    # notify - layers where no features
    if no_feat_count_list:
        print('> These layers are empty: {0}. Must have features.'.format(', '.join(no_feat_count_list)))

    # notify - layers where not x_dim
    if no_x_dim:
        print('> These layers have no x dimension: {0}. Must have.'.format(', '.join(no_x_dim)))
    if not all([e == x_dim_list[0] for e in x_dim_list]):
        print('> Inconsistent x dimensions between layers. Must be consistent.')    

    # notify - layers where not y_dim
    if no_y_dim:
        print('> These layers have no y dimension: {0}. Must have.'.format(', '.join(no_y_dim)))
    if not all([e == y_dim_list[0] for e in y_dim_list]):
        print('> Inconsistent y dimensions between layers. Must be consistent.')      

    # notify - layers where nodata
    if not all([e == nodata_list[0] for e in nodata_list]):
        print('> Inconsistent NoData values between layers. Could cause errors.'.format(', '.join(no_feat_count_list)))

    # raise error if any vital errors detected
    if no_epsg_list or no_proj_list or no_feat_count_list or no_x_dim or no_y_dim:
        raise ValueError('> Errors found in layers (read above). Please fix and re-run the tool.')

# retired
def read_coordinates_shp(shp_path):
    """
    Read observation records from a projected ESRI Shapefile and extracts the x and y values
    located within. This must be a point geometry-type dataset and it must be projected.

    Parameters
    ----------
    shp_path: string
        A single string with full path and filename of shapefile.

    Returns
    ----------
    df_presence : pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """

    # notify user
    print('Reading species point locations from shapefile.')

    # check if string, if not bail
    if not isinstance(shp_path, str):
        raise ValueError('> Shapefile path must be a string. Please check the file path.')

    # check if shp exists
    if not os.path.exists(shp_path):
        raise OSError('> Unable to read species point locations, file not found. Please check the file path.')

    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()

        # check if projected (i.e. in metres)
        is_projected = lyr.GetSpatialRef().IsProjected()

        # get num feats
        num_feats = lyr.GetFeatureCount()

    except Exception:
        raise TypeError('> Could not read species point locations. Is the file corrupt?')

    # check if point/multi point type
    if lyr.GetGeomType() not in [ogr.wkbPoint, ogr.wkbMultiPoint]:
        raise ValueError('> Shapefile is not a point/multi-point type.')

    # check if shapefile is empty
    if num_feats == 0:
        raise ValueError('> Shapefile has no features in it. Please check.')

    # check if shapefile is projected (i.e. in metres)
    if not is_projected:
        raise ValueError('> Shapefile is not projected. Please project into local grid (we recommend MGA or ALBERS).')

    # loop through feats
    coords = []
    for feat in lyr:
        geom = feat.GetGeometryRef()

        # get x and y of individual point type
        if geom.GetGeometryName() == 'POINT':
            coords.append([geom.GetX(), geom.GetY()])

        # get x and y of each point in multipoint type
        elif geom.GetGeometryName() == 'MULTIPOINT':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)   
                coords.append([sub_geom.GetX(), sub_geom.GetY()])

        # error, a non-point type exists
        else:
            raise TypeError('> Unable to read point location, geometry is invalid.')

    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_presence = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    shp, lyr = None, None

    # notify user and return
    print('> Species point presence observations loaded successfully.')
    return df_presence   

# retired - might use one day
def read_mask_shp(shp_path):
    """
    Read mask from a projected ESRI Shapefile. This must be a polygon geometry-type 
    dataset and it must be projected.
    
    Parameters
    ----------
    shp_path: string
        A single string with full path and filename of shapefile.

    Returns
    ----------
    mask_geom : ogr vector geometry
        An ogr vector of type multipolygon geometry with a single feature (dissolved).
    """
    
    # notify user
    print('Reading mask polygon(s) from shapefile.')
    
    # check if string, if not bail
    if not isinstance(shp_path, str):
        raise ValueError('> Shapefile path must be a string. Please check the file path.')
    
    # check if shp exists
    if not os.path.exists(shp_path):
        raise OSError('> Unable to read mask polygon, file not found. Please check the file path.')
    
    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()
        
        # get spatial ref system
        srs = lyr.GetSpatialRef()

        # get num feats
        num_feats = lyr.GetFeatureCount()

    except Exception:
        raise TypeError('> Could not read mask polygon(s). Is the file corrupt?')
        
    # check if point/multi point type
    if lyr.GetGeomType() not in [ogr.wkbPolygon, ogr.wkbMultiPolygon]:
        raise ValueError('> Shapefile is not a polygon/multi-polygon type.')
        
    # check if shapefile is empty
    if num_feats == 0:
        raise ValueError('> Shapefile has no features in it. Please check.')
        
    # check if shapefile is projected (i.e. in metres)
    if not srs.IsProjected():
        raise ValueError('> Shapefile is not projected. Please project into local grid (we recommend MGA or ALBERS).')
          
    # notify
    print('> Compiling geometry and dissolving polygons.')
    
    # loop feats
    mask_geom = ogr.Geometry(ogr.wkbMultiPolygon)
    for feat in lyr:
        geom = feat.GetGeometryRef()
        
        # add geom if individual polygon type
        if geom.GetGeometryName() == 'POLYGON':
            mask_geom.AddGeometry(geom)
            
        # add geom if multi-polygon type
        elif geom.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)
                mask_geom.AddGeometry(sub_geom)
        
        # error, a non-polygon type exists
        else:
            raise TypeError('> Unable to read polygon, geometry is invalid.')
            
    # union all features together (dissolve)
    mask_geom = mask_geom.UnionCascaded()
            
    # check if mask geom is populated
    if not mask_geom or not mask_geom.GetGeometryCount() > 0:
        raise ValueError('> No polygons exist in dissolved mask. Check mask shapefile.')
        
    # drop variables
    shp, lyr, srs = None, None, None
    
    # notify user and return
    print('> Mask polygons loaded and dissolved successfully.')
    return mask_geom

# retired
def rasters_to_dataset(rast_path_list, nodata_value=-9999):
    """
    Read a list of rasters (e.g. tif) and convert them into an xarray dataset, 
    where each raster layer becomes a new dataset variable.

    Parameters
    ----------
    rast_path_list: list
        A list of strings with full path and filename of a raster.
    nodata_value : int or float
        A int or float indicating the no dat avalue expected. Default is -9999.

    Returns
    ----------
    ds : xarray Dataset
    """
    
    # notify user
    print('Converting rasters to an xarray dataset.')
    
    # check if raster exists
    if not rast_path_list:
        raise ValueError('> No raster paths in list. Please check list.')    
    
    # check if list, if not bail
    if not isinstance(rast_path_list, list):
        raise ValueError('> Raster path list must be a list.')
        
    # ensure raster paths in list exist and are strings
    for rast_path in rast_path_list:

        # check if string, if not bail
        if not isinstance(rast_path, str):
            raise ValueError('> Raster path must be a string. Please check the file path.')

        # check if raster exists
        if not os.path.exists(rast_path):
            raise OSError('> Unable to read raster, file not found. Please check the file path.')
            
    # check if no data value is correct
    if type(nodata_value) not in [int, float]:
        raise TypeError('> NoData value is not an int or float.')
         
    # loop thru raster paths and convert to data arrays
    da_list = []
    for rast_path in rast_path_list:
        try:
            # get filename
            rast_filename = os.path.basename(rast_path)
            rast_filename = os.path.splitext(rast_filename)[0]

            # open raster as dataset, rename band to var, add var name 
            da = xr.open_rasterio(rast_path)
            da = da.rename({'band': 'variable'})
            da['variable'] = np.array([rast_filename])
            
            # check if no data val attributes exist, replace with -9999
            if da.attrs and da.attrs.get('nodatavals'):
                nds = da.attrs.get('nodatavals')

                # if iterable, iterate them and replace with -9999
                if type(nds) in [list, tuple]:
                    for nd in nds:
                        da = da.where(da != nd, nodata_value)

                # if single numeric, replace with -9999
                elif type(nds) in [float, int]:
                    da = da.where(da != nd, nodata_value)
        
                # couldnt figure it out, warn user
                else:
                    print('> Unknown NoData value. Check NoData of layer: {0}'.format(rast_path))

            # append to list
            da_list.append(da)
            
            # notify
            print('> Converted raster to xarray data array: {0}'.format(rast_filename))
            
        except Exception:
            raise IOError('Unable to read raster: {0}. Please check.'.format(rast_path))
            
    # check if anything came back, then proceed
    if not da_list:
        raise ValueError('> No data arrays converted from raster paths. Please check rasters are valid.')

    # combine all da together and create dataset
    try:
        ds = xr.concat(da_list, dim='variable')
        ds = ds.to_dataset(dim='variable')
        
    except Exception:
        raise ValueError('Could not concat data arrays. Check your rasters.')
              
    # notify user and return
    print('> Rasters converted to dataset successfully.\n')
    return ds

# retired - might use one day
def generate_absences_from_shp(mask_shp_path, num_abse, occur_shp_path=None, buff_m=None):
    """
    Generates pseudo-absence (random or pseudo-random) locations within provided mask polygon. 
    Pseudo-absence points are key in sdm work. A mask shapefile path and value for number
    of pseudo-absence points to generate is required. Optionally, if user provides an occurrence
    shapefile path and buffer length (in metres), proximity buffers can be also created around
    species occurrences - another often used function sdm work.

    Parameters
    ----------
    mask_shp_path: string
        A single string with full path and filename of shapefile of mask.
    num_abse: int
        A int indicating how many pseudo-absence points to generated.
    occur_shp_path : string (optional)
        A single string with full path and filename of shapefile of occurrence records.
    buff_m : int (optional)
        A int indicating radius of buffers (proximity zones) around occurrence points (in metres).
        
    Returns
    ----------
    df_absence: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """

    # notify user
    print('Generating {0} randomised psuedo-absence locations.'.format(num_abse))

    # check if string, if not bail
    if not isinstance(mask_shp_path, str):
        raise ValueError('> Mask shapefile path must be a string. Please check the file path.')

    # check if shp exists
    if not os.path.exists(mask_shp_path):
        raise OSError('> Unable to read mask shapefile, file not found. Please check the file path.')

    # check if number of absence points is an int
    if not isinstance(num_abse, int):
        raise ValueError('> Num of absence points value is not an integer. Please check the entered value.')

    # check if num of absence points is above 0
    if not num_abse > 0:
        raise ValueError('> Num of absence points must be > 0. Please check the entered value.')  

    # load mask polygon, dissolve it, output multipolygon geometry
    mask_geom = read_mask_shp(shp_path=mask_shp_path)

    # check if mask geom exists
    if not mask_geom or not mask_geom.GetGeometryCount() > 0:
        raise ValueError('> No polygons exist in dissolved mask. Check mask shapefile.')

    # erase proximities from mask if user provides values for it
    buff_geom = None
    if occur_shp_path and buff_m:

        # notify
        print('> Removing proximity buffer areas from mask area.')

        # read proximity buffer geoms
        buff_geom = generate_proximity_areas(occur_shp_path, buff_m)

        # check if proximity buffer geom exists
        if not buff_geom or not buff_geom.GetGeometryCount() > 0:
            raise ValueError('> No proximity buffer polygons exist in dissolved mask. Check mask shapefile.')

        try:
            # difference mask and proximity buffers
            diff_geom = mask_geom.Difference(buff_geom)

        except:
            raise ValueError('> Could not difference mask and proximity buffers.')

        # check if difference geometry exists
        if not diff_geom or not diff_geom.GetArea() > 0:
            raise ValueError('> Nothing returned from difference. Is your buffer length too high?') 
        
        # set mask_geom to buff_geom
        mask_geom = diff_geom
        
    # notify
    print('> Randomising absence points within mask area.')
    
    # get bounds of mask
    env = mask_geom.GetEnvelope()
    x_min, x_max, y_min, y_max = env[0], env[1], env[2], env[3]

    # create random points and fill a list with x and y
    counter = 0
    coords = []
    for i in range(num_abse):
        while counter < num_abse:
            
            # get random x and y coord
            rand_x = random.uniform(x_min, x_max)
            rand_y = random.uniform(y_min, y_max)

            # create point and add x and y to it
            pnt = ogr.Geometry(ogr.wkbPoint)
            pnt.AddPoint(rand_x, rand_y)
            
            # if point in mask, include it
            if pnt.Within(mask_geom):
                coords.append([pnt.GetX(), pnt.GetY()])
                counter += 1
        
    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')
        
    # convert coord array into dataframe
    df_absence = pd.DataFrame(coords, columns=['x', 'y'])
        
    # drop variables
    mask_geom, buff_geom, abse_geom = None, None, None
        
    # notify and return
    print('> Generated pseudo-absence points successfully.')
    return df_absence

# retired
def remove_nodata_records(df_records, nodata_value=-9999):
    """
    Read a numpy record array and remove -9999 (nodata) records.

    Parameters
    ----------
    df_records: pandas dataframe
        A pandas dataframe type containing values extracted from env variables.
    nodata_value : int or float
        A int or float indicating the no dat avalue expected. Default is -9999.

    Returns
    ----------
    df_records : numpy record array
        A numpy record array without nodata values.
    """

    # notify user
    print('Removing records containing NoData (-9999) values.')

    # check if numpy rec array
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('> Not a pands dataframe type.')

    # check if no data value is correct
    if type(nodata_value) not in [int, float]:
        raise TypeError('NoData value is not an int or float.')
        
    # get original num records
    orig_num_recs = df_records.shape[0]

    # remove any record containing nodata value
    df_records = df_records[~df_records.eq(nodata_value).any(axis=1)]

    # check if array exists
    if df_records.shape[0] == 0:
        raise ValueError('> No valid values were detected in dataframe - all were NoData.')
        
    # get num of records removed
    num_removed = orig_num_recs - df_records.shape[0]

    # notify user and return
    print('> Removed {0} records containing NoData values successfully.'.format(num_removed))
    return df_records

# retired
def extract_dataset_values(ds, coords, res_factor=3, nodata_value=-9999):
    """
    Read an xarray dataset and convert them into a numpy records array.

    Parameters
    ----------
    ds: xarray dataset
        A dataset with data variables.
    coords : pandas dataframe
        A pandas dataframe containing x and y columns with records.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.
    nodata_value : int or float
        A int or float indicating the no dat avalue expected. Default is -9999.

    Returns
    ----------
    df_presence_data : pandas dataframe
    """

    # notify user
    print('Extracting xarray dataset values to x and y coordinates.')

    # check if coords is a pandas dataframe
    if not isinstance(coords, pd.DataFrame):
        raise TypeError('> Provided coords is not a numpy ndarray type. Please check input.')

    # check if dataset type provided
    if not isinstance(ds, xr.Dataset):
        raise TypeError('> Provided dataset is not an xarray dataset type. Please check input.')

    # check dimensionality of pandas dataframe. x and y only
    if len(coords.columns) != 2:
        raise ValueError('Num of columns in coords not equal to 2. Please ensure shapefile is valid.')

    # check if res factor is int type
    if not isinstance(res_factor, int):
        raise TypeError('> Resolution factor must be an integer.')

    # check dimensionality of numpy array. xy only
    if not res_factor >= 1:
        raise ValueError('Resolution factor must be value of 1 or greater.')

    # get cell resolution from dataset
    res = get_dataset_resolution(ds)

    # check res exists
    if not res:
        raise ValueError('> No resolution extracted from dataset.')

    # multiply res by res factor
    res = res * res_factor

    # loop through data var and extract values at coords
    values = []
    for i, row in coords.iterrows():
        try:
            # get values from vars at current pixel, tolerence is to 2 pixels either side
            pixel = ds.sel(x=row.get('x'), y=row.get('y'), method='nearest', tolerance=res * res_factor)
            pixel = pixel.to_array().values
            pixel = list(pixel)

        except:
            # fill with list of nan equal to data var size
            pixel = [nodata_value] * len(ds.data_vars)

        # append to list
        values.append(pixel)

    try:
        # convert values list into pandas dataframe
        df_presence_data = pd.DataFrame(values, columns=list(ds.data_vars))

    except:
        raise ValueError('Errors were encoutered when converting data to pandas dataframe.')

    # notify user and return
    print('> Extracted xarray dataset values successfully.\n')
    return df_presence_data


# # # # # COG # # #

# meta, checks
def make_vrt_dataset_xml(x_size, y_size, axis_map, srs, trans):
    """
    take paramets for vrt and create a raster xml object
    """
    
    # imports
    from lxml import etree as et

    # set up root vrt dataset elem
    xml_ds = '<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}"></VRTDataset>'
    xml_ds = et.fromstring(xml_ds.format(x_size=x_size, y_size=y_size))

    # set up srs element and add to vrt dataset
    xml_srs = '<SRS dataAxisToSRSAxisMapping="{axis_map}">{srs}</SRS>'
    xml_ds.append(et.fromstring(xml_srs.format(axis_map=axis_map, srs=srs)))

    # set up geo transform element and add to vrt dataset
    xml_trans = '<GeoTransform>{trans}</GeoTransform>'
    xml_ds.append(et.fromstring(xml_trans.format(trans=trans)))
    
    # return xml dataset
    return xml_ds

# meta, checks
def make_vrt_raster_xml(x_size, y_size, dtype, band_num, nodata, dt, rel_to_vrt, url, src_band):
    """
    take paramets for vrt and create a raster xml object
    """

    # imports
    from lxml import etree as et
    
    # set up root vrt raster elem
    xml_rast = '<VRTRasterBand dataType="{dtype}" band="{band_num}"></VRTRasterBand>'
    xml_rast = et.fromstring(xml_rast.format(dtype=dtype, band_num=band_num))
        
    # add a top-level nodata value element and add to vrt raster
    xml_ndv = '<NoDataValue>{nodata}</NoDataValue>'
    xml_rast.append(et.fromstring(xml_ndv.format(nodata=nodata)))    
    
    # set up top-level complexsource element, dont add it to rast yet
    xml_complex = '<ComplexSource></ComplexSource>'
    xml_complex = et.fromstring(xml_complex)
    
    # add a description elem to hold datetime to the complex source
    xml_desc = '<Description>{dt}</Description>'
    xml_complex.append(et.fromstring(xml_desc.format(dt=dt)))

    # add source filename to complex source
    xml_filename = '<SourceFilename relativeToVRT="{rel_to_vrt}">/vsicurl/{url}</SourceFilename>'
    xml_complex.append(et.fromstring(xml_filename.format(rel_to_vrt=rel_to_vrt, url=url)))
    
    # add source band num to complex source
    xml_src_band = '<SourceBand>{src_band}</SourceBand>'
    xml_complex.append(et.fromstring(xml_src_band.format(src_band=src_band)))
    
    # add source properties to complex source. hardcoded block size for now
    xml_src_props = '<SourceProperties RasterXSize="{x_size}" RasterYSize="{y_size}"' + ' ' + \
                    'DataType="{dtype}" BlockXSize="512" BlockYSize="512"></SourceProperties>'
    xml_complex.append(et.fromstring(xml_src_props.format(x_size=x_size, y_size=y_size, dtype=dtype)))
    
    # add a src rect to complex source. hardcoded offset for now
    xml_src_rect = '<SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}"></SrcRect>'
    xml_complex.append(et.fromstring(xml_src_rect.format(x_size=x_size, y_size=y_size)))
    
    # add a dst rect to complex source. hardedcoded offset for now
    xml_dst_rect = '<DstRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}"></DstRect>'
    xml_complex.append(et.fromstring(xml_dst_rect.format(x_size=x_size, y_size=y_size)))
    
    # add a lower-level nodata elem to complex source
    xml_nd = '<NODATA>{nodata}</NODATA>'
    xml_complex.append(et.fromstring(xml_nd.format(nodata=nodata)))
        
    # finally, add filled in complex source element to rast
    xml_rast.append(xml_complex)
    
    # return xml raster
    return xml_rast

# todo checks, meta
def make_vrt_list(feat_list, band=None):
    """
    take a list of stac features and band(s) names and build gdal
    friendly vrt xml objects in list.
    band : list, str
        Can be a list or string of name of band(s) required.
    """
    
    # imports
    from lxml import etree as et
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    
    # check if band provided, if so and is str, make list
    if band is None:
        bands = []
    elif not isinstance(band, list):
        bands = [band]
    else:
        bands = band
                    
    # check features type, length
    if not isinstance(feat_list, list):
        raise TypeError('Features must be a list of xml objects.')
    elif not len(feat_list) > 0:
        raise ValueError('No features provided.')
        
    # set list vrt of each scene
    vrt_list = []

    # iter stac scenes, build a vrt
    for feat in feat_list:

        # get scene identity and properties
        f_id = feat.get('id')
        f_props = feat.get('properties')

        # get scene-level date
        f_dt = f_props.get('datetime')

        # get scene-level x, y parameters
        f_x_size = f_props.get('proj:shape')[1]
        f_y_size = f_props.get('proj:shape')[0]

        # get scene-level epsg src as wkt
        f_srs = CRS.from_epsg(f_props.get('proj:epsg'))
        f_srs = f_srs.wkt
        #from osgeo.osr import SpatialReference
        #osr_crs = SpatialReference()
        #osr_crs.ImportFromEPSG(f_props.get('proj:epsg'))
        #f_srs = osr_crs.ExportToWkt()
        

        # get scene-level transform
        #from affine import Affine
        aff = Affine(*f_props.get('proj:transform')[0:6])
        f_transform = ', '.join(str(p) for p in Affine.to_gdal(aff))

        # build a top-level vrt dataset xml object
        xml_ds = satfetcher.make_vrt_dataset_xml(x_size=f_x_size,
                                                 y_size=f_y_size,
                                                 axis_map='1,2',  # hardcoded
                                                 srs=f_srs,
                                                 trans=f_transform)
        
        # iterate bands and build raster vrts
        band_idx = 1
        for band in bands:
            if band in feat.get('assets'):

                # get asset
                asset = feat.get('assets').get(band)

                # set dtype to int16... todo bug in rasterio with int8?
                #a_dtype = 'UInt8' if band == 'oa_fmask' else 'Int16'
                a_dtype = 'Int16'

                # get asset raster x, y sizes
                a_x_size = asset.get('proj:shape')[1]
                a_y_size = asset.get('proj:shape')[0]

                # get raster url, replace s3 with https
                a_url = asset.get('href')
                a_url = a_url.replace('s3://dea-public-data', 'https://data.dea.ga.gov.au')
                
                # get nodata value
                a_nodata = 0 if band == 'oa_fmask' else -999

                # build raster xml
                xml_rast = satfetcher.make_vrt_raster_xml(x_size=a_x_size,
                                                          y_size=a_y_size,
                                                          dtype=a_dtype,
                                                          band_num=band_idx,
                                                          nodata=a_nodata,
                                                          dt=f_dt,
                                                          rel_to_vrt=0,  # hardcoded
                                                          url=a_url,
                                                          src_band=1)  # hardcoded

                # append raster xml to vrt dataset xml
                xml_ds.append(xml_rast)

                # increase band index
                band_idx += 1

        # decode to utf-8 string and append to vrt list
        xml_ds = et.tostring(xml_ds).decode('utf-8')
        vrt_list.append(xml_ds)
        
    return vrt_list

# meta, check
def get_dea_landsat_vrt_dict(feat_list):
    """
    this func is designed to take all releveant landsat bands
    on the dea public database for each scene in stac query.
    it results in a list of vrts for each band seperately and maps
    them to a dict where band name is the key, list is the value pair.
    """
        
    # notify
    print('Getting landsat vrts for each relevant bands.')
                        
    # check features type, length
    if not isinstance(feat_list, list):
        raise TypeError('Features must be a list of xml objects.')
    elif not len(feat_list) > 0:
        raise ValueError('No features provided.')
    
    # required dea landsat ard band names
    bands = [
        'nbart_blue', 
        'nbart_green',
        'nbart_red',
        'nbart_nir',
        'nbart_swir_1',
        'nbart_swir_2',
        'oa_fmask'
    ]
    
    # iter each band name and build associated vrt list
    band_vrts_dict = {}
    for band in bands:
        print('Building landsat vrt list for band: {}.'.format(band))
        band_vrts_dict[band] = make_vrt_list(feat_list, band=band)
        
    # notify and return
    print('Got {} landsat vrt band lists successfully.'.format(len(band_vrts_dict)))
    return band_vrts_dict
    

    
# checks, meta - resample, warp tech needed NEEDS WORK!
def build_vrt_file(vrt_list):

    # imports
    import tempfile
    import gdal
    from rasterio.crs import CRS
        
    # check features type, length
    if not isinstance(vrt_list, list):
        raise TypeError('VRT list must be a list of xml objects.')
    elif not len(vrt_list) > 0:
        raise ValueError('No VRT xml objects provided.')
    
    # build vrt    
    with tempfile.NamedTemporaryFile() as tmp:

        # append vrt extension to temp file
        f = tmp.name + '.vrt'

        # create vrt options
        #bb = (602485.0, -2522685.0, 632495.0, -2507775.0)
        #out_crs = CRS.from_epsg(3577).wkt
        opts = gdal.BuildVRTOptions(separate=True)
                                    #outputSRS=out_crs)
                                    #bandList=[1],
                                    #outputBounds=bb)
        
        #resampleAlg='bilinear',
        #resolution='highest',
        #xRes=30.0,
        #yRes=30.0,
        #outputSRS=rasterio.crs.CRS.from_epsg(3577).wkt
        #targetAlignedPixels=True
        
        # warp/translate?
        # todo

        # consutruct vrt in memory, write it with none
        vrt = gdal.BuildVRT(f, vrt_list, options=opts)
        vrt.FlushCache()
        
        # decode ytf-8?
                
        return f

# meta, checks
def combine_vrts_per_band(band_vrt_dict):
    """
    takes a dictionary of band name : vrt list key, value pairs and
    for each band, combines vrts into one vrt using the build vrt file 
    function (just a call to gdal.BuildVRT).
    """
        
    # notify
    print('Combining VRTs into single VRTs per band.')
                        
    # check features type, length
    if not isinstance(band_vrt_dict, dict):
        raise TypeError('Features must be a dict of band : vrt list objects.')
    elif not len(band_vrt_dict) > 0:
        raise ValueError('No band vrts in dictionary.')
    
    # get list of band names in dict
    bands = [band for band in band_vrt_dict]
    
    # iter each band name and build associated vrt dict
    vrt_file_dict = {}
    for band in bands:
        print('Combining VRTs into temp. file for band: {}.'.format(band))
        vrt_list = band_vrt_dict[band]
        vrt_file_dict[band] = satfetcher.build_vrt_file(vrt_list)

    # notify and return
    print('Combined {} band vrt lists successfully.'.format(len(vrt_file_dict)))
    return vrt_file_dict



# meta, checks, rethink it
def parse_vrt_datetimes(vrt_list):
    """
    takes a list of vrt files and extracts datetime from
    the descriptiont tag.
    """
    
    # imports
    from lxml import etree as et
    
    # checks
    
    # set dt map and counter
    dt_map = {}
    i = 1
    
    # iter items and parse description, skip errors
    for item in vrt_list:
        try:
            # parse out description tag as text
            root = et.fromstring(item)
            desc = root.findall('.//Description')[0]
            desc = desc.text

            # add index and datetime to datetime
            dt_map[i] = desc
            i += 1

        except Exception as e:
            print('Warning: {} at index: {}.'.format(e, i))
          
    # return
    return dt_map

# meta, checks, as above
def get_vrt_file_datetimes(vrt_file_dict):
    """
    takes a dictionary of band : vrt files and parses
    datetimes from each vrt file. spits out a dict of
    band name : dicts (band indexes : datetimes)
    """
    
    # imports
    import gdal
    
    # notify
    print('Extracting datetimes for VRTs per band.')
                        
    # check features type, length
    if not isinstance(vrt_file_dict, dict):
        raise TypeError('VRTs must be a dict of band name : vrt files.')
    elif not len(vrt_file_dict) > 0:
        raise ValueError('No vrts in dictionary.')
    
    # get list of band names in dict
    bands = [band for band in vrt_file_dict]
    
    # iter each band name and build associated vrt dict
    dt_dict = {}
    for band in bands:
        print('Extracting datetimes from VRTs for band: {}.'.format(band))
        
        # get vrt file for current band, open with gdal and extract
        vrt_list = vrt_file_dict[band]
        tmp = gdal.Open(vrt_list).GetFileList()
        dt_dict[band] = satfetcher.parse_vrt_datetimes(tmp)

    # notify and return
    print('Extracted {} band vrt datetimes successfully.'.format(len(dt_dict)))
    return dt_dict



# checks, meta
def prepare_full_vrt_dicts(vrt_file_dict, vrt_dt_dict):
    """
    takes vrt file and datetime file dicts and combines
    into one final dict
    """
    
    # imports
    from collections import Counter
    
    # notify
    print('Combining vrt files and datetimes per band.')
    
    # checks
    
    # get list of band names in dict
    file_bands = [band for band in vrt_file_dict]
    dt_bands = [band for band in vrt_dt_dict]
    
    # check if same bands lists identical
    if Counter(file_bands) != Counter(dt_bands):
        raise ValueError('VRT and datetime band names not identical.')
        
    # iter vrt file dict and create as we go
    vrt_dict = {}
    for band in file_bands:
        vrt_dict[band] = {
            'vrt_datetimes': vrt_dt_dict[band],
            'vrt_file': vrt_file_dict[band]}
        
    # notify and return
    print('Combined vrt files and datetimes per band successfully.')
    return vrt_dict



# meta, checks, more features
def build_xr_datasets(vrt_dict):
    """
    """

    # imports
    import xarray as xr
    from dateutil.parser import parse
   
    # notify
    print('Building an xarray dataset from vrt files and datetimes.')

    # checks

    # get list of band names in dict
    bands = [band for band in vrt_dict]

    # iter bands and append to dataset list
    ds_list = []
    for band in bands:
        
        # notify
        print('Working on dataset for band: {}'.format(band))

        # get date dt index and values
        vrt_dt = vrt_dict[band].get('vrt_datetimes')

        # prepare datetimes
        np_dt = {}
        for k, v in vrt_dt.items():
            v = parse(v, ignoretz=True)
            np_dt[k] = np.datetime64(v)

        # get vrt file
        vrt = vrt_dict[band].get('vrt_file')

        # create chunks
        chunks = {'band': 1, 'x': 'auto', 'y': 'auto'}

        # open into xarray via rasterio
        ds = xr.open_rasterio(vrt, chunks=chunks)

        # rename dim band  to time
        ds = ds.rename({'band': 'time'})

        # convert to dataset and name var to band
        ds = ds.to_dataset(name=band, promote_attrs=True)

        # remap times from indexes to datetimes
        ds['time'] = [np_dt[i] for i in ds['time'].values.tolist()]

        # sort datetime
        ds = ds.sortby('time')

        # append
        ds_list.append(ds)

    # bail if nothing worked
    if not ds_list:
        raise ValueError('No datasets were created.')

    # concatenate datasets into one
    ds = xr.merge(ds_list, combine_attrs='override')
    
    # notify and return
    print('Built an xarray dataset successfully.')
    return ds # change back to ds


# checks, meta, check classes good, check method good
def remove_oa_data(ds, oa_classes=[1, 4, 5], max_cloud=10, remove=False):
    """
    """
    
    # checks
        
    
    # notify
    print('Removing times where oa pixels too numerous.')
    
    # compute mask layer
    mask = ds['oa_fmask'].astype('int8')
    
    # notify, compute
    print('Computing mask layer, this can take awhile. Please wait.')
    mask = mask.compute()
    
    # convert requested classes into binary 1, 0
    mask = xr.where(mask.isin(oa_classes), 1, 0)
    
    # calc percentage of errors within each date
    mask = (100 - (mask.sum(['x', 'y']) / mask.count(['x', 'y'])) * 100)
    
    # get list of datetimes where oa low
    np_dts = mask['time'].where(mask <= max_cloud, drop=True)
    
    # keep good oa times, drop rest
    ds = ds.where(ds['time'].isin(np_dts), drop=True)
    
    # if remove, drop from dataset
    if remove:
        ds = ds.drop_vars('oa_fmask')
        
    # check if dask, rechunk
    if bool(ds.chunks):
        ds = ds.chunk(-1)
    
    # notify and return
    print('Removed times successfully.')
    return ds