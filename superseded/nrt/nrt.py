# DEPRECATED
def write_empty_json(filepath):
    with open(filepath, 'w') as f:
        json.dump([], f)


# DEPRECATED
def load_json(filepath):
    """load json file"""
    
    # check if file exists 
    if not os.path.exists(filepath):
        raise ValueError('File does not exist: {}.'.format(filepath))
    
    # read json file
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    return data


# DEPRECATED
def save_json(filepath, data):
    """save json file"""
    
    # check if file exists 
    if not os.path.exists(filepath):
        raise ValueError('File does not exist: {}.'.format(filepath))
    
    # read json file
    with open(filepath, 'w') as f:
        json.dump(data, f)
        
    return data


# DEPRECATED
def get_item_from_json(filepath, global_id):
    """"""
    
    # check if file exists, else none
    if not os.path.exists(filepath):
        return
    elif global_id is None or not isinstance(global_id, str):
        return
    
    # read json file
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        return

    # check json data for item
    for item in data:
        if item.get('global_id') == global_id:
            return item

    # empty handed
    return
    

# DEPRECATED
def get_latest_date_from_json_item(json_item):
    """"""
    
    # check if item is dict, else default
    if json_item is None:
        return '1980-01-01'
    elif not isinstance(json_item, dict):
        return '1980-01-01'
    
    # fetch latest date if exists, else default 
    if json_item.get('data') is not None:
        dates = [dt for dt in json_item.get('data')]
        if dates is not None and len(dates) > 0:
            return dates[-1]
    
    # otherwise default
    return '1980-01-01'


# DEPRECATED
def get_json_array_via_key(json_item, key):
    """"""
    
    # check json item (none, not array, key not in it)
    if json_item is None:
        return np.array([])
    elif not isinstance(json_item, dict):
        return np.array([])
    elif key not in json_item:
        return np.array([])
    
    # fetch values if exists, else default 
    vals = json_item.get(key)
    if vals is not None and len(vals) > 0:
        return np.array(vals)
    
    # otherwise default
    return np.array([])

  
# DEPRECATED meta, checks - DYNAMIC DISABLED, NOT USING 
def build_change_cube(ds, training_start_year=None, training_end_year=None, persistence_per_year=1, add_extra_vars=True):
    """
    """

    # checks

    # notify
    print('Detecting change via static and dynamic methods.')
    
    # get attributes from dataset
    data_attrs = ds.attrs
    band_attrs = ds[list(ds.data_vars)[0]].attrs
    sref_attrs = ds['spatial_ref'].attrs
    
    # sumamrise each image to a single median value
    ds_summary = ds.median(['x', 'y'], keep_attrs=True)

    # perform static ewmacd and add as new var
    print('Generating static model')
    ds_summary['static'] = EWMACD(ds=ds_summary, 
                                  trainingPeriod='static',
                                  trainingStart=training_start_year,
                                  trainingEnd=training_end_year,
                                  persistence_per_year=persistence_per_year)['veg_idx']
    
    # perform dynamic ewmacd and add as new var
    #print('Generating dynamic model')
    #ds_summary['dynamic'] = EWMACD(ds=ds_summary, trainingPeriod='dynamic',
                                   #trainingStart=training_start_year,
                                   #persistence_per_year=persistence_per_year)['veg_idx']

    # rename original veg_idx to summary
    #ds_summary = ds_summary.rename({'veg_idx': 'summary'})

    # broadcast summary back on to original dataset and order axes
    ds_summary, _ = xr.broadcast(ds_summary, ds)
    ds_summary = ds_summary.transpose('time', 'y', 'x')
    
    # add extra empty vars (zones, cands, conseqs) to dataset if new
    if add_extra_vars:
        for var in ['zones', 'cands_inc', 'cands_dec', 'consq_inc', 'consq_dec']:
            if var not in ds_summary:
                ds_summary[var] = xr.full_like(ds_summary['veg_idx'], np.nan)    
                
    # append attrbutes back on
    ds_summary.attrs = data_attrs
    ds_summary['spatial_ref'].attrs = sref_attrs
    for var in list(ds_summary.data_vars):
        ds_summary[var].attrs = band_attrs

    # notify and return
    print('Successfully created detection cube')
    return ds_summary

# DEPRECATED meta, checks, clean!
def perform_change_detection(ds, var_name=None, training_start_year=None, training_end_year=None, persistence=1):
    """"""
    
    # checks
    #
    
    # notify
    print('Detecting change via static method.')
    
    # reduce down to select variable 
    ds = ds[var_name]
    
    # limit to the start of training time
    ds = ds.where(ds['time'] >= training_start_year, drop=True)
    
    # perform it
    result = nrt.EWMACD(ds=ds,
                        trainingPeriod='static',
                        trainingStart=training_start_year,
                        trainingEnd=training_end_year,
                        persistence_per_year=persistence)
    
    return result


# DEPRECATED
def EWMACD_np(dates, arr, trainingPeriod='dynamic', trainingStart=None, testingEnd=None, trainingEnd=None, minTrainingLength=None, maxTrainingLength=np.inf, trainingFitMinimumQuality=0.8, numberHarmonicsSine=2, numberHarmonicsCosine='same as Sine', xBarLimit1=1.5, xBarLimit2= 20, lowthresh=0, _lambda=0.3, lambdaSigs=3, rounding=True, persistence_per_year=1, reverseOrder=False, summaryMethod='date-by-date', outputType='chart.values'):
    """main function"""


    # get day of years and associated year as int 16
    DOYs = dates['time.dayofyear'].data.astype('int16')
    Years = dates['time.year'].data.astype('int16')
    
    # check if training start is < max year 
    if trainingStart >= Years[-1]:
        raise ValueError('Training year must be lower than maximum year in data.')

    # check doys, years
    if len(DOYs) != len(Years):
        raise ValueError('DOYs and Years are not same length.')

    # if no training date provided, choose first year
    if trainingStart is None:
        trainingStart = np.min(Years)

    # if no testing date provided, choose last year + 1
    if testingEnd is None:
        testingEnd = np.max(Years) + 1

    # generate array of nans for every year between start of train and test period
    NAvector = np.repeat(np.nan, len(Years[(Years >= trainingStart) & (Years < testingEnd)]))

    # if not date to date, use year to year (?) may not need this
    if summaryMethod != 'date-by-date':
        num_nans = len(np.unique(Years[(Years >= trainingStart) & (Years < testingEnd)]))
        NAvector = np.repeat(np.nan, num_nans)

    # set cos harmonics value (default 2) to same as sine, if requested
    if numberHarmonicsCosine == 'same as Sine':
        numberHarmonicsCosine = numberHarmonicsSine

    # set simple output if chart values requested (?)
    if outputType == 'chart.values':
        simple_output = True

    # create per-pixel vectorised version of ewmacd per-pixel func       
    try:
        change = EWMACD_pixel_date_by_date(myPixel=arr,
                                           DOYs=DOYs,
                                           Years=Years,
                                           _lambda=_lambda,
                                           numberHarmonicsSine=numberHarmonicsSine,
                                           numberHarmonicsCosine=numberHarmonicsCosine,
                                           trainingStart=trainingStart,
                                           testingEnd=testingEnd,
                                           trainingPeriod=trainingPeriod,
                                           trainingEnd=trainingEnd,
                                           minTrainingLength=minTrainingLength,
                                           maxTrainingLength=maxTrainingLength,
                                           trainingFitMinimumQuality=trainingFitMinimumQuality,
                                           xBarLimit1=xBarLimit1,
                                           xBarLimit2=xBarLimit2,
                                           lowthresh=lowthresh,
                                           lambdaSigs=lambdaSigs,
                                           rounding=rounding,
                                           persistence_per_year=persistence_per_year,
                                           reverseOrder=reverseOrder,
                                           simple_output=simple_output)

        # get change per date from above
        change = change.get('dateByDate')

        # calculate summary method (todo set up for others than just date to date
        final_out = annual_summaries(Values=change,
                                     yearIndex=Years,
                                     summaryMethod=summaryMethod)

    except Exception as e:
        print('Could not train model adequately, please add more years.')
        print(e)
        final_out = NAvector
    
    # rename veg_idx to change and convert to float32
    #arr = arr.astype('float32')
    
    #return dataset
    #return arr
    return final_out


# deprecated! meta
def reproject_ogr_geom(geom, from_epsg=3577, to_epsg=4326):
    """
    """
    
    # check if ogr layer type
    if not isinstance(geom, ogr.Geometry):
        raise TypeError('Layer is not of ogr Geometry type.')
        
    # check if epsg codes are ints
    if not isinstance(from_epsg, int):
        raise TypeError('From epsg must be integer.')
    elif not isinstance(to_epsg, int):
        raise TypeError('To epsg must be integer.')
        
    # notify
    print('Reprojecting layer from EPSG {} to EPSG {}.'.format(from_epsg, to_epsg))
            
    try:
        # init spatial references
        from_srs = osr.SpatialReference()
        to_srs = osr.SpatialReference()
    
        # set spatial references based on epsgs (inplace)
        from_srs.ImportFromEPSG(from_epsg)
        to_srs.ImportFromEPSG(to_epsg)
        
        # transform
        trans = osr.CoordinateTransformation(from_srs, to_srs)
        
        # reproject
        geom.Transform(trans)
        
    except: 
        raise ValueError('Could not transform ogr geometry.')
        
    # notify and return
    print('Successfully reprojected layer.')
    return geom


# deprecated! 
def remove_spikes_np(arr, user_factor=2, win_size=3):
    """
    Takes an numpy array containing vegetation index variable and removes outliers within 
    the timeseries on a per-pixel basis. The resulting dataset contains the timeseries 
    with outliers set to nan.
    
    Parameters
    ----------
    arr: numpy ndarray
        A one-dimensional array containing a vegetation index values.
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. A higher factor 
        value results in few outliers (i.e. only the biggest outliers). Default factor is 2.
    win_size: int
        Number of samples to include in rolling median window.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with all detected outliers in the
        veg_index variable set to nan.
    """
    
    # notify user
    print('Removing spike outliers.')
            
    # check inputs
    if arr is None:
        raise ValueError('Array is empty.')

    # get nan mask (where nan is True)
    cutoffs = np.std(arr) * user_factor

    # do moving win median, back to numpy, fill edge nans 
    roll = pd.Series(arr).rolling(window=win_size, center=True)
    arr_win = roll.median().to_numpy()
    arr_med = np.where(np.isnan(arr_win), arr, arr_win)

    # calc abs diff between orig arr and med arr
    arr_dif = np.abs(arr - arr_med)

    # make mask where absolute diffs exceed cutoff
    mask = np.where(arr_dif > cutoffs, True, False)

    # get value left, right of each outlier and get mean
    l = np.where(mask, np.roll(arr, shift=1), np.nan)  # ->
    r = np.where(mask, np.roll(arr, shift=-1), np.nan) # <-
    arr_mean = (l + r) / 2
    arr_fmax = np.fmax(l, r)

    # mask if middle val < mean of neighbours - cutoff or middle val > max val + cutoffs 
    arr_outliers = ((np.where(mask, arr, np.nan) < (arr_mean - cutoffs)) | 
                    (np.where(mask, arr, np.nan) > (arr_fmax + cutoffs)))

    # apply the mask
    arr = np.where(arr_outliers, np.nan, arr)
    
    return arr


# deprecated!
def interp_nan_np(arr):
    """equal to interpolate_na in xr"""

    # notify user
    print('Interpolating nan values.')
            
    # check inputs
    if arr is None:
        raise ValueError('Array is empty.')
        
    # get range of indexes
    idxs = np.arange(len(arr))
    
    # interpolate linearly 
    arr = np.interp(idxs, 
                    idxs[~np.isnan(arr)], 
                    arr[~np.isnan(arr)])
                    
    return arr


# DEPRECATED meta - deprecated! no longer using
def fill_zeros_with_last(arr):
    """
    forward fills differences of 0 after a 
    decline or positive flag.
    """
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    
    return arr[prev]


# DEPRECATED checks, meta
def sync_new_and_old_cubes(ds_exist, ds_new, out_nc):
    """Takes two structurally idential xarray datasets and 
    combines them into one, where only new data from the latest 
    new dataset is combined with all of the old. Either way, a 
    file of this process is written to output path. This drives 
    the nrt on-going approach of the module."""
    
    # also set rasterio env variables
    rasterio_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS':'tif',
        'VSI_CACHE': True,
        'GDAL_HTTP_MULTIRANGE': 'YES',
        'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
    }
    
    # checks
    # 
    
    # if a new dataset provided only, write and load new
    if ds_exist is None and ds_new is not None:
        print('Existing dataset not provided. Creating and loading for first time.')
        
        # write netcdf file
        with rasterio.Env(**rasterio_env):
            ds_new = ds_new.astype('float32')
            tools.export_xr_as_nc(ds=ds_new, filename=out_nc)
            
        # safeload new dataset and return
        ds_new = safe_load_nc(out_nc)
        return ds_new
            
    elif ds_exist is not None and ds_new is not None:
        print('Existing and New dataset provided. Combining, writing and loading.')  
        
        # ensure existing is not locked via safe load (new always in mem)
        ds_exist = safe_load_nc(out_nc)
                
        # extract only new datetimes from new dataset
        dts = ds_exist['time']
        ds_new = ds_new.where(~ds_new['time'].isin(dts), drop=True)
        
        # check if any new images
        if len(ds_new['time']) > 0:
            print('New images detected ({}), adding and overwriting existing cube.'.format(len(ds_new['time'])))
                        
            # combine new with old (concat converts to float)
            ds_combined = xr.concat([ds_exist, ds_new], dim='time').copy(deep=True) 

            # write netcdf file
            with rasterio.Env(**rasterio_env):
                ds_combined = ds_combined.astype('float32')
                tools.export_xr_as_nc(ds=ds_combined, filename=out_nc)            
             
            # safeload new dataset and return
            ds_combined = safe_load_nc(out_nc)
            return ds_combined
        
        else:
            print('No new images detected, returning existing cube.')
            
            # safeload new dataset and return
            ds_exist = safe_load_nc(out_nc)
            return ds_exist

    else:
        raise ValueError('At a minimum, a new dataset must be provided.')
        return

 
 # deprecated meta, checks, check rule 2 operator is right
def get_candidates(vec, direction='incline', min_consequtives=3, max_consequtives=None, inc_plateaus=False, min_stdv=1, num_zones=1, bidirectional=False, ruleset='1&2|3', binarise=True):
    """
    min_conseq = rule 1
    max_conseq = rule 1
    inc_plateaus = rule 1
    min_stdv = rule 2
    num_zones = rule 3
    rulset = all
    binarise = set out to 1,0 not 1, nan
    """
    
    # checks
    # min_conseq >= 0
    # max_conseq >= 0, > min, or None
    # min_stdv >= 0 
    # operator only > >= < <=
    
    # set up parameters reliant on direction
    if direction == 'incline':
        operator = '>='
    elif direction == 'decline':
        operator = '<='
    
    # calculate rule 1 (consequtive runs)
    print('Calculating rule one: consequtive runs {}.'.format(direction))
    vec_rule_1 = apply_rule_one(arr=vec,
                                direction=direction,
                                min_consequtives=min_consequtives,        # min consequtive before candidate
                                max_consequtives=max_consequtives,        # max num consequtives before reset
                                inc_plateaus=inc_plateaus)                # include plateaus after decline
    
    # calculate rule 2 (zone threshold)
    print('Calculating rule two: zone threshold {}.'.format(direction))
    vec_rule_2 = apply_rule_two(arr=vec,
                                direction=direction,
                                min_stdv=min_stdv,                        # min stdv threshold
                                operator=operator,
                                bidirectional=bidirectional)              # operator e.g. <=
        
    # calculate rule 3 (jumps) increase
    print('Calculating rule three: sharp jump {}.'.format(direction))
    num_stdvs = get_stdv_from_zone(num_zones=num_zones)
    vec_rule_3 = apply_rule_three(arr=vec,
                                  direction=direction,
                                  num_stdv_jumped=num_stdvs,               
                                  min_consequtives=min_consequtives,
                                  max_consequtives=min_consequtives)      # careful !!!!!!

    
    # combine rules 1, 2, 3 decreasing
    print('Combining rule 1, 2, 3 via ruleset {}.'.format(ruleset))
    vec_rules_combo = apply_rule_combo(arr_r1=vec_rule_1, 
                                       arr_r2=vec_rule_2, 
                                       arr_r3=vec_rule_3, 
                                       ruleset=ruleset)
    
    # binarise 1, nan to 1, 0 if requested
    if binarise:
        vec_rules_combo = np.where(vec_rules_combo == 1, 1.0, 0.0)
    
    
    return vec_rules_combo


# deprecated
def build_alerts_xr(ds, ruleset=None, direction=None):
    """
    Builds alert mask (1s and 0s) based on combined rule
    values and assigned ruleset.
    """

    # set up valid rulesets
    valid_rules = [
        '1', 
        '2', 
        '3', 
        '1&2', 
        '1&3', 
        '2&3', 
        '1|2', 
        '1|3', 
        '2|3', 
        '1&2&3', 
        '1|2&3',
        '1&2|3', 
        '1|2|3']

    # check dataset
    if not isinstance(ds, xr.Dataset):
        raise ValueError('Input dataset must be a xarray dataset.')

    # check required static rule vars in dataset
    static_vars = ['static_rule_one', 'static_rule_two', 'static_rule_three', 'static_alerts']
    dynamic_vars = ['dynamic_rule_one', 'dynamic_rule_two', 'dynamic_rule_three', 'dynamic_alerts']
    for var in static_vars + dynamic_vars:
        if var not in ds:
            raise ValueError('Could not find variable: {} in dataset.'.format(var))

    # check if ruleset in allowed rules, direction is valid
    if ruleset not in valid_rules:
        raise ValueError('Ruleset is not supported.')
    elif direction not in ['Incline', 'Decline']:
        raise ValueError('Direction is not supported.')

    # build copy of input dataset for temp working
    ds_alr = ds.copy(deep=True)

    # correct raw rule vals for direction and set 1 if alert, 0 if not
    for var in static_vars + dynamic_vars: 
        if direction == 'Incline':
            ds_alr[var] = xr.where(ds_alr[var] > 0, 1, 0)
        elif direction == 'Decline':
            ds_alr[var] = xr.where(ds_alr[var] < 0, 1, 0)

    # set up short var names for presentation
    sr1, sr2, sr3 = 'static_rule_one', 'static_rule_two', 'static_rule_three'
    dr1, dr2, dr3 = 'dynamic_rule_one', 'dynamic_rule_two', 'dynamic_rule_three'

    # create alert arrays based on singular rule
    if ruleset == '1':
        ds_alr['static_alerts']  = ds_alr[sr1]
        ds_alr['dynamic_alerts'] = ds_alr[dr1]
    elif ruleset == '2':
        ds_alr['static_alerts']  = ds_alr[sr2]
        ds_alr['dynamic_alerts'] = ds_alr[dr2]
    elif ruleset == '3':
        ds_alr['static_alerts']  = ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr3]

    # create alert arrays based on dual "and" rule
    if ruleset == '1&2':
        ds_alr['static_alerts']  = ds_alr[sr1] & ds_alr[sr2]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] & ds_alr[dr2]
    elif ruleset == '1&3':
        ds_alr['static_alerts']  = ds_alr[sr1] & ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] & ds_alr[dr3]
    elif ruleset == '2&3':
        ds_alr['static_alerts']  = ds_alr[sr2] & ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr2] & ds_alr[dr3]    

    # create alert arrays based on dual "or" rule
    if ruleset == '1|2':
        ds_alr['static_alerts']  = ds_alr[sr1] | ds_alr[sr2]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] | ds_alr[dr2]
    elif ruleset == '1|3':
        ds_alr['static_alerts']  = ds_alr[sr1] | ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] | ds_alr[dr3]
    elif ruleset == '2|3':
        ds_alr['static_alerts']  = ds_alr[sr2] | ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr2] | ds_alr[dr3]    

    # create alert arrays based on complex rule
    if ruleset == '1&2&3':  
        ds_alr['static_alerts']  = ds_alr[sr1] & ds_alr[sr2] & ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] & ds_alr[dr2] & ds_alr[dr3]
    elif ruleset == '1|2&3':  
        ds_alr['static_alerts']  = ds_alr[sr1] | (ds_alr[sr2] & ds_alr[sr3])
        ds_alr['dynamic_alerts'] = ds_alr[dr1] | (ds_alr[dr2] & ds_alr[dr3])
    elif ruleset == '1&2|3':  
        ds_alr['static_alerts']  = (ds_alr[sr1] & ds_alr[sr2]) | ds_alr[sr3]
        ds_alr['dynamic_alerts'] = (ds_alr[dr1] & ds_alr[dr2]) | ds_alr[dr3]
    elif ruleset == '1|2|3':  
        ds_alr['static_alerts']  = ds_alr[sr1] | ds_alr[sr2] | ds_alr[sr3]
        ds_alr['dynamic_alerts'] = ds_alr[dr1] | ds_alr[dr2] | ds_alr[dr3]

    # check if array sizes match
    if len(ds['static_alerts']) != len(ds_alr['static_alerts']):
        raise ValueError('Static alert array incorrect size.')
    elif len(ds['dynamic_alerts']) != len(ds_alr['dynamic_alerts']):
        raise ValueError('Dynamic alert array incorrect size.')

    # transfer alert arrays over to original dataset
    ds['static_alerts'] = ds_alr['static_alerts']
    ds['dynamic_alerts'] = ds_alr['dynamic_alerts']
    
    return ds


# deprecated, meta checks 
def apply_rule_two(arr, direction='decline', min_stdv=1, operator='<=', bidirectional=False):
    """
    takes array of smoothed change output and thresholds out
    any values outside of a specified minimum zone e.g. 1.
    """
    
    # check direction
    if direction not in ['incline', 'decline']:
        raise ValueError('Direction must be incline or decline.')
        
    # checks
    if operator not in ['<', '<=', '>', '>=']:
        raise ValueError('Operator must be <, <=, >, >=')
        
    # set stdv to negative if decline
    if direction == 'decline':
        min_stdv = min_stdv * -1
        
    # check operator matches direction 
    if direction == 'incline' and operator not in ['>', '>=']:
        print('Operator must be > or >= when using incline. Setting to >=.')
        operator = '>='
    elif direction == 'decline' and operator not in ['<', '<=']:
        print('Operator must be < or <= when using decline. Setting to <=.')
        operator = '<='

    # operate based on 
    if bidirectional:
        print('Bidrectional enabled, ignoring direction.')
        arr_abs = np.abs(arr)
        
        if '=' in operator:
            arr_thresholded = np.where(arr_abs >= abs(min_stdv), arr, np.nan)
        else:
            arr_thresholded = np.where(arr_abs > abs(min_stdv), arr, np.nan)
        
    elif operator == '<':
        arr_thresholded = np.where(arr < min_stdv, arr, np.nan)
    elif operator == '<=':
        arr_thresholded = np.where(arr <= min_stdv, arr, np.nan)
    elif operator == '>':
        arr_thresholded = np.where(arr > min_stdv, arr, np.nan)
    elif operator == '>=':
        arr_thresholded = np.where(arr >= min_stdv, arr, np.nan)
        
    return arr_thresholded

  
# deprecated meta checks
def apply_rule_three(arr, direction='decline', num_stdv_jumped=3, min_consequtives=3, max_consequtives=3):
    """
    takes array of smoothed (or raw) change output and detects large, multi zone
    jumps. candidates only registered if a specific number of post jump
    runs detected (set min_consequtives to 0 for any spike regardless of runs).
    jump_size is number of zones required to jump - default is 3 stdv (1 zone). max
    consequtives will cut the run off after certain number of indices detected.
    """
    
    # checks
    if direction not in ['incline', 'decline']:
        raise ValueError('Direction must be incline or decline.')
        
    # prepare max consequtives
    if max_consequtives <= 0:
        print('Warning, max consequtives must be > 0. Resetting to three.')
        max_consequtives = 3
            
    # get diffs
    diffs = np.diff(np.insert(arr, 0, arr[0]))
    
    # threshold by magnitude of jump
    if direction == 'incline':
        arr_jumps = diffs > num_stdv_jumped
    elif direction == 'decline':
        arr_jumps = diffs < (num_stdv_jumped * -1)
        
    # iter each spike index and detect post-spike runs (as 1s)
    indices = []
    for i in np.where(arr_jumps)[0]:

        # loop each element in array from current index and calc diff
        for e, v in enumerate(arr[i:]):
            diff = np.abs(np.abs(arr[i]) - np.abs(v))

            # if diff is less than certain jump size record it, else skip
            # todo diff < 3 is check to see if stays within one zone of devs
            if diff < 3 and e <= max_consequtives:
                indices.append(i + e)
            else:
                break 
                
    # set 1 to every flagged index, 0 to all else
    arr_masked = np.zeros_like(arr)
    arr_masked[indices] = 1
    
    # count continuous runs for requested vector value
    arr_extended = np.concatenate(([0], arr_masked, [0]))        # pad array with empty begin and end elements
    idx = np.flatnonzero(arr_extended[1:] != arr_extended[:-1])  # get start and end indexes
    arr_extended[1:][idx[1::2]] = idx[::2] - idx[1::2]           # grab breaks, prev - current, also trim extended elements
    arr_counted = arr_extended.cumsum()[1:-1]                    # apply cumulative sum
    
    # threshold out specific run counts
    if min_consequtives is not None:
        arr_counted = np.where(arr_counted >= min_consequtives, arr_counted, 0)

    # replace 0s with nans
    arr_counted = np.where(arr_counted != 0, arr_counted, np.nan)
    
    return arr_counted


# meta checks todo count runs???
def apply_rule_combo(arr_r1, arr_r2, arr_r3, ruleset='1&2|3'):
    """
    take pre-generated rule arrays and combine where requested.
    """
    
    allowed_rules = [
        '1', '2', '3', '1&2', '1&3', '2&3', 
        '1|2', '1|3', '2|3', '1&2&3', '1|2&3',
        '1&2|3', '1|2|3']
    
    # checks
    if ruleset not in allowed_rules:
        raise ValueError('Ruleset set is not supported.')
    
    # convert rule arrays to binary masks
    arr_r1_mask = ~np.isnan(arr_r1)
    arr_r2_mask = ~np.isnan(arr_r2)
    arr_r3_mask = ~np.isnan(arr_r3)
        
    # set signular rules
    if ruleset == '1':
        arr_comb = np.where(arr_r1_mask, 1, 0)
    elif ruleset == '2':
        arr_comb = np.where(arr_r2_mask, 1, 0)
    elif ruleset == '3':
        arr_comb = np.where(arr_r3_mask, 1, 0)
    
    # set combined dual ruleset
    elif ruleset == '1&2':
        arr_comb = np.where(arr_r1_mask & arr_r2_mask, 1, 0)
    elif ruleset == '1&3':
        arr_comb = np.where(arr_r1_mask & arr_r3_mask, 1, 0)        
    elif ruleset == '2&3':
        arr_comb = np.where(arr_r2_mask & arr_r3_mask, 1, 0)             
        
    # set either or dual ruleset
    elif ruleset == '1|2':
        arr_comb = np.where(arr_r1_mask | arr_r2_mask, 1, 0)  
    elif ruleset == '1|3':
        arr_comb = np.where(arr_r1_mask | arr_r3_mask, 1, 0) 
    elif ruleset == '2|3':
        arr_comb = np.where(arr_r2_mask | arr_r3_mask, 1, 0)     
        
    # set combined several ruleset
    elif ruleset == '1&2&3':  
        arr_comb = np.where(arr_r1_mask & arr_r2_mask & arr_r3_mask, 1, 0)
    elif ruleset == '1|2&3':  
        arr_comb = np.where(arr_r1_mask | (arr_r2_mask & arr_r3_mask), 1, 0)        
    elif ruleset == '1&2|3':  
        arr_comb = np.where((arr_r1_mask & arr_r2_mask) | arr_r3_mask, 1, 0)  
    elif ruleset == '1|2|3':  
        arr_comb = np.where(arr_r1_mask | arr_r2_mask | arr_r3_mask, 1, 0)
        
    # count runs todo
    
        
    # replace 0s with nans
    arr_comb = np.where(arr_comb != 0, arr_comb, np.nan)
        
    return arr_comb


# meta, deprecated
def validate_monitoring_area(row):
    """
    Does relevant checks for information for a
    single monitoring area.
    """
    
    # check if row is tuple
    if not isinstance(row, tuple):
        print('Row must be of type tuple.')
        return False
            
    # parse row info
    area_id = row[0]
    platform = row[1]
    s_year = row[2]
    e_year = row[3]
    index = row[4]
    persistence = row[5]
    rule_1_min_conseqs = row[6]
    rule_1_inc_plateaus = row[7]
    rule_2_min_zone = row[8]
    rule_3_num_zones = row[9]
    ruleset = row[10]
    alert = row[11]
    method = row[12]
    alert_direction = row[13]
    email = row[14]
    ignore = row[15]
    
    # check area id exists
    if area_id is None:
        print('No area id exists, flagging as invalid.')
        return False

    # check platform is Landsat or Sentinel
    if platform is None:
        print('No platform exists, flagging as invalid.')
        return False
    elif platform.lower() not in ['landsat', 'sentinel']:
        print('Platform must be Landsat or Sentinel, flagging as invalid.')
        return False

    # check if start and end years are valid
    if not isinstance(s_year, int) or not isinstance(e_year, int):
        print('Start and end year values must be integers, flagging as invalid.')
        return False
    elif s_year < 1980 or s_year > 2050:
        print('Start year must be between 1980 and 2050, flagging as invalid.')
        return False
    elif e_year < 1980 or e_year > 2050:
        print('End year must be between 1980 and 2050, flagging as invalid.')
        return False
    elif e_year <= s_year:
        print('Start year must be less than end year, flagging as invalid.')
        return False
    elif abs(e_year - s_year) < 2:
        print('Must be at least 2 years between start and end year, flagging as invalid.')
        return False
    elif platform.lower() == 'sentinel' and s_year < 2016:
        print('Start year must not be < 2016 when using Sentinel, flagging as invalid.')
        return False

    # check if index is acceptable
    if index is None:
        print('No index exists, flagging as invalid.')
        return False
    elif index.lower() not in ['ndvi', 'mavi', 'kndvi']:
        print('Index must be NDVI, MAVI or kNDVI, flagging as invalid.')
        return False
    
    # check if persistence is accepptable
    if persistence is None:
        print('No persistence exists, flagging as invalid.')
        return False
    elif persistence < 0.001 or persistence > 9.999:
        print('Persistence must be before 0.0001 and 9.999, flagging as invalid.')
        return False

    # check if rule_1_min_conseqs is accepptable
    if rule_1_min_conseqs is None:
        print('No rule_1_min_conseqs exists, flagging as invalid.')
        return False
    elif rule_1_min_conseqs < 0 or rule_1_min_conseqs > 99:
        print('Rule_1_min_conseqs must be between 0 and 99, flagging as invalid.')
        return False
    
    # check if rule_1_min_conseqs is accepptable
    if rule_1_inc_plateaus is None:
        print('No rule_1_inc_plateaus exists, flagging as invalid.')
        return False
    elif rule_1_inc_plateaus not in ['Yes', 'No']:
        print('Rule_1_inc_plateaus must be Yes or No, flagging as invalid.')
        return False    
    
    # check if rule_2_min_stdv is accepptable
    if rule_2_min_zone is None:
        print('No rule_2_min_zone exists, flagging as invalid.')
        return False
    elif rule_2_min_zone < 0 or rule_2_min_zone > 99:
        print('Rule_2_min_zone must be between 0 and 99, flagging as invalid.')
        return False      

    # check if rule_2_bidirectional is accepptable
    if rule_3_num_zones is None:
        print('No rule_3_num_zones exists, flagging as invalid.')
        return False
    elif rule_3_num_zones < 0 or rule_3_num_zones > 99:
        print('Rule_3_num_zones must be between 0 and 99, flagging as invalid.')
        return False       

    # check if ruleset is accepptable   
    if ruleset is None:
        print('No ruleset exists, flagging as invalid.')
        return False
    
    # check if alert is accepptable
    if alert is None:
        print('No alert exists, flagging as invalid.')
        return False
    elif alert not in ['Yes', 'No']:
        print('Alert must be Yes or No, flagging as invalid.')
        return False
    
    # check method 
    if method.lower() not in ['static', 'dynamic']:
        print('Method must be Static or Dynamic, flagging as invalid.')
        return False
    
    
    # set up alert directions 
    alert_directions = [
        'Incline only (any)', 
        'Decline only (any)', 
        'Incline only (+ zones only)', 
        'Decline only (- zones only)', 
        'Incline or Decline (any)',
        'Incline or Decline (+/- zones only)'
        ]
    
    # check if alert_direction is accepptable
    if alert_direction is None:
        print('No alert_direction exists, flagging as invalid.')
        return False
    elif alert_direction not in alert_directions:
        print('Alert_direction is not supported.')
        return False  

    # check if email address is accepptable
    if alert == 'Yes' and email is None:
        print('Must provide an email if alert is set to Yes.')
        return False
    elif email is not None:
        if '@' not in email or '.' not in email:
            print('No @ or . character in email exists, flagging as invalid.')
            return False

    # check if ignore is accepptable
    if ignore is None:
        print('No ignore exists, flagging as invalid.')
        return False
    elif ignore not in ['Yes', 'No']:
        print('Ignore must be Yes or No, flagging as invalid.')
        return False    

    # all good!
    return True 
 
 
 # deprecated, meta, checks
def safe_load_nc(in_path):
    """Performs a safe load on local netcdf. Reads 
    NetCDF, loads it, and closes connection to file
    whilse maintaining data in memory"""    
    
    # check if file existd and try safe open
    if os.path.exists(in_path):
        try:
            with xr.open_dataset(in_path) as ds_local:
                ds_local.load()
            
            return ds_local
                    
        except:
            print('Could not open cube: {}, returning None.'.format(in_path))
            return
    
    else:
        print('File does not exist: {}, returning None.'.format(in_path))
        return


# deprecated meta 
def validate_xr_site_attrs(ds, feat):
    """
    For NRT monitoring, site information taken from
    the associated shapefile feature are appended to
    the attributes of the associated netcdf file before 
    moving to the next area. We check these attributes
    every new iteration to check if user has changed
    any monitoring area prameters. If key parameters 
    have been changed, we need to ignore the existing
    netcdf and create a new one.
    """
    
    # check if dataset exists (could be first run)
    if ds is None:
        print('Dataset is none. Returning none.')
        return
    elif not isinstance(ds, xr.Dataset):
        print('Dataset is not an xarray type. Returning none.')
        return
    elif not hasattr(ds, 'attrs'):
        print('Dataset has no attributes. Returning none.')
        return
    
    # check if dict is right length
    if not isinstance(feat, (list, tuple)):
        print('Feature not a list or tuple, returning none.')
        return
    elif len(feat) == 0:
        print('Feature has no data, returning none.')
        return
    
    # set up required attributes (key attrs only)
    attrs = [
        'area_id',
        'platform',
        's_year',
        'e_year',
        'index',
        'persistence',
        'rule_1_min_conseqs',
        'rule_1_inc_plateaus',
        'rule_2_min_zone',
        'rule_3_num_zones',
        'ruleset',
        'alert',
        'method',
        'alert_direction',
        'email',
        'ignore'
    ]
    
    # iterate attributes and check
    for attr in attrs:
        if not hasattr(ds, attr):
            print('Attribute {} does not exist, returning none.')
            return
        
    # check if key dataset attributes differ from current feat
    if ds.attrs.get('platform') != str(feat[1]):
        print('Platform has changed, returning none.')
        return
    elif ds.attrs.get('s_year') != str(feat[2]):
        print('Start year has changed, returning none.')
        return
    elif ds.attrs.get('e_year') != str(feat[3]):
        print('End year has changed, returning none.')
        return
    elif ds.attrs.get('index') != str(feat[4]):
        print('Vegetation index has changed, returning none.')
        return    
    elif ds.attrs.get('persistence') != str(feat[5]):
        print('Persistence has changed, returning none.')
        return        
    elif ds.attrs.get('rule_1_min_conseqs') != str(feat[6]):
        print('Rule 1 min consequtives has changed, returning none.')
        return         
    elif ds.attrs.get('rule_1_inc_plateaus') != str(feat[7]):
        print('Rule 1 include plateaus has changed, returning none.')
        return       
    elif ds.attrs.get('rule_2_min_zone') != str(feat[8]):
        print('Rule 2 min zone has changed, returning none.')
        return           
    elif ds.attrs.get('rule_3_num_zones') != str(feat[9]):
        print('Rule 3 num zones has changed, returning none.')
        return  
    elif ds.attrs.get('ruleset') != str(feat[10]):
        print('Ruleset has changed, returning none.')
        return      
    elif ds.attrs.get('method') != str(feat[12]):
        print('Method has changed, returning none.')
        return       
    elif ds.attrs.get('alert_direction') != str(feat[13]):
        print('Alert direction has changed, returning none.')
        return      
    
    return ds
    
# deprecated meta
def get_satellite_params(platform=None):
    """
    Helper function to generate Landsat or Sentinel query information
    for quick use during NRT cube creation or sync only.
    
    Parameters
    ----------
    platform: str
        Name of a satellite platform, Landsat or Sentinel only.
    
    params
    """
    
    # check platform name
    if platform is None:
        raise ValueError('Must provide a platform name.')
    elif platform.lower() not in ['landsat', 'sentinel', 'sentinel_provisional']:
        raise ValueError('Platform must be Landsat or Sentinel.')
        
    # set up dict
    params = {}
    
    # get porams depending on platform
    if platform.lower() == 'landsat':
        
        # get collections
        collections = [
            'ga_ls5t_ard_3', 
            'ga_ls7e_ard_3', 
            'ga_ls8c_ard_3',
            #'ga_ls7e_ard_provisional_3',  # will always be slc-off
            'ga_ls8c_ard_provisional_3']
        
        # get bands
        bands = [
            'nbart_red', 
            'nbart_green', 
            'nbart_blue', 
            'nbart_nir', 
            'nbart_swir_1', 
            'nbart_swir_2', 
            'oa_fmask']
        
        # get resolution
        resolution = 30
        
        # build dict
        params = {
            'collections': collections,
            'bands': bands,
            'resolution': resolution}
        
    # the product 3 is not yet avail on dea. we use s2 for now.
    elif platform.lower() == 'sentinel':
        
        # get collections
        collections = [
            's2a_ard_granule', 
            's2b_ard_granule',
            'ga_s2am_ard_provisional_3', 
            'ga_s2bm_ard_provisional_3'
            ]
        
        # get bands
        bands = [
            'nbart_red', 
            'nbart_green', 
            'nbart_blue', 
            'nbart_nir_1', 
            'nbart_swir_2', 
            'nbart_swir_3', 
            'fmask']
        
        # get resolution
        resolution = 10
        
        # build dict
        params = {
            'collections': collections,
            'bands': bands,
            'resolution': resolution}   

    return params
 
 
 # deprecated meta
def mask_xr_via_polygon(ds, geom, mask_value=1):
    """
    geom object from gdal
    x, y = arrays of coordinates from xr dataset
    bbox 
    transform from geobox
    ncols, nrows = len of x, y
    """
    
    # check dataset
    if 'x' not in ds or 'y' not in ds:
        raise ValueError('Dataset has no x or y dimensions.')
    elif not hasattr(ds, 'geobox'):
        raise ValueError('Dataset does not have a geobox.')
        
    # extract raw x and y value arrays, bbox, transform and num col, row
    x, y = ds['x'].data, ds['y'].data
    bbox = ds.geobox.extent.boundingbox
    transform = ds.geobox.transform
    ncols, nrows = len(ds['x']), len(ds['y'])
    
    # extract bounding box extents
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

    # create ogr transform structure
    geotransform = (transform[2], transform[0], 0.0, 
                    transform[5], 0.0, transform[4])

    # create template raster in memory
    dst_rast = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 , gdal.GDT_Byte)
    dst_rb = dst_rast.GetRasterBand(1)      # get a band
    dst_rb.Fill(0)                          # init raster with zeros
    dst_rb.SetNoDataValue(0)                # set nodata to zero
    dst_rast.SetGeoTransform(geotransform)  # resample, transform

    # rasterise vector and flush
    err = gdal.RasterizeLayer(dst_rast, [1], geom, burn_values=[mask_value])
    dst_rast.FlushCache()

    # get numpy version of classified raster
    arr = dst_rast.GetRasterBand(1).ReadAsArray()

    # create mask
    mask = xr.DataArray(data=arr,
                        dims=['y', 'x'],
                        coords={'y': y, 'x': x})

    return mask
 
 
 # deprecated, meta
def safe_load_ds(ds):
    """
    
    """

    # check if file existd and try safe open
    if ds is not None:
        try:
            ds = ds.load()
            ds.close()
            
            return ds
                    
        except:
            print('Could not open dataset, returning None.')
            return
    else:
        print('Dataset not provided')
        return


# deprecated meta
def interp_nans(ds, drop_edge_nans=False):
    """"""
    
    # check if time dimension exists 
    if isinstance(ds, xr.Dataset) and 'time' not in ds:
        raise ValueError('Dataset has no time dimension.')
    elif isinstance(ds, xr.DataArray) and 'time' not in ds.dims:
        raise ValueError('DataArray has no time dimension.')
        
    try:
        # interpolate all values via linear interp
        ds = ds.interpolate_na('time')
    
        # remove any remaining nan (first and last indices, for e.g.)
        if drop_edge_nans is True:
            ds = ds.where(~ds.isnull(), drop=True)
    except:
        return
    
    return ds
    

# deprecated meta
def add_required_vars(ds):
    """"""

    # set required variable names
    new_vars = [
        'veg_clean', 
        'static_raw', 
        'static_clean',
        'static_rule_one',
        'static_rule_two',
        'static_rule_three',
        'static_zones',
        'static_alerts',
        'dynamic_raw', 
        'dynamic_clean',
        'dynamic_rule_one',
        'dynamic_rule_two',
        'dynamic_rule_three',
        'dynamic_zones',
        'dynamic_alerts']

    # iter each var and add as nans to xr, if not exist
    for new_var in new_vars:
        if new_var not in ds:
            ds[new_var] = xr.full_like(ds['veg_idx'], np.nan)
            
    return ds


# deprecated meta, check the checks
def combine_old_new_xrs(ds_old, ds_new):
    """"""
    
    # check if old provided, else return new
    if ds_old is None:
        print('Old dataset is empty, returning new.')
        return ds_new
    elif ds_new is None:
        print('New dataset is empty, returning old.')
        return ds_old
        
    # check time dimension
    if 'time' not in ds_old or 'time' not in ds_new:
        print('Datasets lack time coordinates.')
        return
    
    # combien old with new
    try:
        ds_cmb = xr.concat([ds_old, ds_new], dim='time')
    except:
        print('Could not combine old and new datasets.')
        return
    
    return ds_cmb
    
    
# deprecated 
def smooth_signal(da):
    """
    Basic func to smooth change signal using
    a savgol filter with win size 3. works best
    for our data. minimal smoothing, yet removes
    small spikes.
    """
    
    # check if data array 
    if not isinstance(da, xr.DataArray):
        print('Only numpy arrays supported, returning original array.')
        return da
    elif len(da) <= 3:
        print('Cannot smooth array <= 3 values. Returning raw array.')
        return da
    
    try:
        # smooth using savgol filter with win size 3
        da_out = xr.apply_ufunc(savgol_filter, da, 3, 1)
        return da_out
    except:
        print('Couldnt smooth, returning raw array.')
    
    return da
    
    
# deprecated
def append_xr_site_attrs(ds, feat):
    """
    Takes a xarray Dataset and appends monitoring area
    attributes to it. Requires a dataset (xarray type) and
    a list or tuple of monitoring area feature values, 
    typically captured from a row via arcpy or ogr.
    """
    
    # check if ds is dataset
    if ds is None:
        raise TypeError('Dataset not an xarray type.')
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    
    # check if dict is right length
    if not isinstance(feat, (list, tuple)):
        print('Feature not a list or tuple, returning original dataset.')
        return ds
    elif len(feat) == 0:
        print('Feature has no data, returning original dataset.')
        return ds

    try:
        # build dict
        attrs = {
            'area_id':             str(feat[0]),
            'platform':            str(feat[1]),
            's_year':              str(feat[2]),
            'e_year':              str(feat[3]),
            'index':               str(feat[4]),
            'persistence':         str(feat[5]),
            'rule_1_min_conseqs':  str(feat[6]),
            'rule_1_inc_plateaus': str(feat[7]),
            'rule_2_min_zone':     str(feat[8]),
            'rule_3_num_zones':    str(feat[9]),
            'ruleset':             str(feat[10]),
            'alert':               str(feat[11]),
            'method':              str(feat[12]),
            'alert_direction':     str(feat[13]),
            'email':               str(feat[14]),
            'ignore':              str(feat[15])
        }
    
        # update attributes on dataset
        ds.attrs = attrs
    
    except:
        raise ValueError('Could not create attribute dictionary.')

    return ds
    
    
# deprecated meta
def create_email_dicts(row_count):
    """meta"""
    
    if row_count is None or row_count == 0:
        print('No rows to build email dictionaries.')
        return
    
    # setup email contents list
    email_contents = []
            
    # pack list full of 'empty' dicts
    for i in range(row_count):
    
        # set up empty email dict
        email_dict = {
            'area_id': None,
            's_year': None,
            'e_year': None,
            'index': None,
            'ruleset': None,
            'alert': None,
            'alert_direction': None,
            'email': None,
            'ignore': None,
            'triggered': None
        }
    
        email_contents.append(email_dict)
        
    return email_contents


# deprecated todo checks, meta
def send_email_alert(sent_from=None, sent_to=None, subject=None, body_text=None, smtp_server=None, smtp_port=None, username=None, password=None):
    """
    """
    
    # check sent from
    if not isinstance(sent_from, str):
        raise TypeError('Sent from must be string.')
    elif not isinstance(sent_to, str):
        raise TypeError('Sent to must be string.')
    elif not isinstance(subject, str):
        raise TypeError('Subject must be string.')
    elif not isinstance(body_text, str):
        raise TypeError('Body text must be string.')     
    elif not isinstance(smtp_server, str):
        raise TypeError('SMTP server must be string.')        
    elif not isinstance(smtp_port, int):
        raise TypeError('SMTP port must be integer.')
    elif not isinstance(username, str):
        raise TypeError('Username must be string.')     
    elif not isinstance(password, str):
        raise TypeError('Password must be string.')
        
    # notify
    print('Emailing alert.')
    
    # construct header text
    msg = MIMEMultipart()
    msg['From'] = sent_from
    msg['To'] = sent_to
    msg['Subject'] = subject

    # construct body text (plain)
    mime_body_text = MIMEText(body_text)
    msg.attach(mime_body_text)

    # create secure connection with server and send
    with smtplib.SMTP(smtp_server, smtp_port) as server:

        # begin ttls
        server.starttls()

        # login to server
        server.login(username, password)

        # send email
        server.sendmail(sent_from, sent_to, msg.as_string())

        # notify
        print('Emailed alert area.')
                
    return


# deprecated need params from feat, conseq count e.g., 3)
def prepare_and_send_alert(ds, back_idx=-2, send_email=False):
    """
    ds = change dataset with required vars
    back_idx = set backwards index (-1 is latest image, -2 is second last, etc)
    """
    
    # check if we have all vars required

    # get second latest date
    latest_date = ds['time'].isel(time=back_idx)
    latest_date = latest_date.dt.strftime('%Y-%m-%d %H:%M:%S')
    latest_date = str(latest_date.values)

    # get latest zone
    latest_zone = ds['zones'].isel(time=back_idx)
    latest_zone = latest_zone.mean(['x', 'y']).values

    # get latest incline candidate
    latest_inc_candidate = ds['cands_inc'].isel(time=back_idx)
    latest_inc_candidate = latest_inc_candidate.mean(['x', 'y']).values

    # get latest decline candidate
    latest_dec_candidate = ds['cands_dec'].isel(time=back_idx)
    latest_dec_candidate = latest_dec_candidate.mean(['x', 'y']).values

    # get latest incline consequtives
    latest_inc_consequtives = ds['consq_inc'].isel(time=back_idx)
    latest_inc_consequtives = latest_inc_consequtives.mean(['x', 'y']).values

    # get latest incline consequtives
    latest_dec_consequtives = ds['consq_dec'].isel(time=back_idx)
    latest_dec_consequtives = latest_dec_consequtives.mean(['x', 'y']).values
    
    
    # alert user via ui and python before email
    if latest_inc_candidate == 1:
        print('- ' * 10)
        print('Alert! Monitoring Area {} has triggered the alert system.'.format('<placeholder>'))
        print('An increasing vegetation trajectory has been detected.')
        print('Alert triggered via image captured on {}.'.format(str(latest_date)))
        print('Area is in zone {}.'.format(int(latest_zone)))
        print('Increase has been on-going for {} images (i.e., dates).'.format(int(latest_inc_consequtives)))       
        print('')

        
    elif latest_dec_candidate == 1:
        print('- ' * 10)
        print('Alert! Monitoring Area {} has triggered the alert system.'.format('<placeholder>'))
        print('An decreasing vegetation trajectory has been detected.')
        print('Alert triggered via image captured  {}.'.format(str(latest_date)))
        print('Area is in zone {}.'.format(int(latest_zone)))
        print('Decrease has been on-going for {} images (i.e., dates).'.format(int(latest_dec_consequtives)))
        print('')  
        
    else:
        print('- ' * 10)
        print('No alert was triggered for Monitoring Area: {}.'.format('<placeholder>'))
        print('')
        
    # if requested, send email
    if send_email:
        print('todo...')


# deprecated meta, stable zone?
def get_stdv_from_zone(num_zones=1):
    """
    """
    
    # checks
    if num_zones is None or num_zones <= 0:
        print('Number of zones must be greater than 0. Setting to 1.')
        num_zones = 1
    
    # multiple by zones (3 per zone)
    std_jumped = num_zones * 3
    
    # todo include stable zone -1 to 1?
    #
    
    return std_jumped


