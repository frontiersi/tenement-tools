# deprecated
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