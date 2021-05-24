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