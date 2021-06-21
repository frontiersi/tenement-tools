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
import dask.array as dask_array
import numpy as np
import pandas as pd
import joblib
from osgeo import ogr, osr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from dask_ml.wrappers import ParallelPostFit
from sklearn.ensemble import RandomForestRegressor

sys.path.append('../../shared')
import tools

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
    
    # notify
    print('Prepared classified dataset successfully.')
    return ds


def reclassify_xr(ds, req_class, nodata_value=-999, inplace=True):
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
    nodata_value : int or float
        A value indicating the NoData values within xarray datatset.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.    

    Returns
    ----------
    ds: pxarray dataset/array
        A reclassified xr dataset/array.
    """
    
    # notify
    print('Reclassifying classes.')
    
    # check xr type, dims in ds a
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset a not an xarray type.')    
        
    # check if req classes is list or int, convert to list
    req_class = req_class if req_class is not None else []
    req_classes = req_class if isinstance(req_class, list) else [req_class]
    
    # create copy ds if not inplace
    if not inplace:
        ds = ds.copy(deep=True)
    
    # add nodata to required classes
    classes = req_classes + [nodata_value]
        
    # remove unrequested classes
    ds = ds.where(ds.isin(classes), 0)
    
    # notify
    print('Reclassified dataset successfully.')
    return ds


def get_xr_classes(ds, nodata_value=-9999):
    """
    Takes an xarray dataset/array and extracts unique class
    values from dataset values. Returns a list of class values.

    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.
    nodata_value : int or float
        A value representing the nodata values in dataset.

    Returns
    ----------
    np_classes : numpy ndarray.
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

    # get all unique classes in dataset and remove nodata
    np_classes = np.unique(ds.to_array())
    np_classes = np_classes[np_classes != nodata_value]
    
    # check if something came back or too much came back
    if len(np_classes) <= 0:
        raise ValueError('No classes detected in dataset.')
    elif len(np_classes) > 100:
        print('Warning: >= 100 classes detected in dataset. Proceed with caution.')
        
    # notify and return
    str_classes = ', '.join([str(c) for c in np_classes])
    print('Detected classes in dataset: {}'.format(str_classes))
    return np_classes


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
    if not isinstance(ds_raw, (xr.Dataset, xr.DataArray)):
        raise ValueError('Raw dataset is not an xarray dataset.')
        
    # check if class dataset is xarray dataeset type
    if not isinstance(ds_class, (xr.Dataset, xr.DataArray)):
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


# fix up meta and checks, reduce code
def create_frequency_windows(ds_raw, ds_class, df_records):
    """
    """

    # notify user
    print('Creating frequency focal windows from random sample points.')

    # check if xarray dataset
    if not isinstance(ds_raw, xr.Dataset):
        raise TypeError('Not a xarray dataset type.')

    # check if pandas array
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Not a pandas dataframe type.')

    # check if no data value is correct
    #if type(nodata_value) not in [int, float]:
        #raise TypeError('> NoData value is not an int or float.')

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

    # get original data type and compute
    np_windows = dask_array.concatenate(window_list).compute()

    # reset index in case rows removed
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
        print('{0} windows generated successfully.'.format(len(df_records)))
    else:
        print('{0} windows generated successfully. {1} omitted.'.format(len(df_records) - skip_count, skip_count))

    # return
    return df_windows

# todo metatdat and checks - is -9999 order important? maybe move to front
def convert_window_counts_to_freqs(df_windows, nodata_value=-9999):
    """
    Take an datframe of pixels in windows, count the classes and counts in each,
    and transform them into numpy classes and frequencies (or proportions) within
    each window.
    """
    
    # notify user
    print('Generating arrays of unique classes and their occurrence frequencies from window pixels.')
    
    # checks
    # check if column called win_vals exists
    
    # create empty class label and freqs columns
    df_windows['class_lbls'], df_windows['class_frqs'] = '', ''

    # iter each row and calc unique classes, counts and freqs
    for i, r in df_windows.iterrows():
        
        # if valid...
        if isinstance(r['win_vals'], np.ndarray) and len(r['win_vals']) > 0:
            
            # get dtype of classes (labels)
            lbl_dtype = r['win_vals'].dtype

            # get size of window
            win_size = len(r['win_vals'])

            # get unique classes, counts, frequencies
            lbls, cnts = np.unique(r['win_vals'], return_counts=True)
            frqs = cnts / win_size

            # append nodata class and freq if missing
            if nodata_value not in lbls:
                lbls = np.insert(lbls, 0, nodata_value)
                frqs = np.insert(frqs, 0, 0.0)

            # add each numpy array to current row
            df_windows.at[i, 'class_lbls'] = lbls.astype(lbl_dtype)
            df_windows.at[i, 'class_frqs'] = frqs.astype(np.float32)

        else:
            # reset values to nan for easy drop later on
            df_windows.at[i, 'class_lbls'] = np.nan
            df_windows.at[i, 'class_frqs'] = np.nan
            
    # notify user
    print('Checking for empty rows and dropping if exist.')
            
    # count nan values for labels and freqs
    lbls_nan_count = df_windows['class_lbls'].isna().sum()
    frqs_nan_count = df_windows['class_frqs'].isna().sum()

    # if nan, tell user and remove rows
    if lbls_nan_count > 0 or frqs_nan_count > 0:
        print('Empty rows detected. Dropping {0} rows.'.format(np.max(lbls_nan_count, frqs_nan_count)))
        df_windows.dropna(subset=['class_lbls', 'class_frqs'])
        
    # drop unneeded boundary cols
    df_windows = df_windows.drop(columns=['win_vals'])
    
    # reset index in case rows removed
    df_windows = df_windows.reset_index(drop=True)
    
    # notify user
    if len(df_windows) > 0:
        print('{0} windows transformed to frequencies successfully.'.format(len(df_windows)))
    else:
        raise ValueError('No frequencies were calculated. Aborting.')
           
    # return
    return df_windows

# metadata, - weird function. can it go to another?
def prepare_freqs_for_analysis(ds_raw, ds_class, df_freqs, override_classes=[], nodata_value=-9999):
    """
    This function takes a dataset with classified pixels and determines if 
    samples taken from it within a pandas dataframe column are missing (i.e. 
    were not sampled.) Can warn or thrown an error if missing classes in samples
    is detected.
    """

    # notify user
    print('Converting dataset to sklearn analysis-ready format.')

    # checks

    # notify 
    print('> Checking classified raster and sampled classes match.')

    # get class datatype
    class_dtype = ds_class.to_array().dtype

    # prepare classes we want for analysis dataframe - override or from dataset
    if len(override_classes) > 0:
        np_dataset_classes = np.unique((np.array(override_classes, dtype=class_dtype)))    
    else:
        np_dataset_classes = np.unique(ds_class.to_array())

    # check if no data col exists, if not, add it
    if nodata_value not in np_dataset_classes:
        np_dataset_classes = np.insert(np_dataset_classes, 0, nodata_value)

    # get unique class values sampled from windows
    np_sampled_classes = np.unique(np.concatenate(df_freqs['class_lbls']))

    # get missing classes - convert to strings to be safe
    np_missing = np.setdiff1d(np_dataset_classes, np_sampled_classes)

    # check if anything exists, throw error if so
    if len(np_missing) > 0:
        raise ValueError('> One or more requested classes not captured within windows.')

    # notify 
    print('> Converting classes and frequencies into analysis-ready format.')

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
    print('> Normalising rows with regards to subset classes and NoData values.')

    # normalise frequencies within row
    for i, r in df_data.iterrows():
        df_data.at[i] = r / (1 - r[nodata_value])

    # notify
    print('> Creating NoData mask.')

    # we create an include column and base it on value in nodata value col
    df_data['include'] = df_data[nodata_value] <= 0

    # drop nodata col
    df_data = df_data.drop(columns=[nodata_value])
    
    # ensure dataframes not empty
    if df_data.isnull().values.all() or df_coords.isnull().values.all():
        raise ValueError('> Dataframe(s) empty or all rows null.')

    # add x and y columns on to data frame
    df_data = pd.concat([df_coords, df_data], axis=1)

    # notify and return
    print('> Data successfully prepared for analysis.')
    return df_data

# needs meta
def perform_optimised_fit(estimator, X, y, parameters, cv=10):
    """
    """
    
    # check for tyoe
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('> X and/or y inputs must be numpy arrays.')
    
    # check structure
    if not len(X.shape) == 2 or not len(y.shape) == 1:
        raise ValueError('> X and/or y inputs are of incorrect size.')
                
    # check parameters
    if not isinstance(parameters, dict):
        raise TypeError('> Parameters must be in a dictionary type.')
        
    # check entered parameters
    allowed_parameters = ['max_features', 'max_depth', 'n_estimators']
    for k, p in parameters.items():
        if k not in allowed_parameters:
            raise ValueError('> Parameter: {0} not supported.'.format(k))
            
    # check cv
    if cv <= 0:
        raise ValueError('> CV (cross-validation) must be > 0.')
        
    # create grid search cv and fit it
    gsc = GridSearchCV(estimator, parameters, cv=cv, n_jobs=-1, scoring='max_error')
    gsc_result = gsc.fit(X, y)
        
    return gsc_result

# do checks
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
            print('> No attributes available. Skipping.')
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

# needs meta, checks
def perform_optimised_validation(estimator, X, y, n_validations=10, split_ratio=0.25):
    """
    """
    
    # do checks
    
    # create array to hold validation results
    r2_list = []
    
    # iterate n validations
    for i in range(0, n_validations):
        
        # split X and y data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, 
                                                            random_state=0, shuffle=True)
        
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
    
# metadata
def perform_fca(ds_raw, ds_class, df_data, df_extract_clean, grid_params, validation_iters, nodata_value=-9999):
    """
    """
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
