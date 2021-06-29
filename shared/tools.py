"""
"""

# import required libraries
import os, sys
import gdal
import numpy as np
import pandas as pd
import xarray as xr


# meta
def calculate_indices(ds, index=None, custom_name=None, rescale=False, drop=False):
    """
    """
        
    # notify
    print('Calculating indices: {0}.'.format(index))
    
    # attempt da convert to ds, check for ds
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Please provide a Dataset.')
    
    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')
                                        
    # prepare index, custom names if empty
    index = index if index is not None else []
    custom_name = custom_name if custom_name is not None else []
            
    # if not none but not list, prepare also
    indices = index if isinstance(index, list) else [index]
    custom_names = custom_name if isinstance(custom_name, list) else [custom_name]
    
    # check if custon names same length as index, if provided
    if custom_names and len(indices) != len(custom_names):
        raise ValueError('Custom names must be provided for all indexes.')
                        
    # get pre-processing band names for drop later
    drop_bands = list(ds.data_vars)
    
    # create copy ds
    ds = ds.copy(deep=True)
    
    # calc index/indices
    for index in indices:
        try:
            # notify
            print('Calculating index: {0}'.format(index))
            
            # ndvi
            if index == 'ndvi':
                ds['ndvi'] = ((ds.nir - ds.red) / 
                              (ds.nir + ds.red))

            # evi with normalise
            elif index == 'evi':
                f = lambda ds: ((2.5 * (ds.nir - ds.red)) / 
                                (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
                ds['evi'] = f(ds / 10000.0)

            # savi with normalise
            elif index == 'savi':       
                f = lambda ds: ((1.5 * (ds.nir - ds.red)) /
                                (ds.nir + ds.red + 0.5))
                ds['savi'] = f(ds / 10000.0)

            # msavi with normalise
            elif index == 'msavi':       
                f = lambda ds: ((2 * ds.nir + 1 - ((2 * ds.nir + 1)**2 - 
                                 8 * (ds.nir - ds.red))**0.5) / 2)
                ds['msavi'] = f(ds / 10000.0)

            # slavi
            elif index == 'slavi':
                ds['slavi'] = ((ds.nir) / 
                               (ds.red + ds.swir2))

            # mavi
            elif index == 'mavi':
                ds['mavi'] = ((ds.nir - ds.red) / 
                              (ds.nir + ds.red + ds.swir1))
                
            # kndvi (non-linear ndvi)
            elif index == 'kndvi':
                ds['kndvi'] = np.tanh(((ds.nir - ds.red) /
                                       (ds.nir + ds.red)) ** 2)

            # tass cap greenness (Crist 1985 coeffs) with normalise
            elif index == 'tcg':        
                f = lambda ds: ((-0.1603 * ds.blue + -0.2819 * ds.green +
                                 -0.4934 * ds.red + 0.7940 * ds.nir +
                                 -0.0002 * ds.swir1 + -0.1446 * ds.swir2))
                ds['tcg'] = f(ds / 10000.0)

            # tass cap brightness (Crist 1985 coeffs) with normalise
            elif index == 'tcb':
                f = lambda ds: ((0.2043 * ds.blue + 0.4158 * ds.green +
                                 0.5524 * ds.red + 0.5741 * ds.nir +
                                 0.3124 * ds.swir1 + -0.2303 * ds.swir2))
                ds['tcb'] = f(ds / 10000.0)

            # tass cap wetness (Crist 1985 coeffs) with normalise
            elif index == 'tcw':       
                f = lambda ds: ((0.0315 * ds.blue + 0.2021 * ds.green +
                                 0.3102 * ds.red + 0.1594 * ds.nir +
                                 -0.6806 * ds.swir1 + -0.6109 * ds.swir2))
                ds['tcw'] = f(ds / 10000.0)

            # ndmi
            elif index == 'ndmi':
                ds['ndmi'] = ((ds.nir - ds.swir1) / 
                              (ds.nir + ds.swir1))

            # gvmi
            elif index == 'gvmi':
                ds['gvmi'] = (((ds.nir + 0.1) - (ds.swir2 + 0.02)) / 
                              ((ds.nir + 0.1) + (ds.swir2 + 0.02)))

            # nbr
            elif index == 'nbr':
                ds['nbr'] = ((ds.nir - ds.swir2) /
                            (ds.nir + ds.swir2))
            
        except:
            raise ValueError('Could not calculate: {0}. Please check bands.'.format(index))
            
    # rescale -1 to 1 -> 0 to 2 if requested
    if rescale:
        for index in indices:
            if index not in ['slavi', 'kndvi', 'tcg', 'tcb', 'tcw', 'nbr']:
                ds[index] = ds[index] + 1
        print('Rescaled index values from -1 to 1 -> 0 to 2.')

    # rename bands if requested
    if custom_names:
        try:
            ds = ds.rename(dict(zip(indices, custom_names)))
            print('Renamed default indices.')
        except:
            print('Warning: could not rename bands. Please check.')
            
    # drop original bands if requested
    if drop:
        ds = ds.drop(drop_bands, errors='ignore')

    # convert back to datarray
    if was_da:
        ds = ds.to_array()
            
    # notify and return
    print('Calculated indices successfully.')
    return ds


def perform_interp(ds, method='full'):
    """
    Takes a xarray dataset/array and performs linear interpolation across
    time dimension. The method can be set to full or half. Full will use the 
    built in xr interpolate_na method, which is robust and dask friendly 
    but very slow. The quicker alternative is half, which only interpolates 
    times that are all nan. Despite being slower, full method recommended.

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
    
    # interpolate using full or half method
    if method == 'full':

        # if dask, rechunk into appropriate shape
        if bool(ds.chunks):
            chunks = ds.chunks
            ds = ds.chunk({'time': -1}) 

        # interpolate all nan pixels linearly
        ds = ds.interpolate_na(dim='time', method='linear')

        # chunk back to orginal size
        if bool(ds.chunks):
            ds = ds.chunk(chunks) 

    elif method == 'half':
        
        # get times where all nan, find diff with original, drop if exist
        nan_dates = ds.dropna(dim='time', how='all').time
        nan_dates = np.setdiff1d(ds['time'], nan_dates)
        ds = ds.dropna(dim='time', how='all')

        # interpolate all nan pixels linearly
        if len(nan_dates):
            ds_interp = ds.interp(time=nan_dates, method='linear')
            ds = xr.concat([ds, ds_interp], dim='time').sortby('time')
            
    return ds


# meta
def get_linear_orpol_contrasts(levels=3):
    """
    """
    
    # check levels
    if not levels:
        raise ValueError('Must provide levels value.')

    # get constant: 1 if odd, 2 if even
    const = 1 if levels % 2 else 2

    # generate line depending on odd, even 
    if const == 1:
        ends = np.floor(levels / 2)
    else:
        ends = levels - 1

    # calc linear orthog poly coeffs
    coeffs = np.linspace(-ends, ends, levels)

    # get sum of squares
    ss = int(np.sum(coeffs**2))
    
    # notify
    print('Got orthogonal polynomial coefficients: {0}'.format(coeffs))
    print('Got sum of squares: {0} and constant {1}: '.format(ss, const))

    return coeffs, ss, const


# meta
def read_shapefile(shp_path=None):
    """
    Read records from a projected ESRI Shapefile table and extracts the field values
    located within. This must be a point geometry-type dataset and it must be projected in
    the GDA94 Albers projection system (EPSG 3577).

    Parameters
    ----------
    shp_path : string
        A single string with full path and filename of shapefile.

    Returns
    ----------
    df_records : pandas dataframe
        A pandas dataframe containing all columns and rows within
        a shapefile.
    """
    
    # imports check
    try:
        from osgeo import ogr
    except:
        raise ImportError('Could not import osgeo.')

    # notify user
    print('Reading records within shapefile.')

    # check if string, if not bail
    if not isinstance(shp_path, str):
        raise ValueError('Shapefile path must be a string. Please check the file path.')

    # check if shp exists
    if not os.path.exists(shp_path):
        raise OSError('Unable to read shapefile. Please check the file path.')

    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()

        # get epsg code
        epsg = int(lyr.GetSpatialRef().GetAttrValue('AUTHORITY', 1))

        # get num feats
        num_feats = lyr.GetFeatureCount()

    except Exception:
        raise TypeError('Could not read shapefile records. Is the file corrupt?')

    # check if point/multi point type
    if lyr.GetGeomType() not in [ogr.wkbPoint, ogr.wkbMultiPoint]:
        raise ValueError('Shapefile is not a point/multi-point type.')

    # check if shapefile is empty
    if num_feats == 0:
        raise ValueError('Shapefile has no features in it. Please check.')

    # check if shapefile is projected (i.e. in metres)
    if epsg != 3577:
        raise ValueError('Shapefile is not projected in GDA94 Albers. Please reproject into EPSG: 3577.')

    # convert shapefile table to pandas df
    rows = []
    for row in lyr:
        
        # get geom
        geom = row.GetGeometryRef()

        # convert row to dict
        row = row.items()
        
        # get x and y of individual point type
        if geom.GetGeometryName() == 'POINT':
            x, y = geom.GetX(), geom.GetY()

        # get x and y of each point in multipoint type
        elif geom.GetGeometryName() == 'MULTIPOINT':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)   
                x, y = sub_geom.GetX(), sub_geom.GetY()

        # update row dict with x, y
        row.update({'x': x, 'y': y})

        # add to rows list
        rows.append(row)

    # check if list is populated
    if not rows:
        raise ValueError('No rows were obtained from shapefile.')

    # convert list of dicts to dataframe
    df_records = pd.DataFrame.from_dict(rows)

    # drop variables
    shp, lyr = None, None

    # notify user and return
    print('Rows read from shapefile successfully.')
    return df_records


# meta
def subset_records(df_records, p_a_column=None):
    """
    takes a pandas dataframe with all columns and limits it down 
    to occurrence field set by user, plus x, y columns.
    if no occurrence column name set, all points considered to be
    presence (1).

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

    # check if pres abse column exists
    if p_a_column and p_a_column not in df_records:
        raise ValueError('Presence column: {0} not in dataframe.'.format(p_a_column))

    # create copy of dataframe
    df_records = df_records.copy()

    # subset dataframe depending on user requests
    if p_a_column:
        df_records = df_records[['x', 'y', p_a_column]]
    else:
        df_records = df_records[['x', 'y']]
        df_records['actual'] = 1

    # rename requested occurrence column, if exists
    if p_a_column in df_records:
        df_records = df_records.rename(columns={p_a_column: 'actual'})

    # remove non 1, 0 values, notify user 
    if 'actual' in df_records:
        old_num = df_records.shape[0]
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


# meta
def intersect_records_with_xr(ds, df_records, extract=False, res_factor=3, if_nodata='any'):
    """
    Takes a pandas dataframe of occurrence records and clips them
    to a xarray dataset or array. It is recommended a binary mask is used 
    as ds input, with nan represented by nodata_value paramter. if_nodata = if any vars in pixel
    are nan, ignore point, if all are, ignore.

    Parameters
    ----------
    ds: xarray dataset
        A dataset with data variables.
    df_records : pandas dataframe
        A pandas dataframe containing at least an x and y columns 
        with records.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_records : pandas dataframe
    """

    # notify user
    print('Clipping pandas dataframe records to xarray dataset.')

    # check if a pandas dataframe
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Records is not a pandas dataframe type.')
            
    # check if x, y exists in df
    if 'x' not in df_records or 'y' not in df_records:
        raise ValueError('No x, y values in records.')
        
    # check if any records exist
    if df_records.shape[0] == 0:
        raise ValueError('No records in pandas dataframe.')
                
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
        
    # check if nodatavals is in dataset
    if not hasattr(ds, 'nodatavals') or ds.nodatavals == 'unknown':
        raise AttributeError('Dataset does not have a nodatavalue attribute.')

    # check if res factor is int type
    if not isinstance(res_factor, int) and res_factor < 1:
        raise TypeError('Resolution factor must be an integer of 1 or greater.')
        
    # check if_nodata
    if if_nodata not in ['any', 'all']:
        raise ValueError('The if_nodata parameter must be all or any.')

    # get cell resolution from dataset, check exists
    res = get_xr_resolution(ds)
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
            pixel = [ds.nodatavals] * len(ds.data_vars)
            
        # check if any/all pixel vars have nodata
        if if_nodata == 'any':
            is_nodata = np.any(np.isnan(pixel) | (pixel == ds.nodatavals))
        elif if_nodata == 'all':
            is_nodata = np.all(np.isnan(pixel) | (pixel == ds.nodatavals))
        
        # append to list if not nodata and add pixel info if extract
        if not is_nodata:
            row = row.to_list() + pixel if extract else row.to_list()
            values.append(row)

    # get original df field dtypes as dict
    df_dtypes_dict = dict(df_records.dtypes)

    # get original ds var dtypes as dict
    ds_dtypes_dict = {}
    for var in ds.data_vars:
        ds_dtypes_dict[var] = ds[var].dtype
        
    # combine dicts depending on extract
    if extract:
        col_names = list(df_records) + list(ds.data_vars)
        col_dtypes = {**df_dtypes_dict, **ds_dtypes_dict}
    else:
        col_names, col_dtypes = list(df_records), df_dtypes_dict
        
    try:
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


#!!!!!!!! I split this into read_shapefile and extract_occurence_data
# todo: remove when sdm code updated
def read_coordinates_shp(shp_path=None, p_a_column=None):
    """
    Read observation records from a projected ESRI Shapefile and extracts the x and y values
    located within. This must be a point geometry-type dataset and it must be projected in
    the GDA94 Albers projection system (EPSG 3577).

    Parameters
    ----------
    shp_path : string
        A single string with full path and filename of shapefile.
    p_a_column : string
        A column of presence/absence values (1, 0). If any value other than
        1, 0 exists, error thrown. Do not enter a name to use every point in
        the dataset as presence (default).

    Returns
    ----------
    df_presence : pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """
    
    # imports check
    try:
        from osgeo import ogr
    except:
        raise ImportError('Could not import osgeo.')
    
    # notify user
    print('Reading species point locations from shapefile.')

    # check if string, if not bail
    if not isinstance(shp_path, str):
        raise ValueError('> Shapefile path must be a string. Please check the file path.')

    # check if shp exists
    if not os.path.exists(shp_path):
        raise OSError('> Unable to read species point locations. Please check the file path.')

    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()

        # get epsg code
        epsg = int(lyr.GetSpatialRef().GetAttrValue('AUTHORITY', 1))

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
    if epsg != 3577:
        raise ValueError('> Shapefile is not projected in GDA94 Albers. Please reproject into EPSG: 3577.')
        
    # check if pres/abse column exists if requested

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


# meta, check if we can just use geobox
def get_xr_resolution(ds):
    """
    Read dataset and get pixel resolution from attributes. If 
    attributes don't exist, fall back to rough approach of minus 
    one pixel to another.

    Parameters
    ----------
    ds: xarray dataset, dataarray
        A single xarray dataset with variables and x and y dims.

    Returns
    ----------
    res : float
        A float containing the cell resolution of xarray dataset.
    """

    # notify user
    print('Extracting cell resolution from dataset.')

    # check if x and y dims exist
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x, y dimensions in dataset.')

    # try getting max res option 1
    try:
        res = abs(max(ds.res))
    except:
        res = None

    # try getting max res option 2
    try:
        if not res:
            res = max(ds.geobox.resolution)
    except:
        res = None

    # try getting max res the dirty way
    try:
        if not res:
            x_res = abs(float(ds['x'].isel(x=0))) - abs(float(ds['x'].isel(x=1)))
            y_res = abs(float(ds['y'].isel(y=0))) - abs(float(ds['y'].isel(y=1)))
            res = abs(float(max(x_res, y_res)))
    except:
        res = None

    # check if something exists
    if not res:
        raise ValueError('Could not extract cell resolution from dataset.')

    # return
    return res


# meta, check if we can just use geobox
def get_xr_crs(ds):
    """
    Read dataset and get crs from attributes. If attributes don't 
    exist, thrown an error.

    Parameters
    ----------
    ds: xarray dataset, dataarray
        A single xarray dataset with variables and x and y dims.

    Returns
    ----------
    crs : int
        A int containing the crs of xarray dataset.
    """

    # notify user
    print('Extracting CRS from dataset.')

    # try getting crs option 1
    try:
        crs = ds.crs
        crs = int(''.join(e for e in crs if e.isdigit()))
    except:
        crs = None

    # try getting crs option 2
    try:
        if not crs:
            crs = int(ds.geobox.crs.epsg)
    except:
        crs = None

    # check if something exists
    if not crs:
        raise ValueError('Could not extract crs from dataset.')

    # notify user and return
    print('CRS extracted successfully from dataset.')
    return crs


def get_xr_extent(ds):
    """
    Read dataset and get raster bounds using dataset method. If fails, do it the
    long, less sophisticated way.

    Parameters
    ----------
    ds: xarray dataset
        A single xarray dataset with variables and x and y dims.
    Returns
    ----------
    extent : dict
        A dict containing the datasets l, b, r, t bounding box coordinates.
    """
    
    # notify user
    print('Extracting spatial extent from dataset.')

    # check if xarray dataset type
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Must provide an xarray dataset/datarray.')
    elif 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('No x and/or y coordinate dimension in dataset.')
        
    try:
        # get bounding box object and convert to dict
        bb = ds.extent.boundingbox
        extent = {'l': bb.left, 'b': bb.bottom, 'r': bb.right, 't': bb.top}
        return extent
        
    except:
        pass
    
    # get bounds the long, hacky, slightly less accurate way
    l, r = float(ds['x'].min()), float(ds['x'].max())
    b, t = float(ds['y'].min()), float(ds['y'].max())    
    
    # store in dict
    extent = {'l': l, 'b': b, 'r': r, 't': t}
    return extent
    

def remove_nodata_records(df_records, nodata_value=np.nan):
    """
    Read a numpy record array and remove -9999 (nodata) records.

    Parameters
    ----------
    df_records: pandas dataframe
        A pandas dataframe type containing values extracted from env variables.
    nodata_value : int, float, np.nan
        A int, float or nan indicating the no data value expected. Default is nan.

    Returns
    ----------
    df_records : numpy record array
        A numpy record array without nodata values.
    """

    # notify user
    print('Removing records containing NoData ({0}) values.'.format(nodata_value))

    # check if numpy rec array
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Not a pands dataframe type.')

    # check if no data value is correct
    if type(nodata_value) not in [int, float, np.nan]:
        raise TypeError('NoData value is not an int, float, nan.')
        
    # get original num records
    orig_num_recs = df_records.shape[0]

    # remove any record containing nodata value
    nan_mask = (df_records.eq(nodata_value) | df_records.isnull())
    df_records = df_records[~nan_mask.any(axis=1)]

    # check if array exists
    if df_records.shape[0] == 0:
        raise ValueError('No valid values remain in data after removal of NoData.')
        
    # get num of records removed
    num_removed = orig_num_recs - df_records.shape[0]

    # notify user and return
    print('Removed {0} records containing NoData values successfully.'.format(num_removed))
    return df_records


def clip_xr_to_xr(ds_a, ds_b, inplace=True):
    """
    Takes an xarray dataset (ds_a) and clips it to match
    the extent of ds_b.

    Parameters
    ----------
    ds_a: xarray dataset/array
        A dataset which will be clipped to match ds_b.
    ds_b: xarray dataset/array
        A dataset to which the extents of ds_a will be
        clipped to.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds_a : xarray dataset/array to match original ds_a.
    """
    
    # notify
    print('Clipping dataset to another.')

    # check xr type, dims in ds a
    if not isinstance(ds_a, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset a not an xarray type.')
    elif 'x' not in list(ds_a.dims) and 'y' not in list(ds_a.dims):
        raise ValueError('No x and/or y coordinate dimension in dataset a.')
        
    # check xr type, dims in ds b
    if not isinstance(ds_b, (xr.Dataset, xr.DataArray)):
        raise TypeError('Dataset b not an xarray type.')
    elif 'x' not in list(ds_b.dims) and 'y' not in list(ds_b.dims):
        raise ValueError('No x and/or y coordinate dimension in dataset b.')   
    
    # create copy ds if not inplace
    if not inplace:
        ds_a = ds.copy(deep=True)
        
    # get extent of ds b
    extent = get_xr_extent(ds_b)
    
    # subset ds high to ds low
    ds_a = ds_a.sel(x=slice(extent.get('l'), extent.get('r')), 
                    y=slice(extent.get('t'), extent.get('v')))
    
    # notify
    print('Clipped dataset successfully.')
    return ds_a


# meta
def export_xr_as_nc(ds, filename):
    """
    """
    
    # notify
    print('Exporting xarray as netcdf file.')
    
    # check if xr ds or da
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError('Only xarray dataset/data array supported.')
        
    # check if attrs exist
    if not hasattr(ds, 'attrs'):
        print('Warning: xarray is missing attributes.')
    elif not hasattr(ds, 'geobox'):
        print('Warning: xarray is missing geobox.')
        
    # check if filename valid
    if not filename:
        raise ValueError('Did not provide filename.')
    
    # check file extension given
    name, ext = os.path.splitext(filename)
    if ext != '.nc':
        raise ValueError('Filename must include .nc extension.')
        
    try:
        # write xr to netcdf
        ds.to_netcdf(filename)
        
    except Exception as e:
        print('Could not export')
        raise ValueError('Could not export netcdf: {0}.'.format(e))
        
    # notify
    print('Exported xarray as netcdf successfully.')
    
    
# vegfrax, older
# todo, long in tooth, move these into funcs when needed
def extract_rast_info(rast_path):
    """
    Read a raster (e.g. tif) and extract geo-transformation, coordinate 
    system, projection type, size of dimensions (x and y), nodata value.

    Parameters
    ----------
    rast_path: string
        A single string with full path and filename of a raster.

    Returns
    ----------
    rast_meta_dict : dictionary with keys:
        layer = name of layer
        type = type of data, vector or raster.
        geo_trans = raster geotransformation info.
        epsg = the epsg code of spatial reference system.
        is_projected = is projection system, true or false.
        x_dim = number of raster cells along x axis.
        y_dim = number of raster cells along y axis.
        nodata_val = no data value embedded in raster.
    """
    
    # check if string, if not bail
    if not isinstance(rast_path, str):
        raise ValueError('> Raster path must be a string. Please check the file path.')

    # check if raster exists
    if not os.path.exists(rast_path):
        raise OSError('> Unable to read raster, file not found. Please check the file path.')    

    # init dict
    rast_info_dict = {
        'layer': os.path.basename(rast_path),
        'type': 'raster',
        'geo_tranform': None,
        'x_dim': None,
        'y_dim': None,
        'epsg': None,
        'is_projected': 0,
        'nodata_val': None
    }
        
    try:
        # open raster
        rast = gdal.Open(rast_path, 0)

        # add transform, dims
        rast_info_dict['geo_tranform'] = rast.GetGeoTransform()
        rast_info_dict['x_dim'] = rast.RasterXSize
        rast_info_dict['y_dim'] = rast.RasterYSize
        rast_info_dict['nodata_val'] = rast.GetRasterBand(1).GetNoDataValue()

        # get spatial ref
        srs = rast.GetSpatialRef()

        # get epsg if exists
        if srs and srs.GetAttrValue('AUTHORITY', 1):
            rast_info_dict['epsg'] = srs.GetAttrValue('AUTHORITY', 1)

        # get is projected if exists
        if srs and srs.IsProjected():
            rast_info_dict['is_projected'] = srs.IsProjected()

        # get nodata value if exists
        if srs and srs.IsProjected():
            rast_info_dict['is_projected'] = srs.IsProjected()    

        # drop
        rast = None

    except Exception:
        raise IOError('Unable to read raster: {0}. Please check.'.format(rast_path))

    return rast_info_dict
     
# 
def extract_xr_values(ds, coords, keep_xy=False, res_factor=3, nodata_value=-9999):
    """
    Read a xarray dataset and convert them into a numpy records array. Based 
    on RSGISLib code.

    Parameters
    ----------
    ds: xarray dataset
        A dataset with data variables.
    coords : pandas dataframe
        A pandas dataframe containing x and y columns with records.
    keep_xy : bool
        Keep the x and y coordinate value columns in output.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.
    nodata_value : int or float
        A int or float indicating the no dat avalue expected. Default is -9999.
        Deprecated.

    Returns
    ----------
    df_samples : pandas dataframe
    """

    # notify user
    print('Extracting xarray dataset values to x and y coordinates.')

    # check if coords is a pandas dataframe
    if not isinstance(coords, pd.DataFrame):
        raise TypeError('> Provided coords is not a numpy ndarray type. Please check input.')

    # check if dataset type provided
    if not isinstance(ds, xr.Dataset):
        raise TypeError('> Provided dataset is not an xarray dataset type. Please check input.')
        
    # check if x and y dims exist
    if 'x' not in list(ds.dims) and 'y' not in list(ds.dims):
        raise ValueError('> No x and/or y coordinate dimension in dataset.') 

    # check dimensionality of pandas dataframe. x and y only
    if len(coords.columns) != 2:
        raise ValueError('Num of columns in coords not equal to 2. Please ensure shapefile is valid.')

    # check if res factor is int type
    if not isinstance(res_factor, int):
        raise TypeError('> Resolution factor must be an integer.')

    # check dimensionality of numpy array. xy only
    if not res_factor >= 1:
        raise ValueError('Resolution factor must be value of 1 or greater.')
        
    # check if xr has nodatavals
    if not hasattr(ds, 'nodatavals'):
        raise ValueError('Dataset does not have nodata value attribute.')
    elif ds.nodatavals == 'unknown':
        raise ValueError('Dataset nodata value is unknown.')    

    # get cell resolution from dataset
    res = get_xr_resolution(ds)

    # check res exists
    if not res:
        raise ValueError('> No resolution extracted from dataset.')

    # multiply res by res factor
    res = res * res_factor

    # loop through data var and extract values at coords
    values = []
    for i, row in coords.iterrows():
        try:            
            # get values from vars at current pixel
            pixel = ds.sel(x=row.get('x'), 
                           y=row.get('y'), 
                           method='nearest', 
                           tolerance=res * res_factor)
            pixel = pixel.to_array().values
            pixel = list(pixel)

        except:
            # fill with list of nan equal to data var size
            pixel = [ds.nodatavals] * len(ds.data_vars)
            
        # add current point x and y columns if wanted
        if keep_xy:
            pixel = [row.get('x'), row.get('y')] + pixel

        # append to list
        values.append(pixel)

    # get x, y dtypes
    x_dtype, y_dtype = coords['x'].dtype, coords['y'].dtype
    
    # get original var dtypes as dict
    col_dtypes_dict = {}
    for var in ds.data_vars:
        col_dtypes_dict[var] = np.dtype(ds[var].dtype)
            
    try:
        # prepare whether to retain or discared coords x and y
        if keep_xy:
            col_names = ['x', 'y'] + list(ds.data_vars)
            col_dtypes_dict.update({'x': x_dtype, 'y': y_dtype})
        else:
            col_names = list(ds.data_vars)

        # convert values list into pandas dataframe, ensure types match dataset
        df_samples = pd.DataFrame(values, columns=col_names)
        df_samples = df_samples.astype(col_dtypes_dict)

    except:
        raise ValueError('Errors were encoutered when converting data to pandas dataframe.')

    # notify user and return
    print('Extracted xarray dataset values successfully.')
    return df_samples 


