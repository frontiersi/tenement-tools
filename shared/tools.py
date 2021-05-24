"""
"""

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import ogr

# meta, checks
def calculate_indices(ds, index=None, custom_name=None, rescale=False, drop=False):
    """
    """
        
    # notify
    print('Calculating indices.'.format(index))
    
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
        
    # check if index is empty
    #if not index:
        #raise ValueError('Must request at least one index.')
                        
    # prepare index, custom names if empty
    index = index if index is not None else []
    custom_name = custom_name if custom_name is not None else []
            
    # if not none but not list, prepare also
    indices = index if isinstance(index, list) else [index]
    custom_names = custom_name if isinstance(custom_name, list) else [custom_name]
                    
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
    to a xarray dataset or array. Existing fields in the dataframe can
    be saved in output if field name is placed in keep_fields parameter.
    It is recommended a binary mask is used as ds input, with nan
    represented by nodata_value paramter. if_nodata = if any vars in pixel
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

    # notify user and return
    print('Resolution extracted successfully from dataset.')
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