# cog odc
'''
This script contains similar functions to cog but uses the odc-stac module
instead of a modified version of stackstac. This is considered a superior
module to stackstac.
https://odc-stac.readthedocs.io/en/latest/

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import pystac_client
from odc import stac
from datetime import datetime
from functools import lru_cache


def fetch_stac_items_odc(stac_endpoint=None, collections=None, start_dt=None, end_dt=None, bbox=None, slc_off=False, limit=250):
    """
    Takes a stac endoint url (e.g., 'https://explorer.sandbox.dea.ga.gov.au/stac'),
    a list of stac assets (e.g., ga_ls5t_ard_3 for Landsat 5 collection 3), a range of
    dates, etc. and a bounding box in lat/lon and queries for all available metadata on the 
    digital earth australia (dea) aws bucket for these parameters. If any data is found for
    provided search query, a pystac items collection is returned.
    
    Parameters
    ----------
    stac_endpoint: str
        A string url for a stac endpoint. We are focused on 
        https://explorer.sandbox.dea.ga.gov.au/stac/search for this
        tool.
    collections : list of strings 
        A list of strings of dea aws product names, e.g., ga_ls5t_ard_3.
    start_dt : date
         A datetime object in format YYYY-MM-DD. This is the starting date
         of required satellite imagery.
    end_dt : date
         A datetime object in format YYYY-MM-DD. This is the ending date
         of required satellite imagery.
    slc_off : bool
        Whether to include Landsat 7 errorneous SLC data. Only relevant
        for Landsat data. Default is False - images where SLC turned off
        are not included.
    bbox : list of ints/floats
        The bounding box of area of interest for which to query for 
        satellite data. Is in latitude and longitudes with format: 
        (min lon, min lat, max lon, max lat).
    limit : int
        Limit number of dea aws items per page in query. Recommended to use
        250. Max is 999. 
     
    Returns
    ----------
    items : a pystac itemcollection object
    """
    
    # notify
    print('Beginning STAC search for items. This can take awhile.')
       
    # check stac endpoint provided
    if stac_endpoint is None:
        raise ValueError('Must provide a STAC endpoint.')
    
    # get catalog
    try:
        catalog = pystac_client.Client.open(stac_endpoint)
    except:
        raise ValueError('Could not reach STAC endpoint. Is it down?')
    
    # prepare collection list
    if collections is None:
        collections = []
    if not isinstance(collections, (list)):
        collections = [collections]
        
    # get dt objects of date strings
    if start_dt is None or end_dt is None:
        raise ValueError('No start or end date provided.')
    elif not isinstance(start_dt, str) or not isinstance(end_dt, str):
        raise ValueError('Start and end date must be strings.')
    else:
        start_dt_obj = datetime.strptime(start_dt, "%Y-%m-%d")
        end_dt_obj = datetime.strptime(end_dt, "%Y-%m-%d")
        
    # check bounding bix
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise TypeError('Bounding box must contain four numbers.')
    
    # loop collections
    items = None
    for collection in collections:
        
        # notify
        print('Searching collection: {}'.format(collection))
        
        # chekc collection type
        if not isinstance(collection, str):
            raise TypeError('Collection must be a string.')
            
        # correct landsat 7 if requested!
        if collection == 'ga_ls7e_ard_3' and not slc_off:
            print('Excluding SLC-off times.')

            # fix date if after slc data ended
            slc_dt = '2003-05-31'
            start_slc_dt = slc_dt if not start_dt_obj.year < 2003 else start_dt
            end_slc_dt = slc_dt if not end_dt_obj.year < 2003 else end_dt
            dt = '{}/{}'.format(start_slc_dt, end_slc_dt)

        else:
            dt = '{}/{}'.format(start_dt, end_dt)
            
        # do last check on limit
        if limit <= 0 or limit > 999:
            limit = 999
                        
        # query current collection
        query = catalog.search(collections=collection,
                               datetime=dt,
                               bbox=bbox,
                               limit=limit)
        
        # if items return, add to queries
        if items is None:
            items = query.get_all_items()
        else:
            items = items + query.get_all_items()
            
    # let user know of item count and return
    print('A total of {} scenes were found.'.format(len(items)))
    return items


def replace_items_s3_to_https(items, from_prefix='s3://dea-public-data', to_prefix='https://data.dea.ga.gov.au'):
    """
    Iterate all items in pystac ItemCollection and replace
    the DEA AWS s3prefix with https (from/to). ArcGIS Pro
    doesn't seem to play nice with s3 at the moment.
    
    Parameters
    ----------
    items: pystac ItemCollection object
        A ItemCollection from pystac typically obtained from the
        fetch_stac_items_odc function.
    from_prefix : string
        The original href url prefix (e.g. s3://dea-public-data) to be 
        replaced for each 
        item (band).
    to_prefix : string
        The new href url prefix (e.g. https://data.dea.ga.gov.au) to 
        replace original.
     
    Returns
    ----------
    items : a pystac itemcollection object with replaced
    href prefix.
    """
    
    # notify user
    print('Replacing url prefix: {} with {}'.format(from_prefix, to_prefix))
    
    # check if from/to are valid
    if not isinstance(from_prefix, str) or not isinstance(to_prefix, str):
        raise TypeError('Both from and to prefixes have to be strings.')
    
    # iter items
    for item in items:

        # iter asset and using name replace href if s3
        for asset_name in item.assets:
            asset = item.assets.get(asset_name)
            href = asset.href

            if href.startswith(from_prefix):
                asset.href = href.replace(from_prefix, to_prefix)

    return items


def build_xr_odc(items=None, bbox=None, bands=None, crs=3577, resolution=None, chunks={}, group_by='solar_day', sort_by=True, skip_broken_datasets=True, like=None):
    """
    Takes a pystac ItemCollection of prepared DEA stac items from the 
    fetch_stac_items_odc function and converts it into lazy-load (or not,
    if chunks is None) xarray dataset with time, x, y, variable dimensions. 
    This is provided by te great work of odc-stac, found here:
    https://odc-stac.readthedocs.io/en/latest/.
    
    Parameters
    ----------
    items: pystac ItemCollection object
        A ItemCollection from pystac typically obtained from the
        fetch_stac_items_odc function.
    bbox: list of int/floats
        The bounding box of area of interest for which to query for 
        satellite data. Must be lat and lon with format: 
        (min lon, min lat, max lon, max lat). Recommended that users
        use wgs84.
    bands : str or list
        List of band names for requested collections.
    crs : int
        Reprojects output satellite images into this coordinate system. 
        Must be a int, later converted to 'EPSG:####' string.
    resolution: tuple
        Output size of raster cells. If higher or greater than raw pixel
        size on dea aws, resampling will occur to up/down scale to user
        provided resolution. Careful - units must be in units of provided
        epsg. Tenement tools sets this to 30, 10 for landsat, sentinel 
        respectively.
    chunks: dict
        Must be a dict of dimension names and chunk size of each. For exmaple
        {'x': 512, 'y': 512}. Set to None for optimal lazy load.
    group_by: str
        Whether to group scenes based on solar day (i.e. overlaps in a single
        pass) or time.
    skip_broken_datasets: bool
        If an error occurs, whether to skip the errorneous data and keep
        going or error.
    sort_by_time: bool
        Sort resulting xr datset by time.
    like: xr dataset
        If another dataset provided, will use the x, y and attributes to
        generate a new xr dataset for new dates for within this same extent.
        Good for sync function.
        
    Returns
    ----------
    ds : xr dataset
    """
    
    # notify
    print('Converting raw STAC data into xarray dataset via odc-stac.')
    
    # check items
    if len(items) < 1:
        raise ValueError('No items in provided ItemCollection.')
        
    # prepare band list
    if bands is None:
        bands = []
    if not isinstance(bands, (list)):
        bands = [bands]
    if len(bands) == 1 and bands[0] is None:
        raise ValueError('Must request at least one asset/band.')      
    
    # check bounding bix
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise TypeError('Bounding box must contain four numbers.')
        
    # correct crs
    if not isinstance(crs, int):
        raise TypeError('the crs provided must be an integer e.g., 4326.')
    else:
        crs = 'EPSG:{}'.format(crs)
        
    # resolution
    if not isinstance(resolution, (int, float)):
        raise TypeError('Resoluton must be a single integer or float.')
    else:
        resolution = (resolution * -1, resolution)
    
    # check chunks
    if chunks is None:
        chunks = {}
    if not isinstance(chunks, dict):
        raise TypeError('Chunks must be a dictionary or None.')
        
    # check group_by is either time or solar_day
    if group_by not in ['solar_day', 'time']:
        raise ValueError('Group_by must be either solar_day or time.')
        
    try:
        # use the odc-stac module to build lazy xr dataset
        ds = stac.load(items=items,
                       bbox=bbox,
                       bands=bands,
                       crs=crs,
                       resolution=resolution,
                       chunks=chunks,
                       group_by=group_by,
                       skip_broken_datasets=skip_broken_datasets,
                       like=like
                      )
        
    except:
        raise ValueError('Could not create xr dataset from stac result.')
        
    # sort by time if requested
    if sort_by:
        ds = ds.sortby('time')
    
    # notify and return
    print('Created xarray dataset via odc-stac successfully.')
    return ds


# may need to add more
def append_query_attrs_odc(ds, bbox, collections, bands, resolution, dtype, fill_value, slc_off, resampling):
    """
    Takes a newly constructed xr dataset and appends some of the original query
    parameters (e.g., collections, bands, bbox, slc-off) to assist in updates later
    on.
    
    Parameters
    -------------
    ds : xr dataset
        An xr dataset object that holds the lazy-loaded raster images obtained
        from the majority of this code base. Attributes are appended to this
        object.
    bbox : list of ints/floats
        The bounding box of area of interest for which to query for 
        satellite data. Is in latitude and longitudes with format: 
        (min lon, min lat, max lon, max lat). Only used to append to 
        attributes, here.
    collections : list
        A list of names for the requested satellite dea collections. For example,
        ga_ls5t_ard_3 for lansat 5 analysis ready data. Not used in analysis here,
        just dded to attributes.
    bands : list
        List of band names requested original query.
    resolution : int or float
        A integer or float containing resolution.
    dtype : str
        Data type of output dataset, e.g., int16, float32. In numpy
        dtype that the output xarray dataset will be encoded in.
    fill_value : int or float
        The value used to fill null or errorneous pixels with from prior methods.
        Not used in analysis here, just included in the attributes. Used to set
        nodatavalue attribute.
    slc_off : bool
        Whether to include Landsat 7 errorneous SLC data. Only relevant
        for Landsat data. Not used in analysis here, only appended to attributes.
    resampling : str
        The rasterio-based resampling method used when pixels are reprojected
        rescaled to different crs from the original. Just used here to append
        to attributes.
        
    Returns
    ----------
    A xr dataset with new attributes appended to it.
    """
    
    # notify
    print('Preparing and appending attributes to dataset.')

    # set resolution
    ds = ds.assign_attrs({'res': resolution})

    # set transform
    ds = ds.assign_attrs({'transform': tuple(ds.geobox.transform)})

    # set nodatavals
    ds = ds.assign_attrs({'nodatavals': fill_value})

    # create top level query parameters
    ds = ds.assign_attrs({'orig_bbox': tuple(bbox)})               # set original bbox
    ds = ds.assign_attrs({'orig_collections': tuple(collections)}) # set original collections
    ds = ds.assign_attrs({'orig_bands': tuple(bands)})             # set original bands
    ds = ds.assign_attrs({'orig_dtype': dtype})                    # set original dtype
    ds = ds.assign_attrs({'orig_slc_off': str(slc_off)})           # set original slc off
    ds = ds.assign_attrs({'orig_resample': resampling})            # set original resample method 

    # notify and return
    print('Attributes appended to dataset successfully.')
    return ds


# CHECK DATA TYPE
def remove_fmask_dates(ds, valid_class=[1, 4, 5], max_invalid=5, mask_band='oa_fmask', nodata_value=np.nan, drop_fmask=False):
    """
    Takes an xr dataset and computes fmask band, if it exists. From mask band,
    calculates percentage of valid vs invalid pixels per image date. Returns a xr
    dataset where all images where too  many invalid pixels were detected have been
    removed.
    
    Parameters
    -------------
    ds : xr dataset
        A xarray dataset with time, x, y, band dimensions.
    valid_classes : list
        List of valid fmask classes. For dea landsat/sentinel data,
        1 = valid, 2 = cloud, 3 = shadow, 4 = snow, 5 = water. See:
        https://docs.dea.ga.gov.au/notebooks/Frequently_used_code/Masking_data.html.
        Default is 1, 4 and 5 (valid, snow, water pixels returned).
    max_invalid : int or float
        The maximum amount of invalid pixels per image to flag whether
        it is invalid. In other words, a max_invalid = 5: means >= 5% invalid
        pixels in an image will remove that image.
    mask_band : str
        Name of mask band in dataset. Default is oa_fmask.
    nodata_value : numpy dtype
        If an invalid pixel is detected, replace with this value. Numpy nan 
        (np.nan) is recommended.
    drop_fmask : bool
        Once fmask images have been removed, drop the fmask band too? Default
        is True.
        
    Returns
    ----------
    ds : xr dataset with invalid pixels masked out and images >= max_invalid
        removed also.
    """
    
    # notify
    print('Removing dates where too many invalid pixels.')

    # check if time, x, y in dataset
    for dim in [dim for dim in list(ds.dims)]:
        if dim not in ['time', 'x', 'y']:
            raise ValueError('Unsupported dim: {} in dataset.'.format(dim))
            
    # check if fmask in dataset
    if mask_band is None:
        raise ValueError('Name of mask band must be provided.')
    else:
        if mask_band not in list(ds.data_vars):
            raise ValueError('Requested mask band name not found in dataset.')

    # calc min number of valid pixels allowed
    min_valid = 1 - (max_invalid / 100)
    num_pix = ds['x'].size * ds['y'].size

    # subset mask band and if dask, compute it
    mask = ds[mask_band]
    if bool(mask.chunks):
        print('Mask band is currently dask. Computing, please wait.')
        mask = mask.compute()

    # set all valid classes to 1, else 0
    mask = xr.where(mask.isin(valid_class), 1.0, 0.0)
    
    # convert to float32 if nodata value is nan
    if nodata_value is np.nan:
        ds = ds.astype('float32')    

    # mask invalid pixels with user value
    print('Filling invalid pixels with requested nodata value.')
    ds = ds.where(mask == 1.0, nodata_value)
    
    # calc proportion of valid pixels to get array of invalid dates
    mask = mask.sum(['x', 'y']) / num_pix
    valid_dates = mask['time'].where(mask >= min_valid, drop=True)

    # remove any non-valid times from dataset
    ds = ds.where(ds['time'].isin(valid_dates), drop=True)

    # drop mask band if requested
    if drop_fmask:
        print('Dropping mask band.')
        ds = ds.drop_vars(mask_band)
        
    # notify and return
    print('Removed invalid images successfully.')
    return ds





# helpers
def convert_type(ds, to_type='int16'):
    """
    Small helper to update an existing odc-stac derived
    xr dataset to type int16, which is needed for tenement
    tools.
    """
    
    return ds.astype(to_type)


def change_nodata_odc(ds, orig_value=0, fill_value=-999):
    """
    Takes original xr dataset  odc-stac nodata values (0) 
    for non-mask bands and converts to int16 with -999 as 
    NoData to match cog fetch.
    
    Parameters
    ----------
    ds: xr dataset
        A single xr dataset object.
    orig_value: int or float
        A value that exists in the xr dataset that represents NoData 
        values.
    fill_value: int or float
        A value to which previous nodata values are converted to.
     
    Returns
    ----------
    ds : xr dataset with modified nodata values.
    """
    
    # loop all vars without mask in name
    for var in list(ds.data_vars):
        if 'mask' not in var.lower():
            ds[var] = ds[var].where(ds[var] != orig_value, fill_value)

    return ds


def fix_xr_time_for_arc_cog(ds):
    """
    Small helper function to strip milliseconds from
    xr dataset - arcgis pro does not play nicely
    with milliseconds.
    """
    
    # convert datetimes
    dts = ds.time.dt.strftime('%Y-%m-%dT%H:%M:%S')
    dts = dts.astype('datetime64[ns]')
    
    # strip milliseconds
    ds['time'] = dts
    
    return ds