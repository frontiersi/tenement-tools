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
#import itertools
#import warnings
#import requests
#import threading
#import pyproj
#import affine
#import osr
#import numpy as np
#import pandas as pd
import xarray as xr
#import dask
#import dask.array as da
#import dask.array as dask_array
import rasterio
#from rasterio.warp import transform_bounds
#from rasterio import windows
#from rasterio.enums import Resampling
#from rasterio.vrt import WarpedVRT
from datetime import datetime
#from functools import lru_cache


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


def prepare_data(feats=None, assets=None, bounds_latlon=None, bounds=None, epsg=3577, resolution=30, snap_bounds=True, force_dea_http=True):
    """
    Prepares raw stac metadata and assets into actual useable data for dask
    computations. This includes preparing coordinate system, bounding boxes,
    resolution and resampling, raster sizes and overlaps, raster snapping,
    etc. Results in an numpy asset table of bounding box and url information, 
    plus and metadata object for later appending to xarray dataset.
    
    Parameters
    ----------
    feats: list
        A list of dicts of stac metadata items produced from fetch stac data
        function.
    assets: list
        A list of satellite band names. Must be correct names for landsat and
        sentinel from dea aws.
    bound_latlon: list of int/float
        The bounding box of area of interest for which to query for 
        satellite data. Must be lat and lon with format: 
        (min lon, min lat, max lon, max lat). Recommended that users
        use wgs84.
    bounds : list of int/float
        As above, but using same coordinate system specified in epsg 
        parameter. Cannot provide both. Currently disabled.
    epsg : int
        Reprojects output satellite images into this coordinate system. 
        If none provided, uses default system of dea aws. Default is
        gda94 albers, 3577. Tenement tools will always use this.
    resolution : int
        Output size of raster cells. If higher or greater than raw pixel
        size on dea aws, resampling will occur to up/down scale to user
        provided resolution. Careful - units must be in units of provided
        epsg. Tenement tools sets this to 30, 10 for landsat, sentinel 
        respectively.
    snap_bounds : bool
        Whether to snap raster bounds to whole number intervals of resolution,
        to prevent fraction-of-a-pixel offsets. Default is true.
    force_dea_http : bool
        Replace s3 aws url path for an http path. Default is true.
        
    Returns
    ----------
    meta : a dictionary of dea aws metadata for appending to xr dataset later
    asset_table : a numpy table with rows of url and associated cleaned bounding box
    """
    
    # notify
    print('Converting raw STAC data into numpy format.')
    
    # set input epsg used by bounds_latlon. for now, we hardcode to wgs84
    # may change and make dynamic later, for now suitable
    in_bounds_epsg = 4326

    # check if both bound parameters provided
    if bounds_latlon is not None and bounds is not None:
        raise ValueError('Cannot provide both bounds latlon and bounds.')
    
    # check if bounds provided - currently disabled, might remove
    if bounds is not None:
        raise ValueError('Bounds input currently disabled. Please use bounds_latlon.')
        
    # check output epsg is albers - currently only support epsg
    if epsg != 3577:
        raise ValueError('EPSG 3577 only supported for now.')
        
    # prepare resolution tuple
    if resolution is None:
        raise ValueError('Must set a resolution value.')
    elif not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
            
    # set output epsg, output bounds, output resolution
    out_epsg, out_bounds, out_resolution = epsg, bounds, resolution
    
    # prepare and check assets list
    if assets is None:
        assets = []
    if not isinstance(assets, list):
        assets = [assets]
    if len(assets) == 0:
        raise ValueError('Must request at least one asset.')
                
    # check data types of assets?
    # we always get rasters of int16 from dea aws... leave for now
        
    # create an numpy asset table to store info, make use of object type for string
    asset_dt = np.dtype([('url', object), ('bounds', 'float64', 4)]) 
    asset_table = np.full((len(feats), len(assets)), None, dtype=asset_dt) 

    # check if feats exist
    if len(feats) == 0:
        raise ValueError('No items to prepare.')
        
    # iter feats, work on bbox, projection, get url
    for feat_idx, feat in enumerate(feats):
        
        # get feat level meta
        feat_props = feat.get('properties')
        feat_epsg = feat_props.get('proj:epsg')
        feat_bbox = feat_props.get('proj:bbox')
        feat_shape = feat_props.get('proj:shape')
        feat_transform = feat_props.get('proj:transform')        
    
        # unpack assets
        feat_bbox_proj = None
        for asset_idx, asset_name in enumerate(assets):
            asset = feat.get('assets').get(asset_name)
            
            # get asset level meta, if not exist, use feat level
            if asset is not None:
                asset_epsg = asset.get('proj:epsg', feat_epsg)
                asset_bbox = asset.get('proj:bbox', feat_bbox)
                asset_shape = asset.get('proj:shape', feat_shape)
                asset_transform = asset.get('proj:transform', feat_transform)
                asset_affine = None
                
                # note: in future, may want to handle using asset epsg 
                # here instead of forcing albers. for now, all good.
                
                # cast output epsg if in case string
                out_epsg = int(out_epsg)
                
                # reproject bounds and out_bounds to user epsg
                if bounds_latlon is not None and out_bounds is None:
                    bounds = reproject_bbox(in_bounds_epsg, out_epsg, bounds_latlon)
                    out_bounds = bounds

                # below could be expanded in future. we always have asset transform, shape
                # but what if we didnt? would need to adapt this to handle. lets leave for now
                # compute bbox from asset level shape and geotransform
                if asset_transform is not None or asset_shape is not None or asset_epsg is not None:
                    asset_affine = affine.Affine(*asset_transform[:6])
                    asset_bbox_proj = bbox_from_affine(asset_affine,
                                                       asset_shape[0],
                                                       asset_shape[1],
                                                       asset_epsg,
                                                       out_epsg)
                else:
                    raise ValueError('No feature-level transform and shape metadata.')
                
                # compute bounds depending on situation
                if bounds is None:
                    if asset_bbox_proj is None:
                        raise ValueError('Not enough STAC infomration to build bounds.')
                        
                    # when output bounds does not exist, use projcted asset bbox, else dounion
                    if out_bounds is None:
                        out_bounds = asset_bbox_proj
                    else:
                        out_bounds = union_bounds(asset_bbox_proj, out_bounds)
                        
                else:
                    # skip asset if bbox does not overlap with requested bounds
                    overlaps = bounds_overlap(asset_bbox_proj, bounds)
                    if asset_bbox_proj is not None and not overlaps:
                        continue
                    
                # note: may want to auto-detect resolution from bounds and affine
                # here. for now, tenement tools requires resolution is set
                
                # extract url from asset
                href = asset.get('href')

                # convert aws s3 url to https if requested. note: not tested with s3 url               
                if force_dea_http:
                    href = href.replace('s3://dea-public-data', 'https://data.dea.ga.gov.au')
                    
                # add info to asset table, 1 row per scene, n columns per band, i.e. (url, [l, b, r, t]) per row
                asset_table[feat_idx, asset_idx] = (href, asset_bbox_proj)

    # snap boundary coordinates if requested
    if snap_bounds:
        out_bounds = snap_bbox(out_bounds, out_resolution)
     
    # transform and get shape for feat level
    transform = do_transform(out_bounds, out_resolution)
    shape = get_shape(out_bounds, out_resolution)
        
    # get table of nans for any feats/assets where asset missing/out bounds
    isnan_table = np.isnan(asset_table['bounds']).all(axis=-1)
    feat_isnan = isnan_table.all(axis=1)
    asset_isnan = isnan_table.all(axis=0)
    
    # remove nan feats/assets. note: ix_ looks up cells at specific col, row
    if feat_isnan.any() or asset_isnan.any():
        asset_table = asset_table[np.ix_(~feat_isnan, ~asset_isnan)]
        feats = [feat for feat, isnan in zip(feats, feat_isnan) if not isnan]
        assets = [asset for asset, isnan in zip(assets, asset_isnan) if not isnan]
        
    # create feat/asset metadata attributes for xr dataset later
    meta = {
        'epsg': out_epsg,
        'bounds': out_bounds,
        'resolutions_xy': out_resolution,
        'shape': shape,
        'transform': transform,
        'feats': feats,
        'assets': assets,
        'vrt_params': {
            'crs': out_epsg,
            'transform': transform,
            'height': shape[0],
            'width': shape[1]
        }
    }
    
    # notify and return
    print('Converted raw STAC data successfully.')
    return meta, asset_table


def convert_to_dask(meta=None, asset_table=None, chunksize=512, resampling='nearest', dtype='int16', fill_value=-999, rescale=True):
    """
    Takes a array of prepared stac items from the prepare_data function 
    and converts it into lazy-load friendly dask arrays prior to 
    converson into a final xr dataset. Some of the smart dask work is
    based on the rasterio/stackstac implementation. For more information
    on stackstac, see: https://github.com/gjoseph92/stackstac. 
    
    Parameters
    ----------
    meta: dict
        A dictionary of metadata attibution extracted from stac results
        in prepare_data function.
    asset_table: numpy array
        A numpy array of structure (url, [bbox]) per row. Url is a 
        url link to a specific band of a specific feature from dea public
        data (i.e., Landsta or Sentinel raster band), and bbox is the
        projected bbox for that scene or scene window.
    chunksize: int
        Rasterio lazy-loading chunk size for cog asset. The dea uses default
        chunksize of 512, but could speed up processing by modifying this.
    resampling : str
        The rasterio-based resampling method used when pixels are reprojected
        rescaled to different crs from the original. Default rasterio nearest
        neighbour. Could use bilinear. We will basically always use this in 
        tenement tools.
    dtype : str or numpy type
        The numpy data type for output rasters for ech dask array row.
        We will typically use int16, to speed up downloads and reduce 
        storage size. Thus, no data is best as -999 instead of np.nan,
        which will cast to float32 and drastically increase data size.
    fill_value : int or float
        The value to use when pixel is detected or converted to no data
        i.e. on errors or missing raster areas. Recommended that -999 is
        used on dea landsat and sentinel data to reduce download and storage
        concerns.
    rescale : bool
        Whether rescaling of pixel vales by the scale and offset set on the
        dataset. Defaults to True.

    Returns
    ----------
    rasters : a dask array of satellite rasters nearly ready for compute!
    """

    # notify
    print('Converting data into dask array.')
    
    # check if meta and array provided
    if meta is None or asset_table is None:
        raise ValueError('Must provide metadata and asset table array.')
        
    # checks
    if resampling not in ['nearest', 'bilinear']:
        raise ValueError('Resampling method not supported.')
        
    # set resampler
    if resampling == 'nearest':
        resampler = Resampling.nearest
    else:
        resampler = Resampling.bilinear
    
    # check dtype, if string cast it, catch error if invalid
    if isinstance(dtype, str):
        try:
            dtype = np.dtype(dtype)
        except:
            raise TypeError('Requested dtype is not valid.')        
            
    # check if dtypes are allowed
    if dtype not in [np.int8, np.int16, np.int32, np.float16, np.float32, np.float64, np.nan]:
        raise TypeError('Requested dtype is not supported. Use int or float.')
        
    # check if we can use fill value with select dtype
    if fill_value is not None and not np.can_cast(fill_value, dtype):
        raise ValueError('Fill value incompatible with output dtype.')
        
    # note: the following approaches are based on rasterio and stackstac
    # methods. the explanation for this is outlined deeply in the stackstac
    # to_dask function. check that code for a deeper explanation.
    # extra note: this might be overkill for our 'dumb' arcgis method
    
    # see their documentation for a deeper explanation.
    da_asset_table = dask_array.from_array(asset_table, 
                                           chunks=1, 
                                           #inline_array=True,  # our version of xr does not support inline
                                           name='assets_' + dask.base.tokenize(asset_table))
    
    # map to blocks. the cog reader class is mapped to each chunk for reading
    ds = da_asset_table.map_blocks(apply_cog_reader,
                                   meta,
                                   resampler,
                                   dtype,
                                   fill_value,
                                   rescale,
                                   meta=da_asset_table._meta)
    
    # generate fake array from shape and chunksize, see stackstac to_dask for approach
    shape = meta.get('shape')
    name = 'slices_' + dask.base.tokenize(chunksize, shape)
    chunks = dask_array.core.normalize_chunks(chunksize, shape)
    keys = itertools.product([name], *(range(len(b)) for b in chunks))
    slices = dask_array.core.slices_from_chunks(chunks)
    
    # stick slices into array container to force dask blockwise logic to handle broadcasts
    fake_slices = dask_array.Array(dict(zip(keys, slices)), 
                                   name, 
                                   chunks, 
                                   meta=ds._meta)
    
    # apply blockwise logic
    rasters = dask_array.blockwise(fetch_raster_window,
                                   'tbyx',
                                   ds,
                                   'tb',
                                   fake_slices,
                                   'yx',
                                   meta=np.ndarray((), dtype=dtype))
        
    # notify and return
    print('Converted data successfully.')
    return rasters



# SIMPLIFY
def build_attributes(ds, meta, collections, bands, slc_off, bbox, dtype, 
                     snap_bounds, fill_value, rescale, cell_align, resampling):
    """
    Takes a newly constructed xr dataset and associated stac metadata attributes
    and appends attributes to the xr dataset itself. This is useful information
    for context in arcmap, but also needed in the dataset update methodology
    for nrt methodology. A heavily modified version of the work done by the 
    great stackstac folk: http://github.com/gjoseph92/stackstac. 
    
    Parameters
    -------------
    ds : xr dataset
        An xr dataset object that holds the lazy-loaded raster images obtained
        from the majority of this code base. Attributes are appended to this
        object.
    meta : dict
        A dictionary of the dea stac metadata associated with the current
        query and satellite data.
    collections : list
        A list of names for the requested satellite dea collections. For example,
        ga_ls5t_ard_3 for lansat 5 analysis ready data. Not used in analysis here,
        just dded to attributes.
    bands : list
        List of band names requested original query.
    slc_off : bool
        Whether to include Landsat 7 errorneous SLC data. Only relevant
        for Landsat data. Not used in analysis here, only appended to attributes.
    bbox : list of ints/floats
        The bounding box of area of interest for which to query for 
        satellite data. Is in latitude and longitudes with format: 
        (min lon, min lat, max lon, max lat). Only used to append to 
        attributes, here.
    dtype : str
        Name of original query dtype, e.g., int16, float32. In numpy
        dtype that the output xarray dataset will be encoded in.
    snap_bounds : bool
        Whether to snap raster bounds to whole number intervals of resolution,
        to prevent fraction-of-a-pixel offsets. Default is true. Only used
        to append to netcdf attirbutes.
    fill_value : int or float
        The value used to fill null or errorneous pixels with from prior methods.
        Not used in analysis here, just included in the attributes.
    rescale : bool
        Whether rescaling of pixel vales by the scale and offset set on the
        dataset. Defaults to True. Only used to append to netcdf attributes.
    cell_align: str
        Alignmented of cell in original query. Either Too-left or Center.
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
    
    # assign spatial_ref coordinate to align with dea odc output
    crs = int(meta.get('epsg'))
    ds = ds.assign_coords({'spatial_ref': crs})
        
    # get wkt from epsg 
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    wkt = srs.ExportToWkt()
    
    # assign wkt to spatial ref attribute. note: this is designed for gda 94 albers
    # if we ever want to include any other output crs, we will need to adapt this
    grid_mapping_name = 'albers_conical_equal_area'
    ds['spatial_ref'] = ds['spatial_ref'].assign_attrs({'spatial_ref': wkt, 
                                                        'grid_mapping_name': grid_mapping_name})
    
    # assign global crs and grid mapping attributes
    ds = ds.assign_attrs({'crs': 'EPSG:{}'.format(crs)})
    ds = ds.assign_attrs({'grid_mapping': 'spatial_ref'})
    
    # get res from transform affine object 
    transform = meta.get('transform')
    res_x, res_y = transform[0], transform[4]
    
    # assign x coordinate attributes 
    ds['x'] = ds['x'].assign_attrs({
        'units': 'metre',
        'resolution': res_x,
        'crs': 'EPSG:{}'.format(crs)
    })
    
    # assign y coordinate attributes
    ds['y'] = ds['y'].assign_attrs({
        'units': 'metre',
        'resolution': res_y,
        'crs': 'EPSG:{}'.format(crs)
    })    
    
    # set range of original query parameters for future sync
    ds = ds.assign_attrs({'transform': tuple(transform)})          # set transform info for cog fetcher
    ds = ds.assign_attrs({'nodatavals': fill_value})               # set original no data values
    ds = ds.assign_attrs({'orig_collections': tuple(collections)}) # set original collections
    ds = ds.assign_attrs({'orig_bands': tuple(bands)})             # set original bands
    ds = ds.assign_attrs({'orig_slc_off': str(slc_off)})           # set original slc off
    ds = ds.assign_attrs({'orig_bbox': tuple(bbox)})               # set original bbox
    ds = ds.assign_attrs({'orig_dtype': dtype})                    # set original dtype
    ds = ds.assign_attrs({'orig_snap_bounds': str(snap_bounds)})   # set original snap bounds (as string)
    ds = ds.assign_attrs({'orig_cell_align': cell_align})          # set original cell align
    ds = ds.assign_attrs({'orig_resample': resampling})            # set original resample method 
    
    # set output resolution depending on type
    res = meta.get('resolutions_xy')
    res = res[0] if res[0] == res[1] else res
    ds = ds.assign_attrs({'res': res})
        
    # iter each var and update attributes 
    for data_var in ds.data_vars:
        ds[data_var] = ds[data_var].assign_attrs({
            'units': '1', 
            'crs': 'EPSG:{}'.format(crs), 
            'grid_mapping': 'spatial_ref'
        })
        
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



