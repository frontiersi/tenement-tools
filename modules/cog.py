# cog
'''
This script contains functions for searching the digital earth australia (dea)
public aws bucket using stac, preparing cog data into dask arrays for lazy 
loading, and computation into a local netcdf with bands as variables, x, y
and time dimensions. This script is a simplified, python 3.7-compatible and 
dea-focused version of the great stackstac python library 
(http://github.com/gjoseph92/stackstac). 

We highly recommended using stackstac if you need any aws bucket other 
than the dea public database, and to cite them when they have a white paper 
available, if using tenement-tools in research.

Note: upcoming DEA ODC-STAC module provides an official stac-xarray workflow, 
consider swapping in if GDV project continues.

See associated Jupyter Notebook cog.ipynb for a basic tutorial on the
main functions and order of execution. https://odc-stac.readthedocs.io/en/latest/

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import itertools
import warnings
import requests
import threading
import pyproj
import affine
import osr
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
import dask.array as dask_array
import rasterio
from rasterio.warp import transform_bounds
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from datetime import datetime
from functools import lru_cache

# # # rasterio reader classes
class COGReader:
    """
    Class for a url/cog reader that returns an array of actual values
    for a url/cog that was were not errorneous or completely empty on 
    the dea aws side of things. Tasked with opening (lazy), reading,
    closing, warping cog to projected bounds, etc. Smart gdal management
    with threading, locking, designed by great team at stakstac:
    http://github.com/gjoseph92/stackstac. This reader is applied
    dask-based url and bounding infomration.
    """
    
    def __init__(self, url, meta, resampler, dtype=None, fill_value=None, rescale=True):
        self.url = url
        self.meta = meta
        self.resampler = resampler
        self.dtype = dtype
        self.rescale = rescale
        self.fill_value = fill_value
        
        # set default rasterio/gdal env parameters
        self._env = {
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
            }
        
        # set private dataset and lock
        self._dataset = None
        self._dataset_lock = threading.Lock()


    def _open(self):
        """
        Opens a cog/url via rasterio to obtain basic info about raster. 
        No reading is done, just a quick check of metadata. If warping
        is required, warping is done via rasterio warpedvrt without read.
        """

        # init rasterio env for efficient open
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
            try:
                # open url via rasterio
                ds = rasterio.open(self.url, sharing=False)
            except:
                # if error, assign nan reader to url
                return NANReader(dtype=self.dtype, fill_value=self.fill_value)           

            # check if datset is only one band, else error - we dont support
            if ds.count != 1:
                ds.close()
                raise ValueError('More than one band in single cog, data not supported.')
                
            # init vrt variable
            vrt = None
                
            # check if cog meta matches uer requested meta (i.e., projection)
            meta_no_match = self.meta.get('vrt_params') != {'crs': ds.crs.to_epsg(),
                                                            'transform': ds.transform,
                                                            'height': ds.height,
                                                            'width': ds.width}
                                                            
            # if no match, then warp via vrt
            if meta_no_match:
                vrt_meta = self.meta.get('vrt_params')
                
                # swap to rasterio env for warping and warp speed!
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
                    vrt = WarpedVRT(ds,
                                    sharing=False, 
                                    resampling=self.resampler,
                                    crs=vrt_meta.get('crs'),
                                    transform=vrt_meta.get('transform'),
                                    height=vrt_meta.get('height'),
                                    width=vrt_meta.get('width'))

        # apply cog reader to data
        if ds.driver in ['GTiff']:
            return ThreadLocalData(ds, vrt=vrt)
        else:
            ds.close()
            raise TypeError('COGreader currently only supports GeoTIFFs or COGs.')
     
    @property
    def dataset(self):
        """In current locked thread, return dataset if exists, else open one."""
        with self._dataset_lock:
            if self._dataset is None:
                self._dataset = self._open()
            return self._dataset
        
 
    def read(self, window, **kwargs):
        """Read cog window, rescale if requested, mask fill values."""
        
        # open dataset if exist, or load and open if not
        reader = self.dataset
        try:
            # read dataset and mask for safer scaling/offsets
            result = reader.read(window=window, masked=True, **kwargs)
        except:
            # on error, return no data reader instead
            return NANReader(dtype=self.dtype, fill_value=self.fill_value)    
    
        # rescale to scale and offset values 
        if self.rescale:
            scale, offset = reader.scale_offset
            if scale != 1 and offset != 0:
                result *= scale
                result += offset
            
        # convert type, fill mask areas with requested nodata values
        result = result.astype(self.dtype, copy=False)
        return np.ma.filled(result, fill_value=self.fill_value)
    
    
    def close(self):
        """In current locked thread, close dataset if exists, else set empty"""
        with self._dataset_lock:
            if self._dataset is None:
                return None
            self._dataset.close()
            self._dataset = None
    
    
    def __del__(self):
        """Called when garbage collected after close. Helps multi-thread issues."""
        try:
            self.close()
        except:
            pass
    
    
    def __getstate__(self):
        """Get pickled meta."""
        state = {
            'url': self.url,
            'meta': self.meta,
            'resampler': self.resampler,
            'dtype': self.dtype,
            'fill_value': self.fill_value,
            'rescale': self.rescale,            
        }
        return state
    
    
    def __setstate__(self, state):
        """Set pickled meta."""
        self.__init__(
            url=state.get('url'),
            meta=state.get('meta'),
            resampler=state.get('resampler'),
            dtype=state.get('dtype'),
            fill_value=state.get('fill_value'),
            rescale=state.get('rescale')
        )


class NANReader:
    """
    Class for a url/cog reader that returns an array of nodata values
    for a url/cog that was errorneous or completely empty on the dea 
    aws side of things. Based off the great work of the stackstac library:
    http://github.com/gjoseph92/stackstac. Is used to wrap dask array
    urls and bounding boxes up for opening, reading, closing tasks during
    xr dataset computation.
    """
    
    # set scale offset constant
    scale_offset = (1.0, 0.0)
    
    def __init__(self, dtype=None, fill_value=None, **kwargs):
        self.dtype = dtype
        self.fill_value = fill_value


    def read(self, window, **kwargs):
        """Read cog window, fill with empty values."""
        return get_nodata_for_window(window, self.dtype, self.fill_value)


    def close(self):
        """Just a pass (nothing to close)."""
        pass


    def __getstate__(self):
        """Get pickled dtype and fill values."""
        return (self.dtype, self.fill_value)


    def __setstate__(self, state):
        """Set pickled dtype and fill value."""
        self.dtype, self.fill_value = state


class ThreadLocalData:
    """
    Creates a copy of the current dataset and vrt for every thread that reads
    from it. This is to avoid limitations of gdal and multi-threading. See
    the method of the same name by the stackstac team, who are the originators:
    http://github.com/gjoseph92/stackstac.
    """
    
    def __init__(self, ds, vrt=None):
        self._url = ds.name
        self._driver = ds.driver
        self._open_options = ds.options
        
        # set env defaults 
        self._env = {
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
            }
        
        # cache scale and offset for non-locking access
        self.scale_offset = (ds.scales[0], ds.offsets[0])
        
        # set up vrt params if exist
        if vrt is not None:
            self._vrt_params = {
                'crs': vrt.crs.to_string(),
                'resampling': vrt.resampling,
                'tolerance': vrt.tolerance,
                'src_nodata': vrt.src_nodata,
                'nodata': vrt.nodata,
                'width': vrt.width,
                'height': vrt.height,
                'src_transform': vrt.src_transform,
                'transform': vrt.transform,
                #'dtype': vrt.working_dtype,     # currently not avail in arcgis 2.8 rasterio
                #'warp_extras': vrt.warp_extras  # currently not avail in arcgis 2.8 rasterio
            }
        else:
            self._vrt_params = None
            
        # set up local threading data
        self._threadlocal = threading.local()
        self._threadlocal.ds = ds
        self._threadlocal.vrt = vrt      
        
        # lock thread!
        self._lock = threading.Lock()


    def _open(self):
        """
        Opens a cog/url via rasterio to obtain basic info about raster. 
        No reading is done, just a quick check of metadata. If warping
        is required, warping is done via rasterio warpedvrt without read.
        """
        
        # init rasterio env for efficient open
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
        
            # open url via rasterio
            result = ds = rasterio.open(self._url, 
                                        sharing=False, 
                                        driver=self._driver,
                                        **self._open_options)
            
            # init vrt variable
            vrt = None
            
            # if vrt exists...
            if self._vrt_params:
            
                # init rasterio env for vrt warp
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
                    result = vrt = WarpedVRT(ds,
                                             sharing=False, 
                                             **self._vrt_params)
                    
        # in current lock, set ds and vrt
        with self._lock:
            self._threadlocal.ds = ds
            self._threadlocal.vrt = vrt
            
        return result
    
    @property
    def dataset(self):
        """In current locked thread, return vrt if exists, else set dataset."""
        try:
            with self._lock:
                if self._threadlocal.vrt:
                    return self._threadlocal.vrt
                else:
                    self._threadlocal.ds
        except AttributeError():
            return self._open()
        
        
    def read(self, window, **kwargs):
        """Read current thead dataset, opening new copy on first thread access."""
        with rasterio.Env(VSI_CACHE=False, **self._env):
            return self.dataset.read(1, window=window, **kwargs)


    def close(self):
        """Release every thread's reference to its dataset, allowing them to close."""
        with self._lock:
            self._threadlocal = threading.local()


    def __getstate__(self):
        """Get pickled meta."""
        raise RuntimeError('Error during get pickle.')

        
    def __setstate__(self, state):
        """Set pickled meta."""
        raise RuntimeError('Error during set pickle')



# # # constructor functions
def fetch_stac_data(stac_endpoint=None, collections=None, start_dt=None, end_dt=None, bbox=None, slc_off=False, sort_time=True, limit=250):
    """
    Takes a stac endoint url (e.g., 'https://explorer.sandbox.dea.ga.gov.au/stac/search'),
    a list of stac assets (e.g., ga_ls5t_ard_3 for Landsat 5 collection 3), a range of
    dates, etc. and a bounding box in lat/lon and queries for all available metadata on the 
    digital earth australia (dea) aws bucket for these parameters. If any data is found for
    provided search query, a list of dictionaries containing image/band urls and other 
    vrt-friendly information is returned for further processing.
    
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
    bbox : list of ints/floats
        The bounding box of area of interest for which to query for 
        satellite data. Is in latitude and longitudes with format: 
        (min lon, min lat, max lon, max lat).
    slc_off : bool
        Whether to include Landsat 7 errorneous SLC data. Only relevant
        for Landsat data. Default is False - images where SLC turned off
        are not included.
    sort_time : bool
        Ensure the returned satellite data items are sorted by time in
        list of dictionaries.
    limit : int
        Limit number of dea aws items per page in query. Recommended to use
        250. Max is 999. 
     
    Returns
    ----------
    feats : list of dicts of returned stac metadata items.
    """
    
    # set headers for stac query
    headers = {
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip',
        'Accept': 'application/geo+json'
    }
    
    # notify
    print('Beginning STAC search for items. This can take awhile.')
            
    # check stac endpoint provided
    if stac_endpoint is None:
        raise ValueError('Must provide a STAC endpoint.')

    # prepare collection list
    if collections is None:
        collections = []
    if not isinstance(collections, (list)):
        collections = [collections]
        
    # get dt objects of date strings
    if start_dt is None or end_dt is None:
        raise ValueError('No start or end date provided.')
    else:
        start_dt_obj = datetime.strptime(start_dt, "%Y-%m-%d")
        end_dt_obj = datetime.strptime(end_dt, "%Y-%m-%d")
        
    # iter each stac collection
    feats = []
    for collection in collections:
        
        # notify
        print('Searching collection: {}'.format(collection))
                
        # set up fresh query
        query = {}

        # add collection to query
        if isinstance(collection, str):
            query.update({'collections': collection})
        else:
            raise TypeError('Collection must be a string.')

        # add bounding box to query if correct
        if isinstance(bbox, list) and len(bbox) == 4:
            query.update({'bbox': bbox})
        else:
            raise TypeError('Bounding box must contain four numbers.')

        # add start and end date to query. correct slc for ls7 if exists, requested
        if isinstance(start_dt, str) and isinstance(end_dt, str):
            
            if collection == 'ga_ls7e_ard_3' and not slc_off:
                print('Excluding SLC-off times.')
                
                # fix date if after slc data ended
                slc_dt = '2003-05-31'
                start_slc_dt = slc_dt if not start_dt_obj.year < 2003 else start_dt
                end_slc_dt = slc_dt if not end_dt_obj.year < 2003 else end_dt
                dt = '{}/{}'.format(start_slc_dt, end_slc_dt)
         
            else:
                dt = '{}/{}'.format(start_dt, end_dt)
            
            # update query
            query.update({'datetime': dt})
            
        else:
            raise TypeError('Start/end date must be of string type.')

        # add query limit
        query.update({'limit': limit})

        # perform initial post query
        res = requests.post(stac_endpoint, headers=headers, json=query)

        # if response, add feats to list
        if res.ok:
            feats += [f for f in res.json().get('features')]
            next_url = get_next_url(res.json())
        else:
            raise ValueError('Could not connect to DEA endpoint.')

        # keep appending next page to list if exists
        while next_url is not None:
            next_res = requests.get(next_url)
            if not next_res.ok:
                raise ValueError('Could not connect to DEA endpoint.')

            # add to larger feats list and get next if exists
            feats += [f for f in next_res.json().get('features')]
            next_url = get_next_url(next_res.json())
            
    # sort by time (low to high)
    if sort_time:
        print('Sorting result by time (old to new).')
        f_sort = lambda feat: feat['properties'].get('datetime', '')
        feats = sorted(feats, key=f_sort)
    
    # notify and return
    print('Found a total of {} scenes.'.format(len(feats)))
    return feats


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



# # # core data functions
def apply_cog_reader(asset_chunk, meta, resampler, dtype, fill_value, rescale):
    """
    Takes a single asset dask chunk (i.e., in our case a single satellite band)
    and associated metadata, subsets that chunk to a rasterio window object
    based on a projected bounds of coordinates, and wraps that chunk subset
    into a COGReader class (which reads rasters into memory).
    
    Parameters
    ----------
    asset_chunk: dask array
        A single chunk from a dask array of structure (url, [bbox]). 
    meta: dict
        A dictionary of metadata obtained from stac metadata.
    resampler: rasterio Resampler object
        A rasterio resampler object type of either NearestNeighbor or
        Bilinear.
    dtype : numpy type
        The numpy data type for output dask chunk.
    fill_value : int or float
        The value to use when pixel is detected or converted to no data.
    rescale : bool
        Whether rescaling of pixel vales by the scale and offset set on the
        dataset. Defaults to True.

    Returns
    ----------
    cog reader and window : a tuple containing the chunk wrapped in a 
    cog reader class with the associated rasterio window to read.
    """
    
    # obtain just the asset chunk element
    asset_chunk = asset_chunk[0, 0]
    
    # extract just the url from asset
    url = asset_chunk['url']
    if url is None:
        return None
        
    # create window of chunk using embedded bounds for current chunk
    win = windows.from_bounds(*asset_chunk['bounds'],
                              transform=meta.get('transform'))
    
    # wrap window into a cog reader class
    cog_reader = COGReader(url=url,
                           meta=meta,
                           resampler=resampler,
                           dtype=dtype,
                           fill_value=fill_value,
                           rescale=rescale)
    
    # return reader and window
    return (cog_reader, win)


def fetch_raster_window(asset_entry, slices):
    """
    Takes an asset chunk and asociated fake slices created from the 
    convert_to_dask method and fetches a window of raster data. If
    the slice covers the window, the cogreader wrapping the asset
    reads the current window and returns data. If no overlap exists,
    an empty window is returned of current window shape.
    
    Parameters
    ----------
    asset_entry: dask array
        A chunked dask array with cog reader mapped to each chunk. 
    slices: numpy array
        Sliced up rows and columns of dask array for parallel reading.

    Returns
    ----------
    Data that as been read within a window. If no data, empty window
    is returned.
    
    """
    
    # read current window from input sliced dimensions
    current_window = windows.Window.from_slices(*slices)
    
    # unpack asset entry into cog reader and window if exists
    if asset_entry is not None:
        reader, asset_window = asset_entry

        # if window being fetched overlaps with the asset, read!
        if windows.intersect(current_window, asset_window):
            data = reader.read(current_window)
            return data[None, None]

    # if no dataset or no intersection with window, return empty array
    return np.broadcast_to(np.nan, (1, 1) + windows.shape(current_window))

# PIXEL CENTERING NEEDS CHECK HERE
def build_coords(feats, assets, meta, pix_loc='topleft'):
    """
    Takes stac feats, assets and metadata outputs and
    generates grid of spatial coordinates. Simplified implementation
    of stackstac method: http://github.com/gjoseph92/stackstac. For tenement
    tools, we are only ever going to need times, bands, and x, y coordinate
    dimensions.
    
    Parameters
    ----------
    feats: list
        A list of dicts of stac metadata items produced from fetch stac data
        function.
    assets: list
        A list of satellite band names. Must be correct names for landsat and
        sentinel from dea aws.
    meta : dict
        A dictionary of metadata information from stac for use in creating
        xr dataset attribution.
    pix_loc : str
        Alignment of pixel cells. If topleft is used, coordinates set at topleft
        of pixel. If center is used, centroid of pixel used. 
        
    Returns
    ----------
    coords : dict of x, y coordinate arrays
    dims : list of dimension names
    """
    
    # notify
    print('Creating dataset coordinates and dimensions.')
    
    # parse datetime from stac features
    times = [f.get('properties').get('datetime') for f in feats]
    times = pd.to_datetime(times, infer_datetime_format=True, errors='coerce')
    
    # remove timezone and milliseconds if exists, xr and arcmap doesnt like
    times = times.tz_convert(None) if times.tz is not None else times
    times = times.strftime('%Y-%m-%dT%H:%M:%S').astype('datetime64[ns]')
        
    # prepare xr dims and coords
    dims = ['time', 'band', 'y', 'x']
    coords = {'time': times, 'band': assets}
    
    # set pixel coordinate position
    if pix_loc == 'center':
        pixel_center = True
    elif pix_loc == 'topleft':
        pixel_center = False
    else:
        raise ValueError('Pixel position not supported')
        
    # generate coordinates
    if meta.get('transform').is_rectilinear:
        
        # as rectilinear, we can just use arange
        min_x, min_y, max_x, max_y = meta.get('bounds')
        res_x, res_y = meta.get('resolutions_xy')
        
        # correct if pixel center
        if pixel_center:
            min_x = min_x + (res_x / 2)
            max_x = max_x + (res_x / 2)
            min_y = min_y + (res_y / 2)
            max_y = max_y + (res_y / 2)
            
        # gen x, y ranges
        h, w = meta.get('shape')
        x_range = pd.Float64Index(np.linspace(min_x, max_x, w, endpoint=False))
        y_range = pd.Float64Index(np.linspace(max_y, min_y, h, endpoint=False))
        
    else:
        # get offset depending on pixel position
        off = 0.5 if pixel_center else 0.0

        # gen x, y ranges
        h, w = meta.get('shape')
        x_range, _ = meta.get('transform') * (np.arange(w) + off, np.zeros(w) + off)
        _, y_range = meta.get('transform') * (np.zeros(h) + off, np.arange(h) + off)
        
    # set coords
    coords['y'] = y_range
    coords['x'] = x_range

    # notify and return
    print('Created coordinates and dimensions successfully.')
    return coords, dims
  
# ADD OTHER ATTRIBUTES NEEDED BY UPDATE FUNC
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



# # # helper functions
def get_next_url(json):
    """
    Small helper function to parse json and look for 'next' url in 
    stac item. Used to move through pages in stac queries.
    
    Parameters
    -------------
    json : dict
        A dictionary of json elements returned from stac.
    
    Returns
    ----------
    A url.
    """
    
    # parse json doc and look for link ref
    for link in json.get('links'):
        if link.get('rel') == 'next':
            return link.get('href')
        
    # else return nothing
    return None


def do_transform(bounds, resolution):
    """
    Helper function to generate an affine transformation on raw
    transformation obtained from stac metadata.
    
    Parameters
    -------------
    bounds : list of bounds (e.g., bbox)
        Two or more lists of bounding box coordinates.
    resolution : tuple or list of int
        Pixel cell size in tuple e.g., (30, 30) for 30 squarem.
        
    Returns
    ----------
    An affine transformation object .
    """
    
    # perform affine transform (xscale, 0.0, xoff, 0.0, yscale, yoff)
    return affine.Affine(resolution[0], 0.0, bounds[0], 0.0, -resolution[1], bounds[3])

@lru_cache(maxsize=64)
def cached_transform(from_epsg=4326, to_epsg=3577, skip_equivalent=True, always_xy=True):
    """
    Helper function to perform a cached pyproj transform. Transforms
    are computationally slow, so caching it during 100s of identical
    operations speeds us up considerablly. Implemented via
    LRU cache.
        
    Parameters
    -------------
    from_epsg : int
        A epsg code as int (e.g., wgs84 as 4326). Default is wgs84.
    to_epsg : int
        As above, except for destination epsg. Default is gda albers.
    skip_equivalent : bool
        Whether to skip re-cache of already cached operations. Obviously,
        we set this to True.
    always_xy : bool
        Whether transform method will accept and return traditional GIS
        coordinate order e.g. lon, lat and east, north. Set to True.
        
    Returns
    ----------
    Cached pyproject transformer object.
    """

    # transform from epsg to epsg
    return pyproj.Transformer.from_crs(from_epsg, to_epsg, skip_equivalent, always_xy)


def bbox_from_affine(aff, ysize, xsize, from_epsg=4326, to_epsg=3577):
    """
    Calculate bounding box from pre-existing affine transform.
    
    Parameters
    -------------
    aff : affine object or list of numerics 
        A pre-existing affine transform array, usually produced
        by pyproj or similar.
    ysize : int
        Number of cells on y-axis.
    xsize : int
        Number of cells on x-axis.
    from_epsg : int
        A epsg code as int (e.g., wgs84 as 4326). Default is wgs84.
    to_epsg : int
        As above, except for destination epsg. Default is gda albers.
        
    Returns
    ----------
    A list of int/floats of bbox.
    """
    
    # affine calculation
    x_nw, y_nw = aff * (0, 0)
    x_sw, y_sw = aff * (0, ysize)
    x_se, y_se = aff * (xsize, ysize)
    x_ne, y_ne = aff * (xsize, 0)

    # set x and y extents
    x_ext = [x_nw, x_sw, x_se, x_ne]
    y_ext = [y_nw, y_sw, y_se, y_ne]

    # transform if from/to epsg differs, else dont transform
    if from_epsg != to_epsg:
        transformer = cached_transform(from_epsg, to_epsg)
        x_ext_proj, y_ext_proj = transformer.transform(x_ext, y_ext, errcheck=True)
    
    else:
        x_ext_proj = x_ext
        y_ext_proj = y_ext
        
    # prepare and return bbox (l, b, r, t)
    return [min(x_ext_proj), min(y_ext_proj), max(x_ext_proj), max(y_ext_proj)]


def reproject_bbox(source_epsg=4326, dest_epsg=3577, bbox=None):
    """
    Helper function to reproject given bounding box (default wgs84)
    to requested coordinate system)
    
    Parameters
    -------------
    source_epsg : int
        A epsg code as int (e.g., wgs84 as 4326). Default is wgs84.
    dest_epsg : int
        As above, except for destination epsg. Default is gda94 albers.
    bbox : list of ints/floats
        The bounding box of area of interest for which to query for 
        satellite data. Is in latitude and longitudes with format: 
        (min lon, min lat, max lon, max lat).
    
    Returns
    ----------
    pbbox : list of reprojected coordinates of bbox.
    """
    
    # deconstruct original bbox array
    l, b, r, t = bbox
    
    # reproject from source epsg to destination epsg
    pbbox = transform_bounds(src_crs=source_epsg,
                             dst_crs=dest_epsg, 
                             left=l, bottom=b, right=r, top=t)
                             
    # return
    return pbbox


def bounds_overlap(*bounds):
    """
    Helper function to check if bounds within list overlap.
    Used to discard assets that may not overlap with user
    defined bbox, or other assets bboxes.
    
    Parameters
    -------------
    bounds : list of bounds (e.g., bbox)
       Two or more lists of bounding box coordinates
    
    Returns
    ----------
    A boolean indicating whether bounds overlap.
    """
    
    # zip up same coordinates across each array
    min_x_vals, min_y_vals, max_x_vals, max_y_vals = zip(*bounds)
    
    # check if overlaps occur and return
    return max(min_x_vals) < min(max_x_vals) and max(min_y_vals) < min(max_y_vals)


def snap_bbox(bounds, resolution):
    """
    Helper function to 'snap' bounding box, e.g., to
    the ceiling and floor depending on min and max values
    with consideration with resolution.
    
    Parameters
    -------------
    bounds : list of bounds (e.g., bbox)
        Two or more lists of bounding box coordinates.
    resolution : tuple or list of int
        Pixel cell size in tuple e.g., (30, 30) for 30 squarem.
        
    Returns
    ----------
    A 'snapped' bounding box.
    """
    
    # unpack coords, resolution from inputs
    min_x, min_y, max_x, max_y = bounds
    res_x, res_y = resolution
    
    # snap bounds!
    min_x = np.floor(min_x / res_x) * res_x
    max_x = np.ceil(max_x / res_x) * res_x
    min_y = np.floor(min_y / res_y) * res_y
    max_y = np.ceil(max_y / res_y) * res_y

    # return new snapped bbox
    return [min_x, min_y, max_x, max_y]


def union_bounds(*bounds):
    """
    Helper function to union all bound extents, typically
    for asset level and outut bound extents during data
    preparation. Essentially gets minimum bounding rectangle
    of all bounding coordinates in input.
    
    Parameters
    -------------
    bounds : multiple lists of bounding box coordinates (e.g., bbox).

    Returns
    ----------
    A list of unioned boundaries.
    """
    
    # zip up all coordinate arrays 
    pairs = zip(*bounds)
    
    # union and return
    return [
        min(next(pairs)),
        min(next(pairs)),
        max(next(pairs)),
        max(next(pairs))
    ]


def get_shape(bounds, resolution):
    """
    Helper function to get the shape (i.e., h, w) of a 
    bounds, with respect to resolution size.
    
    Parameters
    -------------
    bounds : list of bounds (e.g., bbox)
        Two or more lists of bounding box coordinates.
    resolution : tuple or list of int
        Pixel cell size in tuple e.g., (30, 30) for 30 squarem.
        
    Returns
    ----------
    The size of the bounds as list e.g., (h, w)
    """

    # unpack coords and resolution
    min_x, min_y, max_x, max_y = bounds
    res_x, res_y = resolution
    
    # calc shape (width and height) from bounds and resolution
    w = int((max_x - min_x + (res_x / 2)) / res_x)
    h = int((max_y - min_y + (res_y / 2)) / res_y)
    
    # return
    return [h, w]


def get_nodata_for_window(window, dtype, fill_value):
    """
    Helper function to create an window (i.e. a numpy array) with 
    a specific value representing no data (i.e., np.nan or 0) if
    error occurred during cog reader function.
    
    Parameters
    -------------
    window : rasterio window object
        A rasterio-based window subset of actual raster or cog data.
    dtype : str or np dtype
        Data type of output window object
    fill_value : various
        Any type of numpy data type e.g., int8, int16, float16, float32, nan.
        
    Returns
    ----------
    A numpy array representing original window size but of new user-defined
    fill value.
    """
    
    # check if fill value provided
    if fill_value is None:
        raise ValueError('Fill value must be anything but None.')
        
    # get height, width of window and fill new numpy array
    h, w = int(window.height), int(window.width)
    return np.full((h, w), fill_value, dtype)
    
