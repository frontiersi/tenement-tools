# cog
'''
This script contains functions for searching the digital earth australia (dea)
public aws bucket using stac, preparing cog data into dask arrays for lazy 
loading, and computation into a local netcdf with bands as variables, x, y
and time dimensions. This script is a simplified, python 3.8-compatible and 
dea-focused version of the excellent stackstac python library 
(http://github.com/gjoseph92/stackstac). We highly recommended you use
stackstac if you need to use any aws bucket other than the dea public database,
and to cite them when they have a white paper available if using in research.

See associated Jupyter Notebook cog.ipynb for a basic tutorial on the
main functions and order of execution.

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

# meta
class COGReader:
    """
    Method is a very basic version of stackstac and rasterio.
    """
    
    def __init__(self, url, meta, resampler, dtype=None, fill_value=None, rescale=True):
        self.url = url
        self.meta = meta
        self.resampler = resampler
        self.dtype = dtype
        self.rescale = rescale
        self.fill_value = fill_value
        
        # set env defaults 
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
        """

        # open efficient env
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
            try:
                ds = rasterio.open(self.url, sharing=False)
            except:
                return NANReader(dtype=self.dtype, 
                                 fill_value=self.fill_value)           

            # check if 1 band, else fail
            if ds.count != 1:
                ds.close()
                raise ValueError('more than 1 band in single cog. not supported.')
                
            # create vrt if current tif differ from requested params
            vrt = None
            if self.meta.get('vrt_params') != {
                'crs': ds.crs.to_epsg(),
                'transform': ds.transform,
                'height': ds.height,
                'width': ds.width
            }:
                vrt_meta = self.meta.get('vrt_params')
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
                    vrt = WarpedVRT(ds,
                                    sharing=False, 
                                    resampling=self.resampler,
                                    crs=vrt_meta.get('crs'),
                                    transform=vrt_meta.get('transform'),
                                    height=vrt_meta.get('height'),
                                    width=vrt_meta.get('width'))

        # apply reader
        if ds.driver in ['GTiff']:
            return ThreadLocalData(ds, vrt=vrt)
        else:
            ds.close()
            raise TypeError('COGreader currently only supports GTiffs/COGs.')
     
    @property
    def dataset(self):
        """
        Within the current locked thread, return dataset
        if exists, else open one.
        """
        with self._dataset_lock:
            if self._dataset is None:
                self._dataset = self._open()
            return self._dataset
        
        
    def read(self, window, **kwargs):
        """
        Read a window of a dataset. Also will
        rescale if requested. Finally, dtype is
        converted and mask values filled to fill_value.
        """
        # open dataset if exist, or load and open if not
        reader = self.dataset
        try:
            # read the open dataset. mask for safer scaling/offsets
            result = reader.read(window=window, masked=True, **kwargs)
        except:
            return NANReader(dtype=self.dtype, 
                             fill_value=self.fill_value)    
    
        # rescale to scale and offset values 
        if self.rescale:
            scale, offset = reader.scale_offset
            if scale != 1 and offset != 0:
                result *= scale
                result += offset
            
        # convert type, fill mask areas with requested nodata vals
        result = result.astype(self.dtype, copy=False)
        result = np.ma.filled(result, fill_value=self.fill_value)
        return result
    
    
    def close(self):
        """
        Within the current locked thread, close
        dataset if exists, else return None.
        """
        with self._dataset_lock:
            if self._dataset is None:
                return None
            self._dataset.close()
            self._dataset = None
    
    
    def __del__(self):
        """
        Called when garbage collected after close.
        Can get around some multi-threading issues 
        such as "no dataset_lock" exists.
        """
        try:
            self.close()
        except:
            pass
    
    
    def __getstate__(self):
        """
        Get pickled meta.
        """
        print('Getting picked data!')
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
        """
        Set pickled meta.
        """
        print('Setting picked data!')
        self.__init__(
            url=state.get('url'),
            meta=state.get('meta'),
            resampler=state.get('resampler'),
            dtype=state.get('dtype'),
            fill_value=state.get('fill_value'),
            rescale=state.get('rescale')
        )

# meta
class NANReader:
    """
    Reader that returns a constant (nodata) value 
    for all subsequent reads.
    """
    
    # set scale offset
    scale_offset = (1.0, 0.0)
    
    def __init__(self, dtype=None, fill_value=None, **kwargs):
        self.dtype = dtype
        self.fill_value = fill_value
        
        
    def read(self, window, **kwargs):
        """
        """
        
        # todo was this: todo i can delete this safely... how come this didnt fire before though?
        #return nodata_for_window(window, 
                                 #self.dtype, 
                                 #self.fill_value)
                                 
        # changed to this to use my func todo if error, check
        return get_nodata_for_window(window, self.dtype, self.fill_value)
        
      
    def close(self):
        pass
    
    
    def __getstate__(self):
        """
        Get pickled dtype and fill value.
        """
        return (self.dtype, self.fill_value)
    

    def __setstate__(self, state):
        """
        Set pickled dtype and fill value.
        """
        self.dtype, self.fill_value = state
    
# meta
class ThreadLocalData:
    """
    Re-open dataset.
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
                #'dtype': vrt.working_dtype,    # currently not avail in arcgis 2.8 rasterio
                #'warp_extras': vrt.warp_extras # currently not avail in arcgis 2.8 rasterio
            }
        else:
            self._vrt_params = None
            
        # set up local threading data
        self._threadlocal = threading.local()
        self._threadlocal.ds = ds
        self._threadlocal.vrt = vrt      
        
        # lock!
        self._lock = threading.Lock()
        
        
    def _open(self):
        """
        Open COG url and VRT if required.
        """
        # open efficient env and url
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
            result = ds = rasterio.open(self._url, 
                                        sharing=False, 
                                        driver=self._driver,
                                        **self._open_options)
            
            # open efficient env and vrt
            vrt = None
            if self._vrt_params:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True, **self._env):
                    result = vrt = WarpedVRT(ds,
                                             sharing=False, 
                                             **self._vrt_params)
                    
        # in current lock, store ds and vrt
        with self._lock:
            self._threadlocal.ds = ds
            self._threadlocal.vrt = vrt
            
        return result
    
    @property
    def dataset(self):
        """
        Within the current locked thread, return vrt
        if exists, else ds. If neither, open a ds or vrt.
        """
        try:
            with self._lock:
                if self._threadlocal.vrt:
                    return self._threadlocal.vrt
                else:
                    self._threadlocal.ds
        except AttributeError():
            return self._open()
        
        
    def read(self, window, **kwargs):
        """
        Read from current thread's dataset, opening
        a new copy of the dataset on first access
        from each thread.
        """
        with rasterio.Env(VSI_CACHE=False, **self._env):
            return self.dataset.read(1, window=window, **kwargs)
        
        
    def close(self):
        """
        Release every thread's reference to its dataset,
        allowing them to close. 
        """
        with self._lock:
            self._threadlocal = threading.local()
            
    
    def __getstate__(self):
        raise RuntimeError('I shouldnt be getting pickled...')

        
    def __setstate__(self, state):
        raise RuntimeError('I shouldnt be getting un-pickled...')


# # # core functions
def fetch_stac_data(stac_endpoint, collections, start_dt, end_dt, bbox, slc_off=False, sort_time=True, limit=250):
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
    start_dt_obj = datetime.strptime(start_dt, "%Y-%m-%d")
    end_dt_obj = datetime.strptime(start_dt, "%Y-%m-%d")
        
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




# todo checks, meta, union, overlap, resolution, center or corner align/snap? does no data get removed? not sure ins tackstac either
def prepare_data(feats, assets=None, bounds_latlon=None, bounds=None, epsg=3577, resolution=30, snap_bounds=True, force_dea_http=True):
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
                
                    # if projected asset bbox does not exist, error
                    if asset_bbox_proj is None:
                        raise ValueError('Not enough STAC infomration to build bounds.')
                        
                    # now, if output bounds does not exist, use projcted asset bbox or union
                    if out_bounds is None:
                        out_bounds = asset_bbox_proj
                    else:
                        out_bounds = union_bounds(asset_bbox_proj, out_bounds)
    
                else:
                    # skip if asset bbox does not overlap with bounds
                    overlaps = bounds_overlap(asset_bbox_proj, bounds)
                    if asset_bbox_proj is not None and not overlaps:
                        continue # move to next asset
                    
                # do resolution todo: implement auto resolution capture
                if resolution is None:
                    raise ValueError('Need to implement auto-find resolution.')
                    
                # force aws s3 to https todo make this work with aws s3 bucket
                href = asset.get('href')
                if force_dea_http:
                    href = href.replace('s3://dea-public-data', 'https://data.dea.ga.gov.au')
                    
                # add info to asset table
                asset_table[feat_idx, asset_idx] = (href, asset_bbox_proj)
                
        # creates row in array that has 1 row per scene, n columns per requested band where (url, [l, b, r, t])
        href = asset["href"].replace('s3://dea-public-data', 'https://data.dea.ga.gov.au')
        asset_table[feat_idx, asset_idx] = (href, asset_bbox_proj)
        
    # snap boundary coordinates
    if snap_bounds:
        out_bounds = snap_bbox(out_bounds, 
                               out_resolution)
     
    # transform and get shape for top-level
    transform = do_transform(out_bounds, out_resolution)
    shape = get_shape(out_bounds, out_resolution)
        
    # get table of nans where any feats/assets where asset missing/out bounds
    isnan_table = np.isnan(asset_table['bounds']).all(axis=-1)
    feat_isnan = isnan_table.all(axis=1)  # any items all empty?
    asset_isnan = isnan_table.all(axis=0) # any assets all empty?
    
    # remove offending items. np.ix_ removes specific cells (col, row)
    if feat_isnan.any() or asset_isnan.any():
        asset_table = asset_table[np.ix_(~feat_isnan, ~asset_isnan)]
        feats = [feat for feat, isnan in zip(feats, feat_isnan) if not isnan]
        assets = [asset for asset, isnan in zip(assets, asset_isnan) if not isnan]
        
    # create final top-level raster metadata dict
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
    print('Translated raw STAC data successfully.')
    return meta, asset_table










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







# checks, meta, todos, mpve fetch raster window func into here?
def convert_to_dask(meta=None, asset_table=None, chunksize=512, resampling='nearest',
                    dtype='int16', fill_value=-999, rescale=True):
    
    """
    Data is of type dictionary with everything that comes out
    of prepare_data.
    """
       
    # imports    
    import itertools
    import warnings

    # notify
    print('Converting data into dask array.')
    
    # check if meta and array provided
    if meta is None or asset_table is None:
        raise ValueError('Must provide metadata and assets.')
        
    # checks
    if resampling not in ['nearest', 'bilinear']:
        raise ValueError('Resampling method not supported.')
        
    # set resampler
    if resampling == 'nearest':
        resampler = Resampling.nearest
    else:
        resampler = Resampling.bilinear
    
    # check type of dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # check fill value
    #if fill_value is None and errors_as_nodata:
        #raise ValueError('cant do')
        
    # check if we can use fillvalue
    if fill_value is not None and not np.can_cast(fill_value, dtype):
        raise ValueError('Fill value incompatible with output dtype.')
                
    # set errors or empty tuple (not none)
    #errors_as_nodata = errors_as_nodata or ()

    # see stackstac for explanation of this logic here
    
    # make urls into dask array with 1-element chunks (i.e. 1 chunk per asset (i.e.e band))
    da_asset_table = dask_array.from_array(asset_table, 
                                           chunks=1, 
                                           #inline_array=True # need high ver of dask
                                           name='assets_' + dask.base.tokenize(asset_table))
    
    # map to blocks
    ds = da_asset_table.map_blocks(apply_cog_reader,
                                   meta,
                                   resampler,
                                   dtype,
                                   fill_value,
                                   rescale,
                                   meta=da_asset_table._meta)
    
    # generate a fake array from shape and chunksize
    shape = meta.get('shape')
    name = 'slices_' + dask.base.tokenize(chunksize, shape)
    chunks = dask_array.core.normalize_chunks(chunksize, shape)
    keys = itertools.product([name], *(range(len(b)) for b in chunks))
    slices = dask_array.core.slices_from_chunks(chunks)
    
    # stick slices into an array to use dask blockwise logic
    fake_slices = dask_array.Array(dict(zip(keys, slices)), 
                                   name, 
                                   chunks, 
                                   meta=ds._meta)
    
    # apply blockwise
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore", category=dask_array.core.PerformanceWarning)
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

# cogs
def apply_cog_reader(asset_chunk, meta, resampler, dtype, fill_value, rescale):
    """
    For each dask url/bounds chunk, apply the local
    threaded cog reader classes. 
    """
    
    # remove added array dim
    asset_chunk = asset_chunk[0, 0]
    
    # get url
    url = asset_chunk['url']
    if url is None:
        return None
    
    # get requested bounds
    win = windows.from_bounds(*asset_chunk['bounds'],
                              transform=meta.get('transform'))
    
    # wrap in cog reader
    cog_reader = COGReader(
        url=url,
        meta=meta,
        resampler=resampler,
        dtype=dtype,
        fill_value=fill_value,
        rescale=rescale
    )
    
    # return reader and window
    return (cog_reader, win)

# checks, meta
def build_coords(feats, assets, meta, pix_loc='topleft'):
    """
    Very basic version of stackstac. We are only concerned with
    times, bands, ys and xs for our product.
    """
    
    # parse datetime from stac features
    times = [f.get('properties').get('datetime') for f in feats]
    times = pd.to_datetime(times, infer_datetime_format=True, errors='coerce')
    
    # timezone can exist, remove it if so (xarray dont likey)
    times = times.tz_convert(None) if times.tz is not None else times
    
    # strip milliseconds
    #ds['time'] = ds.time.dt.strftime('%Y-%m-%d').astype(np.datetime64)
    times = times.strftime('%Y-%m-%dT%H:%M:%S').astype('datetime64[ns]')
        
    # prepare xr dims and coords
    dims = ['time', 'band', 'y', 'x']
    coords = {
        'time': times,
        'band': assets
    }
    
    # set pixel coordinate position
    if pix_loc == 'center':
        pixel_center = True
    elif pix_loc == 'topleft':
        pixel_center = False
    else:
        raise ValueError('Pixel position not supported')
        
    # generate coordinates
    if meta.get('transform').is_rectilinear:
        
        # we can just use arange for this - quicker
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
        # get offset dependong on pixel position
        off = 0.5 if pixel_center else 0.0

        # gen x, y ranges
        h, w = meta.get('shape')
        x_range, _ = meta.get('transform') * (np.arange(w) + off, np.zeros(w) + off)
        _, y_range = meta.get('transform') * (np.zeros(h) + off, np.arange(h) + off)
        
    # set coords
    coords['y'] = y_range
    coords['x'] = x_range
    
    # get properties as coords
    # not needed
    
    # get band as coords
    # not needed
    
    return coords, dims

# checks, meta, improve dynanism
def build_attributes(ds, meta, fill_value, collections, slc_off, bbox, resampling):
    """
    """
    
    # imports 
    import osr
    #import affine
    
    # assign spatial_ref coordinate
    crs = int(meta.get('epsg'))
    ds = ds.assign_coords({'spatial_ref': crs})
        
    # get wkt from epsg 
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    wkt = srs.ExportToWkt()
    
    # assign wkt to spatial ref 
    grid_mapping_name = 'albers_conical_equal_area' # todo get this dynamically?
    ds['spatial_ref'] = ds['spatial_ref'].assign_attrs({'spatial_ref': wkt, 
                                                        'grid_mapping_name': grid_mapping_name})
    
    # assign global crs and grid mapping 
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
    
    # do same for y coordinate attributes
    ds['y'] = ds['y'].assign_attrs({
        'units': 'metre',
        'resolution': res_y,
        'crs': 'EPSG:{}'.format(crs)
    })    
    
    # add attributes custom for cog fetcher
    ds = ds.assign_attrs({'transform': tuple(transform)})
    
    # set output resolution depending on type
    res = meta.get('resolutions_xy')
    res = res[0] if res[0] == res[1] else res
    ds = ds.assign_attrs({'res': res})
    
    # set no data values
    ds = ds.assign_attrs({'nodatavals': fill_value})
    
    # set collections from original query
    ds = ds.assign_attrs({'orig_collections': tuple(collections)})
    
    # set original bbox
    ds = ds.assign_attrs({'orig_bbox': tuple(bbox)})
    
    # set slc off from original query
    slc_off = 'True' if slc_off else 'False'
    ds = ds.assign_attrs({'orig_slc_off': slc_off})
    
    # set original resample method 
    ds = ds.assign_attrs({'orig_resample': resampling})
    
    # iter each var and update attributes 
    for data_var in ds.data_vars:
        ds[data_var] = ds[data_var].assign_attrs({
            'units': '1', 
            'crs': 'EPSG:{}'.format(crs), 
            'grid_mapping': 'spatial_ref'
        })
   
    return ds

# checks, meta
def remove_fmask_dates(ds, valid_class=[1, 4, 5], max_invalid=5, mask_band='oa_fmask', nodata_value=np.nan, drop_fmask=False):
    """
    Takes a xarray dataset (typically as dask)
    and computes mask band. From mask band,
    calculates percentage of valid vs invalid
    pixels per date. Returns a list of every
    date that is above the max invalid threshold.
    """
    
    # notify
    print('Removing dates where too many invalid pixels.')

    # check if x and y dims
    # todo

    # get 
    min_valid = 1 - (max_invalid / 100)

    # get total num pixels for one slice
    num_pix = ds['x'].size * ds['y'].size

    # subset mask band
    mask = ds[mask_band]

    # if dask, compute it
    if bool(mask.chunks):
        print('Mask band is currently dask. Computing, please wait.')
        mask = mask.compute()

    # set all valid classes to 1, else 0
    mask = xr.where(mask.isin(valid_class), 1.0, 0.0)
    
    # convert to float32 if nodata value is nan
    if nodata_value is np.nan:
        ds = ds.astype('float32')    

    # mask invalid pixels with user value
    print('Filling invalid pixels with nan')
    ds = ds.where(mask == 1.0, nodata_value)
    
    # calc percentage of valdi to total and get invalid dates
    mask = mask.sum(['x', 'y']) / num_pix
    valid_dates = mask['time'].where(mask >= min_valid, drop=True)

    # remove any non-valid times from dataset
    ds = ds.where(ds['time'].isin(valid_dates), drop=True)

    # drop mask band if requested
    if drop_fmask:
        print('Dropping mask band.')
        ds = ds.drop_vars(mask_band)
        
    return ds

# fix this?
def fetch_raster_window(asset_entry, slices):
    current_window = windows.Window.from_slices(*slices)
    
    if asset_entry is not None:
        reader, asset_window = asset_entry

        # check that the window we're fetching overlaps with the asset
        if windows.intersect(current_window, asset_window):
            data = reader.read(current_window)
            return data[None, None]

    # no dataset, or we didn't overlap
    return np.broadcast_to(np.nan, (1, 1) + windows.shape(current_window))



