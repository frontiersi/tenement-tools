# working

# gdal, rasterio env init - todo make this dynamic
#import os, certifi
#os.environ['GDAL_DATA']  = r'C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata'
#os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

import rasterio
import dask
import numpy as np
import pandas as pd
#import dask.array as da
import dask.array as dask_array
import xarray as xr
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import threading
import itertools
import warnings


import pyproj
from functools import lru_cache



def get_next_url(json):
    """
    Small helper function to parse json and
    look for 'next' url. Used during stac
    queries.
    
    Parameters
    -------------
    json : dict
        A dictionary of json elements.
    
    """
    
    # parse json doc and look for link ref
    for link in json.get('links'):
        if link.get('rel') == 'next':
            return link.get('href')
        
    # else, go home empty handed
    return None

# meta, move checks up to top? make the year comparitor cleaner
def fetch_stac_data(stac_endpoint, collections, start_dt, end_dt, bbox, slc_off=False, sort_time=True, limit=250):
    """
    bbox BBOX BBOX BBOX BBOX
    Bounding box (min lon, min lat, max lon, max lat)
    (default: None)
    
    datetime DATETIME   Single date/time or begin and end date/time (e.g.,
                        2017-01-01/2017-02-15) (default: None)
    """
    
    # imports
    import requests
    
    # set headers
    headers = {
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip',
        'Accept': 'application/geo+json'
    }
    
    # notify
    print('Beginning STAC search for items. This can take awhile.')
            
    # check stac_endpoint
    if stac_endpoint is None:
        raise ValueError('Must provide a STAC endpoint.')

    # prepare collection list
    if collections is None:
        collections = []
    if not isinstance(collections, (list)):
        collections = [collections]
        
    # iter each collection. stac doesnt return more than one
    feats = []
    for collection in collections:
        
        # notify
        print('Searching collection: {}'.format(collection))
        
        # set up fresh query
        query = {}

        # collections
        if isinstance(collection, str):
            query.update({'collections': collection})

        # bounding box
        if isinstance(bbox, list) and len(bbox) == 4:
            query.update({'bbox': bbox})

        # start and end date. consider slc for ls7
        if isinstance(start_dt, str) and isinstance(end_dt, str):
            if collection == 'ga_ls7e_ard_3' and not slc_off:
                print('> Excluding SLC-off times.')
                new_end_dt = end_dt if int(end_dt[:4]) < 2003 else '2003-05-31'  # this needs work
                dt = '{}/{}'.format(start_dt, new_end_dt)
            else:
                dt = '{}/{}'.format(start_dt, end_dt)
            query.update({'datetime': dt})

        # query limit
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

# checks, meta
def get_affine(transform):
    """
    Small helper function to computer affine
    from asset transform array.
    """
    
    # imports
    import affine
    
    # checks
    
    # compute affine
    return affine.Affine(*transform[:6]) # ignore scale, shift
    

# checks, meta
def reporject_bbox(source_epsg=4326, dest_epsg=None, bbox=None):
    """
    Basic function to reproject given coordinate
    bounding box (in WGS84).
    """
    
    # imports
    from rasterio.warp import transform_bounds

    # checks
    
    # reproject bounding box
    l, b, r, t = bbox
    return transform_bounds(src_crs=source_epsg, 
                            dst_crs=dest_epsg, 
                            left=l, bottom=b, right=r, top=t)

# meta, checks
@lru_cache(maxsize=64)
def cached_transform(from_epsg, to_epsg, skip_equivalent=True, always_xy=True):
    """
    A cached version of pyproject transform. The 
    transform operation is slow, so caching it
    during 100x operations speeds up the process
    considerablly. LRU cache implements this.
    """
    
    # checks
    
    # transform from epsg to epsg
    return pyproj.Transformer.from_crs(from_epsg, 
                                       to_epsg, 
                                       skip_equivalent, 
                                       always_xy)

# meta, checks
def bbox_from_affine(aff, ysize, xsize, from_epsg, to_epsg):
    """
    Calculate bounding box from affine transform.
    """
    
    # affine calculation
    x_nw, y_nw = aff * (0, 0)
    x_sw, y_sw = aff * (0, ysize)
    x_se, y_se = aff * (xsize, ysize)
    x_ne, y_ne = aff * (xsize, 0)

    # set x and y extents
    x_ext = [x_nw, x_sw, x_se, x_ne]
    y_ext = [y_nw, y_sw, y_se, y_ne]

    # transform if different crs', else just use existing
    if from_epsg != to_epsg:
        transformer = cached_transform(from_epsg, to_epsg)
        x_ext_proj, y_ext_proj = transformer.transform(x_ext, y_ext, errcheck=True)
    else:
        x_ext_proj = x_ext
        y_ext_proj = y_ext
        
    # prepare and return bbox (l, b, r, t)
    bbox = [min(x_ext_proj), min(y_ext_proj), max(x_ext_proj), max(y_ext_proj)]
    return bbox

# meta, check
def bounds_overlap(*bounds):
    """
    """
    
    # get same element across each array
    min_x_vals, min_y_vals, max_x_vals, max_y_vals = zip(*bounds)
    
    overlaps = max(min_x_vals) < min(max_x_vals) and max(min_y_vals) < min(max_y_vals)
    return overlaps


#meta, check
def snap_bbox(bounds, resolution):
    """
    """
    
    # imports
    import math
    
    # checks
    #
    
    # get coords and resolution
    min_x, min_y, max_x, max_y = bounds
    res_x, res_y = resolution
    
    # snap!
    min_x = math.floor(min_x / res_x) * res_x
    max_x = math.ceil(max_x / res_x) * res_x
    min_y = math.floor(min_y / res_y) * res_y
    max_y = math.ceil(max_y / res_y) * res_y

    # out we go
    snapped_bounds = [min_x, min_y, max_x, max_y]
    return snapped_bounds

# meta, checks
def do_transform(bounds, resolution):
    """
    Small helper function to do 
    """
    
    # imports
    from affine import Affine
    
    # checks
    
    # perform affine transform (xscale, 0, xoff, 0, yscale, yoff)
    transform = Affine(resolution[0],
                       0.0,
                       bounds[0],
                       0.0,
                       -resolution[1],
                       bounds[3])
    
    # return
    return transform


# meta, checks
def get_shape(bounds, resolution):
    """
    """
    
    # checks
    
    # get coords and resolution
    min_x, min_y, max_x, max_y = bounds
    res_x, res_y = resolution
    
    # calc shape i.e., width heihjt
    w = int((max_x - min_x + (res_x / 2)) / res_x)
    h = int((max_y - min_y + (res_y / 2)) / res_y)
    
    # pack and return
    hw = [h, w]
    return hw


# todo checks, meta, union, overlap, resolution, center or corner align/snap? does no data get removed? not sure ins tackstac either
def prepare_data(feats, assets=None, bounds_latlon=None, bounds=None, epsg=3577, 
                 resolution=30, snap_bounds=True, force_dea_http=True):
    """
    """
    
    # notify
    print('Translating raw STAC data into numpy format.')

    # checks
    if bounds_latlon is not None and bounds is not None:
        raise ValueError('Cannot provide both bounds latlon and bounds.')
        
    # check epsg
    if epsg != 3577:
        raise ValueError('EPSG 3577 only supported at this stage.')
            
    # set epsg, bounds
    out_epsg = epsg
    out_bounds = bounds

    # prepare resolution tuple
    if resolution is None:
        raise ValueError('Must set a resolution value.')
    elif not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    
    # set output res
    out_resolution = resolution
        
    # prepare and check assets list
    if assets is None:
        assets = []
    if not isinstance(assets, list):
        assets = [assets]
    if len(assets) == 0:
        raise ValueError('Must request at least one asset.')
                
    # todo check data type, get everything if empty list
    #asset_ids = assets
        
    # create an numpy asset table to store info
    asset_dt = np.dtype([('url', object), ('bounds', 'float64', 4)]) 
    asset_table = np.full((len(feats), len(assets)), None, dtype=asset_dt) 

    # check if feats exist
    if len(feats) == 0:
        raise ValueError('No items to prepare.')
        
    # iter feats
    for feat_idx, feat in enumerate(feats):
        
        # get top level meta
        feat_props = feat.get('properties')
        feat_epsg = feat_props.get('proj:epsg')
        feat_bbox = feat_props.get('proj:bbox')
        feat_shape = feat_props.get('proj:shape')
        feat_transform = feat_props.get('proj:transform')        
    
        # unpack assets
        feat_bbox_proj = None
        for asset_idx, asset_name in enumerate(assets):
            asset = feat.get('assets').get(asset_name)
            
            # get asset level meta, if not exist, use top level
            if asset is not None:
                asset_epsg = asset.get('proj:epsg', feat_epsg)
                asset_bbox = asset.get('proj:bbox', feat_bbox)
                asset_shape = asset.get('proj:shape', feat_shape)
                asset_transform = asset.get('proj:transform', feat_transform)
                asset_affine = None
                
                # prepare crs - todo handle when no epsg given. see stackstac
                out_epsg = int(out_epsg)
                
                # reproject bounds and out_bounds to user epsg
                if bounds_latlon is not None and out_bounds is None:
                    bounds = reporject_bbox(4326, out_epsg, bounds_latlon)
                    out_bounds = bounds

                # compute asset bbox via feat bbox. todo: what if no bbox in stac, or asset level exists?
                if asset_transform is None or asset_shape is None or asset_epsg is None:
                    raise ValueError('No feature-level transform and shape metadata.')
                else:
                    asset_affine = get_affine(asset_transform)
                    asset_bbox_proj = bbox_from_affine(asset_affine,
                                                       asset_shape[0],
                                                       asset_shape[1],
                                                       asset_epsg,
                                                       out_epsg)
                
                # compute bounds
                if bounds is None:
                    if asset_bbox_proj is None:
                        raise ValueError('Not enough STAC infomration to build bounds.')
                        
                    if out_bounds is None:
                        out_bounds = asset_bbox_proj
                    else:
                        #bound_union = union_bounds(asset_bbox_proj, out_bounds)
                        #out_bounds = bound_union
                        raise ValueError('Need to implement union.')
    
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


# checks, meta, errors_as_nodata param, play with method - do we need this sophistication?
# asset_entry_to_reader_and_window = change name
def convert_to_dask(meta=None, asset_table=None, chunksize=512, resampling='nearest',
                    dtype='int16', fill_value=-999, rescale=True):
    
    """
    Data is of type dictionary with everything that comes out
    of prepare_data.
    """
       
    # imports    
    import itertools
    import warnings
    from rasterio.enums import Resampling
    
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
        resampling = Resampling.nearest
    else:
        resampling = Resampling.bilinear
    
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
                                           name='asset_table_' + dask.base.tokenize(asset_table))
    
    # map to blocks
    ds = da_asset_table.map_blocks(asset_entry_to_reader_and_window,
                                   meta,
                                   resampling,
                                   dtype,
                                   fill_value,
                                   rescale,
                                   #gdal_env,
                                   #errors_as_nodata,
                                   #reader,
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
























# V these are from rio_reader # # # #

class ThreadLocalRioDataset:
    """
    Creates a copy of the dataset and VRT for every thread that reads from it.
    In GDAL, nothing allows you to read from the same dataset from multiple threads.
    The best concurrency support available is that you can use the same *driver*, on
    separate dataset objects, from different threads (so long as those datasets don't share
    a file descriptor). Also, the thread that reads from a dataset must be the one that creates it.
    This wrapper transparently re-opens the dataset (with ``sharing=False``, to use a separate file
    descriptor) for each new thread that accesses it. Subsequent reads by that thread will reuse that
    dataset.
    Note
    ----
    When using a large number of threads, this could potentially use a lot of memory!
    GDAL datasets are not lightweight objects.
    """

    def __init__(self, env=None, ds=None, vrt=None):
        self._env = env
        self._url = ds.name
        self._driver = ds.driver
        self._open_options = ds.options

        # Cache this for non-locking access
        self.scale_offset = (ds.scales[0], ds.offsets[0])

        if vrt is not None:
            self._vrt_params = dict(
                crs=vrt.crs.to_string(),
                resampling=vrt.resampling,
                tolerance=vrt.tolerance,
                src_nodata=vrt.src_nodata,
                nodata=vrt.nodata,
                width=vrt.width,
                height=vrt.height,
                src_transform=vrt.src_transform,
                transform=vrt.transform,
                dtype='int16'
                #dtype=vrt.working_dtype, # disable for arcgis
                #warp_extras=vrt.warp_extras, # disable for arcgis
            )
        else:
            self._vrt_params = None

        self._threadlocal = threading.local()
        self._threadlocal.ds = ds
        self._threadlocal.vrt = vrt
        self._lock = threading.Lock()

    def _open(self):
        open_env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True)
        with open_env:
            result = ds = SelfCleaningDatasetReader(
                rasterio.parse_path(self._url),
                sharing=False,
                driver=self._driver,
                **self._open_options)

            if self._vrt_params:
                with open_env:
                    result = vrt = WarpedVRT(ds, sharing=False, **self._vrt_params)
            else:
                vrt = None

        with self._lock:
            self._threadlocal.ds = ds
            self._threadlocal.vrt = vrt

        return result

    @property
    def dataset(self):
        try:
            with self._lock:
                return self._threadlocal.vrt or self._threadlocal.ds
        except AttributeError:
            return self._open()

    def read(self, window, **kwargs):
        "Read from the current thread's dataset, opening a new copy of the dataset on first access from each thread."
        read_env = rasterio.Env(VSI_CACHE=False)
        with read_env:
            return self.dataset.read(1, window=window, **kwargs)

    def close(self):
        with self._lock:
            self._threadlocal = threading.local()

    #def __getstate__(self):
        #raise RuntimeError("Don't pickle me bro!")

    #def __setstate__(self, state):
        #raise RuntimeError("Don't un-pickle me bro!")
        
    print('delete this!')


class SelfCleaningDatasetReader(rasterio.DatasetReader):
    def __del__(self):
        self.close()


class AutoParallelRioReader:

    def __init__(self, *, url, spec, resampling, dtype=None, fill_value=None, 
                 rescale=True, gdal_env=None, errors_as_nodata=()):
        self.url = url
        self.spec = spec
        self.resampling = resampling
        self.dtype = dtype
        self.rescale = rescale
        self.fill_value = fill_value
        self.gdal_env = gdal_env
        self.errors_as_nodata = errors_as_nodata

        self._dataset = None
        self._dataset_lock = threading.Lock()

    def _open(self):
        
        open_env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True)
        with open_env:
            try:
                ds = SelfCleaningDatasetReader(
                    rasterio.parse_path(self.url), sharing=False
                )
            except Exception as e:
                msg = f"Error opening {self.url!r}: {e!r}"
                print(msg)
                #if exception_matches(e, self.errors_as_nodata):
                    #warnings.warn(msg)
                    #return NodataReader(
                        #dtype=self.dtype, fill_value=self.fill_value
                    #)
                #raise RuntimeError(msg) from e
                
            if ds.count != 1:
                ds.close()
                raise RuntimeError(
                    f"Assets must have exactly 1 band, but file {self.url!r} has {ds.count}. "
                    "We can't currently handle multi-band rasters (each band has to be "
                    "a separate STAC asset), so you'll need to exclude this asset from your analysis.")

            # Only make a VRT if the dataset doesn't match the spatial spec we want
            if self.spec.get('vrt_params') != {
                "crs": ds.crs.to_epsg(),
                "transform": ds.transform,
                "height": ds.height,
                "width": ds.width}:
                
                with open_env:
                    vrt = WarpedVRT(
                        ds,
                        sharing=False,
                        resampling=self.resampling,
                        **self.spec.get('vrt_params'),
                    )

            else:
                print('skipping vrt')
                vrt = None

        if ds.driver in ['GTiff']:
            return ThreadLocalRioDataset(self.gdal_env, ds, vrt=vrt)
        else:
            #return SingleThreadedRioDataset(self.gdal_env, ds, vrt=vrt)
            print('temp')
            
    @property
    def dataset(self):
        with self._dataset_lock:
            if self._dataset is None:
                self._dataset = self._open()
            return self._dataset

    def read(self, window, **kwargs):
        reader = self.dataset
        try:
            result = reader.read(window=window, masked=True, **kwargs)
        except Exception as e:
            msg = f"Error reading {window} from {self.url!r}: {e!r}"
            print(msg)
            #if exception_matches(e, self.errors_as_nodata):
                #warnings.warn(msg)
                #return nodata_for_window(window, self.fill_value, self.dtype)
            #raise RuntimeError(msg) from e

        if self.rescale:
            scale, offset = reader.scale_offset
            if scale != 1 and offset != 0:
                result *= scale
                result += offset

        result = result.astype(self.dtype, copy=False)
        result = np.ma.filled(result, fill_value=self.fill_value)
        # ^ NOTE: if `self.fill_value` was None, rasterio set the masked array's fill value to the
        # nodata value of the band, which `np.ma.filled` will then use.
        return result

    def close(self) -> None:
        with self._dataset_lock:
            if self._dataset is None:
                return
            self._dataset.close()
            self._dataset = None

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            # AttributeError: 'AutoParallelRioReader' object has no attribute '_dataset_lock'
            # can happen when running multithreaded. I think this somehow occurs when `__del__`
            # happens before `__init__` has even run? Is that possible?
            pass

    #def __getstate__(
        #self,
    #) -> PickleState:
        #return {
            #"url": self.url,
            #"spec": self.spec,
            #"resampling": self.resampling,
            #"dtype": self.dtype,
            #"fill_value": self.fill_value,
            #"rescale": self.rescale,
            #"gdal_env": self.gdal_env,
            #"errors_as_nodata": self.errors_as_nodata,
        #}

    #def __setstate__(
        #self,
        #state: PickleState,
    #):
        #self.__init__(**state)
        # NOTE: typechecking may not catch errors here https://github.com/microsoft/pylance-release/issues/374
        
    print('remove this')


# V these are from to_dask # # # # #
def asset_entry_to_reader_and_window(asset_entry, spec, resampling, dtype=None, 
                                     fill_value=None, rescale=True, gdal_env=None,
                                     errors_as_nodata=(), reader=None):
    
    # to_array adds extra element to this, so subset
    asset_entry = asset_entry[0, 0]

    url = asset_entry['url']
    if url is None:
        return None

    asset_bounds = asset_entry['bounds']
    asset_window = windows.from_bounds(*asset_bounds, transform=spec.get('transform'))

    # Optional[Tuple[ReaderT, windows.Window]]
    return (AutoParallelRioReader(url=url,
                                  spec=spec,
                                  resampling=resampling,
                                  dtype=dtype,
                                  fill_value=fill_value,
                                  rescale=rescale,
                                  gdal_env=gdal_env,
                                  errors_as_nodata=errors_as_nodata), asset_window)


def fetch_raster_window(asset_entry, slices):
    current_window = windows.Window.from_slices(*slices)
    
    if asset_entry is not None:
        reader, asset_window = asset_entry

        # check that the window we're fetching overlaps with the asset
        if windows.intersect(current_window, asset_window):
            # backend: Backend = manager.acquire(needs_lock=False)
            data = reader.read(current_window)
            return data[None, None]

    # no dataset, or we didn't overlap it: return empty data.
    # use the broadcast trick for even fewer memz
    return np.broadcast_to(np.nan, (1, 1) + windows.shape(current_window))



def items_to_dask(meta, asset_table, chunksize=512, resampling='nearest',
                  dtype=np.dtype('int16'), fill_value=-999, rescale=True, reader=None,
                  gdal_env=None, errors_as_nodata=()):
    
    """
    Data is of type dictionary with everything that comes out
    of prepare_data.
    """
       
    # imports 
    from rasterio.enums import Resampling
    if resampling == 'nearest':
        resampling = Resampling.nearest
    elif resampling == 'bilinear':
        resampling = Resampling.bilinear
    else:
        raise ValueError('Resampling method not supported.')
        
    # TEMP
    spec = meta
    
    
    if fill_value is None and errors_as_nodata:
        raise ValueError('cant do')
        
    errors_as_nodata = errors_as_nodata or ()  # be sure it's not None

    if fill_value is not None and not np.can_cast(fill_value, dtype):
        raise ValueError('cant do')
        
    # make urls into dask array with 1-element chunks (i.e. 1 chunk per asset (i.e.e band))
    asset_table_dask = da.from_array(asset_table, 
                                     chunks=1, 
                                     #inline_array=True, doesnt work on this version
                                     name='asset-table-' + dask.base.tokenize(asset_table)
                                    )
    
    # map to blocks
    ds = asset_table_dask.map_blocks(asset_entry_to_reader_and_window,
                                     spec,
                                     resampling,
                                     dtype,
                                     fill_value,
                                     rescale,
                                     gdal_env,
                                     errors_as_nodata,
                                     reader,
                                     meta=asset_table_dask._meta
                                    )
    
    # generate fake array via shape anf chunksize
    shape = spec.get('shape')
    name = "slices-" + dask.base.tokenize(chunksize, shape)
    chunks = da.core.normalize_chunks(chunksize, shape)
    keys = itertools.product([name], *(range(len(bds)) for bds in chunks))
    slices = da.core.slices_from_chunks(chunks)
    
    # make slices of fake array for memory things, see git
    slices_fake_arr = da.Array(
            dict(zip(keys, slices)), name, chunks, meta=ds._meta
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=da.core.PerformanceWarning)

        rasters = da.blockwise(
            fetch_raster_window,
            "tbyx",
            ds,
            "tb",
            slices_fake_arr,
            "yx",
            meta=np.ndarray((), dtype=dtype),
        )

    return rasters


# V these are from accumulate_metadata # # # # #
class _ourlist(list):
    pass


def metadata_to_coords(items, dim_name, fields=True, skip_fields=(),
                       only_allsame=False):
    return dict_to_coords(
        accumulate_metadata(
            items, fields=fields, skip_fields=skip_fields, only_allsame=only_allsame), dim_name
    )


def accumulate_metadata(items, fields=True, skip_fields=(),
                        only_allsame=False):
    """
    Accumulate a sequence of multiple similar dicts into a single dict of lists.
    Each field will contain a list of all the values for that field (equal length to ``items``).
    For items where the field didn't exist, None is used.
    Fields with only one unique value are flattened down to just that single value.
    Parameters
    ----------
    items:
        Iterable of dicts to accumulate
    fields:
        Only use these fields. If True, use all fields.
    skip_fields:
        Skip these fields when ``fields`` is True.
    only_allsame:
        Only return fields that have the same value in every item.
        If ``"ignore-missing"``, ignores this check on items that were missing that field.
    """
    if isinstance(fields, str):
        fields = (fields,)

    all_fields = {}
    i = 0
    for i, item in enumerate(items):
        for existing_field in all_fields.keys():
            value = item.get(existing_field, None)
            if value is None and only_allsame == "ignore-missing":
                continue
            existing_value = all_fields[existing_field]
            if existing_value == value:
                continue

            if isinstance(existing_value, _ourlist):
                existing_value.append(value)
            else:
                if only_allsame:
                    all_fields[existing_field] = None
                else:
                    all_fields[existing_field] = _ourlist(
                        [None] * (i - 1) + [existing_value, value]
                    )

        if fields is True:
            for new_field in item.keys() - all_fields.keys():
                if new_field in skip_fields:
                    continue
                all_fields[new_field] = item[new_field]
        else:
            for field in list(fields): # changes this a bit, incase it breaks
                if field not in all_fields.keys():
                    try:
                        all_fields[field] = item[field]
                    except KeyError:
                        pass

    if only_allsame:
        return {
            field: value for field, value in all_fields.items() if value is not None
        }

    return all_fields


def dict_to_coords(metadata, dim_name):
    """
    Convert the output of `accumulate_metadata` into a dict of xarray Variables.
    1-length lists and scalar values become 0D variables.
    Instances of ``_ourlist`` become 1D variables for ``dim_name``.
    Any other things with >= 1 dimension are dropped, because they probably don't line up
    with the other dimensions of the final array.
    """
    coords = {}
    for field, props in metadata.items():
        while isinstance(props, list) and not isinstance(props, _ourlist):
            # a list scalar (like `instruments = ['OLI', 'TIRS']`).

            # first, unpack (arbitrarily-nested) 1-element lists.
            # keep re-checking if it's still a list
            if len(props) == 1:
                props = props[0]
                continue

            # for now, treat multi-item lists as a set so xarray can interpret them as 0D variables.
            # (numpy very much does not like to create object arrays containing python lists;
            # `set` is basically a hack to make a 0D ndarray containing a Python object with multiple items.)
            try:
                props = set(props)
            except TypeError:
                # if it's not set-able, just give up
                break

        props_arr = np.squeeze(np.array(props))
        if (
            props_arr.ndim > 1
            or props_arr.ndim == 1
            and not isinstance(props, _ourlist)
        ):
            # probably a list-of-lists situation. the other dims likely don't correspond to
            # our "bands", "y", and "x" dimensions, and xarray won't let us use unrelated
            # dimensions. so just skip it for now.
            continue

        coords[field] = xr.Variable(
            (dim_name,) if props_arr.ndim == 1 else (),
            props_arr,
        )

    return coords


# V these are from prepare
def to_attrs(spec):
    attrs = {"spec": spec, "crs": f"epsg:{spec.get('epsg')}", "transform": spec.get('transform')}

    resolutions = spec.get('resolutions_xy')
    if resolutions[0] == resolutions[1]:
        attrs['resolution'] = resolutions[0]
    else:
        attrs['resolution_xy'] = resolutions
    return attrs


# careful - do i want topleft or center?
def to_coords(items, asset_ids, spec, xy_coords='topleft', 
              properties=True, band_coords=True):

    # parse datetime
    times = pd.to_datetime([item['properties']['datetime'] for item in items],
                           infer_datetime_format=True,
                           errors='coerce')
    
    # remove timezone, xr cant handle it
    if times.tz is not None:
        times = times.tz_convert(None)

    # prep dims, coords
    dims = ['time', 'band', 'y', 'x']
    coords = {
        'time': times,
        'id': xr.Variable('time', [item['id'] for item in items]),
        'band': asset_ids,
    }

    if xy_coords is not False:
        if xy_coords == "center":
            pixel_center = True
        elif xy_coords == "topleft":
            pixel_center = False
        else:
            raise ValueError('xy_coords not supported.')

        transform = spec.get('transform')
        if transform.is_rectilinear:
            
            # faster-path for rectilinear transforms: just arange it
            minx, miny, maxx, maxy = spec.get('bounds')
            xres, yres = spec.get('resolutions_xy')

            if pixel_center:
                half_xpixel, half_ypixel = xres / 2, yres / 2
                minx, miny, maxx, maxy = (
                    minx + half_xpixel,
                    miny + half_ypixel,
                    maxx + half_xpixel,
                    maxy + half_ypixel,
                )

            height, width = spec.get('shape')
            xs = pd.Float64Index(np.linspace(minx, maxx, width, endpoint=False))
            ys = pd.Float64Index(np.linspace(maxy, miny, height, endpoint=False))
            
        else:
            height, width = spec.get('shape')
            if pixel_center:
                xs, _ = transform * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
                _, ys = transform * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
            else:
                xs, _ = transform * (np.arange(width), np.zeros(width))
                _, ys = transform * (np.zeros(height), np.arange(height))

        coords["x"] = xs
        coords["y"] = ys

    if properties:
        coords.update(
            metadata_to_coords(
                (item['properties'] for item in items),
                'time',
                fields=properties,
                skip_fields={'datetime'},
                # skip_fields={"datetime", "providers"},
            )
        )

    if band_coords:
        flattened_metadata_by_asset = [
            accumulate_metadata(
                (item['assets'].get(asset_id, {}) for item in items),
                skip_fields={'href', 'type', 'roles'},
                only_allsame='ignore-missing',
                # ^ NOTE: we `ignore-missing` because I've observed some STAC collections
                # missing `eo:bands` on some items.
                # xref https://github.com/sat-utils/sat-api/issues/229
            )
            for asset_id in asset_ids
        ]

        eo_by_asset = []
        for meta in flattened_metadata_by_asset:
            # NOTE: we look for `eo:bands` in each Asset's metadata, not as an Item-level list.
            # This only became available in STAC 1.0.0-beta.1, so we'll fail on older collections.
            # See https://github.com/radiantearth/stac-spec/tree/master/extensions/eo#item-fields
            eo = meta.pop("eo:bands", {})
            if isinstance(eo, list):
                eo = eo[0] if len(eo) == 1 else {}
                # ^ `eo:bands` should be a list when present, but >1 item means it's probably a multi-band asset,
                # which we can't currently handle, so we ignore it. we don't error here, because
                # as long as you don't actually _use_ that asset, everything will be fine. we could
                # warn, but that would probably just get annoying.
            eo_by_asset.append(eo)
            try:
                meta['polarization'] = meta.pop('sar:polarizations')
            except KeyError:
                pass

        coords.update(
            metadata_to_coords(
                flattened_metadata_by_asset,
                'band',
                skip_fields={'href'},
                # skip_fields={"href", "title", "description", "type", "roles"},
            )
        )
        if any(d for d in eo_by_asset):
            coords.update(
                metadata_to_coords(
                    eo_by_asset,
                    'band',
                    fields=['common_name', 'center_wavelength', 'full_width_half_max'],
                )
            )

    # Add `epsg` last in case it's also a field in properties; our data model assumes it's a coordinate
    coords["epsg"] = spec.get('epsg')

    return coords, dims