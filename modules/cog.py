# working

import rasterio
import dask
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import threading
import itertools
import warnings

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
                dtype=vrt.working_dtype, # disable for arcgis
                warp_extras=vrt.warp_extras, # disable for arcgis
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



def items_to_dask(asset_table, spec, chunksize=512, resampling=Resampling.nearest,
                  dtype=np.dtype('int16'), fill_value=-999, rescale=True, reader=None,
                  gdal_env=None, errors_as_nodata=()):
    
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