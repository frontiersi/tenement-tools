import os

import xarray as xr
import numpy as np

from arc.toolbox.globals import STAC_ENDPOINT_ODC, RESULT_LIMIT
from modules import cog, cog_odc
from shared import tools, satfetcher, arc
from arc.toolbox.common_functions.common_functions import check_xarray


class CogFetchODCBase:
    def __init__(self, bbox, rio_settings=None, stac_endpoint=STAC_ENDPOINT_ODC, start_dt=None,
                 end_dt=None, slc_off=None, result_limit=RESULT_LIMIT, bands=None, res=None, 
                 align=None, resampling=None, platform=None):
        self.res = res
        self.ds = None
        self.set_rio_env(rio_settings)
        self.bbox = bbox
        self.stac_endpoint = stac_endpoint
        self.collections = None
        self.prepare_collections(platform)
        self.platform = platform
        self.bands = None
        self.prepare_bands(bands, platform)
        self.start_date = start_dt.strftime('%Y-%m-%d')
        self.end_date = end_dt.strftime('%Y-%m-%d')
        self.slc_off = slc_off
        self.result_limit = result_limit
        self.items = []
        self.resampling = resampling
        self.align = align

    def test_values(self):
        messages = []
        if self.res < 1:
            messages.append('Resolution value must be > 0.')
        if self.resampling not in ['Nearest', 'Bilinear', 'Cubic', 'Average']:
            messages.append('Resampling method not supported.')

        if self.align is not None and (self.align < 0 or self.align > self.res):
            messages.append('Alignment must be > 0 but < resolution.')

        if self.start_date >= self.end_date:
            messages.append('End date must be greater than start date.')

        return messages

    def prepare_collections(self, in_platform):
        self.collections = arc.prepare_collections_list(in_platform)

    def prepare_bands(self, in_bands=None, in_platform=None):
        if in_bands is None:
            in_bands = self.bands
        if in_platform is None:
            in_platform = self.platform
        self.bands = arc.prepare_band_names(in_bands, in_platform)

    def set_rio_env(self, settings=None):
        if not isinstance(settings, dict):
            settings = {
                            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
                            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
                            'VSI_CACHE': 'TRUE',
                            'GDAL_HTTP_MULTIRANGE': 'YES',
                            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
                        }
        for k, v in settings.items():
            os.environ[k] = v

    def test_bbox(self, bbox=None):
        messages = []
        if bbox is None:
            bbox = self.bbox

        if bbox is not None:
            if len(bbox) != 4:
                messages.append('Bounding box is invalid.')

        return messages

    def fetch_items(self, replace_s3_https=True):
        # fetch stac items
        self.items += cog_odc.fetch_stac_items_odc(stac_endpoint=self.stac_endpoint,
                                             collections=self.collections,
                                             start_dt=self.start_date,
                                             end_dt=self.end_date,
                                             bbox=self.bbox,
                                             slc_off=self.slc_off,
                                             limit=self.result_limit)

        # replace s3 prefix with https (pro-friendly)
        if replace_s3_https is True:
            self.items = cog_odc.replace_items_s3_to_https(items=self.items,
                                                  from_prefix='s3://dea-public-data',
                                                  to_prefix='https://data.dea.ga.gov.au')
        return self.items

    def clear_existing_items(self):
        self.items = []
        
    def build_dataset_from_items(self):
        self.ds = cog_odc.build_xr_odc(items=self.items,
                                      bbox=self.bbox,
                                      bands=self.bands,
                                      crs=3577,
                                      res=self.res,
                                      resampling=self.resampling,
                                      align=self.align,
                                      group_by='solar_day',
                                      chunks={},
                                      like=None)

        for idx, var in enumerate(self.ds):
            # increment progress ba
            # compute!
            self.ds[var] = self.ds[var].compute()

        dts = self.ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        self.ds['time'] = dts.astype('datetime64[ns]')

        # set to signed int16
        self.ds = self.ds.astype('int16')

        # set all non-mask bands nodata values (0) to -999
        for var in self.ds:
            if 'mask' not in var:
                self.ds[var] = self.ds[var].where(self.ds[var] != 0, -999)

    def assign_attributes(self):
        attrs = {
            'transform': tuple(self.ds.geobox.transform),
            'res': self.res,
            'nodatavals': -999,
            'orig_bbox': tuple(self.bbox),
            'orig_collections': tuple(self.collections),
            'orig_bands': tuple(self.bands),
            'orig_dtype': 'int16',
            'orig_slc_off': str(self.slc_off),
            'orig_resample': self.resampling
        }

        # assign attrs
        self.ds = self.ds.assign_attrs(attrs)

    def export_dataset(self, out_nc):
        tools.export_xr_as_nc(ds=self.ds, filename=out_nc)

    def close_and_clear_dataset(self):
        self.ds.close()
        self.ds = None






