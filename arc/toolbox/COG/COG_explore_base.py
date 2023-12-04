import xarray as xr
import numpy as np

from modules import cog
from shared import tools, satfetcher, arc
from arc.toolbox.common_functions.common_functions import check_xarray

class CogExploreBase:
    def __init__(self, in_path=None, in_veg_idx=None, in_fmask_flags=('Valid', 'Snow', 'Water'), in_max_cloud=None,
                 out_folder=None):

        if in_fmask_flags is None or len(in_fmask_flags) == 0:
            in_fmask_flags = ('Valid', 'Snow', 'Water')

        # lazy load
        self.inpath = in_path
        self.ds = xr.open_dataset(self.inpath)
        self.ds_loaded = None
        self.mask_band = None
        self.messages = []
        self.check_netcdf_dataset()
        self.veg_idx = in_veg_idx
        self.fmask_flags = in_fmask_flags
        self.fmask_flags_ids = arc.convert_fmask_codes(in_fmask_flags)
        self.max_cloud = in_max_cloud
        self.out_folder = out_folder
        self.attrs = None
        self.band_attrs = None
        self.spatial_ref_attrs = None

    def check_netcdf_dataset(self):
        self.messages.append(check_xarray(self.ds, levels={0,1}))

    def full_load(self):
        self.ds_loaded = satfetcher.load_local_nc(nc_path=self.inpath, use_dask=True, conform_nodata_to=np.nan)

        self.attrs = self.ds_loaded.attrs
        self.band_attrs = self.ds_loaded[list(self.ds_loaded)[0]].attrs
        self.spatial_ref_attrs = self.ds_loaded['spatial_ref'].attrs
        self.mask_band = arc.get_name_of_mask_band(list(self.ds_loaded))
        self.remove_duplicates()

    def remove_duplicates(self):
        self.ds_loaded = satfetcher.group_by_solar_day(self.ds_loaded)

    def remove_pixels_empty_scenes(self):
        self.ds_loaded = cog.remove_fmask_dates(ds=self.ds_loaded,
                               valid_class=self.fmask_flags_ids,
                               max_invalid=self.max_cloud,
                               mask_band=self.mask_band,
                               nodata_value=np.nan,
                               drop_fmask=True)

    def conform_band_names(self):

        in_platform = arc.get_platform_from_dea_attrs(self.attrs)

        # conform dea aws band names based on platform
        self.ds_loaded = satfetcher.conform_dea_ard_band_names(ds=self.ds_loaded,
                                                   platform=in_platform.lower())


    def check_band_names(self):
        errs = []
        for band in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
            if band not in self.ds_loaded:
                errs.append('NetCDF is missing band: {}. Need all bands.'.format(band))
        return errs

    def check_veg_idx(self):
        errs = []
        if self.veg_idx.lower() not in {'ndvi', 'evi', 'savi', 'msavi', 'slavi', 'mavi', 'kndvi'}:
            errs.append('Vegetation index not supported.')

        return errs

    def calc_veg_idx(self):
        self.ds_loaded = tools.calculate_indices(ds=self.ds_loaded,
                                     index=self.veg_idx.lower(),
                                     custom_name='veg_idx',
                                     rescale=False,
                                     drop=True)

        # add band attrs back on
        self.ds_loaded['veg_idx'].attrs = self.band_attrs

    def check_enough_temporal_data(self):
        errs = []
        # check if we have sufficient data temporaly
        if len(self.ds_loaded['time']) == 0:
            errs.append('No dates remaining in data.')
        return errs

    def compute_loaded_ds(self):
        self.ds_loaded = self.ds_loaded.compute()

    def check_null(self):
        errs = []
        if self.ds_loaded.to_array().isnull().all():
            errs.append('NetCDF is empty. Please download again.')
        return errs

    def interpolate(self, method='full'):
        self.ds_loaded = tools.perform_interp(ds=self.ds_loaded, method=method)

    def reattach_attributes(self):
        self.ds_loaded.attrs = self.attrs
        self.ds_loaded['spatial_ref'].attrs = self.spatial_ref_attrs
        for var in self.ds_loaded:
            self.ds_loaded[var].attrs = self.band_attrs
