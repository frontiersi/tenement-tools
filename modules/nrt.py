# nrt
'''
This script contains change detection algorithm
and functions to detect and clean change data.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os
import sys
import time
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import arcpy

from scipy.signal import savgol_filter
from osgeo import gdal
from osgeo import ogr

sys.path.append('../../modules')
import cog_odc, cog

sys.path.append('../../shared')
import arc, satfetcher, tools


class MonitoringAreaStatistics:

    def __init__(self, feat, path, out_nc):
        
        # feature fields
        self.area_id = feat[0]
        self.platform = feat[1]
        self.s_year = feat[2]
        self.e_year = feat[3]
        self.index = feat[4]
        self.persistence = feat[5]
        self.rule_1_min_conseqs = feat[6]
        self.rule_1_inc_plateaus = feat[7]
        self.rule_2_min_zone = feat[8]
        self.rule_3_num_zones = feat[9]
        self.ruleset = feat[10]
        self.alert = feat[11]
        self.method = feat[12]
        self.alert_direction = feat[13]
        self.email = feat[14]
        self.ignore = feat[15]
        self.color = feat[16]
        self.global_id = feat[17]
        
        # feature geometry
        self.raw_geom = feat[18]
        self.prj_geom = None
        
        # path to project folder, output file
        self.path = path
        self.out_nc = out_nc
        
        # xr datasets
        self.ds = None
        self.mask = None


    def validate_area(self):
        """
        Checks all required monitoring area parameters are 
        valid. Raises an error if invalid.
        """

        # check area id
        if self.area_id is None:
            raise ValueError('No area id exists.')

        # check platform
        if self.platform not in ['Landsat', 'Sentinel']:
            raise ValueError('Platform not Landsat or Sentinel.')

        # check start, end years
        if not isinstance(self.s_year, int):
            raise ValueError('Start year not an integer.')
        elif not isinstance(self.e_year, int):
            raise ValueError('End year not an integer.')
        elif self.s_year < 1980 or self.s_year > 2050:
            raise ValueError('Start year not between 1980-2050.')        
        elif self.e_year < 1980 or self.e_year > 2050:
            raise ValueError('End year not between 1980-2050.')          
        elif self.e_year <= self.s_year:
            raise ValueError('End year is <= start year.')           
        elif abs(self.e_year - self.s_year) < 2:
            raise ValueError('Training period < 2 years in length.')  
        elif self.platform == 'Sentinel' and self.s_year < 2016:
            raise ValueError('Start year must be >= 2016 for Sentinel data.')  

        # check index
        if self.index not in ['NDVI', 'MAVI', 'kNDVI']:
            raise ValueError('Index must be NDVI, MAVI or kNDVI.')

        # check persistence
        if self.persistence is None:
            raise ValueError('No persistence exists.')
        elif self.persistence < 0.001 or self.persistence > 9.999:
            raise ValueError('Persistence not between 0.0001 and 9.999.')

        # check rule 1 min consequtives
        if self.rule_1_min_conseqs is None:
            raise ValueError('No rule 1 min conseqs exists.')
        elif self.rule_1_min_conseqs < 0 or self.rule_1_min_conseqs > 999:
            raise ValueError('Rule 1 min conseqs not between 0 and 999.')

        # check rule 1 inc plateaus
        if self.rule_1_inc_plateaus is None:
            raise ValueError('No rule 1 inc plateaus exists.')
        elif self.rule_1_inc_plateaus not in ['Yes', 'No']:
            raise ValueError('Rule 1 inc plateaus must be Yes or No.')

        # check rule 2 min zone
        if self.rule_2_min_zone is None:
            raise ValueError('No rule 2 min zone exists.')
        elif self.rule_2_min_zone < 1 or self.rule_2_min_zone > 11:
            raise ValueError('Rule 2 min zone not between 1 and 11.') 

        # check rule 3 num zones
        if self.rule_3_num_zones is None:
            raise ValueError('No rule 3 num zones exists.')
        elif self.rule_3_num_zones < 1 or self.rule_3_num_zones > 11:
            raise ValueError('rule 3 num zones not between 1 and 11.')             

        # set up allowed rulesets
        rulesets = [
            '1 only',
            '2 only',
            '3 only',
            '1 and 2',
            '1 and 3',
            '2 and 3',
            '1 or 2',
            '1 or 3',
            '2 or 3',
            '1 and 2 and 3',
            '1 or 2 and 3',
            '1 and 2 or 3',
            '1 or 2 or 3'
        ]

        # check ruleset   
        if self.ruleset not in rulesets:
            raise ValueError('Rulset not supported.')

        # check alert
        if self.alert not in ['Yes', 'No']:
            raise ValueError('Alert must be Yes or No.')

        # check method
        if self.method not in ['Static', 'Dynamic']:
            raise ValueError('Method must be Static or Dynamic')

        # set up alert directions 
        alert_directions = [
            'Incline only (any)', 
            'Decline only (any)', 
            'Incline only (+ zones only)', 
            'Decline only (- zones only)', 
            'Incline or Decline (any)',
            'Incline or Decline (+/- zones only)'
        ]

        # check alert direction
        if self.alert_direction not in alert_directions:
            raise ValueError('Alert direction is not supported.')

        # check email address
        if self.alert == 'Yes' and self.email is None:
            raise ValueError('No email provided.')
        elif self.email is not None and '@' not in self.email:
            raise ValueError('Email address invalid.')

        # check ignore
        if self.ignore not in ['Yes', 'No']:
            raise ValueError('Ignore must be Yes or No.')

        # check global id
        if self.global_id is None:
            raise ValueError('No global id exists.')

        # check path provided
        if self.path is None:
            raise ValueError('No project path exists.')

        return
        
    # just ensure the privisonal data mask band correct
    def set_xr(self):
        """
        Fetches all available digital earth australia (dea) 
        landsat/sentinel data for area bounding box. Start date
        is based on provided start year. The resulting data is 
        set to the xr. If an error occurs, an error is raised. 
        """
        
        # set endpoint
        STAC_ENDPOINT = 'https://explorer.sandbox.dea.ga.gov.au/stac'
        
        # check platform is valid
        if self.platform not in ['Landsat', 'Sentinel']:
            raise ValueError('Platform not supported.')

        # prepare dea stac search parameters
        if self.platform == 'Landsat':
            
            # set dea collection names
            collections = [
                'ga_ls5t_ard_3',
                'ga_ls7e_ard_3',
                'ga_ls8c_ard_3',
                'ga_ls8c_ard_provisional_3'
            ]
            
            # set bands
            bands = [
                'nbart_red', 
                'nbart_green', 
                'nbart_blue', 
                'nbart_nir', 
                'nbart_swir_1', 
                'nbart_swir_2', 
                'oa_fmask'
            ]
            
        elif self.platform == 'Sentinel':
            
            # set dea collection names
            collections = [
                's2a_ard_granule',  # todo: use ver 3 when avail
                's2b_ard_granule',  # todo: use ver 3 when avail
                'ga_s2am_ard_provisional_3',
                'ga_s2bm_ard_provisional_3'
            ]
            
            # set bands
            bands = [
                'nbart_red', 
                'nbart_green', 
                'nbart_blue', 
                'nbart_nir_1', 
                'nbart_swir_2', 
                'nbart_swir_3', 
                'fmask'
            ]
        
        try:
            # ensure raw geom is in wgs84 (set prj_geom)
            srs = arcpy.SpatialReference(4326)
            self.prj_geom = self.raw_geom.projectAs(srs)
            
            # convert to bounding box in wgs84
            prj_bbox = [
                self.prj_geom.extent.XMin,
                self.prj_geom.extent.YMin,
                self.prj_geom.extent.XMax,
                self.prj_geom.extent.YMax
            ]
        
        except Exception as e:
            raise ValueError(e)
            
            
        # check if start date provided 
        if self.s_year is None:
            raise ValueError('No start year provided.')
            
        try:
            # prepare start date 
            start_dt = '{}-01-01'.format(self.s_year)
            
            # get all avail dea satellite data without compute
            self.ds = fetch_cube_data(collections=collections, 
                                      bands=bands, 
                                      start_dt=start_dt, 
                                      end_dt='2050-12-31', 
                                      bbox=prj_bbox, 
                                      resolution=10, 
                                      ds_existing=None)
        
            # group duplicate times if exist
            self.ds = satfetcher.group_by_solar_day(self.ds)
            
        except Exception as e:
            raise ValueError(e)
            
        try:
            # enforce none type if no dates
            if len(self.ds['time']) == 0:
                self.ds = None
                
        except Exception as e:
            raise ValueError(e)
            
        return
        
        
    def apply_xr_fmask(self):
        """
        Takes the xr and applies the dea fmask band to remove 
        invalid pixels and dates. If an error occurs, error is 
        raised.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        try:
            # get mask band name (either be oa_fmask or fmask)
            mask = [v for v in self.ds if 'mask' in v][0]

            # mask invalid pixels i.e., not valid, water, snow
            self.ds = cog.remove_fmask_dates(ds=self.ds, 
                                             valid_class=[1, 4, 5],
                                             max_invalid=0,
                                             mask_band=mask, 
                                             nodata_value=np.nan,
                                             drop_fmask=True)
        except Exception as e:
            raise ValueError(e)

        return
    
    
    def apply_xr_index(self):
        """
        Takes the  xr and applies user chosen vegetation
        index. If an error occurs, error is raised. 
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        # check if platform set
        if self.platform not in ['Landsat', 'Sentinel']:
            raise ValueError('Platform not supported.')

        # check if index set
        if self.index not in ['NDVI', 'MAVI', 'kNDVI']:
            raise ValueError('Index not supported.')

        try:
            # conform dea band names and calc vegetation index
            platform = self.platform.lower()
            self.ds = satfetcher.conform_dea_ard_band_names(ds=self.ds, 
                                                            platform=platform) 

            # calculate vegetation index
            index = self.index.lower()
            self.ds = tools.calculate_indices(ds=self.ds, 
                                              index=index, 
                                              custom_name='veg_idx', 
                                              rescale=False, 
                                              drop=True)
        except Exception as e:
            raise ValueError(e)

        return


    def load_xr(self):
        """
        Loads xr values into memory using the xarray 
        load function. This will result in downloading from 
        dea and can take awhile.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No new xr provided.')

        try:
            # load new xr and close connection 
            self.ds.load()
            self.ds.close()

        except Exception as e:
            raise ValueError(e)

        return


    def set_xr_edge_mask(self):
        """
        Sets an xr array useful for masking out edge pixels. Edge pixels 
        can occur within the original bounding box but outside of the 
        vector boundary of the monitoring area. This mask is created 
        as array of 1s and 0s (in boundary, out boundary) 
        and applies it to the xr via the apply_xr_edge_mask function. 
        If an error occurs, no mask is applied.
        """ 

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        # check if raw geometry exists
        if self.raw_geom is None:
            raise ValueError('No raw area geometry provided.')       

        try:
            # rasterize area polygon, set outside pixels to nan
            self.mask = rasterize_polygon(ds=self.ds, 
                                          geom=self.raw_geom)

        except Exception as e:
            self.mask = None
            raise ValueError(e)

        return


    def interp_xr_nans(self):
        """
        Interpolates any existing nan values in xr, linearly.
        If nan values still exist after interpolation (often on
        edge dates due to lack of extrapolation), these will be
        dropped. If error occurs, error is raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')          

        try:
            # interpolate na linearly
            self.ds = self.ds.interpolate_na('time')
            
            # fill any remaining nans
            self.ds = self.ds.fillna(0)
        
        except Exception as e:
            raise ValueError(e)
            
        # check we have data remaining
        if len(self.ds['time']) == 0:
            raise ValueError('No data remaining after nodata dropped.')
            
        return


    def append_xr_vars(self):
        """
        Appends required xr variables to xr if do not exist.
        These variables are required for storing outputs from 
        change detection results, cleaned vegetation, etc. If
        error, error is raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')  
        
        # set required variable names
        new_vars = [
            'veg_clean', 
            'static_raw', 
            'static_clean',
            'static_rule_one',
            'static_rule_two',
            'static_rule_three',
            'static_zones',
            'static_alerts',
            'dynamic_raw', 
            'dynamic_clean',
            'dynamic_rule_one',
            'dynamic_rule_two',
            'dynamic_rule_three',
            'dynamic_zones',
            'dynamic_alerts'
        ]        
        
        # iter var names and append to xr
        for var in new_vars:
            if var not in self.ds:
                da = xr.full_like(self.ds['veg_idx'], np.nan)
                self.ds[var] = da
        
        return
    

    def fix_xr_spikes(self):
        """
        Detects severe vegetation index outliers using the TIMESAT 
        3.3 median spike detection method. Sets spike values to 
        nan and then interpolates them, if they exist. If error, error raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')
        
        try:
            # remove outliers via median spike method
            da = remove_spikes(da=self.ds['veg_idx'], 
                               factor=1, 
                               win_size=3)
            
            # interpolate nans linearly
            da = da.interpolate_na('time')
            
            # fill nan values
            da = da.fillna(0)
            
            # set result to clean var
            self.ds['veg_clean'] = da   
        
        except Exception as e:
            raise ValueError(e)
            
        # check we still have data remaining
        if len(self.ds['time']) == 0:
            raise ValueError('No data remaining after no data dropped.')
            
        return
    
    
    def smooth_xr_index(self):
        """
        Mildly smoothes the xr clean vegetation index values 
        using the savitsky golay filter. If error, error raised.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        try:
            # get time dimension axis
            dims = list(self.ds.dims)
            for idx, dim in enumerate(dims):
                if dim == 'time':
                    axis = idx

            # set up kwargs
            kwargs={
                'window_length': 3, 
                'polyorder': 1,
                'a': axis
            }

            # apply savitsky filter and handle nans
            da = xr.apply_ufunc(safe_savgol, 
                                self.ds['veg_clean'],
                                dask='allowed',
                                kwargs=kwargs)
                        
            # update existing values in xr
            self.ds['veg_clean'] = da
        
        except Exception as e:
            raise ValueError(e)
            
        return

    # TODO play with smoothing, persistence 
    # ALSO TRY MIN TRAINING LENGTH COULD WORK!!! set to 1 year worth of dates, 2 years worth of dates, etc
    def detect_change_xr(self):
        """
        Performs ewmacd change detection (static and dynamic 
        types) on xr. Uses the raw vegetation index time series 
        to detect the change. If error or all nan, error raised.
        """

        # check if anl xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        # check start and end years valid
        if self.s_year is None or self.e_year is None:
            raise ValueError('No start and/or end year provided.')        

        # check if persistence is valid
        if self.persistence is None:
            raise ValueError('No persistence provided.')
            
        try:
            # perform ewmacd change detection
            self.ds = detect_change(ds=self.ds,
                                    method='both',
                                    var='veg_idx',
                                    train_start=self.s_year,
                                    train_end=self.e_year,
                                    persistence=self.persistence)
            
            # ensure static, dynamic dimension order correct
            self.ds['static_raw'] = self.ds['static_raw'].transpose('time', 'y', 'x')
            self.ds['dynamic_raw'] = self.ds['dynamic_raw'].transpose('time', 'y', 'x')
            
        except Exception as e:
            raise ValueError(e)

        # check if we have data
        for var in ['static_raw', 'dynamic_raw']:
            if self.ds[var].isnull().all():
                raise ValueError('Change result is empty.')            

        return

    
    # remove commented out code if happy with method
    def smooth_xr_change(self):
        """
        Mildly smoothes the xr static and dynamic change 
        signal values using the savitsky golay filter. If error, 
        error raised.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')
            
        try:
            # get time dimension axis
            dims = list(self.ds.dims)
            for idx, dim in enumerate(dims):
                if dim == 'time':
                    axis = idx

            # set up kwargs
            kwargs={
                'window_length': 3, 
                'polyorder': 1,
                'a': axis
            }
            
            # apply static savitsky filter and handle nans
            da_static = xr.apply_ufunc(safe_savgol, 
                                       self.ds['static_raw'],
                                       dask='allowed',
                                       kwargs=kwargs)
            
            # apply dynamic savitsky filter and handle nans
            da_dynamic = xr.apply_ufunc(safe_savgol, 
                                        self.ds['dynamic_raw'],
                                        dask='allowed',
                                        kwargs=kwargs)
            
            # update static, dynamic clean values in xr
            self.ds['static_clean'] = da_static
            self.ds['dynamic_clean'] = da_dynamic
            
        except Exception as e:
            raise ValueError(e)

        return       


    def build_zones(self):
        """
        Takes cleaned static and dynamic change deviation
        values and classifies them into 1 of 11 zones based
        on where the change value falls. Honours direction 
        of change by returning zone value with sign (+/-).
        If error occurs, error raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        # check if required vars in xr
        if 'static_clean' not in self.ds:
            raise ValueError('No clean static variable.')
        elif 'dynamic_clean' not in self.ds:
            raise ValueError('No clean dynamic variable.')

        try:
            # seperate dims
            t, y, x = self.ds['time'], self.ds['y'], self.ds['x']

            # get rows of static values per pixel and apply func along rows
            da = self.ds['static_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_zones, axis=0, arr=da)

            # rebuild static xr array 
            da_static = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                     coords={'time': t, 'y': y, 'x': x}, 
                                     dims=['time', 'y', 'x'])

            # get rows of dynamic values per pixel and apply func along rows
            da = self.ds['dynamic_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_zones, axis=0, arr=da)

            # rebuild dynamic xr array 
            da_dynamic = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                      coords={'time': t, 'y': y, 'x': x}, 
                                      dims=['time', 'y', 'x'])

            # update xr dataset
            self.ds['static_zones'] = da_static
            self.ds['dynamic_zones'] = da_dynamic

        except Exception as e:
            raise ValueError(e)

        # check if we have any data
        for var in ['static_zones', 'dynamic_zones']:
            if self.ds[var].isnull().all():
                raise ValueError('Zone result is empty.')

        return
    
    
    def build_rule_one(self):
        """
        Takes cleaned static and dynamic change deviation
        values and applies rule one rules to them. Rule one
        calculates consequtive runs of values across time.
        Honours direction of change by returning value 
        with sign (+/-). If error occurs, error raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No anl xr provided.')
        
        # check if required vars in xr
        if 'static_clean' not in self.ds:
            raise ValueError('No clean static variable.')
        elif 'dynamic_clean' not in self.ds:
            raise ValueError('No clean dynamic variable.')
        
        # check if rule one parameters valid
        if self.rule_1_min_conseqs is None:
            raise ValueError('No minimum consequtives provided.')
        elif self.rule_1_inc_plateaus is None:
            raise ValueError('No include plateaus provided.')   
            
        # prepare plateaus
        if self.rule_1_inc_plateaus == 'Yes':
            plateaus = True
        else:
            plateaus = False
            
        # set kwarg options
        kwargs = {
            'min_conseqs': self.rule_1_min_conseqs,
            'inc_plateaus': plateaus
        }

        try:
            # seperate dims
            t, y, x = self.ds['time'], self.ds['y'], self.ds['x']

            # get rows of static values per pixel and apply rule 1 runs (+/-) along rows
            da = self.ds['static_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_one_runs, axis=0, arr=da, **kwargs)

            # rebuild static xr array 
            da_static = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                     coords={'time': t, 'y': y, 'x': x}, 
                                     dims=['time', 'y', 'x'])

            # get rows of dynamic values per pixel and apply rule 1 runs (+/-) along rows
            da = self.ds['dynamic_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_one_runs, axis=0, arr=da, **kwargs)

            # rebuild dynamic xr array 
            da_dynamic = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                      coords={'time': t, 'y': y, 'x': x}, 
                                      dims=['time', 'y', 'x'])

            # update xr dataset
            self.ds['static_rule_one'] = da_static
            self.ds['dynamic_rule_one'] = da_dynamic

        except Exception as e:
            raise ValueError(e)

        # check if we have any data
        for var in ['static_rule_one', 'dynamic_rule_one']:
            if self.ds[var].isnull().all():
                raise ValueError('Rule one result empty.')

        return
    
    
    def build_rule_two(self):
        """
        Takes cleaned static and dynamic change deviation
        values and applies rule two rules to them. Rule two
        masks out stdv values that fall within a specified
        minimum zone threshold. Honours direction of change 
        by returning value with sign (+/-). If error occurs, 
        error raised.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No anl xr provided.')

        # check if required vars in xr
        if 'static_clean' not in self.ds:
            raise ValueError('No clean static variable.')
        elif 'dynamic_clean' not in self.ds:
            raise ValueError('No clean dynamic variable.')

        # check if rule two parameters valid
        if self.rule_2_min_zone is None:
            raise ValueError('No minimum zone provided.') 

        # convert zone to std
        stdvs = zone_to_std(self.rule_2_min_zone)[0]

        # set kwarg options
        kwargs = {'min_stdv': stdvs}

        try:
            # seperate dims
            t, y, x = self.ds['time'], self.ds['y'], self.ds['x']

            # get rows of static values per pixel and apply rule 2 mask (+/-) along rows
            da = self.ds['static_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_two_mask, axis=0, arr=da, **kwargs)

            # rebuild static xr array 
            da_static = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                     coords={'time': t, 'y': y, 'x': x}, 
                                     dims=['time', 'y', 'x'])

            # get rows of dynamic values per pixel and apply rule 2 mask (+/-) along rows
            da = self.ds['dynamic_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_two_mask, axis=0, arr=da, **kwargs)

            # rebuild dynamic xr array 
            da_dynamic = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                      coords={'time': t, 'y': y, 'x': x}, 
                                      dims=['time', 'y', 'x'])

            # update xr dataset
            self.ds['static_rule_two'] = da_static
            self.ds['dynamic_rule_two'] = da_dynamic

        except Exception as e:
            raise ValueError(e)

        # check if we have any data
        for var in ['static_rule_two', 'dynamic_rule_two']:
            if self.ds[var].isnull().all():
                raise ValueError('Rule two result empty.')

        return


    def build_rule_three(self):
        """
        Takes cleaned static and dynamic change deviation
        values and applies rule three rules to them. Rule three
        detects sharp zone value spikes that occurr between 
        dates. Honours direction of change by returning value 
        with sign (+/-). If error occurs, error raised.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr provided.')

        # check if required vars in xr
        if 'static_clean' not in self.ds:
            raise ValueError('No clean static variable.')
        elif 'dynamic_clean' not in self.ds:
            raise ValueError('No clean dynamic variable.')

        # check if rule three parameters valid
        if self.rule_3_num_zones is None:
            raise ValueError('No number of zones provided.') 

        # convert zone to std and multiple by 2 (2 std per zone)
        stdvs = zone_to_std(self.rule_3_num_zones)[0] 
        stdvs = stdvs * 2

        # set kwarg options
        kwargs = {'min_stdv': stdvs}

        try:
            # seperate dims
            t, y, x = self.ds['time'], self.ds['y'], self.ds['x']

            # get rows of static values per pixel and apply rule 3 spikes (+/-) along rows
            da = self.ds['static_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_three_spikes, axis=0, arr=da, **kwargs)

            # rebuild static xr array 
            da_static = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                     coords={'time': t, 'y': y, 'x': x}, 
                                     dims=['time', 'y', 'x'])

            # get rows of dynamic values per pixel and apply rule 3 spikes (+/-) along rows
            da = self.ds['dynamic_clean'].stack(z=['y', 'x']).values
            da = np.apply_along_axis(build_rule_three_spikes, axis=0, arr=da, **kwargs)

            # rebuild dynamic xr array 
            da_dynamic = xr.DataArray(da.reshape(len(t), len(y), len(x)), 
                                      coords={'time': t, 'y': y, 'x': x}, 
                                      dims=['time', 'y', 'x'])

            # update xr dataset
            self.ds['static_rule_three'] = da_static
            self.ds['dynamic_rule_three'] = da_dynamic

        except Exception as e:
            raise ValueError(e)

        # check if we have any data
        for var in ['static_rule_three', 'dynamic_rule_three']:
            if self.ds[var].isnull().all():
                raise ValueError('Rule three result empty.')

        return
    
    
    def build_alerts(self):
        """
        Takes the previously derived rule one, two, three values
        and combines them into an alert mask (1, 0) variable for 
        static and dynamic methods. This method considers 
        both the user's requested ruleset and the particular
        direction of change (incline or decline) required for
        alert to be set as true (1). If error occurs, error raised.
        """
        
        # check if xr exists
        if self.ds is None:
            raise ValueError('No anl xr provided.')
        
        # set up valid rulesets
        valid_rules = [
            '1 only', 
            '2 only', 
            '3 only', 
            '1 and 2', 
            '1 and 3', 
            '2 and 3', 
            '1 or 2', 
            '1 or 3', 
            '2 or 3', 
            '1 and 2 and 3', 
            '1 or 2 and 3',
            '1 and 2 or 3', 
            '1 or 2 or 3'
        ]
        
        # check if ruleset valid
        if self.ruleset not in valid_rules:
            raise ValueError('Ruleset not supported.')
        
        # set up valid directions 
        valid_directions = [
            'Incline only (any)',
            'Decline only (any)',
            'Incline only (+ zones only)',
            'Decline only (- zones only)',
            'Incline or Decline (any)',
            'Incline or Decline (+/- zones only)'
        ]
        
        # check if direction valid
        if self.ruleset not in valid_rules:
            raise ValueError('Direction not supported.')
        
        # set kwarg options
        kwargs = {
            'ruleset': self.ruleset,
            'direction': self.alert_direction
        }

        try:
            # generate and combines rules into alert for static change 
            self.ds['static_alerts'] = xr.apply_ufunc(build_alerts,
                                                      self.ds['static_rule_one'],
                                                      self.ds['static_rule_two'],
                                                      self.ds['static_rule_three'],
                                                      kwargs=kwargs)

            # generate and combines rules into alert for dynamic change 
            self.ds['dynamic_alerts'] = xr.apply_ufunc(build_alerts,
                                                       self.ds['dynamic_rule_one'],
                                                       self.ds['dynamic_rule_two'],
                                                       self.ds['dynamic_rule_three'],
                                                       kwargs=kwargs)
        except Exception as e:
            raise ValueError(e)
            
        # check if we have any data
        for var in ['static_alerts', 'dynamic_alerts']:
            if self.ds[var].isnull().all():
                raise ValueError('Alert result empty.')
                
        return
    
    
    def apply_xr_edge_mask(self):
            """
            Applies the xr edge mask generated earlier during the 
            set_xr_edge_mask function. Any pixels within this mask 
            will be set to nodata (nan). If an error occurs, no mask 
            is applied.
            """ 

            # check if xr exists
            if self.ds is None:
                raise ValueError('No xr provided.')

            # check if edge mask exists
            if self.mask is None:
                raise ValueError('No edge mask provided.')       

            try:
                # take a copy in case of error
                tmp = self.ds.copy(deep=True)

                # mask edge pixels to nan
                self.ds = self.ds.where(self.mask)

                # check if not all nan
                if self.ds.to_array().isnull().all():
                    raise ValueError('Mask set all pixels to nan, rolling back.')  

            except Exception as e:
                self.ds = tmp
                raise ValueError(e)

            return
    
    
    def perform_kernel_density(self):
        """
        Generates various summary states of vegetation
        and change over time and displays it as
        a kernel density raster. Error will result in
        raised error.
        """

        # ensure x, y and time in dataset
        if 'x' not in self.ds or 'y' not in self.ds:
            raise ValueError('No x, y dimensions.')
        elif 'time' not in self.ds:
            raise ValueError('No time dimensions.')

        try:
            # smooth dataset via mean
            self.ds = self.ds.rolling(x=3, y=3, 
                                      center=True, 
                                      min_periods=1).mean()

            # increase resolution of grid 5-fold
            x_min, x_max = float(self.ds['x'].min()), float(self.ds['x'].max())
            y_min, y_max = float(self.ds['y'].min()), float(self.ds['y'].max())

            # generate high resolution coordinates
            xs = np.linspace(x_min, x_max, len(self.ds['x']) * 5)
            ys = np.linspace(y_min, y_max, len(self.ds['y']) * 5)

            # interpolate values to new grid
            self.ds = self.ds.interp(x=xs, y=ys)

        except Exception as e:
            raise ValueError(e)

        # check if method provided 
        if self.method not in ['Static', 'Dynamic']:
            raise ValueError('Change method not provided.')

        # prepare method name
        method = self.method.lower()

        try:
            # set up mask
            da_mask = self.ds['veg_idx'].mean('time')
            da_mask = xr.where(~da_mask.isnull(), True, False)
            
            # set up core arrays 
            da_veg = self.ds['veg_idx']
            da_chg = self.ds['{}_clean'.format(method)]
            da_zne = self.ds['{}_zones'.format(method)]
            da_alt = self.ds['{}_alerts'.format(method)]
                        
            # get vege avg and std all-time
            da_veg_avg = da_veg.mean('time')
            da_veg_std = da_veg.std('time')
            
            # get latest vege
            da_veg_lts = da_veg.isel(time=-2, drop=True)
            
            # get change max inc, dec all-time
            da_chg_max_inc = da_chg.where(da_chg >= 0, 0).max('time')
            da_chg_max_dec = da_chg.where(da_chg <= 0, 0).min('time')
            
            # get change avg inc, dec all-time
            da_chg_avg_inc = da_chg.where(da_chg >= 0, 0).mean('time')
            da_chg_avg_dec = da_chg.where(da_chg <= 0, 0).mean('time')
            
            # get latest change inc, dec 
            da_chg_lts_inc = da_chg.where(da_chg >= 0, 0).isel(time=-2, drop=True)
            da_chg_lts_dec = da_chg.where(da_chg <= 0, 0).isel(time=-2, drop=True)
            
            # get count alerts all time all dirs
            da_alt_cnt_inc = da_alt.where(da_chg >= 0, 0).sum('time')
            da_alt_cnt_dec = da_alt.where(da_chg <= 0, 0).sum('time')
            
            # set up list of clean datasets
            ds_list = [
                da_veg_avg.to_dataset(name='vege_avg_all_time'),
                da_veg_std.to_dataset(name='vege_std_all_time'),
                da_veg_lts.to_dataset(name='vege_latest_time'),   
                da_chg_max_inc.to_dataset(name='change_max_all_time_incline'),  
                da_chg_max_dec.to_dataset(name='change_max_all_time_decline'),  
                da_chg_avg_inc.to_dataset(name='change_avg_all_time_incline'),  
                da_chg_avg_dec.to_dataset(name='change_avg_all_time_decline'),     
                da_chg_lts_inc.to_dataset(name='change_latest_time_incline'),  
                da_chg_lts_dec.to_dataset(name='change_latest_time_decline'),  
                da_alt_cnt_inc.to_dataset(name='alerts_cnt_all_time_incline'),  
                da_alt_cnt_dec.to_dataset(name='alerts_cnt_all_time_decline'),  
            ]

            # combine into one, apply mask
            ds = xr.merge(ds_list)
            ds = ds.where(da_mask)

        except Exception as e:
            raise ValueError(e)

        # check if anything in dataset
        if ds.to_array().isnull().all():
            raise ValueError('No kernel densities could be generated.')

        # set to class dataset
        self.ds = ds.copy(deep=True)

        return
    

    def append_attrs(self):
        """
        Adds expected attributes to xr
        dataset prior to export.
        """

        # check if xr exists
        if self.ds is None:
            raise ValueError('No xr dataset exists.')

        # manually create attrs for dataset (geotiffs lacking) 
        self.ds = tools.manual_create_xr_attrs(self.ds)
        self.ds.attrs.update({'nodatavals': np.nan})   

        return


    def export_xr(self):
        """
        Exports kernel density-fied xr (which contains 
        everything)  to netcdf named with global id. An
        error will raise an error.
        """
        
        # check if xr valid
        if self.ds is None:
            raise ValueError('No xr provided.')
        elif not isinstance(self.ds, xr.Dataset):
            raise TypeError('The xr is not an xr dataset type.')
            
        # check if path and global id exist
        if self.out_nc is None:
            raise ValueError('No output NetCDF provided.')

        try:           
            # export nc
            self.ds.to_netcdf(self.out_nc)
        
        except Exception as e:
            raise ValueError(e)
        
        return
          
    
    def reset(self):
        """
        Resets all generated area parameters 
        and xr datasets. 
        """

        # set proj geom to none
        self.prj_geom = None

        # set alert info to none
        self.alert_zone =  None
        self.alert_flag =  None
        self.alert_html =  None
        self.alert_graph = None

        # iter xrs and close 
        xrs = [self.ds, self.mask]
        for x in xrs:
            try:
                if x is not None:
                    x.close()
            except:
                pass

        # set xrs to none
        self.ds = None
        self.ds_mask = None

        return  
        
    def run_all(self):
        """
        """
        
        
        
        
    
    

# meta
def validate_monitoring_areas(in_feat):
    """
    Does relevant checks for information for a
    gdb feature class of one or more monitoring areas.
    """
    
    # set up flag
    is_valid = True

    # check input feature is not none and strings
    if in_feat is None:
        print('Monitoring area feature class not provided, flagging as invalid.')
        is_valid = False
    elif not isinstance(in_feat, str):
        print('Monitoring area feature class not string, flagging as invalid.')
        is_valid = False
    elif not os.path.dirname(in_feat).endswith('.gdb'):
        print('Feature class is not in a geodatabase, flagging as invalid.')
        is_valid = False

    # if valid...
    if is_valid:
        try:
            # get feature
            driver = ogr.GetDriverByName("OpenFileGDB")
            data_source = driver.Open(os.path.dirname(in_feat), 0)
            lyr = data_source.GetLayer('monitoring_areas')
            
            # get epsg
            epsg = lyr.GetSpatialRef()
            if 'GDA_1994_Australia_Albers' not in epsg.ExportToWkt():
                print('Could not find GDA94 albers code in shapefile, flagging as invalid.')
                is_valid = False
            
            # check if any duplicate area ids
            area_ids = []
            for feat in lyr:
                area_ids.append(feat['area_id'])
                
            # check if duplicate area ids
            if len(set(area_ids)) != len(area_ids):
                print('Duplicate area ids detected, flagging as invalid.')
                is_valid = False
                
            # check if feature has required fields
            fields = [field.name for field in lyr.schema]
            required_fields = [
                'area_id', 
                'platform', 
                's_year', 
                'e_year', 
                'index',
                'persistence',
                'rule_1_min_conseqs',
                'rule_1_inc_plateaus',
                'rule_2_min_zone', 
                'rule_3_num_zones',
                'ruleset',
                'alert',
                'method',
                'alert_direction',
                'email',
                'ignore',
                'color',
                'global_id'
                ] 
            
            # check if all fields in feat
            if not all(f in fields for f in required_fields):
                print('Not all required fields in monitoring shapefile.')
                is_valid = False
                
            # close data source
            data_source.Destroy()
            
        except:
            print('Could not open monitoring area feature, flagging as invalid.')
            is_valid = False
            data_source.Destroy()

    # return
    return is_valid
 



def fetch_cube_data(collections, bands, start_dt, end_dt, bbox, resolution=30, ds_existing=None):
    """
    Takes a set of metadata (dea product names, band names,
    start and end datesm bbox in wgs84 lat and lons,
    a resolution value and an existing dataset, and fetches
    all available dea satellite data for the query. Returns
    an xarray dataset object (lazy).
    
    Parameters
    ----------
    collections: list
        A list of dea collection names.
    bands : list
        A list of dea band names.
    start_dt : string
        Starting datetime of query in format YYYY-MM-DD.
    end_dt : string
        Ending datetime of query in format YYYY-MM-DD.
    bbox: list 
        List of bounding box lat/lon pairs. XMIN, YMIN, 
        XMAX, YMAX.
    resolution: int, float 
        Resolution of resampled pixels.
    ds_existing: xarray dataset 
        Use an existing xarray dataset to generate query,
        outside of start and end date.

    Returns
    -------
    ds : xarray Dataset
    """

    # notify
    print('Obtaining all satellite data for monitoring area.')

    try:
        # query stac endpoint
        items = cog_odc.fetch_stac_items_odc(stac_endpoint='https://explorer.sandbox.dea.ga.gov.au/stac', 
                                             collections=collections, 
                                             start_dt=start_dt, 
                                             end_dt=end_dt, 
                                             bbox=bbox,
                                             slc_off=False,
                                             limit=250)
               
        # replace s3 prefix with https for each band - arcgis doesnt like s3
        items = cog_odc.replace_items_s3_to_https(items=items, 
                                          from_prefix='s3://dea-public-data', 
                                          to_prefix='https://data.dea.ga.gov.au')                            
    except Exception as e:
        raise ValueError(e)

    try:
        # build xarray dataset from stac data
        ds = cog_odc.build_xr_odc(items=items,
                                  bbox=bbox,
                                  bands=bands,
                                  crs=3577,
                                  res=resolution,
                                  resampling='Nearest',
                                  align=None,
                                  group_by='solar_day',
                                  chunks={},
                                  like=ds_existing)
    
        # convert to float 32 
        ds = ds.astype('float32')
        
        # loop all vars without mask in name
        for var in list(ds.data_vars):
            if 'mask' not in var.lower():
                ds[var] = ds[var].where(ds[var] != 0, -999)
    
        # convert datetimes (strip milliseconds)
        dts = ds.time.dt.strftime('%Y-%m-%dT%H:%M:%S')
        ds['time'] = dts.astype('datetime64[ns]')
    
    
    except Exception as e:
        raise ValueError(e)

    return ds


def rasterize_polygon(ds, geom):
    """
    Takes an esri polygon geometry object associated
    with a monitoring area and rasterizes into a binary 
    mask using ogr and gdal. 
    
    Parameters
    ----------
    ds: xarray dataset
        A dataset containing x, y, geobox and attributes.
    geom : arcpy polygon geometry
        A single polygon geometry object.

    Returns
    -------
    da : xarray Dataset
    """
    
    # check dataset 
    if ds is None:
        raise ValueError('Dataset not provided.')
    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset is not an xarray type.')
    if not hasattr(ds, 'geobox'):
        raise ValueError('Dataset has no geobox.')
    
    # check geometry
    if geom.type != 'polygon':
        raise TypeError('Geometry is not polygon type.')
    elif geom.isMultipart is True:
        raise TypeError('Do not support multi-part geometry.')
    
    try:
        # convert esri geometry to ogr layer
        wkb = ogr.CreateGeometryFromWkb(geom.WKB)
        jsn = wkb.ExportToJson()
        jsn = ogr.Open(jsn, 0)
        lyr = jsn.GetLayer()
        
        # get dataset x, y coords
        x = np.array(ds['x'])
        y = np.array(ds['y'])
        
        # get x, y num cells
        ncol = ds.sizes['x']
        nrow = ds.sizes['y']
        
        # get transform of dataset
        t = ds.geobox.transform
        tran = (t[2], t[0], 0.0, t[5], 0.0, t[4])
        
        # get bounding box and bl, tr
        bbox = ds.geobox.extent.boundingbox
        xmin, ymin, xmax, ymax = bbox 
        
        # create empty gdal raster in memory
        rast = gdal.GetDriverByName('MEM').Create('', ncol, nrow, 1, gdal.GDT_Byte)
        rast.SetGeoTransform(tran)

        # set raster transform and initial band values
        band = rast.GetRasterBand(1)
        band.Fill(0)
        band.SetNoDataValue(0)
        
        # rasterise vector with new raster as canvas
        gdal.RasterizeLayer(rast, [1], lyr, burn_values=[1])
        rast.FlushCache()
        
        # build an xarray data array from result
        da = xr.DataArray(
            data=rast.GetRasterBand(1).ReadAsArray(),
            dims=['y', 'x'],
            coords={'y': y, 'x': x}
        )
        
        return da
    
    except Exception as e:
        raise ValueError(e)


def remove_spikes(da, factor=2, win_size=3):
    """
    Removes outliers using a moving window median filter with 
    a neighbour max and min check technique based on the TIMESAT 
    3.3 software. Only works on an xarray data array type.
    
    Parameters
    ----------
    da: xarray dataarray
        A data array containing x, y, time, geobox and 
        attributes.
    factor : int
        The factor in which to multiply the std threshold.
        A lower value will remove more outliers, a higher 
        value will remove fewer.
    win_size : int 
        The size of the moving window. A higher win size 
        will capture more standard deviation.

    Returns
    -------
    da : xarray Dataset
    """

    # notify user
    print('Removing spike outliers.')

    # check if user factor provided
    if factor <= 0:
        factor = 1

    # check win_size not less than 3 and odd num
    if win_size < 3:
        win_size == 3
    elif win_size % 2 == 0:
        win_size += 1

    # calc cutoff val per pixel (std of pixel) multiply by user-factor 
    cutoff = da.std('time') * factor

    # calc rolling median for whole dataset
    da_med = da.rolling(time=win_size, center=True).median()

    # calc abs diff of orig and med vals
    da_dif = abs(da - da_med)

    # calc mask
    da_mask = da_dif > cutoff

    # shift vals left, right one time index, get mean and fmax per center
    l = da.shift(time=1).where(da_mask)
    r = da.shift(time=-1).where(da_mask)
    da_mean = (l + r) / 2
    da_fmax = xr.ufuncs.fmax(l, r)

    # flag only if mid val < mean of l, r - cutoff or mid val > max val + cutoff
    da_spikes = xr.where((da.where(da_mask) < (da_mean - cutoff)) | 
                         (da.where(da_mask) > (da_fmax + cutoff)), True, False)

    # set spikes to nan
    da = da.where(~da_spikes)

    # notify and return
    print('Spike removal completed successfully.')
    return da
 
 
def safe_savgol(arr, window_length=3, polyorder=1, a=0):
    """
    Small wrapper for the savgol_filter 
    function that ensures no nan go in. 
    'a' is axis.
    """
    
    # correct nan
    if np.isnan(arr).all():
        return arr
    elif np.isnan(arr).any():
        avg = np.nanmean(arr)
        arr = np.nan_to_num(arr, nan=avg)
        
    try:
        # perform savgol
        return savgol_filter(arr, 
                             window_length=3, 
                             polyorder=1,
                             axis=0)
    except:
        # return empty array
        return np.full_like(arr, np.nan)
        
 
def detect_change(ds, method='both', var='veg_idx', train_start=None, train_end=None, persistence=1.0):
    """
    Performs EWMACD change detection on a 1d array of variable 
    values. 

    Parameters
    ----------
    ds: xarray dataset
        A dataset containing x, y, time dimensions. Values are 
        extracted to this dataset.
    method: str
        Can be static, dynamic or both. If both, both
        static and dynamic analyses are done.
    var : str
        A string with the name of the variable to perform 
        change detection on.
    train_start : int 
        Start year of training period.
    end_start : int 
        End year of training period.
    persistence : float 
        Vegetation persistence value.
        
    Returns
    -------
    ds : xarray Dataset
    """
    
    # checks
    if ds is None:
        raise ValueError('Dataset is empty.')
    elif not isinstance(ds, xr.Dataset):
        raise ValueError('Dataset type expected.')
    elif 'time' not in ds:
        raise ValueError('Dataset needs a time dimension.')
        
    # check method is supported, set default if wrong
    if method not in ['static', 'dynamic', 'both']:
        method = 'both'
    
    # check if var in dataset
    if var not in ds:
        raise ValueError('Requested variable not found.')
        
    # check training start
    if train_start is None:
        raise ValueError('Provide a training start year.')
    elif train_start >= ds['time.year'].max():
        raise ValueError('Training start must be lower within dataset range.')
        
    # notify
    print('Beginning change detection.')
    
    # check if any data exists
    if len(ds['time']) == 0:
        raise ValueError('No data in training period.')
        
    try:
        # perform change detection (static)
        ds_stc = EWMACD(ds=ds[var].to_dataset(),
                        trainingPeriod='static',
                        trainingStart=train_start,
                        trainingEnd=train_end,
                        persistence_per_year=persistence)
    except Exception as e:
        raise ValueError(e)
   
    try:
        # perform change detection (dynamic)
        ds_dyn = EWMACD(ds=ds[var].to_dataset(),
                        trainingPeriod='dynamic',
                        trainingStart=train_start,
                        trainingEnd=train_end,
                        persistence_per_year=persistence)
    except Exception as e:
        raise ValueError(e)
                    
    # rename static, dynamic output var
    ds_stc = ds_stc.rename({var: 'static_raw'})
    ds_dyn = ds_dyn.rename({var: 'dynamic_raw'})
    
    # return based on method (both, static, dynamic)
    if method == 'both':
        ds['static_raw'] = ds_stc['static_raw']
        ds['dynamic_raw'] = ds_dyn['dynamic_raw']
    elif method == 'static':
        ds['static_raw'] = ds_stc['static_raw']
    else:
        ds['dynamic_raw'] = ds_dyn['dynamic_raw']

    return ds


def transfer_xr_values(ds_to, ds_from, data_vars):    
    """
    Transfers all values, date-by-date, from the 'from' dataset 
    to the 'to' dataset only where datetimes in 'to' correspond to 
    those in 'from'. As Xarray has no replace value, this is the 
    safest (albeit inefficient) method for moving values between
    datatsets.
    
    Parameters
    ----------
    ds_to: xarray dataset
        A dataset containing x, y, time dimensions. Values are 
        extracted to this dataset.
    ds_from: xarray dataset
        A dataset containing x, y, time dimensions. Values are 
        extracted from this dataset.
    data_vars : list 
        List of variables to extract values to - from.

    Returns
    -------
    ds_to : xarray Dataset
    """
    
    # check data vars provided
    if data_vars is None:
        data_vars = []
    elif isinstance(data_vars, str):
        data_vars = [data_vars]    
    
    # check if time is in datasets
    if 'time' not in ds_to or 'time' not in ds_from:
        raise ValueError('Time dimension not in both datasets.')
    
    # check if variables exist in both datasets
    for var in data_vars:
        if var not in ds_to or var not in ds_from:
            raise ValueError('Requested vars not in both datasets.')
            
    # iter new dates and manual update change vars
    for dt in ds_from['time']:
        da = ds_from.sel(time=dt)
        
        # if time exists in transfer 'to ds', proceed
        if da['time'].isin(ds_to['time']) == True:
            for var in data_vars:
                ds_to[var].loc[{'time': dt}] = da[var]
            
    return ds_to


def build_zones(arr):   
    """
    Takes a static and/or change detection array of values 
    and classifies into 1 of 11 zones. These zones are used 
    to drive the alert and colouring system of the NRT
    method. The output values include zone direction  in way 
    of sign (-/+).
    
    Parameters
    ----------
    arr: numpy arr
        A 1d numpy array of raw change values.

    Returns
    -------
    arr : numpy array
    """

    # set up zone ranges (stdv ranges)
    zones = [
        [0, 1],    # zone 1 - from 0 to 1 (+/-)
        [1, 3],    # zone 2 - between 1 and 3 (+/-)
        [3, 5],    # zone 3 - between 3 and 5 (+/-)
        [5, 7],    # zone 4 - between 5 and 7 (+/-)
        [7, 9],    # zone 5 - between 7 and 9 (+/-)
        [9, 11],   # zone 6 - between 9 and 11 (+/-)
        [11, 13],  # zone 7 - between 11 and 13 (+/-)
        [13, 15],  # zone 8 - between 13 and 15 (+/-)
        [15, 17],  # zone 9 - between 15 and 17 (+/-)
        [17, 19],  # zone 10 - between 17 and 19 (+/-)
        [19]       # zone 11- above 19 (+/-)
    ]

    # create negative sign mask
    arr_neg_mask = np.where(arr < 0, True, False)
    
    # create template arr
    arr_temp = np.full_like(arr, np.nan)
    
    # get abs of arr
    arr = np.abs(arr)

    # iter zones
    for i, z in enumerate(zones, start=1):

        if i == 1:
            arr_temp[np.where((arr >= z[0]) & (arr < z[1]))] = i
            
        elif i == 11:
            arr_temp[np.where(arr >= z[0])] = i
            
        else:
            arr_temp[np.where((arr >= z[0]) & (arr < z[1]))] = i
        
    # check if arr sizes match
    if len(arr) != len(arr_temp):
        raise ValueError('Input array differs in size to output array.')
        
    # mask signs
    arr_temp = np.where(arr_neg_mask, arr_temp * -1, arr_temp)
    
    return arr_temp


def zone_to_std(zone):
    """
    Takes a zone value (0-11 or above) and converts it 
    to its associated standard deviation. For example, if 
    entering zone 2, the stdv value that will be returned 
    is [1, 3]. Returns 0 if zone 0 requested. Returns 999 
    if zone 11, or not supported.
    
    Parameters
    ----------
    zone: int
        A zone value from 0-11 or above.

    Returns
    -------
    arr : An array of lower and upper values of zone.
    """
    
    if zone is None:
        return [0, 0]
    if zone == 0:
        return [0, 0]
    elif zone == 1:
        return [0, 1]    
    elif zone == 2:
        return [1, 3]
    elif zone == 3:
        return [3, 5]
    elif zone == 4:
        return [5, 7]
    elif zone == 5:
        return [7, 9]
    elif zone == 6:
        return [9, 11]
    elif zone == 7:
        return [11, 13]
    elif zone == 8:
        return [13, 15]
    elif zone == 9:
        return [15, 17]
    elif zone == 10:
        return [17, 19]
    elif zone >= 11:
        return [19, 999]
    else:
        return [19, 999]


def build_rule_one_runs(arr, min_conseqs=3, inc_plateaus=False):
    """
    Takes a static and/or change detection array of values 
    and generates consequtive value runs. If raw values continuously
    decline, date after date, a run is started and each time is 
    counted. Once this goes up, the count is stopped and reset until
    the next decline trajectory. Same for incline. The output values 
    include run direction in way of sign (-/+). Users can set the 
    minimum number of consequtives required before run begins, as well
    as to include post-consequtive plateaus of identical values in run 
    count.
    
    Parameters
    ----------
    arr: numpy arr
        A 1d numpy array of raw change values.
    min_conseqs : int 
        Controls the number of consequtive inclines 
        or declines before counts are recorded. 
    inc_plateaus : bool 
        Whether to include plateau values in the run 
        count. 

    Returns
    -------
    arr : numpy array
    """
    
    # check if arr is all nan
    if np.isnan(arr).all():
        raise ValueError('Array is all nan.')
        
    # check min stdvs
    if min_conseqs is None:
        print('No minimum consequtives provided, setting to default (3).')
        min_stdv = 0
    elif not isinstance(min_conseqs, (int, float)):
        print('Minimum consequtives not numeric, returning original array.')
        return arr
    elif min_conseqs < 0:
        print('Minimum consequtives only takes positives, getting absolute.')
        arr = abs(min_stdv)
    
    # check plateaus
    if not isinstance(inc_plateaus, bool):
        print('Include plateaus must be boolean. Setting to False.')
        inc_plateaus = False
        
    # set up empty incline and decline arrays
    arr_incs = np.zeros(len(arr))
    arr_decs = np.zeros(len(arr))
    
    # build runs of consequtive positive values
    direction = 0
    for i in range(1, len(arr)):

        # if curr not 0 and curr > prev, + 1
        if arr[i] != 0 and arr[i] > arr[i - 1]:
            arr_incs[i] = arr_incs[i - 1] + 1  
            direction = 1  

        # if curr == prev (non-zero plateau) and prev dir was incline, + 1
        elif arr[i] != 0 and arr[i] == arr[i - 1] and direction == 1 and inc_plateaus:
            arr_incs[i] = arr_incs[i - 1] + 1  

        # reset dir otherwise
        else:
            direction = 0  
                
    # build runs of consequtive decline values
    direction = 0
    for i in range(1, len(arr)):

        # if curr not 0 and curr < prev, - 1
        if arr[i] != 0 and arr[i] < arr[i - 1]:
            arr_decs[i] = arr_decs[i - 1] - 1
            direction = -1  

        # if curr == prev (non-zero plateau) and prev dir was decline, - 1
        elif arr[i] != 0 and arr[i] == arr[i - 1] and direction == -1 and inc_plateaus:
            arr_decs[i] = arr_decs[i - 1] - 1 

        # reset dir otherwise
        else:
            direction = 0  
    
    # combine both into one
    arr_runs = arr_incs + arr_decs
    
    # remove any run values under min consequtives 
    if min_conseqs > 0:
        arr_runs = np.where(np.abs(arr_runs) >= min_conseqs, arr_runs, 0)
        
    # convert to float32
    arr_runs = arr_runs.astype('float32')
    
    return arr_runs


def build_rule_two_mask(arr, min_stdv=0):
    """
    Takes a static and/or change detection array of values 
    and masks out any change deviations beneath a specified 
    minimum threshold. Masked values set to 0. The output 
    values include direction in way of sign (-/+). Users can 
    set the minimum std of allowed values.
    
    Parameters
    ----------
    min_stdv: int
        Minimum stdv to keep after threshold.

    Returns
    -------
    arr : array of values above min stdv threshold.
    """
    
    # check if arr is all nan
    if np.isnan(arr).all():
        raise ValueError('Array is all nan.')
        
    # check min stdvs
    if min_stdv is None:
        print('No minimum std. dev.provided, setting to default (0).')
        min_stdv = 0
    elif not isinstance(min_stdv, (int, float)):
        print('Minimum std. dev., returning original array.')
        return arr
    elif min_stdv < 0:
        print('Minimum std. dev. only takes positives, getting absolute.')
        arr = abs(min_stdv)
        
    # threshold out all values within threshold area, if 0, ignore it
    if min_stdv == 0:
        arr_thresh = np.where(np.abs(arr) > min_stdv, arr, 0)
    else:
        arr_thresh = np.where(np.abs(arr) >= min_stdv, arr, 0)

    # convert to float32
    arr_thresh = arr_thresh.astype('float32')

    return arr_thresh


def build_rule_three_spikes(arr, min_stdv=4):
    """
    Takes a static and/or change detection array of values 
    and detects any sharp spikes in stdv jumps between
    dates. The output values include direction in way of 
    sign (-/+). Users can set the minimum number of stdv
    required between leaps before an alarm is triggered.
    
    Parameters
    ----------
    min_stdv: int
        Minimum stdv of leap between dates.

    Returns
    -------
    arr : array of values where leap was detected.
    """
    
    # check if arr is all nan
    if np.isnan(arr).all():
        raise ValueError('Array is all nan.')
        
    # check min stdvs
    if min_stdv is None:
        print('No minimum std. dev.provided, setting to default (4).')
        min_stdv = 4
    elif not isinstance(min_stdv, (int, float)):
        print('Minimum std. dev., returning original array.')
        return arr
    elif min_stdv < 0:
        print('Minimum std. dev. only takes positives, getting absolute.')
        arr = abs(min_stdv)
        
    # get differences between values
    arr_diffs = np.diff(arr, prepend=arr[0])
    
    # threshold out all values within threshold area, if 0, ignore it
    if min_stdv == 0:
        arr_spikes = np.where(np.abs(arr_diffs) > min_stdv, arr, 0)
    else:
        arr_spikes = np.where(np.abs(arr_diffs) >= min_stdv, arr, 0)
    
    # convert to float32
    arr_spikes = arr_spikes.astype('float32')
    
    return arr_spikes


def build_alerts(arr_r1, arr_r2, arr_r3, ruleset='1 and 2 or 3', direction='Decline only (any)'):
    """
    Builds alert mask (1s and 0s) based on combined rule
    values and assigned ruleset. Takes vars of rule 1, 
    rule 2, rule 3 of both static and dynamic change methods 
    and combines them using ruleset. Users can also set
    the direction of the change required for an alert to be 
    set to 1 (true).
    
    Parameters
    ----------
    arr_r1: numpy array
        Numpy array of rule one values.
    arr_r2: numpy array
        Numpy array of rule two values.
    arr_r3: numpy array
        Numpy array of rule three values.   
    ruleset : str 
        Set the ruleset combination.
    direction : str
        Set the direction of values required to set alert.

    Returns
    -------
    arr : array of alert mask values.
    """
    
    # set up valid rulesets
    valid_rules = [
        '1 only', 
        '2 only', 
        '3 only', 
        '1 and 2', 
        '1 and 3', 
        '2 and 3', 
        '1 or 2', 
        '1 or 3', 
        '2 or 3', 
        '1 and 2 and 3', 
        '1 or 2 and 3',
        '1 and 2 or 3', 
        '1 or 2 or 3'
        ]
     
    # check if ruleset is valid
    if ruleset not in valid_rules:
        raise ValueError('Ruleset is not supported.')
     
    # set up valid directions 
    valid_directions = [
        'Incline only (any)',
        'Decline only (any)',
        'Incline only (+ zones only)',
        'Decline only (- zones only)',
        'Incline or Decline (any)',
        'Incline or Decline (+/- zones only)'
        ]
    
    # check if direction is valid
    if direction not in valid_directions:
        raise ValueError('Direction is not supported.')
        
    # check if rule arrays are empty 
    if np.isnan(arr_r1).all():
        raise ValueError('Rule one arr is empty.')
    elif np.isnan(arr_r2).all():
        raise ValueError('Rule two arr is empty.')
    elif np.isnan(arr_r3).all():
        raise ValueError('Rule three arr is empty.')
    
    try:
        # correct raw rule vals for direction and set 1 if alert, 0 if not
        if direction == 'Incline only (any)':
            arr_r1 = np.where(arr_r1 > 0, 1, 0)
            arr_r2 = np.where(arr_r2 != 0, 1, 0)  # ignore sign for exclusion zone
            arr_r3 = np.where(arr_r3 > 0, 1, 0)
            
        elif direction == 'Decline only (any)':
            arr_r1 = np.where(arr_r1 < 0, 1, 0)
            arr_r2 = np.where(arr_r2 != 0, 1, 0)  # ignore sign for exclusion zone
            arr_r3 = np.where(arr_r3 < 0, 1, 0)

        elif direction == 'Incline only (+ zones only)':
            arr_r1 = np.where(arr_r1 > 0, 1, 0)
            arr_r2 = np.where(arr_r2 > 0, 1, 0)
            arr_r3 = np.where(arr_r3 > 0, 1, 0)
            
        elif direction == 'Decline only (- zones only)':
            arr_r1 = np.where(arr_r1 < 0, 1, 0)
            arr_r2 = np.where(arr_r2 < 0, 1, 0)
            arr_r3 = np.where(arr_r3 < 0, 1, 0)    

        elif direction == 'Incline or Decline (any)':
            arr_r1 = np.where(arr_r1 != 0, 1, 0)
            arr_r2 = np.where(arr_r2 != 0, 1, 0)
            arr_r3 = np.where(arr_r3 != 0, 1, 0) 

        elif direction == 'Incline or Decline (+/- zones only)':
            arr_r1 = np.where(((arr_r1 > 0) & (arr_r2 > 0)) | 
                              ((arr_r1 < 0) & (arr_r2 < 0)), 1, 0)
            arr_r2 = np.where(arr_r2 != 0, 1, 0)
            arr_r3 = np.where(arr_r3 != 0, 1, 0) 
    
    except Exception as e:
        raise ValueError(e)

    try:
        # create alert arrays based on singular rule
        if ruleset == '1':
            arr_alerts = arr_r1
        elif ruleset == '2':
            arr_alerts = arr_r2
        elif ruleset == '3':
            arr_alerts = arr_r3    
        
        # create alert arrays based on dual "and" rule
        if ruleset == '1 and 2':
            arr_alerts  = arr_r1 & arr_r2
        elif ruleset == '1 and 3':
            arr_alerts  = arr_r1 & arr_r3
        elif ruleset == '2 and 3':
            arr_alerts  = arr_r2 & arr_r3  
        
        # create alert arrays based on dual "or" rule
        if ruleset == '1 or 2':
            arr_alerts  = arr_r1 | arr_r2
        elif ruleset == '1 or 3':
            arr_alerts  = arr_r1 | arr_r3
        elif ruleset == '2 or 3':
            arr_alerts  = arr_r2 | arr_r3    
        
        # create alert arrays based on complex rule
        if ruleset == '1 and 2 and 3':  
            arr_alerts  = arr_r1 & arr_r2 & arr_r3
        elif ruleset == '1 or 2 and 3':  
            arr_alerts  = arr_r1 | arr_r2 & arr_r3
        elif ruleset == '1 and 2 or 3':  
            arr_alerts  = arr_r1 & arr_r2 | arr_r3
        elif ruleset == '1 or 2 or 3':  
            arr_alerts  = arr_r1 | arr_r2 | arr_r3
    
    except Exception as e:
        raise ValueError(e)
    
    # check if result is empty
    if np.isnan(arr_alerts).all():
        raise ValueError('Alert array is empty.')
    
    return arr_alerts










# meta
def extract_new_xr_dates(ds_old, ds_new):
    """
    """
    
    # check if xarray is adequate 
    if not isinstance(ds_old, xr.Dataset) or not isinstance(ds_new, xr.Dataset):
        raise TypeError('Datasets not of Xarray type.')
    elif 'time' not in ds_old or 'time' not in ds_new:
        raise ValueError('Datasets do not have a time coordinate.')
    elif len(ds_old['time']) == 0 or len(ds_new['time']) == 0:
        raise ValueError('Datasets empty.')
    
    try:
        # select only those times greater than latest date in old dataset 
        new_dates = ds_new['time'].where(ds_new['time'] > ds_old['time'].isel(time=-1), drop=True)
        ds_new = ds_new.sel(time=new_dates)
        
        # check if new dates, else return none
        if len(ds_new['time']) != 0:
            return ds_new
    except:
        return

    return 


# meta, checks
def reclassify_signal_to_zones(arr):
    """
    takes a smoothed (or raw) ewmacd change detection
    signal and classifies into 1 of 11 zones based on the
    stdv values. this is used to help flag and colour
    outputs for nrt monitoring. Outputs include 
    zone direction information in way of sign (-/+).
    """   

    # set up zone ranges (stdvs)
    zones = [
        [0, 1],    # zone 1 - from 0 to 1 (+/-)
        [1, 3],    # zone 2 - between 1 and 3 (+/-)
        [3, 5],    # zone 3 - between 3 and 5 (+/-)
        [5, 7],    # zone 4 - between 5 and 7 (+/-)
        [7, 9],    # zone 5 - between 7 and 9 (+/-)
        [9, 11],   # zone 6 - between 9 and 11 (+/-)
        [11, 13],  # zone 7 - between 11 and 13 (+/-)
        [13, 15],  # zone 8 - between 13 and 15 (+/-)
        [15, 17],  # zone 9 - between 15 and 17 (+/-)
        [17, 19],  # zone 10 - between 17 and 19 (+/-)
        [19]       # zone 11- above 19 (+/-)
    ]

    # create template vector
    vec_temp = np.full_like(arr, fill_value=np.nan)

    # iter zones
    for i, z in enumerate(zones, start=1):

        # 
        if i == 1:
            vec_temp[np.where((arr >= z[0]) & (arr <= z[1]))] = i
            vec_temp[np.where((arr < z[0]) & (arr >= z[1] * -1))] = i * -1

        elif i == 11:       
            vec_temp[np.where(arr > z[0])] = i
            vec_temp[np.where(arr < z[0] * -1)] = i * -1

        else:
            vec_temp[np.where((arr > z[0]) & (arr <= z[1]))] = i
            vec_temp[np.where((arr < z[0] * -1) & (arr >= z[1] * -1))] = i * -1
        
    return vec_temp




# EWMACD METHOD
def harmonic_matrix(timeSeries0to2pi, numberHarmonicsSine,  numberHarmonicsCosine):

    # generate harmonic matrix todo 1 or 0? check
    col_ids = np.repeat(1, len(timeSeries0to2pi))

    # get sin harmonics todo need to start at 1, so + 1 to tail
    _ = np.vstack(np.arange(1, numberHarmonicsSine + 1))
    _ = np.repeat(_, len(timeSeries0to2pi), axis=1)
    col_sin = np.sin((_ * timeSeries0to2pi)).T

    # get cos harmonics todo need to start at 1, so + 1 to tail
    _ = np.vstack(np.arange(1, numberHarmonicsCosine + 1))
    _ = np.repeat(_, len(timeSeries0to2pi), axis=1)
    col_cos = np.cos((_ * timeSeries0to2pi)).T

    # stack into columns
    X = np.column_stack([col_ids, col_sin, col_cos])

    return X


def hreg_pixel(Responses, DOYs, numberHarmonicsSine, numberHarmonicsCosine, anomalyThresholdSigmas=1.5, valuesAlreadyCleaned=True):
    """hreg pixel function"""

    # todo this needs a check
    if valuesAlreadyCleaned == False:
        missingIndex = np.flatnonzero(np.isnan(Responses))
        if len(missingIndex) > 0:
            Responses = np.delete(Responses, missingIndex)
            DOYs = np.delete(DOYs, missingIndex)

    # assumes cleaned, non-missing inputs here; screening needs to be done ahead of running!
    Beta = np.repeat(np.nan, (1 + numberHarmonicsSine + numberHarmonicsCosine))
    Rsquared = None
    RMSE = None

    # generate harmonic matrix for given sin, cos numbers
    X = harmonic_matrix(DOYs * 2 * np.pi / 365, numberHarmonicsSine, numberHarmonicsCosine)

    # ensuring design matrix is sufficient rank and nonsingular
    if len(Responses) > (numberHarmonicsSine + numberHarmonicsCosine + 1) and np.abs(np.linalg.det(np.matmul(X.T, X))) >= 0.001:

        # todo check during harmonics > 1
        Preds1 = np.matmul(X, np.linalg.solve(np.matmul(X.T, X), np.vstack(np.matmul(X.T, Responses))))

        # x-bar chart anomaly filtering
        Resids1 = Responses[:, None] - Preds1  # todo i added the new axis [:, None]
        std = np.std(Resids1, ddof=1)
        screen1 = (np.abs(Resids1) > (anomalyThresholdSigmas * std)) + 0
        keeps = np.flatnonzero(screen1 == 0)

        if len(keeps) > (numberHarmonicsCosine + numberHarmonicsSine + 1):
            X_keeps = X[keeps, ]
            Responses_keeps = Responses[keeps]

            # todo check when using harmonics > 1
            Beta = np.linalg.solve(np.matmul(X_keeps.T, X_keeps),
                                   np.vstack(np.matmul(X_keeps.T, Responses_keeps))).flatten()

            fits = np.matmul(X_keeps, Beta)
            Rsquared = 1 - np.sum(np.square(Responses_keeps - fits)) / np.sum(np.square(Responses_keeps - np.sum(Responses_keeps) / len(Responses_keeps)))
            RMSE = np.sum(np.square(Responses_keeps - fits))

        # setup output
        output = {
            'Beta': Beta,
            'Rsquared': Rsquared,
            'RMSE': RMSE
        }

        return output


def optimize_hreg(timeStampsYears, timeStampsDOYs, Values, threshold, minLength, maxLength, ns=1, nc=1, screenSigs=3):
    """optimize hreg function"""

    minHistoryBound = np.min(np.flatnonzero((timeStampsYears >= timeStampsYears[minLength]) &
                                            ((timeStampsYears - timeStampsYears[0]) > 1)))  # todo changed from 1 to 0

    if np.isinf(minHistoryBound):  # todo using inf...
        minHistoryBound = 1

    # NOTE: maxLength applies from the point of minHistoryBound, not from time 1!
    historyBoundCandidates = np.arange(0, np.min(np.append(len(Values) - minHistoryBound, maxLength))) # todo removed the - 1, py dont start at 1!
    historyBoundCandidates = historyBoundCandidates + minHistoryBound

    if np.isinf(np.max(historyBoundCandidates)):  # todo using inf...
        historyBoundCandidates = len(timeStampsYears)

    i = 0
    fitQuality = 0
    while (fitQuality < threshold) and (i < np.min([maxLength, len(historyBoundCandidates)])):

        # Moving Window Approach todo needs a good check
        _ = np.flatnonzero(~np.isnan(Values[(i):(historyBoundCandidates[i])]))
        testResponses = Values[i:historyBoundCandidates[i]][_]

        # call hreg pixel function
        fitQuality = hreg_pixel(Responses=testResponses,
                                numberHarmonicsSine=ns,
                                numberHarmonicsCosine=nc,
                                DOYs=timeStampsDOYs[i:historyBoundCandidates[i]],
                                anomalyThresholdSigmas=screenSigs,
                                valuesAlreadyCleaned=True)

        # get r-squared from fit, set to 0 if empty
        fitQuality = fitQuality.get('Rsquared')
        fitQuality = 0 if fitQuality is None else fitQuality

        # count up
        i += 1

    # get previous history bound and previous fit
    historyBound = historyBoundCandidates[i - 1]  # todo added - 1 here to align with r 1 indexes

    # package output
    opt_output = {
        'historyBound': int(historyBound),
        'fitPrevious': int(minHistoryBound)
    }
    return opt_output


def EWMA_chart(Values, _lambda, histSD, lambdaSigs, rounding):
    """emwa chart"""

    ewma = np.repeat(np.nan, len(Values))
    ewma[0] = Values[0]  # initialize the EWMA outputs with the first present residual

    for i in np.arange(1, len(Values)):  # todo r starts at 2 here, so for us 1
        ewma[i] = ewma[(i - 1)] * (1 - _lambda) + _lambda * Values[i]  # appending new EWMA values for all present data.

    # ewma upper control limit. this is the threshold which dictates when the chart signals a disturbance
    # todo this is not an index, want array of 1:n to calc off those whole nums. start at 1, end at + 1
    UCL = histSD * lambdaSigs * np.sqrt(_lambda / (2 - _lambda) * (1 - (1 - _lambda) ** (2 * np.arange(1, len(Values) + 1))))

    # integer value for EWMA output relative to control limit (rounded towards 0).  A value of +/-1 represents the weakest disturbance signal
    output = None
    if rounding == True:
        output = (np.sign(ewma) * np.floor(np.abs(ewma / UCL)))
        output = output.astype('int16')  # todo added this to remove -0s
    elif rounding == False:
        # EWMA outputs in terms of resdiual scales.
        output = (np.round(ewma, 0))  # 0 is decimals

    return output


def persistence_filter(Values, persistence):
    """persistence filter"""
    # culling out transient values
    # keeping only values for which a disturbance is sustained, using persistence as the threshold
    tmp4 = np.repeat(0, len(Values))

    # ensuring sufficent data for tmp2
    if persistence > 1 and len(Values) > persistence:
        # disturbance direction
        tmpsign = np.sign(Values)

        # Dates for which direction changes # todo check this carefully, especially the two - 1s
        shiftpoints = np.flatnonzero(np.delete(tmpsign, 0) != np.delete(tmpsign, len(tmpsign) - 1))
        shiftpoints = np.append(np.insert(shiftpoints, 0, 0), len(tmpsign) - 1)  # prepend 0 to to start, len to end

        # Counting the consecutive dates in which directions are sustained
        # todo check this
        tmp3 = np.repeat(0, len(tmpsign))
        for i in np.arange(0, len(tmpsign)):
            tmp3lo = 0
            tmp3hi = 0

            while ((i + 1) - tmp3lo) > 0:  # todo added + 1
                if (tmpsign[i] - tmpsign[i - tmp3lo]) == 0:
                    tmp3lo += 1
                else:
                    break

            # todo needs look at index, check
            while (tmp3hi + (i + 1)) <= len(tmpsign):  # todo added + 1
                if (tmpsign[i + tmp3hi] - tmpsign[i]) == 0:
                    tmp3hi += 1
                else:
                    break

            # todo check indexes
            tmp3[i] = tmp3lo + tmp3hi - 1

        tmp4 = np.repeat(0, len(tmp3))
        tmp3[0:persistence, ] = persistence
        Values[0:persistence] = 0

        # if sustained dates are long enough, keep; otherwise set to previous sustained state
        # todo this needs a decent check
        for i in np.arange(persistence, len(tmp3)):  # todo removed + 1
            if tmp3[i] < persistence and np.max(tmp3[0:i]) >= persistence:
                tmpCbind = np.array([np.arange(0, i + 1), tmp3[0:i + 1], Values[0:i + 1]]).T  # todo added + 1
                tmp4[i] = tmpCbind[np.max(np.flatnonzero(tmpCbind[:, 1] >= persistence)), 2]  # todo is 3 or 2 the append value here?
            else:
                tmp4[i] = Values[i]

    return tmp4


def backfill_missing(nonMissing, nonMissingIndex, withMissing):
    """backfill missing"""

    # backfilling missing data
    withMissing = withMissing.copy()  # todo had to do a copy to prevent mem overwrite
    withMissing[nonMissingIndex] = nonMissing

    # if the first date of myPixel was missing/filtered, then assign the EWMA output as 0 (no disturbance).
    if np.isnan(withMissing[0]):
        withMissing[0] = 0

    # if we have EWMA information for the first date, then for each missing/filtered date
    # in the record, fill with the last known EWMA value
    for stepper in np.arange(1, len(withMissing)):
        if np.isnan(withMissing[stepper]):
            withMissing[stepper] = withMissing[stepper - 1]  # todo check this

    return withMissing


def EWMACD_clean_pixel_date_by_date(inputPixel, numberHarmonicsSine, numberHarmonicsCosine, inputDOYs, inputYears, trainingStart, trainingEnd, historyBound, precedents, xBarLimit1=1.5, xBarLimit2=20, lambdaSigs=3, _lambda=0.3, rounding=True, persistence=4):

    # prepare variables
    Dates = len(inputPixel)  # Convenience object
    outputValues = np.repeat(np.nan, Dates)  # Output placeholder
    residualOutputValues = np.repeat(np.nan, Dates)  # Output placeholder
    Beta = np.vstack(np.repeat(np.nan, (numberHarmonicsSine + numberHarmonicsCosine + 1)))

    # get training index and subset pixel
    indexTraining = np.arange(0, historyBound)
    myPixelTraining = inputPixel[indexTraining]            # Training data
    myPixelTesting = np.delete(inputPixel, indexTraining)  # Testing data

    ### Checking if there is data to work with...
    if len(myPixelTraining) > 0:
        out = hreg_pixel(Responses=myPixelTraining[(historyBound - precedents):historyBound],      # todo was a + 1 here
                         DOYs=inputDOYs[indexTraining][(historyBound - precedents):historyBound],  # todo was a + 1 here
                         numberHarmonicsSine=numberHarmonicsSine,
                         numberHarmonicsCosine=numberHarmonicsCosine,
                         anomalyThresholdSigmas=xBarLimit1)

        # extract beta variable
        Beta = out.get('Beta')

        # checking for present Beta
        if Beta[0] is not None:
            XAll = harmonic_matrix(inputDOYs * 2 * np.pi / 365, numberHarmonicsSine, numberHarmonicsCosine)
            myResiduals = (inputPixel - np.matmul(XAll, Beta).T)  # residuals for all present data, based on training coefficients
            residualOutputValues = myResiduals.copy()  # todo added copy for memory write

            myResidualsTraining = myResiduals[indexTraining]  # training residuals only
            myResidualsTesting = np.array([])

            if len(myResiduals) > len(myResidualsTraining):  # Testing residuals
                myResidualsTesting = np.delete(myResiduals, indexTraining)

            SDTraining = np.std(myResidualsTraining, ddof=1)  # first estimate of historical SD
            residualIndex = np.arange(0, len(myResiduals))  # index for residuals
            residualIndexTraining = residualIndex[indexTraining]  # index for training residuals
            residualIndexTesting = np.array([])

            # index for testing residuals
            if len(residualIndex) > len(residualIndexTraining):
                residualIndexTesting = np.delete(residualIndex, indexTraining)

            # modifying SD estimates based on anomalous readings in the training data
            # note that we don't want to filter out the changes in the testing data, so xBarLimit2 is much larger!
            UCL0 = np.concatenate([np.repeat(xBarLimit1, len(residualIndexTraining)),
                                   np.repeat(xBarLimit2, len(residualIndexTesting))])
            UCL0 = UCL0 * SDTraining

            # keeping only dates for which we have some vegetation and aren't anomalously far from 0 in the residuals
            indexCleaned = residualIndex[np.abs(myResiduals) < UCL0]
            myResidualsCleaned = myResiduals[indexCleaned]

            # updating the training SD estimate. this is the all-important modifier for the EWMA control limits.
            SDTrainingCleaned = myResidualsTraining[np.flatnonzero(np.abs(myResidualsTraining) < UCL0[indexTraining])]
            SDTrainingCleaned = np.std(SDTrainingCleaned, ddof=1)

            ### -------
            if SDTrainingCleaned is None:  # todo check if sufficient for empties
                cleanOutput = {
                    'outputValues': outputValues,
                    'residualOutputValues': residualOutputValues,
                    'Beta': Beta
                }
                return cleanOutput

            chartOutput = EWMA_chart(Values=myResidualsCleaned, _lambda = _lambda,
                                     histSD=SDTrainingCleaned, lambdaSigs=lambdaSigs,
                                     rounding=rounding)

            ###  Keeping only values for which a disturbance is sustained, using persistence as the threshold
            # todo this produces the wrong result, check the for loop out
            persistentOutput = persistence_filter(Values=chartOutput, persistence=persistence)

            # Imputing for missing values screened out as anomalous at the control limit stage
            outputValues = backfill_missing(nonMissing=persistentOutput, nonMissingIndex=indexCleaned,
                                            withMissing=np.repeat(np.nan, len(myResiduals)))

    # create output
    cleanOutput = {
        'outputValues': outputValues,
        'residualOutputValues': residualOutputValues,
        'Beta': Beta
    }

    return cleanOutput


def EWMACD_pixel_date_by_date(myPixel, DOYs, Years, _lambda, numberHarmonicsSine, numberHarmonicsCosine, trainingStart, testingEnd, trainingPeriod='dynamic', trainingEnd=None, minTrainingLength=None, maxTrainingLength=np.inf, trainingFitMinimumQuality=None, xBarLimit1=1.5, xBarLimit2=20, lowthresh=0, lambdaSigs=3, rounding=True, persistence_per_year=0.5, reverseOrder=False, simple_output=True):
    """pixel date by date function"""

    # setup breakpoint tracker. note arange ignores the value at stop, must + 1
    breakPointsTracker = np.arange(0, len(myPixel))
    breakPointsStart = np.array([], dtype='int16')
    breakPointsEnd = np.array([], dtype='int16')
    BetaFirst = np.repeat(np.nan, (1 + numberHarmonicsSine + numberHarmonicsCosine)) # setup betas (?)

    ### initial assignment and reverse-toggling as specified
    if reverseOrder == True:
        myPixel = np.flip(myPixel) # reverse array

    # convert doys, years to decimal years for ordering
    DecimalYears = (Years + DOYs / 365)

    ### sort all arrays based on order of decimalyears order via indexes
    myPixel = myPixel[np.argsort(DecimalYears)]
    Years = Years[np.argsort(DecimalYears)]
    DOYs = DOYs[np.argsort(DecimalYears)]
    DecimalYears = DecimalYears[np.argsort(DecimalYears)]

    # if no training end given, default to start year + 3 years
    if trainingEnd == None:
        trainingEnd = trainingStart + 3

    # trim relevent arrays to the user specified timeframe
    # gets indices between starts and end and subset doys, years
    trims = np.flatnonzero((Years >= trainingStart) & (Years < testingEnd))
    DOYs = DOYs[trims]
    Years = Years[trims]
    YearsForAnnualOutput = np.unique(Years)
    myPixel = myPixel[trims]
    breakPointsTracker = breakPointsTracker[trims]

    ### removing missing values and values under the fitting threshold a priori
    dateByDateWithMissing = np.repeat(np.nan, len(myPixel))
    dateByDateResidualsWithMissing = np.repeat(np.nan, len(myPixel))

    # get clean indexes, trim to clean pixel, years, doys, etc
    cleanedInputIndex = np.flatnonzero((~np.isnan(myPixel)) & (myPixel > lowthresh))
    myPixelCleaned = myPixel[cleanedInputIndex]
    YearsCleaned = Years[cleanedInputIndex]
    DOYsCleaned = DOYs[cleanedInputIndex]
    DecimalYearsCleaned = (Years + DOYs / 365)[cleanedInputIndex]
    breakPointsTrackerCleaned = breakPointsTracker[cleanedInputIndex]

    # exit pixel if pixel empty after clean
    if len(myPixelCleaned) == 0:
        output = {
            'dateByDate': np.repeat(np.nan, myPixel),
            'dateByDateResiduals': np.repeat(np.nan, myPixel),
            'Beta': BetaFirst,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }
        return output

    # set min training length if empty (?)
    if minTrainingLength is None:
        minTrainingLength = (1 + numberHarmonicsSine + numberHarmonicsCosine) * 3

    # todo check use of inf... not sure its purpose yet...
    if np.isinf(maxTrainingLength) or np.isnan(maxTrainingLength):
        maxTrainingLength = minTrainingLength * 2

    # calculate persistence
    persistence = np.ceil((len(myPixelCleaned) / len(np.unique(YearsCleaned))) * persistence_per_year)
    persistence = persistence.astype('int16')  # todo added conversion to int, check

    # todo add training period == static
    if trainingPeriod == 'static':
        if minTrainingLength == 0:
            minTrainingLength = 1

        if np.isinf(minTrainingLength):  # todo using inf...
            minTrainingLength = 1

        DecimalYearsCleaned = (YearsCleaned + DOYsCleaned / 365)

        # call optimize hreg
        optimal_outputs = optimize_hreg(DecimalYearsCleaned,
                                        DOYsCleaned,
                                        myPixelCleaned,
                                        trainingFitMinimumQuality,
                                        minTrainingLength,
                                        maxTrainingLength,
                                        ns=1,
                                        nc=1,
                                        screenSigs=xBarLimit1)

        # get bounds, precedents
        historyBound = optimal_outputs.get('historyBound')
        training_precedents = optimal_outputs.get('fitPrevious')

        # combine bp start, tracker, ignore start if empty
        breakPointsStart = np.append(breakPointsStart, breakPointsTrackerCleaned[0])
        breakPointsEnd = np.append(breakPointsEnd, breakPointsTrackerCleaned[historyBound])

        if np.isnan(historyBound):  # todo just check this handles None
            return dateByDateWithMissing

        # call ewmac clean pixel date by date
        tmpOut = EWMACD_clean_pixel_date_by_date(inputPixel=myPixelCleaned,
                                                 numberHarmonicsSine=numberHarmonicsSine,
                                                 numberHarmonicsCosine=numberHarmonicsCosine,
                                                 inputDOYs=DOYsCleaned,
                                                 inputYears=YearsCleaned,
                                                 trainingStart=trainingStart,  # todo added this
                                                 trainingEnd=trainingEnd,  # todo added this
                                                 _lambda=_lambda,
                                                 lambdaSigs=lambdaSigs,
                                                 historyBound=historyBound,
                                                 precedents=training_precedents,
                                                 persistence=persistence)

        # get output values
        runKeeps = tmpOut.get('outputValues')
        runKeepsResiduals = tmpOut.get('residualOutputValues')
        BetaFirst = tmpOut.get('Beta')

    # begin dynamic (Edyn) method
    if trainingPeriod == 'dynamic':
        myPixelCleanedTemp = myPixelCleaned
        YearsCleanedTemp = YearsCleaned
        DOYsCleanedTemp = DOYsCleaned
        DecimalYearsCleanedTemp = (YearsCleanedTemp + DOYsCleanedTemp / 365)
        breakPointsTrackerCleanedTemp = breakPointsTrackerCleaned

        # buckets for edyn outputs
        runKeeps = np.repeat(np.nan, len(myPixelCleaned))
        runKeepsResiduals = np.repeat(np.nan, len(myPixelCleaned))

        # set indexer
        indexer = 0  # todo was 1
        while len(myPixelCleanedTemp) > minTrainingLength and (np.max(DecimalYearsCleanedTemp) - DecimalYearsCleanedTemp[0]) > 1:

            if np.isinf(minTrainingLength): # todo using inf...
                minTrainingLength = 1

            # call optimize hreg
            optimal_outputs = optimize_hreg(DecimalYearsCleanedTemp,
                                            DOYsCleanedTemp,
                                            myPixelCleanedTemp,
                                            trainingFitMinimumQuality,
                                            minTrainingLength,
                                            maxTrainingLength,
                                            ns=1,
                                            nc=1,
                                            screenSigs=xBarLimit1)

            # get bounds, precedents
            historyBound = optimal_outputs.get('historyBound')
            training_precedents = optimal_outputs.get('fitPrevious')

            # combine bp start, tracker, ignore start if empty
            breakPointsStart = np.append(breakPointsStart, breakPointsTrackerCleanedTemp[0])
            breakPointsEnd = np.append(breakPointsEnd, breakPointsTrackerCleanedTemp[historyBound])

            # call ewmac clean pixel date by date
            tmpOut = EWMACD_clean_pixel_date_by_date(inputPixel=myPixelCleanedTemp,
                                                     numberHarmonicsSine=numberHarmonicsSine,
                                                     numberHarmonicsCosine=numberHarmonicsCosine,
                                                     inputDOYs=DOYsCleanedTemp,
                                                     inputYears=YearsCleanedTemp,
                                                     trainingStart=trainingStart,  # todo added this
                                                     trainingEnd=trainingEnd,      # todo added this
                                                     _lambda=_lambda,
                                                     lambdaSigs=lambdaSigs,
                                                     historyBound=historyBound,
                                                     precedents=training_precedents,
                                                     persistence=persistence)
            # get output values
            tmpRun = tmpOut.get('outputValues')
            tmpResiduals = tmpOut.get('residualOutputValues')
            if indexer == 0:
                BetaFirst = tmpOut.get('Beta')

            ## Scratch Work ####------
            # todo move to global method
            def vertex_finder(tsi):
                v1 = tsi[0]
                v2 = tsi[len(tsi) - 1]  # todo added - 1

                res_ind = None
                mse = None
                if np.sum(~np.isnan(tmpRun)) > 1:
                    tmp_mod = scipy.stats.linregress(x=DecimalYearsCleanedTemp[[v1, v2]], y=tmpRun[[v1, v2]])

                    tmp_int = tmp_mod.intercept
                    tmp_slope = tmp_mod.slope

                    tmp_res = tmpRun[tsi] - (tmp_int + tmp_slope * DecimalYearsCleanedTemp[tsi])

                    res_ind = np.argmax(np.abs(tmp_res)) + v1  # todo removed - 1
                    mse = np.sum(tmp_res ** 2)

                # create output
                v_out = {'res_ind': res_ind, 'mse': mse}
                return v_out

            vertices = np.flatnonzero(tmpRun != 0)
            if vertices.size != 0:
                vertices = np.array([np.min(vertices)])  # todo check this works, not fired yet
            else:
                vertices = np.array([historyBound - 1])  # todo added - 1 here

            #time_list = np.arange(vertices[0], len(tmpRun))
            time_list = [np.arange(vertices[0], len(tmpRun), dtype='int16')]  # todo added astype
            #seg_stop = np.prod(np.apply_along_axis(len, 0, time_list) > persistence)  # todo check along axis works in multi dim
            seg_stop = np.prod([len(e) for e in time_list] > persistence)

            vert_indexer = 0
            vert_new = np.array([0])
            while seg_stop == 1 and len(vert_new) >= 1:

                # todo this needs to consider multi dim array
                # todo e.g. for elem in time_list: send to vertex_finder

                # todo for now, do the one dim array
                #vertex_stuff = vertex_finder(tsi=time_list)
                vertex_stuff = [vertex_finder(e) for e in time_list]
                #vertex_stuff = np.array(list(vertex_stuff.values()))
                vertex_stuff = np.array(list(vertex_stuff[0].values())) # todo temp! we dont wanan acess that 0 element like this

                # todo check - started 1, + 1. needed as not indexes
                vertex_mse = vertex_stuff[np.remainder(np.arange(1, len(vertex_stuff) + 1), 2) == 0]
                vertex_ind = vertex_stuff[np.remainder(np.arange(1, len(vertex_stuff) + 1), 2) == 1]

                vert_new = np.flatnonzero(np.prod(abs(vertex_ind - vertices) >= (persistence / 2), axis=0) == 1) # todo apply prod per row

                # todo modified this to handle the above - in r, if array indexed when index doesnt exist, numeric of 0 returned
                if len(vert_new) == 0:
                    vertices = np.unique(np.sort(vertices))
                else:
                    vertices = np.unique(np.sort(np.append(vertices, vertex_ind[vert_new][np.argmax(vertex_mse[vert_new])])))

                # todo this whole thing needs a check, never fired
                if len(vert_new) == 1:
                    #for tl_indexer in np.arange(0, len(vertices)):  # todo check needs - 1
                        #time_list[[tl_indexer]] = np.arange(vertices[tl_indexer], (vertices[tl_indexer + 1] - 1))  # todo check remove - 1?
                    #time_list[[len(vertices)]] = np.arange(vertices[len(vertices)], len(tmpRun))  # todo check

                    for tl_indexer in np.arange(0, len(vertices) - 1):
                        time_list[tl_indexer] = np.arange(vertices[tl_indexer], (vertices[tl_indexer + 1]), dtype='int16')
                    #time_list[len(vertices)] = np.arange(vertices[len(vertices)], len(tmpRun))
                    time_list.append(np.arange(vertices[len(vertices) - 1], len(tmpRun), dtype='int16'))  # todo added - 1 to prevent out of index and append, added astype

                # increase vertex counter
                vert_indexer = vert_indexer + 1

                #seg_stop = np.prod(len(time_list) >= persistence)
                seg_stop = np.prod([len(e) for e in time_list] >= persistence)  # todo check

            # on principle, the second angle should indicate the restabilization!
            if len(vertices) >= 2:
                latestString = np.arange(0, vertices[1] + 1)  # todo added + 1 as we want to include extra index
            else:
                latestString = np.arange(0, len(tmpRun))

            # todo added astype int64 to prevent index float error
            latestString = latestString.astype('int64')

            runStep = np.min(np.flatnonzero(np.isnan(runKeeps)))
            runKeeps[runStep + latestString] = tmpRun[latestString]  # todo check removed - 1 is ok
            runKeepsResiduals[runStep + latestString] = tmpResiduals[latestString]  # todo check removed - 1 is ok

            myPixelCleanedTemp = np.delete(myPixelCleanedTemp, latestString)  # todo check empty array is ok down line
            DOYsCleanedTemp = np.delete(DOYsCleanedTemp, latestString)
            YearsCleanedTemp = np.delete(YearsCleanedTemp, latestString)
            DecimalYearsCleanedTemp = np.delete(DecimalYearsCleanedTemp, latestString)
            breakPointsTrackerCleanedTemp = np.delete(breakPointsTrackerCleanedTemp, latestString)
            indexer = indexer + 1

    # Post-Processing
    # At this point we have a vector of nonmissing EWMACD signals filtered by persistence
    dateByDate = backfill_missing(nonMissing=runKeeps, nonMissingIndex=cleanedInputIndex, withMissing=dateByDateWithMissing)
    dateByDateResiduals = backfill_missing(nonMissing=runKeepsResiduals, nonMissingIndex=cleanedInputIndex, withMissing=dateByDateWithMissing)

    if simple_output == True:
        output = {
            'dateByDate': dateByDate,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }
    else:
        output = {
            'dateByDate': dateByDate,
            'dateByDateResiduals': dateByDateResiduals,
            'Beta': BetaFirst,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }

    return output


def annual_summaries(Values, yearIndex, summaryMethod='date-by-date'):
    """annual summaries"""
    if summaryMethod == 'date-by-date':
        return Values

    finalOutput = np.repeat(np.nan, len(np.unique(yearIndex)))

    if summaryMethod == 'mean':
        # todo mean method, median, extreme, signmed mean methods... do when happy with above
        #finalOutput = (np.round(aggregate(Values, by=list(yearIndex), FUN=mean, na.rm = T)))$x
        ...


def EWMACD(ds, trainingPeriod='dynamic', trainingStart=None, testingEnd=None, trainingEnd=None, minTrainingLength=None, maxTrainingLength=np.inf, trainingFitMinimumQuality=0.8, numberHarmonicsSine=2, numberHarmonicsCosine='same as Sine', xBarLimit1=1.5, xBarLimit2= 20, lowthresh=0, _lambda=0.3, lambdaSigs=3, rounding=True, persistence_per_year=1, reverseOrder=False, summaryMethod='date-by-date', outputType='chart.values'):
    """main function"""

    # notify
    #

    # get day of years and associated year as int 16
    DOYs = ds['time.dayofyear'].data.astype('int16')
    Years = ds['time.year'].data.astype('int16')

    # check doys, years
    if len(DOYs) != len(Years):
        raise ValueError('DOYs and Years are not same length.')

    # if no training date provided, choose first year
    if trainingStart is None:
        trainingStart = np.min(Years)

    # if no testing date provided, choose last year + 1
    if testingEnd is None:
        testingEnd = np.max(Years) + 1

    # generate array of nans for every year between start of train and test period
    NAvector = np.repeat(np.nan, len(Years[(Years >= trainingStart) & (Years < testingEnd)]))

    # if not date to date, use year to year (?) may not need this
    if summaryMethod != 'date-by-date':
        num_nans = len(np.unique(Years[(Years >= trainingStart) & (Years < testingEnd)]))
        NAvector = np.repeat(np.nan, num_nans)

    # set cos harmonics value (default 2) to same as sine, if requested
    if numberHarmonicsCosine == 'same as Sine':
        numberHarmonicsCosine = numberHarmonicsSine

    # set simple output if chart values requested (?)
    if outputType == 'chart.values':
        simple_output = True

    # create per-pixel vectorised version of ewmacd per-pixel func
    def map_ewmacd_to_xr(pixel):
        
        try:
            change = EWMACD_pixel_date_by_date(myPixel=pixel,
                                               DOYs=DOYs,
                                               Years=Years,
                                               _lambda=_lambda,
                                               numberHarmonicsSine=numberHarmonicsSine,
                                               numberHarmonicsCosine=numberHarmonicsCosine,
                                               trainingStart=trainingStart,
                                               testingEnd=testingEnd,
                                               trainingPeriod=trainingPeriod,
                                               trainingEnd=trainingEnd,
                                               minTrainingLength=minTrainingLength,
                                               maxTrainingLength=maxTrainingLength,
                                               trainingFitMinimumQuality=trainingFitMinimumQuality,
                                               xBarLimit1=xBarLimit1,
                                               xBarLimit2=xBarLimit2,
                                               lowthresh=lowthresh,
                                               lambdaSigs=lambdaSigs,
                                               rounding=rounding,
                                               persistence_per_year=persistence_per_year,
                                               reverseOrder=reverseOrder,
                                               simple_output=simple_output)

            # get change per date from above
            change = change.get('dateByDate')

            # calculate summary method (todo set up for others than just date to date
            final_out = annual_summaries(Values=change,
                                         yearIndex=Years,
                                         summaryMethod=summaryMethod)

        except Exception as e:
            print('Could not train model adequately, please add more years.')
            print(e)
            final_out = NAvector

        #return final_out
        return final_out
        

    # map ewmacd func to ds and compute it
    ds = xr.apply_ufunc(map_ewmacd_to_xr,
                        ds,
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                        vectorize=True)
    
    # rename veg_idx to change and convert to float32
    ds = ds.astype('float32')
    
    #return dataset
    return ds









