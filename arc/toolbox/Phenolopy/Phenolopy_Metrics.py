import arcpy
import os, sys
import datetime
import numpy as np
import tempfile
import xarray as xr
import dask

from shared import arc, tools, satfetcher
from modules import cog, phenolopy

from arc.toolbox.globals import GRP_LYR_FILE

class Phenolopy_Metrics:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'Phenolopy Metrics'
        self.description = 'Calculate various metrics that describe various. ' \
                           'aspects of vegetation phenology from a data cube. ' \
                           'Key metrics include Peak of Season (POS), Start and ' \
                           'End of Season (SOS, EOS), and various productivity metrics.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input netcdf file
        par_raw_nc_path = arcpy.Parameter(
            displayName='Input satellite NetCDF file',
            name='in_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_raw_nc_path.filter.list = ['nc']

        # output netcdf file
        par_out_nc_path = arcpy.Parameter(
            displayName='Output Phenometrics NetCDF file',
            name='out_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_path.filter.list = ['nc']

        # use all dates
        par_use_all_dates = arcpy.Parameter(
            displayName='Combine all input dates',
            name='in_use_all_dates',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_use_all_dates.value = True

        # set specific year
        par_specific_years = arcpy.Parameter(
            displayName='Specific year(s) to analyse',
            name='in_specific_years',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=True)
        par_specific_years.filter.type = 'ValueList'
        par_specific_years.filter.list = []
        par_specific_years.value = None

        # input metrics
        par_metrics = arcpy.Parameter(
            displayName='Phenological metrics',
            name='in_metrics',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=True)
        metrics = [
            'POS: Peak of season',
            'VOS: Valley of season',
            'BSE: Base of season',
            'MOS: Middle of season',
            'AOS: Amplitude of season',
            'SOS: Start of season',
            'EOS: End of season',
            'LOS: Length of season',
            'ROI: Rate of increase',
            'ROD: Rate of decrease',
            'SIOS: Short integral of season',
            'LIOS: Long integral of season',
            'SIOT: Short integral of total',
            'LIOT: Long integral of total',
            'NOS: Number of seasons'
        ]
        par_metrics.filter.type = 'ValueList'
        par_metrics.filter.list = metrics
        remove = [
            'MOS: Middle of season',
            'BSE: Base of season',
            'AOS: Amplitude of season',
            'NOS: Number of seasons'
        ]
        par_metrics.values = [m for m in metrics if m not in remove]

        # input vegetation index
        par_veg_idx = arcpy.Parameter(
            displayName='Vegetation index',
            name='in_veg_idx',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_veg_idx.filter.type = 'ValueList'
        par_veg_idx.filter.list = [
            'NDVI',
            'EVI',
            'SAVI',
            'MSAVI',
            'SLAVI',
            'MAVI',
            'kNDVI'
        ]
        par_veg_idx.value = 'MAVI'

        # input method type
        par_method_type = arcpy.Parameter(
            displayName='Season detection method',
            name='in_method_type',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Season Detection',
            multiValue=False)
        par_method_type.filter.list = [
            'First of slope',
            'Mean of slope',
            'Seasonal amplitude',
            'Absolute amplitude',
            'Relative amplitude'
        ]
        par_method_type.values = 'Seasonal amplitude'

        # input amplitude factor (seaamp, relamp)
        par_amp_factor = arcpy.Parameter(
            displayName='Amplitude factor',
            name='in_amp_factor',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            category='Season Detection',
            multiValue=False)
        par_amp_factor.filter.type = 'Range'
        par_amp_factor.filter.list = [0.0, 1.0]
        par_amp_factor.value = 0.5

        # input absolute value (absamp)
        par_abs_value = arcpy.Parameter(
            displayName='Absolute value',
            name='in_abs_value',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            category='Season Detection',
            multiValue=False)
        par_abs_value.value = 0.3

        # input savitsky window length
        par_sav_win_length = arcpy.Parameter(
            displayName='Window size',
            name='in_sav_win_length',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Smoothing',
            multiValue=False)
        par_sav_win_length.filter.type = 'Range'
        par_sav_win_length.filter.list = [3, 99]
        par_sav_win_length.value = 3

        # input polyorder
        par_sav_polyorder = arcpy.Parameter(
            displayName='Polyorder',
            name='in_sav_polyorder',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Smoothing',
            multiValue=False)
        par_sav_polyorder.filter.type = 'Range'
        par_sav_polyorder.filter.list = [1, 100]
        par_sav_polyorder.value = 1

        # input outlier window length
        par_out_win_length = arcpy.Parameter(
            displayName='Window size',
            name='in_out_win_length',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Outlier Correction',
            multiValue=False)
        par_out_win_length.filter.type = 'Range'
        par_out_win_length.filter.list = [3, 99]
        par_out_win_length.value = 3

        # input outlier factor
        par_out_factor = arcpy.Parameter(
            displayName='Outlier removal factor',
            name='in_out_factor',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Outlier Correction',
            multiValue=False)
        par_out_factor.filter.type = 'Range'
        par_out_factor.filter.list = [1, 100]
        par_out_factor.value = 2

        # fix edge dates
        par_fix_edges = arcpy.Parameter(
            displayName='Ignore edge dates',
            name='in_fix_edges',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Outlier Correction',
            multiValue=False)
        par_fix_edges.value = True

        # fill empty pixels
        par_fill_nans = arcpy.Parameter(
            displayName='Fill erroroneous pixels',
            name='in_fill_nans',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Outlier Correction',
            multiValue=False)
        par_fill_nans.value = True

        # input oa fmask
        par_fmask_flags = arcpy.Parameter(
            displayName='Include flags',
            name='in_fmask_flags',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Satellite Quality Options',
            multiValue=True)
        flags = ['NoData', 'Valid', 'Cloud', 'Shadow', 'Snow', 'Water']
        par_fmask_flags.filter.type = 'ValueList'
        par_fmask_flags.filter.list = flags
        par_fmask_flags.values = ['Valid', 'Snow', 'Water']

        # input max cloud cover
        par_max_cloud = arcpy.Parameter(
            displayName='Maximum cloud cover',
            name='in_max_cloud',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            category='Satellite Quality Options',
            multiValue=False)
        par_max_cloud.filter.type = 'Range'
        par_max_cloud.filter.list = [0.0, 100.0]
        par_max_cloud.value = 10.0

        # input add result to map
        par_add_result_to_map = arcpy.Parameter(
            displayName='Add result to map',
            name='in_add_result_to_map',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Outputs',
            multiValue=False)
        par_add_result_to_map.value = True

        # combine parameters
        parameters = [
            par_raw_nc_path,
            par_out_nc_path,
            par_use_all_dates,
            par_specific_years,
            par_metrics,
            par_veg_idx,
            par_method_type,
            par_amp_factor,
            par_abs_value,
            par_sav_win_length,
            par_sav_polyorder,
            par_out_win_length,
            par_out_factor,
            par_fix_edges,
            par_fill_nans,
            par_fmask_flags,
            par_max_cloud,
            par_add_result_to_map
        ]

        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """
        Enable and disable certain parameters when
        controls are changed on ArcGIS Pro panel.
        """

        # imports
        try:
            import numpy as np
            import xarray as xr
        except:
            arcpy.AddError('Python libraries xarray not installed.')
            return

        # globals
        global PHENOLOPY_METRICS

        # unpack global parameter values
        curr_file = PHENOLOPY_METRICS.get('in_file')

        # if input file added, run
        if parameters[0].value is not None:

            # if global has no matching file (or first run), reload all
            if curr_file != parameters[0].valueAsText:
                try:
                    ds = xr.open_dataset(parameters[0].valueAsText)
                    dts = np.unique(ds['time.year']).tolist()
                    ds.close()
                except:
                    dts = []

                # populate years list
                parameters[3].filter.list = dts

                # select last year
                if len(dts) != 0:
                    parameters[3].value = dts[-1]
                else:
                    parameters[3].value = None

                    # update global values
        PHENOLOPY_METRICS = {'in_file': parameters[0].valueAsText}

        # enable year selector based on combine input checkbox
        if parameters[2].value is False:
            parameters[3].enabled = True
        else:
            parameters[3].enabled = False

        # enable amp factor or abs value when methods selected
        if parameters[6].valueAsText in ['Seasonal amplitude', 'Relative amplitude']:
            parameters[7].enabled = True
            parameters[8].enabled = False
        elif parameters[6].valueAsText == 'Absolute amplitude':
            parameters[8].enabled = True
            parameters[7].enabled = False
        else:
            parameters[7].enabled = False
            parameters[8].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the Phenolopy Metrics module.
        """

        # safe imports

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values
        in_nc = parameters[0].valueAsText  # raw input satellite netcdf
        out_nc = parameters[1].valueAsText  # output phenometrics netcdf
        in_use_all_dates = parameters[2].value  # use all dates in nc
        in_specific_years = parameters[3].valueAsText  # set specific year
        in_metrics = parameters[4].valueAsText  # phenometrics
        in_veg_idx = parameters[5].value  # vege index name
        in_method_type = parameters[6].value  # phenolopy method type
        in_amp_factor = parameters[7].value  # amplitude factor
        in_abs_value = parameters[8].value  # absolute value
        in_sav_window_length = parameters[9].value  # savitsky window length
        in_sav_polyorder = parameters[10].value  # savitsky polyorder
        in_out_window_length = parameters[11].value  # outlier window length
        in_out_factor = parameters[12].value  # outlier cutoff user factor
        in_fix_edges = parameters[13].value  # fix edge dates
        in_fill_nans = parameters[14].value  # fill nans
        in_fmask_flags = parameters[15].valueAsText  # fmask flag values
        in_max_cloud = parameters[16].value  # max cloud percentage
        in_add_result_to_map = parameters[17].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Phenolopy Metrics.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=23)

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Loading and checking netcdf...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of netcdf for checking
            ds = xr.open_dataset(in_nc)
        except Exception as e:
            arcpy.AddError('Could not quick load input NetCDF data.')
            arcpy.AddMessage(str(e))
            return

            # check xr type, vars, coords, dims, attrs
        if not isinstance(ds, xr.Dataset):
            arcpy.AddError('Input NetCDF must be a xr dataset.')
            return
        elif len(ds) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds.dims or 'y' not in ds.dims or 'time' not in ds.dims:
            arcpy.AddError('Input NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds.coords or 'y' not in ds.coords or 'time' not in ds.coords:
            arcpy.AddError('Input NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds.coords:
            arcpy.AddError('Input NetCDF must have a spatial_ref coord.')
            return
        elif len(ds['x']) == 0 or len(ds['y']) == 0 or len(ds['time']) == 0:
            arcpy.AddError('Input NetCDF must have all at least one x, y and time index.')
            return
        elif 'oa_fmask' not in ds and 'fmask' not in ds:
            arcpy.AddError('Expected cloud mask band not found in NetCDF.')
            return
        elif not hasattr(ds, 'time.year') or not hasattr(ds, 'time.month'):
            arcpy.AddError('Input NetCDF must have time with year and month component.')
            return
        elif ds.attrs == {}:
            arcpy.AddError('NetCDF must have attributes.')
            return
        elif not hasattr(ds, 'crs'):
            arcpy.AddError('NetCDF CRS attribute not found. CRS required.')
            return
        elif ds.crs != 'EPSG:3577':
            arcpy.AddError('NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
            return
        elif not hasattr(ds, 'nodatavals'):
            arcpy.AddError('NetCDF nodatavals attribute not found.')
            return

            # efficient: if all nan, 0 at first var, assume rest same, so abort
        if ds[list(ds)[0]].isnull().all() or (ds[list(ds)[0]] == 0).all():
            arcpy.AddError('NetCDF has empty variables. Please download again.')
            return

        try:
            # now, do proper open of netcdf properly (and set nodata to nan)
            ds = satfetcher.load_local_nc(nc_path=in_nc,
                                          use_dask=True,
                                          conform_nodata_to=np.nan)
        except Exception as e:
            arcpy.AddError('Could not properly load input NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Getting NetCDF attributes...')
        arcpy.SetProgressorPosition(2)

        # get attributes from dataset
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Grouping dates, if required...')
        arcpy.SetProgressorPosition(3)

        # remove potential datetime duplicates (group by day)
        ds = satfetcher.group_by_solar_day(ds)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Removing invalid pixels and empty dates...')
        arcpy.SetProgressorPosition(4)

        # convert fmask as text to numeric code equivalents
        in_fmask_flags = [e for e in in_fmask_flags.split(';')]
        in_fmask_flags = arc.convert_fmask_codes(in_fmask_flags)

        # check if flags selected, if not, select all
        if len(in_fmask_flags) == 0:
            arcpy.AddWarning('No flags selected, using default.')
            in_fmask_flags = [1, 4, 5]

        # check numeric flags are valid
        for flag in in_fmask_flags:
            if flag not in [0, 1, 2, 3, 4, 5]:
                arcpy.AddError('Pixel flag not supported.')
                return

        # check for duplicate flags
        u, c = np.unique(in_fmask_flags, return_counts=True)
        if len(u[c > 1]) > 0:
            arcpy.AddError('Duplicate pixel flags detected.')
            return

        # get name of mask band
        mask_band = arc.get_name_of_mask_band(list(ds))

        try:
            # remove invalid pixels and empty scenes
            ds = cog.remove_fmask_dates(ds=ds,
                                        valid_class=in_fmask_flags,
                                        max_invalid=in_max_cloud,
                                        mask_band=mask_band,
                                        nodata_value=np.nan,
                                        drop_fmask=True)
        except Exception as e:
            arcpy.AddError('Could not mask pixels.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Conforming satellite band names...')
        arcpy.SetProgressorPosition(5)

        try:
            # get platform name from attributes, error if no attributes
            in_platform = arc.get_platform_from_dea_attrs(ds_attrs)

            # conform dea aws band names based on platform
            ds = satfetcher.conform_dea_ard_band_names(ds=ds,
                                                       platform=in_platform.lower())
        except Exception as e:
            arcpy.AddError('Could not get platform from attributes.')
            arcpy.AddMessage(str(e))
            return

        # check if all expected bands are in dataset
        for band in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
            if band not in ds:
                arcpy.AddError('NetCDF is missing band: {}. Need all bands.'.format(band))
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating vegetation index...')
        arcpy.SetProgressorPosition(6)

        # check if veg idx supported
        if in_veg_idx.lower() not in ['ndvi', 'evi', 'savi', 'msavi', 'slavi', 'mavi', 'kndvi']:
            arcpy.AddError('Vegetation index not supported.')
            return

        try:
            # calculate vegetation index
            ds = tools.calculate_indices(ds=ds,
                                         index=in_veg_idx.lower(),
                                         custom_name='veg_idx',
                                         rescale=False,
                                         drop=True)

            # add band attrs back on
            ds['veg_idx'].attrs = ds_band_attrs

        except Exception as e:
            arcpy.AddError('Could not calculate vegetation index.')
            arcpy.AddMessage(str(e))
            return

        # check if we sufficient data temporally
        if len(ds['time']) == 0:
            arcpy.AddError('Insufficient number of dates in data.')
            return
        elif len(ds['time'].groupby('time.season')) < 3:
            arcpy.AddError('Insufficient number of seasons in data.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Correcting edge dates...')
        arcpy.SetProgressorPosition(7)

        # check if user years is valid
        if in_use_all_dates is None:
            arcpy.AddError('Did not specify combine dates parameter.')
            return
        elif in_use_all_dates is False and in_specific_years is None:
            arcpy.AddError('Did not provide a specific year(s).')
            return

        # get list of years, else empty list
        if in_use_all_dates is False:
            in_specific_years = [int(e) for e in in_specific_years.split(';')]
        else:
            in_specific_years = None

        # check specific years for issues, if specific years exist
        if in_specific_years is not None:
            if datetime.datetime.now().year == max(in_specific_years):
                arcpy.AddError('Cannot use current year, insufficient data.')
                return
            elif 2011 in in_specific_years or 2012 in in_specific_years:
                arcpy.AddError('Cannot use years 2011 or 2012, insufficient data.')
                return

            # check if requested years in dataset
            for year in in_specific_years:
                if year not in ds['time.year']:
                    arcpy.AddError('Year {} was not found in dataset.'.format(year))
                    return

        try:
            # enforce first/last date is 1jan/31dec for user years (or all)
            ds = phenolopy.enforce_edge_dates(ds=ds,
                                              years=in_specific_years)
        except Exception as e:
            arcpy.AddError('Could not correct edge dates.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Subsetting time-series with buffer dates...')
        arcpy.SetProgressorPosition(8)

        try:
            # subset to requested years (or all) with buffer dates (no subset if no years)
            ds = phenolopy.subset_via_years_with_buffers(ds=ds,
                                                         years=in_specific_years)
        except Exception as e:
            arcpy.AddError('Could not subset data with buffer dates.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Resampling time-series to equal-spacing...')
        arcpy.SetProgressorPosition(9)

        try:
            # resample to fortnight medians to ensure equal-spacing
            ds = phenolopy.resample(ds=ds,
                                    interval='SMS')
        except Exception as e:
            arcpy.AddError('Could not resample to equal-spacing.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating initial resample...')
        arcpy.SetProgressorPosition(10)

        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not interpolate initial resample.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Removing spike outliers...')
        arcpy.SetProgressorPosition(11)

        # check window length and factor is valid
        if in_out_window_length < 3 or in_out_window_length > 99:
            arcpy.AddError('Outlier window size must be between 3 and 99.')
            return
        elif in_out_factor < 1 or in_out_factor > 99:
            arcpy.AddError('Outlier factor must be between 1 and 99.')
            return

        try:
            #  remove outliers from data using requeted method
            ds = phenolopy.remove_spikes(ds=ds,
                                         user_factor=in_out_factor,
                                         win_size=in_out_window_length)
        except Exception as e:
            arcpy.AddError('Could not remove spike outliers.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating removed outliers...')
        arcpy.SetProgressorPosition(12)

        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not interpolate removed outliers.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Smoothing time-series...')
        arcpy.SetProgressorPosition(13)

        # check window length
        if in_sav_window_length <= 0 or in_sav_window_length % 2 == 0:
            arcpy.AddWarning('Savitsky window length incompatible, using default.')
            in_sav_window_length = 3

        # check polyorder
        if in_sav_polyorder >= in_sav_window_length:
            arcpy.AddWarning('Savitsky polyorder must be < window length, reducing by one.')
            in_sav_polyorder = in_sav_window_length - 1

        try:
            # smooth dataset across time via savitsky
            ds = phenolopy.smooth(ds=ds,
                                  var='veg_idx',
                                  window_length=in_sav_window_length,
                                  polyorder=in_sav_polyorder)
        except Exception as e:
            arcpy.AddError('Could not smooth data.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Subsetting data to specific years, if requested...')
        arcpy.SetProgressorPosition(14)

        try:
            # subset to requested years, if none, returns input dataset
            ds = phenolopy.subset_via_years(ds=ds,
                                            years=in_specific_years)
        except Exception as e:
            arcpy.AddError('Could not subset to specific year(s).')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Grouping time-series by dates...')
        arcpy.SetProgressorPosition(15)

        try:
            # group years by m-d (1-1, 1-15, 2-1, 2-15, etc)
            ds = phenolopy.group(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not group by dates.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Resampling high resolution curves...')
        arcpy.SetProgressorPosition(16)

        try:
            # resample to 365 days per pixel for higher accuracy metrics
            ds = phenolopy.resample(ds=ds,
                                    interval='1D')
        except Exception as e:
            arcpy.AddError('Could not interpolate high-resolution curves.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating high-resolution curves...')
        arcpy.SetProgressorPosition(17)

        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not interpolate high-resolution curves.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Checking and removing date oversamples...')
        arcpy.SetProgressorPosition(18)

        try:
            # remove potential oversample dates
            ds = phenolopy.drop_overshoot_dates(ds=ds,
                                                min_dates=3)
        except Exception as e:
            arcpy.AddError('Could not remove oversample dates.')
            arcpy.AddMessage(str(e))
            return

        # check if we have 365 days, otherwise warn
        if len(ds['time']) != 365:
            arcpy.AddWarning('Could not create 365-day time-series, output may have errors.')

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(19)

        try:
            # compute!
            ds = ds.compute()
        except Exception as e:
            arcpy.AddError('Could not compute dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('NetCDF is empty. Please download again.')
            return

            # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Calculating phenology metrics...')

        # check if metrics valid
        if in_metrics is None:
            arcpy.AddError('No metrics were selected.')
            return

        # remove single quotes in metric string (due to spaces) and split
        in_metrics = in_metrics.lower().replace("'", '').split(';')
        in_metrics = [e.split(':')[0] for e in in_metrics]

        # convert method to compatible name
        in_method_type = in_method_type.lower()
        in_method_type = in_method_type.replace(' ', '_')

        # check amplitude factors, absolute values
        if in_method_type in ['seasonal_amplitude', 'relative_amplitude']:
            if in_amp_factor is None:
                arcpy.AddError('Must provide an amplitude factor.')
                return
            elif in_amp_factor < 0 or in_amp_factor > 1:
                arcpy.AddError('Amplitude factor must be between 0 and 1.')
                return

        elif in_method_type == 'absolute_amplitude':
            if in_abs_value is None:
                arcpy.AddError('Must provide an absolute value (any value).')
                return

        # check if fix edge dates and fill pixels set
        if in_fix_edges is None:
            arcpy.AddError('Must set the ignore edge dates checkbox.')
            return
        elif in_fill_nans is None:
            arcpy.AddError('Must set the fill erroroneous pixels checkbox.')
            return

        try:
            # calculate phenometrics!
            ds = phenolopy.get_phenometrics(ds=ds,
                                            metrics=in_metrics,
                                            method=in_method_type,
                                            factor=in_amp_factor,
                                            abs_value=in_abs_value,
                                            peak_spacing=12,
                                            fix_edges=in_fix_edges,
                                            fill_nan=in_fill_nans)
        except Exception as e:
            arcpy.AddError('Could not calculate phenometrics.')
            arcpy.AddMessage(str(e))
            return

        # check if any data was returned
        if ds.to_array().isnull().all():
            arcpy.AddError('Metric output contains no values.')
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(20)

        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(21)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(22)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing ensemble layers if exist
                for layer in m.listLayers():
                    if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'metrics':
                        m.removeLayer(layer)

                # setup a group layer via template
                grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
                grp = m.addLayer(grp_lyr)[0]
                grp.name = 'metrics'

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'metrics' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # iter each var and export a seperate tif
                tif_list = []
                for var in ds:
                    # create temp netcdf for one var (prevents 2.9 bug)
                    with tempfile.NamedTemporaryFile() as tmp:
                        tmp_nc = '{}_{}.nc'.format(tmp.name, var)
                        ds[[var]].to_netcdf(tmp_nc)

                    # build in-memory crf for temp netcdf
                    crf = arcpy.md.MakeMultidimensionalRasterLayer(in_multidimensional_raster=tmp_nc,
                                                                   out_multidimensional_raster_layer=var)

                    # export temp tif
                    tmp_tif = os.path.join(out_folder, '{}.tif'.format(var))
                    tif = arcpy.management.CopyRaster(in_raster=crf,
                                                      out_rasterdataset=tmp_tif)

                    # add temp tif to map and get as layer
                    m.addDataFromPath(tif)
                    layer = m.listLayers('{}.tif'.format(var))[0]

                    # hide layer once added
                    # layer.visible = False

                    # add layer to group and then remove outside layer
                    m.addLayerToGroup(grp, layer, 'BOTTOM')
                    m.removeLayer(layer)

                    # success, add store current layer for symbology below
                    tif_list.append('{}.tif'.format(var))

            except Exception as e:
                arcpy.AddWarning('Could not visualise output, aborting visualisation.')
                arcpy.AddMessage(str(e))
                pass

            try:
                # iter tif layer names and update symbology
                for tif in tif_list:
                    layer = m.listLayers(tif)[0]
                    sym = layer.symbology

                    # if layer has stretch coloriser, apply color
                    if hasattr(sym, 'colorizer'):
                        if sym.colorizer.type == 'RasterStretchColorizer':

                            # apply percent clip type
                            sym.colorizer.stretchType = 'PercentClip'

                            # colorize depending on metric
                            if 'roi' in tif or 'rod' in tif:
                                sym.colorizer.minPercent = 1.0
                                sym.colorizer.maxPercent = 1.0
                                cmap = aprx.listColorRamps('Inferno')[0]
                            elif 'aos' in tif or 'los' in tif:
                                sym.colorizer.minPercent = 0.5
                                sym.colorizer.maxPercent = 0.5
                                cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]
                            elif 'nos' in tif:
                                sym.colorizer.stretchType = 'MinimumMaximum'
                                cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]
                            elif 'times' in tif:
                                sym.colorizer.minPercent = 0.25
                                sym.colorizer.maxPercent = 0.25
                                cmap = aprx.listColorRamps('Temperature')[0]
                            elif 'values' in tif:
                                sym.colorizer.minPercent = 1.0
                                sym.colorizer.maxPercent = 1.0
                                cmap = aprx.listColorRamps('Precipitation')[0]

                                # apply color map
                            sym.colorizer.colorRamp = cmap

                            # apply other basic options
                            sym.colorizer.invertColorRamp = False
                            sym.colorizer.gamma = 1.0

                            # update symbology
                            layer.symbology = sym

                            # show layer
                            # layer.visible = True

            except Exception as e:
                arcpy.AddWarning('Could not colorise output, aborting colorisation.')
                arcpy.AddMessage(str(e))
                pass

        # # # # #
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(23)

        # close and del dataset
        ds.close()
        del ds

        # notify user
        arcpy.AddMessage('Generated Phenometrics successfully.')

        return
