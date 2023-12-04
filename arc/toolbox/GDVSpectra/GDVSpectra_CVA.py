import warnings
import arcpy
import os
import datetime
import numpy as np
import xarray as xr
import dask

from modules import gdvspectra, cog
from shared import satfetcher, tools, arc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
class GDVSpectra_CVA:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'GDVSpectra CVA'
        self.description = 'Perform a Change Vector Analysis (CVA) on a data cube.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input netcdf file
        par_raw_nc_path = arcpy.Parameter(
            displayName='Input satellite NetCDF file',
            name='in_raw_nc_path',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_raw_nc_path.filter.list = ['nc']

        # input netcdf mask (thresh) file
        par_mask_nc_path = arcpy.Parameter(
            displayName='Input GDV Threshold mask NetCDF file',
            name='in_mask_nc_path',
            datatype='DEFile',
            parameterType='Optional',
            direction='Input')
        par_mask_nc_path.filter.list = ['nc']

        # output folder location
        par_out_nc_path = arcpy.Parameter(
            displayName='Output CVA NetCDF file',
            name='out_nc_path',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_path.filter.list = ['nc']

        # base start year
        par_base_start_year = arcpy.Parameter(
            displayName='Baseline start year',
            name='in_base_start_year',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_base_start_year.filter.type = 'ValueList'
        par_base_start_year.filter.list = []

        # base end year
        par_base_end_year = arcpy.Parameter(
            displayName='Baseline end year',
            name='in_base_end_year',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_base_end_year.filter.type = 'ValueList'
        par_base_end_year.filter.list = []

        # comparison start year
        par_comp_start_year = arcpy.Parameter(
            displayName='Comparison start year',
            name='in_comp_start_year',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_comp_start_year.filter.type = 'ValueList'
        par_comp_start_year.filter.list = []

        # comparison end year
        par_comp_end_year = arcpy.Parameter(
            displayName='Comparison end year',
            name='in_comp_end_year',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_comp_end_year.filter.type = 'ValueList'
        par_comp_end_year.filter.list = []

        # analysis months
        par_analysis_months = arcpy.Parameter(
            displayName='Set analysis month(s)',
            name='in_analysis_months',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=True)
        par_analysis_months.filter.type = 'ValueList'
        par_analysis_months.filter.list = [m for m in range(1, 13)]
        par_analysis_months.value = [9, 10, 11]

        # cva magnitude threshold
        par_tmf = arcpy.Parameter(
            displayName='Magnitude threshold',
            name='in_tmf',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_tmf.filter.type = 'Range'
        par_tmf.filter.list = [0.0, 100.0]
        par_tmf.value = 2.0

        # set q upper for standardisation
        par_ivt_qupper = arcpy.Parameter(
            displayName='Upper percentile',
            name='in_stand_qupper',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            category='Invariant Standardisation',
            multiValue=False)
        par_ivt_qupper.filter.type = 'Range'
        par_ivt_qupper.filter.list = [0.0, 1.0]
        par_ivt_qupper.value = 0.99

        # set q lower for standardisation
        par_ivt_qlower = arcpy.Parameter(
            displayName='Lower percentile',
            name='in_stand_qlower',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            category='Invariant Standardisation',
            multiValue=False)
        par_ivt_qlower.filter.type = 'Range'
        par_ivt_qlower.filter.list = [0.0, 1.0]
        par_ivt_qlower.value = 0.05

        # set oa class values
        par_fmask_flags = arcpy.Parameter(displayName='Include pixel flags',
                                          name='in_fmask_flags',
                                          datatype='GPString',
                                          parameterType='Required',
                                          direction='Input',
                                          category='Satellite Quality Options',
                                          multiValue=True)
        flags = [
            'NoData',
            'Valid',
            'Cloud',
            'Shadow',
            'Snow',
            'Water'
        ]
        par_fmask_flags.filter.type = 'ValueList'
        par_fmask_flags.filter.list = flags
        par_fmask_flags.values = ['Valid', 'Snow', 'Water']

        # max cloud cover
        par_max_cloud = arcpy.Parameter(
            displayName='Maximum cloud cover',
            name='in_max_cloud',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            category='Satellite Quality Options',
            multiValue=False)
        par_max_cloud.filter.type = 'Range'
        par_max_cloud.filter.list = [0.0, 100.0]
        par_max_cloud.value = 10.0

        # interpolate
        par_interpolate = arcpy.Parameter(
            displayName='Interpolate NoData pixels',
            name='in_interpolate',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Satellite Quality Options',
            multiValue=False)
        par_interpolate.value = True

        # add result to map
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
            par_mask_nc_path,
            par_out_nc_path,
            par_base_start_year,
            par_base_end_year,
            par_comp_start_year,
            par_comp_end_year,
            par_analysis_months,
            par_tmf,
            par_ivt_qupper,
            par_ivt_qlower,
            par_fmask_flags,
            par_max_cloud,
            par_interpolate,
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

        # globals
        global GDVSPECTRA_CVA

        # unpack global parameter values
        curr_file = GDVSPECTRA_CVA.get('in_file')

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

                # populate baseline start and end year lists
                parameters[3].filter.list = dts
                parameters[4].filter.list = dts

                # populate comparison start and end year lists
                parameters[5].filter.list = dts
                parameters[6].filter.list = dts

                # reset baseline start and end year selections
                if len(dts) != 0:
                    parameters[3].value = dts[0]
                    parameters[4].value = dts[0]
                    parameters[5].value = dts[-1]
                    parameters[6].value = dts[-1]
                else:
                    parameters[3].value = None
                    parameters[4].value = None
                    parameters[5].value = None
                    parameters[6].value = None

        # update global values
        GDVSPECTRA_CVA = {
            'in_file': parameters[0].valueAsText,
        }

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra CVA module.
        """




        # grab parameter values
        in_raw_nc = parameters[0].valueAsText  # raw input satellite netcdf
        in_mask_nc = parameters[1].valueAsText  # mask input satellite netcdf
        out_nc = parameters[2].valueAsText  # output gdv likelihood netcdf
        in_base_start_year = parameters[3].value  # base start year
        in_base_end_year = parameters[4].value  # base end year
        in_comp_start_year = parameters[5].value  # comp start year
        in_comp_end_year = parameters[6].value  # comp end year
        in_analysis_months = parameters[7].valueAsText  # analysis months
        in_tmf = parameters[8].value  # magnitude threshold
        in_ivt_qupper = parameters[9].value  # upper quantile for standardisation
        in_ivt_qlower = parameters[10].value  # lower quantile for standardisation
        in_fmask_flags = parameters[11].valueAsText  # fmask flag values
        in_max_cloud = parameters[12].value  # max cloud percentage
        in_interpolate = parameters[13].value  # interpolate missing pixels
        in_add_result_to_map = parameters[14].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra CVA.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=18)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking satellite netcdf...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of satellite netcdf for checking
            ds = xr.open_dataset(in_raw_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input satellite NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds, xr.Dataset):
            arcpy.AddError('Input satellite NetCDF must be a xr dataset.')
            return
        elif len(ds) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds.dims or 'y' not in ds.dims or 'time' not in ds.dims:
            arcpy.AddError('Input satellite NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds.coords or 'y' not in ds.coords or 'time' not in ds.coords:
            arcpy.AddError('Input satellite NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds.coords:
            arcpy.AddError('Input satellite NetCDF must have a spatial_ref coord.')
            return
        elif len(ds['x']) == 0 or len(ds['y']) == 0 or len(ds['time']) == 0:
            arcpy.AddError('Input satellite NetCDF must have all at least one x, y and time index.')
            return
        elif 'oa_fmask' not in ds and 'fmask' not in ds:
            arcpy.AddError('Expected cloud mask band not found in satellite NetCDF.')
            return
        elif not hasattr(ds, 'time.year') or not hasattr(ds, 'time.month'):
            arcpy.AddError('Input satellite NetCDF must have time with year and month component.')
            return
        elif len(ds.groupby('time.year')) < 2:
            arcpy.AddError('Input satellite NetCDF must have >= 2 years of data.')
            return
        elif ds.attrs == {}:
            arcpy.AddError('Satellite NetCDF must have attributes.')
            return
        elif not hasattr(ds, 'crs'):
            arcpy.AddError('Satellite NetCDF CRS attribute not found. CRS required.')
            return
        elif ds.crs != 'EPSG:3577':
            arcpy.AddError('Satellite NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
            return
        elif not hasattr(ds, 'nodatavals'):
            arcpy.AddError('Satellite NetCDF nodatavals attribute not found.')
            return

            # efficient: if all nan, 0 at first var, assume rest same, so abort
        if ds[list(ds)[0]].isnull().all() or (ds[list(ds)[0]] == 0).all():
            arcpy.AddError('Satellite NetCDF has empty variables. Please download again.')
            return

        try:
            # now, do proper open of satellite netcdf properly (and set nodata to nan)
            ds = satfetcher.load_local_nc(nc_path=in_raw_nc,
                                          use_dask=True,
                                          conform_nodata_to=np.nan)
        except Exception as e:
            arcpy.AddError('Could not properly load input satellite NetCDF data.')
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
            arcpy.AddWarning('No flags set, selecting default')
            in_fmask_flags = [1, 4, 5]

        # check numeric flags are valid
        for flag in in_fmask_flags:
            if flag not in [0, 1, 2, 3, 4, 5, 6]:
                arcpy.AddError('Pixel flag not supported.')
                return

        # check if duplicate flags
        u, c = np.unique(in_fmask_flags, return_counts=True)
        if len(u[c > 1]) > 0:
            arcpy.AddError('Duplicate pixel flags detected.')
            return

        # check if mask band exists
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
            arcpy.AddError('Could not cloud mask pixels.')
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
                arcpy.AddError('Satellite NetCDF is missing band: {}. Need all bands.'.format(band))
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset months to requested...')
        arcpy.SetProgressorPosition(6)

        # prepare analysis month(s)
        if in_analysis_months == '':
            arcpy.AddError('Must include at least one analysis month.')
            return

        # unpack month(s)
        analysis_months = [int(e) for e in in_analysis_months.split(';')]

        try:
            # reduce xr dataset into only analysis months
            ds = gdvspectra.subset_months(ds=ds,
                                          month=analysis_months,
                                          inplace=True)
        except Exception as e:
            arcpy.AddError('Could not subset Satellite NetCDF by analysis months.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating tasselled cap vegetation index...')
        arcpy.SetProgressorPosition(7)

        try:
            # calculate tasselled cap green and bare
            ds = tools.calculate_indices(ds=ds,
                                         index=['tcg', 'tcb'],
                                         rescale=False,
                                         drop=True)

            # add band attrs back on
            ds['tcg'].attrs = ds_band_attrs
            ds['tcb'].attrs = ds_band_attrs

        except Exception as e:
            arcpy.AddError('Could not calculate tasselled cap index.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing month(s) into annual medians...')
        arcpy.SetProgressorPosition(8)

        # reduce months into annual medians (year starts, YS)
        try:
            ds = gdvspectra.resample_to_freq_medians(ds=ds,
                                                     freq='YS',
                                                     inplace=True)

            # add band attrs back on
            ds['tcg'].attrs = ds_band_attrs
            ds['tcb'].attrs = ds_band_attrs

        except Exception as e:
            arcpy.AddError('Could not resample months in Satellite NetCDF.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(9)

        try:
            # compute!
            ds = ds.compute()
        except Exception as e:
            arcpy.AddError('Could not compute dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('Satellite NetCDF is empty. Please download again.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Interpolating dataset, if requested...')
        arcpy.SetProgressorPosition(10)

        # if requested...
        if in_interpolate:
            try:
                # interpolate along time dimension (linear)
                ds = tools.perform_interp(ds=ds, method='full')
            except Exception as e:
                arcpy.AddError('Could not interpolate satellite NetCDF.')
                arcpy.AddMessage(str(e))
                return

                # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Standardising data to invariant targets...')
        arcpy.SetProgressorPosition(11)

        # check upper quantile
        if in_ivt_qlower < 0 or in_ivt_qlower >= 0.5:
            arcpy.AddMessage('Lower quantile must be between 0, 0.5. Setting to default.')
            in_ivt_qlower = 0.05

        # do same for upper quantile
        if in_ivt_qupper <= 0.5 or in_ivt_qupper > 1.0:
            arcpy.AddMessage('Upper quantile must be between 0.5, 1.0. Setting to default.')
            in_ivt_qlower = 0.99

            # check if upper <= lower
        if in_ivt_qupper <= in_ivt_qlower:
            arcpy.AddError('Upper quantile must be > than lower quantile value.')
            return

        try:
            # standardise to targets
            ds = gdvspectra.standardise_to_targets(ds,
                                                   q_upper=in_ivt_qupper,
                                                   q_lower=in_ivt_qlower)
        except Exception as e:
            arcpy.AddError('Could not standardise satellite data to invariant targets.')
            arcpy.AddMessage(str(e))
            return

        # final check to see if data exists
        if ds['tcg'].isnull().all() or ds['tcb'].isnull().all():
            arcpy.AddError('Could not standardise satellite NetCDF.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Performing CVA...')
        arcpy.SetProgressorPosition(12)

        # check baseline and comparison start and end years
        if in_base_end_year < in_base_start_year:
            arcpy.AddError('Baseline end year must not be < start year.')
            return
        elif in_comp_end_year < in_comp_start_year:
            arcpy.AddError('Comparison end year must not be < start year.')
            return
        elif in_comp_start_year < in_base_start_year:
            arcpy.AddError('Comparison start year must not be < baseline start year.')
            return

        # check if baseline and comparison years in dataset
        years = ds['time.year']
        if in_base_start_year not in years or in_base_end_year not in years:
            arcpy.AddError('Baseline start and end years not found in dataset.')
            return
        elif in_comp_start_year not in years or in_comp_end_year not in years:
            arcpy.AddError('Comparison start and end years not found in dataset.')
            return

        # check magnitude value
        if in_tmf < 0 or in_tmf > 100:
            arcpy.AddError('CVA threshold magnitude must be between 0 and 100.')
            return

        try:
            # generate cva
            ds_cva = gdvspectra.perform_cva(ds=ds,
                                            base_times=(in_base_start_year, in_base_end_year),
                                            comp_times=(in_comp_start_year, in_comp_end_year),
                                            reduce_comp=False,
                                            vege_var='tcg',
                                            soil_var='tcb',
                                            tmf=in_tmf)
        except Exception as e:
            arcpy.AddError('Could not perform CVA.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Isolating CVA magnitude quartiles...')
        arcpy.SetProgressorPosition(13)

        try:
            # isolate cva magnitude via angle quartiles
            ds_cva = gdvspectra.isolate_cva_quarters(ds=ds_cva,
                                                     drop_orig_vars=True)
        except Exception as e:
            arcpy.AddError('Could not isolate CVA quartiles.')
            arcpy.AddMessage(str(e))
            return

            # check if variables are empty
        if ds_cva.to_array().isnull().all():
            arcpy.AddError('CVA dataset is empty. Check range of years and magnitude threshold.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading, checking and applying mask, if requested...')
        arcpy.SetProgressorPosition(14)

        # if requested...
        if in_mask_nc is not None:

            try:
                # do quick lazy load of mask netcdf for checking
                ds_mask = xr.open_dataset(in_mask_nc)
            except Exception as e:
                arcpy.AddWarning('Could not quick load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return

            # check xr type, vars, coords, dims, attrs
            if not isinstance(ds_mask, xr.Dataset):
                arcpy.AddError('Input mask NetCDF must be a xr dataset.')
                return
            elif len(ds_mask) == 0:
                arcpy.AddError('Input mask NetCDF has no data/variables/bands.')
                return
            elif 'x' not in ds_mask.dims or 'y' not in ds_mask.dims:
                arcpy.AddError('Input mask NetCDF must have x, y dimensions.')
                return
            elif 'x' not in ds_mask.coords or 'y' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have x, y and time coords.')
                return
            elif 'spatial_ref' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have a spatial_ref coord.')
                return
            elif len(ds_mask['x']) == 0 or len(ds_mask['y']) == 0:
                arcpy.AddError('Input mask NetCDF must have at least one x, y index.')
                return
            elif 'time' in ds_mask:
                arcpy.AddError('Input mask NetCDF must not have a time dimension.')
                return
            elif 'thresh' not in ds_mask:
                arcpy.AddError('Input mask NetCDF must have a "thresh" variable. Run GDVSpectra Threshold.')
                return
            elif ds_mask.attrs == {}:
                arcpy.AddError('Input mask NetCDF attributes not found. NetCDF must have attributes.')
                return
            elif not hasattr(ds_mask, 'crs'):
                arcpy.AddError('Input mask NetCDF CRS attribute not found. CRS required.')
                return
            elif ds_mask.crs != 'EPSG:3577':
                arcpy.AddError('Input mask NetCDF CRS is not EPSG:3577. EPSG:3577 required.')
                return
            elif not hasattr(ds_mask, 'nodatavals'):
                arcpy.AddError('Input mask NetCDF nodatavals attribute not found.')
                return

                # check if variables (should only be thresh) are empty
            if ds_mask['thresh'].isnull().all() or (ds_mask['thresh'] == 0).all():
                arcpy.AddError('Input mask NetCDF "thresh" variable is empty. Please download again.')
                return

            try:
                # now, do proper open of mask netcdf (set nodata to nan)
                ds_mask = satfetcher.load_local_nc(nc_path=in_mask_nc,
                                                   use_dask=True,
                                                   conform_nodata_to=np.nan)

                # compute it!
                ds_mask = ds_mask.compute()

            except Exception as e:
                arcpy.AddError('Could not properly load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return

            try:
                # check if like and mask datasets overlap
                if not tools.all_xr_intersect([ds_cva, ds_mask]):
                    arcpy.AddError('Input datasets do not intersect.')
                    return

                # resample mask dataset to match likelihood
                ds_mask = tools.resample_xr(ds_from=ds_mask,
                                            ds_to=ds_cva,
                                            resampling='nearest')

                # squeeze
                ds_mask = ds_mask.squeeze(drop=True)

            except Exception as e:
                arcpy.AddError('Could not intersect input datasets.')
                arcpy.AddMessage(str(e))
                return

            # we made it, so mask cva
            ds_cva = ds_cva.where(~ds_mask['thresh'].isnull())

            # check if variables are empty
            if ds_cva.to_array().isnull().all():
                arcpy.AddError('Masked CVA dataset is empty. Check range of years and magnitude threshold.')
                return

                # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(15)

        # append attrbutes on to dataset and bands
        ds_cva.attrs = ds_attrs
        ds_cva['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_cva:
            ds_cva[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(16)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_cva, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(17)

        # if requested...
        if in_add_result_to_map:

            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove cva layer if already exists
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'cva.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'cva' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'cva.crf')
                crf = arcpy.CopyRaster_management(in_raster=out_nc,
                                                  out_rasterdataset=out_file)

                # add to map
                m.addDataFromPath(crf)

            except Exception as e:
                arcpy.AddWarning('Could not visualise output, aborting visualisation.')
                arcpy.AddMessage(str(e))
                pass

            try:
                # get symbology, update it
                layer = m.listLayers('cva.crf')[0]
                sym = layer.symbology

                # if layer has stretch coloriser, apply color
                if hasattr(sym, 'colorizer'):
                    if sym.colorizer.type == 'RasterStretchColorizer':
                        # apply percent clip type
                        sym.colorizer.stretchType = 'PercentClip'
                        sym.colorizer.minPercent = 0.1
                        sym.colorizer.maxPercent = 0.1

                        # apply color map
                        cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]
                        sym.colorizer.colorRamp = cmap

                        # apply other basic options
                        sym.colorizer.invertColorRamp = False
                        sym.colorizer.gamma = 1.0

                        # update symbology
                        layer.symbology = sym

            except Exception as e:
                arcpy.AddWarning('Could not colorise output, aborting colorisation.')
                arcpy.AddMessage(str(e))
                pass

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(18)

        # close satellite dataset
        ds.close()
        del ds

        # close cva dataset
        ds_cva.close()
        del ds_cva

        # close mask dataset, if exists
        if in_mask_nc is not None:
            ds_mask.close()
            del ds_mask

        # notify user
        arcpy.AddMessage('Generated CVA successfully.')

        return