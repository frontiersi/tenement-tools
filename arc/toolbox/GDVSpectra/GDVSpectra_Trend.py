import arcpy
import os
import datetime
import numpy as np
import xarray as xr
import dask

from modules import gdvspectra
from shared import satfetcher, tools

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class GDVSpectra_Trend:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'GDVSpectra Trend'
        self.description = 'Perform a time-series trend analysis on an existing ' \
                           'GDV Likelihood data cube. Produces a map of areas where ' \
                           'vegetation has continuously increased, decreased or ' \
                           'has not changed.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input like netcdf file
        par_like_nc_file = arcpy.Parameter(
            displayName='Input GDV Likelihood NetCDF file',
            name='in_like_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_like_nc_file.filter.list = ['nc']

        # input mask netcdf file
        par_mask_nc_file = arcpy.Parameter(
            displayName='Input GDV Threshold mask NetCDF file',
            name='in_mask_nc_file',
            datatype='DEFile',
            parameterType='Optional',
            direction='Input')
        par_mask_nc_file.filter.list = ['nc']

        # output netcdf location
        par_out_nc_file = arcpy.Parameter(
            displayName='Output GDV Trend NetCDF file',
            name='out_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # use all years
        par_use_all_years = arcpy.Parameter(
            displayName='Combine all input years',
            name='in_use_all_years',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_use_all_years.value = True

        # set specific start year
        par_start_year = arcpy.Parameter(
            displayName='Start year of trend analysis',
            name='in_start_year',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_start_year.filter.type = 'ValueList'
        par_start_year.filter.list = []

        # set specific end year
        par_end_year = arcpy.Parameter(
            displayName='End year of trend analysis',
            name='in_end_year',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_end_year.filter.type = 'ValueList'
        par_end_year.filter.list = []

        # set analysis type
        par_analysis_type = arcpy.Parameter(
            displayName='Trend analysis method',
            name='in_analysis_type',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_analysis_type.filter.type = 'ValueList'
        par_analysis_type.filter.list = ['Mann-Kendall', 'Theil-Sen Slope']
        par_analysis_type.value = 'Mann-Kendall'

        # mk p-value
        par_mk_pvalue = arcpy.Parameter(
            displayName='P-value',
            name='in_mk_pvalue',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_mk_pvalue.filter.type = 'Range'
        par_mk_pvalue.filter.list = [0.001, 1.0]
        par_mk_pvalue.value = 0.05

        # mk direction
        par_mk_direction = arcpy.Parameter(
            displayName='Trend direction',
            name='in_mk_direction',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_mk_direction.filter.type = 'ValueList'
        par_mk_direction.filter.list = ['Both', 'Increasing', 'Decreasing']
        par_mk_direction.value = 'Both'

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
            par_like_nc_file,
            par_mask_nc_file,
            par_out_nc_file,
            par_use_all_years,
            par_start_year,
            par_end_year,
            par_analysis_type,
            par_mk_pvalue,
            par_mk_direction,
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
        global GDVSPECTRA_TREND

        # unpack global parameter values
        curr_file = GDVSPECTRA_TREND.get('in_file')

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

                # populate start and end year lists
                parameters[4].filter.list = dts
                parameters[5].filter.list = dts

                # reset start and end year selections
                if len(dts) != 0:
                    parameters[4].value = dts[0]
                    parameters[5].value = dts[-1]
                else:
                    parameters[4].value = None
                    parameters[5].value = None

        # update global values
        GDVSPECTRA_TREND = {
            'in_file': parameters[0].valueAsText,
        }

        # enable start and end years based on all years checkbox
        if parameters[3].value is False:
            parameters[4].enabled = True
            parameters[5].enabled = True
        else:
            parameters[4].enabled = False
            parameters[5].enabled = False

        # enable relevant controls when manken or theils
        if parameters[6].valueAsText == 'Mann-Kendall':
            parameters[7].enabled = True
            parameters[8].enabled = True
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
        Executes the GDV Spectra Trend module.
        """

        # disable future warnings


        # grab parameter values
        in_like_nc = parameters[0].valueAsText  # likelihood netcdf
        in_mask_nc = parameters[1].valueAsText  # thresh mask netcdf
        out_nc = parameters[2].valueAsText  # output netcdf
        in_use_all_years = parameters[3].value  # use all years
        in_start_year = parameters[4].value  # start year
        in_end_year = parameters[5].value  # end year
        in_trend_method = parameters[6].value  # trend method
        in_mk_pvalue = parameters[7].value  # mk pvalue
        in_mk_direction = parameters[8].value  # mk direction
        in_add_result_to_map = parameters[9].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra Trend.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=10)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking likelihood netcdf...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of likelihood netcdf for checking
            ds_like = xr.open_dataset(in_like_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input likelihood NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_like, xr.Dataset):
            arcpy.AddError('Input likelihood NetCDF must be a xr dataset.')
            return
        elif len(ds_like) == 0:
            arcpy.AddError('Input likelihood NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds_like.dims or 'y' not in ds_like.dims or 'time' not in ds_like.dims:
            arcpy.AddError('Input likelihood NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds_like.coords or 'y' not in ds_like.coords or 'time' not in ds_like.coords:
            arcpy.AddError('Input likelihood NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds_like.coords:
            arcpy.AddError('Input likelihood NetCDF must have a spatial_ref coord.')
            return
        elif len(ds_like['x']) == 0 or len(ds_like['y']) == 0:
            arcpy.AddError('Input likelihood NetCDF must have at least one x, y index.')
            return
        elif len(ds_like['time']) <= 3:
            arcpy.AddError('Input likelihood NetCDF must have 4 or more times.')
            return
        elif 'like' not in ds_like:
            arcpy.AddError('Input likelihood NetCDF must have a "like" variable. Run GDVSpectra Likelihood.')
            return
        elif not hasattr(ds_like, 'time.year') or not hasattr(ds_like, 'time.month'):
            arcpy.AddError('Input likelihood NetCDF must have time with year and month index.')
            return
        elif ds_like.attrs == {}:
            arcpy.AddError('Input likelihood NetCDF attributes not found. NetCDF must have attributes.')
            return
        elif not hasattr(ds_like, 'crs'):
            arcpy.AddError('Input likelihood NetCDF CRS attribute not found. CRS required.')
            return
        elif ds_like.crs != 'EPSG:3577':
            arcpy.AddError('Input likelihood NetCDF CRS is not EPSG:3577. EPSG:3577 required.')
            return
        elif not hasattr(ds_like, 'nodatavals'):
            arcpy.AddError('Input likelihood NetCDF nodatavals attribute not found.')
            return

            # check if variables (should only be like) are empty
        if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
            arcpy.AddError('Input likelihood NetCDF "like" variable is empty. Please download again.')
            return

        try:
            # now, do proper open of likelihood netcdf (set nodata to nan)
            ds_like = satfetcher.load_local_nc(nc_path=in_like_nc,
                                               use_dask=True,
                                               conform_nodata_to=np.nan)
        except Exception as e:
            arcpy.AddError('Could not properly load input likelihood NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Getting NetCDF attributes...')
        arcpy.SetProgressorPosition(2)

        # get attributes from dataset
        ds_attrs = ds_like.attrs
        ds_band_attrs = ds_like['like'].attrs
        ds_spatial_ref_attrs = ds_like['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Grouping dates, if required...')
        arcpy.SetProgressorPosition(3)

        # remove potential datetime duplicates (group by day)
        ds_like = satfetcher.group_by_solar_day(ds_like)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset based on time, if requested...')
        arcpy.SetProgressorPosition(4)

        # if requested...
        if in_use_all_years is False:

            # check start, end year valid
            if in_start_year is None or in_end_year is None:
                arcpy.AddError('Did not specify start and/or end years.')
                return
            elif in_start_year >= in_end_year:
                arcpy.AddError('Start year must be < end year.')
                return
            elif abs(in_end_year - in_start_year) < 3:
                arcpy.AddError('Require 4 or more years of data.')
                return

            # check if both years in dataset
            years = ds_like['time.year']
            if in_start_year not in years or in_end_year not in years:
                arcpy.AddError('Start year is not in likelihood NetCDF.')
                return

            # subset likelihood dataset based on years
            ds_like = ds_like.where((ds_like['time.year'] >= in_start_year) &
                                    (ds_like['time.year'] <= in_end_year), drop=True)

            # check if more than three years still exist
            if len(ds_like['time']) < 4:
                arcpy.AddError('Subset of likelihood NetCDF resulted in < 4 years of data.')
                return

            # check if variables (should only be like) are empty
            if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
                arcpy.AddError('Subset of likelihood NetCDF is empty. Increase year range.')
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(5)

        try:
            # compute!
            ds_like = ds_like.compute()
        except Exception as e:
            arcpy.AddError('Could not compute likelihood dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if we still have values
        if ds_like.to_array().isnull().all():
            arcpy.AddError('Input likelihood NetCDF is empty. Please download again.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading, checking and applying mask, if requested...')
        arcpy.SetProgressorPosition(6)

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
                arcpy.AddError('Input mask NetCDF must have x, y coords.')
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
                if not tools.all_xr_intersect([ds_like, ds_mask]):
                    arcpy.AddError('Input datasets do not intersect.')
                    return

                # resample mask dataset to match likelihood
                ds_mask = tools.resample_xr(ds_from=ds_mask,
                                            ds_to=ds_like,
                                            resampling='nearest')

                # squeeze it to be safe
                ds_mask = ds_mask.squeeze(drop=True)

            except Exception as e:
                arcpy.AddError('Could not intersect input datasets.')
                arcpy.AddMessage(str(e))
                return

            # we made it, so mask likelihood
            ds_like = ds_like.where(~ds_mask['thresh'].isnull())

            # check if variables (should only be thresh) are empty
            if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
                arcpy.AddError('Masked likelihood dataset is empty.')
                return

                # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Performing trend analysis...')

        # check if trend method is supported
        if in_trend_method not in ['Mann-Kendall', 'Theil-Sen Slope']:
            arcpy.AddError('Trend method not supported.')
            return

        # check and perform mann-kendall or theil sen
        if in_trend_method == 'Mann-Kendall':

            # check mannkendall pvalue
            if in_mk_pvalue is None:
                arcpy.AddError('Mann-Kendall p-value not provided.')
                return
            elif in_mk_pvalue < 0.001 or in_mk_pvalue > 1.0:
                arcpy.AddError('Mann-Kendall p-value must be between 0.001 and 1.0.')
                return

            # check mannkendall direction
            if in_mk_direction not in ['Increasing', 'Decreasing', 'Both']:
                arcpy.AddError('Mann-Kendall direction not supported.')
                return

            # prepare mannkendal direction (must be inc, dec or both)
            if in_mk_direction in ['Increasing', 'Decreasing']:
                in_mk_direction = in_mk_direction.lower()[:3]
            else:
                in_mk_direction = 'both'

            try:
                # perform mann-kendall trend analysis
                ds_trend = gdvspectra.perform_mk_original(ds=ds_like,
                                                          pvalue=in_mk_pvalue,
                                                          direction=in_mk_direction)

                # warn if no change values returned
                if ds_trend['tau'].isnull().all():
                    arcpy.AddError('Trend output is empty, check p-value, date range and/or mask.')
                    return

            except Exception as e:
                arcpy.AddError('Could not perform Mann-Kendall trend analysis.')
                arcpy.AddMessage(str(e))
                return

        else:
            try:
                # perform theil-sen trend analysis
                ds_trend = gdvspectra.perform_theilsen_slope(ds=ds_like,
                                                             alpha=0.95)

                # warn if no change values returned
                if ds_trend['theil'].isnull().all():
                    arcpy.AddError('Trend output is empty, check date range and/or mask.')
                    return

            except Exception as e:
                arcpy.AddError('Could not perform Theil-Sen trend analysis.')
                arcpy.AddMessage(str(e))
                return

                # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(7)

        # append attrbutes on to dataset and bands
        ds_trend.attrs = ds_attrs
        ds_trend['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_trend:
            ds_trend[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(8)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_trend, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(9)

        # if requested...
        if in_add_result_to_map:
            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove trend layer if already exists
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'trend.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'trend' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'trend.crf')
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
                layer = m.listLayers('trend.crf')[0]
                sym = layer.symbology

                # if layer has stretch coloriser, apply color
                if hasattr(sym, 'colorizer'):

                    # apply percent clip type
                    sym.colorizer.stretchType = 'PercentClip'
                    sym.colorizer.minPercent = 0.75
                    sym.colorizer.maxPercent = 0.75

                    # set default trend cmap, override if mannkenn inc or dec used
                    cmap = aprx.listColorRamps('Red-Blue (Continuous)')[0]
                    if in_trend_method == 'Mann-Kendall':
                        if in_mk_direction == 'inc':
                            cmap = aprx.listColorRamps('Yellow-Green-Blue (Continuous)')[0]
                        elif in_mk_direction == 'dec':
                            cmap = aprx.listColorRamps('Yellow-Orange-Red (Continuous)')[0]

                    # apply colormap
                    sym.colorizer.colorRamp = cmap

                    # invert colormap if mannkenn decline
                    if in_trend_method == 'Mann-Kendall' and in_mk_direction == 'dec':
                        sym.colorizer.invertColorRamp = True
                    else:
                        sym.colorizer.invertColorRamp = False

                    # set gamma
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
        arcpy.SetProgressorPosition(10)

        # close likelihood dataset
        ds_like.close()
        del ds_like

        # close trend dataset
        ds_trend.close()
        del ds_trend

        # close mask (if exists)
        if in_mask_nc is not None:
            ds_mask.close()
            del ds_mask

        # notify user
        arcpy.AddMessage('Generated GDV Trend successfully.')

        return