
import os
import datetime
import numpy as np
import arcpy
import xarray as xr
import dask

from arc.toolbox.globals import GRP_LYR_FILE
from shared import satfetcher, tools, arc
from modules import vegfrax, cog
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
class VegFrax_Fractional_Cover:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'VegFrax Fractional Cover'
        self.description = 'Extrapolate class values from a high-resolution ' \
                           'classifed GeoTiff as fractional maps using moderate-' \
                           'resolution satellite imagery from COG Fetch.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input low res netcdf
        par_in_low_nc = arcpy.Parameter(
            displayName='Input satellite NetCDF file',
            name='in_low_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_low_nc.filter.list = ['nc']

        # input high res geotiff
        par_in_high_tif = arcpy.Parameter(
            displayName='Input classified GeoTiff',
            name='in_high_tif',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_high_tif.filter.list = ['tif']

        # output netcdf file
        par_out_nc_file = arcpy.Parameter(
            displayName='Output fractional cover NetCDF file',
            name='out_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # aggregate all dates
        par_aggregate_dates = arcpy.Parameter(
            displayName='Combine all input NetCDF dates',
            name='in_aggregate_dates',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_aggregate_dates.value = True

        # start year of low res
        par_start_date = arcpy.Parameter(
            displayName='Start date of low-resolution NetCDF',
            name='in_start_date',
            datatype='GPDate',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_start_date.values = '2010/01/01'

        # end date of low res
        par_end_date = arcpy.Parameter(
            displayName='End date of low-resolution NetCDF',
            name='in_end_date',
            datatype='GPDate',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_end_date.values = '2020/12/31'

        # high res classes
        par_classes = arcpy.Parameter(
            displayName='Classes to convert',
            name='in_classes',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=True)
        par_classes.filter.type = 'ValueList'
        par_classes.values = []

        # aggregate classes
        par_aggregate_classes = arcpy.Parameter(
            displayName='Combine selected classes',
            name='in_aggregate_classes',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_aggregate_classes.value = False

        # number stratified samples
        par_num_samples = arcpy.Parameter(
            displayName='Number of stratified random samples',
            name='in_num_samples',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_num_samples.filter.type = 'Range'
        par_num_samples.filter.list = [10, 9999999]
        par_num_samples.value = 100

        # maximum window no data
        par_max_nodata = arcpy.Parameter(
            displayName='Maximum NoData fraction',
            name='in_max_nodata',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_max_nodata.filter.type = 'Range'
        par_max_nodata.filter.list = [0.0, 1.0]
        par_max_nodata.value = 0.0

        # smooth
        par_smooth = arcpy.Parameter(
            displayName='Smooth output',
            name='in_smooth',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_smooth.value = False

        # number of model estimators
        par_num_estimators = arcpy.Parameter(
            displayName='Number of model estimators',
            name='in_num_estimators',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Model Parameters',
            multiValue=False)
        par_num_estimators.filter.type = 'Range'
        par_num_estimators.filter.list = [1, 9999999]
        par_num_estimators.value = 100

        # model criterion
        par_criterion = arcpy.Parameter(
            displayName='Criterion',
            name='in_criterion',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Model Parameters',
            multiValue=False)
        par_criterion.filter.type = 'ValueList'
        par_criterion.filter.list = ['Mean Squared Error', 'Mean Absolute Error', 'Poisson']
        par_criterion.value = 'Mean Squared Error'

        # max depth
        par_max_depth = arcpy.Parameter(
            displayName='Maximum tree depth',
            name='in_max_depth',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            category='Model Parameters',
            multiValue=False)
        par_max_depth.filter.type = 'Range'
        par_max_depth.filter.list = [1, 9999999]
        par_max_depth.value = None

        # max_features
        par_max_features = arcpy.Parameter(
            displayName='Maximum features',
            name='in_max_features',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Model Parameters',
            multiValue=False)
        par_max_features.filter.type = 'ValueList'
        par_max_features.filter.list = ['Auto', 'Log2']
        par_max_features.value = 'Auto'

        # boostrap
        par_boostrap = arcpy.Parameter(
            displayName='Boostrap',
            name='in_boostrap',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Model Parameters',
            multiValue=False)
        par_boostrap.value = True

        # mask flags
        par_fmask_flags = arcpy.Parameter(
            displayName='Include pixels flags',
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
            par_in_low_nc,
            par_in_high_tif,
            par_out_nc_file,
            par_aggregate_dates,
            par_start_date,
            par_end_date,
            par_classes,
            par_aggregate_classes,
            par_num_samples,
            par_max_nodata,
            par_smooth,
            par_num_estimators,
            par_criterion,
            par_max_depth,
            par_max_features,
            par_boostrap,
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

        # globals
        global VEGFRAX_FRACTIONAL_COVER

        # unpack global parameter values
        curr_nc_file = VEGFRAX_FRACTIONAL_COVER.get('in_nc_file')
        curr_tif_file = VEGFRAX_FRACTIONAL_COVER.get('in_tif_file')

        # if input file added, run
        if parameters[0].value is not None:

            # if global has no matching file (or first run), reload all
            if curr_nc_file != parameters[0].valueAsText:
                try:
                    ds = xr.open_dataset(parameters[0].valueAsText)
                    s_dt = ds['time'].isel(time=0).dt.strftime('%Y-%m-%d').values
                    e_dt = ds['time'].isel(time=-1).dt.strftime('%Y-%m-%d').values
                    ds.close()
                except:
                    s_dt, e_dt = '2010-01-01', '2020-12-31'

                # populate start, end date controls
                parameters[4].value = str(s_dt)
                parameters[5].value = str(e_dt)

        # if occurrence point feat added, run
        if parameters[1].value is not None:

            # if global has no matching feat (or first run), reload all
            if curr_tif_file != parameters[1].valueAsText:
                try:
                    ds = xr.open_rasterio(parameters[1].valueAsText)
                    if len(ds) != 1 or 'int' not in str(ds.dtype):
                        return

                    # get unique class labels, exclude nodata
                    clss = np.unique(ds)
                    clss = clss[clss != ds.nodatavals]
                    clss = ['Class: {}'.format(c) for c in clss]
                    ds.close()
                except:
                    clss = []

                # populate class list, reset selection
                parameters[6].filter.list = clss
                parameters[6].values = clss

        # update global values
        VEGFRAX_FRACTIONAL_COVER = {
            'in_nc_file': parameters[0].valueAsText,
            'in_tif_file': parameters[1].valueAsText,
        }

        # enable date aggregate if netcdf added
        if parameters[0].value is not None:
            parameters[3].enabled = True
        else:
            parameters[3].enabled = False

        # enable start and end date based on aggregate
        if parameters[3].value is False:
            parameters[4].enabled = True
            parameters[5].enabled = True
        else:
            parameters[4].enabled = False
            parameters[5].enabled = False

        # enable classes and aggregate if tif added
        if parameters[1].value is not None:
            parameters[6].enabled = True
            parameters[7].enabled = True
        else:
            parameters[6].enabled = False
            parameters[7].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the VegFrax Fractional Cover module.
        """
        # grab parameter values
        in_low_nc = parameters[0].valueAsText  # raw input low res netcdf
        in_high_tif = parameters[1].valueAsText  # raw input high res tif
        out_nc = parameters[2].valueAsText  # output vegfrax netcdf
        in_agg_dates = parameters[3].value  # aggregate all dates
        in_start_date = parameters[4].value  # start date of aggregate
        in_end_date = parameters[5].value  # end date of aggregate
        in_classes = parameters[6].valueAsText  # selected classes
        in_agg_classes = parameters[7].value  # aggregate selected classes
        in_num_samples = parameters[8].value  # number of samples
        in_max_nodata = parameters[9].value  # max nodata frequency
        in_smooth = parameters[10].value  # smooth output
        in_num_estimator = parameters[11].value  # number of model estimators
        in_criterion = parameters[12].value  # criterion type
        in_max_depth = parameters[13].value  # max tree depth
        in_max_features = parameters[14].value  # maximum features
        in_bootstrap = parameters[15].value  # boostrap
        in_fmask_flags = parameters[16].valueAsText  # fmask flag values
        in_max_cloud = parameters[17].value  # max cloud percentage
        in_add_result_to_map = parameters[18].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning VegFrax Fractional Cover.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=19)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking satellite NetCDF...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of satellite netcdf for checking
            ds_low = xr.open_dataset(in_low_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input satellite NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_low, xr.Dataset):
            arcpy.AddError('Input satellite NetCDF must be a xr dataset.')
            return
        elif len(ds_low) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds_low.dims or 'y' not in ds_low.dims or 'time' not in ds_low.dims:
            arcpy.AddError('Input satellite NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds_low.coords or 'y' not in ds_low.coords or 'time' not in ds_low.coords:
            arcpy.AddError('Input satellite NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds_low.coords:
            arcpy.AddError('Input satellite NetCDF must have a spatial_ref coord.')
            return
        elif len(ds_low['x']) == 0 or len(ds_low['y']) == 0 or len(ds_low['time']) == 0:
            arcpy.AddError('Input satellite NetCDF must have all at least one x, y and time index.')
            return
        elif 'oa_fmask' not in ds_low and 'fmask' not in ds_low:
            arcpy.AddError('Expected cloud mask band not found in satellite NetCDF.')
            return
        elif not hasattr(ds_low, 'time.year') or not hasattr(ds_low, 'time.month'):
            arcpy.AddError('Input satellite NetCDF must have time with year and month component.')
            return
        elif ds_low.attrs == {}:
            arcpy.AddError('Satellite NetCDF must have attributes.')
            return
        elif not hasattr(ds_low, 'crs'):
            arcpy.AddError('Satellite NetCDF CRS attribute not found. CRS required.')
            return
        elif ds_low.crs != 'EPSG:3577':
            arcpy.AddError('Satellite NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
            return
        elif not hasattr(ds_low, 'nodatavals'):
            arcpy.AddError('Satellite NetCDF nodatavals attribute not found.')
            return

            # efficient: if all nan, 0 at first var, assume rest same, so abort
        if ds_low[list(ds_low)[0]].isnull().all() or (ds_low[list(ds_low)[0]] == 0).all():
            arcpy.AddError('Satellite NetCDF has empty variables. Please download again.')
            return

        try:
            # now, do proper open of satellite netcdf properly (and set nodata to nan)
            ds_low = satfetcher.load_local_nc(nc_path=in_low_nc,
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
        ds_attrs = ds_low.attrs
        ds_band_attrs = ds_low[list(ds_low)[0]].attrs
        ds_spatial_ref_attrs = ds_low['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Grouping dates, if required...')
        arcpy.SetProgressorPosition(3)

        # remove potential datetime duplicates (group by day)
        ds_low = satfetcher.group_by_solar_day(ds_low)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset dates, if requested...')
        arcpy.SetProgressorPosition(4)

        # check if dates are to be aggregated
        if in_agg_dates is None:
            arcpy.AddError('Must specify whether to aggregate dates or not.')
            return

        # if requested...
        if in_agg_dates is False:

            # check start and end dates
            if in_start_date is None or in_end_date is None:
                arcpy.AddError('Did not provide a start or end date.')
                return

            # prepare start, end dates
            in_start_date = in_start_date.strftime('%Y-%m-%d')
            in_end_date = in_end_date.strftime('%Y-%m-%d')

            # check date range is valid
            if in_start_date >= in_end_date:
                arcpy.AddError('End date must be greater than start date.')
                return

            try:
                # subset to requested date range
                ds_low = vegfrax.subset_dates(ds=ds_low,
                                              start_date=in_start_date,
                                              end_date=in_end_date)
            except Exception as e:
                arcpy.AddError('Could not subset satellite NetCDF, see messages for details.')
                arcpy.AddMessage(str(e))
                return

            # check if any dates exist
            if 'time' not in ds_low or len(ds_low['time']) == 0:
                arcpy.AddError('No dates exist in satellite NetCDF for requested date range.')
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Removing invalid pixels and empty dates...')
        arcpy.SetProgressorPosition(5)

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
        mask_band = arc.get_name_of_mask_band(list(ds_low))

        try:
            # remove invalid pixels and empty scenes
            ds_low = cog.remove_fmask_dates(ds=ds_low,
                                            valid_class=in_fmask_flags,
                                            max_invalid=in_max_cloud,
                                            mask_band=mask_band,
                                            nodata_value=np.nan,
                                            drop_fmask=True)
        except Exception as e:
            arcpy.AddError('Could not cloud mask pixels.')
            arcpy.AddMessage(str(e))
            return

        # check if any dates remain
        if 'time' not in ds_low or len(ds_low['time']) == 0:
            arcpy.AddError('No cloud-free data exists in satellite NetCDF for requested date range.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Conforming satellite band names...')
        arcpy.SetProgressorPosition(6)

        try:
            # get platform name from attributes, error if no attributes
            in_platform = arc.get_platform_from_dea_attrs(ds_attrs)

            # conform dea aws band names based on platform
            ds_low = satfetcher.conform_dea_ard_band_names(ds=ds_low,
                                                           platform=in_platform.lower())
        except Exception as e:
            arcpy.AddError('Could not get platform from attributes.')
            arcpy.AddMessage(str(e))
            return

        # check if all expected bands are in dataset
        for band in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
            if band not in ds_low:
                arcpy.AddError('Satellite NetCDF is missing band: {}. Need all bands.'.format(band))
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating tasselled cap index...')
        arcpy.SetProgressorPosition(7)

        try:
            # calculate tasselled cap green, bare, water
            ds_low = tools.calculate_indices(ds=ds_low,
                                             index=['tcg', 'tcb', 'tcw'],
                                             rescale=False,
                                             drop=True)
        except Exception as e:
            arcpy.AddError('Could not calculate tasselled cap index.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset into all-time median...')
        arcpy.SetProgressorPosition(8)

        try:
            # reduce into an all-time median
            ds_low = vegfrax.reduce_to_median(ds=ds_low)

            # add band attrs back on
            ds_low['tcg'].attrs = ds_band_attrs
            ds_low['tcb'].attrs = ds_band_attrs
            ds_low['tcw'].attrs = ds_band_attrs

        except Exception as e:
            arcpy.AddError('Could not reduce satellite NetCDF to all-time median.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing satellite NetCDF into memory, please wait...')
        arcpy.SetProgressorPosition(9)

        try:
            # compute!
            ds_low = ds_low.compute()
        except Exception as e:
            arcpy.AddError('Could not compute satellite NetCDF. See messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check if all nan again
        if ds_low.to_array().isnull().all():
            arcpy.AddError('Satellite NetCDF is empty. Please download again.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking classified GeoTiff...')
        arcpy.SetProgressorPosition(10)

        # check if type is geotiff
        if not in_high_tif.endswith('.tif'):
            arcpy.AddError('High-resolution input is not a GeoTiff.')
            return

        try:
            # do quick lazy load of geotiff for checking
            ds_high = xr.open_rasterio(in_high_tif)
            ds_high = ds_high.to_dataset(dim='band')
        except Exception as e:
            arcpy.AddError('Could not quick load input classified GeoTiff.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_high, xr.Dataset):
            arcpy.AddError('Input GeoTiff must be an xr dataset.')
            return
        elif len(ds_high) == 0:
            arcpy.AddError('Input GeoTiff has no data/variables/bands.')
            return
        elif len(ds_high) != 1:
            arcpy.AddError('Input GeoTiff has multiple bands.')
            return
        elif 'x' not in list(ds_high.coords) or 'y' not in list(ds_high.coords):
            arcpy.AddError('Input GeoTiff must have x, y coords.')
            return
        elif 'x' not in list(ds_high.dims) or 'y' not in list(ds_high.dims):
            arcpy.AddError('Input GeoTiff must have x, y dimensions.')
            return
        elif len(ds_high['x']) == 0 or len(ds_high['y']) == 0:
            arcpy.AddError('Input GeoTiff must have at least one x, y index.')
            return
        elif ds_high.attrs == {}:
            arcpy.AddError('GeoTiff attributes not found. GeoTiff must have attributes.')
            return
        elif not hasattr(ds_high, 'crs'):
            arcpy.AddError('GeoTiff CRS attribute not found. CRS required.')
            return
        elif '3577' not in ds_high.crs:
            arcpy.AddError('GeoTiff CRS is not EPSG:3577. EPSG:3577 required.')
            return
        elif not hasattr(ds_high, 'nodatavals'):
            arcpy.AddError('GeoTiff nodatavals attribute not found.')
            return
        elif 'int' not in str(ds_high.to_array().dtype):
            arcpy.AddError('GeoTiff is not an integer type. Please convert.')
            return
        elif np.nan in ds_high.to_array():
            arcpy.AddError('GeoTiff contains reserved value nan. Please convert.')
            return
        elif -999 in ds_high.to_array():
            arcpy.AddWarning('GeoTiff contains reserved value -999, will be considered as NoData.')
            pass

        try:
            # do proper load with dask, set nodata to -999
            ds_high = satfetcher.load_local_rasters(rast_path_list=in_high_tif,
                                                    use_dask=True,
                                                    conform_nodata_to=-999)

            # rename first and only band, manually build attributes
            ds_high = ds_high.rename({list(ds_high)[0]: 'classes'})
            ds_high = tools.manual_create_xr_attrs(ds=ds_high)

        except Exception as e:
            arcpy.AddError('Could not properly load classified GeoTiff, see messages.')
            arcpy.AddMessage(str(e))
            return

        # check if not all nodata (-999)
        if (ds_high.to_array() == -999).all():
            arcpy.AddError('Input classified GeoTiff is completely empty.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Clipping classified GeoTiff to satellite NetCDF...')
        arcpy.SetProgressorPosition(11)

        # check extents overlap
        if not tools.all_xr_intersect([ds_low, ds_high]):
            arcpy.AddError('Not all input layers intersect.')
            return

        try:
            # clip classified geotiff extent to netcdf
            ds_high = tools.clip_xr_a_to_xr_b(ds_a=ds_high,
                                              ds_b=ds_low)

        except Exception as e:
            arcpy.AddError('Could not clip GeoTiff to NetCDF, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing GeoTiff into memory, please wait...')
        arcpy.SetProgressorPosition(12)

        try:
            # compute geotiff!
            ds_high = ds_high.compute()
        except Exception as e:
            arcpy.AddError('Could not compute GeoTiff, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # ensure geotiff dataset still integer and not empty
        if 'int' not in str(ds_high.to_array().dtype):
            arcpy.AddError('GeoTiff was unable to maintain integer type.')
            return
        elif (ds_high.to_array() == -999).all():
            arcpy.AddError('GeoTiff is completely empty.')
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Generating stratified random samples...')

        # ensure requested classes valid
        if in_classes is None:
            arcpy.AddError('No classes were selected.')
            return

        # prepare requested classes from ui
        in_classes = in_classes.replace('Class: ', '').replace("'", '')
        in_classes = [int(c) for c in in_classes.split(';')]

        # get all available classes in dataset
        all_classes = list(np.unique(ds_high.to_array()))

        # clean and check both class arrays
        for arr in [in_classes, all_classes]:

            # remove nodata if exists
            if -999 in arr:
                arr.remove(-999)

            # check something remains
            if arr is None or len(arr) == 0:
                arcpy.AddError('No classes were obtained from selection and/or dataset.')
                return

        # check if more than one non-nodata classes in geotiff
        if len(all_classes) < 2:
            arcpy.AddError('More than one GeoTiff class required.')
            return

        # ensure all requested classes still available
        for c in in_classes:
            if c not in all_classes:
                arcpy.AddError('Class {} not within satellite NetCDF extent.'.format(c))
                return

        # check number of samples
        if in_num_samples < 1:
            arcpy.AddError('Number of samples must be 1 or more.')
            return

        try:
            # generate stratified random samples (number per class)
            df_samples = vegfrax.build_random_samples(ds_low=ds_low,
                                                      ds_high=ds_high,
                                                      classes=all_classes,
                                                      num_samples=in_num_samples)
        except Exception as e:
            arcpy.AddError('Could not build random samples, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # warn (but continue) if undersampled
        if len(df_samples) < len(all_classes) * in_num_samples:
            arcpy.AddWarning('Caution, smaller classes may be under-sampled.')

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Extracting tasselled cap values...')
        arcpy.SetProgressorPosition(13)

        try:
            # extract tasselled cap band values at each random sample
            df_samples = vegfrax.extract_xr_low_values(df=df_samples,
                                                       ds=ds_low)
        except Exception as e:
            arcpy.AddError('Could not extract values, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # ensure we have samples still
        if len(df_samples) == 0:
            arcpy.AddError('No tasselled cap values were extracted.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Building class fraction arrays...')
        arcpy.SetProgressorPosition(14)

        # ensure max nodata is valid
        if in_max_nodata < 0 or in_max_nodata > 1:
            arcpy.AddError('Maximum NoData value must be >= 0 and <= 1.')
            return

        try:
            # build class fraction windows and arrays
            df_samples = vegfrax.build_class_fractions(df=df_samples,
                                                       ds_low=ds_low,
                                                       ds_high=ds_high,
                                                       max_nodata=in_max_nodata)
        except Exception as e:
            arcpy.AddError('Could not build class fraction arrays, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Performing fractional cover analysis...')

        # check aggregate classes valid
        if in_agg_classes not in [True, False]:
            arcpy.AddError('Combine classes is invalid.')
            return

        # check model parameters are valid
        if in_num_estimator < 1:
            arcpy.AddError('Number of model estimators not between 1 and 10000.')
            return
        elif in_criterion not in ['Mean Squared Error', 'Mean Absolute Error', 'Poisson']:
            arcpy.AddError('Criterion not supported.')
            return
        elif in_max_depth is not None and in_max_depth < 1:
            arcpy.AddError('Maximum depth must be empty or > 0.')
            return
        elif in_max_features not in ['Auto', 'Log2']:
            arcpy.AddError('Maximum features must be Auto or Log2.')
            return
        elif in_bootstrap not in [True, False]:
            arcpy.AddError('Boostrap must be either True or False.')
            return

        # prepare criterion value
        if 'Squared' in in_criterion:
            in_criterion = 'squared_error'
        elif 'Absolute' in in_criterion:
            in_criterion = 'absolute_error'
        else:
            in_criterion = 'poisson'

        # prepare options
        options = {
            'n_estimators': in_num_estimator,
            'criterion': in_criterion,
            'max_depth': in_max_depth,
            'max_features': in_max_features.lower(),
            'bootstrap': in_bootstrap
        }

        try:
            # perform fca and accuracy result message
            ds_frax, result = vegfrax.perform_fcover_analysis(df=df_samples,
                                                              ds=ds_low,
                                                              classes=in_classes,
                                                              combine=in_agg_classes,
                                                              options=options)
            # display accuracy results
            arcpy.AddMessage(result)
        except Exception as e:
            arcpy.AddError('Could not perform fractional cover analysis, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check frax dataset if all nan
        if ds_frax.to_array().isnull().all():
            arcpy.AddError('Fractional cover dataset result is empty.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Smoothing dataset, if requested...')
        arcpy.SetProgressorPosition(15)

        # check if smooth is valid
        if in_smooth not in [True, False]:
            arcpy.AddError('Smooth output is invalid.')
            return

        # if requested...
        if in_smooth:
            try:
                # smooth via median filter
                ds_frax = vegfrax.smooth(ds_frax)
            except Exception as e:
                arcpy.AddError('Could not smooth dataset, see messages for details.')
                arcpy.AddMessage(str(e))
                return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(16)

        # append attrbutes on to dataset and bands
        ds_frax.attrs = ds_attrs
        ds_frax['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_frax:
            ds_frax[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(17)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_frax, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(18)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing fractional layers if exist
                for layer in m.listLayers():
                    if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'fractions':
                        m.removeLayer(layer)

                # setup a group layer via template
                grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
                grp = m.addLayer(grp_lyr)[0]
                grp.name = 'fractions'

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'fractions' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # iter each var and export a seperate tif
                tif_list = []
                for var in ds_frax:
                    # create temp netcdf for one var (prevents 2.9 bug)
                    with tempfile.NamedTemporaryFile() as tmp:
                        tmp_nc = '{}_{}.nc'.format(tmp.name, var)
                        ds_frax[[var]].to_netcdf(tmp_nc)

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
                            # apply percent clip type and threshold
                            sym.colorizer.stretchType = 'PercentClip'
                            sym.colorizer.minPercent = 0.1
                            sym.colorizer.maxPercent = 0.1

                            # create color map and apply
                            cmap = aprx.listColorRamps('Temperature')[0]
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
        arcpy.SetProgressorPosition(19)

        # close and del dataset
        ds_low.close()
        ds_high.close()
        ds_frax.close()
        del ds_low
        del ds_high
        del ds_frax

        # notify user
        arcpy.AddMessage('Generated VegFrax successfully.')

        return