import arcpy
import os
import datetime
import numpy as np
import xarray as xr

from modules import gdvspectra, cog
from shared import arc, satfetcher, tools

class GDVSpectra_Likelihood:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'GDVSpectra Likelihood'
        self.description = 'GDVSpectra Likelihood derives potential groundwater ' \
                           'dependent vegetation (GDV) areas from three or more years of ' \
                           'Landsat or Sentinel NetCDF data. This functions results in ' \
                           'a NetCDF of GDV likelihood with values ranging ' \
                           'from 0 to 1, with 1 being highest probability of GDV.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input netcdf data file
        par_nc_file = arcpy.Parameter(
            displayName='Input satellite NetCDF file',
            name='in_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_nc_file.filter.list = ['nc']

        # output netcdf location
        par_out_nc_file = arcpy.Parameter(
            displayName='Output GDV Likelihood NetCDF file',
            name='out_likelihood_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # input wet month(s)
        par_wet_months = arcpy.Parameter(
            displayName='Wet month(s)',
            name='in_wet_months',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Wet Period',
            multiValue=True)
        par_wet_months.filter.type = 'ValueList'
        par_wet_months.filter.list = [m for m in range(1, 13)]
        par_wet_months.value = [1, 2, 3]

        # input dry month(s)
        par_dry_months = arcpy.Parameter(
            displayName='Dry month(s)',
            name='in_dry_months',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Dry Period',
            multiValue=True)
        par_dry_months.filter.type = 'ValueList'
        par_dry_months.filter.list = [m for m in range(1, 13)]
        par_dry_months.value = [9, 10, 11]

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
            'kNDVI',
            'TCG'
        ]
        par_veg_idx.value = 'MAVI'

        # input moisture index
        par_mst_idx = arcpy.Parameter(
            displayName='Moisture index',
            name='in_mst_idx',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_mst_idx.filter.type = 'ValueList'
        par_mst_idx.filter.list = ['NDMI', 'GVMI']
        par_mst_idx.value = 'NDMI'

        # aggregate likelihood layers
        par_aggregate = arcpy.Parameter(
            displayName='Combine outputs',
            name='in_aggregate',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_aggregate.value = False

        # pvalue for zscore
        par_zscore_pvalue = arcpy.Parameter(
            displayName='Z-Score p-value',
            name='in_zscore_pvalue',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            category='Outlier Correction',
            multiValue=False)
        par_zscore_pvalue.filter.type = 'ValueList'
        par_zscore_pvalue.filter.list = [0.01, 0.05, 0.1]
        par_zscore_pvalue.value = None

        # q upper for standardisation
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

        # q lower for standardisation
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

        # mask flags
        par_fmask_flags = arcpy.Parameter(displayName='Include pixels flags',
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

        # input interpolate
        par_interpolate = arcpy.Parameter(
            displayName='Interpolate NoData pixels',
            name='in_interpolate',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            category='Satellite Quality Options',
            multiValue=False)
        par_interpolate.value = True

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
            par_nc_file,
            par_out_nc_file,
            par_wet_months,
            par_dry_months,
            par_veg_idx,
            par_mst_idx,
            par_aggregate,
            par_zscore_pvalue,
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
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra Likelihood module.
        """

        # grab parameter values
        in_nc = parameters[0].valueAsText  # raw input satellite netcdf
        out_nc = parameters[1].valueAsText  # output gdv likelihood netcdf
        in_wet_months = parameters[2].valueAsText  # wet months
        in_dry_months = parameters[3].valueAsText  # dry months
        in_veg_idx = parameters[4].value  # vege index name
        in_mst_idx = parameters[5].value  # moisture index name
        in_aggregate = parameters[6].value  # aggregate output
        in_zscore_pvalue = parameters[7].value  # zscore pvalue
        in_ivt_qupper = parameters[8].value  # upper quantile for standardisation
        in_ivt_qlower = parameters[9].value  # lower quantile for standardisation
        in_fmask_flags = parameters[10].valueAsText  # fmask flag values
        in_max_cloud = parameters[11].value  # max cloud percentage
        in_interpolate = parameters[12].value  # interpolate missing pixels
        in_add_result_to_map = parameters[13].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra Likelihood.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=20)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking netcdf...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of netcdf for checking
            ds = xr.open_dataset(in_nc)
        except Exception as e:
            arcpy.AddError('Could not quick load input satellite NetCDF data.')
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
        elif len(ds.groupby('time.year')) < 3:
            arcpy.AddError('Input NetCDF must have >= 3 years of data.')
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
        arcpy.SetProgressorLabel('Reducing dataset to wet and dry months...')
        arcpy.SetProgressorPosition(6)

        # prepare wet, dry season lists
        if in_wet_months == '' or in_dry_months == '':
            arcpy.AddError('Must include at least one wet and dry month.')
            return

        # unpack months
        wet_month = [int(e) for e in in_wet_months.split(';')]
        dry_month = [int(e) for e in in_dry_months.split(';')]

        # check if same months in wet and dry
        for v in wet_month:
            if v in dry_month:
                arcpy.AddError('Cannot use same month in wet and dry months.')
                return

        try:
            # reduce xr dataset into only wet, dry months
            ds = gdvspectra.subset_months(ds=ds,
                                          month=wet_month + dry_month,
                                          inplace=True)
        except Exception as e:
            arcpy.AddError('Could not subset dataset into wet and dry months.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating vegetation and moisture indices...')
        arcpy.SetProgressorPosition(7)

        # check if veg idx supported
        if in_veg_idx.lower() not in ['ndvi', 'evi', 'savi', 'msavi', 'slavi', 'mavi', 'kndvi', 'tcg']:
            arcpy.AddError('Vegetation index not supported.')
            return
        elif in_mst_idx.lower() not in ['ndmi', 'gvmi']:
            arcpy.AddError('Moisture index not supported.')
            return

        try:
            # calculate vegetation and moisture index
            ds = tools.calculate_indices(ds=ds,
                                         index=[in_veg_idx.lower(), in_mst_idx.lower()],
                                         custom_name=['veg_idx', 'mst_idx'],
                                         rescale=True,
                                         drop=True)

            # add band attrs back on
            ds['veg_idx'].attrs = ds_band_attrs
            ds['mst_idx'].attrs = ds_band_attrs

        except Exception as e:
            arcpy.AddError('Could not calculate indices.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(8)

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
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Interpolating dataset, if requested...')
        arcpy.SetProgressorPosition(9)

        # if requested...
        if in_interpolate:
            try:
                # interpolate along time dimension (linear)
                ds = tools.perform_interp(ds=ds, method='full')
            except Exception as e:
                arcpy.AddError('Could not interpolate dataset.')
                arcpy.AddMessage(str(e))
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Resampling dataset to annual wet/dry medians...')
        arcpy.SetProgressorPosition(10)

        # extract datetimes for wet and dry seasons
        dts_wet = ds['time'].where(ds['time.month'].isin(wet_month), drop=True)
        dts_dry = ds['time'].where(ds['time.month'].isin(dry_month), drop=True)

        # check if wet/dry months exist in the dataset, arent all empty
        if len(dts_wet) == 0 or len(dts_dry) == 0:
            arcpy.AddError('No wet and/or dry months captured in NetCDF.')
            return
        elif ds.sel(time=dts_wet).to_array().isnull().all():
            arcpy.AddError('Entire wet season is devoid of values in NetCDF.')
            return
        elif ds.sel(time=dts_dry).to_array().isnull().all():
            arcpy.AddError('Entire dry season is devoid of values in NetCDF.')
            return

        try:
            # resample data to annual seasons
            ds = gdvspectra.resample_to_wet_dry_medians(ds=ds,
                                                        wet_month=wet_month,
                                                        dry_month=dry_month,
                                                        inplace=True)
        except Exception as e:
            arcpy.AddError('Could not resample annualised wet and dry seasons.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Removing outliers, if requested...')
        arcpy.SetProgressorPosition(11)

        # prepare zscore selection
        if in_zscore_pvalue not in {0.01, 0.05, 0.1, None}:
            arcpy.AddWarning('Z-score value not supported. Setting to default.')
            in_zscore_pvalue = None

        # if requested...
        if in_zscore_pvalue is not None:
            try:
                # remove outliers
                ds = gdvspectra.nullify_wet_dry_outliers(ds=ds,
                                                         wet_month=wet_month,
                                                         dry_month=dry_month,
                                                         p_value=in_zscore_pvalue,
                                                         inplace=True)
            except Exception as e:
                arcpy.AddError('Could not remove outliers.')
                arcpy.AddMessage(str(e))
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Cleaning years with insufficient seasonality...')
        arcpy.SetProgressorPosition(12)

        try:
            # remove any years missing wet, dry season
            ds = gdvspectra.drop_incomplete_wet_dry_years(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not drop years with insufficient seasons.')
            arcpy.AddMessage(str(e))
            return

        # check if we still have sufficient number of years
        if len(ds.groupby('time.year')) < 3:
            arcpy.AddError('Input NetCDF needs more years. Expand time range in NetCDF.')
            return

        try:
            # fill any empty first, last years using manual back/forward fill
            ds = gdvspectra.fill_empty_wet_dry_edges(ds=ds,
                                                     wet_month=wet_month,
                                                     dry_month=dry_month,
                                                     inplace=True)
        except Exception as e:
            arcpy.AddError('Could not fill empty wet and dry edge dates.')
            arcpy.AddMessage(str(e))
            return

        try:
            # interpolate missing values
            ds = gdvspectra.interp_empty_wet_dry(ds=ds,
                                                 wet_month=wet_month,
                                                 dry_month=dry_month,
                                                 method='full',
                                                 inplace=True)
        except Exception as e:
            arcpy.AddError('Could not interpolate empty wet and dry edge dates.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Standardising data to dry season invariant targets...')
        arcpy.SetProgressorPosition(13)

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
            # standardise data to invariant targets derived from dry times
            ds = gdvspectra.standardise_to_dry_targets(ds=ds,
                                                       dry_month=dry_month,
                                                       q_upper=in_ivt_qupper,
                                                       q_lower=in_ivt_qlower,
                                                       inplace=True)
        except Exception as e:
            arcpy.AddError('Could not standardise data to invariant targets.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating seasonal similarity...')
        arcpy.SetProgressorPosition(14)

        try:
            # calculate seasonal similarity
            ds_similarity = gdvspectra.calc_seasonal_similarity(ds=ds,
                                                                wet_month=wet_month,
                                                                dry_month=dry_month,
                                                                q_mask=0.9,
                                                                inplace=True)
        except Exception as e:
            arcpy.AddError('Could not generate similarity.')
            arcpy.AddMessage(str(e))
            return

        # check similarity dataset is not empty
        if ds_similarity.to_array().isnull().all():
            arcpy.AddError('Similarity modelling returned no data.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating GDV Likelihood...')
        arcpy.SetProgressorPosition(15)

        try:
            # calculate gdv likelihood
            ds = gdvspectra.calc_likelihood(ds=ds,
                                            ds_similarity=ds_similarity,
                                            wet_month=wet_month,
                                            dry_month=dry_month)

            # convert dataset back to float32
            ds = ds.astype('float32')

        except Exception as e:
            arcpy.AddError('Could not generate likelihood data.')
            arcpy.AddMessage(str(e))
            return

        # check likelihood dataset is not empty
        if ds.to_array().isnull().all():
            arcpy.AddError('Likelihood modelling returned no data.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Aggreating dataset, if requested...')
        arcpy.SetProgressorPosition(16)

        # if requested...
        if in_aggregate is True:
            try:
                # reducing full dataset down to one median image without time
                ds = ds.median('time')
            except Exception as e:
                arcpy.AddError('Could not aggregate dataset.')
                arcpy.AddMessage(str(e))
                return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(17)

        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(18)

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
        arcpy.SetProgressorPosition(19)

        # if requested...
        if in_add_result_to_map:
            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove likelihood layer if already exists
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'likelihood.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'likelihood' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'likelihood.crf')
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
                layer = m.listLayers('likelihood.crf')[0]
                sym = layer.symbology

                # if layer has stretch coloriser, apply color
                if hasattr(sym, 'colorizer'):
                    if sym.colorizer.type == 'RasterStretchColorizer':
                        # apply percent clip type
                        sym.colorizer.stretchType = 'PercentClip'
                        sym.colorizer.minPercent = 0.5
                        sym.colorizer.maxPercent = 0.5

                        # apply color map
                        cmap = aprx.listColorRamps('Bathymetric Scale')[0]
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
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(20)

        # close main dataset and del datasets
        ds.close()
        del ds

        # close similarity dataset
        ds_similarity.close()
        del ds_similarity

        # notify user
        arcpy.AddMessage('Generated GDV Likelihood successfully.')

        return