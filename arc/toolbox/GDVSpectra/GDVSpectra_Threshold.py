import arcpy
import os
import datetime
import numpy as np
import xarray as xr


from modules import gdvspectra
from shared import arc, satfetcher, tools

class GDVSpectra_Threshold:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'GDVSpectra Threshold'
        self.description = 'Threshold an existing GDV Likelihood NetCDF using a ' \
                           'shapefile of points or standard deviation. The output ' \
                           'is a layer representing areas of high potential GDV ' \
                           'Likelihood only.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input netcdf data file
        par_nc_file = arcpy.Parameter(
            displayName='Input GDV Likelihood NetCDF file',
            name='in_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_nc_file.filter.list = ['nc']

        # output netcdf location
        par_out_nc_file = arcpy.Parameter(
            displayName='Output GDV Threshold NetCDF file',
            name='out_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # aggregate all dates
        par_aggregate = arcpy.Parameter(
            displayName='Combine all input years',
            name='in_aggregate',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_aggregate.value = True

        # set specific years
        par_specific_years = arcpy.Parameter(
            displayName='Specific year(s) to threshold',
            name='in_specific_years',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=True)
        par_specific_years.filter.type = 'ValueList'
        par_specific_years.filter.list = []

        # threshold type
        par_type = arcpy.Parameter(
            displayName='Threshold type',
            name='in_type',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_type.filter.type = 'ValueList'
        par_type.filter.list = ['Standard Deviation', 'Occurrence Points']
        par_type.value = 'Standard Deviation'

        # standard dev
        par_std_dev = arcpy.Parameter(
            displayName='Standard deviation of threshold',
            name='in_std_dev',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_std_dev.filter.type = 'Range'
        par_std_dev.filter.list = [0.0, 10.0]
        par_std_dev.value = 2.0

        # occurrence points
        par_occurrence_feat = arcpy.Parameter(
            displayName='Occurrence point feature',
            name='in_occurrence_feat',
            datatype='GPFeatureLayer',
            parameterType='Optional',
            direction='Input')
        par_occurrence_feat.filter.list = ['Point']

        # field of presence/absence values
        par_pa_column = arcpy.Parameter(
            displayName='Field with presence and absence labels',
            name='in_pa_column',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_pa_column.filter.type = 'ValueList'
        par_pa_column.filter.list = []

        # remove stray pixels
        par_remove_stray = arcpy.Parameter(
            displayName='Remove stray pixels',
            name='in_remove_stray',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            # category='Additional Options',
            multiValue=False)
        par_remove_stray.value = True

        # binarise checkbox
        par_convert_binary = arcpy.Parameter(
            displayName='Binarise result',
            name='in_convert_binary',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            # category='Additional Options',
            multiValue=False)
        par_convert_binary.value = True

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
            par_nc_file,
            par_out_nc_file,
            par_aggregate,
            par_specific_years,
            par_type,
            par_std_dev,
            par_occurrence_feat,
            par_pa_column,
            par_remove_stray,
            par_convert_binary,
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
        global GDVSPECTRA_THRESHOLD

        # unpack global parameter values
        curr_file = GDVSPECTRA_THRESHOLD.get('in_file')
        curr_feat = GDVSPECTRA_THRESHOLD.get('in_feat')

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

                # populate year list with new years, reset selection
                parameters[3].filter.list = dts
                parameters[3].value = None

        # if occurrence point feat added, run
        if parameters[6].value is not None:

            # if global has no matching feat (or first run), reload all
            if curr_feat != parameters[6].valueAsText:
                try:
                    shp_path = parameters[6].valueAsText
                    cols = [f.name for f in arcpy.ListFields(shp_path)]
                except:
                    cols = []

                # populate field list with new names, reset selection
                parameters[7].filter.list = cols
                parameters[7].value = None

        # update global values
        GDVSPECTRA_THRESHOLD = {
            'in_file': parameters[0].valueAsText,
            'in_feat': parameters[6].valueAsText,
        }

        # enable specifc years based on combine checkbox
        if parameters[2].value is False:
            parameters[3].enabled = True
        else:
            parameters[3].enabled = False

        # enable std dev or shapefile and field based on drop down
        if parameters[4].value == 'Standard Deviation':
            parameters[5].enabled = True
            parameters[6].enabled = False
            parameters[7].enabled = False
        else:
            parameters[5].enabled = False
            parameters[6].enabled = True
            parameters[7].enabled = True

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""

        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra Threshold module.
        """
        # grab parameter values
        in_nc = parameters[0].valueAsText  # likelihood netcdf
        out_nc = parameters[1].valueAsText  # output netcdf
        in_aggregate = parameters[2].value  # aggregate dates
        in_specific_years = parameters[3].valueAsText  # set specific year
        in_type = parameters[4].value  # threshold type
        in_std_dev = parameters[5].value  # std dev threshold value
        in_occurrence_feat = parameters[6]  # occurrence shp path
        in_pa_column = parameters[7].value  # occurrence shp pres/abse col
        in_remove_stray = parameters[8].value  # apply salt n pepper -- requires sa
        in_convert_binary = parameters[9].value  # convert thresh to binary 1, nan
        in_add_result_to_map = parameters[10].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra Threshold.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=13)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking netcdf...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of netcdf for checking
            ds = xr.open_dataset(in_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input likelihood NetCDF data.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds, xr.Dataset):
            arcpy.AddError('Input NetCDF must be a xr dataset.')
            return
        elif len(ds) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds.dims or 'y' not in ds.dims:
            arcpy.AddError('Input NetCDF must have x, y dimensions.')
            return
        elif 'x' not in ds.coords or 'y' not in ds.coords:
            arcpy.AddError('Input NetCDF must have x, y coords.')
            return
        elif 'spatial_ref' not in ds.coords:
            arcpy.AddError('Input NetCDF must have a spatial_ref coord.')
            return
        elif len(ds['x']) == 0 or len(ds['y']) == 0:
            arcpy.AddError('Input NetCDF must have at least one x, y index.')
            return
        elif 'like' not in ds:
            arcpy.AddError('Input NetCDF must have a "like" variable. Run GDVSpectra Likelihood.')
            return
        elif 'time' in ds and (not hasattr(ds, 'time.year') or not hasattr(ds, 'time.month')):
            arcpy.AddError('Input NetCDF must have time with year and month component.')
            return
        elif 'time' in ds.dims and 'time' not in ds.coords:
            arcpy.AddError('Input NetCDF has time dimension but not coordinate.')
            return
        elif ds.attrs == {}:
            arcpy.AddError('NetCDF attributes not found. NetCDF must have attributes.')
            return
        elif not hasattr(ds, 'crs'):
            arcpy.AddError('NetCDF CRS attribute not found. CRS required.')
            return
        elif ds.crs != 'EPSG:3577':
            arcpy.AddError('NetCDF CRS is not EPSG:3577. EPSG:3577 required.')
            return
        elif not hasattr(ds, 'nodatavals'):
            arcpy.AddError('NetCDF nodatavals attribute not found.')
            return

            # check if variables (should only be like) are empty
        if ds['like'].isnull().all() or (ds['like'] == 0).all():
            arcpy.AddError('NetCDF "like" variable is empty. Please download again.')
            return

        try:
            # now, do proper open of netcdf (set nodata to nan)
            ds = satfetcher.load_local_nc(nc_path=in_nc,
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
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Grouping dates, if required...')
        arcpy.SetProgressorPosition(3)

        # remove potential datetime duplicates (group by day)
        if 'time' in ds:
            ds = satfetcher.group_by_solar_day(ds)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset based on time, if requested...')
        arcpy.SetProgressorPosition(4)

        # if time is in dataset...
        if 'time' in ds:

            # check aggregate and specified year(s) is valid
            if in_aggregate is None:
                arcpy.AddError('Did not specify aggregate parameter.')
                return
            elif in_aggregate is False and in_specific_years is None:
                arcpy.AddError('Did not provide a specific year.')
                return

            # if specific years set...
            if in_aggregate is False:
                in_specific_years = [int(e) for e in in_specific_years.split(';')]

            # aggregate depending on user choice
            if in_aggregate is True:
                ds = ds.median('time')
            else:
                ds = ds.where(ds['time.year'].isin(in_specific_years), drop=True)
                ds = ds.median('time')

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(5)

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
        arcpy.SetProgressorLabel('Preparing occurrence points, if provided...')
        arcpy.SetProgressorPosition(6)

        # we need nodataval attr, so ensure it exists
        ds.attrs = ds_attrs

        # if requested...
        if in_type == 'Occurrence Points':

            # check if both shapefile and field provided
            if in_occurrence_feat.value is None or in_pa_column is None:
                arcpy.AddError('No occurrence feature and/or field provided.')
                return

            try:
                # get path to feature instead of map layer
                desc = arcpy.Describe(in_occurrence_feat)
                in_occurrence_feat = os.path.join(desc.path, desc.name)

                # check shapefile is valid
                if desc.shapeType != 'Point':
                    arcpy.AddError('Shapefile is not a point type.')
                    return
                elif desc.spatialReference.factoryCode != 3577:
                    arcpy.AddError('Shapefile is not in GDA94 Albers (EPSG: 3577).')
                    return
                elif int(arcpy.GetCount_management(in_occurrence_feat)[0]) == 0:
                    arcpy.AddError('Shapefile has no points.')
                    return
                elif in_pa_column not in [field.name for field in desc.fields]:
                    arcpy.AddError('Shapefile has no {} field.'.format(in_pa_column))
                    return

                # read shapefile via arcpy, convert feat into dataframe of x, y, actual
                df_records = arc.read_shp_for_threshold(in_occurrence_feat=in_occurrence_feat,
                                                        in_pa_column=in_pa_column)

                # intersect points with dataset and extract likelihood values
                df_records = tools.intersect_records_with_xr(ds=ds,
                                                             df_records=df_records,
                                                             extract=True,
                                                             res_factor=3,
                                                             if_nodata='any')

                # rename column to predicted and check
                df_records = df_records.rename(columns={'like': 'predicted'})

                # check if any records intersected dataset
                if len(df_records.index) == 0:
                    arcpy.AddError('No shapefile points intersect GDV likelihood dataset.')
                    return

                # remove any records where vars contain nodata
                df_records = tools.remove_nodata_records(df_records,
                                                         nodata_value=ds.nodatavals)

                # check again if any records exist
                if len(df_records.index) == 0:
                    arcpy.AddError('No shapefile points remain after empty values removed.')
                    return

            except Exception as e:
                arcpy.AddError('Could not read shapefile, see messages for details.')
                arcpy.AddMessage(str(e))
                return

            # check if some 1s and 0s exist
            unq = df_records['actual'].unique()
            if not np.any(unq == 1) or not np.any(unq == 0):
                arcpy.AddError('Insufficient presence/absence points within NetCDF bounds.')
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Thresholding GDV Likelihood...')
        arcpy.SetProgressorPosition(7)

        try:
            # perform thresholding using either shapefile points or std dev
            if in_type == 'Occurrence Points' and df_records is not None:
                ds = gdvspectra.threshold_likelihood(ds=ds,
                                                     df=df_records,
                                                     res_factor=3,
                                                     if_nodata='any')
            else:
                ds = gdvspectra.threshold_likelihood(ds=ds,
                                                     num_stdevs=in_std_dev,
                                                     res_factor=3,
                                                     if_nodata='any')

            # rename var, convert to float32
            ds = ds.rename({'like': 'thresh'}).astype('float32')

        except Exception as e:
            arcpy.AddError('Could not threshold data.')
            arcpy.AddMessage(str(e))
            # print(str(e))
            return

        # check if any data was returned after threshold
        if ds.to_array().isnull().all():
            arcpy.AddError('Threshold returned no values, try modifying threshold.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Removing stray pixels, if requested...')
        arcpy.SetProgressorPosition(8)

        # if requested...
        if in_remove_stray:
            try:
                # remove salt n pepper
                ds = gdvspectra.remove_salt_pepper(ds, iterations=1)
            except Exception as e:
                arcpy.AddError('Could not remove stray pixels.')
                arcpy.AddMessage(str(e))
                return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Binarising values, if requested...')
        arcpy.SetProgressorPosition(9)

        # if requested...
        if in_convert_binary:
            # set all threshold non-nan values to 1
            ds = ds.where(ds.isnull(), 1)

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(10)

        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(11)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(12)

        # if requested...
        if in_add_result_to_map:
            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove threshold layer if already exists
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'likelihood_threshold.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'likelihood_threshold' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'likelihood_threshold.crf')
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
                layer = m.listLayers('likelihood_threshold.crf')[0]
                sym = layer.symbology

                # if layer has stretch coloriser, apply color
                if hasattr(sym, 'colorizer'):

                    # apply percent clip type
                    sym.colorizer.stretchType = 'PercentClip'
                    sym.colorizer.minPercent = 0.25
                    sym.colorizer.maxPercent = 0.25

                    # colorise deopending on binary or continious
                    if in_convert_binary is True:
                        cmap = aprx.listColorRamps('Yellow to Red')[0]
                    else:
                        cmap = aprx.listColorRamps('Bathymetric Scale')[0]

                    # apply colormap
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
        arcpy.SetProgressorPosition(13)

        # close main dataset
        ds.close()
        del ds

        # notify user
        arcpy.AddMessage('Generated GDV Threshold successfully.')

        return