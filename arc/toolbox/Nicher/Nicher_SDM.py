# safe imports
import os
import datetime
import numpy as np
import arcpy
import xarray as xr
import dask

from shared import arc, satfetcher, tools
from modules import nicher
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class Nicher_SDM:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'Nicher SDM'
        self.description = 'Generate a species distribution model (SDM) using ' \
                           'a shapefile of known species occurrence field points ' \
                           'and two or more digital elevation model (DEM)-derived ' \
                           'topographic variables representing potential habitat.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input continous tif(s)
        par_in_continuous_tifs = arcpy.Parameter(
            displayName='Input GeoTiff(s) of continuous variables',
            name='in_continuous_tifs',
            datatype='GPRasterLayer',
            parameterType='Optional',
            direction='Input',
            multiValue=True)

        # input categorical tif(s)
        par_in_categorical_tifs = arcpy.Parameter(
            displayName='Input GeoTiff(s) of categorical variables',
            name='in_categorical_tifs',
            datatype='GPRasterLayer',
            parameterType='Optional',
            direction='Input',
            multiValue=True)

        # output netcdf file
        par_out_nc_path = arcpy.Parameter(
            displayName='Output Nicher NetCDF file',
            name='out_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_path.filter.list = ['nc']

        # occurrence points
        par_occurrence_feat = arcpy.Parameter(
            displayName='Occurrence point feature',
            name='in_occurrence_feat',
            datatype='GPFeatureLayer',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_occurrence_feat.filter.list = ['Point']

        # number of pseudoabsences
        par_num_absence = arcpy.Parameter(
            displayName='Number of pseudo-absence points',
            name='in_num_pseudos',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_num_absence.filter.type = 'Range'
        par_num_absence.filter.list = [1, 9999999]
        par_num_absence.value = 1000

        # exclusion buffer
        par_exclusion_buffer = arcpy.Parameter(
            displayName='Exclusion buffer (in metres)',
            name='in_exclusion_buffer',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_exclusion_buffer.filter.type = 'Range'
        par_exclusion_buffer.filter.list = [1, 9999999]
        par_exclusion_buffer.value = 500

        # equalise pseudoabsence
        par_equalise_absence = arcpy.Parameter(
            displayName='Equalise number of pseudoabsence points',
            name='in_equalise_pseudos',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_equalise_absence.value = False

        # input test ratio
        par_test_ratio = arcpy.Parameter(
            displayName='Proportion of data for testing',
            name='in_test_ratio',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_test_ratio.filter.type = 'Range'
        par_test_ratio.filter.list = [0, 1]
        par_test_ratio.value = 0.1

        # input resample
        par_resample = arcpy.Parameter(
            displayName='Resample resolution',
            name='in_resample',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        resample_to = ['Highest Resolution', 'Lowest Resolution']
        par_resample.filter.type = 'ValueList'
        par_resample.filter.list = resample_to
        par_resample.value = 'Lowest Resolution'

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
        par_criterion.filter.list = ['Gini', 'Entropy']
        par_criterion.value = 'Gini'

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
            par_in_continuous_tifs,
            par_in_categorical_tifs,
            par_out_nc_path,
            par_occurrence_feat,
            par_num_absence,
            par_exclusion_buffer,
            par_equalise_absence,
            par_test_ratio,
            par_resample,
            par_num_estimators,
            par_criterion,
            par_max_depth,
            par_max_features,
            par_boostrap,
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

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the Nicher SDM module.
        """

        # grab parameter values
        in_cont_tifs = parameters[0].valueAsText  # continous tifs
        in_cate_tifs = parameters[1].valueAsText  # categorical tifs
        out_nc = parameters[2].valueAsText  # output netcdf
        in_occurrence_feat = parameters[3]  # occurrence point shapefile
        in_num_absence = parameters[4].value  # num absences
        in_exclusion_buffer = parameters[5].value  # exclusion buffer
        in_equalise_absence = parameters[6].value  # equalise absences
        in_test_ratio = parameters[7].value  # test ratio
        in_resample = parameters[8].value  # resample
        in_num_estimator = parameters[9].value  # number of estimators
        in_criterion = parameters[10].value  # criterion type
        in_max_depth = parameters[11].value  # max tree depth
        in_max_features = parameters[12].value  # maximum features
        in_bootstrap = parameters[13].value  # boostrap
        in_add_result_to_map = parameters[14].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Nicher SDM.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=14)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Preparing GeoTiffs...')
        arcpy.SetProgressorPosition(1)

        # check if at least on layer provided
        if in_cont_tifs is None and in_cate_tifs is None:
            arcpy.AddError('Must provide at least one GeoTiff.')
            return

        # convert continuous tifs to list if exists
        cont_tifs = []
        if in_cont_tifs is not None:
            for tif in [t.replace("'", '') for t in in_cont_tifs.split(';')]:
                desc = arcpy.Describe(tif)
                cont_tifs.append(os.path.join(desc.path, desc.name))

        # convert categorical tifs to list if exists
        cate_tifs = []
        if in_cate_tifs is not None:
            for tif in [t.replace("'", '') for t in in_cate_tifs.split(';')]:
                desc = arcpy.Describe(tif)
                cate_tifs.append(os.path.join(desc.path, desc.name))

        # ensure all tifs are unique
        u, c = np.unique(cont_tifs + cate_tifs, return_counts=True)
        if len(u[c > 1]) > 0:
            arcpy.AddError('Duplicate input GeoTiffs provided.')
            return

        # ensure we have at least two variables
        if len(cont_tifs + cate_tifs) < 2:
            arcpy.AddError('At least two variables are required.')
            return

        # ensure all files are tifs
        for tif in cont_tifs + cate_tifs:
            if not tif.endswith('.tif'):
                arcpy.AddError('Only GeoTiff raster type supported.')
                return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Loading and checking GeoTiffs...')
        arcpy.SetProgressorPosition(2)

        # iterate layers for check
        ds_list = []
        for tif in cont_tifs + cate_tifs:
            try:
                ds = xr.open_rasterio(tif)
                ds = ds.to_dataset(dim='band')
            except Exception as e:
                arcpy.AddError('Could not quick load input GeoTiff {}.'.format(tif))
                arcpy.AddMessage(str(e))
                return

            # check xr type, vars, coords, dims, attrs
            if not isinstance(ds, xr.Dataset):
                arcpy.AddError('Input GeoTiff must be an xr dataset.')
                return
            elif not tif.endswith('.tif'):
                arcpy.AddError('File is not a GeoTiff.')
                return
            elif len(ds) == 0:
                arcpy.AddError('Input GeoTiff has no data/variables/bands.')
                return
            elif len(ds) != 1:
                arcpy.AddError('Input GeoTiff has multiple bands.')
                return
            elif 'x' not in list(ds.coords) or 'y' not in list(ds.coords):
                arcpy.AddError('Input GeoTiff must have x, y coords.')
                return
            elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
                arcpy.AddError('Input GeoTiff must have x, y dimensions.')
                return
            elif len(ds['x']) == 0 or len(ds['y']) == 0:
                arcpy.AddError('Input GeoTiff must have at least one x, y index.')
                return
            elif ds.attrs == {}:
                arcpy.AddError('GeoTiff attributes not found. GeoTiff must have attributes.')
                return
            elif not hasattr(ds, 'crs'):
                arcpy.AddError('GeoTiff CRS attribute not found. CRS required.')
                return
            elif '3577' not in ds.crs:
                arcpy.AddError('GeoTiff CRS is not EPSG:3577. EPSG:3577 required.')
                return
            elif not hasattr(ds, 'nodatavals'):
                arcpy.AddError('GeoTiff nodatavals attribute not found.')
                return

            try:
                # do proper load with dask, set nodata to nan
                ds = satfetcher.load_local_rasters(rast_path_list=tif,
                                                   use_dask=True,
                                                   conform_nodata_to=np.nan)
                ds = tools.manual_create_xr_attrs(ds=ds)
            except Exception as e:
                arcpy.AddError('Could not properly load input GeoTiff {}.'.format(tif))
                arcpy.AddMessage(str(e))
                return

            # check if xr is all nan
            if ds.to_array().isnull().all():
                arcpy.AddError('Input GeoTiff is completely null.')
                return

            # append to dataset list
            ds_list.append(ds)

        # check xr list isnt empty
        if len(ds_list) == 0:
            arcpy.AddError('No GeoTiffs were successfully loaded.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Conforming GeoTiffs via resampling...')
        arcpy.SetProgressorPosition(3)

        # check extents overlap
        if not tools.all_xr_intersect(ds_list):
            arcpy.AddError('Not all input layers intersect.')
            return

            # check resample
        if in_resample not in ['Lowest Resolution', 'Highest Resolution']:
            arcpy.AddError('Resample type not supported.')
            return

        try:
            # select target resolution dataset
            ds_target = tools.get_target_res_xr(ds_list,
                                                in_resample)
        except Exception as e:
            arcpy.AddError('Could not get target GeoTiff resolution.')
            arcpy.AddMessage(str(e))
            return

        # check target xr captured
        if ds_target is None:
            arcpy.AddError('Could not obtain optimal GeoTiff resolution.')
            return

        try:
            # resample all datasets to target dataset
            for idx in range(len(ds_list)):
                ds_list[idx] = tools.resample_xr(ds_from=ds_list[idx],
                                                 ds_to=ds_target,
                                                 resampling='nearest')

                # squeeze to be safe!
                ds_list[idx] = ds_list[idx].squeeze(drop=True)
        except Exception as e:
            arcpy.AddError('Could not resample GeoTiffs.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Combining GeoTiffs together...')
        arcpy.SetProgressorPosition(4)

        try:
            # merge vars into one dataset, fix attrs
            ds = xr.merge(ds_list)
            ds = tools.manual_create_xr_attrs(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not combine GeoTiffs.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Reducing data to smallest GeoTiff extent...')
        arcpy.SetProgressorPosition(5)

        try:
            # ensure bounding box is fixed to smallest mbr
            ds = tools.remove_nan_xr_bounds(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not reduce to smallest GeoTiff extent.')
            arcpy.AddMessage(str(e))
            return

            # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('GeoTiff data is empty. Please check GeoTiffs.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing GeoTiffs into memory, please wait...')
        arcpy.SetProgressorPosition(6)

        try:
            # compute!
            ds = ds.compute()
        except Exception as e:
            arcpy.AddError('Could not compute GeoTiffs. See messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('GeoTiff data is empty. Please check GeoTiffs.')
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Preparing occurrence points...')
        arcpy.SetProgressorPosition(7)

        # check if shapefile provided
        if in_occurrence_feat.value is None:
            arcpy.AddError('No occurrence point shapefile provided.')
            return

        # ensure dataset has a required nodatavals attribute
        ds.attrs.update({'nodatavals': np.nan})

        try:
            # get full path to feature
            desc = arcpy.Describe(in_occurrence_feat)
            in_occurrence_feat = os.path.join(desc.path, desc.name)

            # check shapefile is valid
            if desc.shapeType != 'Point':
                arcpy.AddError('Shapefile is not a point type.')
                return
            elif desc.spatialReference.factoryCode != 3577:
                arcpy.AddError('Shapefile CRS is not EPSG:3577. EPSG:3577 required.')
                return
            elif int(arcpy.GetCount_management(in_occurrence_feat)[0]) == 0:
                arcpy.AddError('Shapefile has no points.')
                return

            # read shapefile (arcpy), convert to x, y dataframe, check validity
            df_pres = arc.read_shp_for_nicher(in_occurrence_feat)

            # extract values for presence points, keep x, y
            df_pres = tools.extract_xr_values(ds=ds,
                                              coords=df_pres,
                                              keep_xy=True,
                                              res_factor=3)

            # ensure some non-nan values exist
            if df_pres.drop(columns=['x', 'y']).isnull().values.all():
                arcpy.AddError('No occurrence points intersected area of interest.')
                return

                # remove any nodata (nan) values
            df_pres = tools.remove_nodata_records(df_records=df_pres,
                                                  nodata_value=np.nan)
        except Exception as e:
            arcpy.AddError('Could not prepare occurrence points, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # ensure we have presence points remaining
        if len(df_pres) == 0:
            arcpy.AddError('No occurrence points remaining after extraction.')
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Preparing pseudoabsence points...')
        arcpy.SetProgressorPosition(8)

        # check exclusion buffer size and num pseudoabsence
        if in_exclusion_buffer < 1:
            arcpy.AddError('Exclusion buffer not between 1 and 1000000 metres.')
            return
        elif in_num_absence < 1:
            arcpy.AddError('Number of psuedoabsences not between 1 and 10000.')
            return

        # prepare random sample size based on selection
        num_to_sample = in_num_absence
        if in_equalise_absence is True:
            num_to_sample = len(df_pres)

        try:
            # generate absences pixels and presence buffer (arcpy)
            df_abse = arc.generate_absences_for_nicher(df_pres=df_pres,
                                                       ds=ds,
                                                       buff_m=in_exclusion_buffer)

            # randomly sample pseudoabsence points (these are all non-nan)
            df_abse = nicher.random_sample_absences(df_abse=df_abse,
                                                    num_to_sample=num_to_sample)

            # extract values for absence points, keep x, y
            df_abse = tools.extract_xr_values(ds=ds,
                                              coords=df_abse,
                                              keep_xy=True,
                                              res_factor=3)

            # remove any nodata (nan) values
            df_abse = tools.remove_nodata_records(df_records=df_abse,
                                                  nodata_value=np.nan)
        except Exception as e:
            arcpy.AddError('Could not generate psuedoabsence points, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # ensure we have absence points
        if len(df_abse) == 0:
            arcpy.AddError('No pseudoabsence points remain.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Combining presence and absence points...')
        arcpy.SetProgressorPosition(9)

        # get list of all continuous var names
        cont_vars = []
        for c in cont_tifs:
            fp = os.path.splitext(c)[0]
            cont_vars.append(os.path.basename(fp))

        # get list of all categorical var names
        cate_vars = []
        for c in cate_tifs:
            fp = os.path.splitext(c)[0]
            cate_vars.append(os.path.basename(fp))

        try:
            # combine pres/abse data and add "pres_abse" column
            df_pres_abse = nicher.combine_pres_abse_records(df_presence=df_pres,
                                                            df_absence=df_abse)

            # drop all useless vars
            keep_vars = ['pres_abse'] + cont_vars + cate_vars
            for col in df_pres_abse.columns:
                if col not in keep_vars:
                    df_pres_abse = df_pres_abse.drop(columns=[col], errors='ignore')
        except Exception as e:
            arcpy.AddError('Could not combine presence and absence data.')
            arcpy.AddMessage(str(e))
            return

            # check we have columns and data
        if len(df_pres_abse) == 0 or len(df_pres_abse.columns) == 0:
            arcpy.AddError('No columns and/or data exists after processing occurence points.')
            return

        # warn (but proceed) if number records low
        if len(df_pres_abse.index) < 25:
            arcpy.AddWarning('Low number of presence/absence points available.')

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Calculating exploratory statistics...')
        arcpy.SetProgressorPosition(10)

        # if two or more continuous vars, proceed...
        if len(cont_vars) > 1:
            try:
                # make temporary df and drop categoricals, ignore if non
                df_tmp = df_pres_abse.copy(deep=True)
                df_tmp = df_tmp.drop(cate_vars, errors='ignore')

                # calculate pearson correlation as text and display
                result = nicher.generate_correlation_matrix(df_records=df_tmp)
                arcpy.AddMessage(result)

                # calculate variance impact factor as text and display
                result = nicher.generate_vif_scores(df_records=df_tmp)
                arcpy.AddMessage(result)

            except Exception as e:
                arcpy.AddError('Could not calculate exploratory statistics.')
                arcpy.AddMessage(str(e))
                return

                # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Performing species distribution modelling...')

        # check parameters are valid
        if in_num_estimator < 1:
            arcpy.AddError('Number of model estimators not between 1 and 10000.')
            return
        elif in_criterion not in ['Gini', 'Entropy']:
            arcpy.AddError('Criterion must be Gini or Entropy.')
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

        # prepare options
        options = {
            'n_estimators': in_num_estimator,
            'criterion': in_criterion.lower(),
            'max_depth': in_max_depth,
            'max_features': in_max_features.lower(),
            'bootstrap': in_bootstrap
        }

        # check test ratio
        if in_test_ratio <= 0 or in_test_ratio >= 1:
            arcpy.AddError('Testing ratio must be between 0 and 1.')
            return

        try:
            # generate sdm dataset and capture accuracy measures (result)
            ds_sdm, result = nicher.perform_sdm(ds=ds,
                                                df=df_pres_abse,
                                                test_ratio=in_test_ratio,
                                                options=options)
            # display accuracy results
            arcpy.AddMessage(result)
        except Exception as e:
            arcpy.AddError('Could not generate model, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check sdm dataset if all nan
        if ds_sdm.to_array().isnull().all():
            arcpy.AddError('SDM dataset result is empty.')
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(11)

        try:
            # manually create attrs for dataset (geotiffs lacking)
            ds_sdm = tools.manual_create_xr_attrs(ds_sdm)
            ds_sdm.attrs.update({'nodatavals': np.nan})
        except Exception as e:
            arcpy.AddError('Could not append attributes onto dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(12)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_sdm, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(13)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing sdm layer if exist
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'sdm.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'sdm' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'sdm.crf')
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
                layer = m.listLayers('sdm.crf')[0]
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
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(14)

        # close and del satellite dataset
        ds.close()
        del ds

        # close and del satellite dataset
        ds_sdm.close()
        del ds_sdm

        # notify user
        arcpy.AddMessage('Generated Nicher SDM successfully.')

        return