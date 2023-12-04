# safe imports
import os
import datetime
import numpy as np
import arcpy
import xarray as xr
import dask
import tempfile

from shared import satfetcher, tools
from modules import ensemble
from arc.toolbox.globals import GRP_LYR_FILE

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class Ensemble_Model(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'Ensemble Model'
        self.description = 'Combine two or more evidence layers into a ' \
                           'single Dempster-Shafer belief model. The output ' \
                           'produces four layers: belief, disbelief, plausability ' \
                           'and confidence. The benefit of this is areas of ' \
                           'certainty and uncertainty can be derived, providing a ' \
                           'better understanding of where the model can be trusted.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input evidence layers
        par_in_layers = arcpy.Parameter(
            displayName='Input evidence layers',
            name='in_layers',
            datatype='GPValueTable',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_layers.columns = [['DEFile', 'NetCDF File'],
                                 ['GPString', 'Evidence Type']]
        par_in_layers.filters[0].list = ['nc']
        par_in_layers.filters[1].type = 'ValueList'
        par_in_layers.filters[1].list = ['Belief', 'Disbelief']

        # output netcdf
        par_out_nc = arcpy.Parameter(
            displayName='Output ensemble NetCDF file',
            name='out_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc.filter.list = ['nc']

        # resample
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

        # smooth inputs
        par_smooth = arcpy.Parameter(
            displayName='Smooth input layers',
            name='in_smooth',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_smooth.value = False

        # smoothing window size
        par_in_win_size = arcpy.Parameter(
            displayName='Smoothing window size',
            name='in_win_size',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_in_win_size.filter.type = 'Range'
        par_in_win_size.filter.list = [3, 99]
        par_in_win_size.value = 3

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
            par_in_layers,
            par_out_nc,
            par_resample,
            par_smooth,
            par_in_win_size,
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

        # if user has initialised control...
        if parameters[0].value is not None:

            # iter rows...
            rows = []
            for row in parameters[0].value:

                # convert nc object to string if not already
                if not isinstance(row[0], str):
                    row[0] = row[0].value

                # set default if empty and append
                if row[1] == '':
                    row[1] = 'Belief'

                # add to list
                rows.append(row)

            # update value table
            parameters[0].value = rows

        # enable smooth window size
        if parameters[3].value is True:
            parameters[4].enabled = True
        else:
            parameters[4].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the Ensemble Model module.
        """
        # grab parameter values
        in_layers = parameters[0].value  # input layers (as a value array)
        out_nc = parameters[1].valueAsText  # output netcdf
        in_resample = parameters[2].value  # resample resolution
        in_smooth = parameters[3].value  # smooth inputs
        in_win_size = parameters[4].value  # smoothing window size
        in_add_result_to_map = parameters[5].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Ensemble Model.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=11)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking evidence NetCDFs...')
        arcpy.SetProgressorPosition(1)

        # ensure layers are valid
        if in_layers is None or len(in_layers) == 0:
            arcpy.AddError('No input evidence NetCDFs provided.')
            return

        # ensure all layers are unique
        lyrs = [lyr[0].value for lyr in in_layers]
        u, c = np.unique(lyrs, return_counts=True)
        if len(u[c > 1]) > 0:
            arcpy.AddError('Duplicate input NetCDFs provided.')
            return

        # ensure at least one belief and disbelief
        types = [lyr[1] for lyr in in_layers]
        if 'Belief' not in types or 'Disbelief' not in types:
            arcpy.AddError('Must provide at least one belief and disbelief type.')
            return
        elif len(np.unique(types)) != 2:
            arcpy.AddError('Only belief and disbelief types supported.')
            return

        # iterate layers for check
        ds_list = []
        for layer in in_layers:

            try:
                # do quick lazy load of evidence netcdf for checking
                ds = xr.open_dataset(layer[0].value)
            except Exception as e:
                arcpy.AddWarning('Could not quick load evidence NetCDF data, see messages for details.')
                arcpy.AddMessage(str(e))
                return

            # check xr type, vars, coords, dims, attrs
            if not isinstance(ds, xr.Dataset):
                arcpy.AddError('Input NetCDF must be a xr dataset.')
                return
            elif len(ds) == 0:
                arcpy.AddError('Input NetCDF has no data/variables/bands.')
                return
            elif 'sigmoid' not in ds:
                arcpy.AddError('Input NetCDF does not contain a sigmoid variable.')
                return
            elif 'time' in ds:
                arcpy.AddError('Input NetCDF must not have a time dimension.')
                return
            elif 'x' not in ds.dims or 'y' not in ds.dims:
                arcpy.AddError('Input NetCDF must have x, y and time dimensions.')
                return
            elif 'x' not in ds.coords or 'y' not in ds.coords:
                arcpy.AddError('Input NetCDF must have x, y and time coords.')
                return
            elif 'spatial_ref' not in ds.coords:
                arcpy.AddError('Input NetCDF must have a spatial_ref coord.')
                return
            elif len(ds['x']) == 0 or len(ds['y']) == 0:
                arcpy.AddError('Input NetCDF must have all at least one x, y and time index.')
                return
            elif ds.attrs == {}:
                arcpy.AddError('Input NetCDF must have attributes.')
                return
            elif not hasattr(ds, 'crs'):
                arcpy.AddError('Input NetCDF CRS attribute not found. CRS required.')
                return
            elif ds.crs != 'EPSG:3577':
                arcpy.AddError('Input NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
                return
            elif not hasattr(ds, 'nodatavals'):
                arcpy.AddError('Input NetCDF nodatavals attribute not found.')
                return

                # check if xr is all nan
            if ds.to_array().isnull().all():
                arcpy.AddError('Input NetCDF is completely null.')
                return

            try:
                # now, do proper open of satellite netcdf properly (and set nodata to nan)
                ds = satfetcher.load_local_nc(nc_path=layer[0].value,
                                              use_dask=True,
                                              conform_nodata_to=np.nan)

                # add evidence type attr
                ds.attrs.update({'evi_type': layer[1]})

            except Exception as e:
                arcpy.AddError('Could not properly load input NetCDF data, see messages for details.')
                arcpy.AddMessage(str(e))
                return

            # add to list of datasets
            ds_list.append(ds)

        # check we have some datasets
        if len(ds_list) == 0:
            arcpy.AddError('Insufficient NetCDF datasets.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Conforming NetCDFs via resampling...')
        arcpy.SetProgressorPosition(2)

        # check extents overlap
        if not tools.all_xr_intersect(ds_list):
            arcpy.AddError('Not all input NetCDFs intersect.')
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
            arcpy.AddError('Could not get target NetCDF resolution, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check target xr captured
        if ds_target is None:
            arcpy.AddError('Could not obtain optimal NetCDF resolution.')
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
            arcpy.AddError('Could not resample NetCDFs, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Smoothing input NetCDFs, if requested...')
        arcpy.SetProgressorPosition(3)

        # smooth, if requested...
        if in_smooth is True:

            # check window length is valid
            if in_win_size < 3 or in_win_size % 2 == 0:
                arcpy.AddError('Smooth window size is invalid.')
                return

            try:
                # smooth each dataset
                for idx in range(len(ds_list)):
                    ds_list[idx] = ensemble.smooth_xr_dataset(ds=ds_list[idx],
                                                              win_size=in_win_size)
            except Exception as e:
                arcpy.AddError('Could not smooth NetCDFs, see messages for details.')
                arcpy.AddMessage(str(e))
                return

                # check list is still valid
            if len(ds_list) == 0:
                arcpy.AddError('Smoothing resulted in empty NetCDFs.')
                return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing NetCDFs into memory, please wait...')
        arcpy.SetProgressorPosition(4)

        try:
            # iter datasets and load values
            for ds in ds_list:
                ds.load()
        except Exception as e:
            arcpy.AddError('Could not compute NetCDFs, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if each dataset is all nan
        for ds in ds_list:
            if ds.to_array().isnull().all():
                arcpy.AddError('Input NetCDF is empty, please check inputs.')
                return

                # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Splitting datasets into belief and disbelief...')
        arcpy.SetProgressorPosition(5)

        # split into belief list
        beliefs, disbeliefs = [], []
        for ds in ds_list:
            if ds.evi_type == 'Belief':
                beliefs.append(ds)
            else:
                disbeliefs.append(ds)

        # check we have something for both types
        if len(beliefs) == 0 or len(disbeliefs) == 0:
            arcpy.AddError('Could not split NetCDFs into belief and disbelief types.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Performing ensemble modelling...')
        arcpy.SetProgressorPosition(6)

        try:
            # perfom ensemble modelling
            ds = ensemble.perform_modelling(belief=beliefs,
                                            disbelief=disbeliefs)
        except Exception as e:
            arcpy.AddError('Could not perform ensemble modelling, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if dataset exists
        if ds.to_array().isnull().all():
            arcpy.AddError('No ensemble result was produced from modelling.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Reducing ensemble result to minimum extent...')
        arcpy.SetProgressorPosition(7)

        try:
            # ensure bounding box is fixed to smallest mbr
            ds = tools.remove_nan_xr_bounds(ds=ds)
        except Exception as e:
            arcpy.AddError('Could not reduce to minimum extent, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('. Please check GeoTiffs.')
            return

            # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(8)

        try:
            # manually create attrs for dataset (geotiffs lacking)
            ds = tools.manual_create_xr_attrs(ds)
            ds.attrs.update({'nodatavals': np.nan})
        except Exception as e:
            arcpy.AddError('Could not append attributes onto dataset, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(9)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(10)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing ensemble layers if exist
                for layer in m.listLayers():
                    if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'ensemble':
                        m.removeLayer(layer)

                # setup a group layer via template
                grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
                grp = m.addLayer(grp_lyr)[0]
                grp.name = 'ensemble'

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'ensemble' + '_' + dt)
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
                            # apply percent clip type and threshold
                            sym.colorizer.stretchType = 'PercentClip'
                            sym.colorizer.minPercent = 0.1
                            sym.colorizer.maxPercent = 0.1

                            # create color map and apply
                            cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]
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
        arcpy.SetProgressorPosition(11)

        # close and del dataset
        ds.close()
        del ds

        try:
            # close all xr datasets in input lists
            for ds in ds_list + beliefs + disbeliefs:
                ds.close()
        except:
            pass

        # notify user
        arcpy.AddMessage('Generated Ensemble Model successfully.')

        return
