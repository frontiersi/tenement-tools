import os
import datetime
import arcpy
import dask
import tempfile

from arc.toolbox.globals import GRP_LYR_FILE
from modules import nrt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)


class NRT_Visualise_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Visualise Areas'
        self.description = 'Visualise total vegetation and change density using ' \
                           'a kernel density method for a specified monitoring area.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # set input feature class
        par_in_feat = arcpy.Parameter(
            displayName='Input monitoring area feature',
            name='in_feat',
            datatype='GPFeatureLayer',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_feat.filter.list = ['Polygon']

        # set monitoring area
        par_in_set_area = arcpy.Parameter(
            displayName='Select the monitoring area to visualise',
            name='in_set_area',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_set_area.filter.type = 'ValueList'
        par_in_set_area.enabled = False

        # output netcdf
        par_out_nc = arcpy.Parameter(
            displayName='Output NetCDF file',
            name='out_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc.filter.list = ['nc']

        # combine parameters
        parameters = [
            par_in_feat,
            par_in_set_area,
            par_out_nc
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

        # set required fields
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
            'color_border',
            'color_border',
            'global_id'
        ]

        # globals
        global NRT_VISUALISE_AREA

        # unpack global parameter values
        curr_feat = NRT_VISUALISE_AREA.get('in_feat')
        curr_area_id = NRT_VISUALISE_AREA.get('in_area_id')

        # if not first run or no change to area id, skip
        if curr_area_id is not None and curr_area_id == parameters[1].value:
            return

        # check feature input
        if parameters[0].value is not None:
            try:
                # load column names
                in_feat = parameters[0].valueAsText
                cols = [f.name for f in arcpy.ListFields(in_feat)]
            except:
                cols = []

            # if valid fields, proceed to get all rows
            if all(f in cols for f in required_fields):
                try:
                    with arcpy.da.SearchCursor(in_feat, field_names=required_fields) as cursor:
                        rows = [rec for rec in cursor]
                except:
                    return

                # if first time, get first row values, else user selected
                row = None
                if parameters[1].value is None:
                    row = rows[0]
                else:
                    for row in rows:
                        if row[0] == parameters[1].value:
                            break

                # enable, populate and set parameters
                if row is not None:
                    parameters[1].enabled = True
                    parameters[1].filter.list = [rec[0] for rec in rows]
                    parameters[1].value = row[0]

                    # update global
                    NRT_VISUALISE_AREA = {'in_area_id': parameters[1].value}

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Visualise Areas module.
        """

        # set rasterio env
        rio_env = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'VSI_CACHE': 'TRUE',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
        }

        # apply rio env settings
        for k, v in rio_env.items():
            os.environ[k] = v

        # grab parameter values
        in_feat = parameters[0]  # input monitoring areas feature
        in_area_id = parameters[1].value  # input monitoring area id
        out_nc = parameters[2].valueAsText  # output netcdf

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Visualise Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=23)

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Validating monitoring area geodatabase...')
        arcpy.SetProgressorPosition(1)

        # get full path to existing monitoring areas
        feat_desc = arcpy.Describe(parameters[0].value)
        in_feat = os.path.join(feat_desc.path, feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_feat)
        except Exception as e:
            arcpy.AddError('Monitoring areas feature is incompatible, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Obtaining and validating monitoring areas...')
        arcpy.SetProgressorPosition(2)

        # check area id is valid
        if in_area_id is None:
            arcpy.AddError('Did not provide an area identifier value.')
            return

        # set required fields
        fields = [
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
            'color_border',
            'color_fill',
            'global_id',
            'SHAPE@'
        ]

        try:
            # get feature based on area id
            feat = ()
            with arcpy.da.SearchCursor(in_feat, fields) as cursor:
                for row in cursor:
                    if row[0] == in_area_id:
                        feat = row
        except Exception as e:
            arcpy.AddError('Could not open monitoring areas feature.')
            arcpy.AddMessage(str(e))
            return

        # check if feature exists
        if len(feat) == 0:
            arcpy.AddError('Requested monitoring area does not exist.')
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Converting features to monitoring areas objects...')
        arcpy.SetProgressorPosition(3)

        # prepare path to expected geodatabase directory
        in_path = os.path.dirname(in_feat)
        in_path = os.path.splitext(in_path)[0]
        in_path = os.path.dirname(in_path)

        try:
            # create instance of monitoring area
            area = nrt.MonitoringAreaStatistics(feat, path=in_path, out_nc=out_nc)
        except Exception as e:
            arcpy.AddError('Could not convert to monitoring area objects.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Validating monitoring area...')
        arcpy.SetProgressorPosition(4)

        try:
            # validate area
            area.validate_area()
        except Exception as e:
            arcpy.AddError('Area is invalid, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Obtaining new satellite data...')
        arcpy.SetProgressorPosition(5)

        try:
            # get, set all satellite data from start year to now
            area.set_xr()

            # abort if no dates returned, else notify and proceed
            if len(area.ds['time']) == 0:
                arcpy.AddError('No satellite data was obtained.')
                return
            else:
                arcpy.AddMessage('Found {} satellite images.'.format(len(area.ds['time'])))
                pass
        except Exception as e:
            arcpy.AddError('Could not obtain satellite data for area, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and on-going progress bar
        arcpy.SetProgressor('default', 'Removing satellite images with clouds...')
        arcpy.SetProgressorLabel('Removing satellite images with clouds...')

        try:
            # apply fmask on xr. if error or no dates, skip
            area.apply_xr_fmask()

            # abort if no clean dates returned, else notify and proceed
            if len(area.ds['time']) == 0:
                arcpy.AddError('No cloud-free satellite data was obtained.')
                return
            else:
                arcpy.AddMessage('Found {} cloud-free satellite images.'.format(len(area.ds['time'])))
                pass
        except Exception as e:
            arcpy.AddError('Could not apply fmask, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Calculating vegetation index for new satellite images...')
        arcpy.SetProgressorPosition(7)

        try:
            # calculate vege index for xr
            area.apply_xr_index()
        except Exception as e:
            arcpy.AddError('Could not calculate vegetation index, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Downloading satellite data, please wait...')

        try:
            # load xr
            area.load_xr()
        except Exception as e:
            arcpy.AddError('Could not download satellite data, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Building pixel edge mask...')
        arcpy.SetProgressorPosition(9)

        try:
            # set pixel edge mask here, apply later (geobox needed)
            area.set_xr_edge_mask()
        except Exception as e:
            arcpy.AddError('Could not build pixel edge mask.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Interpolating missing pixels...')
        arcpy.SetProgressorPosition(10)

        try:
            # interpolate nans
            area.interp_xr_nans()
        except Exception as e:
            arcpy.AddError('Could not interpolate missing pixels, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Appending required variables...')
        arcpy.SetProgressorPosition(11)

        try:
            # append required vars to xr
            area.append_xr_vars()
        except Exception as e:
            arcpy.AddError('Could not append variables, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Removing vegetation spikes...')
        arcpy.SetProgressorPosition(12)

        try:
            # remove and interpolate spikes
            area.fix_xr_spikes()
        except Exception as e:
            arcpy.AddError('Could not remove spikes, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Performing change detection on dataset, please wait...')

        try:
            # detect static, dynamic change via xr
            area.detect_change_xr()
        except Exception as e:
            arcpy.AddError('Could not perform change detection, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Smoothing change detection on dataset...')
        arcpy.SetProgressorPosition(14)

        try:
            # smooth static, dynamic change in xr
            area.smooth_xr_change()
        except Exception as e:
            arcpy.AddError('Could not smooth change detection data, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Calculating zones...')
        arcpy.SetProgressorPosition(15)

        try:
            # build zones
            area.build_zones()
        except Exception as e:
            arcpy.AddError('Could not calculate zones, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Calculating rule one, two and three...')
        arcpy.SetProgressorPosition(16)

        try:
            # build rule one, two, three
            area.build_rule_one()
            area.build_rule_two()
            area.build_rule_three()
        except Exception as e:
            arcpy.AddError('Could not calculate rules, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Calculating alerts...')
        arcpy.SetProgressorPosition(17)

        try:
            # build alerts
            area.build_alerts()
        except Exception as e:
            arcpy.AddError('Could not calculate alerts, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Applying pixel edge mask...')
        arcpy.SetProgressorPosition(18)

        try:
            # apply pixel edge mask
            area.apply_xr_edge_mask()
        except Exception as e:
            arcpy.AddError('Could not apply pixel edge mask, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Calculating densities...')
        arcpy.SetProgressorPosition(19)

        try:
            # generate kernel densities
            area.perform_kernel_density()
        except Exception as e:
            arcpy.AddError('Could not calculate densities, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Appending spatial attributes to output...')
        arcpy.SetProgressorPosition(20)

        try:
            # append spatial attributes to xr
            area.append_attrs()
        except Exception as e:
            arcpy.AddError('Could not append spatial attributes, see messages.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Exporting NetCDF...')
        arcpy.SetProgressorPosition(21)

        try:
            # export netcdf
            area.export_xr()
        except Exception as e:
            arcpy.AddError('Could not export NetCDF, see messages.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(22)

        try:
            # open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove existing ensemble layers if exist
            for layer in m.listLayers():
                if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'nrt_visual':
                    m.removeLayer(layer)

            # setup a group layer via template
            grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
            grp = m.addLayer(grp_lyr)[0]
            grp.name = 'nrt_visual'

            # create output folder using datetime as name
            dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
            out_folder = os.path.join(os.path.dirname(out_nc), 'nrt_visual' + '_' + dt)
            os.makedirs(out_folder)

            # disable visualise on map temporarily
            arcpy.env.addOutputsToMap = False

            # iter each var and export a seperate tif
            tif_list = []
            for var in area.ds:
                # create temp netcdf for one var (prevents 2.9 bug)
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp_nc = '{}_{}.nc'.format(tmp.name, var)
                    area.ds[[var]].to_netcdf(tmp_nc)

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
                        if 'vege' in tif:
                            sym.colorizer.minPercent = 1.0
                            sym.colorizer.maxPercent = 1.0
                            sym.colorizer.invertColorRamp = False
                            cmap = aprx.listColorRamps('Yellow-Green (Continuous)')[0]
                        elif 'change' in tif and 'incline' in tif:
                            sym.colorizer.minPercent = 1.0
                            sym.colorizer.maxPercent = 1.0
                            sym.colorizer.invertColorRamp = False
                            cmap = aprx.listColorRamps('Yellow-Green-Blue (Continuous)')[0]
                        elif 'change' in tif and 'decline' in tif:
                            sym.colorizer.minPercent = 1.0
                            sym.colorizer.maxPercent = 1.0
                            sym.colorizer.invertColorRamp = True
                            cmap = aprx.listColorRamps('Yellow-Orange-Red (Continuous)')[0]
                        elif 'alerts' in tif and 'incline' in tif:
                            sym.colorizer.minPercent = 1.0
                            sym.colorizer.maxPercent = 1.0
                            sym.colorizer.invertColorRamp = False
                            cmap = aprx.listColorRamps('Yellow-Green-Blue (Continuous)')[0]
                        elif 'alerts' in tif and 'decline' in tif:
                            sym.colorizer.minPercent = 1.0
                            sym.colorizer.maxPercent = 1.0
                            sym.colorizer.invertColorRamp = False
                            cmap = aprx.listColorRamps('Yellow-Orange-Red (Continuous)')[0]

                        # apply color map
                        sym.colorizer.colorRamp = cmap

                        # apply other basic options
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

        try:
            # close area
            area.reset()
            del area
        except Exception as e:
            arcpy.AddError('Could not close data.')
            arcpy.AddMessage(str(e))
            return

            # notify user
        arcpy.AddMessage('Generated NRT Visualise successfully.')

        return