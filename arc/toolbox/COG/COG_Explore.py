import arcpy
import os, sys
import numpy as np
import tempfile
import xarray as xr

from modules import cog
from shared import arc, tools, satfetcher
from arc.toolbox.COG.COG_explore_base import CogExploreBase
class COG_Explore:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'COG Explore'
        self.description = 'Explore an existing multidimensional raster layer ' \
                           'downloaded prior using the COG Fetch tool.'
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

        # output crf data folder
        par_out_folder = arcpy.Parameter(
            displayName='Output folder',
            name='out_folder',
            datatype='DEFolder',
            parameterType='Required',
            direction='Input')

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
            multiValue=False
        )
        par_interpolate.value = True

        # combine parameters
        parameters = [
            par_nc_file,
            par_out_folder,
            par_veg_idx,
            par_fmask_flags,
            par_max_cloud,
            par_interpolate
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


    def add_errs(self, errs):
        for message in errs:
            arcpy.AddError(message)

        if len(errs) != 0:
            return

    def update_progress_bar(self, position, message):
        arcpy.SetProgressorLabel(message)
        arcpy.SetProgressorPosition(position)

    def execute(self, parameters, messages):
        """
        Executes the COG Explore module.
        """

        # grab parameter values
        in_nc = parameters[0].valueAsText  # raw input satellite netcdf
        out_folder = parameters[1].valueAsText  # output crf folder
        in_veg_idx = parameters[2].value  # vege index name
        in_fmask_flags = parameters[3].valueAsText
        in_fmask_flags = tuple(e for e in in_fmask_flags.split(';'))# fmask flag values
        in_max_cloud = parameters[4].value  # max cloud percentage
        in_interpolate = parameters[5].value  # interpolate missing pixels

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning COG Explore.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=11)

        # # # # #
        # notify and increment progress bar
        self.update_progress_bar(1, 'Loading and checking netcdf...')


        dataset = CogExploreBase(in_nc, in_veg_idx, in_fmask_flags, in_max_cloud, in_interpolate, out_folder)
        if len(in_fmask_flags) == 0:
            arcpy.AddWarning(f'No flags selected, using default. {dataset.fmask_flags}')
        self.add_errs(dataset.messages)

        dataset.full_load()

        # # # # #
        # notify and increment progress bar
        self.update_progress_bar(4, 'Removing invalid pixels and empty dates...')

        dataset.remove_pixels_empty_scenes()
        # convert fmask as text to numeric code equivalents

        # check if flags selected, if not, select all

        # # # # #
        # notify and increment progress bar
        self.update_progress_bar(5, 'Conforming satellite band names...')

        dataset.conform_band_names()

        errs = dataset.check_band_names()
        self.add_errs(errs)

        # # # # #
        # notify and increment progress bar
        self.update_progress_bar(6, 'Calculating vegetation index...')

        # check if veg idx supported
        errs = dataset.check_veg_idx()
        self.add_errs(errs)

        try:
            # calculate vegetation index
            dataset.calc_veg_idx()

        except Exception as e:
            arcpy.AddError('Could not calculate vegetation index.')
            arcpy.AddMessage(str(e))
            return

        errs = dataset.check_enough_temporal_data()
        self.add_errs(errs)

            # # # # #
        # notify and increment progress bar
        self.update_progress_bar(7, 'Computing data into memory, please wait...')

        try:
            # compute!
            dataset.compute_loaded_ds()
        except Exception as e:
            arcpy.AddError('Could not compute dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return

            # check if all nan again
        errs = dataset.check_null()
        self.add_errs(errs)

            # # # # #
        # notify and increment progress bar
        self.update_progress_bar(8, 'Interpolating dataset, if requested...')

        # if requested...
        if in_interpolate:
            try:
                # interpolate along time dimension (linear)
                dataset.interpolate()
            except Exception as e:
                arcpy.AddError('Could not interpolate dataset.')
                arcpy.AddMessage(str(e))
                return

                # # # # #
        # notify and increment progess bar
        self.update_progress_bar(9, 'Appending attributes back on to dataset...')

        # append attrbutes on to dataset and bands
        dataset.reattach_attributes()
            # # # # #
        # notify and increment progress bar
        self.update_progress_bar(10, 'Adding output to map...')

        try:
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove explore layer if already exists
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'cog_explore.crf':
                    m.removeLayer(layer)

            # create temp netcdf (prevents 2.9 bug)
            with tempfile.NamedTemporaryFile() as tmp:
                tmp_nc = '{}_{}.nc'.format(tmp.name, 'cog_explore')
                dataset.ds_loaded.to_netcdf(tmp_nc)

            # disable visualise on map temporarily
            arcpy.env.addOutputsToMap = False

            # create crf filename and copy it
            out_file = os.path.join(out_folder, 'cog_explore.crf')
            crf = arcpy.CopyRaster_management(in_raster=tmp_nc,
                                              out_rasterdataset=out_file)

            # add to map
            m.addDataFromPath(crf)

        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass

        try:
            # get symbology, update it
            layer = m.listLayers('cog_explore.crf')[0]
            sym = layer.symbology

            # if layer has stretch coloriser, apply color
            if hasattr(sym, 'colorizer'):
                if sym.colorizer.type == 'RasterStretchColorizer':
                    # apply percent clip type
                    sym.colorizer.stretchType = 'PercentClip'
                    sym.colorizer.minPercent = 0.5
                    sym.colorizer.maxPercent = 0.5

                    # apply color map
                    cmap = aprx.listColorRamps('Precipitation')[0]
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
        arcpy.SetProgressorPosition(11)

        # close main dataset and del datasets
        dataset.ds_loaded.close()
        dataset.ds.close()
        del dataset

        # notify user
        arcpy.AddMessage('Generated COG Explore successfully.')

        return