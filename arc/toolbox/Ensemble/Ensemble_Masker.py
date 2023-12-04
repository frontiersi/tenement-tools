# safe imports
import os
import datetime
import numpy as np
import arcpy
import xarray as xr
import dask
import tempfile

from shared import satfetcher, tools
from arc.toolbox.globals import GRP_LYR_FILE

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class Ensemble_Masker(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'Ensemble Masker'
        self.description = 'Use an existing NetCDF or GeoTiff layer to mask ' \
                           'out areas from previously generated Ensemble modelling ' \
                           'outputs. Useful for removing infrastructure.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input ensemble netcdf
        par_in_ensemble_nc = arcpy.Parameter(
            displayName='Input Ensemble NetCDF file',
            name='in_ensemble_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_ensemble_nc.filter.list = ['nc']

        # output netcdf file
        par_out_nc = arcpy.Parameter(
            displayName='Output masked Ensemble NetCDF file',
            name='out_nc',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc.filter.list = ['nc']

        # input mask netcdf or geotiff
        par_in_mask_file = arcpy.Parameter(
            displayName='Input mask NetCDF or GeoTiff file',
            name='in_mask_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_mask_file.filter.list = ['nc', 'tif']

        # input variable
        par_in_var = arcpy.Parameter(
            displayName='Mask variable',
            name='in_var',
            datatype='GPString',
            parameterType='Required',
            direction='Input')
        par_in_var.filter.type = 'ValueList'
        par_in_var.filter.list = []

        # input mask type
        par_in_type = arcpy.Parameter(
            displayName='Mask type',
            name='in_type',
            datatype='GPString',
            parameterType='Required',
            direction='Input')
        par_in_type.filter.type = 'ValueList'
        par_in_type.filter.list = ['Binary', 'Range']
        par_in_type.value = 'Binary'

        # input binary mask value
        par_in_bin = arcpy.Parameter(
            displayName='Mask value',
            name='in_binary',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input')

        # input range minimum value
        par_range_min = arcpy.Parameter(
            displayName='Minimum mask value',
            name='in_range_min',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input')

        # input range maximum value
        par_range_max = arcpy.Parameter(
            displayName='Maximum mask value',
            name='in_range_max',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input')

        # input replacement value
        par_in_replace = arcpy.Parameter(
            displayName='Replacement',
            name='in_replace',
            datatype='GPString',
            parameterType='Required',
            direction='Input')
        par_in_replace.filter.type = 'ValueList'
        par_in_replace.filter.list = ['NoData', '0', '1']
        par_in_replace.value = 'NoData'

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
            par_in_ensemble_nc,
            par_out_nc,
            par_in_mask_file,
            par_in_var,
            par_in_type,
            par_in_bin,
            par_range_min,
            par_range_max,
            par_in_replace,
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
        global ENSEMBLE_MASKER

        # unpack global parameter values
        curr_file = ENSEMBLE_MASKER.get('in_file')
        curr_var = ENSEMBLE_MASKER.get('in_var')

        # if input file added, run
        if parameters[2].value is not None:

            # if global has no matching file (or first run), reload all
            if curr_file != parameters[2].valueAsText:

                if parameters[2].valueAsText.endswith('.nc'):
                    try:
                        ds = xr.open_dataset(parameters[2].valueAsText)
                        data_vars = [var for var in ds]
                        ds.close()
                    except:
                        data_vars = []

                elif parameters[2].valueAsText.endswith('.tif'):
                    try:
                        ds = xr.open_rasterio(parameters[2].valueAsText)
                        data_vars = [var for var in ds.to_dataset(dim='band')]
                        ds.close()
                    except:
                        data_vars = []

                # populate var list with new vars
                parameters[3].filter.list = data_vars

                # set var and bin, min, max to no selections
                parameters[3].value = None
                parameters[5].value = None
                parameters[6].value = None
                parameters[7].value = None

                # if var has changed, calc min, max values
            elif curr_var != parameters[3].valueAsText:
                new_var = parameters[3].valueAsText
                new_type = parameters[4].value

                if parameters[2].valueAsText.endswith('.nc'):
                    try:
                        # load ds, get nodataval,
                        ds = xr.open_dataset(parameters[2].valueAsText)
                        nd = ds.nodatavals

                        # get only non-nodata vals
                        da = ds[new_var]
                        da = da.where(da != nd)

                        # get mins and max value
                        mins = round(float(da.min()), 3)
                        maxs = round(float(da.max()), 3)
                        ds.close()
                    except:
                        mins, maxs = None, None

                elif parameters[2].valueAsText.endswith('.tif'):
                    try:
                        # load tif, get arrs of nodata vals and vars, get nodata at band
                        ds = xr.open_rasterio(parameters[2].valueAsText)
                        ds = ds.to_dataset(dim='band')
                        nds, dvs = np.array(ds.nodatavals), np.array(ds.data_vars)
                        nd = nds[np.where(dvs == int(new_var))]

                        # get only non-nodata vals
                        da = ds[int(new_var)]
                        da = da.where(da != nd)

                        # get mins and max value
                        mins = round(float(da.min()), 3)
                        maxs = round(float(da.max()), 3)
                        ds.close()
                    except:
                        mins, maxs = None, None

                # set range min, max
                parameters[6].value = mins
                parameters[7].value = maxs

        # update global values
        ENSEMBLE_MASKER = {
            'in_file': parameters[2].valueAsText,
            'in_var': parameters[3].value,
        }

        # enable binary or range parameters based on drop down
        if parameters[4].value == 'Binary':
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
        Executes the Ensemble Masker module.
        """

        # grab parameter values
        in_ensemble_nc = parameters[0].valueAsText  # ensemble netcdf
        out_nc = parameters[1].valueAsText  # output netcdf
        in_mask_file = parameters[2].valueAsText  # mask nc or tif
        in_var = parameters[3].value  # variable
        in_type = parameters[4].value  # mask type
        in_bin = parameters[5].value  # binary value
        in_range_min = parameters[6].value  # range minimum
        in_range_max = parameters[7].value  # range maximum
        in_replace = parameters[8].value  # replacement value
        in_add_result_to_map = parameters[9].value  # add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Ensemble Masker.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=12)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking ensemble NetCDF...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of ensemble netcdf for checking
            ds_ens = xr.open_dataset(in_ensemble_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load ensemble NetCDF data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_ens, xr.Dataset):
            arcpy.AddError('Input ensemble NetCDF must be a xr dataset.')
            return
        elif len(ds_ens) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'time' in ds_ens.dims:
            arcpy.AddError('Input ensemble NetCDF must not have time dimension.')
            return
        elif 'x' not in ds_ens.dims or 'y' not in ds_ens.dims:
            arcpy.AddError('Input ensemble NetCDF must have x, y dimensions.')
            return
        elif 'time' in ds_ens.coords:
            arcpy.AddError('Input ensemble NetCDF must not have time coord.')
            return
        elif 'x' not in ds_ens.coords or 'y' not in ds_ens.coords:
            arcpy.AddError('Input ensemble NetCDF must have x, y coords.')
            return
        elif 'spatial_ref' not in ds_ens.coords:
            arcpy.AddError('Input ensemble NetCDF must have a spatial_ref coord.')
            return
        elif len(ds_ens['x']) == 0 or len(ds_ens['y']) == 0:
            arcpy.AddError('Input ensemble NetCDF must have all at least one x, y index.')
            return
        elif ds_ens.attrs == {}:
            arcpy.AddError('Ensemble NetCDF must have attributes.')
            return
        elif not hasattr(ds_ens, 'crs'):
            arcpy.AddError('Ensemble NetCDF CRS attribute not found. CRS required.')
            return
        elif ds_ens.crs != 'EPSG:3577':
            arcpy.AddError('Ensemble NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
            return
        elif not hasattr(ds_ens, 'nodatavals'):
            arcpy.AddError('Ensemble NetCDF nodatavals attribute not found.')
            return

            # ensure four ensemble variables exist
        for var in ds_ens:
            if var not in ['belief', 'disbelief', 'plausability', 'interval']:
                arcpy.AddError('Ensemble NetCDF is missing variable {}.'.format(var))
                return

        # check if all nan
        if ds_ens.to_array().isnull().all():
            arcpy.AddError('Ensemble NetCDF is empty.')
            return

        try:
            # now open of ensemble netcdf properly (and set nodata to nan)
            ds_ens = satfetcher.load_local_nc(nc_path=in_ensemble_nc,
                                              use_dask=False,
                                              conform_nodata_to=np.nan)
        except Exception as e:
            arcpy.AddError('Could not properly load ensemble NetCDF data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Getting ensemble NetCDF attributes...')
        arcpy.SetProgressorPosition(2)

        # get attributes from dataset
        ds_attrs = ds_ens.attrs
        ds_band_attrs = ds_ens[list(ds_ens)[0]].attrs
        ds_spatial_ref_attrs = ds_ens['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing ensemble NetCDF into memory, please wait...')
        arcpy.SetProgressorPosition(3)

        try:
            # compute!
            ds_ens = ds_ens.compute()
        except Exception as e:
            arcpy.AddError('Could not compute ensemble NetCDF, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking mask NetCDF or GeoTiff...')
        arcpy.SetProgressorPosition(4)

        # get and check mask file extension
        ext = os.path.splitext(in_mask_file)[1]
        if ext not in ['.nc', '.tif']:
            arcpy.AddError('Mask file is not NetCDF or GeoTiff type.')
            return

        try:
            # quick load mask file based on netcdf or geotiff
            if ext == '.nc':
                ds_mask = xr.open_dataset(in_mask_file)
            else:
                ds_mask = xr.open_rasterio(in_mask_file)
                ds_mask = ds_mask.to_dataset(dim='band')

                # convert var to int (geotiff will always be a band int)
                in_var = int(in_var)

        except Exception as e:
            arcpy.AddError('Could not quick load input mask NetCDF or GeoTiff, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_mask, xr.Dataset):
            arcpy.AddError('Input mask file must be an xr dataset.')
            return
        elif len(ds_mask) == 0:
            arcpy.AddError('Input mask file has no data/variables/bands.')
            return
        elif in_var not in ds_mask:
            arcpy.AddError('Input mask file does not contain requested variable.')
            return
        elif 'x' not in list(ds_mask.coords) or 'y' not in list(ds_mask.coords):
            arcpy.AddError('Input mask file must have x, y coords.')
            return
        elif 'x' not in list(ds_mask.dims) or 'y' not in list(ds_mask.dims):
            arcpy.AddError('Input mask file must have x, y dimensions.')
            return
        elif len(ds_mask['x']) == 0 or len(ds_mask['y']) == 0:
            arcpy.AddError('Input mask file must have at least one x, y index.')
            return
        elif ds_mask.attrs == {}:
            arcpy.AddError('Mask file attributes not found. Mask file must have attributes.')
            return
        elif not hasattr(ds_mask, 'crs'):
            arcpy.AddError('Mask file CRS attribute not found. CRS required.')
            return
        elif '3577' not in ds_mask.crs:
            arcpy.AddError('Mask file CRS is not EPSG:3577. EPSG:3577 required.')
            return
        elif not hasattr(ds_mask, 'nodatavals'):
            arcpy.AddError('Mask file nodatavals attribute not found.')
            return

        try:
            # proper load mask based on file nectdf or geotiff
            if ext == '.nc':
                ds_mask = satfetcher.load_local_nc(nc_path=in_mask_file,
                                                   use_dask=False,
                                                   conform_nodata_to=np.nan)
            else:
                ds_mask = xr.open_rasterio(in_mask_file)
                ds_mask = ds_mask.to_dataset(dim='band')
                ds_mask = tools.manual_create_xr_attrs(ds_mask)

                # set nodataval to nan (can be tuple)
                for nd in ds_mask.nodatavals:
                    ds_mask = ds_mask.where(ds_mask != nd)

            # subset mask dataset to requested variable, keep as dataset
            ds_mask = ds_mask[[in_var]]

        except Exception as e:
            arcpy.AddError('Could not properly load input mask data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check if all nan
        if ds_mask.to_array().isnull().all():
            arcpy.AddError('Mask data is empty.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Resampling mask data...')
        arcpy.SetProgressorPosition(5)

        # check extents overlap
        if not tools.all_xr_intersect([ds_ens, ds_mask]):
            arcpy.AddError('Not all input layers intersect.')
            return

        try:
            # resample mask to ensemble dataset, if same, no change
            ds_mask = tools.resample_xr(ds_from=ds_mask,
                                        ds_to=ds_ens,
                                        resampling='nearest')
        except Exception as e:
            arcpy.AddError('Could not clip and resample mask data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # ensure mask still has values
        if ds_mask.to_array().isnull().all():
            arcpy.AddError('No values in mask after resample.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing mask data into memory, please wait...')
        arcpy.SetProgressorPosition(6)

        try:
            # compute!
            ds_mask = ds_mask.compute()
        except Exception as e:
            arcpy.AddError('Could not compute mask data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Preparing mask data...')
        arcpy.SetProgressorPosition(7)

        # check type
        if in_type not in ['Binary', 'Range']:
            arcpy.AddError('Mask type not supported.')
            return

        # check binary value
        if in_type == 'Binary':
            if in_bin is None:
                arcpy.AddError('Must provide a mask value when using binary type.')
                return
            elif in_bin not in np.unique(ds_mask.to_array()):
                arcpy.AddError('Binary value not found in mask.')
                return

        # check range values
        if in_type == 'Range':
            if in_range_min is None or in_range_max is None:
                arcpy.AddError('Must provide a min and max value when using range type.')
                return
            elif in_range_max <= in_range_min:
                arcpy.AddError('Range maximum can not be <= minimum.')
                return

        try:
            # prepare mask depending on user choice
            if in_type == 'Binary':
                ds_mask = xr.where(ds_mask == in_bin, False, True)
            else:
                ds_mask = xr.where((ds_mask >= in_range_min) &
                                   (ds_mask <= in_range_max), False, True)

        except Exception as e:
            arcpy.AddError('Could not prepare mask data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check if any mask values exist
        if not (ds_mask.to_array() == True).any():
            arcpy.AddError('Requested mask value resulted in empty mask.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Masking ensemble dataset via mask...')
        arcpy.SetProgressorPosition(8)

        # check replacement value is valid
        if in_replace not in ['NoData', '1', '0']:
            arcpy.AddError('Replacement value not provided.')
            return

        # prepare replacement value
        if in_replace == 'NoData':
            in_replace = np.nan
        else:
            in_replace = int(in_replace)

        # mask any values to nan and squeeze
        ds_ens = ds_ens.where(ds_mask.to_array(), in_replace)
        ds_ens = ds_ens.squeeze(drop=True)

        # check if any values exist in ensemble dataset
        if ds_ens.to_array().isnull().all():
            arcpy.AddError('Ensemble dataset has no values after mask.')
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to ensemble dataset...')
        arcpy.SetProgressorPosition(9)

        # append attrbutes on to dataset and bands
        ds_ens.attrs = ds_attrs
        ds_ens['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_ens:
            ds_ens[var].attrs = ds_band_attrs

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(10)

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_ens, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(11)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing fractional layers if exist
                for layer in m.listLayers():
                    if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'ensemble_masked':
                        m.removeLayer(layer)

                # setup a group layer via template
                grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
                grp = m.addLayer(grp_lyr)[0]
                grp.name = 'ensemble_masked'

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'ensemble_masked' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # iter each var and export a seperate tif
                tif_list = []
                for var in ds_ens:
                    # create temp netcdf for one var (prevents 2.9 bug)
                    with tempfile.NamedTemporaryFile() as tmp:
                        tmp_nc = '{}_{}.nc'.format(tmp.name, var)
                        ds_ens[[var]].to_netcdf(tmp_nc)

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
        arcpy.SetProgressorPosition(12)

        # close ensemble dataset
        ds_ens.close()
        del ds_ens

        # close mask dataset
        ds_mask.close()
        del ds_mask

        # notify user
        arcpy.AddMessage('Performed Ensemble Masking successfully.')

        return