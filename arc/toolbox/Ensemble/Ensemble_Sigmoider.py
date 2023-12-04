import arcpy

import os
import datetime
import numpy as np

import xarray as xr

from shared import satfetcher, tools
from modules import canopy

class Ensemble_Sigmoider(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'Ensemble Sigmoider'
        self.description = 'Rescale an input NetCDF or GeoTiff of any value ' \
                           'range to 0 - 1 using fuzzy membership functions.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input evidence netcdf, geotiff file
        par_in_file = arcpy.Parameter(
            displayName='Input evidence layer NetCDF or GeoTiff file',
            name='in_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_file.filter.list = ['nc', 'tif']

        # output netcdf file
        par_out_nc = arcpy.Parameter(
            displayName='Output sigmoid NetCDF',
            name='out_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc.filter.list = ['nc']

        # input variables
        par_in_var = arcpy.Parameter(
            displayName='Variable',
            name='in_var',
            datatype='GPString',
            parameterType='Required',
            direction='Input')
        par_in_var.filter.type = 'ValueList'
        par_in_var.filter.list = []

        # input type
        par_in_type = arcpy.Parameter(
            displayName='Membership type',
            name='in_type',
            datatype='GPString',
            parameterType='Required',
            direction='Input')
        par_in_type.filter.type = 'ValueList'
        par_in_type.filter.list = ['Increasing', 'Decreasing', 'Symmetric']
        par_in_type.value = 'Increasing'

        # input minimum (low inflection)
        par_in_min = arcpy.Parameter(
            displayName='Low inflection point',
            name='in_minimum',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input')

        # input middle (middle inflection)
        par_in_mid = arcpy.Parameter(
            displayName='Middle inflection point',
            name='in_middle',
            datatype='GPDouble',
            parameterType='Optional',
            direction='Input')

        # input maximum (high inflection)
        par_in_max = arcpy.Parameter(
            displayName='High inflection point',
            name='in_maximum',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input')

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
            par_in_file,
            par_out_nc,
            par_in_var,
            par_in_type,
            par_in_min,
            par_in_mid,
            par_in_max,
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
        global ENSEMBLE_SIGMOIDER

        # unpack global parameter values
        curr_file = ENSEMBLE_SIGMOIDER.get('in_file')
        curr_var = ENSEMBLE_SIGMOIDER.get('in_var')

        # if input file added, run
        if parameters[0].value is not None:

            # if global has no matching file (or first run), reload all
            if curr_file != parameters[0].valueAsText:

                if parameters[0].valueAsText.endswith('.nc'):
                    try:
                        ds = xr.open_dataset(parameters[0].valueAsText)
                        data_vars = [var for var in ds]
                        ds.close()
                    except:
                        data_vars = []

                elif parameters[0].valueAsText.endswith('.tif'):
                    try:
                        da = xr.open_rasterio(parameters[0].valueAsText)
                        data_vars = ['{}'.format(var) for var in da['band'].values]
                        da.close()
                    except:
                        data_vars = []

                # populate var list with new vars
                parameters[2].filter.list = data_vars

                # set var and min, mid, max to no selections
                parameters[2].value = None
                parameters[4].value = None
                parameters[5].value = None
                parameters[6].value = None

            # calc min, max if var changed
            elif curr_var != parameters[2].valueAsText:
                new_var = parameters[2].valueAsText
                new_type = parameters[3].value

                if parameters[0].valueAsText.endswith('.nc'):
                    try:
                        # load ds, get nodataval,
                        ds = xr.open_dataset(parameters[0].valueAsText)
                        nd = ds.nodatavals

                        # get only non-nodata vals
                        da = ds[new_var]
                        da = da.where(da != nd)

                        # get mins and max value
                        mins = round(float(da.min()), 3)
                        maxs = round(float(da.max()), 3)
                        ds.close()
                    except:
                        mins, mids, maxs = None, None, None

                elif parameters[0].valueAsText.endswith('.tif'):
                    try:
                        # load tif, get arrs of nodata vals and vars, get nodata at band
                        ds = xr.open_rasterio(parameters[0].valueAsText)
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
                        mins, mids, maxs = None, None, None

                # set min, mid, max
                parameters[4].value = mins
                parameters[5].value = (mins + maxs) / 2
                parameters[6].value = maxs

        # update global values
        ENSEMBLE_SIGMOIDER = {
            'in_file': parameters[0].valueAsText,
            'in_var': parameters[2].value,
        }

        # set mid inflection if membership is symmetrical
        if parameters[3].value == 'Symmetric':
            parameters[5].enabled = True
        else:
            parameters[5].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the Ensemble Sigmoider module.
        """



        # grab parameter values
        in_file = parameters[0].valueAsText  # input netcdf or geotiff
        out_nc = parameters[1].valueAsText  # output netcdf
        in_var = parameters[2].value  # input variable
        in_type = parameters[3].value  # input membership type
        in_min = parameters[4].value  # input minimum
        in_mid = parameters[5].value  # input middle
        in_max = parameters[6].value  # input maximum
        in_add_result_to_map = parameters[7].value  # input add result to map

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Ensemble Sigmoider.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=9)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking NetCDF or GeoTiff...')
        arcpy.SetProgressorPosition(1)

        # get and check file extension
        ext = os.path.splitext(in_file)[1]
        if ext not in ['.nc', '.tif']:
            arcpy.AddError('Input file is not NetCDF or GeoTiff type.')
            return

        try:
            # quick load mask file based on netcdf or geotiff
            if ext == '.nc':
                ds = xr.open_dataset(in_file)
            else:
                ds = xr.open_rasterio(in_file)
                ds = ds.to_dataset(dim='band')

                # convert var to int (geotiff will always be a band int)
                in_var = int(in_var)

        except Exception as e:
            arcpy.AddError('Could not quick load input file, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds, xr.Dataset):
            arcpy.AddError('Input file must be an xr dataset.')
            return
        elif 'time' in ds:
            arcpy.AddError('Input NetCDF time dimension must not exist.')
            return
        elif len(ds) == 0:
            arcpy.AddError('Input file has no data/variables/bands.')
            return
        elif in_var not in ds:
            arcpy.AddError('Input file does not contain requested variable.')
            return
        elif 'x' not in list(ds.coords) or 'y' not in list(ds.coords):
            arcpy.AddError('Input file must have x, y coords.')
            return
        elif 'x' not in list(ds.dims) or 'y' not in list(ds.dims):
            arcpy.AddError('Input file must have x, y dimensions.')
            return
        elif len(ds['x']) == 0 or len(ds['y']) == 0:
            arcpy.AddError('Input file must have at least one x, y index.')
            return
        elif ds.attrs == {}:
            arcpy.AddError('Input file attributes not found. Mask file must have attributes.')
            return
        elif not hasattr(ds, 'crs'):
            arcpy.AddError('Input file CRS attribute not found. CRS required.')
            return
        elif '3577' not in ds.crs:
            arcpy.AddError('Input file CRS is not EPSG:3577. EPSG:3577 required.')
            return
        elif not hasattr(ds, 'nodatavals'):
            arcpy.AddError('Input file nodatavals attribute not found.')
            return

        try:
            # proper load mask based on file nectdf or geotiff
            if ext == '.nc':
                ds = satfetcher.load_local_nc(nc_path=in_file,
                                              use_dask=False,
                                              conform_nodata_to=np.nan)
            else:
                ds = xr.open_rasterio(in_file)
                ds = ds.to_dataset(dim='band')
                ds = tools.manual_create_xr_attrs(ds)

                # set nodataval to nan (can be tuple)
                for nd in ds.nodatavals:
                    ds = ds.where(ds != nd)

            # subset mask dataset to requested variable, keep as dataset
            ds = ds[[in_var]]

        except Exception as e:
            arcpy.AddError('Could not properly load input data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check if all nan
        if ds.to_array().isnull().all():
            arcpy.AddError('Input data is empty.')
            return

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Getting input file attributes...')
        arcpy.SetProgressorPosition(2)

        # get attributes from dataset
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing input file into memory, please wait...')
        arcpy.SetProgressorPosition(3)

        try:
            # compute!
            ds = ds.compute()
        except Exception as e:
            arcpy.AddError('Could not compute input file, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Applying sigmoidal function to data...')
        arcpy.SetProgressorPosition(4)

        # check type
        if in_type not in ['Increasing', 'Decreasing', 'Symmetric']:
            arcpy.AddError('Membership type not supported.')
            return

        # check values
        if in_min is None or in_max is None:
            arcpy.AddError('Low and high inflection points must not be empty.')
            return
        elif in_max <= in_min:
            arcpy.AddError('High inflection point can not be <= low inflection.')
            return

        # check midpoint valid if symmetric
        if in_type == 'Symmetric':
            if in_mid is None:
                arcpy.AddError('Middle inflection point not provided.')
                return
            elif in_mid <= in_min or in_mid >= in_max:
                arcpy.AddError('Middle inflection point must be between min and max.')
                return

        try:
            # convert to array and build mask for later
            ds = ds.to_array()
            ds_mask = xr.where(~ds.isnull(), True, False)

            # apply sigmoidal depending on user selection
            if in_type == 'Increasing':
                ds = canopy.inc_sigmoid(ds, a=in_min, b=in_max)

            elif in_type == 'Decreasing':
                ds = canopy.dec_sigmoid(ds, c=in_min, d=in_max)

            elif in_type == 'Symmetric':
                ds = canopy.bell_sigmoid(ds, a=in_min, bc=in_mid, d=in_max)

        except Exception as e:
            arcpy.AddError('Could not perform sigmoidal rescaling, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Cleaning up result...')
        arcpy.SetProgressorPosition(5)

        try:
            # re-apply nan mask and convert to back to dataset
            ds = ds.where(ds_mask)
            ds = ds.to_dataset(name='sigmoid').squeeze(drop=True)
        except Exception as e:
            arcpy.AddError('Could clean result, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to ensemble dataset...')
        arcpy.SetProgressorPosition(6)

        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        ds.attrs.update({'nodatavals': np.nan})
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # remove erroneous 'descriptions' attr if exists (multi-band tif only)
        if 'descriptions' in ds.attrs:
            del ds.attrs['descriptions']

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(7)

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
        arcpy.SetProgressorPosition(8)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing sdm layer if exist
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'sigmoid.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'sigmoid' + '_' + dt)
                os.makedirs(out_folder)

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False

                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'sigmoid.crf')
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
                layer = m.listLayers('sigmoid.crf')[0]
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
        arcpy.SetProgressorPosition(9)

        # close main dataset
        ds.close()
        del ds

        # close mask dataset
        ds_mask.close()
        del ds_mask

        # notify user
        arcpy.AddMessage('Generated Sigmoidal successfully.')

        return