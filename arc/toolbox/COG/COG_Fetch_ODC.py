import arcpy

import os

from arc.toolbox.globals import STAC_ENDPOINT_ODC, RESULT_LIMIT
from modules import cog_odc
from shared import tools, arc


class COG_Fetch_ODC:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'COG Fetch ODC'
        self.description = 'COG Fetch implements the COG Open ' \
                           'Data Cube (ODC) STAC module created ' \
                           'by Digital Earth Australia (DEA). ' \
                           'This allows easy and efficient ' \
                           'downloading of analysis-ready Landsat ' \
                           '5, 7, 8 and Sentinel 2 satellite imagery ' \
                           'for any area in Australia.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input feature
        par_in_feat = arcpy.Parameter(
            displayName='Input area of interest feature',
            name='in_feat',
            datatype='GPFeatureLayer',
            parameterType='Required',
            direction='Input')
        par_in_feat.filter.list = ['Polygon']

        # output file
        par_out_nc = arcpy.Parameter(
            displayName='Output satellite NetCDF file',
            name='out_nc_path',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc.filter.list = ['nc']

        # platform
        par_platform = arcpy.Parameter(
            displayName='Satellite platform',
            name='in_platform',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_platform.filter.list = ['Landsat', 'Sentinel']
        par_platform.values = 'Landsat'

        # include slc off
        par_slc_off = arcpy.Parameter(
            displayName='Include "SLC-off" data',
            name='in_slc_off',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_slc_off.value = False

        # start date
        par_date_start = arcpy.Parameter(
            displayName='Start date',
            name='in_from_date',
            datatype='GPDate',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_date_start.values = '2018/01/01'

        # end date
        par_date_end = arcpy.Parameter(
            displayName='End date',
            name='in_to_date',
            datatype='GPDate',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_date_end.values = '2021/12/31'

        # bands
        par_bands = arcpy.Parameter(
            displayName='Bands',
            name='in_bands',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=True)
        bands = [
            'Blue',
            'Green',
            'Red',
            'NIR',
            'SWIR1',
            'SWIR2',
            'OA_Mask'
        ]
        par_bands.filter.type = 'ValueList'
        par_bands.filter.list = bands
        par_bands.values = bands

        # resolution
        par_res = arcpy.Parameter(
            displayName='Resolution',
            name='in_res',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_res.filter.type = 'Range'
        par_res.filter.list = [1, 10000]
        par_res.value = 30

        # resampling
        par_resampling = arcpy.Parameter(
            displayName='Resampling method',
            name='in_resampling',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_resampling.filter.list = ['Nearest', 'Bilinear', 'Cubic', 'Average']
        par_resampling.values = 'Nearest'

        # alignment
        par_align = arcpy.Parameter(
            displayName='Alignment',
            name='in_align',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_align.filter.type = 'Range'
        par_align.filter.list = [0, 10000]
        par_align.value = None

        # combine parameters
        parameters = [
            par_in_feat,
            par_out_nc,
            par_platform,
            par_slc_off,
            par_date_start,
            par_date_end,
            par_bands,
            par_res,
            par_resampling,
            par_align
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

        # if satellite platform has been validated...
        if parameters[2].hasBeenValidated is False:

            # modify bands list when platform changed
            if parameters[2].value == 'Landsat':

                # enable slc-off control
                parameters[3].enabled = True

                # set band list
                bands = [
                    'Blue',
                    'Green',
                    'Red',
                    'NIR',
                    'SWIR1',
                    'SWIR2',
                    'OA_Mask'
                ]

                # set bands list and select all
                parameters[6].filter.list = bands
                parameters[6].values = bands

                # set default resolution
                parameters[7].value = 30

            elif 'Sentinel' in parameters[2].value:

                # disable slc-off control
                parameters[3].enabled = False

                # set band list
                bands = [
                    'Blue',
                    'Green',
                    'Red',
                    'NIR1',
                    'SWIR2',
                    'SWIR3',
                    'OA_Mask'
                ]

                # set bands list and select all
                parameters[6].filter.list = bands
                parameters[6].values = bands

                # set default resolution
                parameters[7].value = 10

        return

    @staticmethod
    def updateMessages(parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the COG Fetch (ODC) module.
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

        # risky imports (not native to arcgis)


        # grab parameter values
        in_feat = parameters[0].valueAsText  # study area feature
        out_nc = parameters[1].valueAsText  # output nc
        in_platform = parameters[2].value  # platform name
        in_slc_off = parameters[3].value  # slc off
        in_start_date = parameters[4].value  # start date
        in_end_date = parameters[5].value  # end date
        in_bands = parameters[6].valueAsText  # bands
        in_res = parameters[7].value  # resolution
        in_resampling = parameters[8].value  # resampling method
        in_align = parameters[9].value  # alignment

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Performing STAC query...')

        # check if key parameters are valid
        if in_feat is None:
            arcpy.AddError('No input feature provided.')
            return
        elif in_platform not in ['Landsat', 'Sentinel']:
            arcpy.AddError('Platform is not supported.')
            return
        elif in_slc_off not in [True, False]:
            arcpy.AddError('SLC off not provided.')
            return
        elif in_start_date is None or in_end_date is None:
            arcpy.AddError('No start and/or end date provided.')
            return
        elif in_bands is None:
            arcpy.AddError('No platform bands provided.')
            return

        try:
            # prepare collections and bands
            collections = arc.prepare_collections_list(in_platform)
            bands = arc.prepare_band_names(in_bands, in_platform)

            # check collections and bands valid
            if len(collections) == 0 or len(bands) == 0:
                arcpy.AddError('Platform and/or bands not provided.')
                return

            # prepare start, end dates
            in_start_date = in_start_date.strftime('%Y-%m-%d')
            in_end_date = in_end_date.strftime('%Y-%m-%d')

            # check date range si valid
            if in_start_date >= in_end_date:
                arcpy.AddError('End date must be greater than start date.')
                return

            # get bbox (bl, tr) from layer in wgs84
            bbox = arc.get_layer_bbox(in_feat)

            # check bbox is valid
            if len(bbox) != 4:
                arcpy.AddError('Bounding box is invalid.')
                return

            # fetch stac items
            items = cog_odc.fetch_stac_items_odc(stac_endpoint=STAC_ENDPOINT_ODC,
                                                 collections=collections,
                                                 start_dt=in_start_date,
                                                 end_dt=in_end_date,
                                                 bbox=bbox,
                                                 slc_off=in_slc_off,
                                                 limit=RESULT_LIMIT)

            # replace s3 prefix with https (pro-friendly)
            items = cog_odc.replace_items_s3_to_https(items=items,
                                                      from_prefix='s3://dea-public-data',
                                                      to_prefix='https://data.dea.ga.gov.au')

            # notify user of number of images
            arcpy.AddMessage('Found {} satellite items.'.format(len(items)))

        except Exception as e:
            arcpy.AddError('Could not obtain items from DEA AWS, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # ensure any items exist
        if len(items) == 0:
            arcpy.AddError('No satellite items returned.')
            return

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Converting items into dataset...')

        # check if required parameters are valid
        if in_res < 1:
            arcpy.AddError('Resolution value must be > 0.')
            return
        elif in_resampling not in ['Nearest', 'Bilinear', 'Cubic', 'Average']:
            arcpy.AddError('Resampling method not supported.')
            return
        elif in_align is not None and (in_align < 0 or in_align > in_res):
            arcpy.AddError('Alignment must be > 0 but < resolution.')
            return

        try:
            # convert items to xarray dataset
            ds = cog_odc.build_xr_odc(items=items,
                                      bbox=bbox,
                                      bands=bands,
                                      crs=3577,
                                      res=in_res,
                                      resampling=in_resampling,
                                      align=in_align,
                                      group_by='solar_day',
                                      chunks={},
                                      like=None)

        except Exception as e:
            arcpy.AddError('Could not construct xarray dataset, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and set up progress bar
        arcpy.SetProgressor(type='step',
                            message='Downloading data, please wait...',
                            min_range=0,
                            max_range=len(ds) + 1)

        try:
            # iter dataset bands...
            for idx, var in enumerate(ds):
                # increment progress bar
                arcpy.SetProgressorLabel('Downloading band: {}...'.format(var))
                arcpy.SetProgressorPosition(idx)

                # compute!
                ds[var] = ds[var].compute()

        except Exception as e:
            arcpy.AddError('Could not download data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Cleaning up dataset...')

        try:
            # force re-encoding of date time to prevent export bug
            dts = ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            ds['time'] = dts.astype('datetime64[ns]')

            # set to signed int16
            ds = ds.astype('int16')

            # set all non-mask bands nodata values (0) to -999
            for var in ds:
                if 'mask' not in var:
                    ds[var] = ds[var].where(ds[var] != 0, -999)

        except Exception as e:
            arcpy.AddError('Could not finalise dataset, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Assigning extra attributes to dataset...')

        try:
            # set up additional attrs
            attrs = {
                'transform': tuple(ds.geobox.transform),
                'res': in_res,
                'nodatavals': -999,
                'orig_bbox': tuple(bbox),
                'orig_collections': tuple(collections),
                'orig_bands': tuple(bands),
                'orig_dtype': 'int16',
                'orig_slc_off': str(in_slc_off),
                'orig_resample': in_resampling
            }

            # assign attrs
            ds = ds.assign_attrs(attrs)

        except Exception as e:
            arcpy.AddError('Could not assign extra attributes, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Exporting NetCDF file...')

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Finalising process...')

        # close main dataset and del datasets
        ds.close()
        del ds

        # notify user
        arcpy.AddMessage('Generated COG Fetch (ODC) successfully.')

        return