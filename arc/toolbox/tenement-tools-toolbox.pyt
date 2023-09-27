# -*- coding: utf-8 -*-

# imports
import os
import uuid
import certifi
import arcpy
import pyproj
from pathlib import Path

# set default gdal install and certifi envs
install_dir = arcpy.GetInstallInfo().get('InstallDir')
os.environ['GDAL_DATA'] = os.path.join(install_dir, 'Resources\pedata\gdaldata')
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

# set default proj folder via pyproj
env_dir = Path(pyproj.__file__).parents[3]
prj_dir = os.path.join(env_dir, 'Library', 'share', 'proj')
pyproj.datadir.set_data_dir(prj_dir)

# get location of tenement-tool toolbox
tbx_filename = os.path.realpath(__file__)
tbx_folder = os.path.dirname(tbx_filename)
folder = os.path.dirname(tbx_folder)

# globals (non-dev)
FOLDER_MODULES = os.path.join(folder, 'modules')
FOLDER_SHARED = os.path.join(folder, 'shared')
GRP_LYR_FILE = os.path.join(folder, r'arc\lyr\group_template.lyrx')

# globals (dev)
STAC_ENDPOINT_ODC = 'https://explorer.sandbox.dea.ga.gov.au/stac'
STAC_ENDPOINT_LEG = 'https://explorer.sandbox.dea.ga.gov.au/stac/search'
RESULT_LIMIT = 20
# FOLDER_MODULES = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\modules'
# FOLDER_SHARED = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\shared'
# GRP_LYR_FILE = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\arc\lyr\group_template.lyrx'

# globals (tool parameters)
GDVSPECTRA_THRESHOLD = {}
GDVSPECTRA_TREND = {}
GDVSPECTRA_CVA = {}
PHENOLOPY_METRICS = {}
NICHER_MASKER = {}
VEGFRAX_FRACTIONAL_COVER = {}
ENSEMBLE_SIGMOIDER = {}
ENSEMBLE_MASKER = {}
NRT_CREATE_AREA = {}
NRT_MODIFY_AREA = {}
NRT_DELETE_AREA = {}
NRT_VISUALISE_AREA = {}

class Toolbox(object):
    def __init__(self):
        """
        Define the toolbox.
        """   
        
        # set name of toolbox
        self.label = "Toolbox"
        self.alias = "toolbox"

        # tools to display in toolbox
        self.tools = [
            COG_Fetch_ODC,
            COG_Fetch_Legacy,
            COG_Shift,
            COG_Explore, 
            GDVSpectra_Likelihood, 
            GDVSpectra_Threshold, 
            GDVSpectra_Trend,
            GDVSpectra_CVA,
            Phenolopy_Metrics,
            Nicher_SDM,
            Nicher_Masker,
            VegFrax_Fractional_Cover,
            Ensemble_Sigmoider,
            Ensemble_Model,
            Ensemble_Masker,
            NRT_Create_Project,
            NRT_Create_Monitoring_Areas,
            NRT_Modify_Monitoring_Areas,
            NRT_Delete_Monitoring_Areas,
            NRT_Monitor_Areas,
            NRT_Visualise_Areas,
            NRT_Build_Graphs
            ]


class COG_Fetch_ODC(object):
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

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the COG Fetch (ODC) module.
        """
        
        # safe imports
        import os, sys     
        import numpy as np 
        import arcpy       
        
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
        try:
            import xarray as xr
            import dask
            import rasterio
            import pystac_client
            from odc import stac
        except Exception as e:
            arcpy.AddError('Python libraries xarray, dask, rasterio, odc not installed.')
            arcpy.AddMessage(str(e))
            return
            
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import cog_odc
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values
        in_feat = parameters[0].valueAsText             # study area feature
        out_nc = parameters[1].valueAsText              # output nc 
        in_platform = parameters[2].value               # platform name
        in_slc_off = parameters[3].value                # slc off 
        in_start_date = parameters[4].value             # start date
        in_end_date = parameters[5].value               # end date
        in_bands = parameters[6].valueAsText            # bands
        in_res = parameters[7].value                    # resolution
        in_resampling = parameters[8].value             # resampling method 
        in_align = parameters[9].value                  # alignment    



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


class COG_Fetch_Legacy(object):
    def __init__(self):
        """
        Initialise tool.
        """
    
        # set tool name, description, options
        self.label = 'COG Fetch Legacy'
        self.description = 'COG Fetch implements the COG Open ' \
                           'Data Cube (ODC) STAC module created ' \
                           'by Digital Earth Australia (DEA). ' \
                           'This allows easy and efficient ' \
                           'downloading of analysis-ready Landsat ' \
                           '5, 7, 8 and Sentinel 2 satellite imagery ' \
                           'for any area in Australia. This is an older ' \
                           'legacy version of the tool that may work if ODC ' \
                           'version fails.'
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

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the COG Fetch (Legacy) module.
        """
        
        # safe imports
        import os, sys      
        import numpy as np  
        import arcpy        
        
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
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray, dask not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, tools, satfetcher
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import cog
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
        
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
                                            
        # grab parameter values
        in_feat = parameters[0].valueAsText             # study area feature
        out_nc = parameters[1].valueAsText              # output nc 
        in_platform = parameters[2].value               # platform name
        in_slc_off = parameters[3].value                # slc off 
        in_start_date = parameters[4].value             # start date
        in_end_date = parameters[5].value               # end date
        in_bands = parameters[6].valueAsText            # bands
        in_res = parameters[7].value                    # resolution
        


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
            feats = cog.fetch_stac_data(stac_endpoint=STAC_ENDPOINT_LEG, 
                                        collections=collections, 
                                        start_dt=in_start_date, 
                                        end_dt=in_end_date, 
                                        bbox=bbox,
                                        slc_off=in_slc_off,
                                        limit=RESULT_LIMIT)
         
            # notify user of number of images
            arcpy.AddMessage('Found {} satellite items.'.format(len(feats)))
         
        except Exception as e:
            arcpy.AddError('Could not obtain features from DEA AWS, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Preparing STAC results...')

        try:
            # convert raw stac into useable data
            meta, asset_table = cog.prepare_data(feats, 
                                                 assets=bands,
                                                 bounds_latlon=bbox, 
                                                 bounds=None, 
                                                 epsg=3577, 
                                                 resolution=in_res, 
                                                 snap_bounds=True,
                                                 force_dea_http=True)
        except Exception as e:
            arcpy.AddError('Could not prepare STAC results, see messages for details.')
            arcpy.AddMessage(str(e))
            return
        
        
        
        # # # # #
        # notify and start non-progress bar
        arcpy.SetProgressor(type='default', message='Converting to dataset...')
        
        try:
            # convert assets to dask array
            darray = cog.convert_to_dask(meta=meta, 
                                         asset_table=asset_table, 
                                         chunksize=-1,
                                         resampling='nearest', 
                                         dtype='int16', 
                                         fill_value=-999, 
                                         rescale=True)
                                     
            # generate coordinates and dimensions from metadata
            coords, dims = cog.build_coords(feats=feats,
                                            assets=bands, 
                                            meta=meta,
                                            pix_loc='topleft')
        
            # build final xarray data array
            ds = xr.DataArray(darray,
                              coords=coords,
                              dims=dims,
                              name='stac-' + dask.base.tokenize(darray))
                         
            # convert to dataset based on bands
            ds = ds.to_dataset(dim='band')
        
            # append attributes onto dataset
            ds = cog.build_attributes(ds=ds,
                                      meta=meta, 
                                      collections=collections, 
                                      bands=bands,
                                      slc_off=in_slc_off, 
                                      bbox=bbox,
                                      dtype='int16',
                                      snap_bounds=True,
                                      fill_value=-999, 
                                      rescale=True,
                                      cell_align='topleft',
                                      resampling='nearest')
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
        arcpy.AddMessage('Generated COG Fetch (Legacy) successfully.')
        
        return


class COG_Shift(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'COG Shift'
        self.description = 'Shifts the output of the COG Fetch tool by ' \
                           'a specified number of metres in space to correct ' \
                           'for offset issues.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
    
        # input netcdf data file
        par_in_nc_file = arcpy.Parameter(
                           displayName='Input satellite NetCDF file',
                           name='in_nc_file',
                           datatype='DEFile',
                           parameterType='Required',
                           direction='Input')
        par_in_nc_file.filter.list = ['nc']

        # output netcdf data file
        par_out_nc_file = arcpy.Parameter(
                            displayName='Output shifted NetCDF file',
                            name='out_nc_file',
                            datatype='DEFile',
                            parameterType='Required',
                            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # shift x
        par_shift_x = arcpy.Parameter(
                          displayName='Shift X (metres)',
                          name='in_shift_x',
                          datatype='GPDouble',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_shift_x.value = 0.0

        # shift y
        par_shift_y = arcpy.Parameter(
                          displayName='Shift Y (metres)',
                          name='in_shift_y',
                          datatype='GPDouble',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_shift_y.value = 0.0 
        

        # combine parameters
        parameters = [
            par_in_nc_file,
            par_out_nc_file,
            par_shift_x,
            par_shift_y
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
        Executes the COG Shift module.
        """
    
        # safe imports
        import os, sys   
        import arcpy     
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return  
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import tools
        
            # shared modules
            sys.path.append(FOLDER_MODULES)
            import cog
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return   

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)       
                                            
        # grab parameter values 
        in_nc = parameters[0].valueAsText            # raw input satellite netcdf
        out_nc = parameters[1].valueAsText           # shifted output satellite netcdf
        in_shift_x = parameters[2].value             # shift x
        in_shift_y = parameters[3].valueAsText       # shift y



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning COG Shift.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=4)



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
            arcpy.AddError('Input NetCDF must have all at least one x, y index.')
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

        try:
            # do a basic load (i.e. no conversion of nodata, etc.)
            with xr.open_dataset(in_nc) as ds:
                ds.load()
        except Exception as e:
            arcpy.AddError('Could not properly load input satellite NetCDF data.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Shifting NetCDF...')
        arcpy.SetProgressorPosition(2)

        # check x and y shift values are valid 
        if in_shift_x is None or in_shift_y is None:
            arcpy.AddError('No shift x and/or y value provided.')

        try:
            # shift x and y 
            ds['x'] = ds['x'] + float(in_shift_x)
            ds['y'] = ds['y'] + float(in_shift_y)
        except Exception as e:
            arcpy.AddError('Could not shift satellite NetCDF data.')
            arcpy.AddMessage(str(e))
            return
        


        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(3)   

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds, filename=out_nc)
        except Exception as e: 
                arcpy.AddError('Could not export dataset.')
                arcpy.AddMessage(str(e))
                return


        
        # # # # #
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(4)

        # close main dataset and del datasets
        ds.close()
        del ds 

        # notify user
        arcpy.AddMessage('Generated COG Shift successfully.')

        return


class COG_Explore(object):
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

    def execute(self, parameters, messages):
        """
        Executes the COG Explore module.
        """
    
        # safe imports
        import os, sys     
        import numpy as np 
        import tempfile    
        import arcpy       
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return  
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, tools, satfetcher
        
            # shared modules
            sys.path.append(FOLDER_MODULES)
            import cog
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return   

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)       
                                            
        # grab parameter values 
        in_nc = parameters[0].valueAsText            # raw input satellite netcdf
        out_folder = parameters[1].valueAsText       # output crf folder
        in_veg_idx = parameters[2].value             # vege index name
        in_fmask_flags = parameters[3].valueAsText   # fmask flag values
        in_max_cloud = parameters[4].value           # max cloud percentage
        in_interpolate = parameters[5].value         # interpolate missing pixels



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning COG Explore.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=11)



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
        arcpy.SetProgressorLabel('Calculating vegetation index...')
        arcpy.SetProgressorPosition(6) 

        # check if veg idx supported 
        if in_veg_idx.lower() not in ['ndvi', 'evi', 'savi', 'msavi', 'slavi', 'mavi', 'kndvi']:
            arcpy.AddError('Vegetation index not supported.')
            return 

        try:
            # calculate vegetation index
            ds = tools.calculate_indices(ds=ds, 
                                         index=in_veg_idx.lower(), 
                                         custom_name='veg_idx', 
                                         rescale=False, 
                                         drop=True)

            # add band attrs back on
            ds['veg_idx'].attrs = ds_band_attrs   

        except Exception as e: 
            arcpy.AddError('Could not calculate vegetation index.')
            arcpy.AddMessage(str(e))
            return
            
        # check if we sufficient data temporaly
        if len(ds['time']) == 0:
            arcpy.AddError('No dates remaining in data.')
            return      
        


        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(7)

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
        arcpy.SetProgressorPosition(8) 

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
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(9)

        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs       

        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(10)

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
                ds.to_netcdf(tmp_nc)

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
        ds.close()
        del ds 

        # notify user
        arcpy.AddMessage('Generated COG Explore successfully.')

        return


class GDVSpectra_Likelihood(object):
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

        # safe imports
        import os, sys      
        import datetime     
        import numpy as np  
        import arcpy        

        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import gdvspectra, cog 
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
         
        # grab parameter values 
        in_nc = parameters[0].valueAsText            # raw input satellite netcdf
        out_nc = parameters[1].valueAsText           # output gdv likelihood netcdf
        in_wet_months = parameters[2].valueAsText    # wet months 
        in_dry_months = parameters[3].valueAsText    # dry months 
        in_veg_idx = parameters[4].value             # vege index name
        in_mst_idx = parameters[5].value             # moisture index name       
        in_aggregate = parameters[6].value           # aggregate output
        in_zscore_pvalue = parameters[7].value       # zscore pvalue
        in_ivt_qupper = parameters[8].value          # upper quantile for standardisation
        in_ivt_qlower = parameters[9].value          # lower quantile for standardisation
        in_fmask_flags = parameters[10].valueAsText  # fmask flag values
        in_max_cloud = parameters[11].value          # max cloud percentage
        in_interpolate = parameters[12].value        # interpolate missing pixels
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
        if in_zscore_pvalue not in [0.01, 0.05, 0.1, None]:
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


class GDVSpectra_Threshold(object):
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
                             #category='Additional Options',
                             multiValue=False)
        par_remove_stray.value = True
        
        # binarise checkbox
        par_convert_binary = arcpy.Parameter(
                               displayName='Binarise result',
                               name='in_convert_binary',
                               datatype='GPBoolean',
                               parameterType='Required',
                               direction='Input',
                               #category='Additional Options',
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
        
        # imports
        try:
            import numpy as np
            import xarray as xr           
        except:
            arcpy.AddError('Python libraries xarray not installed.')
            return

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
        
        # safe imports
        import os, sys       
        import datetime      
        import numpy as np   
        import pandas as pd  
        import arcpy         
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
                
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import gdvspectra 
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
        
        # grab parameter values 
        in_nc = parameters[0].valueAsText                # likelihood netcdf
        out_nc = parameters[1].valueAsText               # output netcdf
        in_aggregate = parameters[2].value               # aggregate dates
        in_specific_years = parameters[3].valueAsText    # set specific year 
        in_type = parameters[4].value                    # threshold type
        in_std_dev = parameters[5].value                 # std dev threshold value 
        in_occurrence_feat = parameters[6]               # occurrence shp path 
        in_pa_column = parameters[7].value               # occurrence shp pres/abse col 
        in_remove_stray = parameters[8].value            # apply salt n pepper -- requires sa
        in_convert_binary = parameters[9].value          # convert thresh to binary 1, nan
        in_add_result_to_map = parameters[10].value      # add result to map


        
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
            #print(str(e))
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


class GDVSpectra_Trend(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'GDVSpectra Trend'
        self.description = 'Perform a time-series trend analysis on an existing ' \
                           'GDV Likelihood data cube. Produces a map of areas where ' \
                           'vegetation has continuously increased, decreased or ' \
                           'has not changed.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # input like netcdf file
        par_like_nc_file = arcpy.Parameter(
                             displayName='Input GDV Likelihood NetCDF file',
                             name='in_like_nc_file',
                             datatype='DEFile',
                             parameterType='Required',
                             direction='Input')
        par_like_nc_file.filter.list = ['nc']
        
        # input mask netcdf file
        par_mask_nc_file = arcpy.Parameter(
                             displayName='Input GDV Threshold mask NetCDF file',
                             name='in_mask_nc_file',
                             datatype='DEFile',
                             parameterType='Optional',
                             direction='Input')
        par_mask_nc_file.filter.list = ['nc']
        
        # output netcdf location
        par_out_nc_file = arcpy.Parameter(
                                  displayName='Output GDV Trend NetCDF file',
                                  name='out_nc_file',
                                  datatype='DEFile',
                                  parameterType='Required',
                                  direction='Output')
        par_out_nc_file.filter.list = ['nc']
        
        # use all years
        par_use_all_years = arcpy.Parameter(
                          displayName='Combine all input years',
                          name='in_use_all_years',
                          datatype='GPBoolean',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_use_all_years.value = True
        
        # set specific start year
        par_start_year = arcpy.Parameter(
                               displayName='Start year of trend analysis',
                               name='in_start_year',
                               datatype='GPLong',
                               parameterType='Optional',
                               direction='Input',
                               multiValue=False)
        par_start_year.filter.type = 'ValueList'
        par_start_year.filter.list = []
        
        # set specific end year
        par_end_year = arcpy.Parameter(
                               displayName='End year of trend analysis',
                               name='in_end_year',
                               datatype='GPLong',
                               parameterType='Optional',
                               direction='Input',
                               multiValue=False)
        par_end_year.filter.type = 'ValueList'
        par_end_year.filter.list = []        
        
        # set analysis type
        par_analysis_type = arcpy.Parameter(
                              displayName='Trend analysis method',
                              name='in_analysis_type',
                              datatype='GPString',
                              parameterType='Required',
                              direction='Input',
                              multiValue=False)
        par_analysis_type.filter.type = 'ValueList'
        par_analysis_type.filter.list = ['Mann-Kendall', 'Theil-Sen Slope']
        par_analysis_type.value = 'Mann-Kendall'
        
        # mk p-value
        par_mk_pvalue = arcpy.Parameter(
                          displayName='P-value',
                          name='in_mk_pvalue',
                          datatype='GPDouble',
                          parameterType='Optional',
                          direction='Input',
                          multiValue=False)
        par_mk_pvalue.filter.type = 'Range'
        par_mk_pvalue.filter.list = [0.001, 1.0]
        par_mk_pvalue.value = 0.05

        # mk direction
        par_mk_direction = arcpy.Parameter(
                            displayName='Trend direction',
                            name='in_mk_direction',
                            datatype='GPString',
                            parameterType='Optional',
                            direction='Input',
                            multiValue=False)
        par_mk_direction.filter.type = 'ValueList'
        par_mk_direction.filter.list = ['Both', 'Increasing', 'Decreasing']
        par_mk_direction.value = 'Both'
        
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
            par_like_nc_file,
            par_mask_nc_file,
            par_out_nc_file,
            par_use_all_years,
            par_start_year,
            par_end_year,
            par_analysis_type,
            par_mk_pvalue,
            par_mk_direction,
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
        
        # imports
        try:
            import numpy as np
            import xarray as xr           
        except:
            arcpy.AddError('Python libraries xarray not installed.')
            return
        
        # globals 
        global GDVSPECTRA_TREND

        # unpack global parameter values 
        curr_file = GDVSPECTRA_TREND.get('in_file')
        
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

                # populate start and end year lists
                parameters[4].filter.list = dts
                parameters[5].filter.list = dts
                
                # reset start and end year selections
                if len(dts) != 0:
                    parameters[4].value = dts[0]
                    parameters[5].value = dts[-1]
                else:
                    parameters[4].value = None
                    parameters[5].value = None
        
        
        # update global values
        GDVSPECTRA_TREND = {
            'in_file': parameters[0].valueAsText,
        }
        
        # enable start and end years based on all years checkbox 
        if parameters[3].value is False:
            parameters[4].enabled = True
            parameters[5].enabled = True
        else:
            parameters[4].enabled = False
            parameters[5].enabled = False

        # enable relevant controls when manken or theils
        if parameters[6].valueAsText == 'Mann-Kendall':
            parameters[7].enabled = True
            parameters[8].enabled = True
        else:
            parameters[7].enabled = False
            parameters[8].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra Trend module.
        """
        
        # safe imports
        import os, sys     
        import datetime    
        import numpy as np 
        import arcpy       
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
                
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import gdvspectra 
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
            
        # grab parameter values 
        in_like_nc = parameters[0].valueAsText         # likelihood netcdf
        in_mask_nc = parameters[1].valueAsText         # thresh mask netcdf
        out_nc = parameters[2].valueAsText             # output netcdf
        in_use_all_years = parameters[3].value         # use all years
        in_start_year = parameters[4].value            # start year
        in_end_year = parameters[5].value              # end year
        in_trend_method = parameters[6].value          # trend method
        in_mk_pvalue = parameters[7].value             # mk pvalue
        in_mk_direction = parameters[8].value          # mk direction
        in_add_result_to_map = parameters[9].value    # add result to map
        
        

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra Trend.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=10)

        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking likelihood netcdf...')
        arcpy.SetProgressorPosition(1)
        
        try:
            # do quick lazy load of likelihood netcdf for checking
            ds_like = xr.open_dataset(in_like_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input likelihood NetCDF data.')
            arcpy.AddMessage(str(e))
            return
        
        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_like, xr.Dataset):
            arcpy.AddError('Input likelihood NetCDF must be a xr dataset.')
            return
        elif len(ds_like) == 0:
            arcpy.AddError('Input likelihood NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds_like.dims or 'y' not in ds_like.dims or 'time' not in ds_like.dims:
            arcpy.AddError('Input likelihood NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds_like.coords or 'y' not in ds_like.coords or 'time' not in ds_like.coords:
            arcpy.AddError('Input likelihood NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds_like.coords:
            arcpy.AddError('Input likelihood NetCDF must have a spatial_ref coord.')
            return
        elif len(ds_like['x']) == 0 or len(ds_like['y']) == 0:
            arcpy.AddError('Input likelihood NetCDF must have at least one x, y index.')
            return        
        elif len(ds_like['time']) <= 3:
            arcpy.AddError('Input likelihood NetCDF must have 4 or more times.')
            return                    
        elif 'like' not in ds_like:
            arcpy.AddError('Input likelihood NetCDF must have a "like" variable. Run GDVSpectra Likelihood.')
            return
        elif not hasattr(ds_like, 'time.year') or not hasattr(ds_like, 'time.month'):
            arcpy.AddError('Input likelihood NetCDF must have time with year and month index.')
            return
        elif ds_like.attrs == {}:
            arcpy.AddError('Input likelihood NetCDF attributes not found. NetCDF must have attributes.')
            return
        elif not hasattr(ds_like, 'crs'):
            arcpy.AddError('Input likelihood NetCDF CRS attribute not found. CRS required.')
            return
        elif ds_like.crs != 'EPSG:3577':
            arcpy.AddError('Input likelihood NetCDF CRS is not EPSG:3577. EPSG:3577 required.')            
            return 
        elif not hasattr(ds_like, 'nodatavals'):
            arcpy.AddError('Input likelihood NetCDF nodatavals attribute not found.')            
            return 

        # check if variables (should only be like) are empty
        if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
            arcpy.AddError('Input likelihood NetCDF "like" variable is empty. Please download again.')            
            return 

        try:
            # now, do proper open of likelihood netcdf (set nodata to nan)
            ds_like = satfetcher.load_local_nc(nc_path=in_like_nc, 
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
        ds_attrs = ds_like.attrs
        ds_band_attrs = ds_like['like'].attrs
        ds_spatial_ref_attrs = ds_like['spatial_ref'].attrs
        
        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Grouping dates, if required...')
        arcpy.SetProgressorPosition(3)
        
        # remove potential datetime duplicates (group by day)
        ds_like = satfetcher.group_by_solar_day(ds_like)
        
        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset based on time, if requested...')
        arcpy.SetProgressorPosition(4)
        
        # if requested...
        if in_use_all_years is False:
        
            # check start, end year valid
            if in_start_year is None or in_end_year is None:
                arcpy.AddError('Did not specify start and/or end years.')
                return         
            elif in_start_year >= in_end_year:
                arcpy.AddError('Start year must be < end year.')
                return
            elif abs(in_end_year - in_start_year) < 3:
                arcpy.AddError('Require 4 or more years of data.')
                return
                
            # check if both years in dataset
            years = ds_like['time.year']
            if in_start_year not in years or in_end_year not in years:
                arcpy.AddError('Start year is not in likelihood NetCDF.')
                return
                
            # subset likelihood dataset based on years 
            ds_like = ds_like.where((ds_like['time.year'] >= in_start_year) &
                                    (ds_like['time.year'] <= in_end_year), drop=True)
            
            
            # check if more than three years still exist 
            if len(ds_like['time']) < 4:
                arcpy.AddError('Subset of likelihood NetCDF resulted in < 4 years of data.')
                return
            
            # check if variables (should only be like) are empty
            if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
                arcpy.AddError('Subset of likelihood NetCDF is empty. Increase year range.')            
                return
                
                
                
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(5)

        try:
            # compute! 
            ds_like = ds_like.compute()
        except Exception as e: 
            arcpy.AddError('Could not compute likelihood dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return       
        
        # check if we still have values
        if ds_like.to_array().isnull().all():
            arcpy.AddError('Input likelihood NetCDF is empty. Please download again.')            
            return 
            
            

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading, checking and applying mask, if requested...')
        arcpy.SetProgressorPosition(6)

        # if requested...
        if in_mask_nc is not None:
        
            try:
                # do quick lazy load of mask netcdf for checking
                ds_mask = xr.open_dataset(in_mask_nc)
            except Exception as e:
                arcpy.AddWarning('Could not quick load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return
            
            # check xr type, vars, coords, dims, attrs
            if not isinstance(ds_mask, xr.Dataset):
                arcpy.AddError('Input mask NetCDF must be a xr dataset.')
                return
            elif len(ds_mask) == 0:
                arcpy.AddError('Input mask NetCDF has no data/variables/bands.')
                return
            elif 'x' not in ds_mask.dims or 'y' not in ds_mask.dims:
                arcpy.AddError('Input mask NetCDF must have x, y dimensions.')
                return
            elif 'x' not in ds_mask.coords or 'y' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have x, y coords.')
                return
            elif 'spatial_ref' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have a spatial_ref coord.')
                return
            elif len(ds_mask['x']) == 0 or len(ds_mask['y']) == 0:
                arcpy.AddError('Input mask NetCDF must have at least one x, y index.')
                return        
            elif 'time' in ds_mask:
                arcpy.AddError('Input mask NetCDF must not have a time dimension.')
                return                    
            elif 'thresh' not in ds_mask:
                arcpy.AddError('Input mask NetCDF must have a "thresh" variable. Run GDVSpectra Threshold.')
                return
            elif ds_mask.attrs == {}:
                arcpy.AddError('Input mask NetCDF attributes not found. NetCDF must have attributes.')
                return
            elif not hasattr(ds_mask, 'crs'):
                arcpy.AddError('Input mask NetCDF CRS attribute not found. CRS required.')
                return
            elif ds_mask.crs != 'EPSG:3577':
                arcpy.AddError('Input mask NetCDF CRS is not EPSG:3577. EPSG:3577 required.')            
                return 
            elif not hasattr(ds_mask, 'nodatavals'):
                arcpy.AddError('Input mask NetCDF nodatavals attribute not found.')            
                return 

            # check if variables (should only be thresh) are empty
            if ds_mask['thresh'].isnull().all() or (ds_mask['thresh'] == 0).all():
                arcpy.AddError('Input mask NetCDF "thresh" variable is empty. Please download again.')            
                return 

            try:
                # now, do proper open of mask netcdf (set nodata to nan)
                ds_mask = satfetcher.load_local_nc(nc_path=in_mask_nc, 
                                                   use_dask=True, 
                                                   conform_nodata_to=np.nan)
                                                   
                # compute it!
                ds_mask = ds_mask.compute()
                
            except Exception as e:
                arcpy.AddError('Could not properly load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return
                
            try:
                # check if like and mask datasets overlap 
                if not tools.all_xr_intersect([ds_like, ds_mask]):
                    arcpy.AddError('Input datasets do not intersect.')
                    return
                
                # resample mask dataset to match likelihood 
                ds_mask = tools.resample_xr(ds_from=ds_mask, 
                                            ds_to=ds_like,
                                            resampling='nearest')

                # squeeze it to be safe
                ds_mask = ds_mask.squeeze(drop=True)
                
            except Exception as e:
                arcpy.AddError('Could not intersect input datasets.')
                arcpy.AddMessage(str(e))
                return
                
            # we made it, so mask likelihood
            ds_like = ds_like.where(~ds_mask['thresh'].isnull())
            
            # check if variables (should only be thresh) are empty
            if ds_like['like'].isnull().all() or (ds_like['like'] == 0).all():
                arcpy.AddError('Masked likelihood dataset is empty.')            
                return 

        
        
        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Performing trend analysis...')
        
        # check if trend method is supported 
        if in_trend_method not in ['Mann-Kendall', 'Theil-Sen Slope']:
            arcpy.AddError('Trend method not supported.')            
            return
        
        # check and perform mann-kendall or theil sen
        if in_trend_method == 'Mann-Kendall':

            # check mannkendall pvalue 
            if in_mk_pvalue is None:
                arcpy.AddError('Mann-Kendall p-value not provided.')            
                return
            elif in_mk_pvalue < 0.001 or in_mk_pvalue > 1.0:
                arcpy.AddError('Mann-Kendall p-value must be between 0.001 and 1.0.')            
                return
                
            # check mannkendall direction 
            if in_mk_direction not in ['Increasing', 'Decreasing', 'Both']:
                arcpy.AddError('Mann-Kendall direction not supported.')            
                return
                
            # prepare mannkendal direction (must be inc, dec or both)
            if in_mk_direction in ['Increasing', 'Decreasing']:
                in_mk_direction = in_mk_direction.lower()[:3]
            else:
                in_mk_direction = 'both'
                
            try:
                # perform mann-kendall trend analysis
                ds_trend = gdvspectra.perform_mk_original(ds=ds_like, 
                                                          pvalue=in_mk_pvalue, 
                                                          direction=in_mk_direction)
                                                          
                # warn if no change values returned
                if ds_trend['tau'].isnull().all():
                    arcpy.AddError('Trend output is empty, check p-value, date range and/or mask.')
                    return
                    
            except Exception as e:
                arcpy.AddError('Could not perform Mann-Kendall trend analysis.')
                arcpy.AddMessage(str(e))
                return    

        else:            
            try:
                # perform theil-sen trend analysis
                ds_trend = gdvspectra.perform_theilsen_slope(ds=ds_like, 
                                                             alpha=0.95)
                                                             
                # warn if no change values returned
                if ds_trend['theil'].isnull().all():
                    arcpy.AddError('Trend output is empty, check date range and/or mask.')
                    return
            
            except Exception as e:
                arcpy.AddError('Could not perform Theil-Sen trend analysis.')
                arcpy.AddMessage(str(e))
                return    



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(7)
        
        # append attrbutes on to dataset and bands
        ds_trend.attrs = ds_attrs
        ds_trend['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_trend:
            ds_trend[var].attrs = ds_band_attrs
            
        
        
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(8)   

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_trend, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return
            


        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(9)

        # if requested...
        if in_add_result_to_map:
            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap
                
                # remove trend layer if already exists 
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'trend.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'trend' + '_' + dt)
                os.makedirs(out_folder)                

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False
                
                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'trend.crf')
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
                layer = m.listLayers('trend.crf')[0]
                sym = layer.symbology

                # if layer has stretch coloriser, apply color
                if hasattr(sym, 'colorizer'):

                    # apply percent clip type
                    sym.colorizer.stretchType = 'PercentClip'
                    sym.colorizer.minPercent = 0.75
                    sym.colorizer.maxPercent = 0.75

                    # set default trend cmap, override if mannkenn inc or dec used
                    cmap = aprx.listColorRamps('Red-Blue (Continuous)')[0]
                    if in_trend_method == 'Mann-Kendall':
                        if in_mk_direction == 'inc':
                            cmap = aprx.listColorRamps('Yellow-Green-Blue (Continuous)')[0]
                        elif in_mk_direction == 'dec':
                            cmap = aprx.listColorRamps('Yellow-Orange-Red (Continuous)')[0]
                        
                    # apply colormap
                    sym.colorizer.colorRamp = cmap

                    # invert colormap if mannkenn decline 
                    if in_trend_method == 'Mann-Kendall' and in_mk_direction == 'dec':
                        sym.colorizer.invertColorRamp = True
                    else:
                        sym.colorizer.invertColorRamp = False
                    
                    # set gamma
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
        arcpy.SetProgressorPosition(10)
        
        # close likelihood dataset
        ds_like.close()
        del ds_like
        
        # close trend dataset
        ds_trend.close() 
        del ds_trend
        
        # close mask (if exists)
        if in_mask_nc is not None:
            ds_mask.close()
            del ds_mask
            
        # notify user
        arcpy.AddMessage('Generated GDV Trend successfully.')  
                
        return


class GDVSpectra_CVA(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'GDVSpectra CVA'
        self.description = 'Perform a Change Vector Analysis (CVA) on a data cube.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # input netcdf file
        par_raw_nc_path = arcpy.Parameter(
                            displayName='Input satellite NetCDF file',
                            name='in_raw_nc_path',
                            datatype='DEFile',
                            parameterType='Required',
                            direction='Input')
        par_raw_nc_path.filter.list = ['nc']
        
        # input netcdf mask (thresh) file
        par_mask_nc_path = arcpy.Parameter(
                             displayName='Input GDV Threshold mask NetCDF file',
                             name='in_mask_nc_path',
                             datatype='DEFile',
                             parameterType='Optional',
                             direction='Input')
        par_mask_nc_path.filter.list = ['nc']
        
        # output folder location
        par_out_nc_path = arcpy.Parameter(
                            displayName='Output CVA NetCDF file',
                            name='out_nc_path',
                            datatype='DEFile',
                            parameterType='Required',
                            direction='Output')
        par_out_nc_path.filter.list = ['nc']
        
        # base start year
        par_base_start_year = arcpy.Parameter(
                                displayName='Baseline start year',
                                name='in_base_start_year',
                                datatype='GPLong',
                                parameterType='Required',
                                direction='Input',
                                multiValue=False)
        par_base_start_year.filter.type = 'ValueList'
        par_base_start_year.filter.list = []
        
        # base end year
        par_base_end_year = arcpy.Parameter(
                              displayName='Baseline end year',
                              name='in_base_end_year',
                              datatype='GPLong',
                              parameterType='Required',
                              direction='Input',
                              multiValue=False)
        par_base_end_year.filter.type = 'ValueList'
        par_base_end_year.filter.list = []

        # comparison start year
        par_comp_start_year = arcpy.Parameter(
                                displayName='Comparison start year',
                                name='in_comp_start_year',
                                datatype='GPLong',
                                parameterType='Required',
                                direction='Input',
                                multiValue=False)
        par_comp_start_year.filter.type = 'ValueList'
        par_comp_start_year.filter.list = []
        
        # comparison end year
        par_comp_end_year = arcpy.Parameter(
                              displayName='Comparison end year',
                              name='in_comp_end_year',
                              datatype='GPLong',
                              parameterType='Required',
                              direction='Input',
                              multiValue=False)
        par_comp_end_year.filter.type = 'ValueList'
        par_comp_end_year.filter.list = []

        # analysis months
        par_analysis_months = arcpy.Parameter(
                                displayName='Set analysis month(s)',
                                name='in_analysis_months',
                                datatype='GPLong',
                                parameterType='Required',
                                direction='Input',
                                multiValue=True)
        par_analysis_months.filter.type = 'ValueList'
        par_analysis_months.filter.list = [m for m in range(1, 13)]
        par_analysis_months.value = [9, 10, 11]

        # cva magnitude threshold 
        par_tmf = arcpy.Parameter(
                    displayName='Magnitude threshold',
                    name='in_tmf',
                    datatype='GPDouble',
                    parameterType='Required',
                    direction='Input',
                    multiValue=False)
        par_tmf.filter.type = 'Range'
        par_tmf.filter.list = [0.0, 100.0]  
        par_tmf.value = 2.0  

        # set q upper for standardisation
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
        
        # set q lower for standardisation
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
        
        # set oa class values
        par_fmask_flags = arcpy.Parameter(displayName='Include pixel flags',
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
        
        # interpolate
        par_interpolate = arcpy.Parameter(
                            displayName='Interpolate NoData pixels',
                            name='in_interpolate',
                            datatype='GPBoolean',
                            parameterType='Required',
                            direction='Input',
                            category='Satellite Quality Options',
                            multiValue=False)
        par_interpolate.value = True
        
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
            par_raw_nc_path,
            par_mask_nc_path,
            par_out_nc_path,
            par_base_start_year,
            par_base_end_year,
            par_comp_start_year,
            par_comp_end_year,
            par_analysis_months,
            par_tmf,
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
        """
        Enable and disable certain parameters when
        controls are changed on ArcGIS Pro panel.
        """
        
        # imports
        try:
            import numpy as np
            import xarray as xr
        except:
            arcpy.AddError('Python library xarray not installed.')
            return
            
        # globals 
        global GDVSPECTRA_CVA

        # unpack global parameter values 
        curr_file = GDVSPECTRA_CVA.get('in_file')
                
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

                # populate baseline start and end year lists
                parameters[3].filter.list = dts
                parameters[4].filter.list = dts
                
                # populate comparison start and end year lists
                parameters[5].filter.list = dts
                parameters[6].filter.list = dts
                
                # reset baseline start and end year selections
                if len(dts) != 0:
                    parameters[3].value = dts[0]
                    parameters[4].value = dts[0]
                    parameters[5].value = dts[-1]
                    parameters[6].value = dts[-1]  
                else:
                    parameters[3].value = None
                    parameters[4].value = None
                    parameters[5].value = None
                    parameters[6].value = None
                    
        # update global values
        GDVSPECTRA_CVA = {
            'in_file': parameters[0].valueAsText,
        }
            
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra CVA module.
        """
        
        # safe imports
        import os, sys       
        import datetime      
        import numpy as np   
        import pandas as pd  
        import arcpy         
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
                
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import gdvspectra, cog 
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
        
        # grab parameter values 
        in_raw_nc = parameters[0].valueAsText               # raw input satellite netcdf
        in_mask_nc = parameters[1].valueAsText              # mask input satellite netcdf
        out_nc = parameters[2].valueAsText                  # output gdv likelihood netcdf
        in_base_start_year =  parameters[3].value           # base start year
        in_base_end_year =  parameters[4].value             # base end year
        in_comp_start_year =  parameters[5].value           # comp start year
        in_comp_end_year =  parameters[6].value             # comp end year
        in_analysis_months = parameters[7].valueAsText      # analysis months
        in_tmf = parameters[8].value                        # magnitude threshold
        in_ivt_qupper = parameters[9].value                 # upper quantile for standardisation
        in_ivt_qlower = parameters[10].value                # lower quantile for standardisation
        in_fmask_flags = parameters[11].valueAsText         # fmask flag values
        in_max_cloud = parameters[12].value                 # max cloud percentage
        in_interpolate = parameters[13].value               # interpolate missing pixels
        in_add_result_to_map = parameters[14].value         # add result to map



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning GDVSpectra CVA.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=18)
                            
                            
                            
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking satellite netcdf...')
        arcpy.SetProgressorPosition(1)
        
        try:
            # do quick lazy load of satellite netcdf for checking
            ds = xr.open_dataset(in_raw_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load input satellite NetCDF data.')
            arcpy.AddMessage(str(e))
            return
        
        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds, xr.Dataset):
            arcpy.AddError('Input satellite NetCDF must be a xr dataset.')
            return
        elif len(ds) == 0:
            arcpy.AddError('Input NetCDF has no data/variables/bands.')
            return
        elif 'x' not in ds.dims or 'y' not in ds.dims or 'time' not in ds.dims:
            arcpy.AddError('Input satellite NetCDF must have x, y and time dimensions.')
            return
        elif 'x' not in ds.coords or 'y' not in ds.coords or 'time' not in ds.coords:
            arcpy.AddError('Input satellite NetCDF must have x, y and time coords.')
            return
        elif 'spatial_ref' not in ds.coords:
            arcpy.AddError('Input satellite NetCDF must have a spatial_ref coord.')
            return
        elif len(ds['x']) == 0 or len(ds['y']) == 0 or len(ds['time']) == 0:
            arcpy.AddError('Input satellite NetCDF must have all at least one x, y and time index.')
            return
        elif 'oa_fmask' not in ds and 'fmask' not in ds:
            arcpy.AddError('Expected cloud mask band not found in satellite NetCDF.')
            return
        elif not hasattr(ds, 'time.year') or not hasattr(ds, 'time.month'):
            arcpy.AddError('Input satellite NetCDF must have time with year and month component.')
            return
        elif len(ds.groupby('time.year')) < 2:
            arcpy.AddError('Input satellite NetCDF must have >= 2 years of data.')
            return
        elif ds.attrs == {}:
            arcpy.AddError('Satellite NetCDF must have attributes.')
            return
        elif not hasattr(ds, 'crs'):
            arcpy.AddError('Satellite NetCDF CRS attribute not found. CRS required.')
            return
        elif ds.crs != 'EPSG:3577':
            arcpy.AddError('Satellite NetCDF CRS is not in GDA94 Albers (EPSG:3577).')            
            return 
        elif not hasattr(ds, 'nodatavals'):
            arcpy.AddError('Satellite NetCDF nodatavals attribute not found.')            
            return 
            
        # efficient: if all nan, 0 at first var, assume rest same, so abort
        if ds[list(ds)[0]].isnull().all() or (ds[list(ds)[0]] == 0).all():
            arcpy.AddError('Satellite NetCDF has empty variables. Please download again.')            
            return 
            
        try:
            # now, do proper open of satellite netcdf properly (and set nodata to nan)
            ds = satfetcher.load_local_nc(nc_path=in_raw_nc, 
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
            arcpy.AddError('Could not cloud mask pixels.')
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
                arcpy.AddError('Satellite NetCDF is missing band: {}. Need all bands.'.format(band))
                return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing dataset months to requested...')
        arcpy.SetProgressorPosition(6)

        # prepare analysis month(s)
        if in_analysis_months == '':
            arcpy.AddError('Must include at least one analysis month.')
            return
            
        # unpack month(s)
        analysis_months = [int(e) for e in in_analysis_months.split(';')]

        try:
            # reduce xr dataset into only analysis months
            ds = gdvspectra.subset_months(ds=ds, 
                                          month=analysis_months,
                                          inplace=True)
        except Exception as e: 
            arcpy.AddError('Could not subset Satellite NetCDF by analysis months.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Calculating tasselled cap vegetation index...')
        arcpy.SetProgressorPosition(7)        
        
        try:
            # calculate tasselled cap green and bare
            ds = tools.calculate_indices(ds=ds, 
                                         index=['tcg', 'tcb'], 
                                         rescale=False, 
                                         drop=True)

            # add band attrs back on
            ds['tcg'].attrs = ds_band_attrs   
            ds['tcb'].attrs = ds_band_attrs

        except Exception as e: 
            arcpy.AddError('Could not calculate tasselled cap index.')
            arcpy.AddMessage(str(e))
            return

        

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Reducing month(s) into annual medians...')
        arcpy.SetProgressorPosition(8)
        
        # reduce months into annual medians (year starts, YS)
        try:
            ds = gdvspectra.resample_to_freq_medians(ds=ds,
                                                     freq='YS',
                                                     inplace=True)
                                                     
            # add band attrs back on
            ds['tcg'].attrs = ds_band_attrs   
            ds['tcb'].attrs = ds_band_attrs
            
        except Exception as e: 
            arcpy.AddError('Could not resample months in Satellite NetCDF.')
            arcpy.AddMessage(str(e))
            return
            


        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(9)

        try:
            # compute! 
            ds = ds.compute()
        except Exception as e: 
            arcpy.AddError('Could not compute dataset. See messages for details.')
            arcpy.AddMessage(str(e))
            return
        

        # check if all nan again
        if ds.to_array().isnull().all():
            arcpy.AddError('Satellite NetCDF is empty. Please download again.')            
            return     
            


        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Interpolating dataset, if requested...')
        arcpy.SetProgressorPosition(10) 

        # if requested...
        if in_interpolate:
            try:
                # interpolate along time dimension (linear)
                ds = tools.perform_interp(ds=ds, method='full')
            except Exception as e: 
                arcpy.AddError('Could not interpolate satellite NetCDF.')
                arcpy.AddMessage(str(e))
                return        



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Standardising data to invariant targets...')
        arcpy.SetProgressorPosition(11)

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
            # standardise to targets
            ds = gdvspectra.standardise_to_targets(ds, 
                                                   q_upper=in_ivt_qupper, 
                                                   q_lower=in_ivt_qlower)
        except Exception as e: 
            arcpy.AddError('Could not standardise satellite data to invariant targets.')
            arcpy.AddMessage(str(e))
            return
            
        # final check to see if data exists
        if ds['tcg'].isnull().all() or ds['tcb'].isnull().all():
            arcpy.AddError('Could not standardise satellite NetCDF.')
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Performing CVA...')
        arcpy.SetProgressorPosition(12)
        
        # check baseline and comparison start and end years
        if in_base_end_year < in_base_start_year:
            arcpy.AddError('Baseline end year must not be < start year.')
            return
        elif in_comp_end_year < in_comp_start_year:
            arcpy.AddError('Comparison end year must not be < start year.')
            return
        elif in_comp_start_year < in_base_start_year:
            arcpy.AddError('Comparison start year must not be < baseline start year.')
            return
        
        # check if baseline and comparison years in dataset
        years = ds['time.year']
        if in_base_start_year not in years or in_base_end_year not in years:
            arcpy.AddError('Baseline start and end years not found in dataset.')
            return
        elif in_comp_start_year not in years or in_comp_end_year not in years:
            arcpy.AddError('Comparison start and end years not found in dataset.')
            return
            
        # check magnitude value 
        if in_tmf < 0 or in_tmf > 100:
            arcpy.AddError('CVA threshold magnitude must be between 0 and 100.')
            return

        try:
            # generate cva
            ds_cva = gdvspectra.perform_cva(ds=ds,
                                            base_times=(in_base_start_year, in_base_end_year),
                                            comp_times=(in_comp_start_year, in_comp_end_year),
                                            reduce_comp=False,
                                            vege_var = 'tcg',
                                            soil_var = 'tcb',
                                            tmf=in_tmf)
        except Exception as e: 
            arcpy.AddError('Could not perform CVA.')
            arcpy.AddMessage(str(e))
            return


        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Isolating CVA magnitude quartiles...')
        arcpy.SetProgressorPosition(13)                                        
        
        try:
            # isolate cva magnitude via angle quartiles
            ds_cva = gdvspectra.isolate_cva_quarters(ds=ds_cva, 
                                                     drop_orig_vars=True)
        except Exception as e: 
            arcpy.AddError('Could not isolate CVA quartiles.')
            arcpy.AddMessage(str(e))
            return                                                     

        # check if variables are empty
        if ds_cva.to_array().isnull().all():
            arcpy.AddError('CVA dataset is empty. Check range of years and magnitude threshold.')            
            return   

        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading, checking and applying mask, if requested...')
        arcpy.SetProgressorPosition(14)

        # if requested...
        if in_mask_nc is not None:
        
            try:
                # do quick lazy load of mask netcdf for checking
                ds_mask = xr.open_dataset(in_mask_nc)
            except Exception as e:
                arcpy.AddWarning('Could not quick load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return
        
            # check xr type, vars, coords, dims, attrs
            if not isinstance(ds_mask, xr.Dataset):
                arcpy.AddError('Input mask NetCDF must be a xr dataset.')
                return
            elif len(ds_mask) == 0:
                arcpy.AddError('Input mask NetCDF has no data/variables/bands.')
                return
            elif 'x' not in ds_mask.dims or 'y' not in ds_mask.dims:
                arcpy.AddError('Input mask NetCDF must have x, y dimensions.')
                return
            elif 'x' not in ds_mask.coords or 'y' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have x, y and time coords.')
                return
            elif 'spatial_ref' not in ds_mask.coords:
                arcpy.AddError('Input mask NetCDF must have a spatial_ref coord.')
                return
            elif len(ds_mask['x']) == 0 or len(ds_mask['y']) == 0:
                arcpy.AddError('Input mask NetCDF must have at least one x, y index.')
                return        
            elif 'time' in ds_mask:
                arcpy.AddError('Input mask NetCDF must not have a time dimension.')
                return                    
            elif 'thresh' not in ds_mask:
                arcpy.AddError('Input mask NetCDF must have a "thresh" variable. Run GDVSpectra Threshold.')
                return
            elif ds_mask.attrs == {}:
                arcpy.AddError('Input mask NetCDF attributes not found. NetCDF must have attributes.')
                return
            elif not hasattr(ds_mask, 'crs'):
                arcpy.AddError('Input mask NetCDF CRS attribute not found. CRS required.')
                return
            elif ds_mask.crs != 'EPSG:3577':
                arcpy.AddError('Input mask NetCDF CRS is not EPSG:3577. EPSG:3577 required.')            
                return 
            elif not hasattr(ds_mask, 'nodatavals'):
                arcpy.AddError('Input mask NetCDF nodatavals attribute not found.')            
                return 
        
            # check if variables (should only be thresh) are empty
            if ds_mask['thresh'].isnull().all() or (ds_mask['thresh'] == 0).all():
                arcpy.AddError('Input mask NetCDF "thresh" variable is empty. Please download again.')            
                return 
        
            try:
                # now, do proper open of mask netcdf (set nodata to nan)
                ds_mask = satfetcher.load_local_nc(nc_path=in_mask_nc, 
                                                   use_dask=True, 
                                                   conform_nodata_to=np.nan)

                # compute it!
                ds_mask = ds_mask.compute()

            except Exception as e:
                arcpy.AddError('Could not properly load input mask NetCDF data.')
                arcpy.AddMessage(str(e))
                return
                
            try:
                # check if like and mask datasets overlap 
                if not tools.all_xr_intersect([ds_cva, ds_mask]):
                    arcpy.AddError('Input datasets do not intersect.')
                    return

                # resample mask dataset to match likelihood 
                ds_mask = tools.resample_xr(ds_from=ds_mask, 
                                            ds_to=ds_cva,
                                            resampling='nearest')

                # squeeze
                ds_mask = ds_mask.squeeze(drop=True)

            except Exception as e:
                arcpy.AddError('Could not intersect input datasets.')
                arcpy.AddMessage(str(e))
                return
                    
            # we made it, so mask cva
            ds_cva = ds_cva.where(~ds_mask['thresh'].isnull())

            # check if variables are empty
            if ds_cva.to_array().isnull().all():
                arcpy.AddError('Masked CVA dataset is empty. Check range of years and magnitude threshold.')            
                return   
                
                
                
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(15)

        # append attrbutes on to dataset and bands
        ds_cva.attrs = ds_attrs
        ds_cva['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_cva:
            ds_cva[var].attrs = ds_band_attrs
            
            
            
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(16)   

        try:
            # export netcdf file
            tools.export_xr_as_nc(ds=ds_cva, filename=out_nc)
        except Exception as e:
            arcpy.AddError('Could not export dataset.')
            arcpy.AddMessage(str(e))
            return
        
        
        

        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Adding output to map, if requested...')
        arcpy.SetProgressorPosition(17)

        # if requested...
        if in_add_result_to_map:
            
            try:
                # for current project, open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap
                
                # remove cva layer if already exists 
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'cva.crf':
                        m.removeLayer(layer)

                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'cva' + '_' + dt)
                os.makedirs(out_folder)                

                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False
                
                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'cva.crf')
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
                layer = m.listLayers('cva.crf')[0]
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
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(18)
        
        # close satellite dataset
        ds.close()
        del ds
        
        # close cva dataset
        ds_cva.close()
        del ds_cva
        
        # close mask dataset, if exists 
        if in_mask_nc is not None:
            ds_mask.close()
            del ds_mask

        # notify user
        arcpy.AddMessage('Generated CVA successfully.')  
                
        return


class Phenolopy_Metrics(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'Phenolopy Metrics'
        self.description = 'Calculate various metrics that describe various. ' \
                           'aspects of vegetation phenology from a data cube. ' \
                           'Key metrics include Peak of Season (POS), Start and ' \
                           'End of Season (SOS, EOS), and various productivity metrics.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # input netcdf file
        par_raw_nc_path = arcpy.Parameter(
                            displayName='Input satellite NetCDF file',
                            name='in_nc',
                            datatype='DEFile',
                            parameterType='Required',
                            direction='Input')
        par_raw_nc_path.filter.list = ['nc']
        
        # output netcdf file
        par_out_nc_path = arcpy.Parameter(
                            displayName='Output Phenometrics NetCDF file',
                            name='out_nc',
                            datatype='DEFile',
                            parameterType='Required',
                            direction='Output')
        par_out_nc_path.filter.list = ['nc']
        
        # use all dates
        par_use_all_dates = arcpy.Parameter(
                              displayName='Combine all input dates',
                              name='in_use_all_dates',
                              datatype='GPBoolean',
                              parameterType='Required',
                              direction='Input',
                              multiValue=False)
        par_use_all_dates.value = True
        
        # set specific year
        par_specific_years = arcpy.Parameter(
                              displayName='Specific year(s) to analyse',
                              name='in_specific_years',
                              datatype='GPLong',
                              parameterType='Optional',
                              direction='Input',
                              multiValue=True)
        par_specific_years.filter.type = 'ValueList'
        par_specific_years.filter.list = []
        par_specific_years.value = None
        
        # input metrics
        par_metrics = arcpy.Parameter(
                        displayName='Phenological metrics',
                        name='in_metrics',
                        datatype='GPString',
                        parameterType='Required',
                        direction='Input',
                        multiValue=True)
        metrics = [
            'POS: Peak of season',
            'VOS: Valley of season',
            'BSE: Base of season',
            'MOS: Middle of season',
            'AOS: Amplitude of season',
            'SOS: Start of season',
            'EOS: End of season',
            'LOS: Length of season',
            'ROI: Rate of increase',
            'ROD: Rate of decrease',
            'SIOS: Short integral of season',
            'LIOS: Long integral of season',
            'SIOT: Short integral of total',
            'LIOT: Long integral of total',
            'NOS: Number of seasons'
            ]
        par_metrics.filter.type = 'ValueList'        
        par_metrics.filter.list = metrics
        remove = [
            'MOS: Middle of season', 
            'BSE: Base of season', 
            'AOS: Amplitude of season',
            'NOS: Number of seasons'
            ]
        par_metrics.values = [m for m in metrics if m not in remove]

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
            'kNDVI'
            ]
        par_veg_idx.value = 'MAVI'  

        # input method type
        par_method_type = arcpy.Parameter(
                            displayName='Season detection method',
                            name='in_method_type',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            category='Season Detection',
                            multiValue=False)
        par_method_type.filter.list = [
            'First of slope',
            'Mean of slope',
            'Seasonal amplitude',
            'Absolute amplitude',
            'Relative amplitude'
            ]
        par_method_type.values = 'Seasonal amplitude'
        
        # input amplitude factor (seaamp, relamp)
        par_amp_factor = arcpy.Parameter(
                           displayName='Amplitude factor',
                           name='in_amp_factor',
                           datatype='GPDouble',
                           parameterType='Optional',
                           direction='Input',
                           category='Season Detection',
                           multiValue=False)
        par_amp_factor.filter.type = 'Range'
        par_amp_factor.filter.list = [0.0, 1.0]
        par_amp_factor.value = 0.5
        
        # input absolute value (absamp)
        par_abs_value = arcpy.Parameter(
                          displayName='Absolute value',
                          name='in_abs_value',
                          datatype='GPDouble',
                          parameterType='Optional',
                          direction='Input',
                          category='Season Detection',
                          multiValue=False)
        par_abs_value.value = 0.3
        
        # input savitsky window length
        par_sav_win_length = arcpy.Parameter(
                               displayName='Window size',
                               name='in_sav_win_length',
                               datatype='GPLong',
                               parameterType='Required',
                               direction='Input',
                               category='Smoothing',
                               multiValue=False)
        par_sav_win_length.filter.type = 'Range'
        par_sav_win_length.filter.list = [3, 99]
        par_sav_win_length.value = 3
        
        # input polyorder
        par_sav_polyorder = arcpy.Parameter(
                          displayName='Polyorder',
                          name='in_sav_polyorder',
                          datatype='GPLong',
                          parameterType='Required',
                          direction='Input',
                          category='Smoothing',
                          multiValue=False)
        par_sav_polyorder.filter.type = 'Range'
        par_sav_polyorder.filter.list = [1, 100]
        par_sav_polyorder.value = 1       

        # input outlier window length
        par_out_win_length = arcpy.Parameter(
                               displayName='Window size',
                               name='in_out_win_length',
                               datatype='GPLong',
                               parameterType='Required',
                               direction='Input',
                               category='Outlier Correction',
                               multiValue=False)
        par_out_win_length.filter.type = 'Range'
        par_out_win_length.filter.list = [3, 99]
        par_out_win_length.value = 3
        
        # input outlier factor
        par_out_factor = arcpy.Parameter(
                               displayName='Outlier removal factor',
                               name='in_out_factor',
                               datatype='GPLong',
                               parameterType='Required',
                               direction='Input',
                               category='Outlier Correction',
                               multiValue=False)
        par_out_factor.filter.type = 'Range'
        par_out_factor.filter.list = [1, 100]
        par_out_factor.value = 2                  

        # fix edge dates
        par_fix_edges = arcpy.Parameter(
                          displayName='Ignore edge dates',
                          name='in_fix_edges',
                          datatype='GPBoolean',
                          parameterType='Required',
                          direction='Input',
                          category='Outlier Correction',
                          multiValue=False)
        par_fix_edges.value = True 

        # fill empty pixels
        par_fill_nans = arcpy.Parameter(
                          displayName='Fill erroroneous pixels',
                          name='in_fill_nans',
                          datatype='GPBoolean',
                          parameterType='Required',
                          direction='Input',
                          category='Outlier Correction',
                          multiValue=False)
        par_fill_nans.value = True

        # input oa fmask 
        par_fmask_flags = arcpy.Parameter(
                            displayName='Include flags',
                            name='in_fmask_flags',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            category='Satellite Quality Options',
                            multiValue=True)
        flags = ['NoData', 'Valid', 'Cloud', 'Shadow', 'Snow', 'Water']
        par_fmask_flags.filter.type = 'ValueList'      
        par_fmask_flags.filter.list = flags
        par_fmask_flags.values = ['Valid', 'Snow', 'Water']
        
        # input max cloud cover
        par_max_cloud = arcpy.Parameter(
                          displayName='Maximum cloud cover',
                          name='in_max_cloud',
                          datatype='GPDouble',
                          parameterType='Optional',
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
            par_raw_nc_path,
            par_out_nc_path,
            par_use_all_dates,
            par_specific_years,
            par_metrics,
            par_veg_idx,
            par_method_type,
            par_amp_factor,
            par_abs_value,
            par_sav_win_length,
            par_sav_polyorder,
            par_out_win_length,
            par_out_factor,
            par_fix_edges,
            par_fill_nans,
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
        
        # imports
        try:
            import numpy as np
            import xarray as xr           
        except:
            arcpy.AddError('Python libraries xarray not installed.')
            return
        
        # globals 
        global PHENOLOPY_METRICS
        
        # unpack global parameter values 
        curr_file = PHENOLOPY_METRICS.get('in_file')
        
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

                # populate years list
                parameters[3].filter.list = dts

                # select last year
                if len(dts) != 0:
                    parameters[3].value = dts[-1]
                else:
                    parameters[3].value = None      
        
        # update global values
        PHENOLOPY_METRICS = {'in_file': parameters[0].valueAsText}
        
        # enable year selector based on combine input checkbox
        if parameters[2].value is False:
            parameters[3].enabled = True
        else:
            parameters[3].enabled = False

        # enable amp factor or abs value when methods selected
        if parameters[6].valueAsText in ['Seasonal amplitude', 'Relative amplitude']:
            parameters[7].enabled = True
            parameters[8].enabled = False
        elif parameters[6].valueAsText == 'Absolute amplitude':
            parameters[8].enabled = True
            parameters[7].enabled = False
        else: 
            parameters[7].enabled = False
            parameters[8].enabled = False
        

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the Phenolopy Metrics module.
        """
        
        # safe imports
        import os, sys      
        import datetime     
        import numpy as np  
        import tempfile     
        import arcpy        

        # risk imports (non-native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools  
            
            # module folder
            sys.path.append(FOLDER_MODULES)
            import phenolopy, cog   
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)
        
        # grab parameter values 
        in_nc = parameters[0].valueAsText                 # raw input satellite netcdf
        out_nc = parameters[1].valueAsText                # output phenometrics netcdf
        in_use_all_dates = parameters[2].value            # use all dates in nc 
        in_specific_years = parameters[3].valueAsText     # set specific year 
        in_metrics = parameters[4].valueAsText            # phenometrics
        in_veg_idx = parameters[5].value                  # vege index name
        in_method_type = parameters[6].value              # phenolopy method type
        in_amp_factor = parameters[7].value               # amplitude factor
        in_abs_value = parameters[8].value                # absolute value
        in_sav_window_length = parameters[9].value        # savitsky window length 
        in_sav_polyorder = parameters[10].value           # savitsky polyorder 
        in_out_window_length = parameters[11].value       # outlier window length 
        in_out_factor = parameters[12].value              # outlier cutoff user factor
        in_fix_edges = parameters[13].value               # fix edge dates
        in_fill_nans = parameters[14].value               # fill nans
        in_fmask_flags = parameters[15].valueAsText       # fmask flag values
        in_max_cloud = parameters[16].value               # max cloud percentage
        in_add_result_to_map = parameters[17].value       # add result to map



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Phenolopy Metrics.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...', 
                            min_range=0, max_range=23)



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Loading and checking netcdf...')
        arcpy.SetProgressorPosition(1)
        
        try:
            # do quick lazy load of netcdf for checking
            ds = xr.open_dataset(in_nc)
        except Exception as e:
            arcpy.AddError('Could not quick load input NetCDF data.')
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
            arcpy.AddError('Could not properly load input NetCDF data.')
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
        arcpy.SetProgressorLabel('Calculating vegetation index...')
        arcpy.SetProgressorPosition(6) 

        # check if veg idx supported 
        if in_veg_idx.lower() not in ['ndvi', 'evi', 'savi', 'msavi', 'slavi', 'mavi', 'kndvi']:
            arcpy.AddError('Vegetation index not supported.')
            return 

        try:
            # calculate vegetation index
            ds = tools.calculate_indices(ds=ds, 
                                         index=in_veg_idx.lower(), 
                                         custom_name='veg_idx', 
                                         rescale=False, 
                                         drop=True)

            # add band attrs back on
            ds['veg_idx'].attrs = ds_band_attrs   

        except Exception as e: 
            arcpy.AddError('Could not calculate vegetation index.')
            arcpy.AddMessage(str(e))
            return
            
        # check if we sufficient data temporally
        if len(ds['time']) == 0:
            arcpy.AddError('Insufficient number of dates in data.')
            return
        elif len(ds['time'].groupby('time.season')) < 3:
            arcpy.AddError('Insufficient number of seasons in data.')
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Correcting edge dates...')
        arcpy.SetProgressorPosition(7)
        
        # check if user years is valid 
        if in_use_all_dates is None:
            arcpy.AddError('Did not specify combine dates parameter.')
            return
        elif in_use_all_dates is False and in_specific_years is None:
            arcpy.AddError('Did not provide a specific year(s).')
            return
            
        # get list of years, else empty list
        if in_use_all_dates is False:
            in_specific_years = [int(e) for e in in_specific_years.split(';')]
        else:
            in_specific_years = None

        # check specific years for issues, if specific years exist 
        if in_specific_years is not None:
            if datetime.datetime.now().year == max(in_specific_years):
                arcpy.AddError('Cannot use current year, insufficient data.')
                return
            elif 2011 in in_specific_years or 2012 in in_specific_years:
                arcpy.AddError('Cannot use years 2011 or 2012, insufficient data.')
                return
           
            # check if requested years in dataset 
            for year in in_specific_years:
                if year not in ds['time.year']:
                    arcpy.AddError('Year {} was not found in dataset.'.format(year))
                    return

        try:
            # enforce first/last date is 1jan/31dec for user years (or all)
            ds = phenolopy.enforce_edge_dates(ds=ds, 
                                              years=in_specific_years)
        except Exception as e: 
            arcpy.AddError('Could not correct edge dates.')
            arcpy.AddMessage(str(e))
            return
            
            
            
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Subsetting time-series with buffer dates...')
        arcpy.SetProgressorPosition(8)

        try:
            # subset to requested years (or all) with buffer dates (no subset if no years)
            ds = phenolopy.subset_via_years_with_buffers(ds=ds, 
                                                         years=in_specific_years)
        except Exception as e: 
            arcpy.AddError('Could not subset data with buffer dates.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Resampling time-series to equal-spacing...')
        arcpy.SetProgressorPosition(9)
        
        try:
            # resample to fortnight medians to ensure equal-spacing
            ds = phenolopy.resample(ds=ds, 
                                    interval='SMS')        
        except Exception as e: 
            arcpy.AddError('Could not resample to equal-spacing.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating initial resample...')
        arcpy.SetProgressorPosition(10)
        
        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e: 
            arcpy.AddError('Could not interpolate initial resample.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Removing spike outliers...')
        arcpy.SetProgressorPosition(11)    

        # check window length and factor is valid 
        if in_out_window_length < 3 or in_out_window_length > 99:
            arcpy.AddError('Outlier window size must be between 3 and 99.')
            return
        elif in_out_factor < 1 or in_out_factor > 99:
            arcpy.AddError('Outlier factor must be between 1 and 99.')
            return

        try:
            #  remove outliers from data using requeted method
            ds = phenolopy.remove_spikes(ds=ds, 
                                         user_factor=in_out_factor, 
                                         win_size=in_out_window_length)
        except Exception as e: 
            arcpy.AddError('Could not remove spike outliers.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating removed outliers...')
        arcpy.SetProgressorPosition(12)
        
        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e: 
            arcpy.AddError('Could not interpolate removed outliers.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Smoothing time-series...')
        arcpy.SetProgressorPosition(13)

        # check window length 
        if in_sav_window_length <= 0 or in_sav_window_length % 2 == 0:
            arcpy.AddWarning('Savitsky window length incompatible, using default.')
            in_sav_window_length = 3
                
        # check polyorder 
        if in_sav_polyorder >= in_sav_window_length:
            arcpy.AddWarning('Savitsky polyorder must be < window length, reducing by one.')
            in_sav_polyorder = in_sav_window_length - 1

        try:
            # smooth dataset across time via savitsky
            ds = phenolopy.smooth(ds=ds, 
                                  var='veg_idx', 
                                  window_length=in_sav_window_length, 
                                  polyorder=in_sav_polyorder)
        except Exception as e: 
            arcpy.AddError('Could not smooth data.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Subsetting data to specific years, if requested...')
        arcpy.SetProgressorPosition(14)

        try:
            # subset to requested years, if none, returns input dataset
            ds = phenolopy.subset_via_years(ds=ds, 
                                            years=in_specific_years)
        except Exception as e: 
            arcpy.AddError('Could not subset to specific year(s).')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Grouping time-series by dates...')
        arcpy.SetProgressorPosition(15)
        
        try:
            # group years by m-d (1-1, 1-15, 2-1, 2-15, etc)
            ds = phenolopy.group(ds=ds)
        except Exception as e: 
            arcpy.AddError('Could not group by dates.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Resampling high resolution curves...')
        arcpy.SetProgressorPosition(16)
        
        try:
            # resample to 365 days per pixel for higher accuracy metrics
            ds = phenolopy.resample(ds=ds, 
                                    interval='1D')
        except Exception as e: 
            arcpy.AddError('Could not interpolate high-resolution curves.')
            arcpy.AddMessage(str(e))
            return        



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Interpolating high-resolution curves...')
        arcpy.SetProgressorPosition(17)
        
        try:
            # interpolate our initial resample gaps
            ds = phenolopy.interpolate(ds=ds)
        except Exception as e: 
            arcpy.AddError('Could not interpolate high-resolution curves.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Checking and removing date oversamples...')
        arcpy.SetProgressorPosition(18)
        
        try:
            # remove potential oversample dates
            ds = phenolopy.drop_overshoot_dates(ds=ds, 
                                                min_dates=3)
        except Exception as e: 
            arcpy.AddError('Could not remove oversample dates.')
            arcpy.AddMessage(str(e))
            return

        # check if we have 365 days, otherwise warn 
        if len(ds['time']) != 365:
            arcpy.AddWarning('Could not create 365-day time-series, output may have errors.')



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing data into memory, please wait...')
        arcpy.SetProgressorPosition(19)

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
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Calculating phenology metrics...')

        # check if metrics valid 
        if in_metrics is None:
            arcpy.AddError('No metrics were selected.')
            return
            
        # remove single quotes in metric string (due to spaces) and split 
        in_metrics = in_metrics.lower().replace("'", '').split(';')
        in_metrics = [e.split(':')[0] for e in in_metrics]
        
        # convert method to compatible name
        in_method_type = in_method_type.lower()
        in_method_type = in_method_type.replace(' ', '_')

        # check amplitude factors, absolute values 
        if in_method_type in ['seasonal_amplitude', 'relative_amplitude']:
            if in_amp_factor is None:
                arcpy.AddError('Must provide an amplitude factor.')
                return
            elif in_amp_factor < 0 or in_amp_factor > 1:
                arcpy.AddError('Amplitude factor must be between 0 and 1.')
                return
                
        elif in_method_type == 'absolute_amplitude':
            if in_abs_value is None:
                arcpy.AddError('Must provide an absolute value (any value).')
                return

        # check if fix edge dates and fill pixels set 
        if in_fix_edges is None:
            arcpy.AddError('Must set the ignore edge dates checkbox.')
            return
        elif in_fill_nans is None:
            arcpy.AddError('Must set the fill erroroneous pixels checkbox.')
            return

        try:
            # calculate phenometrics!
            ds = phenolopy.get_phenometrics(ds=ds, 
                                            metrics=in_metrics,
                                            method=in_method_type,
                                            factor=in_amp_factor, 
                                            abs_value=in_abs_value,
                                            peak_spacing=12,
                                            fix_edges=in_fix_edges,
                                            fill_nan=in_fill_nans)
        except Exception as e: 
            arcpy.AddError('Could not calculate phenometrics.')
            arcpy.AddMessage(str(e))
            return

        # check if any data was returned
        if ds.to_array().isnull().all():
            arcpy.AddError('Metric output contains no values.')
            return       



        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to dataset...')
        arcpy.SetProgressorPosition(20)
        
        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs
            
            
            
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(21)   

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
        arcpy.SetProgressorPosition(22)
        
        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap

                # remove existing ensemble layers if exist
                for layer in m.listLayers():
                    if layer.isGroupLayer and layer.supports('NAME') and layer.name == 'metrics':
                        m.removeLayer(layer)

                # setup a group layer via template
                grp_lyr = arcpy.mp.LayerFile(GRP_LYR_FILE)
                grp = m.addLayer(grp_lyr)[0]
                grp.name = 'metrics'
        
                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'metrics' + '_' + dt)
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
                    #layer.visible = False
                    
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
                            if 'roi' in tif or 'rod' in tif:
                                sym.colorizer.minPercent = 1.0
                                sym.colorizer.maxPercent = 1.0
                                cmap = aprx.listColorRamps('Inferno')[0]                                                           
                            elif 'aos' in tif or 'los' in tif:
                                sym.colorizer.minPercent = 0.5
                                sym.colorizer.maxPercent = 0.5
                                cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]  
                            elif 'nos' in tif:
                                sym.colorizer.stretchType = 'MinimumMaximum'
                                cmap = aprx.listColorRamps('Spectrum By Wavelength-Full Bright')[0]  
                            elif 'times' in tif:
                                sym.colorizer.minPercent = 0.25
                                sym.colorizer.maxPercent = 0.25
                                cmap = aprx.listColorRamps('Temperature')[0]
                            elif 'values' in tif:
                                sym.colorizer.minPercent = 1.0
                                sym.colorizer.maxPercent = 1.0
                                cmap = aprx.listColorRamps('Precipitation')[0]                                

                            # apply color map
                            sym.colorizer.colorRamp = cmap

                            # apply other basic options
                            sym.colorizer.invertColorRamp = False
                            sym.colorizer.gamma = 1.0

                            # update symbology
                            layer.symbology = sym
                            
                            # show layer 
                            #layer.visible = True
                            
            except Exception as e:
                arcpy.AddWarning('Could not colorise output, aborting colorisation.')
                arcpy.AddMessage(str(e))
                pass



        # # # # #
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(23)
        
        # close and del dataset
        ds.close()
        del ds

        # notify user
        arcpy.AddMessage('Generated Phenometrics successfully.')

        return


class Nicher_SDM(object):
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
        
        # safe imports
        import os, sys      
        import datetime     
        import numpy as np  
        import pandas as pd 
        import arcpy        

        # risk imports (non-native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools  
            
            # module folder
            sys.path.append(FOLDER_MODULES)
            import nicher  
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_cont_tifs = parameters[0].valueAsText          # continous tifs
        in_cate_tifs = parameters[1].valueAsText          # categorical tifs
        out_nc = parameters[2].valueAsText                # output netcdf
        in_occurrence_feat = parameters[3]                # occurrence point shapefile   
        in_num_absence = parameters[4].value              # num absences 
        in_exclusion_buffer = parameters[5].value         # exclusion buffer
        in_equalise_absence = parameters[6].value         # equalise absences
        in_test_ratio = parameters[7].value               # test ratio
        in_resample = parameters[8].value                 # resample
        in_num_estimator = parameters[9].value            # number of estimators
        in_criterion = parameters[10].value               # criterion type
        in_max_depth = parameters[11].value               # max tree depth
        in_max_features = parameters[12].value            # maximum features
        in_bootstrap = parameters[13].value               # boostrap
        in_add_result_to_map = parameters[14].value       # add result to map



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


class Nicher_Masker(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'Nicher Masker'
        self.description = 'Use an existing NetCDF or GeoTiff layer to mask ' \
                           'out areas from previously generated Niche modelling ' \
                           'outputs. Useful for removing infrastructure.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input nicher netcdf
        par_in_nicher_nc = arcpy.Parameter(
                               displayName='Input Nicher NetCDF file',
                               name='in_nicher_nc',
                               datatype='DEFile',
                               parameterType='Required',
                               direction='Input')
        par_in_nicher_nc.filter.list = ['nc']
        
        # output netcdf file
        par_out_nc = arcpy.Parameter(
                       displayName='Output masked Nicher NetCDF file',
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
            par_in_nicher_nc, 
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
        
        # imports
        try:
            import numpy as np
            import xarray as xr
            import rasterio
        except:
            arcpy.AddError('Python libraries xarray, rasterio not installed.')
            return
            
        # globals
        global NICHER_MASKER

        # unpack global parameter values 
        curr_file = NICHER_MASKER.get('in_file')
        curr_var = NICHER_MASKER.get('in_var')

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
        NICHER_MASKER = {
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
        Executes the Nicher Masker module.
        """

        # safe imports
        import os
        import datetime
        import numpy as np
        import tempfile

        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_nicher_nc = parameters[0].valueAsText        # nicher netcdf
        out_nc = parameters[1].valueAsText              # output netcdf
        in_mask_file = parameters[2].valueAsText        # mask nc or tif
        in_var = parameters[3].value                    # variable
        in_type = parameters[4].value                   # mask type
        in_bin = parameters[5].value                    # binary value
        in_range_min = parameters[6].value              # range minimum
        in_range_max = parameters[7].value              # range maximum
        in_replace = parameters[8].value                # replacement value
        in_add_result_to_map = parameters[9].value      # add result to map



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning Nicher Masker.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=12)



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking nicher NetCDF...')
        arcpy.SetProgressorPosition(1)

        try:
            # do quick lazy load of nicher netcdf for checking
            ds_sdm = xr.open_dataset(in_nicher_nc)
        except Exception as e:
            arcpy.AddWarning('Could not quick load nicher NetCDF data, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # check xr type, vars, coords, dims, attrs
        if not isinstance(ds_sdm, xr.Dataset):
            arcpy.AddError('Input nicher NetCDF must be a xr dataset.')
            return
        elif len(ds_sdm) == 0:
            arcpy.AddError('Input nicher NetCDF has no data/variables/bands.')
            return
        elif 'time' in ds_sdm.dims:
            arcpy.AddError('Input nicher NetCDF must not have time dimension.')
            return
        elif 'x' not in ds_sdm.dims or 'y' not in ds_sdm.dims:
            arcpy.AddError('Input nicher NetCDF must have x, y dimensions.')
            return
        elif 'time' in ds_sdm.coords:
            arcpy.AddError('Input nicher NetCDF must not have time coord.')
            return
        elif 'x' not in ds_sdm.coords or 'y' not in ds_sdm.coords:
            arcpy.AddError('Input nicher NetCDF must have x, y coords.')
            return
        elif 'spatial_ref' not in ds_sdm.coords:
            arcpy.AddError('Input nicher NetCDF must have a spatial_ref coord.')
            return
        elif len(ds_sdm['x']) == 0 or len(ds_sdm['y']) == 0:
            arcpy.AddError('Input nicher NetCDF must have all at least one x, y index.')
            return
        elif ds_sdm.attrs == {}:
            arcpy.AddError('Nicher NetCDF must have attributes.')
            return
        elif not hasattr(ds_sdm, 'crs'):
            arcpy.AddError('Nicher NetCDF CRS attribute not found. CRS required.')
            return
        elif ds_sdm.crs != 'EPSG:3577':
            arcpy.AddError('Nicher NetCDF CRS is not in GDA94 Albers (EPSG:3577).')            
            return 
        elif not hasattr(ds_sdm, 'nodatavals'):
            arcpy.AddError('Nicher NetCDF nodatavals attribute not found.')            
            return 
        elif 'result' not in ds_sdm:
            arcpy.AddError('Nicher NetCDF is missing result variable.')
            return

        # check if all nan
        if ds_sdm.to_array().isnull().all():
            arcpy.AddError('Nicher NetCDF is empty.')            
            return 

        try:
            # now open of nicher netcdf properly (and set nodata to nan)
            ds_sdm = satfetcher.load_local_nc(nc_path=in_nicher_nc, 
                                              use_dask=False, 
                                              conform_nodata_to=np.nan)
        except Exception as e:
            arcpy.AddError('Could not properly load nicher NetCDF data, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Getting nicher NetCDF attributes...')
        arcpy.SetProgressorPosition(2)

        # get attributes from dataset
        ds_attrs = ds_sdm.attrs
        ds_band_attrs = ds_sdm[list(ds_sdm)[0]].attrs
        ds_spatial_ref_attrs = ds_sdm['spatial_ref'].attrs



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Computing nicher NetCDF into memory, please wait...')
        arcpy.SetProgressorPosition(3)

        try:
            # compute! 
            ds_sdm = ds_sdm.compute()
        except Exception as e: 
            arcpy.AddError('Could not compute nicher NetCDF, see messages for details.')
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
        if not tools.all_xr_intersect([ds_sdm, ds_mask]):
            arcpy.AddError('Not all input layers intersect.')            
            return 

        try:        
            # resample mask to nicher dataset, if same, no change
            ds_mask = tools.resample_xr(ds_from=ds_mask, 
                                        ds_to=ds_sdm,
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
        arcpy.SetProgressorLabel('Masking nicher dataset via mask...')
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
        ds_sdm = ds_sdm.where(ds_mask.to_array(), in_replace)
        ds_sdm = ds_sdm.squeeze(drop=True)
        
        # check if any values exist in nicher dataset
        if ds_sdm.to_array().isnull().all():
            arcpy.AddError('Nicher dataset has no values after mask.')
            return
            
            
            
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Appending attributes back on to nicher dataset...')
        arcpy.SetProgressorPosition(9)

        # append attrbutes on to dataset and bands
        ds_sdm.attrs = ds_attrs
        ds_sdm['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds_sdm:
            ds_sdm[var].attrs = ds_band_attrs


            
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(10)   

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
        arcpy.SetProgressorPosition(11)

        # if requested...
        if in_add_result_to_map:
            try:
                # open current map
                aprx = arcpy.mp.ArcGISProject('CURRENT')
                m = aprx.activeMap
                
                # remove existing sdm layer if exist
                for layer in m.listLayers():
                    if layer.supports('NAME') and layer.name == 'sdm_masked.crf':
                        m.removeLayer(layer)
                        
                # create output folder using datetime as name
                dt = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                out_folder = os.path.join(os.path.dirname(out_nc), 'sdm_masked' + '_' + dt)
                os.makedirs(out_folder)
                
                # disable visualise on map temporarily
                arcpy.env.addOutputsToMap = False
                
                # create crf filename and copy it
                out_file = os.path.join(out_folder, 'sdm_masked.crf')
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
                layer = m.listLayers('sdm_masked.crf')[0]
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
        arcpy.SetProgressorPosition(12)

        # close ensemble dataset 
        ds_sdm.close()
        del ds_sdm

        # close mask dataset
        ds_mask.close()
        del ds_mask

        # notify user
        arcpy.AddMessage('Performed Nicher Masking successfully.')

        return


class VegFrax_Fractional_Cover(object):
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
        
        # imports
        try:
            import numpy as np
            import xarray as xr           
        except:
            arcpy.AddError('Python libraries xarray not installed.')
            return
        
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
        
        # safe imports
        import os, sys                         
        import datetime                        
        import numpy as np                     
        import pandas as pd                    
        import tempfile                        

        # risk imports (non-native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools  
            
            # module folder
            sys.path.append(FOLDER_MODULES)
            import vegfrax, cog  
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_low_nc = parameters[0].valueAsText             # raw input low res netcdf
        in_high_tif = parameters[1].valueAsText           # raw input high res tif
        out_nc = parameters[2].valueAsText                # output vegfrax netcdf
        in_agg_dates = parameters[3].value                # aggregate all dates
        in_start_date = parameters[4].value               # start date of aggregate
        in_end_date = parameters[5].value                 # end date of aggregate
        in_classes = parameters[6].valueAsText            # selected classes
        in_agg_classes = parameters[7].value              # aggregate selected classes       
        in_num_samples = parameters[8].value              # number of samples
        in_max_nodata = parameters[9].value               # max nodata frequency
        in_smooth = parameters[10].value                  # smooth output
        in_num_estimator = parameters[11].value           # number of model estimators
        in_criterion = parameters[12].value               # criterion type
        in_max_depth = parameters[13].value               # max tree depth
        in_max_features = parameters[14].value            # maximum features
        in_bootstrap = parameters[15].value               # boostrap
        in_fmask_flags = parameters[16].valueAsText       # fmask flag values
        in_max_cloud = parameters[17].value               # max cloud percentage
        in_add_result_to_map = parameters[18].value       # add result to map



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
        elif tools.get_xr_crs(ds_high) != 3577:
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
                    #layer.visible = False

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
                            #layer.visible = True

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
        
        # imports
        try:
            import numpy as np
            import xarray as xr
            import rasterio
        except:
            arcpy.AddError('Python libraries xarray, rasterio not installed.')
            return

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
        
        # safe imports
        import os             # arcgis comes with these
        import datetime       # arcgis comes with these

        # risky imports (not native to arcgis)
        try:
            import numpy as np
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import satfetcher, tools

            # module folder
            sys.path.append(FOLDER_MODULES)
            import canopy
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_file = parameters[0].valueAsText          # input netcdf or geotiff
        out_nc = parameters[1].valueAsText           # output netcdf
        in_var = parameters[2].value                 # input variable
        in_type = parameters[3].value                # input membership type
        in_min = parameters[4].value                 # input minimum
        in_mid = parameters[5].value                 # input middle
        in_max = parameters[6].value                 # input maximum
        in_add_result_to_map = parameters[7].value   # input add result to map



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

        # safe imports
        import os          
        import datetime
        import numpy as np
        import tempfile    

        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import satfetcher, tools

            # module folder
            sys.path.append(FOLDER_MODULES)
            import ensemble
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_layers = parameters[0].value              # input layers (as a value array)
        out_nc = parameters[1].valueAsText           # output netcdf
        in_resample = parameters[2].value            # resample resolution
        in_smooth = parameters[3].value              # smooth inputs
        in_win_size = parameters[4].value            # smoothing window size
        in_add_result_to_map = parameters[5].value   # add result to map



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
                    #layer.visible = False

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
                            #layer.visible = True

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
        
        # imports
        try:
            import numpy as np
            import xarray as xr
            import rasterio
        except:
            arcpy.AddError('Python libraries xarray, rasterio not installed.')
            return
            
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

        # safe imports
        import os
        import datetime
        import numpy as np
        import tempfile

        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
        except Exception as e:
            arcpy.AddError('Python libraries xarray and dask not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_ensemble_nc = parameters[0].valueAsText      # ensemble netcdf
        out_nc = parameters[1].valueAsText              # output netcdf
        in_mask_file = parameters[2].valueAsText        # mask nc or tif
        in_var = parameters[3].value                    # variable
        in_type = parameters[4].value                   # mask type
        in_bin = parameters[5].value                    # binary value
        in_range_min = parameters[6].value              # range minimum
        in_range_max = parameters[7].value              # range maximum
        in_replace = parameters[8].value                # replacement value
        in_add_result_to_map = parameters[9].value      # add result to map



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
                    #layer.visible = False

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
                            #layer.visible = True

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


class NRT_Create_Project(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'NRT Create Project'
        self.description = 'Create a new project geodatabase to hold ' \
                           'monitoring areas.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # output gdb folder
        par_out_folder = arcpy.Parameter(
                           displayName='Output project folder',
                           name='out_folder',
                           datatype='DEFolder',
                           parameterType='Required',
                           direction='Input')
                           
        # combine parameters
        parameters = [par_out_folder]
        
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
        Executes the NRT Create Project module.
        """
        
        # safe imports
        import os
        import arcpy
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # grab parameter values 
        out_folder = parameters[0].valueAsText      # output gdb folder path



        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Create Project.')
        arcpy.SetProgressor(type='step', 
                            message='Preparing parameters...',
                            min_range=0, max_range=8)
                            
                            
                            
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Checking input parameters...')
        arcpy.SetProgressorPosition(1)
                            
        # check inputs are not none and are strings
        if out_folder is None:
            arcpy.AddError('No project folder provided.')
            return
        elif not isinstance(out_folder, str):
            arcpy.AddError('Project folder is not a string.')
            return

        # check if monitoring area gdb already exists 
        gdb_path = os.path.join(out_folder, 'monitoring_areas.gdb')
        if os.path.exists(gdb_path):
            arcpy.AddError('Project folder already exists, provide a different folder.')
            return
            


        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Creating new project geodatabase...')
        arcpy.SetProgressorPosition(2)

        try:
            # build project geodatbase
            out_filepath = arcpy.management.CreateFileGDB(out_folder_path=out_folder, 
                                                          out_name='monitoring_areas.gdb')
        except Exception as e:
            arcpy.AddError('Could not create file geodatabase.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database feature class...')
        arcpy.SetProgressorPosition(3)

        # temporarily disable auto-visual of outputs
        arcpy.env.addOutputsToMap = False

        try:
            # create feature class and aus albers spatial ref sys
            srs = arcpy.SpatialReference(3577)
            out_feat = arcpy.management.CreateFeatureclass(out_path=out_filepath, 
                                                           out_name='monitoring_areas', 
                                                           geometry_type='POLYGON',
                                                           spatial_reference=srs)
        except Exception as e:
            arcpy.AddError('Could not create featureclass.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database domains...')
        arcpy.SetProgressorPosition(4)

        try:
            # create platform domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_platforms', 
                                          domain_description='Platform name (Landsat or Sentinel)',
                                          field_type='TEXT', 
                                          domain_type='CODED')
                                          
            # generate coded values to platform domain
            dom_values = {'Landsat': 'Landsat', 'Sentinel': 'Sentinel'}
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_platforms', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))

            # create start year domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_start_years', 
                                          domain_description='Training years (1980 - 2050)',
                                          field_type='LONG', 
                                          domain_type='RANGE')

            # generate range values to year domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_start_years', 
                                                    min_value=1980, 
                                                    max_value=2050)
                                                    
            # create train length domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_train_length', 
                                          domain_description='Training length (5 - 99999)',
                                          field_type='LONG', 
                                          domain_type='RANGE')

            # generate range values to train length domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_train_length', 
                                                    min_value=5, 
                                                    max_value=99999)

            # create index domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_indices', 
                                          domain_description='Vegetation index name',
                                          field_type='TEXT', 
                                          domain_type='CODED')

            # generate coded values to index domain
            dom_values = {'NDVI': 'NDVI', 'MAVI': 'MAVI', 'kNDVI': 'kNDVI'}
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_indices', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))

            # create persistence domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_persistence', 
                                          domain_description='Vegetation persistence (0.001 - 9.999)',
                                          field_type='FLOAT', 
                                          domain_type='RANGE')

            # generate range values to persistence domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_persistence', 
                                                    min_value=0.001, 
                                                    max_value=9.999)
                                                    
            # create rule 1 min consequtives domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_rule_1_consequtives', 
                                          domain_description='Rule 1 Consequtives (0 - 999)',
                                          field_type='LONG', 
                                          domain_type='RANGE')

            # generate range values to consequtives domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_rule_1_consequtives', 
                                                    min_value=0, 
                                                    max_value=999)
                                                    
            # create rule 2 min zone domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_rule_2_min_zone', 
                                          domain_description='Rule 2 Minimum Zone (1 - 11)',
                                          field_type='LONG', 
                                          domain_type='RANGE')

            # generate range values for min zone domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_rule_2_min_zone', 
                                                    min_value=1, 
                                                    max_value=11)

            # create rule 3 num zones domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_rule_3_num_zones', 
                                          domain_description='Rule 3 Number of Zones (1 - 11)',
                                          field_type='LONG', 
                                          domain_type='RANGE')

            # generate range values to num zones domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                                    domain_name='dom_rule_3_num_zones', 
                                                    min_value=1, 
                                                    max_value=11)

            # create ruleset domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_ruleset', 
                                          domain_description='Various rulesets',
                                          field_type='TEXT', 
                                          domain_type='CODED')

            # generate coded values to ruleset domain   
            dom_values = {
                '1 only': '1 only',
                '2 only': '2 only',
                '3 only': '3 only',
                '1 and 2': '1 and 2',
                '1 and 3': '1 and 3',
                '2 and 3': '2 and 3',
                '1 or 2': '1 or 2',
                '1 or 3': '1 or 3',
                '2 or 3': '2 or 3',
                '1 and 2 and 3': '1 and 2 and 3',
                '1 or 2 and 3': '1 or 2 and 3',
                '1 and 2 or 3': '1 and 2 or 3',
                '1 or 2 or 3': '1 or 2 or 3'
                }      
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_ruleset', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))                                            

            # create alert method domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_alert_method', 
                                          domain_description='Alert method',
                                          field_type='TEXT', 
                                          domain_type='CODED')

            # generate coded values to alert method domain 
            dom_values = {'Static': 'Static', 'Dynamic': 'Dynamic'}
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_alert_method', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))
        
            # create alert direction domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_alert_direction', 
                                          domain_description='Alert directions',
                                          field_type='TEXT', 
                                          domain_type='CODED')

            # generate coded values to boolean domain
            dom_values = {
                'Incline only (any)': 'Incline only (any)', 
                'Decline only (any)': 'Decline only (any)', 
                'Incline only (+ zones only)': 'Incline only (+ zones only)', 
                'Decline only (- zones only)': 'Decline only (- zones only)', 
                'Incline or Decline (any)': 'Incline or Decline (any)',
                'Incline or Decline (+/- zones only)': 'Incline or Decline (+/- zones only)'
                }
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_alert_direction', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))

            # create boolean domain
            arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                          domain_name='dom_boolean', 
                                          domain_description='Boolean (Yes or No)',
                                          field_type='TEXT', 
                                          domain_type='CODED')

            # generate coded values to boolean domain
            dom_values = {'Yes': 'Yes', 'No': 'No'}
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                                       domain_name='dom_boolean', 
                                                       code=dom_value, 
                                                       code_description=dom_values.get(dom_value))
        except Exception as e:
            arcpy.AddError('Could not create domains.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database fields...')
        arcpy.SetProgressorPosition(5)

        try:
            # add area id field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='area_id', 
                                      field_type='TEXT', 
                                      field_alias='Area ID',
                                      field_length=200,
                                      field_is_required='REQUIRED')

            # add platforms field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='platform', 
                                      field_type='TEXT', 
                                      field_alias='Platform',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_platforms')    

            # add s_year field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='s_year', 
                                      field_type='LONG', 
                                      field_alias='Start Year of Training Period',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_start_years')

            # add e_year field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='e_year', 
                                      field_type='LONG', 
                                      field_alias='Training Period Length',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_train_length')

            # add index field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='index', 
                                      field_type='TEXT', 
                                      field_alias='Vegetation Index',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_indices')

            # add persistence field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='persistence', 
                                      field_type='FLOAT', 
                                      field_alias='Vegetation Persistence',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_persistence')

            # add rule 1 min consequtives field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='rule_1_min_conseqs', 
                                      field_type='LONG', 
                                      field_alias='Rule 1 Minimum Consequtives',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_1_consequtives')

            # add include plateaus field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='rule_1_inc_plateaus', 
                                      field_type='TEXT', 
                                      field_alias='Rule 1 Include Pleateaus',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')

            # add rule 2 min stdv field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='rule_2_min_zone', 
                                      field_type='LONG', 
                                      field_alias='Rule 2 Minimum Zone',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_2_min_zone')

            # add rule 3 num zones field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='rule_3_num_zones', 
                                      field_type='LONG', 
                                      field_alias='Rule 3 Number of Zones',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_3_num_zones')                              

            # add ruleset field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='ruleset', 
                                      field_type='TEXT', 
                                      field_alias='Ruleset',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_ruleset')      

            # add alert field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='alert', 
                                      field_type='TEXT', 
                                      field_alias='Alert via Email',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')
   
            # add method field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='method', 
                                      field_type='TEXT', 
                                      field_alias='Alert via Method',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_alert_method')
   
            # add alert direction field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='alert_direction', 
                                      field_type='TEXT', 
                                      field_alias='Change Direction for Alert',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_alert_direction')

            # add email field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='email', 
                                      field_type='TEXT', 
                                      field_alias='Email of User',
                                      field_is_required='REQUIRED')

            # add ignore field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='ignore', 
                                      field_type='TEXT', 
                                      field_alias='Ignore When Run',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')   
                                  
            # add color border field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='color_border', 
                                      field_type='LONG', 
                                      field_alias='Color Code (Border)',
                                      field_is_required='REQUIRED')  
            
            # add color fill field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='color_fill', 
                                      field_type='LONG', 
                                      field_alias='Color Code (Fill)',
                                      field_is_required='REQUIRED')  

            # add global_id field to featureclass   
            arcpy.management.AddField(in_table=out_feat, 
                                      field_name='global_id', 
                                      field_type='TEXT', 
                                      field_alias='Global ID',
                                      field_length=200,
                                      field_is_required='REQUIRED')
        except Exception as e:
            arcpy.AddError('Could not create fields.')
            arcpy.AddMessage(str(e))
            return

        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database defaults...')
        arcpy.SetProgressorPosition(6)

        try:
            # set default platform
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='platform',
                                                  default_value='Landsat')   

            # set default index
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='index',
                                                  default_value='MAVI')  

            # set default persistence
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='persistence',
                                                  default_value=0.5)

            # set default min conseqs
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='rule_1_min_conseqs',
                                                  default_value=3)

            # set default inc plateaus
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='rule_1_inc_plateaus',
                                                  default_value='No')

            # set default min zone
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='rule_2_min_zone',
                                                  default_value=2)

            # set default num zones
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='rule_3_num_zones',
                                                  default_value=2)

            # set default ruleset
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='ruleset',
                                                  default_value='1 and 2 or 3')

            # set default alert
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='alert',
                                                  default_value='No')   
        
            # set default alert method
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='method',
                                                  default_value='Static')  
        
            # set default alert direction
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='alert_direction',
                                                  default_value='Decline only (any)')   

            # set default ignore
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='ignore',
                                                  default_value='No')  
                                              
            # set default border color
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='color_border',
                                                  default_value=0) 
                                                  
            # set default fill color
            arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                                  field_name='color_fill',
                                                  default_value=0) 
        except Exception as e:
            arcpy.AddError('Could not assign default values.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(7)

        try:           
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(out_feat)
        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass
            
        try:
            # apply symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)
        except Exception as e:
            arcpy.AddWarning('Could not colorise output, aborting colorisation.')
            arcpy.AddMessage(str(e))
            pass
            
        try:
            # show labels
            for layer in m.listLayers('monitoring_areas'):
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "'Zone: ' +  Text($feature.color_fill)"
                layer.showLabels = True
        except Exception as e:
            arcpy.AddWarning('Could not show labels, aborting label display.')
            arcpy.AddMessage(str(e))
            pass  
        
        
        
        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(8)

        # finish up
        arcpy.AddMessage('Created new NRT Project.')
        return


class NRT_Create_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'NRT Create Monitoring Areas'
        self.description = 'Create new monitoring area boundaries and set ' \
                           'monitoring options.'
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
        
        # set input feature class
        par_in_new_feat = arcpy.Parameter(
                            displayName='Draw new monitoring area',
                            name='in_new_feat',
                            datatype='GPFeatureRecordSetLayer',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_new_feat.filter.list = ['Polygon']
        par_in_new_feat.enabled = False
        
        # set monitoring area
        par_in_set_area = arcpy.Parameter(
                            displayName='Unique identifier',
                            name='in_area_id',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_set_area.enabled = False
        
        # platform
        par_in_platform = arcpy.Parameter(
                            displayName='Satellite platform',
                            name='in_platform',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_platform.filter.type = 'ValueList'
        par_in_platform.filter.list = ['Landsat', 'Sentinel']
        par_in_platform.value = 'Landsat'
        par_in_platform.enabled = False
        
        # start year
        par_in_s_year = arcpy.Parameter(
                          displayName='Pre-impact start year',
                          name='in_s_year',
                          datatype='GPLong',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_in_s_year.filter.type = 'Range'
        par_in_s_year.filter.list = [1980, 2050]
        par_in_s_year.enabled = False
        
        # training length (was end year)
        par_in_e_year = arcpy.Parameter(
                          displayName='Minimum number of training dates',
                          name='in_e_year',
                          datatype='GPLong',
                          parameterType='Optional',
                          direction='Input',
                          multiValue=False)
        par_in_e_year.filter.type = 'Range'
        par_in_e_year.filter.list = [5, 99999]
        par_in_e_year.enabled = False

        # vegetation index
        par_in_veg_idx = arcpy.Parameter(
                           displayName='Vegetation index',
                           name='in_veg_idx',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           multiValue=False)
        par_in_veg_idx.filter.type = 'ValueList'
        par_in_veg_idx.filter.list = ['NDVI', 'MAVI', 'kNDVI']
        par_in_veg_idx.value = 'MAVI'
        par_in_veg_idx.enabled = False
        
        # persistence
        par_in_persistence = arcpy.Parameter(
                               displayName='Vegetation persistence',
                               name='in_persistence',
                               datatype='GPDouble',
                               parameterType='Required',
                               direction='Input',
                               multiValue=False)
        par_in_persistence.filter.type = 'Range'
        par_in_persistence.filter.list = [0.001, 9.999]
        par_in_persistence.value = 0.5
        par_in_persistence.enabled = False
       
        # rule 1 min conseqs
        par_in_min_conseqs = arcpy.Parameter(
                               displayName='Rule 1: minimum consequtives',
                               name='in_min_conseqs',
                               datatype='GPLong',
                               parameterType='Required',
                               direction='Input',
                               category='Rules',
                               multiValue=False)
        par_in_min_conseqs.filter.type = 'Range'
        par_in_min_conseqs.filter.list = [0, 99]
        par_in_min_conseqs.value = 3
        par_in_min_conseqs.enabled = False
        
        # rule 1 include plateaus
        par_in_inc_plateaus = arcpy.Parameter(
                                displayName='Rule 1: include plateaus',
                                name='in_inc_plateaus',
                                datatype='GPString',
                                parameterType='Required',
                                direction='Input',
                                category='Rules',
                                multiValue=False)
        par_in_inc_plateaus.filter.type = 'ValueList'
        par_in_inc_plateaus.filter.list = ['Yes', 'No']
        par_in_inc_plateaus.value = 'No'
        par_in_inc_plateaus.enabled = False
        
        # rule 2 minimum zone
        par_in_min_stdv = arcpy.Parameter(
                            displayName='Rule 2: minimum zone',
                            name='in_min_zone',
                            datatype='GPLong',
                            parameterType='Required',
                            category='Rules',
                            direction='Input',
                            multiValue=False)
        par_in_min_stdv.filter.type = 'Range'
        par_in_min_stdv.filter.list = [1, 99]
        par_in_min_stdv.value = 2
        par_in_min_stdv.enabled = False  
        
        # rule 3 number of zones
        par_in_num_zones = arcpy.Parameter(
                             displayName='Rule 3: number of zones',
                             name='in_num_zones',
                             datatype='GPLong',
                             parameterType='Required',
                             category='Rules',
                             direction='Input',
                             multiValue=False)
        par_in_num_zones.filter.type = 'Range'
        par_in_num_zones.filter.list = [1, 99]
        par_in_num_zones.value = 2
        par_in_num_zones.enabled = False       
        
        # ruleset
        par_in_ruleset = arcpy.Parameter(
                           displayName='Ruleset',
                           name='in_ruleset',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           category='Rules',
                           multiValue=False)
        par_in_ruleset.filter.type = 'ValueList'
        ruleset = [
            '1 only',
            '2 only',
            '3 only',
            '1 and 2',
            '1 and 3',
            '2 and 3',
            '1 or 2',
            '1 or 3',
            '2 or 3',
            '1 and 2 and 3',
            '1 or 2 and 3',
            '1 and 2 or 3',
            '1 or 2 or 3'
            ] 
        par_in_ruleset.filter.list = ruleset
        par_in_ruleset.value = '1 and 2 or 3'
        par_in_ruleset.enabled = False       

        # alert user 
        par_in_alert_user = arcpy.Parameter(
                              displayName='Send alerts',
                              name='in_alert_user',
                              datatype='GPString',
                              parameterType='Required',
                              direction='Input',
                              category='Alerts and Email',
                              multiValue=False)
        par_in_alert_user.filter.type = 'ValueList'
        par_in_alert_user.filter.list = ['Yes', 'No']
        par_in_alert_user.value = 'No'
        par_in_alert_user.enabled = False     

        # alert method
        par_in_alert_method = arcpy.Parameter(
                           displayName='Alert based on method',
                           name='in_alert_method',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           category='Alerts and Email',
                           multiValue=False)
        par_in_alert_method.filter.type = 'ValueList'
        par_in_alert_method.filter.list = ['Static', 'Dynamic']
        par_in_alert_method.value = 'Static'
        par_in_alert_method.enabled = False        
        
        # alert direction 
        par_in_alert_direction = arcpy.Parameter(
                                   displayName='Direction of change to trigger alert',
                                   name='in_alert_direction',
                                   datatype='GPString',
                                   parameterType='Required',
                                   direction='Input',
                                   category='Alerts and Email',
                                   multiValue=False)
        directions = [
            'Incline only (any)', 
            'Decline only (any)', 
            'Incline only (+ zones only)', 
            'Decline only (- zones only)', 
            'Incline or Decline (any)',
            'Incline or Decline (+/- zones only)'
            ]
        par_in_alert_direction.filter.type = 'ValueList'
        par_in_alert_direction.filter.list = directions
        par_in_alert_direction.value = 'Decline only (any)'
        par_in_alert_direction.enabled = False    

        # email 
        par_in_email = arcpy.Parameter(
                         displayName='Email to recieve alerts',
                         name='in_email',
                         datatype='GPString',
                         parameterType='Optional',
                         direction='Input',
                         category='Alerts and Email',
                         multiValue=False)
        par_in_email.enabled = False   

        # ignore 
        par_in_ignore = arcpy.Parameter(
                          displayName='Ignore during monitoring',
                          name='in_ignore_user',
                          datatype='GPString',
                          parameterType='Required',
                          direction='Input',
                          category='Other',
                          multiValue=False)
        par_in_ignore.filter.type = 'ValueList'
        par_in_ignore.filter.list = ['Yes', 'No']
        par_in_ignore.value = 'No'
        par_in_ignore.enabled = False  

        # combine parameters
        parameters = [
            par_in_feat,
            par_in_new_feat,
            par_in_set_area,
            par_in_platform,
            par_in_s_year,
            par_in_e_year,
            par_in_veg_idx,
            par_in_persistence,
            par_in_min_conseqs,
            par_in_inc_plateaus,
            par_in_min_stdv,
            par_in_num_zones,
            par_in_ruleset,
            par_in_alert_user,
            par_in_alert_method,
            par_in_alert_direction,
            par_in_email,
            par_in_ignore
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
            'color_fill',
            'global_id'
            ] 

        # globals 
        global NRT_CREATE_AREA

        # unpack global parameter values 
        curr_feat = NRT_CREATE_AREA.get('in_feat')

        # check existing feature input
        if parameters[0].valueAsText != curr_feat:            
            try:
                # load column names
                in_feat = parameters[0].valueAsText                
                cols = [f.name for f in arcpy.ListFields(in_feat)]
            except:
                cols = []

            # if invalid fields, abort
            if all(f in cols for f in required_fields):
                for i in range(1, 18):
                    parameters[i].enabled = True
            else:
                for i in range(1, 18):
                    parameters[i].enabled = False

        # update global values
        NRT_CREATE_AREA = {'in_feat': parameters[0].valueAsText}

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Create Monitoring Areas tool.
        """
        
        # safe imports
        import os
        import uuid
        import arcpy
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import nrt
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
        
        # grab parameter values 
        in_exist_feat = parameters[0]                     # input monitoring areas feature
        in_new_feat = parameters[1]                       # input new monitoring areas feature
        in_area_id = parameters[2].value                  # input monitoring area id
        in_platform = parameters[3].value                 # input platform
        in_s_year = parameters[4].value                   # input start year
        in_e_year = parameters[5].value                   # input end year
        in_veg_idx = parameters[6].value                  # input vegetation index
        in_persistence = parameters[7].value              # input persistence
        in_rule_1_min_conseqs = parameters[8].value       # input rule 1 min conseqs
        in_rule_1_inc_plateaus = parameters[9].value      # input rule 1 include plateaus
        in_rule_2_min_zone = parameters[10].value         # input rule 2 min stdvs
        in_rule_3_num_zones = parameters[11].value        # input rule 3 num zones 
        in_ruleset = parameters[12].value                 # input rulesets
        in_alert_user = parameters[13].value              # input alert user 
        in_alert_method = parameters[14].value            # input alert method 
        in_alert_direction = parameters[15].value         # input alert direction
        in_email = parameters[16].value                   # input email
        in_ignore = parameters[17].value                  # input ignore
        
        
        
        # # # # #
        # notify user and set up progress bar 
        arcpy.AddMessage('Beginning NRT Create Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=8)


        
        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Validating existing feature...') 
        arcpy.SetProgressorPosition(1) 

        # get full path to existing monitoring areas
        exist_feat_desc = arcpy.Describe(parameters[0].value)
        in_exist_feat = os.path.join(exist_feat_desc.path, exist_feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_exist_feat)
        except Exception as e:
            arcpy.AddError('Existing monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Validating new feature...') 
        arcpy.SetProgressorPosition(2)         

        # get path to new layer 
        new_feat_desc = arcpy.Describe(in_new_feat)
        in_new_feat = os.path.join(new_feat_desc.path, new_feat_desc.name)        

        # check if new feature is valid based on name
        if 'NRT Create Monitoring Areas Draw' not in parameters[1].valueAsText:
            arcpy.AddError('New monitoring feature is incompatible.')
            return

        try:
            # check if only one new feature drawn
            with arcpy.da.SearchCursor(in_new_feat, field_names=['SHAPE@']) as cursor:
                row_count = len([row for row in cursor])
        except Exception as e:
            arcpy.AddError('Could not count new feature records.')
            arcpy.AddMessage(str(e))
            return

        # check if row count is one
        if row_count != 1:
            arcpy.AddError('Only one new monitoring area can be added at a time.')
            return

        try:
            # fetch a list of all existing feature ids
            with arcpy.da.SearchCursor(in_exist_feat, field_names=['area_id']) as cursor:
                existing_area_ids = [row[0] for row in cursor]
        except Exception as e:
            arcpy.AddError('Could not count new feature records.')
            arcpy.AddMessage(str(e))
            return

        # check if existing id already exists
        if in_area_id in existing_area_ids:
            arcpy.AddError('New area identifier {} already used.'.format(in_area_id))
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Preparing new feature geometry...') 
        arcpy.SetProgressorPosition(3)          
        
        try:
            # temporarily disable auto add to map 
            arcpy.env.addOutputsToMap = False
            
            # ensure new polygon is in albers 
            poly_wgs = arcpy.management.Project(in_dataset=in_new_feat, 
                                                out_dataset='poly_prj', 
                                                out_coor_system=3577)

            # extract binary of current geometry and set as poly
            cursor = arcpy.da.SearchCursor(poly_wgs, ['SHAPE@WKB'])
            poly = cursor.next()[0]
            
            # re-enable add to map
            arcpy.env.addOutputsToMap = True

        except Exception as e:
            arcpy.AddError('Could not prepare new feature geometry.')
            arcpy.AddMessage(str(e))    
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Inserting new feature into geodatabase...') 
        arcpy.SetProgressorPosition(4) 
        
        # set required fields 
        data = {
            'area_id':             in_area_id, 
            'platform':            in_platform, 
            's_year':              in_s_year, 
            'e_year':              in_e_year, 
            'index':               in_veg_idx,
            'persistence':         in_persistence,
            'rule_1_min_conseqs':  in_rule_1_min_conseqs,
            'rule_1_inc_plateaus': in_rule_1_inc_plateaus,
            'rule_2_min_zone':     in_rule_2_min_zone, 
            'rule_3_num_zones':    in_rule_3_num_zones,
            'ruleset':             in_ruleset,
            'alert':               in_alert_user,
            'method':              in_alert_method,
            'alert_direction':     in_alert_direction,
            'email':               in_email,
            'ignore':              in_ignore,
            'color_border':        0,                 # default border color
            'color_fill':          0,                 # default fill color
            'global_id':           uuid.uuid4().hex,  # generate guid
            'SHAPE@WKB':           poly
            }

        try:
            # validate new area before insertion
            nrt.validate_monitoring_area(tuple(data.values())[:-4])
        except Exception as e:
            arcpy.AddError('New monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        try:
            # insert new area into existing feature
            with arcpy.da.InsertCursor(in_exist_feat, field_names=list(data.keys())) as cursor:
                inserted = cursor.insertRow(list(data.values()))
        except Exception as e:
            arcpy.AddError('Could not insert new area into existing features.')
            arcpy.AddMessage(str(e))
            return
            


        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Refreshing spatial index...') 
        arcpy.SetProgressorPosition(5)        
        
        try:
            # recalculate spatial index 
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping..')
            arcpy.AddMessage(str(e))
            pass   
        
        
        
        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Removing features on map...') 
        arcpy.SetProgressorPosition(6) 
        
        try:           
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            
            # remove all layers associated with monitoring areas
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'monitoring_areas':
                    m.removeLayer(layer)
                elif layer.supports('NAME') and 'NRT Create Monitoring Areas Draw' in layer.name:
                    m.removeLayer(layer)
        except Exception as e:
            arcpy.AddWarning('Could not remove new feature from map, skipping.')
            arcpy.AddMessage(str(e))
            pass



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(7)

        try:           
            # for current project, open current map, re-add area feature
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(in_exist_feat)
            
            # update all monitoring area features symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)
        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass

        try:
            # show labels
            for layer in m.listLayers('monitoring_areas'):
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "'Zone: ' +  Text($feature.color_fill)"
                layer.showLabels = True
        except Exception as e:
            arcpy.AddWarning('Could not show labels, aborting label display.')
            arcpy.AddMessage(str(e))
            pass  



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(8)

        # notify user
        arcpy.AddMessage('Created new NRT Monitoring Area.')
        
        return


class NRT_Modify_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set input feature class
        self.label = 'NRT Modify Monitoring Areas'
        self.description = 'Modify existing NRT monitoring area features.'
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
                            displayName='Select the monitoring area to modify',
                            name='in_set_area',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_set_area.filter.type = 'ValueList'
        par_in_set_area.enabled = False
        
        # platform
        par_in_platform = arcpy.Parameter(
                            displayName='Satellite platform',
                            name='in_platform',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_platform.filter.type = 'ValueList'
        par_in_platform.filter.list = ['Landsat', 'Sentinel']
        par_in_platform.value = 'Landsat'
        par_in_platform.enabled = False
        
        # start year
        par_in_s_year = arcpy.Parameter(
                          displayName='Pre-impact start year',
                          name='in_s_year',
                          datatype='GPLong',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_in_s_year.filter.type = 'Range'
        par_in_s_year.filter.list = [1980, 2050]
        par_in_s_year.enabled = False
        
        # training length (was end year)
        par_in_e_year = arcpy.Parameter(
                          displayName='Minimum number of training dates',
                          name='in_e_year',
                          datatype='GPLong',
                          parameterType='Optional',
                          direction='Input',
                          multiValue=False)
        par_in_e_year.filter.type = 'Range'
        par_in_e_year.filter.list = [5, 99999]
        par_in_e_year.enabled = False

        # vegetation index
        par_in_veg_idx = arcpy.Parameter(
                           displayName='Vegetation index',
                           name='in_veg_idx',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           multiValue=False)
        par_in_veg_idx.filter.type = 'ValueList'
        par_in_veg_idx.filter.list = ['NDVI', 'MAVI', 'kNDVI']
        par_in_veg_idx.value = 'MAVI'
        par_in_veg_idx.enabled = False
        
        # persistence
        par_in_persistence = arcpy.Parameter(
                               displayName='Vegetation persistence',
                               name='in_persistence',
                               datatype='GPDouble',
                               parameterType='Required',
                               direction='Input',
                               multiValue=False)
        par_in_persistence.filter.type = 'Range'
        par_in_persistence.filter.list = [0.001, 9.999]
        par_in_persistence.value = 0.5
        par_in_persistence.enabled = False
       
        # rule 1 min conseqs
        par_in_min_conseqs = arcpy.Parameter(
                               displayName='Rule 1: minimum consequtives',
                               name='in_min_conseqs',
                               datatype='GPLong',
                               parameterType='Required',
                               direction='Input',
                               category='Rules',
                               multiValue=False)
        par_in_min_conseqs.filter.type = 'Range'
        par_in_min_conseqs.filter.list = [0, 99]
        par_in_min_conseqs.value = 3
        par_in_min_conseqs.enabled = False
        
        # rule 1 include plateaus
        par_in_inc_plateaus = arcpy.Parameter(
                                displayName='Rule 1: include plateaus',
                                name='in_inc_plateaus',
                                datatype='GPString',
                                parameterType='Required',
                                direction='Input',
                                category='Rules',
                                multiValue=False)
        par_in_inc_plateaus.filter.type = 'ValueList'
        par_in_inc_plateaus.filter.list = ['Yes', 'No']
        par_in_inc_plateaus.value = 'No'
        par_in_inc_plateaus.enabled = False
        
        # rule 2 minimum zone
        par_in_min_zone = arcpy.Parameter(
                            displayName='Rule 2: minimum zone',
                            name='in_min_zone',
                            datatype='GPLong',
                            parameterType='Required',
                            category='Rules',
                            direction='Input',
                            multiValue=False)
        par_in_min_zone.filter.type = 'Range'
        par_in_min_zone.filter.list = [1, 99]
        par_in_min_zone.value = 2
        par_in_min_zone.enabled = False  
        
        # rule 3 number of zones
        par_in_num_zones = arcpy.Parameter(
                             displayName='Rule 3: number of zones',
                             name='in_num_zones',
                             datatype='GPLong',
                             parameterType='Required',
                             category='Rules',
                             direction='Input',
                             multiValue=False)
        par_in_num_zones.filter.type = 'Range'
        par_in_num_zones.filter.list = [1, 99]
        par_in_num_zones.value = 2
        par_in_num_zones.enabled = False       
        
        # ruleset
        par_in_ruleset = arcpy.Parameter(
                           displayName='Ruleset',
                           name='in_ruleset',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           category='Rules',
                           multiValue=False)
        par_in_ruleset.filter.type = 'ValueList'
        ruleset = [
            '1 only',
            '2 only',
            '3 only',
            '1 and 2',
            '1 and 3',
            '2 and 3',
            '1 or 2',
            '1 or 3',
            '2 or 3',
            '1 and 2 and 3',
            '1 or 2 and 3',
            '1 and 2 or 3',
            '1 or 2 or 3'
            ] 
        par_in_ruleset.filter.list = ruleset
        par_in_ruleset.value = '1 and 2 or 3'
        par_in_ruleset.enabled = False       

        # alert user 
        par_in_alert_user = arcpy.Parameter(
                              displayName='Send alerts',
                              name='in_alert_user',
                              datatype='GPString',
                              parameterType='Required',
                              direction='Input',
                              category='Alerts and Email',
                              multiValue=False)
        par_in_alert_user.filter.type = 'ValueList'
        par_in_alert_user.filter.list = ['Yes', 'No']
        par_in_alert_user.value = 'No'
        par_in_alert_user.enabled = False     

        # alert method
        par_in_alert_method = arcpy.Parameter(
                           displayName='Alert based on method',
                           name='in_alert_method',
                           datatype='GPString',
                           parameterType='Required',
                           direction='Input',
                           category='Alerts and Email',
                           multiValue=False)
        par_in_alert_method.filter.type = 'ValueList'
        par_in_alert_method.filter.list = ['Static', 'Dynamic']
        par_in_alert_method.value = 'Static'
        par_in_alert_method.enabled = False        
        
        # alert direction 
        par_in_alert_direction = arcpy.Parameter(
                                   displayName='Direction of change to trigger alert',
                                   name='in_alert_direction',
                                   datatype='GPString',
                                   parameterType='Required',
                                   direction='Input',
                                   category='Alerts and Email',
                                   multiValue=False)
        directions = [
            'Incline only (any)', 
            'Decline only (any)', 
            'Incline only (+ zones only)', 
            'Decline only (- zones only)', 
            'Incline or Decline (any)',
            'Incline or Decline (+/- zones only)'
            ]
        par_in_alert_direction.filter.type = 'ValueList'
        par_in_alert_direction.filter.list = directions
        par_in_alert_direction.value = 'Decline only (any)'
        par_in_alert_direction.enabled = False    

        # email 
        par_in_email = arcpy.Parameter(
                         displayName='Email to recieve alerts',
                         name='in_email',
                         datatype='GPString',
                         parameterType='Optional',
                         direction='Input',
                         category='Alerts and Email',
                         multiValue=False)
        par_in_email.enabled = False   

        # ignore 
        par_in_ignore = arcpy.Parameter(
                          displayName='Ignore during monitoring',
                          name='in_ignore_user',
                          datatype='GPString',
                          parameterType='Required',
                          direction='Input',
                          category='Other',
                          multiValue=False)
        par_in_ignore.filter.type = 'ValueList'
        par_in_ignore.filter.list = ['Yes', 'No']
        par_in_ignore.value = 'No'
        par_in_ignore.enabled = False  

        # combine parameters
        parameters = [
            par_in_feat,
            par_in_set_area,
            par_in_platform,
            par_in_s_year,
            par_in_e_year,
            par_in_veg_idx,
            par_in_persistence,
            par_in_min_conseqs,
            par_in_inc_plateaus,
            par_in_min_zone,
            par_in_num_zones,
            par_in_ruleset,
            par_in_alert_user,
            par_in_alert_method,
            par_in_alert_direction,
            par_in_email,
            par_in_ignore
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
            'color_fill',
            'global_id'
            ] 
        
        # globals 
        global NRT_MODIFY_AREA 
        
        # unpack global parameter values
        curr_feat = NRT_MODIFY_AREA.get('in_feat')
        curr_area_id = NRT_MODIFY_AREA.get('in_area_id')   

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
                    
                    # use loop for rest, they're all the same 
                    for i in range(1, 16):
                        parameters[i + 1].enabled = True
                        parameters[i + 1].value = row[i]

                    # update global
                    NRT_MODIFY_AREA = {'in_area_id': parameters[1].value}

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Modify Monitoring Areas tool.
        """
        
        # safe imports
        import os
        import arcpy

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc
            
            # module folder
            sys.path.append(FOLDER_MODULES)
            import nrt
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return         
        
        # grab parameter values 
        in_feat = parameters[0]                           # input monitoring areas feature
        in_area_id = parameters[1].value                  # input monitoring area id
        in_platform = parameters[2].value                 # input platform
        in_s_year = parameters[3].value                   # input start year
        in_e_year = parameters[4].value                   # input end year
        in_veg_idx = parameters[5].value                  # input vegetation index
        in_persistence = parameters[6].value              # input persistence
        in_rule_1_min_conseqs = parameters[7].value       # input rule 1 min conseqs
        in_rule_1_inc_plateaus = parameters[8].value      # input rule 1 include plateaus
        in_rule_2_min_zone = parameters[9].value          # input rule 2 min stdvs
        in_rule_3_num_zones = parameters[10].value        # input rule 3 num zones 
        in_ruleset = parameters[11].value                 # input rulesets
        in_alert_user = parameters[12].value              # input alert user 
        in_alert_method = parameters[13].value            # input alert method 
        in_alert_direction = parameters[14].value         # input alert direction
        in_email = parameters[15].value                   # input email
        in_ignore = parameters[16].value                  # input ignore
        
        
        
        # # # # #
        # notify user and set up progress bar 
        arcpy.AddMessage('Beginning NRT Modify Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=7)
        
        
        
        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Validating existing feature...') 
        arcpy.SetProgressorPosition(1) 

        # get full path to existing monitoring areas
        exist_feat_desc = arcpy.Describe(parameters[0].value)
        in_exist_feat = os.path.join(exist_feat_desc.path, exist_feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_exist_feat)
        except Exception as e:
            arcpy.AddError('Existing monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return 
        


        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Validating new information...') 
        arcpy.SetProgressorPosition(2) 

        # set required fields 
        data = {
            'area_id':             in_area_id, 
            'platform':            in_platform, 
            's_year':              in_s_year, 
            'e_year':              in_e_year, 
            'index':               in_veg_idx,
            'persistence':         in_persistence,
            'rule_1_min_conseqs':  in_rule_1_min_conseqs,
            'rule_1_inc_plateaus': in_rule_1_inc_plateaus,
            'rule_2_min_zone':     in_rule_2_min_zone, 
            'rule_3_num_zones':    in_rule_3_num_zones,
            'ruleset':             in_ruleset,
            'alert':               in_alert_user,
            'method':              in_alert_method,
            'alert_direction':     in_alert_direction,
            'email':               in_email,
            'ignore':              in_ignore
            }

        try:
            # validate new area before insertion
            nrt.validate_monitoring_area(tuple(data.values()))
        except Exception as e:
            arcpy.AddError('Modified monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Updating feature...') 
        arcpy.SetProgressorPosition(3) 

        try:
            # open feature and update it at current id
            with arcpy.da.UpdateCursor(in_exist_feat, field_names=list(data.keys())) as cursor:
                for row in cursor:
                    
                    # update for current id
                    if row[0] == in_area_id:
                        row[1] = data.get('platform')
                        row[2] = data.get('s_year')
                        row[3] = data.get('e_year')
                        row[4] = data.get('index')
                        row[5] = data.get('persistence')
                        row[6] = data.get('rule_1_min_conseqs')
                        row[7] = data.get('rule_1_inc_plateaus')
                        row[8] = data.get('rule_2_min_zone')
                        row[9] = data.get('rule_3_num_zones')
                        row[10] = data.get('ruleset')
                        row[11] = data.get('alert')
                        row[12] = data.get('method')
                        row[13] = data.get('alert_direction')
                        row[14] = data.get('email')
                        row[15] = data.get('ignore')

                        # update row
                        cursor.updateRow(row)
        except Exception as e:
            arcpy.AddError('Could not update existing feature.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Refreshing spatial index...') 
        arcpy.SetProgressorPosition(4)        

        try:
            # recalculate spatial index 
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping..')
            arcpy.AddMessage(str(e))
            pass   



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Removing features from map...') 
        arcpy.SetProgressorPosition(5) 

        try:           
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove all layers associated with monitoring areas
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'monitoring_areas':
                    m.removeLayer(layer)
        except Exception as e:
            arcpy.AddWarning('Could not remove new feature from map, skipping.')
            arcpy.AddMessage(str(e))
            pass



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(6)

        try:           
            # for current project, open current map, re-add area feature
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(in_exist_feat)

            # update all monitoring area features symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)

        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass
            
        try:
            # show labels
            for layer in m.listLayers('monitoring_areas'):
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "'Zone: ' +  Text($feature.color_fill)"
                layer.showLabels = True
        except Exception as e:
            arcpy.AddWarning('Could not show labels, aborting label display.')
            arcpy.AddMessage(str(e))
            pass  



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(7)

        # notify user
        arcpy.AddMessage('Modified existing NRT Monitoring Area.')
        
        return


class NRT_Delete_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'NRT Delete Monitoring Areas'
        self.description = 'Delete existing NRT monitoring area features.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # set input feature
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
                            displayName='Select the monitoring area to delete',
                            name='in_area_id',
                            datatype='GPString',
                            parameterType='Required',
                            direction='Input',
                            multiValue=False)
        par_in_set_area.enabled = False
        
        # combine parameters
        parameters = [
            par_in_feat,
            par_in_set_area
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
            'color_fill',
            'global_id'
            ] 
        
        # globals 
        global NRT_DELETE_AREA

        # unpack global parameter values 
        curr_feat = NRT_DELETE_AREA.get('in_feat')
        
        # check existing feature input
        if parameters[0].valueAsText != curr_feat:
            try:
                # load column names
                in_feat = parameters[0].valueAsText                
                cols = [f.name for f in arcpy.ListFields(in_feat)]
            except:
                cols = []

            # if invalid fields, abort
            if all(f in cols for f in required_fields):
                try:
                    # get all rows as list of tuples with specific field order
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
            
                # if row exists, enable + populate controls with first row values 
                if row is not None:
                    parameters[1].enabled = True
                    parameters[1].filter.list = [rec[0] for rec in rows]
                    parameters[1].value = row[0]
                    
                else:
                    parameters[1].enabled = True
                    parameters[1].filter.list = []
                    parameters[1].value = None

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Delete Monitoring Areas tool.
        """
        
        # safe imports
        import os
        import arcpy
        
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc
            
            # module folder
            sys.path.append(FOLDER_MODULES)
            import nrt
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return   
        
        # grab parameter values 
        in_exist_feat = parameters[0]       # input monitoring areas feature
        in_area_id = parameters[1].value    # input monitoring area id

        
        
        # # # # #
        # notify user and set up progress bar 
        arcpy.AddMessage('Beginning NRT Delete Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=6)        
        
        
        
        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Validating existing feature...') 
        arcpy.SetProgressorPosition(1) 

        # get full path to existing monitoring areas
        exist_feat_desc = arcpy.Describe(parameters[0].value)
        in_exist_feat = os.path.join(exist_feat_desc.path, exist_feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_exist_feat)
        except Exception as e:
            arcpy.AddError('Existing monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Deleting monitoring area feature...') 
        arcpy.SetProgressorPosition(2)        
        
        try:
            # delete feature at current id
            with arcpy.da.UpdateCursor(in_exist_feat, field_names=['area_id']) as cursor:
                for row in cursor:
                    if row[0] == in_area_id:
                        cursor.deleteRow()
        except Exception as e:
            arcpy.AddError('Could not delete existing feature.')
            arcpy.AddMessage(str(e))
            return
            
            
            
        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Refreshing spatial index...') 
        arcpy.SetProgressorPosition(3)        

        try:
            # recalculate spatial index 
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping.')
            arcpy.AddMessage(str(e))
            pass  



        # # # # #
        # notify user and increment progress bar 
        arcpy.SetProgressorLabel('Removing features from map...') 
        arcpy.SetProgressorPosition(4) 

        try:           
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove all layers associated with monitoring areas
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'monitoring_areas':
                    m.removeLayer(layer)
        except Exception as e:
            arcpy.AddWarning('Could not remove new feature from map, skipping.')
            arcpy.AddMessage(str(e))
            pass



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(5)

        try:           
            # for current project, open current map, re-add area feature
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(in_exist_feat)

            # update all monitoring area features symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)
        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass

        try:
            # show labels
            for layer in m.listLayers('monitoring_areas'):
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "'Zone: ' +  Text($feature.color_fill)"
                layer.showLabels = True
        except Exception as e:
            arcpy.AddWarning('Could not show labels, aborting label display.')
            arcpy.AddMessage(str(e))
            pass  
            
            

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(6)

        # notify user
        arcpy.AddMessage('Deleted existing NRT Monitoring Area.')
        
        return


class NRT_Monitor_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'NRT Monitor Areas'
        self.description = 'Perform on-demand or on-going vegetation ' \
                           'change monitoring for pre-created monitoring ' \
                           'areas. Optionally send emails when certain changes ' \
                           'occur.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # input monitoring area feature
        par_in_feat = arcpy.Parameter(
                        displayName='Input monitoring area feature',
                        name='in_feat',
                        datatype='GPFeatureLayer',
                        parameterType='Required',
                        direction='Input',
                        multiValue=False)
        par_in_feat.filter.list = ['Polygon']

        # continuous monitor
        par_continuous = arcpy.Parameter(
                           displayName='Continuously monitor areas',
                           name='in_continuous',
                           datatype='GPBoolean',
                           parameterType='Required',
                           direction='Input',
                           multiValue=False)
        par_continuous.value = False

        # num days for checks
        par_num_days = arcpy.Parameter(
                         displayName='Number of days between cycles',
                         name='in_num_days',
                         datatype='GPLong',
                         parameterType='Required',
                         direction='Input',
                         multiValue=False)
        par_num_days.filter.type = 'Range'
        par_num_days.filter.list = [1, 365]
        par_num_days.value = 1

        # send email alerts
        par_send_email = arcpy.Parameter(
                           displayName='Send email alerts',
                           name='in_send_email',
                           datatype='GPBoolean',
                           parameterType='Required',
                           direction='Input',
                           multiValue=False)
        par_send_email.value = False

        # email host (i.e. from) 
        par_email_host = arcpy.Parameter(
                           displayName='Host email address',
                           name='in_email_host',
                           datatype='GPString',
                           parameterType='Optional',
                           direction='Input',
                           multiValue=False)
        par_email_host.value = None

        # email smtp server
        par_email_server = arcpy.Parameter(
                             displayName='Host server address',
                             name='in_email_server',
                             datatype='GPString',
                             parameterType='Optional',
                             direction='Input',
                             multiValue=False)
        par_email_server.value = None
        
        # email smtp port
        par_email_port = arcpy.Parameter(
                           displayName='Host server port',
                           name='in_email_port',
                           datatype='GPLong',
                           parameterType='Optional',
                           direction='Input',
                           multiValue=False)
        par_email_port.value = None

        # email smtp username
        par_email_username = arcpy.Parameter(
                               displayName='Host username',
                               name='in_email_username',
                               datatype='GPString',
                               parameterType='Optional',
                               direction='Input',
                               multiValue=False)
        par_email_username.value = None
        
        # email smtp password
        par_email_password = arcpy.Parameter(
                               displayName='Host password',
                               name='in_email_password',
                               datatype='GPStringHidden',
                               parameterType='Optional',
                               direction='Input',
                               multiValue=False)
        par_email_password.value = None

        # combine parameters
        parameters = [
            par_in_feat,
            par_continuous,
            par_num_days,
            par_send_email,
            par_email_host,
            par_email_server,
            par_email_port,
            par_email_username,
            par_email_password
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
    
        # enable number of days selector
        if parameters[1].value is True:
            parameters[2].enabled = True
        else:
            parameters[2].enabled = False
        
        # enable email parameters
        if parameters[3].value is True:
            parameters[4].enabled = True
            parameters[5].enabled = True
            parameters[6].enabled = True
            parameters[7].enabled = True
            parameters[8].enabled = True
        else:
            parameters[4].enabled = False
            parameters[5].enabled = False
            parameters[6].enabled = False
            parameters[7].enabled = False
            parameters[8].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Monitor Areas module.
        """
        
        # safe imports
        import os, sys
        import time
        import datetime   
        import numpy as np
        import tempfile
        import arcpy

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
        try:
            import xarray as xr
            import dask
            import rasterio
            import pystac_client
            from odc import stac
        except Exception as e:
            arcpy.AddError('Python libraries xarray, dask, rasterio, odc not installed.')
            arcpy.AddMessage(str(e))
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools

            # module folder
            sys.path.append(FOLDER_MODULES)
            import cog_odc, nrt
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return

        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)   
         
        # grab parameter values 
        in_feat = parameters[0]                            # input monitoring area feature
        in_continuous = parameters[1].value                # continuous monitoring
        in_num_days = parameters[2].value                  # days between checks
        in_send_email = parameters[3].value                # send email alerts
        in_email_host = parameters[4].value                # host email address
        in_email_server = parameters[5].value              # host email server
        in_email_port = parameters[6].value                # host email port
        in_email_username = parameters[7].value            # host email username
        in_email_password = parameters[8].value            # host email password
        
        
        
        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Validating monitoring area geodatabase...')

        # get full path to existing monitoring areas
        feat_desc = arcpy.Describe(parameters[0].value)
        in_feat = os.path.join(feat_desc.path, feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_feat)
        except Exception as e:
            arcpy.AddError('Monitoring areas feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return



        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Obtaining and validating monitoring areas...')

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
            # get features
            with arcpy.da.SearchCursor(in_feat, fields) as cursor:
                feats = [row for row in cursor]
        except Exception as e:
            arcpy.AddError('Could not read monitoring areas.')
            arcpy.AddMessage(str(e))
            return

        # check if feature exists 
        if len(feats) == 0:
            arcpy.AddError('No monitoring areas exist.')
            return
            
            
            
        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Converting features to monitoring areas objects...')
        
        # prepare path to current geodatabase folder
        in_path = os.path.dirname(in_feat)
        in_path = os.path.splitext(in_path)[0]
        in_path = os.path.dirname(in_path)

        try:
            # iter feature and convert to monitoring area objects
            areas = [nrt.MonitoringArea(feat, path=in_path) for feat in feats]
        except Exception as e:
            arcpy.AddError('Could not convert to monitoring area objects.')
            arcpy.AddMessage(str(e))
            return
            
        # check if areas exist
        if len(areas) == 0:
            arcpy.AddError('No monitoring area objects exist.')
            return



        # # # # #
        # notify and set on-going progess bar
        arcpy.AddMessage('Beginning monitoring cycle...')
        arcpy.SetProgressor('default', 'Monitoring existing areas...')

        # check continuous and num days are valid 
        if in_continuous not in [True, False]:
            arcpy.AddError('Did not provide continuous monitoring option.')
            return
        elif in_num_days < 1:
            arcpy.AddError('Number of days between cycles must be >= 1.')
            return
            
        # prepare monitor iterator 
        iterations = 99999 if in_continuous else 1
        
        # begin monitoring cycles...
        for _ in range(iterations):

            # iter monitoring area objects...
            for area in areas:
            
                # # # # #
                # notify user and set up progress bar
                arcpy.AddMessage(u'\u200B')
                arcpy.AddMessage('Starting area {}.'.format(area.area_id))
                arcpy.SetProgressor(type='step',
                                    message='Starting area: {}...'.format(area.area_id),
                                    min_range=0, max_range=30)
                
                # notify and skip area if user requested
                if area.ignore is True:
                    arcpy.AddMessage('Area set to ignore, skipping.')
                    continue

                

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Validating monitoring area...')
                arcpy.SetProgressorLabel('Validating monitoring area...') 
                arcpy.SetProgressorPosition(1) 
            
                try:
                    # validate area, skip if error
                    area.validate_area()
                except Exception as e:
                    arcpy.AddWarning('Area is invalid, see messages for details.')
                    arcpy.AddMessage(str(e))
                    continue
                    
                    
                    
                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Checking for existing area NetCDF...')
                arcpy.SetProgressorLabel('Checking for existing area NetCDF...') 
                arcpy.SetProgressorPosition(2)
                
                try:
                    # get old xr, if none set old to none, if error set old to none
                    area.set_old_xr()
                    
                    # notify if no old xr and proceed
                    if area.ds_old is None:
                        arcpy.AddMessage('No existing NetCDF found.')
                        pass
                    else:
                        arcpy.AddMessage('Found existing NetCDF.')
                        pass
                except:
                    arcpy.AddMessage('No existing area NetCDF, fetching a new NetCDF.')
                    pass
                    
                    
                    
                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Validating existing NetCDF...')
                arcpy.SetProgressorLabel('Validating existing NetCDF...') 
                arcpy.SetProgressorPosition(3)
                
                try:
                    # validate old xr, skip if none or attrs differ, skip if error
                    area.validate_old_xr()
                    
                    # notify if no old xr and proceed
                    if area.ds_old is None:
                        arcpy.AddMessage('Existing NetCDF is invalid (or does not exist).')
                        pass
                    else:
                        arcpy.AddMessage('Existing NetCDF is valid.')
                        pass
                except:
                    arcpy.AddWarning('Attributes of existing NetCDF have changed, resetting.')
                    pass



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Obtaining new satellite data, if exists...')
                arcpy.SetProgressorLabel('Obtaining new satellite data, if exists...') 
                arcpy.SetProgressorPosition(4)

                try:
                    # get all sat data, set new xr to new dates if old exists else all new, on error set new to none and skip
                    area.set_new_xr()  

                    # skip if no new dates, else notify and proceed
                    if area.new_xr_dates_found() is False:
                        arcpy.AddMessage('No new satellite images, skipping area.')
                        continue
                    else:
                        arcpy.AddMessage('Found {} new satellite images.'.format(len(area.ds_new['time'])))
                        pass
                except Exception as e:
                    arcpy.AddWarning('Could not obtain new satellite data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue    



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing satellite images with clouds...')
                arcpy.SetProgressor('default', 'Removing satellite images with clouds...')

                try:
                    # apply fmask on new xr. if error or no dates, skip
                    area.apply_new_xr_fmask()

                    # check if any valid images remain, else notify and skip
                    if area.new_xr_dates_found() is False:
                        arcpy.AddMessage('No cloud-free satellite images remain, skipping area.')
                        continue 
                    else:
                        arcpy.AddMessage('Found {} cloud-free satellite images'.format(len(area.ds_new['time'])))
                        pass
                except Exception as e:
                    arcpy.AddWarning('Could not apply fmask, see messages.')
                    arcpy.AddMessage(str(e))
                    continue   



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating vegetation index for new satellite images...')
                arcpy.SetProgressorLabel('Calculating vegetation index for new satellite images...') 
                arcpy.SetProgressorPosition(6)

                try:
                    # calculate vege index for new xr. if error or no dates, skip
                    area.apply_new_xr_index() 
                except Exception as e:
                    arcpy.AddWarning('Could not calculate vegetation index, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and on-going progress bar
                arcpy.AddMessage('Downloading new satellite data, please wait...')
                arcpy.SetProgressor('default', 'Downloading new satellite data, please wait...')

                try:
                    # load new xr, skip if error
                    area.load_new_xr() 
                except Exception as e:
                    arcpy.AddWarning('Could not download satellite data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing edge pixels via mask...')
                arcpy.SetProgressorLabel('Removing edge pixels via mask...') 
                arcpy.SetProgressorPosition(8)

                try:
                    # remove edge pixels. if error or empty, return original xr
                    area.remove_new_xr_edges() 
                except Exception as e:
                    arcpy.AddWarning('Could not remove edge pixels, proceeding without mask.')
                    arcpy.AddMessage(str(e))
                    pass



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Reducing new satellite images down to temporal medians...')
                arcpy.SetProgressorLabel('Reducing new satellite images down to temporal medians...') 
                arcpy.SetProgressorPosition(9)

                try:
                    # reduce new xr to median value per date. skip if error
                    area.reduce_new_xr() 
                except Exception as e:
                    arcpy.AddWarning('Could not reduce satellite images, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Interpolating missing pixels...')
                arcpy.SetProgressorLabel('Interpolating missing pixels...') 
                arcpy.SetProgressorPosition(10)

                try:
                    # interp nans. if any remain, drop. if empty, error and skip
                    area.interp_new_xr_nans() 
                except Exception as e:
                    arcpy.AddWarning('Could not interpolate missing pixels, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Appending required variables...')
                arcpy.SetProgressorLabel('Appending required variables...') 
                arcpy.SetProgressorPosition(11)

                try:
                    # append required vars to new xr. skip if error
                    area.append_new_xr_vars() 
                except Exception as e:
                    arcpy.AddWarning('Could not append variables, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Combining old and new satellite images, if required...')
                arcpy.SetProgressorLabel('Combining old and new satellite images, if required...') 
                arcpy.SetProgressorPosition(12)

                try:
                    # combine old and new xrs. if no old xr, new xr used. if error, skip
                    area.set_cmb_xr() 
                except Exception as e:
                    arcpy.AddWarning('Could not combine old and new images, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing vegetation spikes in combined images...')
                arcpy.SetProgressorLabel('Removing vegetation spikes in combined images...') 
                arcpy.SetProgressorPosition(13)

                try:
                    # replace cmb spikes, interp removed. set result to var veg_clean. skip if error or empty.
                    area.fix_cmb_xr_spikes() 
                except Exception as e:
                    arcpy.AddWarning('Could not remove spikes, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Smoothing vegetation in combined images...')
                arcpy.SetProgressorLabel('Smoothing vegetation spike in combined images...') 
                arcpy.SetProgressorPosition(14)

                try:
                    # smooth cmb veg_clean var with savitsky filter. skip if error.
                    area.smooth_cmb_xr_index() 
                except Exception as e:
                    arcpy.AddWarning('Could not smooth vegetation, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Building analysis dataset...')
                arcpy.SetProgressorLabel('Building analysis dataset...') 
                arcpy.SetProgressorPosition(15)

                try:
                    # build anl xr, remove pre-training years. skip if error.
                    area.set_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not build analysis dataset, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Performing change detection on analysis dataset...')
                arcpy.SetProgressorLabel('Performing change detection on analysis dataset...') 
                arcpy.SetProgressorPosition(16)

                try:
                    # detect static, dynamic change via anl xr, skip if error.
                    area.detect_change_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not perform change detection, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Smoothing change detection on analysis dataset...')
                arcpy.SetProgressorLabel('Smoothing change detection on analysis dataset...') 
                arcpy.SetProgressorPosition(17)

                try:
                    # smooth static, dynamic change via anl xr, skip if error.
                    area.smooth_anl_xr_change()
                except Exception as e:
                    arcpy.AddWarning('Could not smooth change detection data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Transfering old data to analysis dataset...')
                arcpy.SetProgressorLabel('Transfering old data to analysis dataset...') 
                arcpy.SetProgressorPosition(18)

                try:
                    # transfer old xr change values to anl xr, skip if error.
                    area.transfer_old_to_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not transfer data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating zones...')
                arcpy.SetProgressorLabel('Calculating zones...') 
                arcpy.SetProgressorPosition(19)
                
                try:
                    # build zones, skip if error.
                    area.build_zones()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate zones, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating rule one, two and three...')
                arcpy.SetProgressorLabel('Calculating rule one, two and three...') 
                arcpy.SetProgressorPosition(20)
                
                try:
                    # build rule one, two, three. skip if error.
                    area.build_rule_one()
                    area.build_rule_two()
                    area.build_rule_three()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate rules, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating alerts...')
                arcpy.SetProgressorLabel('Calculating alerts...') 
                arcpy.SetProgressorPosition(21)

                try:
                    # build alerts, skip if error.
                    area.build_alerts()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate alerts, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Transfering analysis data to combined dataset...')
                arcpy.SetProgressorLabel('Transfering analysis data to combined dataset...') 
                arcpy.SetProgressorPosition(22)
                
                try:
                    # transfer anl xr to cmb xr. skip if error.
                    area.transfer_anl_to_cmb_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not transfer analysis data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Appending area attributes on to combined dataset...')
                arcpy.SetProgressorLabel('Appending area attributes on to combined dataset...') 
                arcpy.SetProgressorPosition(23)
                
                try:
                    # append area field attrs on to cmb xr. skip if error.
                    area.append_cmb_xr_attrs()
                except Exception as e:
                    arcpy.AddWarning('Could not append area attributes, see messages.')
                    arcpy.AddMessage(str(e))
                    continue   



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Generating area info, html and graph...')
                arcpy.SetProgressorLabel('Generating area info, html and graph...') 
                arcpy.SetProgressorPosition(24)
                
                try:
                    # generate alert info, html, graph data. skip if error.
                    area.set_alert_data()
                except Exception as e:
                    arcpy.AddWarning('Could not generate alert information, html, graphs.')
                    arcpy.AddMessage(str(e))
                    continue  



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Refreshing monitoring area symbology...')
                arcpy.SetProgressorLabel('Refreshing monitoring area symbology...') 
                arcpy.SetProgressorPosition(25)
                
                try:
                    # refresh monitoring area symbology. skip if error.
                    area.refresh_area_symbology()
                except Exception as e:
                    arcpy.AddWarning('Could not refresh symbology, see messages.')
                    arcpy.AddMessage(str(e))
                    pass  



                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Exporting combined NetCDF...')
                arcpy.SetProgressorLabel('Exporting combined data...') 
                arcpy.SetProgressorPosition(26)

                try:
                    area.export_cmb_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not export NetCDF, see messages.')
                    arcpy.AddMessage(str(e))
                    continue  
                    
                
                
                # # # # #
                # notify user and show on-going progress bar               
                arcpy.AddMessage('Cooling off for 10 seconds...')
                arcpy.SetProgressor('default', 'Cooling off for 10 seconds...')
                
                # pause for 10 seconds 
                for sec in range(10):
                    time.sleep(1.0)



            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage('Preparing and sending email alert, if requested...')
            arcpy.SetProgressor('default', 'Preparing and sending email alert, if requested...')
            
            # if requested...
            if in_send_email is True:            
                try:
                    # prepare email alerts for each area and send, skip if error
                    nrt.email_alerts(areas, 
                                     host_email=in_email_host, 
                                     host_server=in_email_server, 
                                     host_port=in_email_port, 
                                     host_user=in_email_username, 
                                     host_pass=in_email_password)
                except Exception as e:
                    arcpy.AddWarning('Could not send email, see messages for details.')
                    arcpy.AddMessage(str(e))
                    pass


            
            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage(u'\u200B')
            arcpy.AddMessage('Resetting monitoring areas...')
            arcpy.SetProgressor('default', 'Resetting monitoring areas...')
            
            try:
                # reset each monitor area's internal data 
                for area in areas:
                    area.reset()
            except Exception as e:
                arcpy.AddWarning('Could not reset areas, see messages for details.')
                arcpy.AddMessage(str(e))
                pass



            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage('Waiting for next cycle, if requested...')
            arcpy.SetProgressor('default', 'Waiting for next cycle, if requested...')
            
            # if requested, were done!
            if not in_continuous:
                break 

            # prepare new progressor for wait
            arcpy.SetProgressor(type='step', 
                                message='Waiting for next monitoring cycle...', 
                                min_range=0, 
                                max_range=in_num_days * 86400)
        
            # begin iter until next cycle
            for sec in range(in_num_days * 86400):
                arcpy.SetProgressorPosition(sec)
                time.sleep(1.0)
                
                # ensure we leave on tool cancel properly
                if arcpy.env.isCancelled:
                    sys.exit(0)



        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(3)

        # notify user
        arcpy.AddMessage('Monitored areas successfully.')
        return


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

        # safe imports
        import os, sys   
        import datetime   
        import numpy as np
        import tempfile
        import arcpy
        
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
        try:
            import xarray as xr
            import dask
            import rasterio
            import pystac_client
            from odc import stac
        except Exception as e:
            arcpy.AddError('Python libraries xarray, dask, rasterio, odc not installed.')
            arcpy.AddMessage(str(e))
            return
            
        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, satfetcher, tools
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import cog_odc, nrt
        except Exception as e:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            arcpy.AddMessage(str(e))
            return
            
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

        # grab parameter values 
        in_feat = parameters[0]              # input monitoring areas feature
        in_area_id = parameters[1].value     # input monitoring area id
        out_nc = parameters[2].valueAsText   # output netcdf



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
                #layer.visible = False
                
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
                        #layer.visible = True
                        
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


class NRT_Build_Graphs(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = 'NRT Build Graphs'
        self.description = 'Builds google charts when ' \
                           'user clicks on monitoring area. ' \
                           'Not intended for manual use.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """
        
        # input netcdf path
        par_in_path = arcpy.Parameter(
                        displayName='Input NetCDF path',
                        name='in_nc',
                        datatype='GPString',
                        parameterType='Required',
                        direction='Input',
                        multiValue=False)
        par_in_path.value = ''
        
        # input area parameters
        par_in_params = arcpy.Parameter(
                          displayName='Parameters',
                          name='in_parameters',
                          datatype='GPString',
                          parameterType='Required',
                          direction='Input',
                          multiValue=False)
        par_in_params.value = ''
        
        # combine parameters
        parameters = [
            par_in_path, 
            par_in_params
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
        Executes the NRT Build Graphs module.
        """
        
        # safe imports
        import os
        import numpy as np
        import arcpy
        
        # risky imports
        try:
            import xarray as xr
        except:
            arcpy.AddMessage('<h3>Could not generate graph information.</h3>')
            return
       
        # set up non-chart html 
        _html_overview = """
        <html>
            <head>
                <style type="text/css">
                    h4 {
                        margin-top: 0px;
                        margin-bottom: 0px;
                    }
                    p {
                        margin-top: 0px;
                        margin-bottom: 0px;
                    }
                </style>
            </head>
            <body>
                <center>
                    <h3>Area Overview</h3>
                </center>

                <h4>Area identifier</h4>
                <p>//data.OvrAreaId</p>
                <br />

                <h4>Satellite platform</h4>
                <p>//data.OvrPlatform</p>
                <br />

                <h4>Starting year of pre-impact period</h4>
                <p>//data.OvrSYear</p>
                <br />

                <h4>Minimum number of training dates</h4>
                <p>//data.OvrEYear</p>
                <br />

                <h4>Vegetation index</h4>
                <p>//data.OvrIndex</p>
                <br />

                <h4>Model Persistence</h4>
                <p>//data.OvrPersistence</p>
                <br />

                <h4>Rule 1: Minimum consequtives</h4>
                <p>//data.OvrRule1MinConseq</p>
                <br />

                <h4>Rule 1: Include Plateaus</h4>
                <p>//data.OvrRule1IncPlateaus</p>
                <br />

                <h4>Rule 2: Minimum Zone</h4>
                <p>//data.OvrRule2MinZone</p>
                <br />

                <h4>Rule 3: Number of Zones</h4>
                <p>//data.OvrRule3NumZones</p>
                <br />

                <h4>Ruleset</h4>
                <p>//data.OvrRuleset</p>
                <br />

                <h4>Alert via email</h4>
                <p>//data.OvrAlert</p>
                <br />
                
                <h4>Alert Method</h4>
                <p>//data.OvrMethod</p>
                <br />

                <h4>Alert direction</h4>
                <p>//data.OvrDirection</p>
                <br />

                <h4>User email</h4>
                <p>//data.OvrEmail</p>
                <br />

                <h4>Ignore during monitoring</h4>
                <p>//data.OvrIgnore</p>
                <br />
            </body>
        </html>
        """

        # set up full veg html template
        _html_full_veg = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Vegetation (Raw)');
                data.addColumn('number', 'Vegetation (Smooth)');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Health (Median)',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'grey',
                      lineWidth: 1,
                      enableInteractivity: false
                    },
                    1: {
                      color: 'green'
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }
            </script>
          </head>
          <body>
            <center>
              <h3>Vegetation History (Full)</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>
        </html>
        """
      
        # set up sub veg html template
        _html_sub_veg = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Vegetation (Raw)');
                data.addColumn('number', 'Vegetation (Smooth)');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Health (Median)',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'grey',
                      lineWidth: 1,
                      enableInteractivity: false
                    },
                    1: {
                      color: 'green'
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }

            </script>
          </head>

          <body>
            <center>
              <h3>Vegetation History (Analysis Only)</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>

        </html>
        """

        # set up sub change html template
        _html_sub_change = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Change');
                data.addColumn('number', 'Alert');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Change Deviation',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'red',
                      //lineWidth: 1,
                      //enableInteractivity: false
                    },
                    1: {
                        color: 'maroon',
                        lineWidth: 0,
                        pointSize: 5
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }

            </script>
          </head>

          <body>
            <center>
              <h3>Change Deviation & Alerts</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>

        </html>
        """

        # set up sub zone html template
        _html_sub_zone = """
        <html>
            <head>
                <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
                <script type="text/javascript">
                    google.charts.load('current', { packages: ['corechart', 'bar']});
                    google.charts.setOnLoadCallback(drawChart);

                    function drawChart() {

                        //data.addRows

                        var options = {
                            legend: {
                                position: 'none',
                            },
                            chartArea: {
                                width: '100%', 
                                height: '100%',
                                top: 20,
                                bottom: 100,
                                left: 75,
                                right: 25
                                },
                            hAxis: {
                                title: 'Date',
                                textStyle: {fontSize: '9'},
                                slantedText: true, 
                                slantedTextAngle: 90
                            },
                            vAxis: {
                                title: 'Zone',
                                textStyle: {fontSize: '9'},
                            },
                        };

                        var chart = new google.visualization.ColumnChart(document.getElementById('column_chart'));
                        chart.draw(data, options);
                    }
                </script>
            </head>
            <body>
                <center>
                    <h3>Zone & Alert History</h3>
                    <div id="column_chart" style="width: 100%; height: 90%"></div>
                </center>
                
            </body>
        </html>
        """

        # set up legend html template
        _html_legend = """
        <html>
          <head>
            <style>
              td, th {
                border: 1px solid transparent;
                text-align: left;
                padding: 0px;
              }
            </style>
          </head>

          <body>
            <center>
              <h3>Zone Legend</h3>
            </center>

            <table style="width: 100%;">
              <colgroup>
                <col span="1" style="width: 15%;">
                <col span="1" style="width: 15%;">
                <col span="1" style="width: 70%;">
              </colgroup>
              <tr>
                <th>Symbology</th>
                <th>Zone</th>
                <th>Description</th>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FF7F7F"></div>
                  </div>
                </td>
                <td>-11</td>
                <td>Change deviation is below -19. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFA77F"></div>
                  </div>
                </td>
                <td>-10</td>
                <td>Change deviation is between -17 and -19. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFD37F"></div>
                  </div>
                </td>
                <td>-9</td>
                <td>Change deviation is between -15 and -17. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFFF73"></div>
                  </div>
                </td>
                <td>-8</td>
                <td>Change deviation is between -13 and -15. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #D1FF73"></div>
                  </div>
                </td>
                <td>-7</td>
                <td>Change deviation is between -11 and -13. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #A3FF73"></div>
                  </div>
                </td>
                <td>-6</td>
                <td>Change deviation is between -9 and -11. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73FFDF"></div>
                  </div>
                </td>
                <td>-5</td>
                <td>Change deviation is between -7 and -9. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73DFFF"></div>
                  </div>
                </td>
                <td>-4</td>
                <td>Change deviation is between -5 and -7. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73B2FF"></div>
                  </div>
                </td>
                <td>-3</td>
                <td>Change deviation is between -3 and -5. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #DF73FF"></div>
                  </div>
                </td>
                <td>-2</td>
                <td>Change deviation is between -1 and -3. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FF73DF"></div>
                  </div>
                </td>
                <td>-1</td>
                <td>Change deviation is between 0 and -1. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid black; background-color: white"></div>
                  </div>
                </td>
                <td>0</td>
                <td>No change in either direction.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FF73DF"></div>
                  </div>
                </td>
                <td>1</td>
                <td>Change deviation is between 0 and 1. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #DF73FF"></div>
                  </div>
                </td>
                <td>2</td>
                <td>Change deviation is between 1 and 3. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73B2FF"></div>
                  </div>
                </td>
                <td>3</td>
                <td>Change deviation is between 3 and 5. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73DFFF"></div>
                  </div>
                </td>
                <td>4</td>
                <td>Change deviation is between 5 and 7. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73FFDF"></div>
                  </div>
                </td>
                <td>5</td>
                <td>Change deviation is between 7 and 9. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #A3FF73"></div>
                  </div>
                </td>
                <td>6</td>
                <td>Change deviation is between 9 and 11. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #D1FF73"></div>
                  </div>
                </td>
                <td>7</td>
                <td>Change deviation is between 11 and 13. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFFF73"></div>
                  </div>
                </td>
                <td>8</td>
                <td>Change deviation is between 13 and 15. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFD37F"></div>
                  </div>
                </td>
                <td>9</td>
                <td>Change deviation is between 15 and 17. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFA77F"></div>
                  </div>
                </td>
                <td>10</td>
                <td>Change deviation is between 17 and 19. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FF7F7F"></div>
                  </div>
                </td>
                <td>11</td>
                <td>Change deviation is above 19. Growth.</td>
              </tr>
            </table>
          </body>
        </html>
        """
        
        # grab parameter values
        in_path = parameters[0].value         # filepath to selected netcdf
        in_params = parameters[1].value       # string of area params seperated by ";"



        # # # # #
        #  prepare overview data html 
        
        try:
            # unpack parameters 
            params = in_params.split(';')
            
            # set over html values 
            _html_overview = _html_overview.replace('//data.OvrAreaId', params[0])
            _html_overview = _html_overview.replace('//data.OvrPlatform', params[1])
            _html_overview = _html_overview.replace('//data.OvrSYear', params[2])
            _html_overview = _html_overview.replace('//data.OvrEYear', params[3])
            _html_overview = _html_overview.replace('//data.OvrIndex', params[4])
            _html_overview = _html_overview.replace('//data.OvrPersistence', params[5])
            _html_overview = _html_overview.replace('//data.OvrRule1MinConseq', params[6])
            _html_overview = _html_overview.replace('//data.OvrRule1IncPlateaus', params[7])
            _html_overview = _html_overview.replace('//data.OvrRule2MinZone', params[8])
            _html_overview = _html_overview.replace('//data.OvrRule3NumZones', params[9])
            _html_overview = _html_overview.replace('//data.OvrRuleset', params[10])
            _html_overview = _html_overview.replace('//data.OvrAlert', params[11])
            _html_overview = _html_overview.replace('//data.OvrMethod', params[12])
            _html_overview = _html_overview.replace('//data.OvrDirection', params[13])
            _html_overview = _html_overview.replace('//data.OvrEmail', params[14])
            html = _html_overview.replace('//data.OvrIgnore', params[15])            
        except:
            html = '<h3>Could not generate overview information.</h3>'

        # add to output 
        arcpy.AddMessage(html)



        # # # # #
        # check and load input netcdf at path
        
        try:
            # check if input path is valid
            if in_path is None or in_path == '':
                raise
            elif not os.path.exists(in_path):
                raise

            # safe open current dataset
            with xr.open_dataset(in_path) as ds:
                ds.load() 

            # check if dataset is valid
            if 'time' not in ds:
                raise
            elif len(ds['time']) == 0:
                raise
            elif ds.to_array().isnull().all():
                raise
                
            # remove last date (-1 when slice)
            ds = ds.isel(time=slice(0, -1))
        except:
            pass  # proceed, causing error messages below



        # # # # #
        # prepare full vegetation history chart
        
        try:
            # unpack full time, veg raw, veg clean values
            ds_full = ds[['veg_idx', 'veg_clean']]
            dts = ds_full['time'].dt.strftime('%Y-%m-%d').values
            raw = ds_full['veg_idx'].values
            cln = ds_full['veg_clean'].values

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], raw[i], cln[i]])
                
            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')
            
            # construct and relay full veg line chart html
            data = "data.addRows(" + data + ");"
            html = _html_full_veg.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate full vegetation history chart.</h3>'

        # add to output 
        arcpy.AddMessage(html)
        


        # # # # #
        # prepare analysis vegetation history chart

        try:
            # drop dates < start year and get veg raw, clean
            ds_sub = ds.where(ds['time.year'] >= int(params[2]), drop=True)
            dts = ds_sub['time'].dt.strftime('%Y-%m-%d').values
            raw = ds_sub['veg_idx'].values
            cln = ds_sub['veg_clean'].values
            
            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], raw[i], cln[i]])
            
            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')
            
            # construct and relay full veg line chart html
            data = "data.addRows(" + data + ");"
            html = _html_sub_veg.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate analysis-only vegetation history chart.</h3>'

        # add to output 
        arcpy.AddMessage(html)



        # # # # #
        # prepare change history chart

        try:
            # get method name and prepare vars
            method = params[12].lower()
            change_var = '{}_clean'.format(method)
            alert_var = '{}_alerts'.format(method)
        
            # get change
            chg = ds_sub[change_var].values
            
            # get where alerts exist 
            alt = ds_sub[change_var].where(ds_sub[alert_var] == 1.0, np.nan)
            alt = alt.values
            
            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], chg[i], alt[i]])
            
            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')
            
            # construct and relay static change line chart html
            data = "data.addRows(" + data + ");"
            html = _html_sub_change.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate change history chart.</h3>'

        # add to output 
        arcpy.AddMessage(html)



        # # # # #
        # prepare zone history chart
        
        try:
            # get method name and prepare vars
            method = params[12].lower()
            zone_var = '{}_zones'.format(method)
            alert_var = '{}_alerts'.format(method)
            
            # get zones where alerts exist 
            zne = ds_sub[zone_var].where(ds_sub[alert_var] == 1.0, 0.0)
            zne = zne.values

            # prepare data statement and header row 
            data_block = "var data = google.visualization.arrayToDataTable(["
            data_block += "['Date', 'Zone', {role: 'style'}],"
            
            # set zone colours 
            cmap = {
                -12: "black",
                -11: "#FF7F7F",
                -10: "#FFA77F",
                -9:  "#FFD37F",
                -8:  "#FFFF73",
                -7:  "#D1FF73",
                -6:  "#A3FF73",
                -5:  "#73FFDF",
                -4:  "#73DFFF",
                -3:  "#73B2FF",
                -2:  "#DF73FF",
                -1:  "#FF73DF",
                0:   "#FFFFFF",
                1:   "#FF73DF",
                2:   "#DF73FF",
                3:   "#73B2FF",
                4:   "#73DFFF",
                5:   "#73FFDF",
                6:   "#A3FF73",
                7:   "#D1FF73",
                8:   "#FFFF73",
                9:   "#FFD37F",
                10:  "#FFA77F",
                11:  "#FF7F7F",
                12:  "black"
                }

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], zne[i], cmap.get(zne[i])])
            
            # construct string data array
            data = ','.join([str(s) for s in data])
            
            # replace nan with null, if exists
            data = data.replace('nan', 'null')

            # finalise block
            data_block += data + "]);"
            
            # # prepare data
            html = _html_sub_zone.replace('//data.addRows', data_block)
        except:
            html = '<h3>Could not generate zone history chart.</h3>'

        # add to output 
        arcpy.AddMessage(html)
        
        # finally, add static legend chart 
        arcpy.AddMessage(_html_legend)

        return


class Tool(object):

    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Tool"
        self.description = "Tool Template"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        params = None
        return params

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
        """The source code of the tool."""
                    
        return
