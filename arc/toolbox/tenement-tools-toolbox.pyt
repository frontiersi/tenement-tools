# -*- coding: utf-8 -*-# https://pro.arcgis.com/en/pro-app/latest/arcpy/geoprocessing_and_python/a-quick-tour-of-python-toolboxes.htm# imports todo - fix this up properlyimport os, certifios.environ['GDAL_DATA']  = r'C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata'os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())# importsimport arcpyclass Toolbox(object):    def __init__(self):        """Define the toolbox (the name of the toolbox is the name of the        .pyt file)."""           self.label = "Toolbox"        self.alias = "toolbox"        # list of tool classes associated with this toolbox        self.tools = [Tool, COG_Fetch, GDVSpectra]        class Tool(object):    def __init__(self):        """Define the tool (tool name is the name of the class)."""        self.label = "Tool"        self.description = ""        self.canRunInBackground = False    def getParameterInfo(self):        """Define parameter definitions"""        params = None        return params    def isLicensed(self):        """Set whether tool is licensed to execute."""        return True    def updateParameters(self, parameters):        """Modify the values and properties of parameters before internal        validation is performed.  This method is called whenever a parameter        has been changed."""        return    def updateMessages(self, parameters):        """Modify the messages created by internal validation for each tool        parameter.  This method is called after internal validation."""        return    def execute(self, parameters, messages):        """The source code of the tool."""        returnclass COG_Fetch(object):    def __init__(self):        """Define the tool (tool name is the name of the class)."""              self.label = "COG Fetch"        self.description = "COG contains functions that " \                           "allow for efficient download of " \                           "analysis ready data (ARD) Landsat " \                           "5, 7, 8 or Sentinel 2A, 2B images " \                           "from the Digital Earth Australia " \                           "(DEA) public AWS server."        self.canRunInBackground = False    def getParameterInfo(self):        """Define parameter definitions"""                # input study area shapefile (temp)        par_studyarea_feat = arcpy.Parameter(                                displayName="Input study area feature",                                name="in_studyarea_feat",                                datatype="GPFeatureLayer",                                parameterType="Required",                                direction="Input"                                )                                        # set study area to be polygon only        par_studyarea_feat.filter.list = ['Polygon']                                        # output folder location        par_out_folder_path = arcpy.Parameter(                                displayName="Output folder",                                name="out_folder_path",                                datatype="DEWorkspace",                                parameterType="Required",                                direction="Input"                                )        # in_platform        par_platform = arcpy.Parameter(                            displayName="Satellite platform",                            name="in_platform",                            datatype="GPString",                            parameterType="Required",                            direction="Input",                            #category='Satellite query',                            multiValue=False                            )                                    # set default platform        par_platform.values = 'Landsat'        par_platform.filter.list = ['Landsat', 'Sentinel']                # in_from_date        par_date_from = arcpy.Parameter(                            displayName="Date from",                            name="in_from_date",                            datatype="GPDate",                            parameterType="Required",                            direction="Input",                            #category='Satellite query',                            multiValue=False                            )                                    # set in_from_date value        par_date_from.values = '2010/01/01'                # in_to_date        par_date_to = arcpy.Parameter(                        displayName="Date to",                        name="in_to_date",                        datatype="GPDate",                        parameterType="Required",                        direction="Input",                        #category='Satellite query',                        multiValue=False                        )        # set in_from_date value        par_date_to.values = '2021/01/01'        # set bands        par_bands = arcpy.Parameter(displayName="Bands",                                    name="in_bands",                                    datatype="GPString",                                    parameterType="Required",                                    direction="Input",                                    category='Satellite Bands',                                    multiValue=True                                    )                         # in_max_cloud        par_max_cloud = arcpy.Parameter(                            displayName="Maximum Cloud Cover",                            name="in_max_cloud",                            datatype="GPDouble",                            parameterType="Optional",                            direction="Input",                            multiValue=False                            )                                    # set default platform        par_max_cloud.filter.type = 'Range'        par_max_cloud.filter.list = [0.0, 100.0]        par_max_cloud.value = 10.0        # set landsat bands        bands = ['Blue',                  'Green',                  'Red',                  'NIR',                  'SWIR1',                  'SWIR2',                  'OA_Mask'                 ]                # set default bands        par_bands.filter.type = "ValueList"                par_bands.filter.list = bands        par_bands.values = bands                # set slc-off        par_slc_off = arcpy.Parameter(displayName="SLC-Off",                                          name="in_slc_off",                                          datatype="GPBoolean",                                          parameterType="Required",                                          direction="Input",                                          multiValue=False                                          )                # set slc-off value        par_slc_off.value = False                # set oa class values        par_fmask_flags = arcpy.Parameter(displayName="Include Pixels",                                            name="in_fmask_flags",                                            datatype="GPString",                                            parameterType="Required",                                            direction="Input",                                            category='Quality Mask',                                            multiValue=True                                          )                                                                        # set landsat bands        fmask_flags = ['NoData',                        'Valid',                        'Cloud',                        'Shadow',                        'Snow',                        'Water'                       ]                # set default bands        par_fmask_flags.filter.type = "ValueList"                par_fmask_flags.filter.list = fmask_flags        par_fmask_flags.values = ['Valid', 'Snow', 'Water']                # set open as multidim raster values        par_open_as_mdr = arcpy.Parameter(displayName="Load as Multidimensional Raster",                                          name="in_open_as_mdr",                                          datatype="GPBoolean",                                          parameterType="Optional",                                          direction="Input",                                          multiValue=False                                          )                # set multidim raster value        par_open_as_mdr.value = True                # to add        # stac limit         # epsg         # resolution         # snap bounds         # force dea http                # combine parameters        parameters = [            par_studyarea_feat,            par_out_folder_path,            par_platform,            par_date_from,            par_date_to,            par_bands,            par_slc_off,            par_fmask_flags,            par_max_cloud,            par_open_as_mdr        ]                return parameters    def isLicensed(self):        """Set whether tool is licensed to execute."""        return True    def updateParameters(self, parameters):        """Modify the values and properties of parameters before internal        validation is performed.  This method is called whenever a parameter        has been changed."""        return    def updateMessages(self, parameters):        """Modify the messages created by internal validation for each tool        parameter.  This method is called after internal validation."""        return    def execute(self, parameters, messages):        """The source code of the tool."""                # temp! remove future warning on pandas        #import warnings        #warnings.simplefilter(action='ignore', category=FutureWarning)                # imports todo - fix this up properly        import os, certifi        os.environ['GDAL_DATA']  = r'C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata'        os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())                        # imports        import os, sys        import numpy as np        import xarray as xr        import dask        import dask.array as da        import arcpy        # import tools        #sys.path.append('../../../shared') temp for demo        #sys.path.append(r'C:\Users\262272G\Documents\GitHub\tenement-tools\shared')        #sys.path.append(r'C:\Users\Lewis\Documents\GitHub\tenement-tools\shared')        #import satfetcher, tools                # import gdvspectra module        #sys.path.append('../../../modules') temp for demo        #sys.path.append(r'C:\Users\262272G\Documents\GitHub\tenement-tools\modules')        sys.path.append(r'C:\Users\Lewis\Documents\GitHub\tenement-tools\modules')        import cog                # globals         AWS_KEY = ''        AWS_SECRET = ''        STAC_ENDPOINT = 'https://explorer.sandbox.dea.ga.gov.au/stac/search'                # progress bar        arcpy.SetProgressor(type='step', message='Initialising...', min_range=0, max_range=5)        arcpy.SetProgressorLabel('Preparing input parameters...')        arcpy.SetProgressorPosition(0)                            # notify user        arcpy.AddMessage('Preparing input parameters.')                        # get study area feature        in_studyarea_feat = parameters[0].value        arcpy.AddMessage('Recieved study area file: {0}.'.format(in_studyarea_feat))        # get output folder        out_folder_path = parameters[1].valueAsText        arcpy.AddMessage('Recieved output folder path: {0}.'.format(out_folder_path))               # get platform        in_platform = parameters[2].value        arcpy.AddMessage('Recieved platform type: {0}.'.format(in_platform))                # get in_from_date        in_from_date = parameters[3].value        arcpy.AddMessage('Recieved from date: {0}.'.format(in_from_date))                # get in_to_date        in_to_date = parameters[4].value        arcpy.AddMessage('Recieved to date: {0}.'.format(in_to_date))                # get bands        in_bands = parameters[5].value        arcpy.AddMessage('Recieved bands: {0}.'.format(in_bands))                # get slc-off        in_slc_off = parameters[6].value        arcpy.AddMessage('Recieved slc-off: {0}.'.format(in_slc_off))              # get fmask flags        in_fmask_flags = parameters[7].value        arcpy.AddMessage('Recieved fmask flags: {0}.'.format(in_fmask_flags))                    # get max cloud        in_max_cloud = parameters[8].value        arcpy.AddMessage('Recieved max cloud: {0}.'.format(in_max_cloud))                # get zscore p-value        in_open_as_mdr = parameters[9].value        arcpy.AddMessage('Recieved open as multidim raster: {0}.'.format(in_open_as_mdr))                # notify user        arcpy.AddMessage('Initialising COG tool.')                def get_shp_extent_via_arcpy(shp_path):            # get shapefile path            #shp_path = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\data\arc\yandi_study_area.shp'            # check if exists            if not arcpy.Exists(shp_path):                raise ValueError('Selected shapefile does not exist.')            # check if feature is polygon or multipolygon            desc = arcpy.Describe(shp_path)            if not desc.shapeType == 'Polygon':                raise TypeError('Selected shapefile must be a polygon type.')            # check if feature has a spatial system            if not desc.hasSpatialIndex:                raise ValueError('Selected shapefile must have a coordinate system.')            # project extent to wgs84            srs = arcpy.SpatialReference(4326)            prj_extent = desc.extent.projectAs(srs)            # get extent            l, r = prj_extent.XMin, prj_extent.XMax            b, t = prj_extent.YMin, prj_extent.YMax            return [l, b, r, t]                # get bounding box of study area         # todo        #bbox = [            #118.92837524414061,            #-22.816061209792938,            #119.16526794433592,            #-22.68118293381927            #]                   shp_path = in_studyarea_feat.dataSource        shp_path = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\data\arc\yandi_study_area_wgs84.shp'        bbox = get_shp_extent_via_arcpy(shp_path)                arcpy.AddMessage(bbox)                # get collections based on platform         in_platform = in_platform.lower()        if in_platform == 'landsat':            collections = [            'ga_ls5t_ard_3',             'ga_ls7e_ard_3',            'ga_ls8c_ard_3'            ]                    # exclude slc        slc_off = False                    # convert start, end dates        start_dt, end_dt = '1990-01-01', '1995-12-31'                # set search limits        limit = 200                # increase progress        arcpy.SetProgressorLabel('Querying STAC...')        arcpy.SetProgressorPosition(1)                # test        import io         stdout = sys.stdout         sys.stdout = io.StringIO()                # fetch stac data         arcpy.AddMessage('Querying STAC data. Please wait.')        feats = cog.fetch_stac_data(stac_endpoint=STAC_ENDPOINT,                                     collections=collections,                                     start_dt=start_dt,                                     end_dt=end_dt,                                     bbox=bbox,                                    slc_off=slc_off,                                    limit=limit)                                            output = sys.stdout.getvalue()        sys.stdout = stdout        arcpy.AddMessage(output)                # increase progress        arcpy.SetProgressorLabel('Converting STAC data...')        arcpy.SetProgressorPosition(2)                # count number of items        arcpy.AddMessage('Found {} {} scenes.'.format(in_platform, len(feats)))                # todo convert bands to dea friendly        assets = [            'nbart_blue',             'nbart_green',             'nbart_red',             'nbart_nir',            'nbart_swir_1',            'nbart_swir_2',            'oa_fmask'            ]                    # set resampling algorithm        resampling = 'nearest'        # set output dtype        out_dtype = 'int16'        # set out resolution        out_resolution = 30        # set fill value        out_fill_value = -999        # output epsg        out_epsg = 3577        # snap boundary coords        snap_bounds = True        # set output rescale        out_rescale = True        # chunk size        chunk_size=-1 #512        # set pixel location for cell alignment        out_pixel_loc = 'topleft'                            # convert raw stac into dict with coord reproject, etc.        arcpy.AddMessage('Preparing raw STAC data into useable format.')        meta, asset_table = cog.prepare_data(feats,                                              assets=assets,                                             bounds_latlon=bbox,                                              bounds=None,                                              epsg=out_epsg,                                              resolution=out_resolution,                                              snap_bounds=snap_bounds,                                             force_dea_http=True)                                                     # increase progress        arcpy.SetProgressorLabel('Setting up Dask array...')        arcpy.SetProgressorPosition(3)                                                     # convert assets to dask array. todo: make this yours        arcpy.AddMessage('Converting asset data to dask array.')        darray = cog.convert_to_dask(meta=meta,                                      asset_table=asset_table,                                      chunksize=chunk_size,                                     resampling=resampling,                                      dtype=out_dtype,                                      fill_value=out_fill_value,                                      rescale=out_rescale)                                         # generate coordinates and dimensions from metadata        coords, dims = cog.build_coords(feats=feats,                                        assets=assets,                                         meta=meta,                                        pix_loc=out_pixel_loc)        # generate attributes        attrs = cog.build_attributes(meta=meta,                                      fill_value=out_fill_value,                                      collections=collections,                                      slc_off=slc_off,                                      bbox=bbox)        # build final xarray data array        ds = xr.DataArray(darray,                          coords=coords,                          dims=dims,                          attrs=attrs,                          name='stac-' + dask.base.tokenize(da)                         )                                 # comvert to cleaner xarray dataset        ds = ds.to_dataset(dim='band')                arcpy.AddMessage(ds)                                         # increase progress        arcpy.SetProgressorLabel('Computing into memory... This can take awhile...')        arcpy.SetProgressorPosition(4)                                                 # todo: keep going        arcpy.AddMessage('TEMP: attempting to compute array.')        #import time        #start = time.time()        #arcpy.SetProgressor(type='default')        # save and re-open        #from dask.diagnostics import ProgressBar        #arcpy.AddMessage(ProgressBar().register())        #fn = r'C:\Users\Lewis\Desktop\out_nc\test2.nc'        #ds.to_netcdf(fn)        ds = ds.compute()         arcpy.AddMessage(ds)        #end = time.time()        #elapsed = end - start        #arcpy.AddMessage(elapsed)                        # increase progress        #arcpy.SetProgressorLabel('Step 5')        #arcpy.SetProgressorPosition(5)                # progress bar        #arcpy.SetProgressor(type='step', message='Initialising...', min_range=0, max_range=5)        #arcpy.SetProgressorLabel('Finished!')        #arcpy.SetProgressorPosition(5)                returnclass GDVSpectra(object):    def __init__(self):        """Define the tool (tool name is the name of the class)."""        self.label = "GDVSpectra"        self.description = "GDVSpectra contains functions that derive " \                           "potential groundwater dependent vegetation from " \                           "a time series of Landsat or Sentinel data."        self.canRunInBackground = False    def getParameterInfo(self):        """Define parameter definitions"""                # input netcdf file (temp)        par_nc_path = arcpy.Parameter(                        displayName="Input NetCDF file",                        name="in_nc_path",                        datatype="DEFile",                        parameterType="Required",                        direction="Input"                        )        # set file type to be netcdf only        par_nc_path.filter.list = ['nc']                # input study area shapefile (temp)        par_studyarea_feat = arcpy.Parameter(                                displayName="Input study area feature",                                name="in_studyarea_feat",                                datatype="GPFeatureLayer",                                parameterType="Required",                                direction="Input"                                )                                        # set study area to be polygon only        #par_studyarea_feat.filter.list = ['Polygon']                         # output folder location        par_out_folder_path = arcpy.Parameter(                                displayName="Output folder",                                name="out_folder_path",                                datatype="DEWorkspace",                                parameterType="Required",                                direction="Input"                                )        # in_platform        par_platform = arcpy.Parameter(                            displayName="Satellite platform",                            name="in_platform",                            datatype="GPString",                            parameterType="Required",                            direction="Input",                            category='Satellite query',                            multiValue=False                            )                                    # set default platform        par_platform.values = 'landsat'        par_platform.filter.list = ['landsat', 'sentinel']        # in_from_date        par_date_from = arcpy.Parameter(                            displayName="Date from",                            name="in_from_date",                            datatype="GPDate",                            parameterType="Required",                            direction="Input",                            category='Satellite query',                            multiValue=False                            )                                    # set in_from_date value        par_date_from.values = '2010/01/01'                # in_to_date        par_date_to = arcpy.Parameter(                        displayName="Date to",                        name="in_to_date",                        datatype="GPDate",                        parameterType="Required",                        direction="Input",                        category='Satellite query',                        multiValue=False                        )        # set in_from_date value        par_date_to.values = '2021/01/01'        # in_wet_months        par_wet_months = arcpy.Parameter(                            displayName="Wet month(s)",                            name="in_wet_months",                            datatype="GPString",                            parameterType="Required",                            direction="Input",                            multiValue=False                            )                                    # set default wet months        par_wet_months.values = '1, 2, 3'                # in_dry_months        par_dry_months = arcpy.Parameter(                            displayName="Dry month(s)",                            name="in_dry_months",                            datatype="GPString",                            parameterType="Required",                            direction="Input",                            multiValue=False                            )                                    # set default dry months        par_dry_months.values = '9, 10, 11'                # in_veg_idx        par_veg_idx = arcpy.Parameter(                        displayName="Vegetation index",                        name="in_veg_idx",                        datatype="GPString",                        parameterType="Required",                        direction="Input",                        multiValue=False                        )                                    # set default veg idx        par_veg_idx.value = 'mavi'        par_veg_idx.filter.type = 'ValueList'        par_veg_idx.filter.list = [            'ndvi',            'evi',             'savi',            'msavi',            'slavi',            'mavi',            'kndvi',            'tcg',            'tcb',            'tcw'            ]                # in_mst_idx        par_mst_idx = arcpy.Parameter(                        displayName="Moisture index",                        name="in_mst_idx",                        datatype="GPString",                        parameterType="Required",                        direction="Input",                        multiValue=False)                                    # set default mst idx        par_mst_idx.value = 'ndmi'        par_mst_idx.filter.type = 'ValueList'        par_mst_idx.filter.list = [            'ndmi',            'gvmi'            ]                    # set pvalue for zscore        par_zscore_pvalue = arcpy.Parameter(                                displayName="Z-score p-value",                                name="in_zscore_pvalue",                                datatype="GPDouble",                                parameterType="Optional",                                direction="Input",                                category='Outlier correction',                                multiValue=False)                                    # set default mst idx        par_zscore_pvalue.value = None                # set q upper for standardisation        par_ivt_qupper = arcpy.Parameter(                                displayName="Upper percentile",                                name="in_stand_qupper",                                datatype="GPDouble",                                parameterType="Optional",                                direction="Input",                                category='Standardisation',                                multiValue=False)                                    # set default mst idx        par_ivt_qupper.value = 0.99                # set q lower for standardisation        par_ivt_qlower = arcpy.Parameter(                                displayName="Lower percentile",                                name="in_stand_qlower",                                datatype="GPDouble",                                parameterType="Optional",                                direction="Input",                                category='Standardisation',                                multiValue=False)                                    # set default mst idx        par_ivt_qlower.value = 0.05        # combine parameters        parameters = [            par_nc_path,            par_studyarea_feat,            par_out_folder_path,            par_platform,            par_date_from,            par_date_to,            par_wet_months,             par_dry_months,             par_veg_idx,             par_mst_idx,             par_zscore_pvalue,            par_ivt_qupper,            par_ivt_qlower            ]                return parameters    def isLicensed(self):        """Set whether tool is licensed to execute."""        return True    def updateParameters(self, parameters):        """Modify the values and properties of parameters before internal        validation is performed.  This method is called whenever a parameter        has been changed."""                        return    def updateMessages(self, parameters):        """Modify the messages created by internal validation for each tool        parameter.  This method is called after internal validation."""        return    def execute(self, parameters, messages):        """The source code of the tool."""                # temp! remove future warning on pandas        import warnings        warnings.simplefilter(action='ignore', category=FutureWarning)                # imports        import os, sys        import numpy as np        import pandas as pd        import xarray as xr                # import tools        #sys.path.append('../../../shared') temp for demo        sys.path.append(r'C:\Users\262272G\Documents\GitHub\tenement-tools\shared')        sys.path.append(r'C:\Users\Lewis\Documents\GitHub\tenement-tools\shared')        import satfetcher, tools                # import gdvspectra module        #sys.path.append('../../../modules') temp for demo        sys.path.append(r'C:\Users\262272G\Documents\GitHub\tenement-tools\modules')        sys.path.append(r'C:\Users\Lewis\Documents\GitHub\tenement-tools\modules')        import gdvspectra                # notify user        arcpy.AddMessage('Preparing input parameters.')        # get netcdf file        in_nc_path = parameters[0].valueAsText        arcpy.AddMessage('Recieved NetCDF file: {0}.'.format(in_nc_path))                        # get study area feature        in_studyarea_feat = parameters[1].value        arcpy.AddMessage('Recieved study area file: {0}.'.format(in_studyarea_feat))                # get output folder        out_folder_path = parameters[2].valueAsText        arcpy.AddMessage('Recieved output folder path: {0}.'.format(out_folder_path))               # get platform        in_platform = parameters[3].value        arcpy.AddMessage('Recieved platform type: {0}.'.format(in_platform))                # get in_from_date        in_from_date = parameters[4].value        arcpy.AddMessage('Recieved from date: {0}.'.format(in_from_date))                # get in_to_date        in_to_date = parameters[5].value        arcpy.AddMessage('Recieved to date: {0}.'.format(in_to_date))                # clean up wet months         wet_month = parameters[6].value        if wet_month:            wet_month = [int(e) for e in wet_month.split(',')]                # clean up dry months         dry_month = parameters[7].value        if dry_month:            dry_month = [int(e) for e in dry_month.split(',')]                    # get veg_idx, mst_idx from parameters         veg_idx, mst_idx = parameters[8].value, parameters[9].value                # get zscore p-value        zscore_pvalue = parameters[10].value # fix                # get interpolation method         # todo full or half                # get q upper, q lower for invariant target         # todo                # set veg mask quantile for similairity        # todo                                        # notify user        arcpy.AddMessage('Initialising GDVSpectra tool.')        # load netcdf        arcpy.AddMessage('Loading provided NetCDF file.')        ds = satfetcher.load_local_nc(nc_path=in_nc_path,                                       use_dask=False,                                       conform_nodata_to=-9999)                  # conform band names from dea to basic        arcpy.AddMessage('Conforming band names.')        ds = satfetcher.conform_dea_ard_band_names(ds=ds,                                                    platform=in_platform)                # notify user        #arcpy.AddMessage('Creating a backup dataset for later.')                # make copy for ds for later cva work        #ds_backup = ds.copy(deep=True)                # get subset fo data for wet and dry season months        arcpy.AddMessage('Reducing dataset down to wet, dry months.')        ds = gdvspectra.subset_months(ds=ds,                                       month=wet_month + dry_month,                                      inplace=True)                # calculate vegetation and moisture indices        arcpy.AddMessage('Generating vege/moist indices: {0}'.format(veg_idx, mst_idx))        ds = tools.calculate_indices(ds=ds,                                      index=['mavi', 'ndmi'],                                      custom_name=['veg_idx', 'mst_idx'],                                      rescale=True,                                      drop=True)        # perform resampling        arcpy.AddMessage('Resampling dataset to annual wet/dry medians.')        ds = gdvspectra.resample_to_wet_dry_medians(ds=ds,                                                     wet_month=wet_month,                                                     dry_month=dry_month,                                                    inplace=True)        # persist memoru        #ds = ds.persist()                # perform outlier removal        if zscore_pvalue:            arcpy.AddMessage('Removing outliers via Z-Score.')            ds = gdvspectra.nullify_wet_dry_outliers(ds=ds,                                                      wet_month=wet_month,                                                      dry_month=dry_month,                                                      p_value=0.01, # todo link this to param                                                     inplace=True)        # remove any years missing wet, dry season         arcpy.AddMessage('Removing any years missing wet, dry seasons.')        ds = gdvspectra.drop_incomplete_wet_dry_years(ds=ds)        # fill any empty first, last years using back/forward fill        arcpy.AddMessage('Filling any empty first and last years.')        # temp! todo for demo        #ds = gdvspectra.fill_empty_wet_dry_edges(ds=ds,                                                 #wet_month=wet_month,                                                  #dry_month=dry_month,                                                 #inplace=True)                                                         # interpolate missing values         arcpy.AddMessage('Interpolating missing values for wet, dry seasons.')        # temp! todo for demo        #ds = gdvspectra.interp_empty_wet_dry(ds=ds,                                             #wet_month=wet_month,                                             #dry_month=dry_month,                                             #method='full', #todo link to param                                             #inplace=True)                                                     # standardise data to invariant targets derived from dry times        arcpy.AddMessage('Standardising data to dry season invariant targets.')        ds = gdvspectra.standardise_to_dry_targets(ds=ds,                                                    dry_month=dry_month,                                                    q_upper=0.99, # todo link to params                                                   q_lower=0.05, # todo link to params                                                   inplace=True)                                                           # calculate seasonal similarity        arcpy.AddMessage('Calculating seasonal similarity.')        ds_similarity = gdvspectra.calc_seasonal_similarity(ds=ds,                                                            wet_month=wet_month,                                                            dry_month=dry_month,                                                            q_mask=0.9, # todo link to params                                                            inplace=True)                                                                    # calculate gdv likelihood        arcpy.AddMessage('Calculating groundwater dependent vegetation likelihood.')        ds = gdvspectra.calc_likelihood(ds=ds,                                         ds_similarity=ds_similarity,                                        wet_month=wet_month,                                         dry_month=dry_month)                # todo - temp - get median all time         ds = ds.median('time')                                                                         # export likelihood as netcdf         out_path_file = os.path.join(out_folder_path, 'ds_like.nc')        arcpy.AddMessage('Exporting GDV likelihood as netcdf to: {0}.'.format(out_path_file))        tools.export_xr_as_nc(ds=ds, filename=out_path_file)                # notify user        arcpy.AddMessage('Preparing data for map.')                                      # get current project and map todo handle exception        aprx = arcpy.mp.ArcGISProject("CURRENT")        m = aprx.listMaps("Map")[0]                # temp halt adding to map for now        arcpy.env.addOutputsToMap = False                # todo - temp - show a layer        nc_layer = arcpy.MakeNetCDFRasterLayer_md(in_netCDF_file=out_path_file,                                                   variable='like',                                                   x_dimension='x',                                                   y_dimension='y',                                                  out_raster_layer='like',                                                   band_dimension='',                                                   dimension_values='',                                                   value_selection_method='')                                                           # create layer        out_path_tif = os.path.join(out_folder_path, 'like.tif')        tif_layer = arcpy.CopyRaster_management(in_raster=nc_layer,                                                 out_rasterdataset=out_path_tif)        # turn on add to map        arcpy.env.addOutputsToMap = True        # define projection in albers        tif_layer = arcpy.management.DefineProjection(in_dataset=tif_layer,                                                       coor_system=arcpy.SpatialReference(3577))        # refresh toc and map        m.addDataFromPath(tif_layer)                                                      # close xarrays, arcpy objects        del ds, ds_similarity #, ds_backup        del nc_layer, aprx, m         # notify user        arcpy.AddMessage('Generated GDV likelihood successfully.')                return