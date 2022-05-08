
# deprecated
class COG_Fetch(object):
    def __init__(self):
        """
        Initialise tool.
        """
        
        # set tool name, description, options
        self.label = "COG Fetch"
        self.description = "COG contains functions that " \
                           "allow for efficient download of " \
                           "analysis ready data (ARD) Landsat " \
                           "5, 7, 8 or Sentinel 2A, 2B images " \
                           "from the Digital Earth Australia " \
                           "(DEA) public AWS server."
        self.canRunInBackground = False

    def getParameterInfo(self):
        
        # input study area shapefile
        par_studyarea_feat = arcpy.Parameter(
                                displayName="Input study area feature",
                                name="in_studyarea_feat",
                                datatype="GPFeatureLayer",
                                parameterType="Required",
                                direction="Input"
                                )
                                
        # set study area to be polygon only
        par_studyarea_feat.filter.list = ['Polygon']
                                
        # output file location
        par_out_nc_path = arcpy.Parameter(
                                displayName="Output NetCDF file",
                                name="out_nc_path",
                                datatype="DEFile",
                                parameterType="Required",
                                direction="Output"
                                )
                                
        # set file type to be netcdf only
        par_out_nc_path.filter.list = ['nc']

        # in_platform
        par_platform = arcpy.Parameter(
                            displayName="Satellite platform",
                            name="in_platform",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input",
                            multiValue=False
                            )
                            
        # set default platform
        par_platform.values = 'Landsat'
        par_platform.filter.list = ['Landsat', 'Sentinel']  # 'Sentinel 2A', 'Sentinel 2B'
        
        # in_from_date
        par_date_from = arcpy.Parameter(
                            displayName="Date from",
                            name="in_from_date",
                            datatype="GPDate",
                            parameterType="Required",
                            direction="Input",
                            multiValue=False
                            )
                            
        # set in_from_date value
        par_date_from.values = '2015/01/01'
        
        # in_to_date
        par_date_to = arcpy.Parameter(
                        displayName="Date to",
                        name="in_to_date",
                        datatype="GPDate",
                        parameterType="Required",
                        direction="Input",
                        multiValue=False
                        )

        # set in_from_date value
        par_date_to.values = '2020/12/31'

        # set bands
        par_bands = arcpy.Parameter(
                        displayName="Bands",
                        name="in_bands",
                        datatype="GPString",
                        parameterType="Required",
                        direction="Input",
                        category='Satellite Bands',
                        multiValue=True
                        )
                 
        # set landsat bands
        bands = [
            'Blue', 
            'Green', 
            'Red', 
            'NIR', 
            'SWIR1', 
            'SWIR2', 
            'OA_Mask'
            ]
        
        # set default bands
        par_bands.filter.type = "ValueList"        
        par_bands.filter.list = bands
        par_bands.values = bands
        
        # set slc-off
        par_slc_off = arcpy.Parameter(
                        displayName="SLC Off",
                        name="in_slc_off",
                        datatype="GPBoolean",
                        parameterType="Required",
                        direction="Input",
                        multiValue=False
                        )
        
        # set slc-off value
        par_slc_off.value = False
               
        # set output datatype
        par_output_dtype = arcpy.Parameter(
                            displayName="Output data type",
                            name="in_output_dtype",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
                            
        # set default platform
        par_output_dtype.filter.list = ['int8', 'int16', 'float32', 'float64']
        par_output_dtype.values = 'int16'
        
        # todo make this changeh when sent/landsat changed
        # set output resolution
        par_output_res = arcpy.Parameter(
                            displayName="Output pixel resolution",
                            name="in_output_res",
                            datatype="GPLong",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
                            
        # set default platform
        par_output_res.filter.type = 'Range'
        par_output_res.filter.list = [0, 1000]
        par_output_res.value = 30
 
        # todo allow this to handle np.nan
        # set output nodata value
        par_output_fill_value = arcpy.Parameter(
                                    displayName="Output NoData value",
                                    name="in_output_fill_value",
                                    datatype="GPString",
                                    parameterType="Required",
                                    direction="Input",
                                    category='Warping Options',
                                    multiValue=False
                                    )
                            
        # set default nodata value
        par_output_fill_value.value = "-999"
        
        # set output epsg
        par_output_epsg = arcpy.Parameter(
                            displayName="Output EPSG",
                            name="in_output_epsg",
                            datatype="GPLong",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
                            
        # set default epsg
        par_output_epsg.filter.list = [3577]
        par_output_epsg.values = 3577
        
        # set resampling type
        par_output_resampling = arcpy.Parameter(
                            displayName="Resampling type",
                            name="in_output_resampling",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
                            
        # set default resampling
        par_output_resampling.filter.list = ['Nearest', 'Bilinear']
        par_output_resampling.values = 'Nearest'
        
        # set snap boundary
        par_output_snap = arcpy.Parameter(
                            displayName="Snap boundaries",
                            name="in_snap_bounds",
                            datatype="GPBoolean",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
        
        # set snap boundary value
        par_output_snap.value = True
        
        # set rescale
        par_output_rescale = arcpy.Parameter(
                        displayName="Rescale",
                        name="in_rescale",
                        datatype="GPBoolean",
                        parameterType="Required",
                        direction="Input",
                        category='Warping Options',
                        multiValue=False
                        )
        
        # set rescale value
        par_output_rescale.value = True
        
        # set cell alignment
        par_output_cell_align = arcpy.Parameter(
                            displayName="Cell alignment",
                            name="in_output_cell_align",
                            datatype="GPString",
                            parameterType="Required",
                            direction="Input",
                            category='Warping Options',
                            multiValue=False
                            )
                            
        # set default cell align
        par_output_cell_align.filter.list = ['Top-left', 'Center']
        par_output_cell_align.values = 'Top-left'
        
        # set chunks
        par_output_chunk_size = arcpy.Parameter(
                            displayName="Chunk size",
                            name="in_output_chunk_size",
                            datatype="GPLong",
                            parameterType="Required",
                            direction="Input",
                            category='Parallelisation',
                            multiValue=False
                            )
                            
        # set default chunksize
        par_output_chunk_size.value = -1
        
        # set dea aws stac url
        par_output_stac_url = arcpy.Parameter(
                                displayName="Digital Earth Australia STAC URL",
                                name="in_output_stac_url",
                                datatype="GPString",
                                parameterType="Required",
                                direction="Input",
                                category='STAC Options',
                                multiValue=False
                                )
        
         # set default dea aws stac url
        par_output_stac_url.value = STAC_ENDPOINT

        # set dea aws key
        par_output_aws_key = arcpy.Parameter(
                                displayName="Digital Earth Australia AWS Key",
                                name="in_output_aws_key",
                                datatype="GPString",
                                parameterType="Optional",
                                direction="Input",
                                category='STAC Options',
                                multiValue=False
                                )
        
         # set default dea aws key value
        par_output_aws_key.value = AWS_KEY 

        # set dea aws secret
        par_output_aws_secret = arcpy.Parameter(
                                displayName="Digital Earth Australia AWS Secret Key",
                                name="in_output_aws_secret",
                                datatype="GPString",
                                parameterType="Optional",
                                direction="Input",
                                category='STAC Options',
                                multiValue=False
                                )
        
         # set default dea aws secret value
        par_output_aws_secret.value = AWS_SECRET
        
        # combine parameters
        parameters = [
            par_studyarea_feat,
            par_out_nc_path,
            par_platform,
            par_date_from,
            par_date_to,
            par_bands,
            par_slc_off,
            par_output_dtype,
            par_output_res,
            par_output_fill_value,
            par_output_epsg,
            par_output_resampling,
            par_output_snap,
            par_output_rescale,
            par_output_cell_align,
            par_output_chunk_size,
            par_output_stac_url,
            par_output_aws_key,
            par_output_aws_secret
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
        
        # modify bands list when platform changed
        if parameters[2].value == 'Landsat' and not parameters[2].hasBeenValidated:
        
            # enable slc-off control
            parameters[6].enabled = True
            
            # update landsat band list
            bands = [
                'Blue', 
                'Green', 
                'Red', 
                'NIR', 
                'SWIR1', 
                'SWIR2', 
                'OA_Mask'
                ]
            
            # update bands and set to default resolution
            parameters[5].filter.list = bands
            parameters[5].values = bands
            parameters[8].value = 30

        elif 'Sentinel' in parameters[2].value and not parameters[2].hasBeenValidated:
        
            # disable slc-off control
            parameters[6].enabled = False
            
            # update sentinel band list
            bands = [
                'Blue', 
                'Green', 
                'Red', 
                'NIR1', 
                'SWIR2', 
                'SWIR3', 
                'OA_Mask'
                ]
            
            # update values in control
            parameters[5].filter.list = bands
            parameters[5].values = bands
            
            # set resolution to original 10x10m
            parameters[8].value = 10

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the GDV Spectra Likelihood module.
        """
        
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
                        
        # safe imports
        import os, sys
        import io
        import time
        import numpy as np
        import arcpy
        
        # risky imports (not native to arcgis)
        try:
            import xarray as xr
            import dask
            import dask.array as da
        except:
            arcpy.AddError('Python libraries xarray, dask not installed.')
            return

        # import tools
        try:
            # shared folder
            sys.path.append(FOLDER_SHARED)
            import arc, tools, satfetcher
        
            # module folder
            sys.path.append(FOLDER_MODULES)
            import cog
        except:
            arcpy.AddError('Could not find tenement tools python scripts (modules, shared).')
            return
        
        # notify 
        arcpy.AddMessage('Performing COG Fetch.')
                                            
        # grab parameter values 
        in_studyarea_feat = parameters[0].value         # study area feat 
        out_nc = parameters[1].valueAsText              # output nc 
        in_platform = parameters[2].value               # platform name
        in_from_date = parameters[3].value              # from date
        in_to_date = parameters[4].value                # to date
        in_bands = parameters[5].valueAsText            # bands
        in_slc_off = parameters[6].value                # slc off 
        in_dtype = parameters[7].value                  # output pixel dtype
        in_res = parameters[8].value                    # output pixel resolution
        in_fill_value = parameters[9].value             # todo processing string to int, float or np.nan
        in_epsg = parameters[10].value                  # output epsg 
        in_resampling = parameters[11].value            # output resampler method 
        in_snap = parameters[12].value                  # output snap alignment 
        in_rescale = parameters[13].value               # output rescale
        in_cell_align = parameters[14].value            # output cell alignmnent     
        in_chunk_size = parameters[15].value            # chunk size
        in_stac_endpoint = parameters[16].value         # stac endpoint
        in_aws_key = parameters[17].value               # dea aws key
        in_aws_secret = parameters[18].value            # dea aws secret
        
        # let user know that aws key and secret not yet implemented
        if in_aws_key is not None or in_aws_secret is not None:
            arcpy.AddWarning('AWS Credentials not yet supported. Using DEA AWS.')
        
        # set up progess bar
        arcpy.SetProgressor(type='default', message='Preparing query parameters...')
                
        # get minimum bounding geom from input 
        bbox = arc.get_selected_layer_extent(in_studyarea_feat)
        
        # get collections based on platform 
        collections = arc.prepare_collections_list(in_platform)
            
        # prepare start and end date times
        in_from_date = arc.datetime_to_string(in_from_date)
        in_to_date = arc.datetime_to_string(in_to_date)
        
        # fetch stac data 
        arcpy.SetProgressorLabel('Performing STAC query...')
        feats = cog.fetch_stac_data(stac_endpoint=in_stac_endpoint, 
                                    collections=collections, 
                                    start_dt=in_from_date, 
                                    end_dt=in_to_date, 
                                    bbox=bbox,
                                    slc_off=in_slc_off,
                                    limit=RESULT_LIMIT)
        
        # count number of items
        arcpy.AddMessage('Found {} {} scenes.'.format(len(feats), in_platform))

        # prepare band (i.e. stac assets) names
        assets = arc.prepare_band_names(in_bands=in_bands, 
                                        in_platform=in_platform)
                                                    
        # convert raw stac into dict with coord reproject, etc.
        arcpy.SetProgressorLabel('Converting STAC data into useable format...')
        meta, asset_table = cog.prepare_data(feats, 
                                             assets=assets,
                                             bounds_latlon=bbox, 
                                             bounds=None, 
                                             epsg=in_epsg, 
                                             resolution=in_res, 
                                             snap_bounds=in_snap,
                                             force_dea_http=True)
                                             
        # prepare resample and fill value types
        resampling = in_resampling.lower()
        fill_value = arc.prepare_fill_value_type(in_fill_value)
                                                                                          
        # convert assets to dask array
        arcpy.SetProgressorLabel('Parallelising data...')
        darray = cog.convert_to_dask(meta=meta, 
                                     asset_table=asset_table, 
                                     chunksize=in_chunk_size,
                                     resampling=resampling, 
                                     dtype=in_dtype, 
                                     fill_value=fill_value, 
                                     rescale=in_rescale)
                                     
        # prepare alignment type
        cell_align = arc.prepare_cell_align_type(in_cell_align)

        # generate coordinates and dimensions from metadata
        arcpy.SetProgressorLabel('Building dataset metadata...')
        coords, dims = cog.build_coords(feats=feats,
                                        assets=assets, 
                                        meta=meta,
                                        pix_loc=cell_align)
        
        # build final xarray data array
        arcpy.SetProgressorLabel('Finalising dataset...')
        ds_name = 'stac-' + dask.base.tokenize(darray)
        ds = xr.DataArray(darray,
                          coords=coords,
                          dims=dims,
                          name=ds_name
                          )
                         
        # comvert to cleaner xarray dataset
        ds = ds.to_dataset(dim='band')
        
        # append attributes onto dataset
        ds = cog.build_attributes(ds=ds,
                                  meta=meta, 
                                  collections=collections, 
                                  bands=assets,
                                  slc_off=in_slc_off, 
                                  bbox=bbox,
                                  dtype=in_dtype,
                                  snap_bounds=in_snap,
                                  fill_value=fill_value, 
                                  rescale=in_rescale,
                                  cell_align=in_cell_align,
                                  resampling=in_resampling)
                                     
        # set up proper progress bar
        arcpy.SetProgressor(type='step', 
                            message='Preparing data download...', 
                            min_range=0, 
                            max_range=len(ds.data_vars) + 1)

        # get list of dataset vars and iterate compute on each
        for counter, data_var in enumerate(list(ds.data_vars), start=1):
        
            # start clock
            start = time.time()
        
            # update progress bar
            arcpy.SetProgressorLabel('Downloading band: {}...'.format(data_var))
            arcpy.SetProgressorPosition(counter)
        
            # compute!
            ds[data_var] = ds[data_var].compute()
            
            # notify time 
            duration = round(time.time() - start, 2)
            arcpy.AddMessage('Band: {} took: {}s to download.'.format(data_var, duration))                          
        
        # wrap up 
        arcpy.SetProgressorLabel('Exporting NetCDF...')
        arcpy.SetProgressorPosition(counter + 1)
                
        # export netcdf to output folder
        tools.export_xr_as_nc(ds=ds, filename=out_nc)

        # notify finish
        arcpy.AddMessage('COG Fetch completed successfully.')
        
        return


# deprecated
class COG_Sync(object):
    def __init__(self):
    
        # set tool name
        self.label = "COG Sync"
        
        # set tool description
        self.description = "Sync COG to update cube with latest " \
                           "data."
                           
        # set false for pro
        self.canRunInBackground = False

    def getParameterInfo(self):
    
        # input netcdf file
        par_nc_file = arcpy.Parameter(
                        displayName="Input NetCDF file",
                        name="in_nc_file",
                        datatype="DEFile",
                        parameterType="Required",
                        direction="Input"
                        )
                                
        # set options
        par_nc_file.filter.list = ['nc']
        
        # combine parameters
        parameters = [
            par_nc_file
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
        
        # imports
        import os, sys
        #import io
        #import time
        import pandas as pd
        import numpy as np
        import xarray as xr
        import arcpy

        # import tools
        sys.path.append(FOLDER_SHARED)
        import arc, tools, satfetcher
        
        # import gdvspectra module
        sys.path.append(FOLDER_MODULES)
        import cog
        
        # globals 
        AWS_KEY = ''
        AWS_SECRET = ''
        STAC_ENDPOINT = 'https://explorer.sandbox.dea.ga.gov.au/stac/search'
        RESULT_LIMIT = 250
        
        # notify 
        arcpy.AddMessage('Performing COG Sync.')
                                            
        # grab parameter values 
        in_nc = parameters[0].valueAsText      # raw netcdf

        # set up progess bar
        arcpy.SetProgressor(type='default', message='Loading and checking netcdf...')
        
        # load netcdf file as xr
        ds = satfetcher.load_local_nc(nc_path=in_nc, 
                                      use_dask=True, 
                                      conform_nodata_to=np.nan)  #nodatavals?
        
        # checks
        if 'time' not in ds:
            arcpy.AddError('No time dimension detected.')
        
        #tod other checks 
        
        # get original query attributes 
        arcpy.SetProgressorLabel('Getting original query parameters...')

        # check attributes
        in_bands = list(ds.data_vars)
        collections = list(ds.orig_collections)
        bbox = list(ds.orig_bbox)
        in_res = ds.res # use get xr res method
        crs = ds.crs
        in_slc_off = ds.orig_slc_off
        resampling = ds.orig_resample
        nodatavals = ds.nodatavals

        # need to do
        in_epsg = int(crs.split(':')[1])
        in_platform = 'Landsat'
        dtype = 'int16'
        fill_value = -999
        in_snap = True
        rescale = True
        cell_align = 'Top-left'
        chunk_size = -1
        
        # get datetimes
        arcpy.SetProgressorLabel('Assessing dates...')

        # get now, earliest, latest datetimes in dataset
        dt_now = np.datetime64('now')
        dt_first = ds['time'].isel(time=0).values
        dt_last = ds['time'].isel(time=-1).values

        # conver to stac format
        in_from_date = arc.datetime_to_string(pd.Timestamp(dt_last))
        in_to_date = arc.datetime_to_string(pd.Timestamp(dt_now))

        # check if xr dt less than now (will be for now, but not if override)
        if dt_last < dt_now:
            
            # fetch cog
            arcpy.SetProgressorLabel('Performing STAC query...')
            feats = cog.fetch_stac_data(stac_endpoint=STAC_ENDPOINT, 
                                        collections=collections, 
                                        start_dt=in_from_date, 
                                        end_dt=in_to_date, 
                                        bbox=bbox,
                                        slc_off=in_slc_off,
                                        limit=RESULT_LIMIT)
                                        
            # count number of items
            arcpy.AddMessage('Found {} {} scenes.'.format(len(feats), in_platform))
                    
            # prepare band (i.e. stac assets) names
            assets = in_bands
            #assets = arc.prepare_band_names(in_bands=in_bands, 
                                            #in_platform=in_platform)
                
            # convert raw stac into dict with coord reproject, etc.
            arcpy.SetProgressorLabel('Converting STAC data into useable format...')
            meta, asset_table = cog.prepare_data(feats, 
                                                 assets=assets,
                                                 bounds_latlon=bbox, 
                                                 bounds=None, 
                                                 epsg=in_epsg, 
                                                 resolution=in_res, 
                                                 snap_bounds=in_snap,
                                                 force_dea_http=True)
                                        
            
            
            
        else:
            arcpy.AddMessage('No new scenes available. No sync required.')

    
        return
