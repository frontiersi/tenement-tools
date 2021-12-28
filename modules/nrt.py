# nrt
'''
Temp.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os
import sys
import shutil
import datetime
import xarray as xr
import rasterio
from osgeo import gdal

sys.path.append('../../modules')
import cog_odc

sys.path.append('../../shared')
import arc, satfetcher, tools


def create_nrt_project(out_folder, out_filename):
    """
    Creates a new empty geodatabase with required features
    for nrt monitoring tools.
    
    Parameters
    ----------
    out_folder: str
        An output path for new project folder.
    out_filename: str
        An output filename for new project.
    """
    
    # notify
    print('Creating new monitoring project database...')
    
    # check inputs are not none and strings
    if out_folder is None or out_filename is None:
        raise ValueError('Blank folder or filename provided.')
    elif not isinstance(out_folder, str) or not isinstance(out_folder, str):
        raise TypeError('Folder or filename not strings.')
    
    # get full path
    out_filepath = os.path.join(out_folder, out_filename + '.gdb')
    
    # check folder exists
    if not os.path.exists(out_folder):
        raise ValueError('Requested folder does not exist.')
        
    # check file does not already exist
    if os.path.exists(out_filepath):
        raise ValueError('Requested file location arleady exists. Choose a different name.')
    
    # build project geodatbase
    out_filepath = arcpy.management.CreateFileGDB(out_folder, out_filename)
    
    
    # notify
    print('Generating database feature class...')
    
    # temporarily disable auto-visual of outputs
    arcpy.env.addOutputsToMap = False
    
    # create feature class and aus albers spatial ref sys
    srs = arcpy.SpatialReference(3577)
    out_feat = arcpy.management.CreateFeatureclass(out_path=out_filepath, 
                                                   out_name='monitoring_areas', 
                                                   geometry_type='POLYGON',
                                                   spatial_reference=srs)
    
    
    # notify
    print('Generating database domains...')
    
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

    # create year domain
    arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                  domain_name='dom_years', 
                                  domain_description='Training years (1980 - 2050)',
                                  field_type='LONG', 
                                  domain_type='RANGE')
    
    # generate range values to year domain
    arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                            domain_name='dom_years', 
                                            min_value=1980, 
                                            max_value=2050)
    
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
        

    # notify
    print('Generating database fields...') 
    
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
                              field_domain='dom_years')
    
    # add e_year field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='e_year', 
                              field_type='LONG', 
                              field_alias='End Year of Training Period',
                              field_is_required='REQUIRED',
                              field_domain='dom_years')
    
    # add index field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='index', 
                              field_type='TEXT', 
                              field_alias='Vegetation Index',
                              field_length=20,
                              field_is_required='REQUIRED',
                              field_domain='dom_indices')
    
    # add alert field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='alert', 
                              field_type='TEXT', 
                              field_alias='Alert User',
                              field_length=20,
                              field_is_required='REQUIRED',
                              field_domain='dom_boolean')
    
    # add email field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='email', 
                              field_type='TEXT', 
                              field_alias='Email of User',
                              field_is_required='REQUIRED')
    
    # add last_run field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='last_run', 
                              field_type='DATE', 
                              field_alias='Last Run',
                              field_is_required='NON_REQUIRED')   
    
    # notify todo - delete if we dont want defaults
    print('Generating database defaults...')  
    
    # set default platform
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='platform',
                                          default_value='Landsat')   

    # set default index
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='index',
                                          default_value='MAVI')        

    # set default alert
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='alert',
                                          default_value='No')    
           
           
    # notify
    print('Creating NetCDF data folder...') 
    
    # create output folder
    out_nc_folder = os.path.join(out_folder, '{}_cubes'.format(out_filename))
    if os.path.exists(out_nc_folder):
        try:
            shutil.rmtree(out_nc_folder)
        except:
            raise ValueError('Could not delete {}'.format(out_nc_folder))

    # create new folder
    os.makedirs(out_nc_folder)
    
    
    # notify
    print('Adding data to current map...') 
    
    # enable auto-visual of outputs
    arcpy.env.addOutputsToMap = True
    
    try:
        # get active map, add feat
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(out_feat)
    
    except:
        arcpy.AddWarning('Could not find active map. Add monitor areas manually.')        
        
    # notify
    print('Created new monitoring project database successfully.')


# checks, dtype, fillvalueto, fillvalue_from need doing, metadata
def sync_nrt_cube(out_nc, collections, bands, start_dt, end_dt, bbox, in_epsg=3577, slc_off=False, resolution=30, ds_existing=None, chunks={}):
    """
    Takes a path to a netcdf file, a start and end date, bounding box and
    obtains the latest satellite imagery from DEA AWS. If an existing
    dataset is provided, the metadata from that is used to define the
    coordinates, etc. This function is used to 'sync' existing cubes
    to the latest scene (time = now). New scenes are appended on to
    the existing cube and re-exported to a new file (overwrite).
    
    Parameters
    ----------
    in_feat: str
        A path to an existing monitoring areas gdb feature class.
    in_epsg: int
        A integer representing a specific epsg code for coordinate system.
      
    """
    
    # checks
    
    # notify
    print('Syncing cube for monitoring area: {}'.format(out_nc))

    # query stac endpoint
    items = cog_odc.fetch_stac_items_odc(stac_endpoint='https://explorer.sandbox.dea.ga.gov.au/stac', 
                                         collections=collections, 
                                         start_dt=start_dt, 
                                         end_dt=end_dt, 
                                         bbox=bbox,
                                         slc_off=slc_off,
                                         limit=250)

    # replace s3 prefix with https for each band - arcgis doesnt like s3
    items = cog_odc.replace_items_s3_to_https(items=items, 
                                              from_prefix='s3://dea-public-data', 
                                              to_prefix='https://data.dea.ga.gov.au')

    # construct an xr of items (lazy)
    ds = cog_odc.build_xr_odc(items=items,
                              bbox=bbox,
                              bands=bands,
                              crs=in_epsg,
                              resolution=resolution,
                              group_by='solar_day',
                              skip_broken_datasets=True,
                              like=ds_existing,
                              chunks=chunks)

    # prepare lazy ds with data type, type, time etc
    ds = cog_odc.convert_type(ds=ds, to_type='int16')  # input?
    ds = cog_odc.change_nodata_odc(ds=ds, orig_value=0, fill_value=-999)  # input?
    ds = cog_odc.fix_xr_time_for_arc_cog(ds)

    # return dataset instantly if new, else append
    if ds_existing is None:
        return ds
    else:
        # existing netcdf - append
        ds_new = ds_existing.combine_first(ds).copy(deep=True)

        # close everything safely
        ds.close()
        ds_existing.close()

        return ds_new

  
# todo - include provisional products too. finish meta
def get_satellite_params(platform=None):
    """
    Helper function to generate Landsat or Sentinel query information
    for quick use during NRT cube creation or sync only.
    
    Parameters
    ----------
    platform: str
        Name of a satellite platform, Landsat or Sentinel only.
    
    params
    """
    
    # check platform name
    if platform is None:
        raise ValueError('Must provide a platform name.')
    elif platform.lower() not in ['landsat', 'sentinel']:
        raise ValueError('Platform must be Landsat or Sentinel.')
        
    # set up dict
    params = {}
    
    # get porams depending on platform
    if platform.lower() == 'landsat':
        
        # get collections
        collections = [
            'ga_ls5t_ard_3', 
            'ga_ls7e_ard_3', 
            'ga_ls8c_ard_3']
        
        # get bands
        bands = [
            'nbart_red', 
            'nbart_green', 
            'nbart_blue', 
            'nbart_nir', 
            'nbart_swir_1', 
            'nbart_swir_2', 
            'oa_fmask']
        
        # get resolution
        resolution = 30
        
        # build dict
        params = {
            'collections': collections,
            'bands': bands,
            'resolution': resolution}
        
    else:
        
        # get collections
        collections = [
            's2a_ard_granule', 
            's2b_ard_granule']
        
        # get bands
        bands = [
            'nbart_red', 
            'nbart_green', 
            'nbart_blue', 
            'nbart_nir_1', 
            'nbart_swir_2', 
            'nbart_swir_3', 
            'fmask']
        
        # get resolution
        resolution = 10
        
        # build dict
        params = {
            'collections': collections,
            'bands': bands,
            'resolution': resolution}        
        
    return params
 
 
# todo - meta
def validate_monitoring_areas(in_feat):
    """
    Does relevant checks for information for a
    gdb feature class of one or more monitoring areas.
    """

    is_valid = True

    # check input feature is not none and strings
    if in_feat is None:
        raise ValueError('Monitoring area feature class not provided.')
    elif not isinstance(in_feat, str):
        raise TypeError('Monitoring area feature class not string.')
        
    try:
        # get feature epsg and check
        lyr = arcpy.Describe(in_feat)
        epsg = lyr.spatialReference.factoryCode
        
        # check if epsg is correct
        if not isinstance(epsg, int) or epsg != 3577:
            print('Monitoring area feature EPSG is incorrect, flagging as invalid.')
            is_valid = False           
    except:
        print('Could not open monitoring area feature, flagging as invalid.')
        is_valid = False
        
    # get num rows and all area ids
    fields = ['area_id']
    with arcpy.da.SearchCursor(in_feat, fields) as cursor:
        feat_count = 0
        area_ids = []
        
        for row in cursor:
            feat_count += 1
            area_ids.append(row[0])
            

    # check if number of features > 0
    if feat_count == 0:
        print('No monitoring areas found in feature, flagging as invalid.')
        is_valid = False
        
    print(area_ids)
        
    # check if duplicate area ids
    if len(set(area_ids)) != len(area_ids):
        print('Duplicate area ids detected, flagging as invalid.')
        is_valid = False
        
    return is_valid
 
 
# todo - meta
def validate_monitoring_area(area_id, platform, s_year, e_year, index):
    """
    Does relevant checks for information for a
    single monitoring area.
    """

    # set valid flag
    is_valid = True

    # check area id exists
    if area_id is None:
        print('No area id exists, flagging as invalid.')
        is_valid = False

    # check platform is Landsat or Sentinel
    if platform is None:
        print('No platform exists, flagging as invalid.')
        is_valid = False
    elif platform.lower() not in ['landsat', 'sentinel']:
        print('Platform must be Landsat or Sentinel, flagging as invalid.')
        is_valid = False

    # check if start and end years are valid
    if not isinstance(s_year, int) or not isinstance(e_year, int):
        print('Start and end year values must be integers, flagging as invalid.')
        is_valid = False
    elif s_year < 1980 or s_year > 2050:
        print('Start year must be between 1980 and 2050, flagging as invalid.')
        is_valid = False
    elif e_year < 1980 or e_year > 2050:
        print('End year must be between 1980 and 2050, flagging as invalid.')
        is_valid = False
    elif e_year <= s_year:
        print('Start year must be less than end year, flagging as invalid.')
        is_valid = False
    elif abs(e_year - s_year) < 2:
        print('Must be at least 2 years between start and end year, flagging as invalid.')
        is_valid = False

    # check if index is acceptable
    if index is None:
        print('No index exists, flagging as invalid.')
        is_valid = False
    elif index.lower() not in ['ndvi', 'mavi', 'kndvi']:
        print('Index must be NDVI, MAVI or kNDVI, flagging as invalid.')
        is_valid = False

    return is_valid 
 
 
 # todo do checks, do meta
def mask_xr_via_polygon(geom, x, y, bbox, transform, ncols, nrows, mask_value=1):
    """
    geom object from gdal
    x, y = arrays of coordinates from xr dataset
    bbox 
    transform from geobox
    ncols, nrows = len of x, y

    """

    # extract bounding box extents
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

    # create ogr transform structure
    geotransform = (transform[2], transform[0], 0.0, 
                    transform[5], 0.0, transform[4])

    # create template raster in memory
    dst_rast = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 , gdal.GDT_Byte)
    dst_rb = dst_rast.GetRasterBand(1)      # get a band
    dst_rb.Fill(0)                          # init raster with zeros
    dst_rb.SetNoDataValue(0)                # set nodata to zero
    dst_rast.SetGeoTransform(geotransform)  # resample, transform

    # rasterise vector and flush
    err = gdal.RasterizeLayer(dst_rast, [1], geom, burn_values=[mask_value])
    dst_rast.FlushCache()

    # get numpy version of classified raster
    arr = dst_rast.GetRasterBand(1).ReadAsArray()

    # create mask
    mask = xr.DataArray(data=arr,
                        dims=['y', 'x'],
                        coords={'y': y, 'x': x})

    return mask