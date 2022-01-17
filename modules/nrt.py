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
import time
import datetime
import smtplib
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import rasterio
from osgeo import gdal
from osgeo import ogr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    sys.path.append('../../modules')
    import cog_odc

    sys.path.append('../../shared')
    import arc, satfetcher, tools
    
except:
    print('TEMP DELETE THIS TRYCATCH LATER ONCE WEMACD UP AND RUNNING')

# prints
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
    
    # imports 
    try:
        import arcpy 
    except:
        raise ValueError('Could not import arcpy.')
    
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
 
 
# todo - meta, remove arcpy dependency
def validate_monitoring_areas(in_feat):
    """
    Does relevant checks for information for a
    gdb feature class of one or more monitoring areas.
    """
    
    # set up flag
    is_valid = True

    # check input feature is not none and strings
    if in_feat is None:
        print('Monitoring area feature class not provided, flagging as invalid.')
        is_valid = False
    elif not isinstance(in_feat, str):
        print('Monitoring area feature class not string, flagging as invalid.')
        is_valid = False
    elif not os.path.dirname(in_feat).endswith('.gdb'):
        print('Feature class is not in a geodatabase, flagging as invalid.')
        is_valid = False
        
    # if we've made it, check shapefile
    if is_valid:
        try:
            # get feature
            driver = ogr.GetDriverByName("OpenFileGDB")
            data_source = driver.Open(os.path.dirname(in_feat), 0)
            lyr = data_source.GetLayer('monitoring_areas')
            
            # get and check feat count
            feat_count = lyr.GetFeatureCount()
            if feat_count == 0:
                print('No monitoring areas found in feature, flagging as invalid.')
                is_valid = False
            
            # get epsg
            epsg = lyr.GetSpatialRef()
            if 'GDA_1994_Australia_Albers' not in epsg.ExportToWkt():
                print('Could not find GDA94 albers code in shapefile, flagging as invalid.')
                is_valid = False
            
            # check if any duplicate area ids
            area_ids = []
            for feat in lyr:
                area_ids.append(feat['area_id'])
                
            # check if duplicate area ids
            if len(set(area_ids)) != len(area_ids):
                print('Duplicate area ids detected, flagging as invalid.')
                is_valid = False
                
            # close data source
            data_source.Destroy()
            
        except:
            print('Could not open monitoring area feature, flagging as invalid.')
            is_valid = False
            data_source.Destroy()

    # return
    return is_valid
 
 
# todo - meta
def validate_monitoring_area(area_id, platform, s_year, e_year, index):
    """
    Does relevant checks for information for a
    single monitoring area.
    """
    
    # check area id exists
    if area_id is None:
        print('No area id exists, flagging as invalid.')
        return False

    # check platform is Landsat or Sentinel
    if platform is None:
        print('No platform exists, flagging as invalid.')
        return False
    elif platform.lower() not in ['landsat', 'sentinel']:
        print('Platform must be Landsat or Sentinel, flagging as invalid.')
        return False

    # check if start and end years are valid
    if not isinstance(s_year, int) or not isinstance(e_year, int):
        print('Start and end year values must be integers, flagging as invalid.')
        return False
    elif s_year < 1980 or s_year > 2050:
        print('Start year must be between 1980 and 2050, flagging as invalid.')
        return False
    elif e_year < 1980 or e_year > 2050:
        print('End year must be between 1980 and 2050, flagging as invalid.')
        return False
    elif e_year <= s_year:
        print('Start year must be less than end year, flagging as invalid.')
        return False
    elif abs(e_year - s_year) < 2:
        print('Must be at least 2 years between start and end year, flagging as invalid.')
        return False

    # check if index is acceptable
    if index is None:
        print('No index exists, flagging as invalid.')
        return False
    elif index.lower() not in ['ndvi', 'mavi', 'kndvi']:
        print('Index must be NDVI, MAVI or kNDVI, flagging as invalid.')
        return False

    # all good!
    return True 
 
 
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
    


# EWMACD EWMACD EWMACD
# TODO LIST
# todo 0 : current error: historyBound is wrong.
# todo 1: any numpy we "copy" must use .copy(), or we overwrite mem...!
# todo 2: force type where needed... important!

# note: check the pycharm project pyEWMACD for original, working code if i break this!!!


def harmonic_matrix(timeSeries0to2pi, numberHarmonicsSine,  numberHarmonicsCosine):

    # generate harmonic matrix todo 1 or 0? check
    col_ids = np.repeat(1, len(timeSeries0to2pi))

    # get sin harmonics todo need to start at 1, so + 1 to tail
    _ = np.vstack(np.arange(1, numberHarmonicsSine + 1))
    _ = np.repeat(_, len(timeSeries0to2pi), axis=1)
    col_sin = np.sin((_ * timeSeries0to2pi)).T

    # get cos harmonics todo need to start at 1, so + 1 to tail
    _ = np.vstack(np.arange(1, numberHarmonicsCosine + 1))
    _ = np.repeat(_, len(timeSeries0to2pi), axis=1)
    col_cos = np.cos((_ * timeSeries0to2pi)).T

    # stack into columns
    X = np.column_stack([col_ids, col_sin, col_cos])

    return X


def hreg_pixel(Responses, DOYs, numberHarmonicsSine, numberHarmonicsCosine, anomalyThresholdSigmas=1.5, valuesAlreadyCleaned=True):
    """hreg pixel function"""

    # todo this needs a check
    if valuesAlreadyCleaned == False:
        missingIndex = np.flatnonzero(np.isnan(Responses))
        if len(missingIndex) > 0:
            Responses = np.delete(Responses, missingIndex)
            DOYs = np.delete(DOYs, missingIndex)

    # assumes cleaned, non-missing inputs here; screening needs to be done ahead of running!
    Beta = np.repeat(np.nan, (1 + numberHarmonicsSine + numberHarmonicsCosine))
    Rsquared = None
    RMSE = None

    # generate harmonic matrix for given sin, cos numbers
    X = harmonic_matrix(DOYs * 2 * np.pi / 365, numberHarmonicsSine, numberHarmonicsCosine)

    # ensuring design matrix is sufficient rank and nonsingular
    if len(Responses) > (numberHarmonicsSine + numberHarmonicsCosine + 1) and np.abs(np.linalg.det(np.matmul(X.T, X))) >= 0.001:

        # todo check during harmonics > 1
        Preds1 = np.matmul(X, np.linalg.solve(np.matmul(X.T, X), np.vstack(np.matmul(X.T, Responses))))

        # x-bar chart anomaly filtering
        Resids1 = Responses[:, None] - Preds1  # todo i added the new axis [:, None]
        std = np.std(Resids1, ddof=1)
        screen1 = (np.abs(Resids1) > (anomalyThresholdSigmas * std)) + 0
        keeps = np.flatnonzero(screen1 == 0)

        if len(keeps) > (numberHarmonicsCosine + numberHarmonicsSine + 1):
            X_keeps = X[keeps, ]
            Responses_keeps = Responses[keeps]

            # todo check when using harmonics > 1
            Beta = np.linalg.solve(np.matmul(X_keeps.T, X_keeps),
                                   np.vstack(np.matmul(X_keeps.T, Responses_keeps))).flatten()

            fits = np.matmul(X_keeps, Beta)
            Rsquared = 1 - np.sum(np.square(Responses_keeps - fits)) / np.sum(np.square(Responses_keeps - np.sum(Responses_keeps) / len(Responses_keeps)))
            RMSE = np.sum(np.square(Responses_keeps - fits))

        # setup output
        output = {
            'Beta': Beta,
            'Rsquared': Rsquared,
            'RMSE': RMSE
        }

        return output


def optimize_hreg(timeStampsYears, timeStampsDOYs, Values, threshold, minLength, maxLength, ns=1, nc=1, screenSigs=3):
    """optimize hreg function"""

    minHistoryBound = np.min(np.flatnonzero((timeStampsYears >= timeStampsYears[minLength]) &
                                            ((timeStampsYears - timeStampsYears[0]) > 1)))  # todo changed from 1 to 0

    if np.isinf(minHistoryBound):  # todo using inf...
        minHistoryBound = 1

    # NOTE: maxLength applies from the point of minHistoryBound, not from time 1!
    historyBoundCandidates = np.arange(0, np.min(np.append(len(Values) - minHistoryBound, maxLength))) # todo removed the - 1, py dont start at 1!
    historyBoundCandidates = historyBoundCandidates + minHistoryBound

    if np.isinf(np.max(historyBoundCandidates)):  # todo using inf...
        historyBoundCandidates = len(timeStampsYears)

    i = 0
    fitQuality = 0
    while (fitQuality < threshold) and (i < np.min([maxLength, len(historyBoundCandidates)])):

        # Moving Window Approach todo needs a good check
        _ = np.flatnonzero(~np.isnan(Values[(i):(historyBoundCandidates[i])]))
        testResponses = Values[i:historyBoundCandidates[i]][_]

        # call hreg pixel function
        fitQuality = hreg_pixel(Responses=testResponses,
                                numberHarmonicsSine=ns,
                                numberHarmonicsCosine=nc,
                                DOYs=timeStampsDOYs[i:historyBoundCandidates[i]],
                                anomalyThresholdSigmas=screenSigs,
                                valuesAlreadyCleaned=True)

        # get r-squared from fit, set to 0 if empty
        fitQuality = fitQuality.get('Rsquared')
        fitQuality = 0 if fitQuality is None else fitQuality

        # count up
        i += 1

    # get previous history bound and previous fit
    historyBound = historyBoundCandidates[i - 1]  # todo added - 1 here to align with r 1 indexes

    # package output
    opt_output = {
        'historyBound': int(historyBound),
        'fitPrevious': int(minHistoryBound)
    }
    return opt_output


def EWMA_chart(Values, _lambda, histSD, lambdaSigs, rounding):
    """emwa chart"""

    ewma = np.repeat(np.nan, len(Values))
    ewma[0] = Values[0]  # initialize the EWMA outputs with the first present residual

    for i in np.arange(1, len(Values)):  # todo r starts at 2 here, so for us 1
        ewma[i] = ewma[(i - 1)] * (1 - _lambda) + _lambda * Values[i]  # appending new EWMA values for all present data.

    # ewma upper control limit. this is the threshold which dictates when the chart signals a disturbance
    # todo this is not an index, want array of 1:n to calc off those whole nums. start at 1, end at + 1
    UCL = histSD * lambdaSigs * np.sqrt(_lambda / (2 - _lambda) * (1 - (1 - _lambda) ** (2 * np.arange(1, len(Values) + 1))))

    # integer value for EWMA output relative to control limit (rounded towards 0).  A value of +/-1 represents the weakest disturbance signal
    output = None
    if rounding == True:
        output = (np.sign(ewma) * np.floor(np.abs(ewma / UCL)))
        output = output.astype('int16')  # todo added this to remove -0s
    elif rounding == False:
        # EWMA outputs in terms of resdiual scales.
        output = (np.round(ewma, 0))  # 0 is decimals

    return output


def persistence_filter(Values, persistence):
    """persistence filter"""
    # culling out transient values
    # keeping only values for which a disturbance is sustained, using persistence as the threshold
    tmp4 = np.repeat(0, len(Values))

    # ensuring sufficent data for tmp2
    if persistence > 1 and len(Values) > persistence:
        # disturbance direction
        tmpsign = np.sign(Values)

        # Dates for which direction changes # todo check this carefully, especially the two - 1s
        shiftpoints = np.flatnonzero(np.delete(tmpsign, 0) != np.delete(tmpsign, len(tmpsign) - 1))
        shiftpoints = np.append(np.insert(shiftpoints, 0, 0), len(tmpsign) - 1)  # prepend 0 to to start, len to end

        # Counting the consecutive dates in which directions are sustained
        # todo check this
        tmp3 = np.repeat(0, len(tmpsign))
        for i in np.arange(0, len(tmpsign)):
            tmp3lo = 0
            tmp3hi = 0

            while ((i + 1) - tmp3lo) > 0:  # todo added + 1
                if (tmpsign[i] - tmpsign[i - tmp3lo]) == 0:
                    tmp3lo += 1
                else:
                    break

            # todo needs look at index, check
            while (tmp3hi + (i + 1)) <= len(tmpsign):  # todo added + 1
                if (tmpsign[i + tmp3hi] - tmpsign[i]) == 0:
                    tmp3hi += 1
                else:
                    break

            # todo check indexes
            tmp3[i] = tmp3lo + tmp3hi - 1

        tmp4 = np.repeat(0, len(tmp3))
        tmp3[0:persistence, ] = persistence
        Values[0:persistence] = 0

        # if sustained dates are long enough, keep; otherwise set to previous sustained state
        # todo this needs a decent check
        for i in np.arange(persistence, len(tmp3)):  # todo removed + 1
            if tmp3[i] < persistence and np.max(tmp3[0:i]) >= persistence:
                tmpCbind = np.array([np.arange(0, i + 1), tmp3[0:i + 1], Values[0:i + 1]]).T  # todo added + 1
                tmp4[i] = tmpCbind[np.max(np.flatnonzero(tmpCbind[:, 1] >= persistence)), 2]  # todo is 3 or 2 the append value here?
            else:
                tmp4[i] = Values[i]

    return tmp4


def backfill_missing(nonMissing, nonMissingIndex, withMissing):
    """backfill missing"""

    # backfilling missing data
    withMissing = withMissing.copy()  # todo had to do a copy to prevent mem overwrite
    withMissing[nonMissingIndex] = nonMissing

    # if the first date of myPixel was missing/filtered, then assign the EWMA output as 0 (no disturbance).
    if np.isnan(withMissing[0]):
        withMissing[0] = 0

    # if we have EWMA information for the first date, then for each missing/filtered date
    # in the record, fill with the last known EWMA value
    for stepper in np.arange(1, len(withMissing)):
        if np.isnan(withMissing[stepper]):
            withMissing[stepper] = withMissing[stepper - 1]  # todo check this

    return withMissing


def EWMACD_clean_pixel_date_by_date(inputPixel, numberHarmonicsSine, numberHarmonicsCosine, inputDOYs, inputYears, trainingStart, trainingEnd, historyBound, precedents, xBarLimit1=1.5, xBarLimit2=20, lambdaSigs=3, _lambda=0.3, rounding=True, persistence=4):

    # prepare variables
    Dates = len(inputPixel)  # Convenience object
    outputValues = np.repeat(np.nan, Dates)  # Output placeholder
    residualOutputValues = np.repeat(np.nan, Dates)  # Output placeholder
    Beta = np.vstack(np.repeat(np.nan, (numberHarmonicsSine + numberHarmonicsCosine + 1)))

    # get training index and subset pixel
    indexTraining = np.arange(0, historyBound)
    myPixelTraining = inputPixel[indexTraining]            # Training data
    myPixelTesting = np.delete(inputPixel, indexTraining)  # Testing data

    ### Checking if there is data to work with...
    if len(myPixelTraining) > 0:
        out = hreg_pixel(Responses=myPixelTraining[(historyBound - precedents):historyBound],      # todo was a + 1 here
                         DOYs=inputDOYs[indexTraining][(historyBound - precedents):historyBound],  # todo was a + 1 here
                         numberHarmonicsSine=numberHarmonicsSine,
                         numberHarmonicsCosine=numberHarmonicsCosine,
                         anomalyThresholdSigmas=xBarLimit1)

        # extract beta variable
        Beta = out.get('Beta')

        # checking for present Beta
        if Beta[0] is not None:
            XAll = harmonic_matrix(inputDOYs * 2 * np.pi / 365, numberHarmonicsSine, numberHarmonicsCosine)
            myResiduals = (inputPixel - np.matmul(XAll, Beta).T)  # residuals for all present data, based on training coefficients
            residualOutputValues = myResiduals.copy()  # todo added copy for memory write

            myResidualsTraining = myResiduals[indexTraining]  # training residuals only
            myResidualsTesting = np.array([])

            if len(myResiduals) > len(myResidualsTraining):  # Testing residuals
                myResidualsTesting = np.delete(myResiduals, indexTraining)

            SDTraining = np.std(myResidualsTraining, ddof=1)  # first estimate of historical SD
            residualIndex = np.arange(0, len(myResiduals))  # index for residuals
            residualIndexTraining = residualIndex[indexTraining]  # index for training residuals
            residualIndexTesting = np.array([])

            # index for testing residuals
            if len(residualIndex) > len(residualIndexTraining):
                residualIndexTesting = np.delete(residualIndex, indexTraining)

            # modifying SD estimates based on anomalous readings in the training data
            # note that we don't want to filter out the changes in the testing data, so xBarLimit2 is much larger!
            UCL0 = np.concatenate([np.repeat(xBarLimit1, len(residualIndexTraining)),
                                   np.repeat(xBarLimit2, len(residualIndexTesting))])
            UCL0 = UCL0 * SDTraining

            # keeping only dates for which we have some vegetation and aren't anomalously far from 0 in the residuals
            indexCleaned = residualIndex[np.abs(myResiduals) < UCL0]
            myResidualsCleaned = myResiduals[indexCleaned]

            # updating the training SD estimate. this is the all-important modifier for the EWMA control limits.
            SDTrainingCleaned = myResidualsTraining[np.flatnonzero(np.abs(myResidualsTraining) < UCL0[indexTraining])]
            SDTrainingCleaned = np.std(SDTrainingCleaned, ddof=1)

            ### -------
            if SDTrainingCleaned is None:  # todo check if sufficient for empties
                cleanOutput = {
                    'outputValues': outputValues,
                    'residualOutputValues': residualOutputValues,
                    'Beta': Beta
                }
                return cleanOutput

            chartOutput = EWMA_chart(Values=myResidualsCleaned, _lambda = _lambda,
                                     histSD=SDTrainingCleaned, lambdaSigs=lambdaSigs,
                                     rounding=rounding)

            ###  Keeping only values for which a disturbance is sustained, using persistence as the threshold
            # todo this produces the wrong result, check the for loop out
            persistentOutput = persistence_filter(Values=chartOutput, persistence=persistence)

            # Imputing for missing values screened out as anomalous at the control limit stage
            outputValues = backfill_missing(nonMissing=persistentOutput, nonMissingIndex=indexCleaned,
                                            withMissing=np.repeat(np.nan, len(myResiduals)))

    # create output
    cleanOutput = {
        'outputValues': outputValues,
        'residualOutputValues': residualOutputValues,
        'Beta': Beta
    }

    return cleanOutput


def EWMACD_pixel_date_by_date(myPixel, DOYs, Years, _lambda, numberHarmonicsSine, numberHarmonicsCosine, trainingStart, testingEnd, trainingPeriod='dynamic', trainingEnd=None, minTrainingLength=None, maxTrainingLength=np.inf, trainingFitMinimumQuality=None, xBarLimit1=1.5, xBarLimit2=20, lowthresh=0, lambdaSigs=3, rounding=True, persistence_per_year=0.5, reverseOrder=False, simple_output=True):
    """pixel date by date function"""

    # setup breakpoint tracker. note arange ignores the value at stop, must + 1
    breakPointsTracker = np.arange(0, len(myPixel))
    breakPointsStart = np.array([], dtype='int16')
    breakPointsEnd = np.array([], dtype='int16')
    BetaFirst = np.repeat(np.nan, (1 + numberHarmonicsSine + numberHarmonicsCosine)) # setup betas (?)

    ### initial assignment and reverse-toggling as specified
    if reverseOrder == True:
        myPixel = np.flip(myPixel) # reverse array

    # convert doys, years to decimal years for ordering
    DecimalYears = (Years + DOYs / 365)

    ### sort all arrays based on order of decimalyears order via indexes
    myPixel = myPixel[np.argsort(DecimalYears)]
    Years = Years[np.argsort(DecimalYears)]
    DOYs = DOYs[np.argsort(DecimalYears)]
    DecimalYears = DecimalYears[np.argsort(DecimalYears)]

    # if no training end given, default to start year + 3 years
    if trainingEnd == None:
        trainingEnd = trainingStart + 3

    # trim relevent arrays to the user specified timeframe
    # gets indices between starts and end and subset doys, years
    trims = np.flatnonzero((Years >= trainingStart) & (Years < testingEnd))
    DOYs = DOYs[trims]
    Years = Years[trims]
    YearsForAnnualOutput = np.unique(Years)
    myPixel = myPixel[trims]
    breakPointsTracker = breakPointsTracker[trims]

    ### removing missing values and values under the fitting threshold a priori
    dateByDateWithMissing = np.repeat(np.nan, len(myPixel))
    dateByDateResidualsWithMissing = np.repeat(np.nan, len(myPixel))

    # get clean indexes, trim to clean pixel, years, doys, etc
    cleanedInputIndex = np.flatnonzero((~np.isnan(myPixel)) & (myPixel > lowthresh))
    myPixelCleaned = myPixel[cleanedInputIndex]
    YearsCleaned = Years[cleanedInputIndex]
    DOYsCleaned = DOYs[cleanedInputIndex]
    DecimalYearsCleaned = (Years + DOYs / 365)[cleanedInputIndex]
    breakPointsTrackerCleaned = breakPointsTracker[cleanedInputIndex]

    # exit pixel if pixel empty after clean
    if len(myPixelCleaned) == 0:
        output = {
            'dateByDate': np.repeat(np.nan, myPixel),
            'dateByDateResiduals': np.repeat(np.nan, myPixel),
            'Beta': BetaFirst,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }
        return output

    # set min training length if empty (?)
    if minTrainingLength is None:
        minTrainingLength = (1 + numberHarmonicsSine + numberHarmonicsCosine) * 3

    # todo check use of inf... not sure its purpose yet...
    if np.isinf(maxTrainingLength) or np.isnan(maxTrainingLength):
        maxTrainingLength = minTrainingLength * 2

    # calculate persistence
    persistence = np.ceil((len(myPixelCleaned) / len(np.unique(YearsCleaned))) * persistence_per_year)
    persistence = persistence.astype('int16')  # todo added conversion to int, check

    # todo add training period == static
    if trainingPeriod == 'static':
        if minTrainingLength == 0:
            minTrainingLength = 1

        if np.isinf(minTrainingLength):  # todo using inf...
            minTrainingLength = 1

        DecimalYearsCleaned = (YearsCleaned + DOYsCleaned / 365)

        # call optimize hreg
        optimal_outputs = optimize_hreg(DecimalYearsCleaned,
                                        DOYsCleaned,
                                        myPixelCleaned,
                                        trainingFitMinimumQuality,
                                        minTrainingLength,
                                        maxTrainingLength,
                                        ns=1,
                                        nc=1,
                                        screenSigs=xBarLimit1)

        # get bounds, precedents
        historyBound = optimal_outputs.get('historyBound')
        training_precedents = optimal_outputs.get('fitPrevious')

        # combine bp start, tracker, ignore start if empty
        breakPointsStart = np.append(breakPointsStart, breakPointsTrackerCleaned[0])
        breakPointsEnd = np.append(breakPointsEnd, breakPointsTrackerCleaned[historyBound])

        if np.isnan(historyBound):  # todo just check this handles None
            return dateByDateWithMissing

        # call ewmac clean pixel date by date
        tmpOut = EWMACD_clean_pixel_date_by_date(inputPixel=myPixelCleaned,
                                                 numberHarmonicsSine=numberHarmonicsSine,
                                                 numberHarmonicsCosine=numberHarmonicsCosine,
                                                 inputDOYs=DOYsCleaned,
                                                 inputYears=YearsCleaned,
                                                 trainingStart=trainingStart,  # todo added this
                                                 trainingEnd=trainingEnd,  # todo added this
                                                 _lambda=_lambda,
                                                 lambdaSigs=lambdaSigs,
                                                 historyBound=historyBound,
                                                 precedents=training_precedents,
                                                 persistence=persistence)

        # get output values
        runKeeps = tmpOut.get('outputValues')
        runKeepsResiduals = tmpOut.get('residualOutputValues')
        BetaFirst = tmpOut.get('Beta')

    # begin dynamic (Edyn) method
    if trainingPeriod == 'dynamic':
        myPixelCleanedTemp = myPixelCleaned
        YearsCleanedTemp = YearsCleaned
        DOYsCleanedTemp = DOYsCleaned
        DecimalYearsCleanedTemp = (YearsCleanedTemp + DOYsCleanedTemp / 365)
        breakPointsTrackerCleanedTemp = breakPointsTrackerCleaned

        # buckets for edyn outputs
        runKeeps = np.repeat(np.nan, len(myPixelCleaned))
        runKeepsResiduals = np.repeat(np.nan, len(myPixelCleaned))

        # set indexer
        indexer = 0  # todo was 1
        while len(myPixelCleanedTemp) > minTrainingLength and (np.max(DecimalYearsCleanedTemp) - DecimalYearsCleanedTemp[0]) > 1:

            if np.isinf(minTrainingLength): # todo using inf...
                minTrainingLength = 1

            # call optimize hreg
            optimal_outputs = optimize_hreg(DecimalYearsCleanedTemp,
                                            DOYsCleanedTemp,
                                            myPixelCleanedTemp,
                                            trainingFitMinimumQuality,
                                            minTrainingLength,
                                            maxTrainingLength,
                                            ns=1,
                                            nc=1,
                                            screenSigs=xBarLimit1)

            # get bounds, precedents
            historyBound = optimal_outputs.get('historyBound')
            training_precedents = optimal_outputs.get('fitPrevious')

            # combine bp start, tracker, ignore start if empty
            breakPointsStart = np.append(breakPointsStart, breakPointsTrackerCleanedTemp[0])
            breakPointsEnd = np.append(breakPointsEnd, breakPointsTrackerCleanedTemp[historyBound])

            # call ewmac clean pixel date by date
            tmpOut = EWMACD_clean_pixel_date_by_date(inputPixel=myPixelCleanedTemp,
                                                     numberHarmonicsSine=numberHarmonicsSine,
                                                     numberHarmonicsCosine=numberHarmonicsCosine,
                                                     inputDOYs=DOYsCleanedTemp,
                                                     inputYears=YearsCleanedTemp,
                                                     trainingStart=trainingStart,  # todo added this
                                                     trainingEnd=trainingEnd,      # todo added this
                                                     _lambda=_lambda,
                                                     lambdaSigs=lambdaSigs,
                                                     historyBound=historyBound,
                                                     precedents=training_precedents,
                                                     persistence=persistence)
            # get output values
            tmpRun = tmpOut.get('outputValues')
            tmpResiduals = tmpOut.get('residualOutputValues')
            if indexer == 0:
                BetaFirst = tmpOut.get('Beta')

            ## Scratch Work ####------
            # todo move to global method
            def vertex_finder(tsi):
                v1 = tsi[0]
                v2 = tsi[len(tsi) - 1]  # todo added - 1

                res_ind = None
                mse = None
                if np.sum(~np.isnan(tmpRun)) > 1:
                    tmp_mod = scipy.stats.linregress(x=DecimalYearsCleanedTemp[[v1, v2]], y=tmpRun[[v1, v2]])

                    tmp_int = tmp_mod.intercept
                    tmp_slope = tmp_mod.slope

                    tmp_res = tmpRun[tsi] - (tmp_int + tmp_slope * DecimalYearsCleanedTemp[tsi])

                    res_ind = np.argmax(np.abs(tmp_res)) + v1  # todo removed - 1
                    mse = np.sum(tmp_res ** 2)

                # create output
                v_out = {'res_ind': res_ind, 'mse': mse}
                return v_out

            vertices = np.flatnonzero(tmpRun != 0)
            if vertices.size != 0:
                vertices = np.array([np.min(vertices)])  # todo check this works, not fired yet
            else:
                vertices = np.array([historyBound - 1])  # todo added - 1 here

            #time_list = np.arange(vertices[0], len(tmpRun))
            time_list = [np.arange(vertices[0], len(tmpRun), dtype='int16')]  # todo added astype
            #seg_stop = np.prod(np.apply_along_axis(len, 0, time_list) > persistence)  # todo check along axis works in multi dim
            seg_stop = np.prod([len(e) for e in time_list] > persistence)

            vert_indexer = 0
            vert_new = np.array([0])
            while seg_stop == 1 and len(vert_new) >= 1:

                # todo this needs to consider multi dim array
                # todo e.g. for elem in time_list: send to vertex_finder

                # todo for now, do the one dim array
                #vertex_stuff = vertex_finder(tsi=time_list)
                vertex_stuff = [vertex_finder(e) for e in time_list]
                #vertex_stuff = np.array(list(vertex_stuff.values()))
                vertex_stuff = np.array(list(vertex_stuff[0].values())) # todo temp! we dont wanan acess that 0 element like this

                # todo check - started 1, + 1. needed as not indexes
                vertex_mse = vertex_stuff[np.remainder(np.arange(1, len(vertex_stuff) + 1), 2) == 0]
                vertex_ind = vertex_stuff[np.remainder(np.arange(1, len(vertex_stuff) + 1), 2) == 1]

                vert_new = np.flatnonzero(np.prod(abs(vertex_ind - vertices) >= (persistence / 2), axis=0) == 1) # todo apply prod per row

                # todo modified this to handle the above - in r, if array indexed when index doesnt exist, numeric of 0 returned
                if len(vert_new) == 0:
                    vertices = np.unique(np.sort(vertices))
                else:
                    vertices = np.unique(np.sort(np.append(vertices, vertex_ind[vert_new][np.argmax(vertex_mse[vert_new])])))

                # todo this whole thing needs a check, never fired
                if len(vert_new) == 1:
                    #for tl_indexer in np.arange(0, len(vertices)):  # todo check needs - 1
                        #time_list[[tl_indexer]] = np.arange(vertices[tl_indexer], (vertices[tl_indexer + 1] - 1))  # todo check remove - 1?
                    #time_list[[len(vertices)]] = np.arange(vertices[len(vertices)], len(tmpRun))  # todo check

                    for tl_indexer in np.arange(0, len(vertices) - 1):
                        time_list[tl_indexer] = np.arange(vertices[tl_indexer], (vertices[tl_indexer + 1]), dtype='int16')
                    #time_list[len(vertices)] = np.arange(vertices[len(vertices)], len(tmpRun))
                    time_list.append(np.arange(vertices[len(vertices) - 1], len(tmpRun), dtype='int16'))  # todo added - 1 to prevent out of index and append, added astype

                # increase vertex counter
                vert_indexer = vert_indexer + 1

                #seg_stop = np.prod(len(time_list) >= persistence)
                seg_stop = np.prod([len(e) for e in time_list] >= persistence)  # todo check

            # on principle, the second angle should indicate the restabilization!
            if len(vertices) >= 2:
                latestString = np.arange(0, vertices[1] + 1)  # todo added + 1 as we want to include extra index
            else:
                latestString = np.arange(0, len(tmpRun))

            # todo added astype int64 to prevent index float error
            latestString = latestString.astype('int64')

            runStep = np.min(np.flatnonzero(np.isnan(runKeeps)))
            runKeeps[runStep + latestString] = tmpRun[latestString]  # todo check removed - 1 is ok
            runKeepsResiduals[runStep + latestString] = tmpResiduals[latestString]  # todo check removed - 1 is ok

            myPixelCleanedTemp = np.delete(myPixelCleanedTemp, latestString)  # todo check empty array is ok down line
            DOYsCleanedTemp = np.delete(DOYsCleanedTemp, latestString)
            YearsCleanedTemp = np.delete(YearsCleanedTemp, latestString)
            DecimalYearsCleanedTemp = np.delete(DecimalYearsCleanedTemp, latestString)
            breakPointsTrackerCleanedTemp = np.delete(breakPointsTrackerCleanedTemp, latestString)
            indexer = indexer + 1

    # Post-Processing
    # At this point we have a vector of nonmissing EWMACD signals filtered by persistence
    dateByDate = backfill_missing(nonMissing=runKeeps, nonMissingIndex=cleanedInputIndex, withMissing=dateByDateWithMissing)
    dateByDateResiduals = backfill_missing(nonMissing=runKeepsResiduals, nonMissingIndex=cleanedInputIndex, withMissing=dateByDateWithMissing)

    if simple_output == True:
        output = {
            'dateByDate': dateByDate,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }
    else:
        output = {
            'dateByDate': dateByDate,
            'dateByDateResiduals': dateByDateResiduals,
            'Beta': BetaFirst,
            'breakPointsStart': breakPointsStart,
            'breakPointsEnd': breakPointsEnd
        }

    return output


def annual_summaries(Values, yearIndex, summaryMethod='date-by-date'):
    """annual summaries"""
    if summaryMethod == 'date-by-date':
        return Values

    finalOutput = np.repeat(np.nan, len(np.unique(yearIndex)))

    if summaryMethod == 'mean':
        # todo mean method, median, extreme, signmed mean methods... do when happy with above
        #finalOutput = (np.round(aggregate(Values, by=list(yearIndex), FUN=mean, na.rm = T)))$x
        ...


# todo check use of inf... not sure its purpose yet...
def EWMACD(ds, trainingPeriod='dynamic', trainingStart=None, testingEnd=None, trainingEnd=None, minTrainingLength=None, maxTrainingLength=np.inf, trainingFitMinimumQuality=0.8, numberHarmonicsSine=2, numberHarmonicsCosine='same as Sine', xBarLimit1=1.5, xBarLimit2= 20, lowthresh=0, _lambda=0.3, lambdaSigs=3, rounding=True, persistence_per_year=1, reverseOrder=False, summaryMethod='date-by-date', outputType='chart.values'):
    """main function"""

    # notify
    #

    # get day of years and associated year as int 16
    DOYs = ds['time.dayofyear'].data.astype('int16')
    Years = ds['time.year'].data.astype('int16')

    # check doys, years
    if len(DOYs) != len(Years):
        raise ValueError('DOYs and Years are not same length.')

    # if no training date provided, choose first year
    if trainingStart is None:
        trainingStart = np.min(Years)

    # if no testing date provided, choose last year + 1
    if testingEnd is None:
        testingEnd = np.max(Years) + 1

    # generate array of nans for every year between start of train and test period
    NAvector = np.repeat(np.nan, len(Years[(Years >= trainingStart) & (Years < testingEnd)]))

    # if not date to date, use year to year (?) may not need this
    if summaryMethod != 'date-by-date':
        num_nans = len(np.unique(Years[(Years >= trainingStart) & (Years < testingEnd)]))
        NAvector = np.repeat(np.nan, num_nans)

    # set cos harmonics value (default 2) to same as sine, if requested
    if numberHarmonicsCosine == 'same as Sine':
        numberHarmonicsCosine = numberHarmonicsSine

    # set simple output if chart values requested (?)
    if outputType == 'chart.values':
        simple_output = True

    # create per-pixel vectorised version of ewmacd per-pixel func
    def map_ewmacd_to_xr(pixel):
        
        try:
            change = EWMACD_pixel_date_by_date(myPixel=pixel,
                                               DOYs=DOYs,
                                               Years=Years,
                                               _lambda=_lambda,
                                               numberHarmonicsSine=numberHarmonicsSine,
                                               numberHarmonicsCosine=numberHarmonicsCosine,
                                               trainingStart=trainingStart,
                                               testingEnd=testingEnd,
                                               trainingPeriod=trainingPeriod,
                                               trainingEnd=trainingEnd,
                                               minTrainingLength=minTrainingLength,
                                               maxTrainingLength=maxTrainingLength,
                                               trainingFitMinimumQuality=trainingFitMinimumQuality,
                                               xBarLimit1=xBarLimit1,
                                               xBarLimit2=xBarLimit2,
                                               lowthresh=lowthresh,
                                               lambdaSigs=lambdaSigs,
                                               rounding=rounding,
                                               persistence_per_year=persistence_per_year,
                                               reverseOrder=reverseOrder,
                                               simple_output=simple_output)

            # get change per date from above
            change = change.get('dateByDate')

            # calculate summary method (todo set up for others than just date to date
            final_out = annual_summaries(Values=change,
                                         yearIndex=Years,
                                         summaryMethod=summaryMethod)

        except Exception as e:
            print('ERROR CHECK!')
            print(e)
            final_out = NAvector

        #return final_out
        return final_out
        

    # map ewmacd func to ds and compute it
    ds = xr.apply_ufunc(map_ewmacd_to_xr,
                        ds,
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                        vectorize=True)
    
    # rename veg_idx to change and convert to float32
    ds = ds.rename({'veg_idx': 'change'}).astype('float32')
    
    #return dataset
    return ds


# todo checks
def send_email_alert(sent_from=None, sent_to=None, subject=None, body_text=None, smtp_server=None, smtp_port=None, username=None, password=None):
    """
    """
    
    # check sent from
    if not isinstance(sent_from, str):
        raise TypeError('Sent from must be string.')
    elif not isinstance(sent_to, str):
        raise TypeError('Sent to must be string.')
    elif not isinstance(subject, str):
        raise TypeError('Subject must be string.')
    elif not isinstance(body_text, str):
        raise TypeError('Body text must be string.')     
    elif not isinstance(smtp_server, str):
        raise TypeError('SMTP server must be string.')        
    elif not isinstance(smtp_port, int):
        raise TypeError('SMTP port must be integer.')
    elif not isinstance(username, str):
        raise TypeError('Username must be string.')     
    elif not isinstance(password, str):
        raise TypeError('Password must be string.')
        
    # notify
    print('Emailing alert.')
    
    # construct header text
    msg = MIMEMultipart()
    msg['From'] = sent_from
    msg['To'] = sent_to
    msg['Subject'] = subject

    # construct body text (plain)
    mime_body_text = MIMEText(body_text)
    msg.attach(mime_body_text)

    # create secure connection with server and send
    with smtplib.SMTP(smtp_server, smtp_port) as server:

        # begin ttls
        server.starttls()

        # login to server
        server.login(username, password)

        # send email
        server.sendmail(sent_from, sent_to, msg.as_string())

        # notify
        print('Emailed alert area.')
                
    return


# temp
if __name__ == '__main__':

    ds_summary = xr.open_dataset(r"C:\Users\Lewis\Desktop\testing ds\ds_static.nc")

    #ds_summary = ds.median(['x', 'y'])

    output = EWMACD(ds=ds_summary, trainingPeriod='static')

    # todo - the order of dims is wrong on output
    ds_final_veg, _ = xr.broadcast(ds_summary, ds)   # want same median value for every pixel per image
    ds_final_change, _ = xr.broadcast(output , ds)   # want same change value for every pixel per image

    # ensure dimensions in original order
    ds_final_veg = ds_final_veg.transpose('time', 'y', 'x')
    ds_final_change = ds_final_change.transpose('time', 'y', 'x')
    print(output)
    
    
# what i need to do
# ui buttons:
# create project button - calls nrt create project that builds new, empty gdb with monitoring areas features
# create manage areas button - opens tab that has basic controls for adding, modifying, deleting monitoring areas? leave for last
# create monitor tool button - opens tab that allows for running tool


# todo :
# prevent 2011, 2012 deom being used

# nrt 

