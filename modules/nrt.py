# nrt
'''
Temp.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# set gdal global environ
# import os
# os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
# os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS '] = 'tif'
# os.environ['VSI_CACHE '] = 'TRUE'
# os.environ['GDAL_HTTP_MULTIRANGE '] = 'YES'
# os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES '] = 'YES'

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
import json
from scipy.signal import savgol_filter
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    sys.path.append('../../modules')
    import cog_odc

    sys.path.append('../../shared')
    import arc, satfetcher, tools
    
except:
    print('TEMP DELETE THIS TRYCATCH LATER ONCE WEMACD UP AND RUNNING')


def write_empty_json(filepath):
    with open(filepath, 'w') as f:
        json.dump([], f)




# meta
def load_json(filepath):
    """load json file"""
    
    # check if file exists 
    if not os.path.exists(filepath):
        raise ValueError('File does not exist: {}.'.format(filepath))
    
    # read json file
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    return data

def save_json(filepath, data):
    """save json file"""
    
    # check if file exists 
    if not os.path.exists(filepath):
        raise ValueError('File does not exist: {}.'.format(filepath))
    
    # read json file
    with open(filepath, 'w') as f:
        json.dump(data, f)
        
    return data


def get_item_from_json(filepath, global_id):
    """"""
    
    # check if file exists, else none
    if not os.path.exists(filepath):
        return
    elif global_id is None or not isinstance(global_id, str):
        return
    
    # read json file
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        return

    # check json data for item
    for item in data:
        if item.get('global_id') == global_id:
            return item

    # empty handed
    return
    

def get_latest_date_from_json_item(json_item):
    """"""
    
    # check if item is dict, else default
    if json_item is None:
        return '1980-01-01'
    elif not isinstance(json_item, dict):
        return '1980-01-01'
    
    # fetch latest date if exists, else default 
    dates = json_item.get('dates')
    if dates is not None and len(dates) > 0:
        return dates[-1]
    
    # otherwise default
    return '1980-01-01'


def get_json_array_via_key(json_item, key):
    """"""
    
    # check json item (none, not array, key not in it)
    if json_item is None:
        return np.array([])
    elif not isinstance(json_item, dict):
        return np.array([])
    elif key not in json_item:
        return np.array([])
    
    # fetch values if exists, else default 
    vals = json_item.get(key)
    if vals is not None and len(vals) > 0:
        return np.array(vals)
    
    # otherwise default
    return np.array([])

  
def extract_new_dates(ds, date_string):
    """"""
    
    # check if xarray is adequate 
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset is not of Xarray type.')
    elif 'time' not in ds:
        raise ValueError('Dataset does not have a time coordinate.')
    elif len(ds['time']) == 0:
        raise ValueError('Dataset is empty.')
    
    # check date string 
    if not isinstance(date_string, str):
        raise TypeError('Date must be of type string in format YYYY-MM-DD.')

    # get a list of all xr dates in string format 
    all_times = ds['time'].dt.strftime('%Y-%m-%d')
    
    # select only those times greater than current string 
    new_dates = ds['time'].where(all_times > date_string, drop=True)
    ds = ds.sel(time=new_dates)
    
    # return 
    return ds
    
    
def remove_spikes_xr(ds, user_factor=2, win_size=3):
    """
    Takes an xarray dataset containing vegetation index variable and removes outliers within 
    the timeseries on a per-pixel basis. The resulting dataset contains the timeseries 
    with outliers set to nan. Can work on datasets with or without existing nan values. Note:
    Zscore method will compute memory.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation 
        index variable (i.e. 'veg_index').
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. A higher factor 
        value results in few outliers (i.e. only the biggest outliers). Default factor is 2.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with all detected outliers in the
        veg_index variable set to nan.
    """
    
    # notify user
    print('Removing spike outliers.')
            
    # check xr type, dims
    if ds is None:
        raise ValueError('Dataset is empty.')
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset not an xarray type.')
    elif 'time' not in ds:
        raise ValueError('No time dimension in dataset.')

    # check if user factor provided
    if user_factor <= 0:
        print('User factor is less than 0, setting to 1.')
        user_factor = 1
                
    # calc cutoff val per pixel i.e. stdv of pixel multiply by user-factor 
    cutoffs = ds.std('time') * user_factor

    # calc mask of existing nan values (nan = True) in orig ds
    ds_mask = xr.where(ds.isnull(), True, False)

    # calc win size via num of dates in dataset
    #win_size = int(len(ds['time']) / 7)
    #win_size = int(win_size / int(len(ds.resample(time='1Y'))))
    
    

    if win_size < 3:
        win_size = 3
        print('Generated roll window size less than 3, setting to default (3).')
    elif win_size % 2 == 0:
        win_size = win_size + 1
        print('Generated roll window size is an even number, added 1 to make it odd ({0}).'.format(win_size))
    else:
        print('Generated roll window size is: {0}'.format(win_size))

    # calc rolling median for whole dataset
    ds_med = ds.rolling(time=win_size, center=True).median()

    # calc nan mask of start/end nans from roll, replace them with orig vals
    med_mask = xr.where(ds_med.isnull(), True, False)
    med_mask = xr.where(ds_mask != med_mask, True, False)
    ds_med = xr.where(med_mask, ds, ds_med)

    # calc abs diff between orig ds and med ds vals at each pixel
    ds_diffs = abs(ds - ds_med)

    # calc mask of outliers (outlier = True) where absolute diffs exceed cutoff
    outlier_mask = xr.where(ds_diffs > cutoffs, True, False)

    # shift values left and right one time index and combine, get mean and max for each window
    lefts = ds.shift(time=1).where(outlier_mask)
    rights = ds.shift(time=-1).where(outlier_mask)
    nbr_means = (lefts + rights) / 2
    nbr_maxs = xr.ufuncs.fmax(lefts, rights)

    # keep nan only if middle val < mean of neighbours - cutoff or middle val > max val + cutoffs
    outlier_mask = xr.where((ds.where(outlier_mask) < (nbr_means - cutoffs)) | 
                            (ds.where(outlier_mask) > (nbr_maxs + cutoffs)), True, False)

    # flag outliers as nan in original da
    ds = ds.where(~outlier_mask)


    # notify user and return
    print('Outlier removal successful.')
    return ds


def interp_nans_xr(ds):
    """"""
    
    # check if xarray is adequate 
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Dataset is not of Xarray type.')
    elif 'time' not in ds:
        raise ValueError('Dataset does not have a time coordinate.')
    elif len(ds['time']) == 0:
        raise ValueError('Dataset is empty.')
        
    try:
        # interpolate all values via linear interp
        ds = ds.interpolate_na('time')
    
        # remove any remaining nan (first and last indices, for e.g.)
        ds = ds.where(~ds.isnull(), drop=True)
    except:
        return
    
    return ds

    
def load_nc(ds):
    """this method loads existing dataset"""

    # check if file existd and try safe open
    if ds is not None:
        try:
            ds = ds.load()
            return ds
                    
        except:
            print('Could not open dataset, returning None.')
            return
    else:
        print('Dataset not provided')
        return



def remove_spikes_np(arr, user_factor=2, win_size=3):
    """
    Takes an numpy array containing vegetation index variable and removes outliers within 
    the timeseries on a per-pixel basis. The resulting dataset contains the timeseries 
    with outliers set to nan.
    
    Parameters
    ----------
    arr: numpy ndarray
        A one-dimensional array containing a vegetation index values.
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. A higher factor 
        value results in few outliers (i.e. only the biggest outliers). Default factor is 2.
    win_size: int
        Number of samples to include in rolling median window.
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with all detected outliers in the
        veg_index variable set to nan.
    """
    
    # notify user
    print('Removing spike outliers.')
            
    # check inputs
    if arr is None:
        raise ValueError('Array is empty.')

    # get nan mask (where nan is True)
    cutoffs = np.std(arr) * user_factor

    # do moving win median, back to numpy, fill edge nans 
    roll = pd.Series(arr).rolling(window=win_size, center=True)
    arr_win = roll.median().to_numpy()
    arr_med = np.where(np.isnan(arr_win), arr, arr_win)

    # calc abs diff between orig arr and med arr
    arr_dif = np.abs(arr - arr_med)

    # make mask where absolute diffs exceed cutoff
    mask = np.where(arr_dif > cutoffs, True, False)

    # get value left, right of each outlier and get mean
    l = np.where(mask, np.roll(arr, shift=1), np.nan)  # ->
    r = np.where(mask, np.roll(arr, shift=-1), np.nan) # <-
    arr_mean = (l + r) / 2
    arr_fmax = np.fmax(l, r)

    # mask if middle val < mean of neighbours - cutoff or middle val > max val + cutoffs 
    arr_outliers = ((np.where(mask, arr, np.nan) < (arr_mean - cutoffs)) | 
                    (np.where(mask, arr, np.nan) > (arr_fmax + cutoffs)))

    # apply the mask
    arr = np.where(arr_outliers, np.nan, arr)
    
    return arr


def interp_nan_np(arr):
    """equal to interpolate_na in xr"""

    # notify user
    print('Interpolating nan values.')
            
    # check inputs
    if arr is None:
        raise ValueError('Array is empty.')
        
    # get range of indexes
    idxs = np.arange(len(arr))
    
    # interpolate linearly 
    arr = np.interp(idxs, 
                    idxs[~np.isnan(arr)], 
                    arr[~np.isnan(arr)])
                    
    return arr








# meta, checks
def safe_load_nc(in_path):
    """Performs a safe load on local netcdf. Reads 
    NetCDF, loads it, and closes connection to file
    whilse maintaining data in memory"""
    
    # check if file existd and try safe open
    if os.path.exists(in_path):
        try:
            with xr.open_dataset(in_path) as ds_local:
                ds_local.load()
            
            return ds_local
                    
        except:
            print('Could not open cube: {}, returning None.'.format(in_path))
            return
    
    else:
        print('File does not exist: {}, returning None.'.format(in_path))
        return


# checks, metadata
def fetch_cube_data(collections, bands, start_dt, end_dt, bbox, resolution=30, ds_existing=None):
    """
    Takes a path to a netcdf file, a start and end date, bounding box and
    obtains the latest satellite imagery from DEA AWS. If an existing
    dataset is provided, the metadata from that is used to define the
    coordinates, etc. This function is used to 'grab' all existing cubes
    Note: currently this func contains a temp solution to the sentinel nrt 
    fmask band name issue
    
    Parameters
    ----------
    out_nc: str
        A path to an existing monitoring areas gdb feature class.
    """
    
    # checks
    # todo
    
    # notify
    print('Obtaining all satellite data for monitoring area.')

    # query stac endpoint
    try:
        items = cog_odc.fetch_stac_items_odc(stac_endpoint='https://explorer.sandbox.dea.ga.gov.au/stac', 
                                             collections=collections, 
                                             start_dt=start_dt, 
                                             end_dt=end_dt, 
                                             bbox=bbox,
                                             slc_off=False,    # never want slc-off data
                                             limit=250)
               
        # replace s3 prefix with https for each band - arcgis doesnt like s3
        items = cog_odc.replace_items_s3_to_https(items=items, 
                                          from_prefix='s3://dea-public-data', 
                                          to_prefix='https://data.dea.ga.gov.au')                            
    except:
        raise ValueError('Error occurred during fetching of stac metadata.')

    # build xarray dataset from stac data
    try:
        ds = cog_odc.build_xr_odc(items=items,
                                  bbox=bbox,
                                  bands=bands,
                                  crs=3577,
                                  resolution=resolution,
                                  group_by='solar_day',
                                  skip_broken_datasets=True,  
                                  like=ds_existing,
                                  chunks={})
                                  
        # prepare lazy ds with data type, type, time etc
        ds = cog_odc.convert_type(ds=ds, to_type='float32')
        ds = cog_odc.change_nodata_odc(ds=ds, orig_value=0, fill_value=-999)
        ds = cog_odc.fix_xr_time_for_arc_cog(ds)
    except:
        raise ValueError('Error occurred building of xarray dataset.')

    return ds


# checks, meta
def sync_new_and_old_cubes(ds_exist, ds_new, out_nc):
    """Takes two structurally idential xarray datasets and 
    combines them into one, where only new data from the latest 
    new dataset is combined with all of the old. Either way, a 
    file of this process is written to output path. This drives 
    the nrt on-going approach of the module."""
    
    # also set rasterio env variables
    rasterio_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS':'tif',
        'VSI_CACHE': True,
        'GDAL_HTTP_MULTIRANGE': 'YES',
        'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
    }
    
    # checks
    # 
    
    # if a new dataset provided only, write and load new
    if ds_exist is None and ds_new is not None:
        print('Existing dataset not provided. Creating and loading for first time.')
        
        # write netcdf file
        with rasterio.Env(**rasterio_env):
            ds_new = ds_new.astype('float32')
            tools.export_xr_as_nc(ds=ds_new, filename=out_nc)
            
        # safeload new dataset and return
        ds_new = safe_load_nc(out_nc)
        return ds_new
            
    elif ds_exist is not None and ds_new is not None:
        print('Existing and New dataset provided. Combining, writing and loading.')  
        
        # ensure existing is not locked via safe load (new always in mem)
        ds_exist = safe_load_nc(out_nc)
                
        # extract only new datetimes from new dataset
        dts = ds_exist['time']
        ds_new = ds_new.where(~ds_new['time'].isin(dts), drop=True)
        
        # check if any new images
        if len(ds_new['time']) > 0:
            print('New images detected ({}), adding and overwriting existing cube.'.format(len(ds_new['time'])))
                        
            # combine new with old (concat converts to float)
            ds_combined = xr.concat([ds_exist, ds_new], dim='time').copy(deep=True) 

            # write netcdf file
            with rasterio.Env(**rasterio_env):
                ds_combined = ds_combined.astype('float32')
                tools.export_xr_as_nc(ds=ds_combined, filename=out_nc)            
             
            # safeload new dataset and return
            ds_combined = safe_load_nc(out_nc)
            return ds_combined
        
        else:
            print('No new images detected, returning existing cube.')
            
            # safeload new dataset and return
            ds_exist = safe_load_nc(out_nc)
            return ds_exist

    else:
        raise ValueError('At a minimum, a new dataset must be provided.')
        return

  
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
    elif platform.lower() not in ['landsat', 'sentinel', 'sentinel_provisional']:
        raise ValueError('Platform must be Landsat or Sentinel.')
        
    # set up dict
    params = {}
    
    # get porams depending on platform
    if platform.lower() == 'landsat':
        
        # get collections
        collections = [
            'ga_ls5t_ard_3', 
            'ga_ls7e_ard_3', 
            'ga_ls8c_ard_3',
            #'ga_ls7e_ard_provisional_3',  # will always be slc-off
            'ga_ls8c_ard_provisional_3']
        
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
        
    # the product 3 is not yet avail on dea. we use s2 for now.
    elif platform.lower() == 'sentinel':
        
        # get collections
        collections = [
            's2a_ard_granule', 
            's2b_ard_granule',
            'ga_s2am_ard_provisional_3', 
            'ga_s2bm_ard_provisional_3'
            ]
        
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
        
    # prepare path toe xpected json file
    in_path = os.path.dirname(in_feat)
    in_path = os.path.splitext(in_path)[0]
    in_path = os.path.dirname(in_path)
    in_data_path = os.path.join(in_path, 'data.json')
    
    # check json file exists
    if not os.path.exists(in_data_path):
        print('Associated data json not found, flagging as invalid.')
        is_valid = False

    # if we've made it, check featureclass
    if is_valid:
        try:
            # get feature
            driver = ogr.GetDriverByName("OpenFileGDB")
            data_source = driver.Open(os.path.dirname(in_feat), 0)
            lyr = data_source.GetLayer('monitoring_areas')
            
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
                
                
            # check if feature has required fields
            fields = [field.name for field in lyr.schema]
            required_fields = ['area_id', 'platform', 's_year', 'e_year', 
                               'index', 'persistence', 'rule_1_min_conseqs', 
                               'rule_1_inc_plateaus', 'rule_2_min_stdv', 
                               'rule_2_bidirectional', 'rule_3_num_zones',
                               'ruleset', 'alert', 'alert_direction', 'email', 
                               'ignore', 'color', 'global_id']
            
            # check if all fields in feat
            if not all(f in fields for f in required_fields):
                raise ValueError('Not all required fields in monitoring shapefile.')
                is_valid = False
                
            # close data source
            data_source.Destroy()
            
        except:
            print('Could not open monitoring area feature, flagging as invalid.')
            is_valid = False
            data_source.Destroy()

    # return
    return is_valid
 
 
# todo - meta, consider other fields
def validate_monitoring_area(row):
    """
    Does relevant checks for information for a
    single monitoring area.
    """
    
    # check if row is tuple
    if not isinstance(row, tuple):
        raise ValueError('Row must be of type tuple.')
            
    # parse row info
    area_id = row[0]
    platform = row[1]
    s_year = row[2]
    e_year = row[3]
    index = row[4]
    persistence = row[5]
    rule_1_min_conseqs = row[6]
    rule_1_inc_plateaus = row[7]
    rule_2_min_stdv = row[8]
    rule_2_bidirectional = row[9]
    rule_3_num_zones = row[10]
    ruleset = row[11]
    alert = row[12]
    alert_direction = row[13]
    email = row[14]
    ignore = row[15]
    
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
    
    # check if persistence is accepptable
    if persistence is None:
        print('No persistence exists, flagging as invalid.')
        return False
    elif persistence < 0.001 or persistence > 9.999:
        print('Persistence must be before 0.0001 and 9.999, flagging as invalid.')
        return False

    # check if rule_1_min_conseqs is accepptable
    if rule_1_min_conseqs is None:
        print('No rule_1_min_conseqs exists, flagging as invalid.')
        return False
    elif rule_1_min_conseqs < 0 or rule_1_min_conseqs > 99:
        print('Rule_1_min_conseqs must be between 0 and 99, flagging as invalid.')
        return False
    
    # check if rule_1_min_conseqs is accepptable
    if rule_1_inc_plateaus is None:
        print('No rule_1_inc_plateaus exists, flagging as invalid.')
        return False
    elif rule_1_inc_plateaus not in ['Yes', 'No']:
        print('Rule_1_inc_plateaus must be Yes or No, flagging as invalid.')
        return False    
    
    # check if rule_2_min_stdv is accepptable
    if rule_2_min_stdv is None:
        print('No rule_2_min_stdv exists, flagging as invalid.')
        return False
    elif rule_2_min_stdv < 0 or rule_2_min_stdv > 99:
        print('Rule_2_min_stdv must be between 0 and 99, flagging as invalid.')
        return False    

    # check if rule_2_bidirectional is accepptable
    if rule_2_bidirectional is None:
        print('No rule_2_bidirectional exists, flagging as invalid.')
        return False
    elif rule_2_bidirectional not in ['Yes', 'No']:
        print('Rule_2_bidirectional must be Yes or No, flagging as invalid.')
        return False     

    # check if rule_2_bidirectional is accepptable
    if rule_3_num_zones is None:
        print('No rule_3_num_zones exists, flagging as invalid.')
        return False
    elif rule_3_num_zones < 0 or rule_2_min_stdv > 99:
        print('Rule_3_num_zones must be between 0 and 99, flagging as invalid.')
        return False       

    # check if ruleset is accepptable   
    if ruleset is None:
        print('No ruleset exists, flagging as invalid.')
        return False
    
    # check if alert is accepptable
    if alert is None:
        print('No alert exists, flagging as invalid.')
        return False
    elif alert not in ['Yes', 'No']:
        print('Alert must be Yes or No, flagging as invalid.')
        return False
    
    # check if alert_direction is accepptable
    if alert_direction is None:
        print('No alert_direction exists, flagging as invalid.')
        return False
    elif alert_direction not in ['Incline Only', 'Decline Only', 'Both']:
        print('Alert_direction must be Incline Only, Decline Only, Both, flagging as invalid.')
        return False  

    # check if alert_direction is accepptable
    if email is not None:
        if '@' not in email or '.' not in email:
            print('No @ or . character in email exists, flagging as invalid.')
            return False

    # check if ignore is accepptable
    if ignore is None:
        print('No ignore exists, flagging as invalid.')
        return False
    elif ignore not in ['Yes', 'No']:
        print('Ignore must be Yes or No, flagging as invalid.')
        return False    

    # all good!
    return True 
 
 
# todo do checks, do meta
def mask_xr_via_polygon(ds, geom, mask_value=1):
    """
    geom object from gdal
    x, y = arrays of coordinates from xr dataset
    bbox 
    transform from geobox
    ncols, nrows = len of x, y
    """
    
    # check dataset
    if 'x' not in ds or 'y' not in ds:
        raise ValueError('Dataset has no x or y dimensions.')
    elif not hasattr(ds, 'geobox'):
        raise ValueError('Dataset does not have a geobox.')
        
    # extract raw x and y value arrays, bbox, transform and num col, row
    x, y = ds['x'].data, ds['y'].data
    bbox = ds.geobox.extent.boundingbox
    transform = ds.geobox.transform
    ncols, nrows = len(ds['x']), len(ds['y'])
    
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
    
    
# deprecated! meta
def reproject_ogr_geom(geom, from_epsg=3577, to_epsg=4326):
    """
    """
    
    # check if ogr layer type
    if not isinstance(geom, ogr.Geometry):
        raise TypeError('Layer is not of ogr Geometry type.')
        
    # check if epsg codes are ints
    if not isinstance(from_epsg, int):
        raise TypeError('From epsg must be integer.')
    elif not isinstance(to_epsg, int):
        raise TypeError('To epsg must be integer.')
        
    # notify
    print('Reprojecting layer from EPSG {} to EPSG {}.'.format(from_epsg, to_epsg))
            
    try:
        # init spatial references
        from_srs = osr.SpatialReference()
        to_srs = osr.SpatialReference()
    
        # set spatial references based on epsgs (inplace)
        from_srs.ImportFromEPSG(from_epsg)
        to_srs.ImportFromEPSG(to_epsg)
        
        # transform
        trans = osr.CoordinateTransformation(from_srs, to_srs)
        
        # reproject
        geom.Transform(trans)
        
    except: 
        raise ValueError('Could not transform ogr geometry.')
        
    # notify and return
    print('Successfully reprojected layer.')
    return geom


# meta, checks - DYNAMIC DISABLED, NOT USING
def build_change_cube(ds, training_start_year=None, training_end_year=None, persistence_per_year=1, add_extra_vars=True):
    """
    """

    # checks

    # notify
    print('Detecting change via static and dynamic methods.')
    
    # get attributes from dataset
    data_attrs = ds.attrs
    band_attrs = ds[list(ds.data_vars)[0]].attrs
    sref_attrs = ds['spatial_ref'].attrs
    
    # sumamrise each image to a single median value
    ds_summary = ds.median(['x', 'y'], keep_attrs=True)

    # perform static ewmacd and add as new var
    print('Generating static model')
    ds_summary['static'] = EWMACD(ds=ds_summary, 
                                  trainingPeriod='static',
                                  trainingStart=training_start_year,
                                  trainingEnd=training_end_year,
                                  persistence_per_year=persistence_per_year)['veg_idx']
    
    # perform dynamic ewmacd and add as new var
    #print('Generating dynamic model')
    #ds_summary['dynamic'] = EWMACD(ds=ds_summary, trainingPeriod='dynamic',
                                   #trainingStart=training_start_year,
                                   #persistence_per_year=persistence_per_year)['veg_idx']

    # rename original veg_idx to summary
    #ds_summary = ds_summary.rename({'veg_idx': 'summary'})

    # broadcast summary back on to original dataset and order axes
    ds_summary, _ = xr.broadcast(ds_summary, ds)
    ds_summary = ds_summary.transpose('time', 'y', 'x')
    
    # add extra empty vars (zones, cands, conseqs) to dataset if new
    if add_extra_vars:
        for var in ['zones', 'cands_inc', 'cands_dec', 'consq_inc', 'consq_dec']:
            if var not in ds_summary:
                ds_summary[var] = xr.full_like(ds_summary['veg_idx'], np.nan)    
                
    # append attrbutes back on
    ds_summary.attrs = data_attrs
    ds_summary['spatial_ref'].attrs = sref_attrs
    for var in list(ds_summary.data_vars):
        ds_summary[var].attrs = band_attrs

    # notify and return
    print('Successfully created detection cube')
    return ds_summary

# meta
def create_email_dicts(row_count):
    """meta"""
    
    if row_count is None or row_count == 0:
        print('No rows to build email dictionaries.')
        return
    
    # setup email contents list
    email_contents = []
            
    # pack list full of 'empty' dicts
    for i in range(row_count):
    
        # set up empty email dict
        email_dict = {
            'area_id': None,
            's_year': None,
            'e_year': None,
            'index': None,
            'ruleset': None,
            'alert': None,
            'alert_direction': None,
            'email': None,
            'ignore': None,
            'triggered': None
        }
    
        email_contents.append(email_dict)
        
    return email_contents


# todo checks, meta
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


# meta, checks 
def smooth_change(arr):
    """
    Basic func to smooth change signal using
    a savgol filter with win size 3. works best
    for our data. minimal smoothing, yet removes
    small spikes.
    """
    
    # check if arr larger than win size
    if len(arr) <= 3:
        print('Cannot smooth signal less than window size, returning raw array.')
        return arr
    
    # smooth using savgol filter with win size 3
    return savgol_filter(arr, 3, 1)


# meta, checks
def count_runs(arr, vector_value=-1):
    
    # count continuous runs for requested vector value
    arr_masked = np.where(arr == vector_value, 1, 0)           # binarise
    arr_extended = np.concatenate(([0], arr_masked, [0]))        # pad array with empty begin and end elements
    idx = np.flatnonzero(arr_extended[1:] != arr_extended[:-1])  # get start and end indexes
    arr_extended[1:][idx[1::2]] = idx[::2] - idx[1::2]           # grab breaks, prev - current, also trim extended elements
    arr_counted = arr_extended.cumsum()[1:-1]                    # apply cumulative sum
    
    return arr_counted


# meta - deprecated! no longer using
def fill_zeros_with_last(arr):
    """
    forward fills differences of 0 after a 
    decline or positive flag.
    """
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    
    return arr[prev]


# meta, check
def classify_signal(arr):
    """
    takes array, calcs differences between on-going values and classifies 
    in 0 (no change), 1 (incline), -1 (decline), 2 (plateau after incline), 
    -2 (plateau after decline).
    """
    
    # classify stable (0), incline (1), decline (-1)
    diffs = np.diff(arr, prepend=arr[0])
    diffs = np.where(diffs == 0, 0, diffs)  # inc. for clarity)
    diffs = np.where(diffs > 0, 1, diffs)
    diffs = np.where(diffs < 0, -1, diffs)
    
    # classify plateau post-incline (2) and post-decline (-2)
    for i in np.arange(1, len(diffs)):
        if diffs[i] == 0 and diffs[i - 1] > 0:
            diffs[i] = 2
        elif diffs[i] == 0 and diffs[i - 1] < 0:
            diffs[i] = -2
            
    return diffs

  
# meta, stable zone?
def get_stdv_from_zone(num_zones=1):
    """
    """
    
    # checks
    if num_zones is None or num_zones <= 0:
        print('Number of zones must be greater than 0. Setting to 1.')
        num_zones = 1
    
    # multiple by zones (3 per zone)
    std_jumped = num_zones * 3
    
    # todo include stable zone -1 to 1?
    #
    
    return std_jumped

  
# meta checks
def apply_rule_one(arr, direction='decline', min_consequtives=3, max_consequtives=None, inc_plateaus=False):
    """
    takes array of smoothed change output, classifies
    array into binary 1s 0s depending on vector value,
    calculates consequtive runs, then thresholds out 
    minimum consequtive values and maximum consequtives. note min_consequitive
    is inclusive of the value added. if plateaus are
    included in runs, plateaus following declines/incline 
    flags will also be considered in runs.
    """
    
    # check direction
    if direction not in ['incline', 'decline']:
        raise ValueError('Direction must be incline or decline.')
        
    # check min and max consequtives 
    if min_consequtives is not None and min_consequtives < 0:
        print('Minimum consequtives must be either None or >= 0. Setting to None.')
        min_consequtives = None

    # check min and max consequtives 
    if max_consequtives is not None and max_consequtives <= 0:
        print('Maximum consequtives must be either None or > 0. Setting to 1.')
        max_consequtives = 1
        
    # get required directional values
    dir_values = [1, 2] if direction == 'incline' else [-1, -2]
        
    # classify signal into -2, -1, 0, 1, 2
    diffs = classify_signal(arr=arr)
    
    # generate runs
    arr_runs = count_runs(arr=diffs, 
                              vector_value=dir_values[0])
    
    # flag plateaus if they are after a decline
    if inc_plateaus:
        for i in np.arange(1, len(arr_runs)):
            if arr_runs[i] == 0 and arr_runs[i - 1] != 0:
                if diffs[i] == dir_values[1]:
                    arr_runs[i] = arr_runs[i - 1] + 1
                    
    # threshold out less than specific number consequtives
    if min_consequtives is not None:
        arr_runs = np.where(arr_runs >= min_consequtives, arr_runs, 0)

    # threshold out more than specific number consequtives
    if max_consequtives is not None:
        arr_temp = np.where(arr_runs > 0, 1, 0)
        arr_temp = count_runs(arr=arr_temp, vector_value=1)
        arr_runs = np.where(arr_temp > max_consequtives, 0, arr_runs)
        
    # replace 0s with nans and re-count to merge runs and plateaus
    arr_counted = np.where(arr_runs != 0, 1, 0)
    arr_counted = count_runs(arr=arr_counted, vector_value=1)
    
    # finally, replace 0s with nans
    arr_counted = np.where(arr_runs == 0, np.nan, arr_runs)
    
    return arr_counted


# meta checks 
def apply_rule_two(arr, direction='decline', min_stdv=1, operator='<=', bidirectional=False):
    """
    takes array of smoothed change output and thresholds out
    any values outside of a specified minimum zone e.g. 1.
    """
    
    # check direction
    if direction not in ['incline', 'decline']:
        raise ValueError('Direction must be incline or decline.')
        
    # checks
    if operator not in ['<', '<=', '>', '>=']:
        raise ValueError('Operator must be <, <=, >, >=')
        
    # set stdv to negative if decline
    if direction == 'decline':
        min_stdv = min_stdv * -1
        
    # check operator matches direction 
    if direction == 'incline' and operator not in ['>', '>=']:
        print('Operator must be > or >= when using incline. Setting to >=.')
        operator = '>='
    elif direction == 'decline' and operator not in ['<', '<=']:
        print('Operator must be < or <= when using decline. Setting to <=.')
        operator = '<='

    # operate based on 
    if bidirectional:
        print('Bidrectional enabled, ignoring direction.')
        arr_abs = np.abs(arr)
        
        if '=' in operator:
            arr_thresholded = np.where(arr_abs >= abs(min_stdv), arr, np.nan)
        else:
            arr_thresholded = np.where(arr_abs > abs(min_stdv), arr, np.nan)
        
    elif operator == '<':
        arr_thresholded = np.where(arr < min_stdv, arr, np.nan)
    elif operator == '<=':
        arr_thresholded = np.where(arr <= min_stdv, arr, np.nan)
    elif operator == '>':
        arr_thresholded = np.where(arr > min_stdv, arr, np.nan)
    elif operator == '>=':
        arr_thresholded = np.where(arr >= min_stdv, arr, np.nan)
        
    return arr_thresholded

  
# meta checks
def apply_rule_three(arr, direction='decline', num_stdv_jumped=3, min_consequtives=3, max_consequtives=3):
    """
    takes array of smoothed (or raw) change output and detects large, multi zone
    jumps. candidates only registered if a specific number of post jump
    runs detected (set min_consequtives to 0 for any spike regardless of runs).
    jump_size is number of zones required to jump - default is 3 stdv (1 zone). max
    consequtives will cut the run off after certain number of indices detected.
    """
    
    # checks
    if direction not in ['incline', 'decline']:
        raise ValueError('Direction must be incline or decline.')
        
    # prepare max consequtives
    if max_consequtives <= 0:
        print('Warning, max consequtives must be > 0. Resetting to three.')
        max_consequtives = 3
            
    # get diffs
    diffs = np.diff(np.insert(arr, 0, arr[0]))
    
    # threshold by magnitude of jump
    if direction == 'incline':
        arr_jumps = diffs > num_stdv_jumped
    elif direction == 'decline':
        arr_jumps = diffs < (num_stdv_jumped * -1)
        
    # iter each spike index and detect post-spike runs (as 1s)
    indices = []
    for i in np.where(arr_jumps)[0]:

        # loop each element in array from current index and calc diff
        for e, v in enumerate(arr[i:]):
            diff = np.abs(np.abs(arr[i]) - np.abs(v))

            # if diff is less than certain jump size record it, else skip
            # todo diff < 3 is check to see if stays within one zone of devs
            if diff < 3 and e <= max_consequtives:
                indices.append(i + e)
            else:
                break 
                
    # set 1 to every flagged index, 0 to all else
    arr_masked = np.zeros_like(arr)
    arr_masked[indices] = 1
    
    # count continuous runs for requested vector value
    arr_extended = np.concatenate(([0], arr_masked, [0]))        # pad array with empty begin and end elements
    idx = np.flatnonzero(arr_extended[1:] != arr_extended[:-1])  # get start and end indexes
    arr_extended[1:][idx[1::2]] = idx[::2] - idx[1::2]           # grab breaks, prev - current, also trim extended elements
    arr_counted = arr_extended.cumsum()[1:-1]                    # apply cumulative sum
    
    # threshold out specific run counts
    if min_consequtives is not None:
        arr_counted = np.where(arr_counted >= min_consequtives, arr_counted, 0)

    # replace 0s with nans
    arr_counted = np.where(arr_counted != 0, arr_counted, np.nan)
    
    return arr_counted


# meta checks todo count runs???
def apply_rule_combo(arr_r1, arr_r2, arr_r3, ruleset='1&2|3'):
    """
    take pre-generated rule arrays and combine where requested.
    """
    
    allowed_rules = [
        '1', '2', '3', '1&2', '1&3', '2&3', 
        '1|2', '1|3', '2|3', '1&2&3', '1|2&3',
        '1&2|3', '1|2|3']
    
    # checks
    if ruleset not in allowed_rules:
        raise ValueError('Ruleset set is not supported.')
    
    # convert rule arrays to binary masks
    arr_r1_mask = ~np.isnan(arr_r1)
    arr_r2_mask = ~np.isnan(arr_r2)
    arr_r3_mask = ~np.isnan(arr_r3)
        
    # set signular rules
    if ruleset == '1':
        arr_comb = np.where(arr_r1_mask, 1, 0)
    elif ruleset == '2':
        arr_comb = np.where(arr_r2_mask, 1, 0)
    elif ruleset == '3':
        arr_comb = np.where(arr_r3_mask, 1, 0)
    
    # set combined dual ruleset
    elif ruleset == '1&2':
        arr_comb = np.where(arr_r1_mask & arr_r2_mask, 1, 0)
    elif ruleset == '1&3':
        arr_comb = np.where(arr_r1_mask & arr_r3_mask, 1, 0)        
    elif ruleset == '2&3':
        arr_comb = np.where(arr_r2_mask & arr_r3_mask, 1, 0)             
        
    # set either or dual ruleset
    elif ruleset == '1|2':
        arr_comb = np.where(arr_r1_mask | arr_r2_mask, 1, 0)  
    elif ruleset == '1|3':
        arr_comb = np.where(arr_r1_mask | arr_r3_mask, 1, 0) 
    elif ruleset == '2|3':
        arr_comb = np.where(arr_r2_mask | arr_r3_mask, 1, 0)     
        
    # set combined several ruleset
    elif ruleset == '1&2&3':  
        arr_comb = np.where(arr_r1_mask & arr_r2_mask & arr_r3_mask, 1, 0)
    elif ruleset == '1|2&3':  
        arr_comb = np.where(arr_r1_mask | (arr_r2_mask & arr_r3_mask), 1, 0)        
    elif ruleset == '1&2|3':  
        arr_comb = np.where((arr_r1_mask & arr_r2_mask) | arr_r3_mask, 1, 0)  
    elif ruleset == '1|2|3':  
        arr_comb = np.where(arr_r1_mask | arr_r2_mask | arr_r3_mask, 1, 0)
        
    # count runs todo
    
        
    # replace 0s with nans
    arr_comb = np.where(arr_comb != 0, arr_comb, np.nan)
        
    return arr_comb


# meta, checks, check rule 2 operator is right
def get_candidates(vec, direction='incline', min_consequtives=3, max_consequtives=None, inc_plateaus=False, min_stdv=1, num_zones=1, bidirectional=False, ruleset='1&2|3', binarise=True):
    """
    min_conseq = rule 1
    max_conseq = rule 1
    inc_plateaus = rule 1
    min_stdv = rule 2
    num_zones = rule 3
    rulset = all
    binarise = set out to 1,0 not 1, nan
    """
    
    # checks
    # min_conseq >= 0
    # max_conseq >= 0, > min, or None
    # min_stdv >= 0 
    # operator only > >= < <=
    
    # set up parameters reliant on direction
    if direction == 'incline':
        operator = '>='
    elif direction == 'decline':
        operator = '<='
    
    # calculate rule 1 (consequtive runs)
    print('Calculating rule one: consequtive runs {}.'.format(direction))
    vec_rule_1 = apply_rule_one(arr=vec,
                                direction=direction,
                                min_consequtives=min_consequtives,        # min consequtive before candidate
                                max_consequtives=max_consequtives,        # max num consequtives before reset
                                inc_plateaus=inc_plateaus)                # include plateaus after decline
    
    # calculate rule 2 (zone threshold)
    print('Calculating rule two: zone threshold {}.'.format(direction))
    vec_rule_2 = apply_rule_two(arr=vec,
                                direction=direction,
                                min_stdv=min_stdv,                        # min stdv threshold
                                operator=operator,
                                bidirectional=bidirectional)              # operator e.g. <=
        
    # calculate rule 3 (jumps) increase
    print('Calculating rule three: sharp jump {}.'.format(direction))
    num_stdvs = get_stdv_from_zone(num_zones=num_zones)
    vec_rule_3 = apply_rule_three(arr=vec,
                                  direction=direction,
                                  num_stdv_jumped=num_stdvs,               
                                  min_consequtives=min_consequtives,
                                  max_consequtives=min_consequtives)      # careful !!!!!!

    
    # combine rules 1, 2, 3 decreasing
    print('Combining rule 1, 2, 3 via ruleset {}.'.format(ruleset))
    vec_rules_combo = apply_rule_combo(arr_r1=vec_rule_1, 
                                       arr_r2=vec_rule_2, 
                                       arr_r3=vec_rule_3, 
                                       ruleset=ruleset)
    
    # binarise 1, nan to 1, 0 if requested
    if binarise:
        vec_rules_combo = np.where(vec_rules_combo == 1, 1.0, 0.0)
    
    
    return vec_rules_combo


# meta, checks
def reclassify_signal_to_zones(arr):
    """
    takes a smoothed (or raw) ewmacd change detection
    signal and classifies into 1 of 11 zones based on the
    stdv values. this is used to help flag and colour
    outputs for nrt monitoring. Outputs include 
    zone direction information in way of sign (-/+).
    """   

    # set up zone ranges (stdvs)
    zones = [
        [0, 1],    # zone 1 - from 0 to 1 (+/-)
        [1, 3],    # zone 2 - between 1 and 3 (+/-)
        [3, 5],    # zone 3 - between 3 and 5 (+/-)
        [5, 7],    # zone 4 - between 5 and 7 (+/-)
        [7, 9],    # zone 5 - between 7 and 9 (+/-)
        [9, 11],   # zone 6 - between 9 and 11 (+/-)
        [11, 13],  # zone 7 - between 11 and 13 (+/-)
        [13, 15],  # zone 8 - between 13 and 15 (+/-)
        [15, 17],  # zone 9 - between 15 and 17 (+/-)
        [17, 19],  # zone 10 - between 17 and 19 (+/-)
        [19]       # zone 11- above 19 (+/-)
    ]

    # create template vector
    vec_temp = np.full_like(arr, fill_value=np.nan)

    # iter zones
    for i, z in enumerate(zones, start=1):

        # 
        if i == 1:
            vec_temp[np.where((arr >= z[0]) & (arr <= z[1]))] = i
            vec_temp[np.where((arr < z[0]) & (arr >= z[1] * -1))] = i * -1

        elif i == 11:       
            vec_temp[np.where(arr > z[0])] = i
            vec_temp[np.where(arr < z[0] * -1)] = i * -1

        else:
            vec_temp[np.where((arr > z[0]) & (arr <= z[1]))] = i
            vec_temp[np.where((arr < z[0] * -1) & (arr >= z[1] * -1))] = i * -1
        
    return vec_temp


# need params from feat, conseq count e.g., 3)
def prepare_and_send_alert(ds, back_idx=-2, send_email=False):
    """
    ds = change dataset with required vars
    back_idx = set backwards index (-1 is latest image, -2 is second last, etc)
    """
    
    # check if we have all vars required

    # get second latest date
    latest_date = ds['time'].isel(time=back_idx)
    latest_date = latest_date.dt.strftime('%Y-%m-%d %H:%M:%S')
    latest_date = str(latest_date.values)

    # get latest zone
    latest_zone = ds['zones'].isel(time=back_idx)
    latest_zone = latest_zone.mean(['x', 'y']).values

    # get latest incline candidate
    latest_inc_candidate = ds['cands_inc'].isel(time=back_idx)
    latest_inc_candidate = latest_inc_candidate.mean(['x', 'y']).values

    # get latest decline candidate
    latest_dec_candidate = ds['cands_dec'].isel(time=back_idx)
    latest_dec_candidate = latest_dec_candidate.mean(['x', 'y']).values

    # get latest incline consequtives
    latest_inc_consequtives = ds['consq_inc'].isel(time=back_idx)
    latest_inc_consequtives = latest_inc_consequtives.mean(['x', 'y']).values

    # get latest incline consequtives
    latest_dec_consequtives = ds['consq_dec'].isel(time=back_idx)
    latest_dec_consequtives = latest_dec_consequtives.mean(['x', 'y']).values
    
    
    # alert user via ui and python before email
    if latest_inc_candidate == 1:
        print('- ' * 10)
        print('Alert! Monitoring Area {} has triggered the alert system.'.format('<placeholder>'))
        print('An increasing vegetation trajectory has been detected.')
        print('Alert triggered via image captured on {}.'.format(str(latest_date)))
        print('Area is in zone {}.'.format(int(latest_zone)))
        print('Increase has been on-going for {} images (i.e., dates).'.format(int(latest_inc_consequtives)))       
        print('')

        
    elif latest_dec_candidate == 1:
        print('- ' * 10)
        print('Alert! Monitoring Area {} has triggered the alert system.'.format('<placeholder>'))
        print('An decreasing vegetation trajectory has been detected.')
        print('Alert triggered via image captured  {}.'.format(str(latest_date)))
        print('Area is in zone {}.'.format(int(latest_zone)))
        print('Decrease has been on-going for {} images (i.e., dates).'.format(int(latest_dec_consequtives)))
        print('')  
        
    else:
        print('- ' * 10)
        print('No alert was triggered for Monitoring Area: {}.'.format('<placeholder>'))
        print('')
        
    # if requested, send email
    if send_email:
        print('todo...')




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
            print('Could not train model adequately, please add more years.')
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
    ds = ds.astype('float32')
    
    #return dataset
    return ds

