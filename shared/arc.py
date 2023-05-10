# arc
'''
This script contains functions that are entirely for use
within the ArcGIS Pro/arcpy environment. These are really just
helper functions for managing UI and IO type operations - an
middle-manager between the ArcGIS Pro UI and the underlying
tenement tool modules. Only for use in ArcGIS Pro UI. 

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import arcpy
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


def datetime_to_numpy(dt):
    """
    Quick function to convert a datetime
    object into a numpy format datetime64 for use in
    indexing and querying xr time dimensions.
    """
    
    return np.datetime64(dt)


def get_layer_bbox(feat_path):
    """
    Basic function that takes a arcpy feature layer 
    and extracts extent of layer in WGS84 lat/lons. Used
    as the bounding box in stac query.
    
    Parameters
    -------------
    feat_path : str
        A str representing either a full path to a feature 
        or a feature name current in a map in ArcGIS Pro.
    
    Returns
    ----------
    A list in form of west, south, east, north coordinates 
    in WGS84.
    """
    
    # check layer is valid
    if feat_path is None:
        raise ValueError('No feature path provided.')
    elif not isinstance(feat_path, str):
        raise ValueError('Feature is not a string.')
        
    try:
        # get feature definition
        desc = arcpy.Describe(feat_path)
    except Exception as e:
        raise ValueError(e)
        
    # check layer definition is valid
    if not desc.shapeType == 'Polygon':
        raise TypeError('Feature is not a polygon type.')
    elif not desc.hasSpatialIndex:
        raise ValueError('Feature has no coordinate system.')
        
    # create spatial reference system for wgs84
    srs = arcpy.SpatialReference(4326)
    
    # get extents of selected or all features
    with arcpy.da.SearchCursor(feat_path, ['OID@', 'SHAPE@']) as cursor:
        exts = [r[1].extent for r in cursor]
        
    # project extent(s) to wgs84
    exts = [e.projectAs(srs) for e in exts]
    
    # construct bounding box
    xmin = min([e.XMin for e in exts])
    ymin = min([e.YMin for e in exts])
    xmax = max([e.XMax for e in exts])
    ymax = max([e.YMax for e in exts])
    bbox = [xmin, ymin, xmax, ymax]
    
    # check if something exists
    if len(bbox) == 0:
        raise ValueError('Bounding box is empty.')
        
    return bbox


def prepare_collections_list(in_platform):
    """
    Basic function that takes the name of a satellite 
    platform (Landsat or Sentinel) and converts 
    into collections name recognised by dea aws. Used only
    for ArcGIS Pro UI/Toolbox.
    
    Parameters
    -------------
    in_platform : str
        Name of satellite platform, Landsat or Sentinel.
    
    Returns
    ----------
    A string of dea aws collections associated with input 
    satellite.
    """
    
    # checks
    if in_platform not in ['Landsat', 'Sentinel', 'Sentinel 2A', 'Sentinel 2B']:
        raise ValueError('Platform must be Landsat or Sentinel.')
        
    # prepare collections
    if in_platform == 'Landsat':
        return ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3']
    
    elif in_platform == 'Sentinel':
        return ['ga_s2am_ard_3', 'ga_s2bm_ard_3']
    
    elif in_platform == 'Sentinel 2A':
        return ['s2a_ard_granule']
        
    elif in_platform == 'Sentinel 2B':
        return ['s2b_ard_granule']


def prepare_band_names(in_bands, in_platform):
    """
    Basic function that takes the name of raw
    Landsat or Sentinel satellite band names (e.g.,
    Blue;Green;Red;NIR) and translates them into
    naming convention recognised by dea aws (e.g., 
    nbart_blue, nbart_green, nbart_red, nbart_nir_1).
    Depends on platform (set via in_platform), as 
    dea aws uses different band names between Landsat
    and Sentinel, for some reason.
    
    Parameters
    -------------
    in_bands : str
        The raw names of satellite bands known to most
        users and embedded in the ArcGIS Pro UI. For
        example: Blue;Green;Red.
    in_platform : str
        A string of the name of current platform the bands
        are associated with (Landsat or Sentinel).
    
    Returns
    ----------
    A list of dea aws friendly band names.
    """
    
    # check some inputs
    if in_bands is None:
        raise ValueError('No bands in band request.')
    
    elif not isinstance(in_bands, str):
        raise TypeError('Expecting a string of band names.')
    
    elif in_platform not in ['Landsat', 'Sentinel', 'Sentinel 2A', 'Sentinel 2B']:
        raise ValueError('Only Landsat or Sentinel platforms supported.')

    # split bands by semi-colon
    bands = in_bands.lower().split(';')
    
    # prepare bands and names depending on platform
    out_bands = []
    for band in bands:
    
        if band in ['blue', 'green', 'red']:
            out_bands.append('nbart' + '_' + band)
                        
        elif in_platform == 'Landsat':
            if band == 'nir':
                out_bands.append('nbart' + '_' + band)
            elif band in ['swir1', 'swir2']:
                out_bands.append('nbart' + '_' + band[:-1] + '_' + band[-1])
            elif band == 'oa_mask':
                out_bands.append('oa_fmask')

        elif 'Sentinel' in in_platform:
            if band == 'nir1':
                out_bands.append('nbart' + '_' + band[:-1] + '_' + band[-1])
            elif band in ['swir2', 'swir3']:
                out_bands.append('nbart' + '_' + band[:-1] + '_' + band[-1])
            elif band == 'oa_mask':
                out_bands.append('oa_fmask')

        else:
            raise ValeuError('Reuqested band {} does not exist.'.format(band))

    # return list of renamed band names
    return out_bands


def read_shp_for_threshold(in_occurrence_feat=None, in_pa_column=None):
    """
    Alternative fucntion to read_shapefile which
    uses arcpy instead of gdal. This function is 
    only suitable for GDVSpectra Threshold. Quicker 
    and safer, but only useful for UI toolbox work. 
    Use read_shapefile for non-arcgis work. Input
    occurrence file and presence absence column are
    strings.
    """
    
    # check inputs
    if in_occurrence_feat is None or in_pa_column is None:
        raise ValueError('Did not provide a shapefile and field.')
    elif not isinstance(in_occurrence_feat, str):
        raise ValueError('Input shapefile must be a string path.')
    elif not isinstance(in_pa_column, str):
        raise ValueError('Input column must be a string.')

    try:
        # get description of file
        shp_desc = arcpy.Describe(in_occurrence_feat)
    except:
        raise ValueError('Could not describe input shapefile.')

    # check if empty shapefile, crs, field exists
    if int(arcpy.GetCount_management(in_occurrence_feat)[0]) == 0:
        raise ValueError('Shapefile is empty.')
    elif shp_desc.spatialReference.factoryCode != 3577:
        raise ValueError('Shapefile is not projected in GDA94 Albers (EPSG:3566).')
    elif in_pa_column not in [field.name for field in shp_desc.fields]:
        raise ValueError('Requested field is not in shapefile.')

    # check if presence/absence column is integer
    for field in shp_desc.fields:
        if field.name == in_pa_column and field.type != 'Integer':
            raise TypeError('Presence/absence column must be integer type.')

    # check if any non 1s and 0s in pres/abse column
    with arcpy.da.SearchCursor(in_occurrence_feat, [in_pa_column]) as cursor:
        vals = np.unique([row[0] for row in cursor])
        if len(vals) != 2 or (0 not in vals or 1 not in vals):
            raise ValueError('Presence/absence column does not contain just 1s and 0s.')

    # check if any non 1s and 0s in pres/abse column
    fields = ['SHAPE@X', 'SHAPE@Y', in_pa_column]
    with arcpy.da.SearchCursor(in_occurrence_feat, fields) as cursor:
        rows = [{'x': r[0], 'y': r[1], 'actual': r[2]} for r in cursor]
        df_records = pd.DataFrame.from_records(rows)

    # secondary check of various dataframe requirements
    if df_records.shape[0] == 0:
        raise ValueError('No records in pandas dataframe.')
    elif 'x' not in df_records or 'y' not in df_records:
        raise ValueError('No x, y columns in dataframe.')
    elif len(df_records.columns) != 3:
        raise ValueError('Number of columns in dataframe is incorrect.')
    elif 'actual' not in df_records:
        raise ValueError('Presence column not in dataframe.')
    elif len(np.unique(df_records['actual'])) != 2:
        raise ValueError('More than 1s and 0s in presence/absence in dataframe.')

    return df_records
   
   
def read_shp_for_nicher(in_occurrence_feat=None):
    """
    Alternative fucntion to read_shapefile which
    uses arcpy instead of gdal. This function is 
    only suitable for Nicher SDM. Quicker and safer, 
    but only useful for UI toolbox work. Use read_shapefile 
    for non-arcgis work. Input occurrence file is a
    string.
    """
    
    # check inputs
    if in_occurrence_feat is None:
        raise ValueError('Did not provide a shapefile.')
    elif not isinstance(in_occurrence_feat, str):
        raise ValueError('Input shapefile must be a string path.')

    try:
        # get description of file
        shp_desc = arcpy.Describe(in_occurrence_feat)
    except:
        raise ValueError('Could not describe input shapefile.')

    # check if empty shapefile, crs valid
    if int(arcpy.GetCount_management(in_occurrence_feat)[0]) == 0:
        raise ValueError('Shapefile is empty.')
    elif shp_desc.spatialReference.factoryCode != 3577:
        raise ValueError('Shapefile is not projected in GDA94 Albers (EPSG:3566).')

    # check if any non 1s and 0s in pres/abse column
    fields = ['SHAPE@X', 'SHAPE@Y']
    with arcpy.da.SearchCursor(in_occurrence_feat, fields) as cursor:
        rows = [{'x': r[0], 'y': r[1]} for r in cursor]
        df_records = pd.DataFrame.from_records(rows)

    # secondary check of various dataframe requirements
    if df_records.shape[0] == 0:
        raise ValueError('No records in pandas dataframe.')
    elif 'x' not in df_records or 'y' not in df_records:
        raise ValueError('No x, y columns in dataframe.')
    elif len(df_records.columns) != 2:
        raise ValueError('Number of columns in dataframe is incorrect.')

    return df_records
   
   
def generate_absences_for_nicher(df_pres, ds, buff_m):
    """
    Alternative fucntion to nicher generate_absence. This
    version uses arcpy instead of gdal. This function is 
    only suitable for Nicher SDM. Quicker and safer, 
    but only useful for UI toolbox work. Use generate_absence 
    and generate_proximity_areas for non-arcgis work. Input 
    df_pres is a pandas dataframe with x and y columns representing
    presence points. The ds is a xr dataset with x, y dimensions
    and at least one var. Buffer_m is the distance in metres for
    the exclusion zone buffer
    """

    # generate presence points via arcpy
    if df_pres is None:
        raise ValueError('No presence dataframe provided.')
    elif 'x' not in df_pres or 'y' not in df_pres:
        raise ValueError('No x or y fields in presence dataframe.')
    elif len(df_pres) == 0:
        raise ValueError('No records in presence dataframe.')

    try:
        # convert presence points to numpy and geometry points
        coords_pres = df_pres.to_numpy()
        points_pres = [arcpy.Point(c[0], c[1]) for c in coords_pres]

        # create empty presence feature class in memory
        results_pres = arcpy.management.CreateFeatureclass(out_path='memory',
                                                           out_name='presence_points',
                                                           geometry_type='POINT',
                                                           spatial_reference=3577)
        # access presence feature and insert points
        feat_pres = results_pres[0]
        with arcpy.da.InsertCursor(feat_pres, ['SHAPE@']) as cursor:
            [cursor.insertRow([p]) for p in points_pres]
    
    except Exception as e:
        raise ValueError(e)
    

    # check xr is valid
    if ds is None:
        raise ValueError('No xarray dataset provided.')
    elif not isinstance(ds, xr.Dataset):
        raise ValueError('Did not provide an xarray dataset type.') 
    elif 'x' not in ds or 'y' not in ds:
        raise ValueError('No x or y dimensions in xarray dataset.')
    elif len(ds) == 0:
        raise ValueError('No variables in xarray dataset.')

    try:
        # generate valid pixel mask (non-nans are 1)
        mask = xr.where(ds.to_array().isnull(), 0, 1)
        mask = mask.min('variable')

        # convert mask to dataframe, select valid, reset cols
        df_mask = mask.to_dataframe(name='valid')
        df_mask = df_mask[df_mask.valid == 1]
        df_mask = df_mask.reset_index(drop=False)
    
    except Exception as e:
        raise ValueError(e)

    # check we have something left
    if len(df_mask) == 0:
        raise ValueError('No valid pixels for pseudoabsences in dataframe.')

    try:
        # random sample if over 100k to reduce size 
        if len(df_mask) > 100000:
            df_mask = df_mask.sample(n=100000)

        # convert absence points to numpy and geometry points
        coords_abse = df_mask[['x', 'y']].to_numpy()
        points_abse = [arcpy.Point(c[0], c[1]) for c in coords_abse]
        
        # create empty absence feature class in memory
        results_abse = arcpy.management.CreateFeatureclass(out_path='memory',
                                                           out_name='absence_points',
                                                           geometry_type='POINT',
                                                           spatial_reference=3577)
        # access absence feature and insert points
        feat_abse = results_abse[0]
        with arcpy.da.InsertCursor(feat_abse, ['SHAPE@']) as cursor:
            [cursor.insertRow([a]) for a in points_abse]

    except Exception as e:
        raise ValueError(e)
            
            
    # check buffer is valid, if so, format parameter
    if buff_m < 1 or buff_m > 1000000:
        raise ValueError('Exclusion buffer distance invalid.')       
    
    
    try:
        # select all pseudoabsence not in exclusion buffer
        points_pabse = arcpy.management.SelectLayerByLocation(in_layer=feat_abse, 
                                                              overlap_type='WITHIN_A_DISTANCE',
                                                              select_features=feat_pres,
                                                              search_distance='{} Meters'.format(buff_m),
                                                              invert_spatial_relationship='INVERT')
    except Exception as e:
        raise ValueError(e)    

    # check number of pres abse that came back outside exlcusion
    num_pabse = int(points_pabse[2])
    if num_pabse == 0:
        raise ValueError('No pseudoabsence points generated outside buffer.')

        
    try:
        # convert presabse feature to numpy
        np_pabse = arcpy.da.FeatureClassToNumPyArray(in_table=points_pabse[0], 
                                                     field_names=['SHAPE@X', 'SHAPE@Y'])
        
        # then into pandas...
        df_pabse = pd.DataFrame(np_pabse)
        df_pabse = df_pabse.rename(columns={'SHAPE@X': 'x', 'SHAPE@Y': 'y'})
        
    except Exception as e:
        raise ValueError(e)  
        
    # check number of pseudoabse
    if len(df_pabse) == 0:
        raise ValueError('No pseudoabsence points generated.')
        
    return df_pabse  
   
   
def convert_fmask_codes(flags):
    """Takes a list of arcgis dea aws satellite data
    fmask flags (e.g., Water, Valid) and converts them
    into their DEA AWS numeric code equivalent."""
    
    out_flags = []
    for flag in flags:
        if flag == 'NoData':
            out_flags.append(0)
        elif flag == 'Valid':
            out_flags.append(1)
        elif flag == 'Cloud':
            out_flags.append(2)
        elif flag == 'Shadow':
            out_flags.append(3)
        elif flag == 'Snow':
            out_flags.append(4)
        elif flag == 'Water':
            out_flags.append(5)
            
    return out_flags


def get_name_of_mask_band(bands):
    """For a list of band names from a DEA AWS 
    data cube NetCDF, check if expected mask band
    exists and return it. Error if not found."""
    
    if 'oa_fmask' in bands:
        return 'oa_fmask'
    elif 'fmask' in bands:
        return 'fmask'
    else:
        arcpy.AddError('Expected mask band not found.')
        raise


def get_platform_from_dea_attrs(attr_dict):
    """We can get the name of our DEA AWS data cubes 
    platform if we extract the collections information
    and check the code at the start of the collection name.
    We need this a lot, so lets create a helper!"""
    
    # try and get collection from attribute... fail if empty
    collections = attr_dict.get('orig_collections')
    if collections is None:
        arcpy.AddError('Input NetCDF has no collection attribute.')
        raise
        
    # grab collections whether string or tuple/list
    if isinstance(collections, (list, tuple)) and len(collections) > 0:
        platform = collections[0]
    elif isinstance(collections, str):
        platform = collections
    else:
        arcpy.AddError('Input NetCDF ha attributes but no values in collections.')
        raise
        
    # parse dea aws platform code from collections attirbute
    if platform[:5] == 'ga_ls':
        platform = 'Landsat'
    elif platform[:2] == 's2' or platform[:5] == 'ga_s2':
        # the new collection naming begins with `ga_s2`, but we'll support
        # the older `s2` prefix in case users have already created
        # NetCDF files from the old collection on DEA
        platform = 'Sentinel'
    else:
        arcpy.AddError('Platform in NetCDF is not supported.')
        raise
        
    return platform


def apply_monitoring_area_symbology(layer):
    """
    Smaller helper function that akes an arcgis map 
    layer type and applies monitoring area zone symbology.
    """

    # check layer
    if not isinstance(layer, arcpy._mp.Layer):
        raise TypeError('Layer must be an arcpy map layer type.')
    elif not layer.isFeatureLayer:
        raise TypeError('Layer must be feature layer type')

    # get description object
    desc = arcpy.Describe(layer)
    
    # check if color border field in layer
    if 'color_border' not in [field.name for field in desc.fields]:
        raise ValueError('Field called color_border not in layer.')
        
    # check if color fill field in layer
    if 'color_fill' not in [field.name for field in desc.fields]:
        raise ValueError('Field called color_fill not in layer.')
        
    # set transparency
    alpha = 80

    # get symbology, update renderer, target color border field
    sym = layer.symbology
    sym.updateRenderer('UniqueValueRenderer')
    sym.renderer.fields = ['color_border', 'color_fill']

    # iter group items and colorize
    for grp in sym.renderer.groups:
        for itm in grp.items:
            try:
                # get class value and convert to int
                border_val = int(itm.values[0][0])
                fill_val = int(itm.values[0][1])
                
                # apply border color
                if border_val == 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [0, 0, 0, alpha]}
                elif border_val > 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [0, 112, 255, alpha]}
                elif border_val < 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [255, 0, 0, alpha]}

                # apply fill color
                if fill_val == 0:
                    itm.symbol.color = {'RGB': [255, 255, 255, alpha]} 
                elif abs(fill_val) == 1:
                    itm.symbol.color = {'RGB': [255, 115, 223, alpha]}
                elif abs(fill_val) == 2:
                    itm.symbol.color = {'RGB': [223, 115, 255, alpha]}
                elif abs(fill_val) == 3:
                    itm.symbol.color = {'RGB': [115, 178, 255, alpha]}
                elif abs(fill_val) == 4:
                    itm.symbol.color = {'RGB': [115, 223, 255, alpha]}
                elif abs(fill_val) == 5:
                    itm.symbol.color = {'RGB': [115, 255, 223, alpha]}
                elif abs(fill_val) == 6:
                    itm.symbol.color = {'RGB': [163, 255, 115, alpha]}
                elif abs(fill_val) == 7:
                    itm.symbol.color = {'RGB': [209, 255, 115, alpha]}
                elif abs(fill_val) == 8:
                    itm.symbol.color = {'RGB': [255, 255, 115, alpha]}
                elif abs(fill_val) == 9:
                    itm.symbol.color = {'RGB': [255, 211, 127, alpha]}
                elif abs(fill_val) == 10:
                    itm.symbol.color = {'RGB': [255, 167, 127, alpha]}
                elif abs(fill_val) == 11:
                    itm.symbol.color = {'RGB': [255, 127, 127, alpha]}
            except:
                print('Symbology fill class value not supported, skipping.')
                continue

    # finally, apply the symbology
    layer.symbology = sym
   
   
# deprecated
def apply_cmap(aprx, lyr_name, cmap_name='Precipitation', cutoff_pct=0.5):
    """
    For current ArcGIS Project which runs this function,
    finds particular layer in table of contents (lyr_name),
    applies specific color map to it (cmap_name) with a
    percentage cutoff (cutoff_pct). Set cutoff_pct to 0
    for no thresholding (i.e., min-max colouring).
    
    Parameters
    -------------
    aprx : arcpy aprx object
        The currently selected ArcGIS Pro project.
    lyr_name : str
        Name of layer in current map to colourise.
    cmap_name : str
        Name of official ESRI colormaps to visualise 
        layer to.
    cutoff_pct : float
        Value to threshold layer colours to, 0 - 1.
        
    Returns
    ----------
    Original input layer file with new colourmap.
    """
    
    # get cmap if exists, precipitation if not
    try:
        cmap = aprx.listColorRamps(cmap_name)[0]
    except:
        print('Requested cmap does not exist. Using default.')
        cmap = aprx.listColorRamps('Precipitation')[0]
        
    # get active map and ensure it isnt empty
    m = aprx.activeMap
    if m is None:
        raise ValueError('No active map found. Please open a map first.')
    
    # get requested lyr
    lyr = m.listLayers(lyr_name)
    if len(lyr) != 1:
        raise ValueError('Requested layer not found.')
    elif not lyr[0].isRasterLayer:
        raise TypeError('Requested layer is not a raster type.')
        
    # prepare lyr and symbology objects
    lyr = lyr[0]
    sym = lyr.symbology
    
    # set symbology parameters
    sym.colorizer.stretchType = 'PercentClip'
    sym.colorizer.colorRamp = cmap
    sym.colorizer.minPercent = cutoff_pct
    sym.colorizer.maxPercent = cutoff_pct   
    sym.colorizer.invertColorRamp = False
    
    # set symbology of requested layer
    lyr.symbology = sym
    
    # return coloursed layer
    return lyr


# deprecated
def convert_arcpy_geom_to_gjson(arcpy_geom):
    """Arcpy and ogr geom dont play nicely. The easiest
    way to convert arcpy geometry to ogr geometry is via
    geojson conversion from arcpy to gjson first."""
    
    
    try:
        # check if input is arcpy geometry type
        if arcpy_geom.type != 'polygon':
            print('Arcpy geometry object is not polygon type.')
            return
    except:
        print('Could not read input arcpy geometry.')
        return
        
    try:
        # create temp named file for geojson
        tmp = tempfile.NamedTemporaryFile().name + '.geojson'

        # perform feature conversion to geojson via arcpy
        arcpy.conversion.FeaturesToJSON(in_features=arcpy_geom, 
                                        out_json_file=tmp, 
                                        geoJSON='GEOJSON')
        
        return tmp
    
    except Exception as e:
        print(e)
        print('Could not convert geojson to OGR geometry.')
        return


# deprecated
def datetime_to_string(dt):
    """
    Basic function that takes a python datetime
    type of format that ArcGIS Pro datetime
    picker creates from UI and converts into a string of
    format YYYY-MM-DD suitable for dea aws stac query.
    
    Parameters
    -------------
    dt : datetime
        Datetime type produced by the ArcGIS Pro 
        UI date picker control.
    
    Returns
    ----------
    A string of date in format YYYY-MM-DD for use in dea
    aws stac query.
    """

    # get year, month, day as strings with zero padding
    y, m, d = str(dt.year), str(dt.month).zfill(2), str(dt.day).zfill(2)
    return '{}-{}-{}'.format(y, m, d)   


# deprecated
def convert_resample_interval_code(code):
    """Takes a resample interval code from ArcGIS UI and
    translates it to xarray friendly format"""
    
    if code == 'Weekly':
        return '1W'
    elif code == 'Bi-monthly':
        return '1SM'
    elif code == 'Monthly':
        return '1M'
    else:
        arcpy.AddError('Resample method not supported.')
        raise


# deprecated
def convert_smoother_code(code):
    """Takes a smoother code from ArcGIS UI and
    translates it to xarray friendly format"""
    
    if code == 'Savitsky-Golay':
        return 'savitsky'
    elif code == 'Symmetrical Gaussian':
        return 'symm_gaussian'
    else:
        arcpy.AddError('Smoother is not supported.')
        raise


# deprecated
def get_bbox_from_geom(in_geom):
    """Helper func to take an arcpy geom type and
    project it to albers. The bbox is obtained from 
    this projected geometry"""       
    
    # make a copy of input geom
    out_geom = in_geom
    
    # transform from albers to wgs84 if needed
    if out_geom.spatialReference.factoryCode == 3577:
        srs = arcpy.SpatialReference(4326)
        out_geom = out_geom.projectAs(srs)

    # get bbox in wgs84 from geometry
    bbox = [out_geom.extent.XMin, out_geom.extent.YMin, 
            out_geom.extent.XMax, out_geom.extent.YMax]

    return bbox
   

# deprecated
def check_savitsky_inputs(window_length, polyorder):
    """Based on scipy, savitsk golay window length 
    must be odd, positive number, and polyorder must
    always be lower than window_length."""
    
    # check window length is odd and > 0
    if window_length <= 0:
        arcpy.AddWarning('Savitsky window length must be > 0. Defaulting to 3.')
        window_length = 3
    
    elif window_length % 2 == 0:
        arcpy.AddWarning('Savitsky window length must be odd number. Subtracting 1.')
        window_length = window_length - 1

    # now do polyorder
    if polyorder >= window_length:
        arcpy.AddWarning('Savitsky polyorder must be less than window length. Correcting.')
        polyorder = window_length - 1
    
    return window_length, polyorder


# deprecated
def get_selected_layer_extent(lyr):
    """
    Basic function that takes a arcpy feature layer 
    and extracts extent of layer in WGS84 lat/lons. Used
    as the bounding box in stac query.
    
    Parameters
    -------------
    lyr : arcpy feature layer type
        A feature layer of arcpy type geometry. Expects
        a polygon type. Rejects feature if no spatial
        coordinate system detected.
    
    Returns
    ----------
    A list in form of west, south, east, north coordinates 
    in WGS84.
    """
    
    # set output epsg (for now, always wgs84)
    out_epsg = 4326
    
    # get layer description
    desc = arcpy.Describe(lyr)
    
    # check if polygon
    if not desc.shapeType == 'Polygon':
        raise TypeError('Shapefile is not a polygon type.')
            
    # check for coordinate system
    if not desc.hasSpatialIndex:
        raise ValueError('Shapefile does not have a coordinate system.')
           
    # union all selected geometries into one
    uni = None
    with arcpy.da.SearchCursor(lyr, ['SHAPE@']) as cursor:
        for idx, row in enumerate(cursor):
            if row[0] is not None:
                if idx == 0:
                    uni = row[0]
                else:
                    uni = uni.union(row[0])

    # check if user selected a feat
    if uni is None:
        print('Warning, no polygon(s) selected. Using whole shapefile extent!')
        uni = desc
        
    # extract extent
    ext = uni.extent
        
    # project extent to wgs84
    srs = arcpy.SpatialReference(out_epsg)
    ext_prj = ext.projectAs(srs)
    
    # return bbox coords
    return [ext_prj.XMin, ext_prj.YMin, ext_prj.XMax, ext_prj.YMax]


# deprecated
def prepare_fill_value_type(in_fill_value):
    """
    Basic function that checks and prepares
    the value user set within ArcGIS Pro
    COG Fetcher UI when setting the value
    to fill empty raster cells during COG
    fetch. ArcGIS Pro sets this value as
    a string, so we want the actual numeric
    version of this string. Limits the allowed
    no data values. If error, defaults to np.nan.
    
    Parameters
    -------------
    in_fill_value : str
        The ArcGIS Pro UI string for setting
        the xarray raster no data value. Could be
        a string, nan, or numeric. But always will 
        input into this as a string, from UI.
    
    Returns
    ----------
    Processed fill value in correct type. If enters
    as '-999', leaves here as an integer -999.
    """
    
    # set allowed nodata types
    nodata_types = [
        'np.nan', 
        'nan', 
        'nodata', 
        'no data', 
        'none', 
        ''
    ]
    
    # check if value is negative
    is_neg = True if in_fill_value[0] == '-' else False
    if is_neg:
        in_fill_value = in_fill_value[1:]
    
    # convert datatype to appropriate dtype
    try:
        if in_fill_value is None:
            return np.nan

        elif in_fill_value.lower() in nodata_types:
            return np.nan
        
        elif in_fill_value.isdigit():
            out_val = int(in_fill_value)
            return out_val * -1 if is_neg else out_val

        else:
            out_val = float(in_fill_value)
            return out_val * -1 if is_neg else out_val
        
    except:
        print('Could not determine fill value. Defaulting to np.nan.')
        return np.nan


# deprecated
def prepare_cell_align_type(in_cell_align):
    """
    Basic function that renames the cell alignment
    string from the ArcGIS Pro UI and makes it dea
    aws stac query compatible. 
    
    Parameters
    -------------
    in_cell_align : str
        The ArcGIS Pro UI string for stac cell alignment.
        Example: Top-left, Center.
    
    Returns
    ----------
    Processed cell alignment string name for use in dea
    aws compatible stac query. Example: topleft, center.
    """
    
    # checks
    if not isinstance(in_cell_align, str):
        raise TypeError('Cell alignment must be a string.')
    
    elif in_cell_align not in ['Top-left', 'Center']:
        raise ValueError('Cell alignment must be either Top-left or Center.')
        
    # prepare and return
    return in_cell_align.lower().replace('-', '')
    

# deprecated
def prepare_vegfrax_classes(classes):
    """
    Unpacks a string of selected classes from ArcGIS Pro
    interface. Will always be in format "'Class: X'; 'Class: Y'"
    etc. We only support integer classes at this stage.
    """
    
    # convert arcgis multi-value format to list of values and notify       
    classes = classes.replace('Class: ', '').replace("'", "").split(';')

    if len(classes) == 0:
        arcpy.AddError('Not classes could be obtained from selection.')
        raise

    # convert into integers... strings not supported
    _ = []
    for c in classes:
        try:
            _.append(int(c))
        except:
            arcpy.AddError('Selected class not an integer. Only support integers, e.g., 1, 2, 3.')
            raise

    # return classes
    return _