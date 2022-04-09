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
from datetime import datetime


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
        return ['s2a_ard_granule', 's2b_ard_granule']
    
    elif in_platform == 'Sentinel 2A':
        return ['s2a_ard_granule']
        
    elif in_platform == 'Sentinel 2B':
        return ['s2b_ard_granule']


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


def datetime_to_numpy(dt):
    """
    Quick function to convert a datetime
    object into a numpy format datetime64 for use in
    indexing and querying xr time dimensions.
    """
    
    return np.datetime64(dt)


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
                out_bands.append('fmask')

        else:
            raise ValeuError('Reuqested band {} does not exist.'.format(band))

    # return list of renamed band names
    return out_bands


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
    
   
# checks, meta
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
    elif platform[:2] == 's2':
        platform = 'Sentinel'
    else:
        arcpy.AddError('Platform in NetCDF is not supported.')
        raise
        
    return platform


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

# meta
def apply_monitoring_area_symbology(layer):
    """takes a arcgis map layer type"""

    # check layer
    if not isinstance(layer, arcpy._mp.Layer):
        raise TypeError('Layer must be an arcpy map layer type.')
    elif not layer.isFeatureLayer:
        raise TypeError('Layer must be feature layer type')

    # check if color field in layer
    desc = arcpy.Describe(layer)
    if 'color' not in [field.name for field in desc.fields]:
        raise ValueError('Field called color not in layer.')
        
    # set transparency
    alpha = 80
        
    # get symbology, update renderer, target color field
    sym = layer.symbology
    sym.updateRenderer('UniqueValueRenderer')
    sym.renderer.fields = ['color']
    
    # iter group items and colorize
    for grp in sym.renderer.groups:
        for itm in grp.items:
            try:
                # get class value and convert to int
                val = int(itm.values[0][0])

                # apply fill color
                if val == 0:
                    itm.symbol.color = {'RGB': [255, 255, 255, alpha]}
                elif abs(val) == 1:
                    itm.symbol.color = {'RGB': [255, 115, 223, alpha]} 
                elif abs(val) == 2:
                    itm.symbol.color = {'RGB': [115, 178, 255, alpha]}
                elif abs(val) == 3:
                    itm.symbol.color = {'RGB': [115, 178, 255, alpha]}
                elif abs(val) == 4:
                    itm.symbol.color = {'RGB': [115, 223, 255, alpha]}
                elif abs(val) == 5:
                    itm.symbol.color = {'RGB': [115, 255, 223, alpha]}
                elif abs(val) == 6:
                    itm.symbol.color = {'RGB': [163, 255, 115, alpha]}
                elif abs(val) == 7:
                    itm.symbol.color = {'RGB': [209, 255, 115, alpha]}
                elif abs(val) == 8:
                    itm.symbol.color = {'RGB': [255, 255, 115, alpha]}
                elif abs(val) == 9:
                    itm.symbol.color = {'RGB': [255, 211, 127, alpha]}
                elif abs(val) == 10:
                    itm.symbol.color = {'RGB': [255, 167, 127, alpha]}
                elif abs(val) == 11:
                    itm.symbol.color = {'RGB': [255, 127, 127, alpha]}

                # apply border style now for stable, incline, decline
                if val == 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [0, 0, 0, alpha]} 
                elif val > 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [0, 112, 255, alpha]}
                elif val < 0:
                    itm.symbol.size = 2
                    itm.symbol.outlineColor = {'RGB': [255, 0, 0, alpha]}         

            except:
                print('Symbology class value not supported, skipping.')
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

