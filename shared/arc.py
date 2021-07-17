# arc

# imports 
import arcpy

# meta
def prepare_collections_list(in_platform):
    """
    """
    
    # checks
    if in_platform not in ['Landsat', 'Sentinel']:
        raise ValueError('Platform must be Landsat or Sentinel.')
        
    # prepare collections
    if in_platform == 'Landsat':
        return ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3']
    elif in_platform == 'Sentinel':
        raise ValueError('Sentinel not yet implemented.')

# meta, checks 
def datetime_to_string(dt):
    
    # imports
    from datetime import datetime
    
    # get year, month, day as strings with zero padding
    y, m, d = str(dt.year), str(dt.month).zfill(2), str(dt.day).zfill(2)
    return '{}-{}-{}'.format(y, m, d)   

# meta
def get_selected_layer_extent(lyr):
    """
    """
    
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
            geom = row[0]
            if idx == 0:
                uni = geom
            else:
                uni = uni.union(geom)

    # check if user selected a feat
    if uni is None:
        raise ValueError('No feature was selected.')
        
    # extract extent
    ext = uni.extent
        
    # project extent to wgs84
    srs = arcpy.SpatialReference(4326)
    ext_prj = ext.projectAs(srs)
    
    # return bbox coords
    return [ext_prj.XMin, ext_prj.YMin, ext_prj.XMax, ext_prj.YMax]

# meta
def prepare_band_names(in_bands, in_platform):
    """
    """
    
    # checks
    if in_bands is None:
        raise ValueError('No bands in band request.')
    elif not isinstance(in_bands, str):
        raise TypeError('Expecting a string of band names.')
    elif in_platform not in ['Landsat', 'Sentinel']:
        raise ValueError('Only Landsat or Sentinel platforms supported.')

    # split bands by semi-colon
    bands = in_bands.lower().split(';')
    
    # prepare bands and names depending on platform
    out_bands = []
    for band in bands:
        if band in ['blue', 'green', 'red']:
            out_bands.append('nbart' + '_' + band)
            
        elif band == 'oa_mask':
            out_bands.append('oa_fmask')
            
        elif in_platform == 'Landsat':
            if band == 'nir':
                out_bands.append('nbart' + '_' + band)
            elif band in ['swir1', 'swir2']:
                out_bands.append('nbart' + '_' + band[:-1] + '_' + band[-1])

        elif in_platform == 'Sentinel':
            if band == 'nir_1':
                out_bands.append('nbart' + '_' + band)
            elif band in ['swir2', 'swir3']:
                out_bands.append('nbart' + '_' + band[:-1] + '_' + band[-1])

        else:
            raise ValeuError('Reuqested band {} does not exist.'.format(band))

    return out_bands

# meta
def prepare_cell_align_type(in_cell_align):
    """
    """
    
    # checks
    if not isinstance(in_cell_align, str):
        raise TypeError('Cell alignment must be a string.')
    elif in_cell_align not in ['Top-left', 'Center']:
        raise ValueError('Cell alignment must be either Top-left or Center.')
        
    # prepare and return
    return in_cell_align.lower().replace('-', '')
   
# meta   
def prepare_fill_value_type(in_fill_value):
    """
    """
    
    # imports
    import numpy as np
    
    # checks
    
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
        print('Could not determine fill value. Setting to np.nan.')
        return np.nan