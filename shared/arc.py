# arc

# imports 
import arcpy

# meta, checks
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
