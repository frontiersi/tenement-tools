"""
Raster functions allow you to define processing operations that will be applied to one or more rasters.
These functions are applied to the raster data on the fly as the data is accessed and viewed; therefore,
they can be applied quickly without having to endure the time it would otherwise take to create a
processed product on disk, for which raster analytics tools like arcgis.raster.analytics.generate_raster can be used.

Functions can be applied to various rasters (or images), including the following:

* Imagery layers
* Rasters within imagery layers

"""
# Raster dataset layers
# Mosaic datasets
# Rasters within mosaic datasets
from .._layer import ImageryLayer, Raster, _ArcpyRaster, RasterCollection
from .utility import _raster_input, _get_raster, _replace_raster_url, _get_raster_url, _get_raster_ra, \
                     _pixel_type_string_to_long
from arcgis.gis import Item
import copy
import numbers
from . import gbl
import arcgis as _arcgis
import json as _json
from arcgis.geoprocessing._support import _analysis_job, _analysis_job_results, \
                                          _analysis_job_status
from .utility import _raster_input_rft, _get_raster_ra_rft, _input_rft, _find_object_ref, \
                     _python_variable_name
from arcgis.features.layer import FeatureLayer as _FeatureLayer
from .._RasterInfo import RasterInfo
import logging
_LOGGER = logging.getLogger(__name__)
from datetime import datetime
import time

key_value_dict={}
hidden_inputs = ["ToolName","PrimaryInputParameterName", "OutputRasterParameterName"]


#
# def _raster_input(raster):
#
#     if isinstance(raster, ImageryLayer):
#         layer = raster
#         raster = raster._fn #filtered_rasters()
#     # elif isinstance(raster, dict) and 'function_chain' in raster:
#     #     layer = raster['layer']
#     #     raster = raster['function_chain']
#     elif isinstance(raster, list):
#         r0 = raster[0]
#         if 'function_chain' in r0:
#             layer = r0['layer']
#             raster = [r['function_chain'] for r in raster]
#     else:
#         layer = None
#
#     return layer, raster


def _clone_layer(layer, function_chain, raster_ra, raster_ra2=None, variable_name='Raster'):

    _set_multidimensional_rules(function_chain)

    if isinstance(layer, Raster) or isinstance(layer, RasterCollection):
        return _clone_layer_raster(layer, function_chain, raster_ra, raster_ra2, variable_name)
    if isinstance(layer, Item):
        layer = layer.layers[0]

    function_chain_ra = copy.deepcopy(function_chain)
    function_chain_ra['rasterFunctionArguments'][variable_name] = raster_ra
    if raster_ra2 is not None:
        function_chain_ra['rasterFunctionArguments']['Raster2'] = raster_ra2
    if layer._datastore_raster:
        if isinstance(layer._uri, dict) or isinstance(layer._uri,bytes):
            newlyr = ImageryLayer(function_chain_ra, layer._gis)
        else:
            newlyr = ImageryLayer(layer._uri, layer._gis)

    else:
        newlyr = ImageryLayer(layer._url, layer._gis)

    # if layer._fn is not None: # chain the functions
    #     old_chain = layer._fn
    #     newlyr._fn = function_chain
    #     newlyr._fn['rasterFunctionArguments']['Raster'] = old_chain
    # else:
    newlyr._fn = function_chain
    newlyr._fnra = function_chain_ra
    if layer._datastore_raster:
        if not isinstance(layer._uri, dict) and not isinstance(layer._uri,bytes):
            newlyr._fn = function_chain_ra

    newlyr._where_clause = layer._where_clause
    newlyr._spatial_filter = layer._spatial_filter
    newlyr._temporal_filter = layer._temporal_filter
    newlyr._mosaic_rule = layer._mosaic_rule
    newlyr._filtered = layer._filtered
    newlyr._uses_gbl_function = layer._uses_gbl_function
    newlyr._raster_info = layer._raster_info

    if hasattr(layer, "_lazy_token"):
        newlyr._lazy_token = layer._lazy_token
    else:
        newlyr._lazy_token = layer._token

    if layer._extent_set:
        newlyr._extent = layer._extent
        newlyr._extent_set = layer._extent_set

    return newlyr

def _clone_layer_without_copy(layer, function_chain, function_chain_ra):
    _set_multidimensional_rules(function_chain, function_chain_ra)
    if isinstance(layer, Raster) or isinstance(layer, RasterCollection):
        return _clone_layer_raster_without_copy(layer, function_chain, function_chain_ra)
    if isinstance(layer, Item):
        layer = layer.layers[0]
   
    if layer._datastore_raster:
        if isinstance(layer._uri, dict) or isinstance(layer._uri,bytes):
            newlyr = ImageryLayer(function_chain_ra, layer._gis)
        else:
            newlyr = ImageryLayer(layer._uri, layer._gis)

    else:
        newlyr = ImageryLayer(layer._url, layer._gis)

    # if layer._fn is not None: # chain the functions
    #     old_chain = layer._fn
    #     newlyr._fn = function_chain
    #     newlyr._fn['rasterFunctionArguments']['Raster'] = old_chain
    # else:
    newlyr._fn = function_chain
    newlyr._fnra = function_chain_ra

    if layer._datastore_raster:
        if not isinstance(layer._uri, dict) and not isinstance(layer._uri,bytes):
            newlyr._fn = function_chain_ra

    newlyr._where_clause = layer._where_clause
    newlyr._spatial_filter = layer._spatial_filter
    newlyr._temporal_filter = layer._temporal_filter
    newlyr._mosaic_rule = layer._mosaic_rule
    newlyr._filtered = layer._filtered
    newlyr._uses_gbl_function = layer._uses_gbl_function
    newlyr._raster_info = layer._raster_info

    if hasattr(layer, "_lazy_token"):
        newlyr._lazy_token = layer._lazy_token
    else:
        newlyr._lazy_token = layer._token
    if layer._extent_set:
        newlyr._extent = layer._extent
        newlyr._extent_set = layer._extent_set
    return newlyr

def _clone_layer_raster(layer, function_chain, raster_ra, raster_ra2=None, variable_name='Raster'):

    function_chain_ra = copy.deepcopy(function_chain)
    function_chain_ra['rasterFunctionArguments'][variable_name] = raster_ra
    if raster_ra2 is not None:
        function_chain_ra['rasterFunctionArguments']['Raster2'] = raster_ra2

    if layer._datastore_raster:
        if isinstance(layer._uri, dict) or isinstance(layer._uri,bytes):
            newlyr= Raster(function_chain_ra, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
        else:
            newlyr= Raster(layer._uri, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
    else:
        newlyr = Raster(layer._url, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)

    if layer._engine==_ArcpyRaster:
        try:            
            import arcpy, json
            arcpylyr=arcpy.ia.Apply(layer._uri,json.dumps(function_chain_ra))
            newlyr = Raster(arcpylyr, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
        except Exception as err:
            _LOGGER.warning(err)

    #newlyr.properties = layer.properties
    newlyr._engine_obj._fn = function_chain
    newlyr._engine_obj._fnra = function_chain_ra

    if (hasattr(layer, '_datastore_raster')) and layer._datastore_raster:
        if not isinstance(layer._uri, dict) and not isinstance(layer._uri,bytes):
            newlyr._engine_obj._fn  = copy.deepcopy(function_chain_ra)

    newlyr._engine_obj._where_clause = layer._where_clause
    newlyr._engine_obj._spatial_filter = layer._spatial_filter
    newlyr._engine_obj._temporal_filter = layer._temporal_filter
    newlyr._engine_obj._mosaic_rule = layer._mosaic_rule
    newlyr._engine_obj._filtered = layer._filtered
    newlyr._engine_obj._uses_gbl_function = layer._uses_gbl_function
    newlyr._engine_obj._do_not_hydrate = layer._do_not_hydrate
    #newlyr._engine_obj.extent = layer.extent
    if hasattr(layer, "_lazy_token"):
        newlyr._engine_obj._lazy_token = layer._lazy_token
    else:
        if layer._engine!=_ArcpyRaster:
            newlyr._lazy_token = layer._token

    if layer._extent_set:
        newlyr._engine_obj._extent = layer._extent
        newlyr._engine_obj._extent_set = layer._extent_set

    return newlyr

def _clone_layer_raster_without_copy(layer, function_chain, function_chain_ra):

    if layer._datastore_raster:
        if isinstance(layer._uri, dict) or isinstance(layer._uri,bytes):
            newlyr= Raster(function_chain_ra, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
        else:
            newlyr= Raster(layer._uri, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
    else:
        newlyr = Raster(layer._url, is_multidimensional= layer._is_multidimensional, engine= layer._engine,  gis=layer._gis)

    if layer._engine==_ArcpyRaster:
        try:
            import arcpy, json
            arcpylyr=arcpy.ia.Apply(layer._uri,json.dumps(function_chain_ra))
            newlyr = Raster(arcpylyr, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
        except Exception as err:
            _LOGGER.warning(err)

    #newlyr.properties = layer.properties
    newlyr._engine_obj._fn = function_chain
    newlyr._engine_obj._fnra = function_chain_ra

    if (hasattr(layer, '_datastore_raster')) and layer._datastore_raster:
        if not isinstance(layer._uri, dict) and not isinstance(layer._uri,bytes):
            newlyr._engine_obj._fn  = copy.deepcopy(function_chain_ra)

    newlyr._engine_obj._where_clause = layer._where_clause
    newlyr._engine_obj._spatial_filter = layer._spatial_filter
    newlyr._engine_obj._temporal_filter = layer._temporal_filter
    newlyr._engine_obj._mosaic_rule = layer._mosaic_rule
    newlyr._engine_obj._filtered = layer._filtered
    newlyr._engine_obj._uses_gbl_function = layer._uses_gbl_function
    newlyr._engine_obj._do_not_hydrate = layer._do_not_hydrate
    newlyr._engine_obj.extent = layer.extent
    if hasattr(layer, "_lazy_token"):
        newlyr._engine_obj._lazy_token = layer._lazy_token
    else:
        if layer._engine!=_ArcpyRaster:
            newlyr._lazy_token = layer._token

    if layer._extent_set:
        newlyr._engine_obj._extent = layer._extent
        newlyr._engine_obj._extent_set = layer._extent_set

    return newlyr

def _set_multidimensional_rules(function_chain = None, function_chain_ra = None):
    match_variables = _arcgis.env.match_variables
    union_dimension = _arcgis.env.union_dimension

    if (match_variables is not None) and isinstance(match_variables, bool):
        if (function_chain is not None) and 'MatchVariable' not in function_chain.keys():
            function_chain['rasterFunctionArguments']['MatchVariable'] =  match_variables
        if (function_chain_ra is not None) and 'MatchVariable' not in function_chain_ra.keys():
            function_chain_ra['rasterFunctionArguments']['MatchVariable'] = match_variables
    if (union_dimension is not None) and isinstance(union_dimension, bool):
        if (function_chain is not None) and 'UnionDimension' not in function_chain.keys():
            function_chain['rasterFunctionArguments']['UnionDimension'] = union_dimension
        if (function_chain_ra is not None) and 'UnionDimension' not in function_chain.keys():
            function_chain_ra['rasterFunctionArguments']['UnionDimension'] =  union_dimension




def arg_statistics(rasters, stat_type=None, min_value=None, max_value=None, undefined_class=None, astype=None):
    """
    The arg_statistics function produces an output with a pixel value that represents a statistical metric from all
    bands of input rasters. The statistics can be the band index of the maximum, minimum, or median value, or the
    duration (number of bands) between a minimum and maximum value

    See http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/argstatistics-function.htm

    :param rasters: the imagery layers filtered by where clause, spatial and temporal filters
    :param stat_type: one of "max", "min", "median", "duration"
    :param min_value: double, required if the type is duration
    :param max_value: double, required if the type is duration
    :param undefined_class: int, required if the type is maximum or minimum
    :return: the output raster with this function applied to it
    """
    # find oids given spatial and temporal filter and where clause

    layer, raster, raster_ra = _raster_input(rasters)

    stat_types = {
        'max': 0,
        'min': 1,
        'median': 2,
        'duration': 3
    }
        
    template_dict = {
        "rasterFunction": "ArgStatistics",
        "rasterFunctionArguments": {            
            "Rasters": raster,
        },
        "variableName": "Rasters"
    }

    if stat_type is not None:       
        template_dict["rasterFunctionArguments"]['ArgStatisticsType'] = stat_types[stat_type.lower()]
    if min_value is not None:
        template_dict["rasterFunctionArguments"]['MinValue'] = min_value
    if max_value is not None:
        template_dict["rasterFunctionArguments"]['MaxValue'] = max_value
    if undefined_class is not None:
        template_dict["rasterFunctionArguments"]['UndefinedClass'] = undefined_class

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')

def arg_max(rasters, undefined_class=None, astype=None):
    """
    In the ArgMax method, all raster bands from every input raster are assigned a 0-based incremental band index,
    which is first ordered by the input raster index, as shown in the table below, and then by the relative band order
    within each input raster.

    See http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/argstatistics-function.htm

    :param rasters: the imagery layers filtered by where clause, spatial and temporal filters
    :param undefined_class: int, required
    :return: the output raster with this function applied to it
    """
    return arg_statistics(rasters, "max", undefined_class=undefined_class, astype=astype)

def arg_min(rasters, undefined_class=None, astype=None):
    """
    ArgMin is the argument of the minimum, which returns the Band index for which the given pixel attains
    its minimum value.

    See http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/argstatistics-function.htm

    :param rasters: the imagery layers filtered by where clause, spatial and temporal filters
    :param undefined_class: int, required
    :return: the output raster with this function applied to it
    """
    return arg_statistics(rasters, "min", undefined_class=undefined_class, astype=astype)

def arg_median(rasters, undefined_class=None, astype=None):
    """
    The ArgMedian method returns the Band index for which the given pixel attains the median value of values
    from all bands.

    Consider values from all bands as an array. After sorting the array in ascending order, the median is the
    one value separating the lower half of the array from the higher half. More specifically, if the ascend-sorted
    array has n values, the median is the ith (0-based) value, where: i = ( (n-1) / 2 )

    See http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/argstatistics-function.htm

    :param rasters: the imagery layers filtered by where clause, spatial and temporal filters
    :param undefined_class: int, required
    :return: the output raster with this function applied to it
    """
    return arg_statistics(rasters, "median", undefined_class=undefined_class, astype=astype)

def duration(rasters, min_value=None, max_value=None, undefined_class=None, astype=None):
    """
    Returns the duration (number of bands) between a minimum and maximum value.
    The Duration method finds the longest consecutive elements in the array, where each element has a value greater
    than or equal to min_value and less than or equal to max_value, and then returns its length.

    See http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/argstatistics-function.htm

    :param rasters: the imagery layers filtered by where clause, spatial and temporal filters
    :param undefined_class: int, required
    :return: the output raster with this function applied to it
    """
    return arg_statistics(rasters, "max",  min_value=min_value, max_value=max_value,
                          undefined_class=undefined_class, astype=astype)


def arithmetic(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None, operation_type=1):
    """
    The Arithmetic function performs an arithmetic operation between two rasters or a raster and a scalar, and vice versa.

    :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
    :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
    :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
    :param operation_type: int 1 = Plus, 2 = Minus, 3 = Multiply, 4=Divide, 5=Power, 6=Mode
    :return: the output raster with this function applied to it
    """

    layer1, raster_1, raster_ra1 = _raster_input(raster1)
    layer2, raster_2, raster_ra2 = _raster_input(raster1, raster2)

    if layer1 is not None and (layer2 is None or ((layer2 is not None) and layer2._datastore_raster is False)):
        layer = layer1
    else:
        layer = layer2
    #layer = layer1 if layer1 is not None else layer2

    extent_types = {
        "FirstOf" : 0,
        "IntersectionOf" : 1,
        "UnionOf" : 2,
        "LastOf" : 3
    }

    cellsize_types = {
        "FirstOf" : 0,
        "MinOf" : 1,
        "MaxOf" : 2,
        "MeanOf" : 3,
        "LastOf" : 4
    }

    in_extent_type = extent_types[extent_type]
    in_cellsize_type = cellsize_types[cellsize_type]

    template_dict = {
        "rasterFunction": "Arithmetic",
        "rasterFunctionArguments": {
            "Operation": operation_type,
            "Raster": raster_1,
            "Raster2": raster_2
        }
    }

    if in_extent_type is not None:
        template_dict["rasterFunctionArguments"]['ExtentType'] = in_extent_type
    if in_cellsize_type is not None:
        template_dict["rasterFunctionArguments"]['CellsizeType'] = in_cellsize_type

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra1, raster_ra2)

#
#
# def plus(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Adds two rasters or a raster and a scalar, and vice versa
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 1)
#
# def minus(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Subtracts a raster or a scalar from another raster or a scaler
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 2)
#
# def multiply(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Multiplies two rasters or a raster and a scalar, and vice versa
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 3)
#
# def divide(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Divides two rasters or a raster and a scalar, and vice versa
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 4)
#
# def power(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Adds two rasters or a raster and a scalar, and vice versa
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 5)
#
# def mode(raster1, raster2, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
#     """
#     Adds two rasters or a raster and a scalar, and vice versa
#
#     :param raster1: the first raster- imagery layers filtered by where clause, spatial and temporal filters
#     :param raster2: the 2nd raster - imagery layers filtered by where clause, spatial and temporal filters
#     :param extent_type: one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
#     :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
#     :return: the output raster with this function applied to it
#     """
#
#     return arithmetic(raster1, raster2, extent_type, cellsize_type, astype, 6)
#

def aspect(raster):
    """
    aspect identifies the downslope direction of the maximum rate of change in value from each cell to its neighbors.
    Aspect can be thought of as the slope direction. The values of the output raster will be the compass direction of
    the aspect. For more information, see
    `Aspect function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/aspect-function.htm>`__
    and `How Aspect works <http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/how-aspect-works.htm>`__.

    :param raster: the input raster / imagery layer
    :return: aspect applied to the input raster
    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Aspect",
        "rasterFunctionArguments": {
            "Raster" : raster,
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)


def band_arithmetic(raster, band_indexes=None, astype=None, method=0):
    """
    The band_arithmetic function performs an arithmetic operation on the bands of a raster. For more information,
    see Band Arithmetic function at http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/band-arithmetic-function.htm

    :param raster: the input raster / imagery layer
    :param band_indexes: band indexes or expression. Band indexes can be given as a space seperated string or a list of integers or floating point values. e.g., "4 3" or [4,3]. For user defined methods the band index can be given as an expression such as "(B3 - B1)/(B3 + B1)"
    :param astype: output pixel type
    :param method: int. The type of band arithmetic algorithm you want to deploy. 
                   You can define your custom algorithm, or choose a predefined index.
                   0 = UserDefined, 
                   1 = NDVI,
                   2 = SAVI,
                   3 = TSAVI,
                   4 = MSAVI,
                   5 = GEMI,
                   6 = PVI,
                   7 = GVITM,
                   8 = Sultan,
                   9 = VARI,
                   10 = GNDVI,
                   11 = SR,
                   12 = NDVIre,
                   13 = SRre,
                   14 = MTVI2,
                   15 = RTVICore,
                   16 = CIre,
                   17 = CIg,
                   18 = NDWI,
                   19 = EVI,
                   20 = IronOxide,
                   21 = FerrousMinerals,
                   22 = ClayMinerals,
                   23 = WNDWI,
                   24 = BAI,
                   25 = NBR,
                   26 = NDBI,
                   27 = NDMI,
                   28 = NDSI,
                   29 = MNDWI

    :return: band_arithmetic applied to the input raster
    """

    layer, raster, raster_ra = _raster_input(raster)

    if isinstance(band_indexes, list):
         band_indexes = " ".join(str(index) for index in band_indexes)

    template_dict = {
        "rasterFunction": "BandArithmetic",
        "rasterFunctionArguments": {
            "Method": method,
            "BandIndexes": band_indexes,
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)

def ndvi(raster, band_indexes="4 3", astype=None):
    """
    Normalized Difference Vegetation Index
    NDVI = ((NIR - Red)/(NIR + Red))

    :param raster: the input raster / imagery layer
    :param band_indexes: Band Indexes "NIR Red", e.g., "4 3" or [4,3]
    :param astype: output pixel type
    :return: Normalized Difference Vegetation Index raster
    """
    return band_arithmetic(raster, band_indexes, astype, 1)

def savi(raster, band_indexes="4 3 0.33", astype=None):
    """
    Soil-Adjusted Vegetation Index
    SAVI = ((NIR - Red) / (NIR + Red + L)) x (1 + L)
    where L represents amount of green vegetative cover, e.g., 0.5

    :param raster: the input raster / imagery layer
    :param band_indexes: "BandIndexes": "NIR Red L", for example, "4 3 0.33" or [4,3,0.33]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 2)

def tsavi(raster, band_indexes= "4 3 0.33 0.50 1.50", astype=None):
    """
    Transformed Soil Adjusted Vegetation Index

    TSAVI = (s(NIR-s*Red-a))/(a*NIR+Red-a*s+X*(1+s^2))

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red s a X", e.g., "4 3 0.33 0.50 1.50" or [4,3,0.33,0.50,1.50] where a = the soil line intercept, s = the soil line slope, X = an adjustment factor that is set to minimize soil noise
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 3)

def msavi(raster, band_indexes="4 3", astype=None):
    """
    Modified Soil Adjusted Vegetation Index
    
    MSAVI2 = (1/2)*(2(NIR+1)-sqrt((2*NIR+1)^2-8(NIR-Red)))

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red", e.g., "4 3" or [4,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 4)

def gemi(raster, band_indexes="4 3", astype=None):
    """
    Global Environmental Monitoring Index

    GEMI = eta*(1-0.25*eta)-((Red-0.125)/(1-Red))
    
    where eta = (2*(NIR^2-Red^2)+1.5*NIR+0.5*Red)/(NIR+Red+0.5)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red", e.g., "4 3" or [4,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 5)

def pvi(raster, band_indexes="4 3 0.3 0.5", astype=None):
    """
    Perpendicular Vegetation Index
    PVI = (NIR-a*Red-b)/(sqrt(1+a^2))

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red a b", e.g., "4 3 0.3 0.5" or [4,3,0.3,0.5]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 6)

def gvitm(raster, band_indexes= "1 2 3 4 5 6", astype=None):
    """
    Green Vegetation Index - Landsat TM

    GVITM = -0.2848*Band1-0.2435*Band2-0.5436*Band3+0.7243*Band4+0.0840*Band5-0.1800*Band7

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red", e.g., "4 3" or [4,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 7)

def sultan(raster, band_indexes="1 2 3 4 5 6", astype=None):
    """
    Sultan's Formula (transform to 3 band 8 bit image)

        | Band 1 = (Band5 / Band6) x 100
        | Band 2 = (Band5 / Band1) x 100
        | Band 3 = (Band3 / Band4) x (Band5 / Band4) x 100

    :param raster: the input raster / imagery layer
    :param band_indexes: "Band1 Band2 Band3 Band4 Band5 Band6", e.g., "1 2 3 4 5 6" or [1,2,3,4,5,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 8)

def vari(raster, band_indexes="3 2 1", astype=None):
    """
    Visible Atmospherically Resistant Index

    VARI = (Green - Red)/(Green + Red - Blue)

    :param raster: the input raster / imagery layer
    :param band_indexes: "Red Green Blue", e.g., "3 2 1" or [3,2,1]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 9)

def gndvi(raster, band_indexes="4 2", astype=None):
    """
    Green Normalized Difference Vegetation Index

    GNDVI = (NIR-Green)/(NIR+Green)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Green", e.g., "5 3" or [5,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 10)

def sr(raster, band_indexes="4 3", astype=None):
    """
    Simple Ratio (SR)

    SR = NIR / Red

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red", e.g., "4 3" or [4,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 11)

def ndvire(raster, band_indexes="7 6", astype=None):
    """
    Red-Edge NDVI (NDVIre)
    The Red-Edge NDVI (NDVIre) is a vegetation index for estimating 
    vegetation health using the red-edge band. It is especially useful 
    for estimating crop health in the mid to late stages of growth where 
    the chlorophyll concentration is relatively higher. Also, NDVIre can
    be used to map the within-field variability of nitrogen foliage to 
    understand the fertilizer requirements of crops.

    NDVIre = (NIR-RedEdge)/(NIR+RedEdge)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR RedEdge", e.g., "7 6" or [7,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 12)

def srre(raster, band_indexes="7 6", astype=None):
    """
    The Red-Edge Simple Ratio (SRre) is a vegetation index for estimating the 
    amount of healthy and stressed vegetation. It is the ratio of light scattered 
    in the NIR and red-edge bands, which reduces the effects of atmosphere and topography.

    Values are high for vegetation with high canopy closure and healthy vegetation, 
    lower for high canopy closure and stressed vegetation, and low for soil, water, 
    and nonvegetated features. The range of values is from 0 to about 30, where healthy 
    vegetation generally falls between values of 1 to 10.

    SRre = NIR / RedEdge

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR RedEdge", e.g., "7 6" or [7,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 13)

def mtvi2(raster, band_indexes="7 5 3", astype=None):
    """
    The Modified Triangular Vegetation Index (MTVI2) is a vegetation index 
    for detecting leaf chlorophyll content at the canopy scale while being 
    relatively insensitive to leaf area index. It uses reflectance in the green, 
    red, and near-infrared (NIR) bands

    MTVI2 = (1.5*(1.2*(NIR-Green)-2.5*(Red-Green))/sqrt((2*NIR+1)^2-(6*NIR-5*sqrt(Red))-0.5))

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red Green", e.g., "7 5 3" or [7,5,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 14)

def rtvi_core(raster, band_indexes="7 6 3", astype=None):
    """
    The Red-Edge Triangulated Vegetation Index (RTVICore) is a vegetation index 
    for estimating leaf area index and biomass. This index uses reflectance 
    in the NIR, red-edge, and green spectral bands

    RTVICore = [100(NIR-RedEdge)-10(NIR-Green)]

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR RedEdge Green", e.g., "7 6 3" or [7,6,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 15)

def cire(raster, band_indexes="7 6", astype=None):
    """
    The Chlorophyll Index - Red-Edge (CIre) is a vegetation index for estimating 
    the chlorophyll content in leaves using the ratio of reflectivity in the 
    near-infrared (NIR) and red-edge bands.

    CIre = [(NIR / RedEdge)-1]

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR RedEdge", e.g., "7 6" or [7,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 16)

def cig(raster, band_indexes="7 3", astype=None):
    """
    The Chlorophyll Index - Green (CIg) is a vegetation index for estimating 
    the chlorophyll content in leaves using the ratio of reflectivity in 
    the near-infrared (NIR) and green bands.

    CIg = [(NIR / Green)-1]

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Green", e.g., "7 3" or [7,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 17)

def ndwi(raster, band_indexes="5 3", astype=None):
    """
    The Normalized Difference Water Index (NDWI) is an index for delineating and 
    monitoring content changes in surface water. It is computed with the near-infrared 
    (NIR) and green bands.

    NDWI = (Green - NIR)/(Green +NIR)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Green", e.g., "5 3" or [5,3]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 18)

def evi(raster, band_indexes="5 4 2", astype=None):
    """
    The Enhanced Vegetation Index (EVI) is an optimized vegetation index that accounts 
    for atmospheric influences and vegetation background signal. It's similar to NDVI, 
    but is less sensitive to background and atmospheric noise, and it does not become 
    saturated NDVI when viewing areas with very dense green vegetation.

    EVI =  2.5 * [(NIR - Red)/(NIR + (6*Red) - (7.5*Blue) + 1)]

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR Red Blue", e.g., "5 4 2" or [5,4,2]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 19)

def iron_oxide(raster, band_indexes="4 2", astype=None):
    """
    The Iron Oxide (IO) ratio is a geological index for identifying rock 
    features that have experienced oxidation of iron-bearing sulfides 
    using the red and blue bands. IO is useful in identifying iron oxide 
    features below vegetation canopies, and is used in mineral composite mapping.

    IronOxide = Red / Blue

    :param raster: the input raster / imagery layer
    :param band_indexes: "Red Blue", e.g., "4 2" or [4,2]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 20)

def ferrous_minerals(raster, band_indexes="6 5", astype=None):
    """
    The Ferrous Minerals (FM) ratio is a geological index for identifying 
    rock features containing some quantity of iron-bearing minerals using
    the shortwave infrared (SWIR) and near-infrared (NIR) bands. FM is used
    in mineral composite mapping.

    FM = SWIR / NIR

    :param raster: the input raster / imagery layer
    :param band_indexes: "SWIR NIR", e.g., "6 5" or [6,5]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 21)

def clay_minerals(raster, band_indexes="6 7", astype=None):
    """
    The Clay Minerals (CM) ratio is a geological index for identifying 
    mineral features containing clay and alunite using two shortwave 
    infrared (SWIR) bands. CM is used in mineral composite mapping.

    CM = SWIR1 / SWIR2

    :param raster: the input raster / imagery layer
    :param band_indexes: "SWIR1 SWIR2", e.g., "6 7" or [6,7]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 22)

def wndwi(raster, band_indexes="2 5 6 0.5", astype=None):
    """
    The Weighted Normalized Difference Water Index (WNDWI) is a water index
    developed to reduce error typically encountered in other water indices,
    including water turbidity, small water bodies, or shadow in remote sensing scenes.
    Supported from 10.8.

    WNDWI = [Green – α * NIR – (1 – α) * SWIR ] / [Green + α * NIR + (1 – α) * SWIR]

    :param raster: the input raster / imagery layer
    :param band_indexes: "Green NIR SWIR α", e.g., "2 5 6 0.5" or [2,5,6,0.5]
    :param astype: output pixel type
    :return: output raster

    """
    return band_arithmetic(raster, band_indexes, astype, 23)

def bai(raster, band_indexes="3 4", astype=None):
    """
    The Burn Area Index (BAI) uses the reflectance values in the red and NIR portion of the spectrum to identify
    the areas of the terrain affected by fire.

    BAI = 1/((0.1 -RED)^2 + (0.06 - NIR)^2)

    :param raster: the input raster / imagery layer
    :param band_indexes: "Red NIR", e.g., "3 4" or [3,4]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 24)

def nbr(raster, band_indexes="5 7", astype=None):
    """
    The Normalized Burn Ratio Index (NBRI) uses the NIR and SWIR bands to emphasize burned areas,
    while mitigating illumination and atmospheric effects. Your images should be corrected to reflectance values
    before using this index.

    NBR = (NIR - SWIR) / (NIR+ SWIR)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR SWIR", e.g., "5 7" or [5,7]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 25)

def ndbi(raster, band_indexes="6 5", astype=None):
    """
    The Normalized Difference Built-up Index (NDBI) uses the NIR and SWIR bands to emphasize  man-made built-up areas.
    It is ratio based to mitigate the effects of terrain illumination differences as well as atmospheric effects.

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    :param raster: the input raster / imagery layer
    :param band_indexes: "SWIR NIR", e.g., "6 5" or [6,5]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 26)

def ndmi(raster, band_indexes="5 6", astype=None):
    """
    The Normalized Difference Moisture Index (NDMI) is sensitive to the moisture levels in vegetation.
    It is used to monitor droughts as well as monitor fuel levels in fire-prone areas. It uses NIR and SWIR bands to
    create a ratio designed to mitigate illumination and atmospheric effects.

    NDMI = (NIR - SWIR1)/(NIR + SWIR1)

    :param raster: the input raster / imagery layer
    :param band_indexes: "NIR SWIR1", e.g., "5 6" or [5,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 27)

def ndsi(raster, band_indexes="4 6", astype=None):
    """
    The Normalized Difference Snow Index (NDSI) is designed to use MODIS (band 4 and band 6) and
    Landsat TM (band 2 and band 5) for identification of snow cover while ignoring cloud cover. Since it is ratio based,
    it also mitigates atmospheric effects.

    NDSI = (Green - SWIR) / (Green + SWIR)

    :param raster: the input raster / imagery layer
    :param band_indexes: "Green SWIR", e.g., "4 6" or [4,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 28)

def mndwi(raster, band_indexes="3 6", astype=None):
    """
    The Modified Normalized Difference Water Index (MNDWI) uses green and SWIR bands for the enhancement 
    of open water features. It also diminishes built-up area features that are often correlated with open
    water in other indices.

    MNDWI = (Green - SWIR) / (Green + SWIR)

    :param raster: the input raster / imagery layer
    :param band_indexes: "Green SWIR", e.g., "3 6" or [3,6]
    :param astype: output pixel type
    :return: output raster
    """
    return band_arithmetic(raster, band_indexes, astype, 29)

def expression(raster, expression="(B3 - B1 / B3 + B1)", astype=None):
    """
    Use a single-line algebraic formula to create a single-band output. The supported operators are -, +, /, *, and unary -.
    To identify the bands, prepend the band number with a B or b. For example: "BandIndexes":"(B1 + B2) / (B3 * B5)"

    :param raster: the input raster / imagery layer
    :param expression: the algebric formula
    :param astype: output pixel type
    :return: output raster
    :return:
    """
    return band_arithmetic(raster, expression, astype, 0)

def classify(raster1, raster2=None, classifier_definition=None, astype=None):
    """
    classifies a segmented raster to a categorical raster.

    :param raster1: the first raster - imagery layers filtered by where clause, spatial and temporal filters
    :param raster2: Optional segmentation raster -  If provided, pixels in each segment will get same class assignments. 
                    imagery layers filtered by where clause, spatial and temporal filters
    :param classifier_definition: the classifier parameters as a Python dictionary / json format

    :return: the output raster with this function applied to it
    """

    layer1, raster_1, raster_ra1 = _raster_input(raster1)
    layer2=None
    if raster2 is not None:
        layer2, raster_2, raster_ra2 = _raster_input(raster1, raster2)

    if layer1 is not None or (layer2 is not None and layer2._datastore_raster is False):
        layer = layer1
    else:
        layer = layer2

    template_dict = {
        "rasterFunction": "Classify",
        "rasterFunctionArguments": {
            "ClassifierDefinition": classifier_definition,
            "Raster": raster_1
        }
    }
    if classifier_definition is None:
        raise RuntimeError("classifier_definition cannot be empty")
    template_dict["rasterFunctionArguments"]["ClassifierDefinition"] = classifier_definition

    if raster2 is not None:
        template_dict["rasterFunctionArguments"]["Raster2"] = raster_2

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if raster2 is not None:
        return _clone_layer(layer, template_dict, raster_ra1, raster_ra2)
    return _clone_layer(layer, template_dict, raster_ra1)

def clip(raster, geometry=None, clip_outside=True, astype=None):
    """
    Clips a raster using a rectangular shape according to the extents defined or will clip a raster to the shape of an
    input polygon. The shape defining the clip can clip the extent of the raster or clip out an area within the raster.

    :param raster: input raster
    :param geometry: clipping geometry
    :param clip_outside: boolean, If True, the imagery outside the extents will be removed, else the imagery within the
            clipping_geometry will be removed.
    :param astype: output pixel type
    :return: the clipped raster
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Clip",
        "rasterFunctionArguments": {
            "ClippingGeometry": geometry,
            "ClipType": 1 if clip_outside else 2,
            "Raster": raster
        }
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)


def colormap(raster, colormap_name=None, colormap=None, colorramp=None, astype=None):
    """
    Transforms the pixel values to display the raster data as a color (RGB) image, based on specific colors in
    a color map. For more information, see Colormap function at
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/colormap-function.htm

    :param raster: input raster
    :param colormap_name: colormap name, if one of Random | NDVI | Elevation | Gray
    :param colormap: | [
                     | [<value1>, <red1>, <green1>, <blue1>], //[int, int, int, int]
                     | [<value2>, <red2>, <green2>, <blue2>]
                     | ],
    :param colorramp: Can be a string specifiying color ramp name like <Black To White|Yellow To Red|Slope|more..>
                      or a color ramp object. 
                      For more information about colorramp object, see color ramp object at
                      https://developers.arcgis.com/documentation/common-data-types/color-ramp-objects.htm)
    :param astype: output pixel type
    :return: the colorized raster
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Colormap",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if colormap_name is not None:
        template_dict["rasterFunctionArguments"]['ColormapName'] = colormap_name
    if colormap is not None:
        template_dict["rasterFunctionArguments"]['Colormap'] = colormap
    if colorramp is not None and isinstance(colorramp,str):
        template_dict["rasterFunctionArguments"]['ColorrampName'] = colorramp
    if colorramp is not None and isinstance(colorramp, dict):
        template_dict["rasterFunctionArguments"]['Colorramp'] = colorramp

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)


def composite_band(rasters, astype=None, cellsize_type='MaxOf'):
    """
    Combines multiple images to form a multiband image.

    :param rasters: input rasters
    :param astype: output pixel type
    :param cellsize_type: The cell size used to create the output raster.
                          one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
    :return: the multiband image
    """
    layer, raster, raster_ra = _raster_input(rasters)

    template_dict = {
        "rasterFunction": "CompositeBand",
        "rasterFunctionArguments": {
            "Rasters": raster
        },
        "variableName": "Rasters"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    cellsize_types = {
        "firstof" : 0,
        "minof" : 1,
        "maxof" : 2,
        "meanof" : 3,
        "lastof" : 4
    }      

    if cellsize_type is not None:
        if isinstance(cellsize_type, str):
            template_dict["rasterFunctionArguments"]['CellsizeType'] = cellsize_types[cellsize_type.lower()]
        elif isinstance(cellsize_type,int):
            template_dict["rasterFunctionArguments"]['CellsizeType'] = cellsize_type

    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')

def contrast_brightness(raster, contrast_offset=2, brightness_offset=1, astype=None):
    """
    The ContrastBrightness function enhances the appearance of raster data (imagery) by modifying the brightness or
    contrast within the image. This function works on 8-bit input raster only.

    :param raster: input raster
    :param contrast_offset: double, -100 to 100
    :param brightness_offset: double, -100 to 100
    :param astype: pixel type of result raster
    :return: output raster
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
      "rasterFunction" : "ContrastBrightness",
      "rasterFunctionArguments" : {
        "Raster": raster,
        "ContrastOffset" : contrast_offset,
        "BrightnessOffset" : brightness_offset
      },
      "variableName" : "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)


def convolution(raster, kernel=None, astype=None):
    """
    The Convolution function performs filtering on the pixel values in an image, which can be used for sharpening an
    image, blurring an image, detecting edges within an image, or other kernel-based enhancements. For more information,
    see Convolution function at http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/convolution-function.htm

    :param raster: input raster
    :param kernel: well known kernel from arcgis.raster.kernels or user defined kernel passed as a list of list
    :param astype: pixel type of result raster
    :return: output raster
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
      "rasterFunction" : "Convolution",
      "rasterFunctionArguments" : {
        "Raster": raster,
      },
      "variableName" : "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if (isinstance(kernel, int)):
        template_dict["rasterFunctionArguments"]['Type'] = kernel
    elif (isinstance(kernel, list)):
        numrows = len(kernel)
        numcols = len(kernel[0])
        flattened = [item for sublist in kernel for item in sublist]
        template_dict["rasterFunctionArguments"]['Columns'] = numcols
        template_dict["rasterFunctionArguments"]['Rows'] = numrows
        template_dict["rasterFunctionArguments"]['Kernel'] = flattened
    else:
        raise RuntimeError('Invalid kernel type - pass int or list of list: [[][][]...]')

    return _clone_layer(layer, template_dict, raster_ra)


def curvature(raster, curvature_type='standard', z_factor=1, astype=None):
    """
    The Curvature function displays the shape or curvature of the slope. A part of a surface can be concave or convex;
    you can tell that by looking at the curvature value. The curvature is calculated by computing the second derivative
    of the surface. Refer to this conceptual help on how it works.

    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/curvature-function.htm

    :param raster: input raster
    :param curvature_type: 'standard', 'planform', 'profile'
    :param z_factor: double
    :param astype: output pixel type
    :return: the output raster
    """
    layer, raster, raster_ra = _raster_input(raster)


    curv_types = {
        'standard': 0,
        'profile': 1,
        'planform': 2
    }

    in_curv_type = curv_types[curvature_type.lower()]

    template_dict = {
        "rasterFunction": "Curvature",
        "rasterFunctionArguments": {
            "Raster": raster,
            "Type": in_curv_type,
            "ZFactor": z_factor
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)


def NDVI(raster, visible_band=2, ir_band=1, astype=None):
    """
    The Normalized Difference Vegetation Index (ndvi) is a standardized index that allows you to generate an image
    displaying greenness (relative biomass). This index takes advantage of the contrast of the characteristics of
    two bands from a multispectral raster dataset the chlorophyll pigment absorptions in the red band and the
    high reflectivity of plant materials in the near-infrared (NIR) band. For more information, see ndvi function.
    The arguments for the ndvi function are as follows:

    :param raster: input raster
    :param visible_band_id: int (zero-based band id, e.g. 2)
    :param infrared_band_id: int (zero-based band id, e.g. 1)
    :param astype: output pixel type
    :return: the output raster
    The following equation is used by the NDVI function to generate a 0 200 range 8 bit result:
    NDVI = ((IR - R)/(IR + R)) * 100 + 100
    If you need the specific pixel values (-1.0 to 1.0), use the lowercase ndvi method.
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
      "rasterFunction" : "NDVI",
      "rasterFunctionArguments" : {
        "Raster": raster,
        "VisibleBandID" : visible_band,
        "InfraredBandID" : ir_band
      },
      "variableName" : "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra)


def elevation_void_fill(raster, max_void_width=0, astype=None):
    """
    The elevation_void_fill function is used to create pixels where holes exist in your elevation. Refer to
    `this conceptual help <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/elevation-void-fill-function.htm>`__
    on how it works. The arguments for the elevation_void_fill function are as follows:

    :param raster: input raster
    :param max_void_width: number. Maximum void width to fill. 0: fill all
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "ElevationVoidFill",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if max_void_width is not None:
        template_dict["rasterFunctionArguments"]["MaxVoidWidth"] = max_void_width

    return _clone_layer(layer, template_dict, raster_ra)


def extract_band(raster, band_ids=None, band_names=None, band_wavelengths=None, missing_band_action=None,
                 wavelength_match_tolerance=None, astype=None):
    """
    The extract_band function allows you to extract one or more bands from a raster, or it can reorder the bands in a
    multiband image. The arguments for the extract_band function are as follows:

    :param raster: input raster
    :param band_ids: array of int, band_ids uses one-based indexing.
    :param band_names: array of string
    :param band_wavelengths: array of double
    :param missing_band_action: int, 0 = esriMissingBandActionFindBestMatch, 1 = esriMissingBandActionFail
    :param wavelength_match_tolerance: double
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "ExtractBand",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if band_ids is not None:
        band_ids_mod=[]
        if isinstance(band_ids,list):
            for index, item in enumerate(band_ids):
                band_ids_mod.append(item-1)
            template_dict["rasterFunctionArguments"]["BandIDs"] = band_ids_mod
        else:
            raise RuntimeError("band_ids should be of type list")
    if band_names is not None:
        template_dict["rasterFunctionArguments"]["BandNames"] = band_names
    if band_wavelengths is not None:
        template_dict["rasterFunctionArguments"]["BandWavelengths"] = band_wavelengths
    if missing_band_action is not None:
        template_dict["rasterFunctionArguments"]["MissingBandAction"] = missing_band_action
    if wavelength_match_tolerance is not None:
        template_dict["rasterFunctionArguments"]["WavelengthMatchTolerance"] = wavelength_match_tolerance

    return _clone_layer(layer, template_dict, raster_ra)


def geometric(raster, geodata_transforms=None, append_geodata_xform=None, z_factor=None, z_offset=None, constant_z=None,
              correct_geoid=None, astype=None, tolerance=None, dem= None):
    """
    The geometric function transforms the image (for example, orthorectification) based on a sensor definition and a
    terrain model.This function was added at 10.1.The arguments for the geometric function are as follows:

    :param raster: The input raster.
    :param geodata_transforms: Please refer to the Geodata Transformations documentation for more details.
    :param append_geodata_xform: Optional boolean. Indicates whether the geodata transformation is appended to the existing one from the input raster.
    :param z_factor: Optional double. Satellite rational polynomial coefficients (RPCs) are scaled for 
                     elevation datasets with vertical units in meters. If your elevation uses other 
                     vertical units, enter a Z Factor to rescale to meters. For example, if your 
                     elevation units are in feet, you would use a value of 0.3048 to convert your 
                     elevation units from feet to meters.
    :param z_offset: Optional double. The base value to be added to the elevation value in the DEM. 
                     This could be used to offset elevation values that do not start at sea level.
    :param constant_z: Optional double. Specify a constant elevation to use for the Geometric function.
    :param correct_geoid: Optional boolean. Set it to True to apply the geoid (EGM96) correction to the z-values, 
                          unless your DEM is already referenced to ellipsoidal heights.
    :param tolerance: Optional double. Specify the maximum tolerable error in the geometric function, given in number of pixels.
    :param dem: Optional Raster. Specify the DEM to use for the Geometric function

    """

    layer1, raster_1, raster_ra1 = _raster_input(raster)

    layer2=None
    if dem is not None:
        layer2, raster_2, raster_ra2 = _raster_input(raster, dem)

    template_dict = {
        "rasterFunction": "Geometric",
        "rasterFunctionArguments": {
            "Raster": raster_1
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if geodata_transforms is not None:
        template_dict["rasterFunctionArguments"]["GeodataTransforms"] = geodata_transforms
    if append_geodata_xform is not None:
        template_dict["rasterFunctionArguments"]["AppendGeodataXform"] = append_geodata_xform
    if z_factor is not None:
        template_dict["rasterFunctionArguments"]["ZFactor"] = z_factor
    if z_offset is not None:
        template_dict["rasterFunctionArguments"]["ZOffset"] = z_offset
    if constant_z is not None:
        template_dict["rasterFunctionArguments"]["ConstantZ"] = constant_z
    if correct_geoid is not None:
        template_dict["rasterFunctionArguments"]["CorrectGeoid"] = correct_geoid
    if tolerance is not None:
        template_dict["rasterFunctionArguments"]["Tolerance"] = tolerance

    if dem is not None:
        template_dict["rasterFunctionArguments"]["DEM"] = raster_2
        return _clone_layer(layer1, template_dict, raster_ra1, raster_ra2)
    return _clone_layer(layer1, template_dict, raster_ra1)


def hillshade(dem, azimuth=215.0, altitude=75.0, z_factor=0.3, slope_type=1, ps_power=None, psz_factor=None,
              remove_edge_effect=None, astype=None, hillshade_type=0):
    """
    A hillshade is a grayscale 3D model of the surface taking the sun's relative position into account to shade the image.
    For more information, see
    `hillshade function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/hillshade-function.htm>`__ and `How hillshade works <http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/how-hillshade-works.htm>`__.
    
    The arguments for the hillshade function are as follows:

    :param dem: input DEM
    :param azimuth: double (e.g. 215.0)
    :param altitude: double (e.g. 75.0)
    :param z_factor: double (e.g. 0.3)
    :param slope_type: new at 10.2. 1=DEGREE, 2=PERCENTRISE, 3=SCALED. default is 1.
    :param ps_power: new at 10.2. double, used together with SCALED slope type
    :param psz_factor: new at 10.2. double, used together with SCALED slope type
    :param remove_edge_effect: new at 10.2. boolean, true of false
    :param astype: output pixel type
    :param hillshade_type: new at 10.5.1 0 = traditional, 1 = multi - directional; default is 0
    :return: the output raster

    """
    raster = dem

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Hillshade",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if azimuth is not None:
        template_dict["rasterFunctionArguments"]["Azimuth"] = azimuth
    if altitude is not None:
        template_dict["rasterFunctionArguments"]["Altitude"] = altitude
    if z_factor is not None:
        template_dict["rasterFunctionArguments"]["ZFactor"] = z_factor
    if slope_type is not None:
        template_dict["rasterFunctionArguments"]["SlopeType"] = slope_type
    if ps_power is not None:
        template_dict["rasterFunctionArguments"]["PSPower"] = ps_power
    if psz_factor is not None:
        template_dict["rasterFunctionArguments"]["PSZFactor"] = psz_factor
    if remove_edge_effect is not None:
        template_dict["rasterFunctionArguments"]["RemoveEdgeEffect"] = remove_edge_effect
    if hillshade_type is not None:
        template_dict["rasterFunctionArguments"]["HillshadeType"] = hillshade_type

    return _clone_layer(layer, template_dict, raster_ra)


def local(rasters, operation, extent_type="FirstOf", cellsize_type="FirstOf", astype=None, process_as_multiband=None):
    """
    The local function allows you to perform bitwise, conditional, logical, mathematical, and statistical operations on
    a pixel-by-pixel basis. For more information, see `local function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/local-function.htm>`__.

    License:At 10.5, you must license your ArcGIS Server as ArcGIS Server 10.5.1 Enterprise Advanced or
    ArcGIS Image Server to use this resource. At versions prior to 10.5, the hosting ArcGIS Server needs to have a Spatial Analyst license.

    The arguments for the local function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param operation: int. see reference `here <https://desktop.arcgis.com/en/arcobjects/latest/net/webframe.htm#esriGeoAnalysisFunctionEnum.htm>`__.
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
                                 Applicable for operations - Majority, Maximum, Mean, Median, Minimum, Minority, Range, Standard Deviation, Sum, and Variety.
    :return: the output raster

    """
    # redacted - The local function works on single band or the first band of an image only, and the output is single band.
    raster = rasters

    layer, raster, raster_ra = _raster_input(raster)


    extent_types = {
        "FirstOf" : 0,
        "IntersectionOf" : 1,
        "UnionOf" : 2,
        "LastOf" : 3
    }

    cellsize_types = {
        "FirstOf" : 0,
        "MinOf" : 1,
        "MaxOf" : 2,
        "MeanOf" : 3,
        "LastOf" : 4
    }

    in_extent_type = extent_types[extent_type]
    in_cellsize_type = cellsize_types[cellsize_type]

    template_dict = {
        "rasterFunction": "Local",
        "rasterFunctionArguments": {
            "Rasters": raster
        },
        "variableName": "Rasters"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if operation is not None:
        template_dict["rasterFunctionArguments"]["Operation"] = operation
    if extent_type is not None:
        template_dict["rasterFunctionArguments"]["ExtentType"] = in_extent_type
    if cellsize_type is not None:
        template_dict["rasterFunctionArguments"]["CellsizeType"] = in_cellsize_type

    if process_as_multiband is not None:
        if isinstance(process_as_multiband, bool):
            template_dict["rasterFunctionArguments"]["ProcessAsMultiband"] = process_as_multiband
        else:
            raise RuntimeError('process_as_multiband should be an instance of bool')


    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')


###############################################  LOCAL FUNCTIONS  ######################################################

def plus(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The binary Plus (addition,+) operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 1, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def minus(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The binary Minus (subtraction,-) operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 2, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def times(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Times (multiplication,*) operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 3, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def sqrt(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Square Root operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 4, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def power(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Power operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 5, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def acos(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The acos operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 6, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def asin(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The asin operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 7, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def atan(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The ATan operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 8, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def atanh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The ATanH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 9, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def abs(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Abs operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 10, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_and(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseAnd operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 11, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_left_shift(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseLeftShift operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 12, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_not(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseNot operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 13, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_or(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseOr operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 14, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_right_shift(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseRightShift operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 15, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def bitwise_xor(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BitwiseXOr operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 16, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def boolean_and(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BooleanAnd operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 17, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def boolean_not(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BooleanNot operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 18, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def boolean_or(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BooleanOr operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 19, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def boolean_xor(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The BooleanXOr operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 20, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def cos(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Cos operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 21, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def cosh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The CosH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 22, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def divide(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Divide operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 23, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def equal_to(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The EqualTo operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 24, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def exp(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Exp operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 25, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def exp10(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Exp10 operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 26, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def exp2(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Exp2 operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 27, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def greater_than(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The GreaterThan operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 28, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def greater_than_equal(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The GreaterThanEqual operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 29, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def INT(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Int operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 30, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def is_null(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The IsNull operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 31, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def FLOAT(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Float operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 32, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def less_than(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The LessThan operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 33, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def less_than_equal(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The LessThanEqual operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 34, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def ln(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Ln operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 35, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def log10(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Log10 operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 36, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def log2(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Log2 operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 37, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def majority(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Majority operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: Set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 66 if ignore_nodata else 38
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband= process_as_multiband)


def max(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Max operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :param process_as_multiband: Set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 67 if ignore_nodata else 39
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def mean(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Mean operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: Set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 68 if ignore_nodata else 40
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def med(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Med operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: Set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 69 if ignore_nodata else 41
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def min(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Min operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: Set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 70 if ignore_nodata else 42
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def minority(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Minority operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 71 if ignore_nodata else 43
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def mod(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Mod operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 44, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def negate(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Negate operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 45, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def not_equal(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The NotEqual operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 46, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def cellstats_range(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Range operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 72 if ignore_nodata else 47
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def round_down(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The RoundDown operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 48, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def round_up(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The RoundUp operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 49, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def set_null(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The SetNull operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 50, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def sin(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Sin operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 51, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def sinh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The SinH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 52, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def square(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Square operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 53, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def std(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Std operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 73 if ignore_nodata else 54
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def sum(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False,  astype=None, process_as_multiband=None):
    """
    The Sum operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 74 if ignore_nodata else 55
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def tan(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The Tan operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 56, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def tanh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The TanH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 57, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def variety(rasters, extent_type="FirstOf", cellsize_type="FirstOf", ignore_nodata=False, astype=None, process_as_multiband=None):
    """
    The Variety operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param ignore_nodata: True or False, set to True to ignore NoData values
    :param astype: output pixel type
    :param process_as_multiband: True or False, set to True to process as multiband. 
    :return: the output raster

    """
    opnum = 75 if ignore_nodata else 58
    return local(rasters, opnum, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype, process_as_multiband=process_as_multiband)


def acosh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The ACosH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 59, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def asinh(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The ASinH operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 60, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def atan2(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The ATan2 operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 61, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def float_divide(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The FloatDivide operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 64, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def floor_divide(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The FloorDivide operation

    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 65, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def con(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The con operation.Performs a conditional if/else evaluation on each of the input cells of an input raster. For more information see, http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/con-.htm
    
    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :returns: the output raster  

    .. code-block:: python

        USAGE EXAMPLE: To extract raster from flow direction raster layer that only covers the watershed. 
                       rasters: 
                       ["Input raster representing the true or false result of the desired condition. It can be of integer or floating point type.", 
                        "The input whose values will be used as the output cell values if the condition is true. It can be an integer or a floating point raster, or a constant value.", 
                        "The input whose values will be used as the output cell values if the condition is false. It can be an integer or a floating point raster, or a constant value."]

        con([stowe_watershed_lyr, Stowe_fill_flow_direction_lyr, 0])

    """
    return local(rasters, 78, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)


def _pick(rasters, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The value from a position raster is used to determine from which raster in 
    a list of input rasters the output cell value will be obtained.
    The arguments for this function are as follows:

    :param rasters: array of rasters. If a scalar is needed for the operation, the scalar can be a double or string
    :param extent_type: one of "FirstOf", "IntersectionOf", "UnionOf", "LastOf"
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf, "MeanOf", "LastOf"
    :param astype: output pixel type
    :return: the output raster

    """
    return local(rasters, 84, extent_type=extent_type, cellsize_type=cellsize_type, astype=astype)

###############################################  LOCAL FUNCTIONS  ######################################################


def mask(raster, no_data_values=None, included_ranges=None, no_data_interpretation=None, astype=None):
    """
    The mask function changes the image by specifying a certain pixel value or a range of pixel values as no data.
    The arguments for the mask function are as follows:

    :param raster: input raster
    :param no_data_values: list of strings ["band0_val","band1_val",...]. The NoData values can be specified 
                           for each band. The index of each element in no_data_values list 
                           represents the no data value in the corresponding band.

                           You can specify more than one value by entering a space-delimited string for each index.
                           e.g., ["band0_val1 band0_val2", "band1_val1 band1_val2",...]
    :param included_ranges: list of floats [band0_lowerbound,band0_upperbound,band1_lowerbound,band1_upperbound, band2_.....], 
                            The included ranges can be specified for each band by specifying a minimum and maximum value.
    :param no_data_interpretation: int 0=MatchAny, 1=MatchAll. This parameter refers to how the NoData 
                                   values will impact the output image.

                                   - 0 (MatchAny) : If the NoData value you specify occurs for a cell in a
                                     specified band, then that cell in the output image will be NoData.
                                   - 1 (MatchAll) :  The NoData values you specify for each band must occur 
                                     in the same cell for the output image to contain the NoData cell.
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Mask",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if no_data_values is not None:
        template_dict["rasterFunctionArguments"]["NoDataValues"] = no_data_values
    if included_ranges is not None:
        template_dict["rasterFunctionArguments"]["IncludedRanges"] = included_ranges
    if no_data_interpretation is not None:
        template_dict["rasterFunctionArguments"]["NoDataInterpretation"] = no_data_interpretation

    return _clone_layer(layer, template_dict, raster_ra)


def ml_classify(raster, signature, astype=None):
    """
    The ml_classify function allows you to perform a supervised classification using the maximum likelihood classification
    algorithm. The hosting ArcGIS Server needs to have a Spatial Analyst license.LicenseLicense:At 10.5, you must license
    your ArcGIS Server as ArcGIS Server 10.5.1 Enterprise Advanced or ArcGIS Image Server to use this resource.
    At versions prior to 10.5, the hosting ArcGIS Server needs to have a Spatial Analyst license.
    
    The arguments for the ml_classify function are as follows:

    :param raster: input raster
    :param signature: string. a signature string returned from computeClassStatistics (GSG)
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "MLClassify",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if signature is not None:
        template_dict["rasterFunctionArguments"]["SignatureFile"] = signature

    return _clone_layer(layer, template_dict, raster_ra)

# See NDVI() above
# def ndvi(raster, visible_band_id=None, infrared_band_id=None, astype=None):
#     """
#     The Normalized Difference Vegetation Index (ndvi) is a standardized index that allows you to generate an image displaying greenness (relative biomass). This index takes advantage of the contrast of the characteristics of two bands from a multispectral raster dataset the chlorophyll pigment absorptions in the red band and the high reflectivity of plant materials in the near-infrared (NIR) band. For more information, see ndvi function.The arguments for the ndvi function are as follows:
#
#     :param raster: input raster
#     :param visible_band_id: int (zero-based band id, e.g. 2)
#     :param infrared_band_id: int (zero-based band id, e.g. 1)
#     :param astype: output pixel type
#     :return: the output raster
#
#     """
#
#     layer, raster, raster_ra = _raster_input(raster)
#
#     template_dict = {
#         "rasterFunction": "NDVI",
#         "rasterFunctionArguments": {
#             "Raster": raster
#         },
#         "variableName": "Raster"
#     }
#
#     if astype is not None:
#         template_dict["outputPixelType"] = astype.upper()
#
#     if visible_band_id is not None:
#         template_dict["rasterFunctionArguments"]["VisibleBandID"] = visible_band_id
#     if infrared_band_id is not None:
#         template_dict["rasterFunctionArguments"]["InfraredBandID"] = infrared_band_id
#
#     return {
#         'layer': layer,
#         'function_chain': template_dict
#     }

# TODO: how does recast work?
# def recast(raster, < _argument_name1 >= None, < _argument_name2 >= None, astype=None):
#     """
#     The recast function reassigns argument values in an existing function template.The arguments for the recast function are based on the function it is overwriting.
#
#     :param raster: input raster
#     :param <_argument_name1>: ArgumentName1 will be reassigned with ArgumentValue1
#     :param <_argument_name2>: ArgumentName1 will be reassigned with ArgumentValue2
#     :param astype: output pixel type
#     :return: the output raster
#
#     """
#
#     layer, raster, raster_ra = _raster_input(raster)
#
#     template_dict = {
#         "rasterFunction": "Recast",
#         "rasterFunctionArguments": {
#             "Raster": raster
#         },
#         "variableName": "Raster"
#     }
#
#     if astype is not None:
#         template_dict["outputPixelType"] = astype.upper()
#
#     if < _argument_name1 > is not None:
#         template_dict["rasterFunctionArguments"]["<ArgumentName1>"] = < _argument_name1 >
#     if < _argument_name2 > is not None:
#         template_dict["rasterFunctionArguments"]["<ArgumentName2>"] = < _argument_name2 >
#
#     return {
#         'layer': layer,
#         'function_chain': template_dict
#     }


def remap(raster, input_ranges=None, output_values=None, geometry_type=None, geometries=None, no_data_ranges=None,
          allow_unmatched=None, astype=None):
    """
    The remap function allows you to change or reclassify the pixel values of the raster data. For more information,
    see `remap function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/remap-function.htm>`__.

    The arguments for the remap function are as follows:

    :param raster: input raster
    :param input_ranges: [double, double,...], input ranges are specified in pairs: from (inclusive) and to (exclusive).
    :param output_values: [double, ...], output values of corresponding input ranges
    :param geometry_type: added at 10.3
    :param geometries: added at 10.3
    :param no_data_ranges: [double, double, ...], nodata ranges are specified in pairs: from (inclusive) and to (exclusive).
    :param allow_unmatched: Boolean, specify whether to keep the unmatched values or turn into nodata.
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Remap",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if input_ranges is not None:
        template_dict["rasterFunctionArguments"]["InputRanges"] = input_ranges
    if output_values is not None:
        template_dict["rasterFunctionArguments"]["OutputValues"] = output_values
    if geometry_type is not None:
        template_dict["rasterFunctionArguments"]["GeometryType"] = geometry_type
    if geometries is not None:
        template_dict["rasterFunctionArguments"]["Geometries"] = geometries
    if no_data_ranges is not None:
        template_dict["rasterFunctionArguments"]["NoDataRanges"] = no_data_ranges
    if allow_unmatched is not None:
        template_dict["rasterFunctionArguments"]["AllowUnmatched"] = allow_unmatched

    return _clone_layer(layer, template_dict, raster_ra)


def resample(raster, resampling_type=None, input_cellsize=None, output_cellsize=None, astype=None):
    """
    The resample function resamples pixel values from a given resolution.The arguments for the resample function are as follows:

    :param raster: input raster
    :param resampling_type: one of NearestNeighbor,Bilinear,Cubic,Majority,BilinearInterpolationPlus,BilinearGaussBlur,
            BilinearGaussBlurPlus, Average, Minimum, Maximum,VectorAverage(require two bands)
    :param input_cellsize: point that defines cellsize in source spatial reference
    :param output_cellsize: point that defines output cellsize
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)
    resample_types = {
        'NearestNeighbor': 0,
        'Bilinear': 1,
        'Cubic': 2,
        'Majority': 3,
        'BilinearInterpolationPlus': 4,
        'BilinearGaussBlur': 5,
        'BilinearGaussBlurPlus': 6,
        'Average': 7,
        'Minimum': 8,
        'Maximum': 9,
        'VectorAverage':10
    }

    if isinstance(resampling_type, str):
        resampling_type = resample_types[resampling_type]

    template_dict = {
        "rasterFunction": "Resample",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if resampling_type is not None:
        template_dict["rasterFunctionArguments"]["ResamplingType"] = resampling_type
    if input_cellsize is not None:
        template_dict["rasterFunctionArguments"]["InputCellsize"] = input_cellsize
    if output_cellsize is not None:
        template_dict["rasterFunctionArguments"]["OutputCellsize"] = output_cellsize

    return _clone_layer(layer, template_dict, raster_ra)


def segment_mean_shift(raster, spectral_detail=None, spatial_detail=None, spectral_radius=None, spatial_radius=None,
                       min_num_pixels_per_segment=None, astype=None):
    """
    The segment_mean_shift function produces a segmented output. Pixel values in the output image represent the
    converged RGB colors of the segment. The input raster needs to be a 3-band 8-bit image. If the imagery layer is not
    a 3-band 8-bit unsigned image, you can use the Stretch function before the segment_mean_shift function.

    License:At 10.5, you must license your ArcGIS Server as ArcGIS Server 10.5.1 Enterprise Advanced or
    ArcGIS Image Server to use this resource.
    At versions prior to 10.5, the hosting ArcGIS Server needs to have a Spatial Analyst license.

    When specifying arguments for SegmentMeanShift, use either SpectralDetail,SpatialDetail as a pair, or use
    SpectralRadius, SpatialRadius. They have an inverse relationship. SpectralRadius = 21 - SpectralDetail,
    SpatialRadius = 21 - SpectralRadius

    The arguments for the segment_mean_shift function are as follows:

    :param raster: input raster
    :param spectral_detail: double 0-21. Bigger value is faster and has more segments.
    :param spatial_detail: int 0-21. Bigger value is faster and has more segments.
    :param spectral_radius: double. Bigger value is slower and has less segments.
    :param spatial_radius: int. Bigger value is slower and has less segments.
    :param min_num_pixels_per_segment: int
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "SegmentMeanShift",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if spectral_detail is not None:
        template_dict["rasterFunctionArguments"]["SpectralDetail"] = spectral_detail
    if spatial_detail is not None:
        template_dict["rasterFunctionArguments"]["SpatialDetail"] = spatial_detail
    if spectral_radius is not None:
        template_dict["rasterFunctionArguments"]["SpectralRadius"] = spectral_radius
    if spatial_radius is not None:
        template_dict["rasterFunctionArguments"]["SpatialRadius"] = spatial_radius
    if min_num_pixels_per_segment is not None:
        template_dict["rasterFunctionArguments"]["MinNumPixelsPerSegment"] = min_num_pixels_per_segment

    return _clone_layer(layer, template_dict, raster_ra)


def shaded_relief(raster, azimuth=None, altitude=None, z_factor=None, colormap=None, slope_type=None, ps_power=None,
                  psz_factor=None, remove_edge_effect=None, astype=None, colorramp=None, hillshade_type=0):
    """
    Shaded relief is a color 3D model of the terrain, created by merging the images from the Elevation-coded and
    Hillshade methods. For more information, see `Shaded relief <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/shaded-relief-function.htm>`__ function.

    The arguments for the shaded_relief function are as follows:

    :param raster: input raster
    :param azimuth: double (e.g. 215.0)
    :param altitude: double (e.g. 75.0)
    :param z_factor: double (e.g. 0.3)
    :param colormap: [[<value1>, <red1>, <green1>, <blue1>], [<value2>, <red2>, <green2>, <blue2>]]
    :param slope_type: 1=DEGREE, 2=PERCENTRISE, 3=SCALED. default is 1.
    :param ps_power: double, used together with SCALED slope type
    :param psz_factor: double, used together with SCALED slope type
    :param remove_edge_effect: boolean, True or False
    :param astype: output pixel type
    :param colorramp: string, specifying color ramp name like <Black To White|Yellow To Red|Slope|more..>
                      or a color ramp object. 
                      For more information about colorramp object, see color ramp object at
                      https://developers.arcgis.com/documentation/common-data-types/color-ramp-objects.htm)
    :param hillshade_type: new at 10.8.1. int, 0 = traditional, 1 = multi - directional; default is 0
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "ShadedRelief",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if azimuth is not None:
        template_dict["rasterFunctionArguments"]["Azimuth"] = azimuth
    if altitude is not None:
        template_dict["rasterFunctionArguments"]["Altitude"] = altitude
    if z_factor is not None:
        template_dict["rasterFunctionArguments"]["ZFactor"] = z_factor
    if colormap is not None:
        template_dict["rasterFunctionArguments"]["Colormap"] = colormap
    if colorramp is not None:
        template_dict["rasterFunctionArguments"]['Colorramp'] = colorramp
    if slope_type is not None:
        template_dict["rasterFunctionArguments"]["SlopeType"] = slope_type
    if ps_power is not None:
        template_dict["rasterFunctionArguments"]["PSPower"] = ps_power
    if psz_factor is not None:
        template_dict["rasterFunctionArguments"]["PSZFactor"] = psz_factor
    if remove_edge_effect is not None:
        template_dict["rasterFunctionArguments"]["RemoveEdgeEffect"] = remove_edge_effect
    if hillshade_type is not None:
        template_dict["rasterFunctionArguments"]["HillshadeType"] = hillshade_type

    return _clone_layer(layer, template_dict, raster_ra)


def slope(dem, z_factor=None, slope_type=None, ps_power=None, psz_factor=None, remove_edge_effect=None,
          astype=None):
    """
    slope represents the rate of change of elevation for each pixel. For more information, see
    `slope function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/slope-function.htm>`__
    and `How slope works <http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/how-slope-works.htm>`__.
    
    The arguments for the slope function are as follows:

    :param dem: input DEM
    :param z_factor: double (e.g. 0.3)
    :param slope_type: new at 10.2. 1=DEGREE, 2=PERCENTRISE, 3=SCALED. default is 1.
    :param ps_power: new at 10.2. double, used together with SCALED slope type
    :param psz_factor: new at 10.2. double, used together with SCALED slope type
    :param remove_edge_effect: new at 10.2. boolean, true of false
    :param astype: output pixel type
    :return: the output raster

    """
    raster = dem

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Slope",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if z_factor is not None:
        template_dict["rasterFunctionArguments"]["ZFactor"] = z_factor
    if slope_type is not None:
        template_dict["rasterFunctionArguments"]["SlopeType"] = slope_type
    if ps_power is not None:
        template_dict["rasterFunctionArguments"]["PSPower"] = ps_power
    if psz_factor is not None:
        template_dict["rasterFunctionArguments"]["PSZFactor"] = psz_factor
    if remove_edge_effect is not None:
        template_dict["rasterFunctionArguments"]["RemoveEdgeEffect"] = remove_edge_effect
    # if dem is not None:
    #     template_dict["rasterFunctionArguments"]["DEM"] = raster

    return _clone_layer(layer, template_dict, raster_ra)


def focal_statistics(raster, kernel_columns=None, kernel_rows=None, stat_type=None, columns=None, rows=None,
               fill_no_data_only=None, astype=None):
    """
    The focal_statistics function calculates focal statistics for each pixel of an image based on a defined focal neighborhood.
    For more information, see `statistics function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/statistics-function.htm>`__.
    
    The arguments for the statistics function are as follows:

    :param raster: input raster
    :param kernel_columns: int (e.g. 3)
    :param kernel_rows: int (e.g. 3)
    :param stat_type: int or string.
                      There are four types of focal statistical functions:
                      
                      1=Min, 2=Max, 3=Mean, 4=StandardDeviation, 5=Median, 6=Majority, 7=Minority

                      - Min - Calculates the minimum value of the pixels within the neighborhood
                      - Max - Calculates the maximum value of the pixels within the neighborhood
                      - Mean - Calculates the average value of the pixels within the neighborhood. This is the default.
                      - StandardDeviation - Calculates the standard deviation value of the pixels within the neighborhood
                      - Median - Calculates the median value of pixels within the neighborhood.
                      - Majority - Calculates the majority value, or the value that occurs most frequently, of the pixels within the neighborhood.
                      - Minority - Calculates the minority value, or the value that occurs least frequently, of the pixels within the neighborhood.
    :param columns: int (e.g. 3). The number of pixel rows to use in your focal neighborhood dimension.
    :param rows: int (e.g. 3). The number of pixel columns to use in your focal neighborhood dimension.
    :param fill_no_data_only: bool
    :param astype: output pixel type
    :return: the output raster

    .. note::
        The focal_statistics() function is different from the focal_stats() function in the following aspects:

        The focal_statistics() function supports  Minimum, Maximum, Mean, and Standard Deviation, Median, Majority, Minority.  
        The focal_stats() function supports Mean, Majority, Maximum, Median, Minimum, Minority, Range, Standard deviation, Sum, and Variety.

        The focal_statistics() function supports only Rectangle.
        The focal_stats() function supports Rectangle, Circle, Annulus, Wedge, Irregular, and Weight neighborhoods.

        The option to determine if NoData pixels are to be processed out is available in the focal_statistics() function by setting a bool value for fill_no_data_only param.
        This option is not present in the focal_stats() function.

        The option to determine whether NoData values are ignored or not is available in the focal_stats() function by setting a bool value for ignore_no_data param.
        This option is not present in the focal_statistics() function.

    """

    layer, raster, raster_ra = _raster_input(raster)

    statistics_types = ["Min", "Max", "Mean", "StandardDeviation", "Median", "Majority", "Minority"]

    template_dict = {
        "rasterFunction": "Statistics",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if kernel_columns is not None:
        template_dict["rasterFunctionArguments"]["KernelColumns"] = kernel_columns
    if kernel_rows is not None:
        template_dict["rasterFunctionArguments"]["KernelRows"] = kernel_rows
    if stat_type is not None: 
        if isinstance(stat_type, str) and stat_type in statistics_types:
            template_dict["rasterFunctionArguments"]['Type'] = stat_type
        elif isinstance(stat_type, int):
            template_dict["rasterFunctionArguments"]['Type'] = stat_type
    if columns is not None:
        template_dict["rasterFunctionArguments"]["Columns"] = columns
    if rows is not None:
        template_dict["rasterFunctionArguments"]["Rows"] = rows
    if fill_no_data_only is not None:
        template_dict["rasterFunctionArguments"]["FillNoDataOnly"] = fill_no_data_only

    return _clone_layer(layer, template_dict, raster_ra)


def stretch(raster, stretch_type=0, min=None, max=None, num_stddev=None, statistics=None,
            dra=None, min_percent=None, max_percent=None, gamma=None, compute_gamma=None, sigmoid_strength_level=None,
            astype=None):
    """
    The stretch function enhances an image through multiple stretch types. For more information, see
    `stretch function <http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/stretch-function.htm>`__.

    Gamma stretch works with all stretch types. The Gamma parameter is needed when UseGamma is set to true. Min and Max
    can be used to define output minimum and maximum. DRA is used to get statistics from the extent in the export_image request.
    ComputeGamma will automatically calculate best gamma value to render exported image based on empirical model.

    Stretch type None does not require other parameters.
    Stretch type StdDev requires NumberOfStandardDeviations, Statistics, or DRA (true).
    Stretch type Histogram (Histogram Equalization) requires the source dataset to have histograms or additional DRA (true).
    Stretch type MinMax requires Statistics or DRA (true).
    Stretch type PercentClip requires MinPercent, MaxPercent, and DRA (true), or histograms from the source dataset.
    Stretch type Sigmoid does not require other parameters.

    Optionally, set the SigmoidStrengthLevel (1 to 6) to adjust the curvature of Sigmoid curve used in color stretch.


    The arguments for the stretch function are as follows:

    :param raster: input raster
    :param stretch_type: str, one of None, StdDev, Histogram, MinMax, PercentClip, 9 = Sigmoid
    :param min: double
    :param max: double
    :param num_stddev: double (e.g. 2.5)
    :param statistics: double (e.g. 2.5)[<min1>, <max1>, <mean1>, <standardDeviation1>], //[double, double, double, double][<min2>, <max2>, <mean2>, <standardDeviation2>]],
    :param dra: boolean. derive statistics from current request, Statistics parameter is ignored when DRA is true
    :param min_percent: double (e.g. 0.25), applicable to PercentClip
    :param max_percent: double (e.g. 0.5), applicable to PercentClip
    :param gamma: array of doubles
    :param compute_gamma: optional, applicable to any stretch type when "UseGamma" is "true"
    :param sigmoid_strength_level: int (1~6), applicable to Sigmoid
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    str_types = {
        'none': 0,
        'stddev': 3,
        'histogram' : 4,
        'minmax': 5,
        'percentclip' : 6,
        'sigmoid': 9
    }

    if isinstance(stretch_type, str):
        in_str_type = str_types[stretch_type.lower()]
    else:
        in_str_type = stretch_type

    template_dict = {
        "rasterFunction": "Stretch",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if stretch_type is not None:
        template_dict["rasterFunctionArguments"]["StretchType"] = in_str_type
    if min is not None:
        template_dict["rasterFunctionArguments"]["Min"] = min
    if max is not None:
        template_dict["rasterFunctionArguments"]["Max"] = max
    if num_stddev is not None:
        template_dict["rasterFunctionArguments"]["NumberOfStandardDeviations"] = num_stddev
    if statistics is not None:
        template_dict["rasterFunctionArguments"]["Statistics"] = statistics
    if dra is not None:
        template_dict["rasterFunctionArguments"]["DRA"] = dra
    if min_percent is not None:
        template_dict["rasterFunctionArguments"]["MinPercent"] = min_percent
    if max_percent is not None:
        template_dict["rasterFunctionArguments"]["MaxPercent"] = max_percent
    if gamma is not None:
        template_dict["rasterFunctionArguments"]["Gamma"] = gamma
    if compute_gamma is not None:
        template_dict["rasterFunctionArguments"]["ComputeGamma"] = compute_gamma
    if sigmoid_strength_level is not None:
        template_dict["rasterFunctionArguments"]["SigmoidStrengthLevel"] = sigmoid_strength_level

    if compute_gamma is not None or gamma is not None:
        template_dict["rasterFunctionArguments"]["UseGamma"] = True

    return _clone_layer(layer, template_dict, raster_ra)


def threshold(raster, astype=None):
    """
    The binary threshold function produces the binary image. It uses the Otsu method and assumes the input image to have
    a bi-modal histogram. 
    
    The arguments for the threshold function are as follows:

    :param raster: input raster
    :param astype: output pixel type
    :return: the output raster

    """
    threshold_type = 1
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Threshold",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if threshold_type is not None:
        template_dict["rasterFunctionArguments"]["ThresholdType"] = threshold_type

    return _clone_layer(layer, template_dict, raster_ra)


def transpose_bits(raster, input_bit_positions=None, output_bit_positions=None, constant_fill_check=None,
                   constant_fill_value=None, fill_raster=None, astype=None):
    """
    The transpose_bits function performs a bit operation. It extracts bit values from the source data and assigns them
    to new bits in the output data.
    
    The arguments for the transpose_bits function are as follows:

    If constant_fill_check is False, it assumes there is an input fill_raster. If an input fill_raster is not given,
    it falls back constant_fill_check to True and looks for constant_fill_value.
    Filling is used to initialize pixel values of the output raster.
    Landsat 8 has a quality assessment band. The following are the example input and output bit positions to extract
    confidence levels by mapping them to 0-3:

    * Landsat 8 Water: {"input_bit_positions":[4,5],"output_bit_positions":[0,1]}
    * Landsat 8 Cloud Shadow: {"input_bit_positions":[6,7],"output_bit_positions":[0,1]}
    * Landsat 8 Vegetation: {"input_bit_positions":[8,9],"output_bit_positions":[0,1]}
    * Landsat 8 Snow/Ice: {"input_bit_positions":[10,11],"output_bit_positions":[0,1]}
    * Landsat 8 Cirrus: {"input_bit_positions":[12,13],"output_bit_positions":[0,1]}
    * Landsat 8 Cloud: {"input_bit_positions":[14,15],"output_bit_positions":[0,1]}
    * Landsat 8 Designated Fill: {"input_bit_positions":[0],"output_bit_positions":[0]}
    * Landsat 8 Dropped Frame: {"input_bit_positions":[1],"output_bit_positions":[0]}
    * Landsat 8 Terrain Occlusion: {"input_bit_positions":[2],"output_bit_positions":[0]}

    :param raster: input raster
    :param input_bit_positions: array of long, required
    :param output_bit_positions: array of long, required
    :param constant_fill_check: bool, optional
    :param constant_fill_value: int, required
    :param fill_raster: optional, the fill raster
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "TransposeBits",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if input_bit_positions is not None:
        template_dict["rasterFunctionArguments"]["InputBitPositions"] = input_bit_positions
    if output_bit_positions is not None:
        template_dict["rasterFunctionArguments"]["OutputBitPositions"] = output_bit_positions
    if constant_fill_check is not None:
        template_dict["rasterFunctionArguments"]["ConstantFillCheck"] = constant_fill_check
    if constant_fill_value is not None:
        template_dict["rasterFunctionArguments"]["ConstantFillValue"] = constant_fill_value
    if fill_raster is not None:
        template_dict["rasterFunctionArguments"]["FillRaster"] = fill_raster

    return _clone_layer(layer, template_dict, raster_ra)


def unit_conversion(raster, from_unit=None, to_unit=None, astype=None):
    """
    The unit_conversion function performs unit conversions. 
    
    The arguments for the unit_conversion function are as follows:
    
    from_unit and to_unit take the following str values:
    Speed Units: MetersPerSecond, KilometersPerHour, Knots, FeetPerSecond, MilesPerHour
    Temperature Units: Celsius,Fahrenheit,Kelvin
    Distance Units: str, one of Inches, Feet, Yards, Miles, NauticalMiles, Millimeters, Centimeters, Meters

    :param raster: input raster
    :param from_unit: units constant listed below (int)
    :param to_unit: units constant listed below (int)
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    unit_types = {
        'inches': 1,
        'feet': 3,
        'yards': 4,
        'miles': 5,
        'nauticalmiles': 6,
        'millimeters': 7,
        'centimeters': 8,
        'meters': 9,
        'celsius': 200,
        'fahrenheit': 201,
        'kelvin': 202,
        'meterspersecond': 100,
        'kilometersperhour': 101,
        'knots': 102,
        'feetpersecond': 103,
        'milesperhour': 104
    }

    if isinstance(from_unit, str):
        from_unit = unit_types[from_unit.lower()]

    if isinstance(to_unit, str):
        to_unit = unit_types[to_unit.lower()]


    template_dict = {
        "rasterFunction": "UnitConversion",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if from_unit is not None:
        template_dict["rasterFunctionArguments"]["FromUnit"] = from_unit
    if to_unit is not None:
        template_dict["rasterFunctionArguments"]["ToUnit"] = to_unit

    return _clone_layer(layer, template_dict, raster_ra)


def vector_field_renderer(raster, is_uv_components=None, reference_system=None, mass_flow_angle_representation=None,
                          calculation_method="Vector Average", symbology_name="Single Arrow", astype=None):
    """
    The vector_field_renderer function symbolizes a U-V or Magnitude-Direction raster.
    
    The arguments for the vector_field_renderer function are as follows:

    :param raster: input raster
    :param is_uv_components: bool
    :param reference_system: int 1=Arithmetic, 2=Angular
    :param mass_flow_angle_representation: int 0=from 1=to
    :param calculation_method: string, "Vector Average" |
    :param symbology_name: string, "Single Arrow" |
    :param astype: output pixel type
    :return: the output raster

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "VectorFieldRenderer",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    if is_uv_components is not None:
        template_dict["rasterFunctionArguments"]["IsUVComponents"] = is_uv_components
    if reference_system is not None:
        template_dict["rasterFunctionArguments"]["ReferenceSystem"] = reference_system
    if mass_flow_angle_representation is not None:
        template_dict["rasterFunctionArguments"]["MassFlowAngleRepresentation"] = mass_flow_angle_representation
    if calculation_method is not None:
        template_dict["rasterFunctionArguments"]["CalculationMethod"] = calculation_method
    if symbology_name is not None:
        template_dict["rasterFunctionArguments"]["SymbologyName"] = symbology_name

    return _clone_layer(layer, template_dict, raster_ra)


def apply(raster, fn_name, **kwargs):
    """
    Applies a server side raster function template defined by the imagery layer (image service)
    The name of the raster function template is available in the imagery layer properties.rasterFunctionInfos.

    Function arguments are optional; argument names and default values are created by the author of the raster function
    template and are not known through the API. A client can simply provide the name of the raster function template
    only or, optionally, provide arguments to overwrite the default values.
    For more information about authoring server-side raster function templates, see
    `Server-side raster functions <http://server.arcgis.com/en/server/latest/publish-services/windows/server-side-raster-functions.htm>`__.

    :param raster: the input raster, or imagery layer
    :param fn_name: name of the server side raster function template, See imagery layer properties.rasterFunctionInfos
    :param kwargs: keyword arguments to override the default values of the raster function template, including astype
    :return: the output raster
    """
    
    variable_name = kwargs.pop("variable_name", None)
    raster_layer = raster
    if variable_name is not None:
        layer, raster, raster_ra = _raster_input(kwargs.pop(variable_name))
    else:
        layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": fn_name,
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    for key, value in kwargs.items():
        template_dict["rasterFunctionArguments"][key] = value

    astype = kwargs.pop('astype', None)
    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()
        template_dict["rasterFunctionArguments"].pop('astype', None)

    if variable_name is not None:
        template_dict["variableName"] = variable_name        
        template_dict["rasterFunctionArguments"][variable_name] = raster        
        template_dict["rasterFunctionArguments"].pop('variable_name', None)
        template_dict["rasterFunctionArguments"].pop('Raster', None)

    function_chain_ra = {
        "rasterFunction" : "Identity",
        "rasterFunctionArguments": {
            "Raster" : {"renderingRule":copy.deepcopy(template_dict),
                         "url":layer._lyr_json['url']},
        }
    }

    if raster_layer._mosaic_rule is not None:
        function_chain_ra["rasterFunctionArguments"]["Raster"]["mosaicRule"] = raster_layer._mosaic_rule
    return _clone_layer_without_copy(layer, template_dict, function_chain_ra)


def vector_field(raster_u_mag, raster_v_dir, input_data_type='Vector-UV', angle_reference_system='Geographic',
                 output_data_type='Vector-UV', astype=None):
    """
    The VectorField function is used to composite two single-band rasters (each raster represents U/V or Magnitude/Direction)
    into a two-band raster (each band represents U/V or Magnitude/Direction). Data combination type (U-V or Magnitude-Direction)
    can also be converted interchangeably with this function.
    For more information, see Vector Field function
    (http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/vector-field-function.htm)

    :param raster_u_mag: raster item representing 'U' or 'Magnitude' - imagery layers filtered by where clause, spatial and temporal filters
    :param raster_v_dir: raster item representing 'V' or 'Direction' - imagery layers filtered by where clause, spatial and temporal filters
    :param input_data_type: string, 'Vector-UV' or 'Vector-MagDir' per input used in 'raster_u_mag' and 'raster_v_dir'
    :param angle_reference_system: string, optional when 'input_data_type' is 'Vector-UV', one of "Geographic", "Arithmetic"
    :param output_data_type: string, 'Vector-UV' or 'Vector-MagDir'
    :return: the output raster with this function applied to it
    """

    layer1, raster_u_mag_1, raster_ra1 = _raster_input(raster_u_mag)
    layer2, raster_v_dir_1, raster_ra2 = _raster_input(raster_u_mag, raster_v_dir)

    if layer1 is not None and layer2._datastore_raster is False:
        layer = layer1
    else:
        layer = layer2
    #layer = layer1 if layer1 is not None else layer2

    angle_reference_system_types = {
        "Geographic" : 0,
        "Arithmetic" : 1
    }

    in_angle_reference_system = angle_reference_system_types[angle_reference_system]

    template_dict = {
        "rasterFunction": "VectorField",
        "rasterFunctionArguments": {
            "Raster1": raster_u_mag_1,
            "Raster2": raster_v_dir_1,            
        }
    }

    if in_angle_reference_system is not None:
        template_dict["rasterFunctionArguments"]["AngleReferenceSystem"] = in_angle_reference_system
    if input_data_type is not None and input_data_type in ["Vector-UV", "Vector-MagDir"]:
        template_dict["rasterFunctionArguments"]['InputDataType'] = input_data_type
    if output_data_type is not None and output_data_type in ["Vector-UV", "Vector-MagDir"]:
        template_dict["rasterFunctionArguments"]['OutputDataType'] = output_data_type
    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra1, raster_ra2)


def complex(raster):
    """
    Complex function computes magnitude from complex values. It is used when
    input raster has complex pixel type. It computes magnitude from complex
    value to convert the pixel type to floating point for each pixel. It takes
    no argument but an optional input raster. For more information, see 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/complex-function.htm

    :param raster: the input raster / imagery layer
    :return: Output raster obtained after applying the function
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Complex",
        "rasterFunctionArguments" : {
            "Raster" : raster,
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)


def colormap_to_rgb(raster):
    """
    The function is designed to work with single band image service that has
    internal colormap. It will convert the image into a three-band 8-bit RGB
    raster. This function takes no arguments except an input raster. For 
    qualified image service, there are two situations when ColormapToRGB 
    function is automatically applied: The "colormapToRGB" property of the 
    image service is set to true; or, client asks to export image into jpg 
    or png format. For more information, see 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/colormap-to-rgb-function.htm)

    :param raster: the input raster / imagery layer
    :return: Three band raster
    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "ColormapToRGB",
        "rasterFunctionArguments" : {
            "Raster" : raster,
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)


def statistics_histogram(raster, statistics=None, histograms=None):
    """"
    The function is used to define the statistics and histogram of a raster.
    It is normally used for control the default display of exported image. 
    For more information, see Statistics and Histogram function, 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/statistics-and-histogram-function.htm

    :param raster: the input raster / imagery layer
    :param statistics: array of statistics objects. (Predefined statistics for each band)
    :param histograms: array of histogram objects. (Predefined histograms for each band)
    :return: Statistics and Histogram defined raster
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "StatisticsHistogram",
        "rasterFunctionArguments" : {            
            "Raster" : raster,
        }
    }

    if statistics is not None:
        template_dict["rasterFunctionArguments"]['Statistics'] = statistics
    if histograms is not None:
        template_dict["rasterFunctionArguments"]['Histograms'] = histograms

    return _clone_layer(layer, template_dict, raster_ra)


def tasseled_cap(raster):
    """
    The function is designed to analyze and map vegetation and urban development
    changes detected by various satellite sensor systems. It is known as the 
    Tasseled Cap transformation due to the shape of the graphical distribution
    of data. This function takes no arguments except a raster. The input for 
    this function is the source raster of image service. There are no other 
    parameters for this function because all the information is derived from 
    the input's properties and key metadata (bands, data type, and sensor name). 
    Only imagery from the Landsat MSS, Landsat TM, Landsat ETM+, IKONOS, 
    QuickBird, WorldView-2 and RapidEye sensors are supported. Prior to applying
    this function, there should not be any functions that would alter the pixel
    values in the function chain, such as the Stretch, Apparent Reflectance or
    Pansharpening function. The only exception is for Landsat ETM+; when using 
    Landsat ETM+, the Apparent Reflectance function must precede the Tasseled 
    Cap function. For more information, see 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/tasseled-cap-transformation.htm

    :param raster: the input raster / imagery layer
    :return: the output raster with TasseledCap function applied to it
    """
 
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "TasseledCap",
        "rasterFunctionArguments" : {
            "Raster" : raster,
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)


def identity(raster):
    """"
    The function is used to define the source raster as part of the default
    mosaicking behavior of the mosaic dataset. This function is a no-op function
    and takes no arguments except a raster. For more information, see
    (http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/identity-function.htm)

    :param raster: the input raster / imagery layer
    :return: the input raster
    """
 
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Identity",
        "rasterFunctionArguments": {
            "Raster" : raster,
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)

def colorspace_conversion(raster, conversion_type="rgb_to_hsv"):
    """
    The ColorspaceConversion function converts the color model of a three-band
    unsigned 8-bit image from either the hue, saturation, and value (HSV)
    to red, green, and blue (RGB) or vice versa. An ExtractBand function and/or
    a Stretch function are sometimes used for converting the imagery into valid
    input of ColorspaceConversion function. For more information, see
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/color-model-conversion-function.htm

    :param raster: the input raster
    :param conversion_type: sting type, one of "rgb_to_hsv" or "hsv_to_rgb". Default is "rgb_to_hsv"
    :return: the output raster with this function applied to it
    """
 
    layer, raster, raster_ra = _raster_input(raster)

    conversion_types = {
        "rgb_to_hsv" : 0,
        "hsv_to_rgb" : 1
        }
            
    template_dict = {
        "rasterFunction" : "ColorspaceConversion",
        "rasterFunctionArguments" : {
            "Raster" : raster,            
        }
    }
    
    template_dict["rasterFunctionArguments"]['ConversionType'] = conversion_types[conversion_type]
         
    return _clone_layer(layer, template_dict, raster_ra)


def grayscale(raster, conversion_parameters=None):
    """
    The Grayscale function converts a multi-band image into a single-band grayscale
    image. Specified weights are applied to each of the input bands, and a 
    normalization is applied for output. For more information, see
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/grayscale-function.htm

    :param raster: the input raster
    :param conversion_parameters: array of double (A length of N array representing weights for each band, where N=band count.)
    :return: the output raster with this function applied to it
    """
 
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Grayscale",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }
    
    if conversion_parameters is not None and isinstance(conversion_parameters, list):
        template_dict["rasterFunctionArguments"]['ConversionParameters'] = conversion_parameters

    return _clone_layer(layer, template_dict, raster_ra)


def spectral_conversion(raster, conversion_matrix):
    """
    The SpectralConversion function applies a matrix to a multi-band image to
    affect the spectral values of the output. In the matrix, different weights
    can be assigned to all the input bands to calculate each of the output 
    bands. The column/row size of the matrix equals to the band count of input 
    raster. For more information, see Spectral Conversion function
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/spectral-conversion-function.htm

    :param raster: the input raster
    :param conversion_parameters: array of double (A NxN length one-dimension matrix, where N=band count.)
    :return: the output raster with this function applied to it
    """
 
    layer, raster, raster_ra = _raster_input(raster)
       
    template_dict = {
        "rasterFunction" : "SpectralConversion",
        "rasterFunctionArguments": {
            "Raster" : raster,  
            "ConversionMatrix" : conversion_matrix
        }
    }    
    
    return _clone_layer(layer, template_dict, raster_ra)


def raster_calculator(rasters, input_names, expression, extent_type="FirstOf", cellsize_type="FirstOf", astype=None):
    """
    The RasterCalculator function provides access to all existing math functions
    so you can make calls to them when building your expressions. The calculator
    function requires single-band inputs. If you need to perform expressions on
    bands in a multispectral image as part of a function chain, you can use 
    the Extract Bands Function before the RasterCalculator function. 
    For more info including operators supported, see Calculator function 
    http://pro.arcgis.com/en/pro-app/help/data/imagery/calculator-function.htm

    :param raster: array of rasters
    :param input_names: array of strings for arbitrary raster names.
    :param expression: string, expression to calculate output raster from input rasters
    :param extent_type: string, one of "FirstOf", "IntersectionOf" "UnionOf", "LastOf". Default is "FirstOf".
    :param cellsize_type: one of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf". Default is "FirstOf".
    :param astype: output pixel type
    :return: output raster with function applied
    """
    
    layer, raster, raster_ra = _raster_input(rasters)
    
    extent_types = {
        "FirstOf" : 0,
        "IntersectionOf" : 1,
        "UnionOf" : 2,
        "LastOf" : 3
    }

    cellsize_types = {
        "FirstOf" : 0,
        "MinOf" : 1,
        "MaxOf" : 2,
        "MeanOf" : 3,
        "LastOf" : 4
    }      

    template_dict = {
        "rasterFunction" : "RasterCalculator",
        "rasterFunctionArguments": {
            "InputNames" : input_names,
            "Expression" : expression,
            "Rasters" : raster            
        },
        "variableName" : "Rasters"
    }
    
    template_dict["rasterFunctionArguments"]['ExtentType'] = extent_types[extent_type]    
    template_dict["rasterFunctionArguments"]['CellsizeType'] = cellsize_types[cellsize_type]

    if astype is not None:
        template_dict["outputPixelType"] = astype.upper()

    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')


def speckle(raster, 
            filter_type="Lee", 
            filter_size="3x3", 
            noise_model="Multiplicative", 
            noise_var=None,
            additive_noise_mean=None, 
            multiplicative_noise_mean=1,
            nlooks=1, 
            damp_factor=None):
    """
    The Speckle function filters the speckled radar dataset to smooth out the 
    noise while retaining the edges or sharp features in the image. Four speckle
    reduction filtering algorithms are provided through this function. For more
    information including required and optional parameters for each filter and 
    the default parameter values, see Speckle function 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm

    :param raster: input raster type
    :param filter_type: string, one of "Lee", "EnhancedLee" "Frost", "Kaun". Default is "Lee".
    :param filter_size: string, kernel size. One of "3x3", "5x5", "7x7", "9x9", "11x11". Default is "3x3".
    :param noise_model: string, For Lee filter only. One of "Multiplicative", "Additive", "AdditiveAndMultiplicative"
    :param noise_var: double, for Lee filter with noise_model "Additive" or "AdditiveAndMultiplicative"
    :param additive_noise_mean: string, for Lee filter witth noise_model "AdditiveAndMultiplicative" only
    :param multiplicative_noise_mean: double, For Lee filter with noise_model "Additive" or "AdditiveAndMultiplicative"
    :param nlooks: int, for Lee, EnhancedLee and Kuan Filters
    :param damp_factor: double, for EnhancedLee and Frost filters
    :return: output raster with function applied
    """
   
    layer, raster, raster_ra = _raster_input(raster)
   
    filter_types = {
        "Lee" : 0,
        "EnhancedLee" : 1,
        "Frost" : 2,
        "Kuan" : 3
    }

    filter_sizes = {
        "3x3" : 0,
        "5x5" : 1,
        "7x7" : 2,
        "9x9" : 3,
        "11x11" : 4
    }

    noise_models = {
        "Multiplicative" : 0,
        "Additive" : 1,
        "AdditiveAndMultiplicative" : 2
    }    
    
    template_dict = {
        "rasterFunction" : "Speckle",
        "rasterFunctionArguments" : {            
            "Raster": raster,            
        }
    }
        
    template_dict["rasterFunctionArguments"]['FilterType'] = filter_types[filter_type]    
    template_dict["rasterFunctionArguments"]['FilterSize'] = filter_sizes[filter_size]    
    template_dict["rasterFunctionArguments"]['NoiseModel'] = noise_models[noise_model]

    if noise_var is not None:
        template_dict["rasterFunctionArguments"]['NoiseVar'] = noise_var
    if additive_noise_mean is not None:
        template_dict["rasterFunctionArguments"]['AdditiveNoiseMean'] = additive_noise_mean
    if multiplicative_noise_mean is not None:
        template_dict["rasterFunctionArguments"]['MultiplicativeNoiseMean'] = multiplicative_noise_mean
    if nlooks is not None:
        template_dict["rasterFunctionArguments"]['NLooks'] = nlooks
    if damp_factor is not None:
        template_dict["rasterFunctionArguments"]['DampFactor'] = damp_factor
    
    return _clone_layer(layer, template_dict, raster_ra)


def pansharpen(pan_raster,
               ms_raster,
               ir_raster=None,
               fourth_band_of_ms_is_ir = True,
               weights = [0.166, 0.167, 0.167, 0.5],               
               type="ESRI",                                
               sensor=None):
    """
    The Pansharpening function uses a higher-resolution panchromatic raster to
    fuse with a lower-resolution, multiband raster. It can generate colorized 
    multispectral image with higher resolution. For more information, see 
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/pansharpening-function.htm

    :param pan_raster: raster, which is panchromatic
    :param ms_raster: raster, which is multispectral
    :param ir_raster: Optional, if fourth_band_of_ms_is_ir is true or selected pansharpening method doesn't require near-infrared image
    :param fourth_band_of_ms_is_ir: Boolean, "true" if "ms_raster" has near-infrared image on fourth band
    :param weights: Weights applied for Red, Green, Blue, Near-Infrared bands. 4-elements array, Sum of values is 1
    :param type: string, describes the Pansharpening method one of "IHS", "Brovey" "ESRI", "SimpleMean", "Gram-Schmidt". Default is "ESRI"
    :param sensor: string, it is an optional parameter to specify the sensor name
    :return: output raster with function applied
    """

    layer1, pan_raster_1, raster_ra1 = _raster_input(pan_raster)
    layer2, ms_raster_1, raster_ra2 = _raster_input(pan_raster, ms_raster)
    
    layer3=None
    if ir_raster is not None:
        layer3, ir_raster_1, raster_ra3 = _raster_input(pan_raster, ir_raster)

    layer = None
    if layer1._datastore_raster is True:
        layer = layer1
    elif layer2._datastore_raster is True:
        layer= layer2
    elif (layer3 is not None and layer3._datastore_raster is True):
        layer = layer3
    
    if layer is not None:
        pan_raster_1 = raster_ra1
        ms_raster_1 = raster_ra2
        if ir_raster is not None:
            ir_raster_1 = raster_ra3
    else:
        if layer1 is not None:
            layer = layer1
        elif layer2 is not None:
            layer = layer2
        elif layer3 is not None:
            layer = layer3

    pansharpening_types = {
        "IHS" : 0,
        "Brovey" : 1,
        "ESRI" : 2,
        "SimpleMean" : 3,
        "Gram-Schmidt" : 4
    }

    template_dict = {
        "rasterFunction" : "Pansharpening",
        "rasterFunctionArguments" : {      
            "Weights" : weights,            
            "PanImage": pan_raster_1,
            "MSImage" : ms_raster_1
        }
    }

    if type is not None:
        template_dict["rasterFunctionArguments"]['PansharpeningType'] = pansharpening_types[type]

    if ir_raster is not None:
        template_dict["rasterFunctionArguments"]['InfraredImage'] = ir_raster_1

    if isinstance(fourth_band_of_ms_is_ir, bool):
        template_dict["rasterFunctionArguments"]['UseFourthBandOfMSAsIR'] = fourth_band_of_ms_is_ir

    if sensor is not None:
        template_dict["rasterFunctionArguments"]['Sensor'] = sensor

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']['PanImage'] = raster_ra1
    function_chain_ra['rasterFunctionArguments']['MSImage'] = raster_ra2
    if ir_raster is not None:
        function_chain_ra['rasterFunctionArguments']['InfraredImage'] = raster_ra3

    return _clone_layer_without_copy(layer, template_dict, function_chain_ra)


def weighted_overlay(rasters, fields, influences, remaps, eval_from, eval_to):
               
    """
    The WeightedOverlay function allows you to overlay several rasters using a common measurement scale and weights each according to its importance. For more information, see
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/weighted-overlay-function.htm

    :param raster: array of rasters
    :param fields: array of string fields of the input rasters to be used for weighting.				 
    :param influences: array of double, Each input raster is weighted according to its importance, or 
				       its influence. The sum of the influence weights must equal 1
    :param remaps: array of strings, Each value in an input raster is assigned a new value based on the remap. The remap value can be a valid value or a NoData value.    
    :param eval_from: required, numeric value of evaluation scale from
    :param eval_to: required, numeric value of evaluation scale to
    :return: output raster with function applied
    """

    layer, raster, raster_ra = _raster_input(rasters)   

    template_dict = {
        "rasterFunction" : "WeightedOverlay",
        "rasterFunctionArguments" : { 
            "Rasters" : raster,
            "Fields" : fields,
            "Influences": influences,
            "Remaps" : remaps,
            "EvalFrom" : eval_from,
            "EvalTo": eval_to
        },
        "variableName": "Rasters"
    }   
    
    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')


def weighted_sum(rasters, fields, weights):
               
    """
    The WeightedSum function allows you to overlay several rasters, multiplying each by their given weight and summing them together.  For more information, see
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/weighted-sum-function.htm

    :param raster: array of rasters
    :param fields: array of string fields of the input rasters to be used for weighting.				 
    :param weights: array of double, The weight value by which to multiply the raster. 
				    It can be any positive or negative decimal value.    
    :return: output raster with function applied
    """

    layer, raster, raster_ra = _raster_input(rasters)   
     
    template_dict = {
        "rasterFunction" : "WeightedSum",
        "rasterFunctionArguments" : { 
            "Rasters" : raster,
            "Fields" : fields,
            "Weights" : weights
        },
        "variableName": "Rasters"
    }   
    
    return _clone_layer(layer, template_dict, raster_ra, variable_name='Rasters')


def focal_stats(raster, neighborhood_type=1 , width=3, height=3, 
                inner_radius=1 , outer_radius=3, radius=3, start_angle=0, end_angle=90, neighborhood_values=None,
                stat_type=3, percentile_value=90, ignore_no_data=True):
    """
    Calculates for each input cell location a statistic of the values within a specified neighborhood around it.
    For more information see, https://pro.arcgis.com/en/pro-app/help/data/imagery/focal-statistics-function.htm

    :param raster: input raster
    :param neighborhood_type: int, default is 1. The shape of the area around each cell used to calculate the statistic.
                               1 = Rectangle
                               2 = Circle
                               3 = Annulus
                               4 = Wedge
                               5 = Irregular
                               6 = Weight
                               
    :param width: int, default is 3. Specified when neighborhood_type is Rectangle
    :param height: int, default is 3. Specified when neighborhood_type is Rectangle
    :param inner_radius: int, default is 1. Specified when neighborhood_type is Annulus
    :param outer_radius: int, default is 3. Specified when neighborhood_type is Annulus
    :param radius: int, default is 3. Specified when neighborhood_type is Circle or Wedge
    :param start_angle: float, default is 0. Specified when neighborhood_type is Wedge
    :param end_angle: float, default is 90. Specified when neighborhood_type is Wedge
    :param neighborhood_values: Specified when neighborhood_type is Irregular or Weight.
                                It can be a list of lists, in which the width and height will be automatically set from the columns and rows 
                                of the two dimensional list, respectively. 
                                Alternatively, it can be a one dimensional list obtained from flattening a two dimensional list. In this case, 
                                the dimensions need to be specified explicitly with the width and height parameters.
    :param stat_type: int, default is 3(Mean)

                      There are 11 types of statistics available:
                      1=Majority, 2=Maximum, 3=Mean , 4=Median, 5= Minimum, 6 = Minority,
                      7=Range, 8=Standard deviation, 9=Sum, 10=Variety, 12=Percentile

                          - Majority = Calculates the majority (value that occurs most often) of the cells in the neighborhood.

                          - Maximum = Calculates the maximum (largest value) of the cells in the neighborhood.

                          - Mean = Calculates the mean (average value) of the cells in the neighborhood.

                          - Median = Calculates the median of the cells in the neighborhood.

                          - Minimum = Calculates the minimum (smallest value) of the cells in the neighborhood.

                          - Minority = Calculates the minority (value that occurs least often) of the cells in the neighborhood.

                          - Range = Calculates the range (difference between largest and smallest value) of the cells in the neighborhood.

                          - Standard deviation =  Calculates the standard deviation of the cells in the neighborhood.

                          - Sum = Calculates the sum (total of all values) of the cells in the neighborhood.

                          - Variety = Calculates the variety (the number of unique values) of the cells in the neighborhood.

                          - Percentile = Calculates a specified percentile of the cells in the neighborhood.

    :param ignore_no_data: boolean, default is True.

                        - True - Specifies that if a NoData value exists within a neighborhood, \
                        the NoData value will be ignored. Only cells within the neighborhood \
                        that have data values will be used in determining the output value. \
                        This is the default.

                        - False - Specifies that if any cell in a neighborhood has a value of \
                        NoData, the output for the processing cell will be NoData.
    :param percentile_value: float, default is 90. Denotes which percentile to calculate when the stat_type is Percentile.   
                             The value can range from 0 to 100.

    :return: the output raster

    .. note::
        The focal_stats() function is different from the focal_statistics() function in the following aspects:

        The focal_stats() function supports Mean, Majority, Maximum, Median, Minimum, Minority, Percentile, Range, Standard deviation, Sum, and Variety.
        The focal_statistics() function supports only Minimum, Maximum, Mean, and Standard Deviation.

        The focal_stats() function supports Rectangle, Circle, Annulus, Wedge, Irregular, and Weight neighborhoods. The focal_statistics() function supports only Rectangle.

        The option to determine whether NoData values are ignored or not is available in the focal_stats() function by setting a bool value for ignore_no_data param.
        This option is not present in the focal_statistics() function.

        The option to determine if NoData pixels are to be processed out is available in the focal_statistics() function by setting a bool value for fill_no_data_only param.
        This option is not present in the focal_stats() function.

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Focal",
        "rasterFunctionArguments": {
            "Raster": raster
        },
        "variableName": "Raster"
    }

    if stat_type is not None:
        template_dict["rasterFunctionArguments"]["StatisticType"] = stat_type
    if percentile_value is not None:
        template_dict["rasterFunctionArguments"]["Percentile"] = percentile_value
    if neighborhood_type is not None:
        template_dict["rasterFunctionArguments"]["NeighborhoodType"] = neighborhood_type
    if width is not None:
        template_dict["rasterFunctionArguments"]["Width"] = width
    if height is not None:
        template_dict["rasterFunctionArguments"]["Height"] = height
    if inner_radius is not None:
        template_dict["rasterFunctionArguments"]["InnerRadius"] = inner_radius
    if outer_radius is not None:
        template_dict["rasterFunctionArguments"]["OuterRadius"] = outer_radius
    if radius is not None:
        template_dict["rasterFunctionArguments"]["Radius"] = radius
    if start_angle is not None:
        template_dict["rasterFunctionArguments"]["StartAngle"] = start_angle
    if end_angle is not None:
        template_dict["rasterFunctionArguments"]["EndAngle"] = end_angle
    if neighborhood_values is not None:
        flattened = [item for sublist in neighborhood_values if isinstance(sublist, list) for item in sublist]
        if flattened == []:
            flattened = neighborhood_values
        else:
            template_dict["rasterFunctionArguments"]["Height"] = len(neighborhood_values)
            if isinstance(neighborhood_values[0], list):
                template_dict["rasterFunctionArguments"]["Width"] = len(neighborhood_values[0])
        template_dict["rasterFunctionArguments"]["NeighborhoodValues"] = flattened
    if ignore_no_data is not None:
        template_dict["rasterFunctionArguments"]["NoDataPolicy"] = ignore_no_data

    return _clone_layer(layer, template_dict, raster_ra)


def lookup(raster, field=None):
    """
    Creates a new raster by looking up values found in another field in the table of the input raster. 
    For more information see, https://pro.arcgis.com/en/pro-app/help/data/imagery/lookup-function.htm

    :param raster: The input raster that contains a field from which to create a new raster.
    :param field: Field containing the desired values for the new raster.

    :return: the output raster with this function applied to it
    """

    layer, raster, raster_ra = _raster_input(raster)
       
    template_dict = {
        "rasterFunction" : "Lookup",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }
    
    if field is not None:
        template_dict["rasterFunctionArguments"]['Field'] = field

    return _clone_layer(layer, template_dict, raster_ra)


def raster_collection_function(raster, item_function=None, 
                               aggregation_function=None, 
                               processing_function=None, 
                               aggregation_definition_type="ALL",
                               dimension=None,
                               interval_keyword=None,
                               interval_value=None,
                               interval_unit=None,
                               interval_ranges=None,
                               ):
    """
    Creates a new raster by applying item, aggregation and processing function

    :param raster: Input Imagery Layer. The image service the layer is based on should be a mosaic dataset
    :param item_function: The raster function template to be applied on each item of the mosaic dataset. 
                          Create an RFT object out of the raster function template item on the portal and 
                          specify that as the input to item_function 
    :param aggregation_function: The aggregation function to be applied on the mosaic dataset.
                                 Create an RFT object out of the raster function template item on the portal and 
                                 specify that as the input to aggregation_function 
    :param processing_function: The processing template to be applied on the imagery layer.
                                Create an RFT object out of the raster function template item on the portal and 
                                specify that as the input to processing_function.
    :param aggregation_definition_type: Optional String. Specifies the dimension interval for which the data
                                        will be aggregated.

                                        - ALL : The data values will be aggregated across all slices. This is the default.

                                        - INTERVAL_KEYWORD : The variable data will be aggregated using a commonly known interval.

                                        - INTERVAL_VALUE : The variable data will be aggregated using a user-specified interval and unit.

                                        - INTERVAL_RANGES : The variable data will be aggregated between specified pairs of values or dates.
    :param dimension: Optional String. This is the dimension along which the variables will be aggregated.
    :param interval_keyword: Optional String. Specifies the keyword interval that will be used
                                             when aggregating along the dimension. This parameter is required
                                             when the aggregation_def parameter is set to INTERVAL_KEYWORD, and
                                             the aggregation must be across time.

                                             - HOURLY : The data values will be aggregated into hourly time steps, 
                                               and the result will include every hour in the time series.

                                             - DAILY : The data values will be aggregated into daily time steps, 
                                               and the result will include every day in the time series.

                                             - WEEKLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include every week in the time series.

                                             - DEKADLY : Divides each month into 3 periods of 10 days each 
                                               (last period might have more or less than 10 days)
                                               and each month would output 3 slices.

                                             - PENTADLY : Divides each month into 6 periods of 5 days each 
                                               (last period might have more or less than 5 days)
                                               and each month would output 6 slices.

                                             - MONTHLY : The data values will be aggregated into monthly time steps, 
                                               and the result will include every month in the time series.

                                             - QUARTERLY : The data values will be aggregated into quarterly time steps, 
                                               and the result will include every quarter in the time series.

                                             - YEARLY : The data values will be aggregated into yearly time steps, 
                                               and the result will include every year in the time series.

                                             - RECURRING_DAILY : The data values will be aggregated into daily time steps, 
                                               and the result includes each one aggregated value per day. 
                                               The output will include, at most, 366 daily time slices

                                             - RECURRING_WEEKLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per week. 
                                               The output will include, at most, 53 weekly time slices.

                                             - RECURRING_MONTHLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per month. 
                                               The output will include, at most, 12 monthly time slices.

                                             - RECURRING_QUARTERLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per quarter. 
                                               The output will include, at most, 4 quarterly time slices.

    :param interval_value: Optional String. The size of the interval that will be used for the
                           aggregation. This parameter is required when the aggregation_def
                           parameter is set to INTERVAL_VALUE.

                           For example, to aggregate 30 years of monthly temperature data into
                           5-year increments, enter 5 as the interval_value, and specify
                           interval_unit as YEARS.

    :param interval_unit: Optional String. The unit that will be used for the interval value.
                          This parameter is required when the dimension parameter is set to a
                          time field and the aggregation_def parameter is set to INTERVAL_VALUE.

                          If you are aggregating over anything other than time, this option
                          will not be available and the unit for the interval value will match
                          the variable unit of the input multidimensional raster data.

                          - HOURS : The data values will be aggregated into hourly time slices at the interval provided.
                          - DAYS : The data values will be aggregated into daily time slices at the interval provided.
                          - WEEKS : The data values will be aggregated into weekly time slices at the interval provided.
                          - MONTHS : The data values will be aggregated into monthly time slices at the interval provided.
                          - YEARS : The data values will be aggregated into yearly time slices at the interval provided.
    :param interval_ranges: Optional List of dictionary objects. Interval ranges specified as list of dictionary objects 
                            that will be used to aggregate groups of values. 

                            This parameter is required when the aggregation_definition parameter is set to INTERVAL_RANGE.
                            If dimension is StdTime, then the value must be specified in human readable time format (YYYY-MM-DDTHH:MM:SS).

                            Syntax: 
                                [{"minValue":"<min value>","maxValue":"<max value>"},
                                {"minValue":"<min value>","maxValue":"<max value>"}]

                            Example:
                                [{"minValue":"2012-01-15T03:00:00","maxValue":"2012-01-15T09:00:00"},
                                {"minValue":"2012-01-15T12:00:00","maxValue":"2012-01-15T21:00:00"}]
    

    :return: the output raster with function applied on it
    """

    layer, raster, raster_ra = _raster_input(raster)
       
    template_dict = {
        "rasterFunction": "RasterCollection",
        "rasterFunctionArguments": {
            "RasterCollection": raster
        },
        "variableName": "RasterCollection"
    }
    
    if item_function is not None:
        if isinstance(item_function, RFT):
            template_dict["rasterFunctionArguments"]['ItemFunction']=item_function._rft_json
        else:
            template_dict["rasterFunctionArguments"]['ItemFunction']=item_function
        if "type" not in template_dict["rasterFunctionArguments"]["ItemFunction"]:
            template_dict["rasterFunctionArguments"]["ItemFunction"].update({'type':'RasterFunctionTemplate'})

    if aggregation_function is not None:
        if isinstance(aggregation_function, RFT):
            template_dict["rasterFunctionArguments"]["AggregationFunction"]=aggregation_function._rft_json
        else:
            template_dict["rasterFunctionArguments"]["AggregationFunction"]=aggregation_function
        if "type" not in template_dict["rasterFunctionArguments"]["AggregationFunction"]:
            template_dict["rasterFunctionArguments"]["AggregationFunction"].update({'type':'RasterFunctionTemplate'})

    if processing_function is not None:
        if isinstance(processing_function, RFT):
            template_dict["rasterFunctionArguments"]["ProcessingFunction"]=processing_function._rft_json
        else:
            template_dict["rasterFunctionArguments"]["ProcessingFunction"]=processing_function
        if "type" not in template_dict["rasterFunctionArguments"]["ProcessingFunction"]:
            template_dict["rasterFunctionArguments"]["ProcessingFunction"].update({'type':'RasterFunctionTemplate'})

    aggregation_definition = {}
    aggregation_definition.update({"definitionType": aggregation_definition_type})

    if dimension is not None:
         aggregation_definition.update({"dimension":dimension})

    if aggregation_definition_type == "INTERVAL_VALUE":
        if interval_value is None:
            raise RuntimeError("interval_value cannot be None, if aggregation_definition_type is INTERVAL_VALUE")

        aggregation_definition.update({"intervalValue":interval_value})
        if interval_unit is not None:
            aggregation_definition.update({"intervalUnits":interval_unit})

    elif aggregation_definition_type == "INTERVAL_KEYWORD":
        if interval_keyword is None:
            raise RuntimeError("interval_keyword cannot be None, if aggregation_definition_type is INTERVAL_KEYWORD")

        interval_keyword_val=interval_keyword
        if interval_keyword is not None:
            interval_keyword_allowed_values = ['HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'YEARLY', 'RECURRING_DAILY', 'RECURRING_WEEKLY',
                                               'RECURRING_MONTHLY', 'RECURRING_QUARTERLY', 'PENTADLY', 'DEKADLY']
            if [element.lower() for element in interval_keyword_allowed_values].count(interval_keyword.lower()) <= 0 :
                raise RuntimeError('interval_keyword can only be one of the following: '+str(interval_keyword_allowed_values))
            interval_keyword_val=interval_keyword
            for element in interval_keyword_allowed_values:
                if interval_keyword.upper() == element:
                    interval_keyword_val = element
        aggregation_definition.update({"intervalKeyword":interval_keyword})

    elif aggregation_definition_type == "INTERVAL_RANGES":
        if interval_ranges is None:
            raise RuntimeError("interval_ranges cannot be None, if aggregation_definition_type is INTERVAL_RANGES")
            
        min_list=[]
        max_list=[]
        if isinstance(interval_ranges, list):
            for ele in interval_ranges:
                if isinstance(ele,dict):
                    min_list.append(ele["minValue"])
                    max_list.append(ele["maxValue"])

        if isinstance(interval_ranges, dict):
            min_list.append(interval_ranges["minValue"])
            max_list.append(interval_ranges["maxValue"])

        aggregation_definition.update({"minValues":min_list})
        aggregation_definition.update({"maxValues":max_list})

    if aggregation_definition is not None:
        template_dict["rasterFunctionArguments"]["AggregationDefinition"] = _json.dumps(aggregation_definition)

    #if where_clause is not None:
    #    template_dict["rasterFunctionArguments"]["WhereClause"] = where_clause

    return _clone_layer(layer, template_dict, raster_ra)

def monitor_vegetation(raster, method='NDVI', band_indexes=None, astype=None):
    """
    The monitor_vegetation function performs an arithmetic operation on the bands 
    of a performs an arithmetic operation on the bands of a multiband raster layer 
    to reveal vegetation coverage information of the study area.
    see Band Arithmetic function at http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/band-arithmetic-function.htm

    :param raster: the input raster / imagery layer
    :param method: String. The method to create the vegetation index layer. 
                    The different vegetation indexes can help highlight certain features or reduce various noise. 
                    NDVI, SAVI, TSAVI, MSAVI, GEMI, PVI, GVITM, Sultan, VARI, GNDVI, SR, NDVIre, SRre, MTVI2,
                    RTVICore, CIre, CIg, NDWI, EVI
                    Default is NDVI.
    :param band_indexes: band indexes
    :param astype: output pixel type

    :return: output raster 
    """
    if band_indexes is None:
        raise RuntimeError('band_indexes cannot be None')
    if isinstance(method, str):
        if method.upper() == 'NDVI':
            return band_arithmetic(raster, band_indexes, astype, 1)
        elif method.upper() == 'SAVI':
            return band_arithmetic(raster, band_indexes, astype, 2)
        elif method.upper() == 'TSAVI':
            return band_arithmetic(raster, band_indexes, astype, 3)
        elif method.upper() == 'MSAVI':
            return band_arithmetic(raster, band_indexes, astype, 4)
        elif method.upper() == 'GEMI':
            return band_arithmetic(raster, band_indexes, astype, 5)
        elif method.upper() == 'PVI':
            return band_arithmetic(raster, band_indexes, astype, 6)
        elif method.upper() == 'GVITM':
            return band_arithmetic(raster, band_indexes, astype, 7)
        elif method.upper() == 'SULTAN':
            return band_arithmetic(raster, band_indexes, astype, 8)
        elif method.upper() == 'VARI':
            return band_arithmetic(raster, band_indexes, astype, 9)
        elif method.upper() == 'GNDVI':
            return band_arithmetic(raster, band_indexes, astype, 10)
        elif method.upper() == 'SR':
            return band_arithmetic(raster, band_indexes, astype, 11)
        elif method.upper() == 'NDVIRE':
            return band_arithmetic(raster, band_indexes, astype, 12)
        elif method.upper() == 'SRRE':
            return band_arithmetic(raster, band_indexes, astype, 13)
        elif method.upper() == 'MTVI2':
            return band_arithmetic(raster, band_indexes, astype, 14)
        elif method.upper() == 'RTVICORE':
            return band_arithmetic(raster, band_indexes, astype, 15)
        elif method.upper() == 'CIRE':
            return band_arithmetic(raster, band_indexes, astype, 16)
        elif method.upper() == 'CIG':
            return band_arithmetic(raster, band_indexes, astype, 17)
        elif method.upper() == 'NDWI':
            return band_arithmetic(raster, band_indexes, astype, 18)
        elif method.upper() == 'EVI':
            return band_arithmetic(raster, band_indexes, astype, 19)

    return band_arithmetic(raster, band_indexes, astype, method)

def constant_raster(constant, raster_info, gis=None):
    """
    Creates a virtual raster with a single pixel value.

    :param constant: Required list. The value of the constant to be added to the virtual raster.
    :param raster_info: Required Raster info dictionary or arcgis.raster.RasterInfo object or ImageryLayer object to set the properties of the output raster.
                        if ImageryLayer is specified then the raster information is obtained from the ImageryLayer specified. 

                        Example for RasterInfo dict - 

                        {'bandCount': 3, 'extent': {"xmin": 4488761.95, 
                        "ymin": 5478609.805, 
                        "xmax": 4489727.05, 
                        "ymax": 5479555.305, 
                        "spatialReference": {
                        "wkt": "PROJCS[\"Deutsches_Hauptdreiecksnetz_Transverse_Mercator\", 
                        GEOGCS[\"GCS_Deutsches_Hauptdreiecksnetz\", DATUM[\"D_Deutsches_Hauptdreiecksnetz\", 
                        SPHEROID[\"Bessel_1841\", 6377397.155,299.1528128]], PRIMEM[\"Greenwich\", 0.0], 
                        UNIT[\"Degree\", 0.0174532925199433]], PROJECTION[\"Transverse_Mercator\"], 
                        PARAMETER[\"false_easting\", 4500000.0], PARAMETER[\"false_northing\", 0.0], 
                        PARAMETER[\"central_meridian\", 12.0], PARAMETER[\"scale_factor\", 1.0], 
                        PARAMETER[\"latitude_of_origin\", 0.0], UNIT[\"Meter\", 1.0]]"
                        }}, 
                        'pixelSizeX': 0.0999999999999614, 
                        'pixelSizeY': 0.1, 
                        'pixelType': 'U8'}


    :param gis: Optional gis. gis parameter can be specified to render the output raster dynamically using the raster rendering service of the gis.
                If not provided, active gis will be used to do this.
                If gis parameter is not specified the output of constant_raster() cannot be displayed. 

    :return: output raster 
    """

    template_dict = {
        "rasterFunction" : "Constant",
        "rasterFunctionArguments": {
        }
    }

    if constant is not None:
        template_dict["rasterFunctionArguments"]["Constant"] = constant

    if raster_info is not None:
        layer_raster_info = {}
        if isinstance(raster_info, ImageryLayer):
            layer_raster_info = copy.deepcopy(raster_info.raster_info)
        elif isinstance(raster_info, RasterInfo):
            layer_raster_info = copy.deepcopy(raster_info.to_dict())
        else:
            layer_raster_info = copy.deepcopy(raster_info)
        if "pixelType" in layer_raster_info.keys():
            if isinstance(layer_raster_info["pixelType"], str):
                layer_raster_info["pixelType"] = _pixel_type_string_to_long(layer_raster_info["pixelType"])
        layer_raster_info.update({'type': 'RasterInfo'})
        template_dict["rasterFunctionArguments"]['RasterInfo'] = layer_raster_info
    else:
        raise RuntimeError('raster_info cannot be None')

    if gis is not None:
        newlyr = ImageryLayer(template_dict, gis)
    else:
        newlyr = ImageryLayer(template_dict,None)
    #_LOGGER.warning("""Set the desired extent on the output Imagery Layer before viewing it""")
    newlyr._fn = template_dict
    newlyr._fnra = template_dict
    return newlyr

def random_raster(raster_info, distribution=1, min_uniform=0.0, max_uniform=1.0, min_integer=1, 
                  max_integer=10, normal_mean=0.0, std_dev=1.0, exp_mean=1.0, poisson_mean= 1.0,
                  alpha=1.0, beta=1.0, N=10, r=10, probability=0.5, seed=1, generator_type=2, 
                  gis=None):
    """
    Creates a virtual raster with random values for each cell.

    :param raster_info: Required Raster info dictionary or arcgis.raster.RasterInfo object or ImageryLayer object to set the properties of the output raster.
                        if ImageryLayer is specified then the raster information is obtained from the ImageryLayer specified. 

                        Example for RasterInfo dict - 

                        {'bandCount': 3, 'extent': {"xmin": 4488761.95, 
                        "ymin": 5478609.805, 
                        "xmax": 4489727.05, 
                        "ymax": 5479555.305, 
                        "spatialReference": {
                        "wkt": "PROJCS[\"Deutsches_Hauptdreiecksnetz_Transverse_Mercator\", 
                        GEOGCS[\"GCS_Deutsches_Hauptdreiecksnetz\", DATUM[\"D_Deutsches_Hauptdreiecksnetz\", 
                        SPHEROID[\"Bessel_1841\", 6377397.155,299.1528128]], PRIMEM[\"Greenwich\",0.0], 
                        UNIT[\"Degree\", 0.0174532925199433]], PROJECTION[\"Transverse_Mercator\"], 
                        PARAMETER[\"false_easting\", 4500000.0], PARAMETER[\"false_northing\", 0.0], 
                        PARAMETER[\"central_meridian\", 12.0], PARAMETER[\"scale_factor\", 1.0], 
                        PARAMETER[\"latitude_of_origin\", 0.0], UNIT[\"Meter\", 1.0]]"
                        }}, 
                        'pixelSizeX': 0.0999999999999614, 
                        'pixelSizeY': 0.1, 
                        'pixelType': 'U8'}

    :param distribution: Optional int. Specify the random value distribution method to use.
     Default 1. i,e; Uniform
                         Choice list:
                           Uniform = 1
                           UniformInteger = 2
                           Normal = 3
                           Exponential = 4
                           Poisson = 5
                           Gamma = 6
                           Binomial = 7
                           Geometric = 8
                           NegativeBinomial = 9

                        - Uniform - A uniform distribution with the defined range.

                        - UniformInteger - An integer distribution with the defined range.

                        - Normal - A normal distribution with a defined {normal_mean} and {std_dev}. 

                        - Exponential - An exponential distribution with a defined {exp_mean}.

                        - Poisson - A Poisson distribution with a defined {Mean}.

                        - Gamma - A gamma distribution with a defined {alpha} and {beta}.
                                
                        - Binomial - A binomial distribution with a defined {N} and {probability}.

                        - Geometric - A geometric distribution with a defined {probability}. 

                        - NegativeBinomial - A Pascal distribution with a defined {r} and {probability}.

    :param min_uniform: Optional float. The default value is 0.0
    :param max_uniform: Optional float. The default value is 1.0
    :param min_integer: Optional int. The default value is 1
    :param max_integer: Optional int. The default value is 10
    :param normal_mean: Optional float. The default value is 0.0
    :param std_dev: Optional float. The default value is 1.0
    :param exp_mean: Optional float. The default value is 1.0
    :param poisson_mean: Optional float. The default value is 1.0
    :param alpha: Optional float. The default value is 1.0
    :param beta: Optional float. The default value is 1.0
    :param N: Optional int. The default value is 0.0
    :param r: Optional int. The default value is 0.0
    :param probability: Optional float. The default value is 0.5
    :param seed: Optional int. The default value is 0.0
    :param generator_type: Optional int. Default 2. i.e; MersenneTwister
                           Choice list:
                           Standard C Rand = 0
                           ACM collected algorithm 599 = 1
                           MersenneTwister = 2

    :param gis: Optional gis. gis parameter can be specified to render the output raster dynamically using the raster rendering service of the gis.
                If not provided, active gis will be used to do this.
                If gis parameter is not specified the output of constant_raster() cannot be displayed. 

    :return: output raster 
    """

    template_dict = {
        "rasterFunction" : "Random",
        "rasterFunctionArguments": {
        }
    }

    if distribution is not None:
        template_dict["rasterFunctionArguments"]["Distribution"] = distribution

    if min_uniform is not None:
        template_dict["rasterFunctionArguments"]["MinimumUniform"] = min_uniform

    if max_uniform is not None:
        template_dict["rasterFunctionArguments"]["MaximumUniform"] = max_uniform

    if min_integer is not None:
        template_dict["rasterFunctionArguments"]["MinimumInteger"] = min_integer

    if max_integer is not None:
        template_dict["rasterFunctionArguments"]["MaximumInteger"] = max_integer

    if normal_mean is not None:
        template_dict["rasterFunctionArguments"]["NormalMean"] = normal_mean

    if std_dev is not None:
        template_dict["rasterFunctionArguments"]["StandardDeviation"] = std_dev

    if exp_mean is not None:
        template_dict["rasterFunctionArguments"]["ExponentialMean"] = exp_mean

    if poisson_mean is not None:
        template_dict["rasterFunctionArguments"]["ExponentialMean"] = poisson_mean

    if alpha is not None:
        template_dict["rasterFunctionArguments"]["Alpha"] = alpha

    if beta is not None:
        template_dict["rasterFunctionArguments"]["Beta"] = beta

    if N is not None:
        template_dict["rasterFunctionArguments"]["N"] = N

    if r is not None:
        template_dict["rasterFunctionArguments"]["r"] = r

    if probability is not None:
        template_dict["rasterFunctionArguments"]["Probability"] = probability

    if seed is not None:
        template_dict["rasterFunctionArguments"]["Seed"] = seed

    if generator_type is not None:
        template_dict["rasterFunctionArguments"]["GeneratorType"] = generator_type

    if raster_info is not None:
        layer_raster_info = {}
        if isinstance(raster_info, ImageryLayer):
            layer_raster_info = copy.deepcopy(raster_info.raster_info)
        elif isinstance(raster_info, RasterInfo):
            layer_raster_info = copy.deepcopy(raster_info.to_dict())
        else:
            layer_raster_info = copy.deepcopy(raster_info)
        if "pixelType" in layer_raster_info.keys():
            if isinstance(layer_raster_info["pixelType"], str):
                layer_raster_info["pixelType"] = _pixel_type_string_to_long(layer_raster_info["pixelType"])
        layer_raster_info.update({'type': 'RasterInfo'})
        template_dict["rasterFunctionArguments"]['RasterInfo'] = layer_raster_info
    else:
        raise RuntimeError('raster_info cannot be None')

    if gis is not None:
        newlyr = ImageryLayer(template_dict, gis)
    else:
        newlyr = ImageryLayer(template_dict,None)
    #_LOGGER.warning("""Set the desired extent on the output Imagery Layer before viewing it""")
    newlyr._fn = template_dict
    newlyr._fnra = template_dict
    return newlyr

def aggregate_cells(raster, cell_factor=2, aggregation_type=9, extent_handling=False, ignore_nodata=False):

    """
    Generates a reduced-resolution version of a raster.

    :param raster: the input raster
    :param cell_factor: The factor by which to multiply the cell size of the input raster to 
                        obtain the desired resolution for the output raster.
                        For example, a cell factor value of three would result in an output cell 
                        size three times larger than that of the input raster.
                        The value must be an integer greater than 1.
    :param aggregation_type: Optional int. Establishes how the value for each output cell will be determined.
                             The values of the input cells encompassed by the coarser output cell 
                             are aggregated by one of the following statistics:

                                - 2 (MAXIMUM) : The largest value of the input cells.
                                - 3 (MEAN) : The average value of the input cells.
                                - 4 (MEDIAN) : The median value of the input cells.
                                - 5 (MINIMUM) : The smallest value of the input cells.
                                - 9 (SUM) : The sum (total) of the input cell values.This is the default.

    :param extent_handling: Optional boolean. Defines how to handle the boundaries of the input raster when its rows 
                            or columns are not a multiple of the cell factor.
                                - True : Expands the top or right boundaries of the input raster so 
                                  the total number of cells in a row or column is a multiple 
                                  of the cell factor. Those expanded cells are given a value 
                                  of NoData. With this option, the output raster can cover a 
                                  larger spatial extent than the input raster. This is the default.
                                - False : Reduces the number of rows or columns in the output raster by 1. 
                                  This will truncate the remaining cells on the top or right 
                                  boundaries of the input raster, making the number of rows 
                                  or columns in the input raster a multiple of the cell factor.
                                  With this option, the output raster can cover a smaller 
                                  spatial extent than the input raster.

    :param ignore_nodata: Optional boolean. Denotes whether NoData values are ignored by the aggregation calculation.

                          - True : Specifies that if NoData values exist for any of the cells that fall within the 
                            spatial extent of a larger cell on the output raster, the NoData values will be 
                            ignored when determining the value for output cell locations. Only input cells 
                            within the extent of the output cell that have data values will be used in
                            determining the value of the output cell. This is the default.
                          - False : Specifies that if any cell that falls within the spatial extent of a larger 
                            cell on the output raster has a value of NoData, the value for that output 
                            cell location will be NoData.When the this option is used, it is implied 
                            that when cells within an aggregation contain the NoData value, 
                            there is insufficient information to perform the specified calculations 
                            necessary to determine an output value.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Aggregate",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }
    
    if cell_factor is not None:
        template_dict["rasterFunctionArguments"]['CellFactor'] = cell_factor

    if aggregation_type is not None:
        template_dict["rasterFunctionArguments"]['AggregationType'] = aggregation_type

    if extent_handling is not None:
        template_dict["rasterFunctionArguments"]['ExpandHandling'] = extent_handling

    if ignore_nodata is not None:
        template_dict["rasterFunctionArguments"]['IgnoreNoData'] = ignore_nodata

    return _clone_layer(layer, template_dict, raster_ra)

def _raster_item(raster, raster_id=None):
    """
    :param raster: the input raster
    :param conversion_parameters: array of double (A length of N array representing weights for each band, where N=band count.)
    :return: the output raster with this function applied to it
    """
 


    template_dict = {
        "rasterFunction" : "RasterItem",
        "rasterFunctionArguments": {}
    }
    
    if raster is not None and isinstance(raster, ImageryLayer):
        template_dict["rasterFunctionArguments"]['URL'] = raster.url

    if raster is not None and isinstance(raster, str):
        template_dict["rasterFunctionArguments"]['URL'] = raster

    if raster_id is not None:
        template_dict["rasterFunctionArguments"]['RasterID'] = raster_id

    layer, raster, raster_ra = _raster_input(raster)

    return _clone_layer(layer, template_dict, raster_ra)

def generate_trend(raster, dimension_name, regression_type=0, cycle_length = 1, cycle_unit = "YEARS",
                   harmonic_frequency=1, polynomial_order=2, ignore_nodata=True, rmse=True, r2=False,slope_p_value=False,
                   seasonal_period="DAYS"):

    """
    Estimates the trend for each pixel along a dimension for one or more variables in a multidimensional raster.

    :param raster: The input multidimensional raster.
    :param dimension_name: Required String. The dimension along which a trend will be extracted for the 
                           variable or variables selected in the analysis.
    :param regression_type: Optional Integer or String. Specifies the type of line to be used to fit to the pixel values along a dimension.

                            - 0 (LINEAR) : Fits the pixel values for a variable along a linear trend line. This is the default.
                            - 1 (HARMONIC) : Fits the pixel values for a variable along a harmonic trend line.
                            - 2 (POLYNOMIAL) : Fits the pixel values for a variable along a second-order polynomial trend line.
                            - 3 (MANN-KENDALL) : Variable pixel values will be evaluated using the Mann-Kendall trend test. 
                            - 4 (SEASONAL-KENDALL) : Variable pixel values will be evaluated using the Seasonal-Kendall trend test. 

    :param cycle_length: Optional Integer. The length of periodic variation to model. This parameter is required when Trend Line Type is
                         set to Harmonic. For example, leaf greenness often has one strong cycle of variation in a single year, so the 
                         cycle length is 1 year. Hourly temperature data has one strong cycle of variation throughout a single day, 
                         so the cycle length is 1 day. Default is 1.
    :param cycle_unit: Optional String. Specifies the time unit to be used for the length of harmonic cycle. Options: "DAYS", "YEARS" (This is the default).

    :param harmonic_frequency: Optional Integer. The frequency number to use in the trend fitting. This parameter specifies the 
                               frequency of cycles in a year. The default value is 1, or one harmonic cycle per year. 
                               This parameter is only included in the trend analysis for a harmonic regression (regression_type=1).
    :param polynomial_order: Optional Integer. The polynomial order number to use in the trend fitting. This parameter specifies the 
                             polynomial order. The default value is 2, or second-order polynomial. This parameter 
                             is only included in the trend analysis for a polynomial regression (regression_type=2).
    :param ignore_nodata: Optional Boolean. Specifies whether NoData values are ignored in the analysis.
                          
                          - True : The analysis will include all valid pixels along a given dimension and ignore any NoData pixels. This is the default.
                          - False : The analysis will result in NoData if there are any NoData values for the pixels along the given dimension.

    :param rmse: Optional Boolean. Specifies whether to generate the root mean square error (RMSE) of the trend fit line.

                 - True : The RMSE will be calculated and displayed when the tool is finished running. This is the default.
                 - False : The RMSE will not be calculated.

    :param r2: Optional Boolean. Specifies whether to calculate the R-squared goodness-of-fit statistic for the trend fit line.

               - True : The R-squared will be calculated and displayed when the tool is finished running.
               - False : The R-squared will not be calculated. This is the default.

    :param slope_p_value: Optional Boolean. Specifies whether to calculate the p-value statistic for the slope coefficient of the trend line.

                          - True : The p-value will be calculated and displayed when the tool is finished running. 
                          - False : The p-value will not be calculated. This is the default.

    :param seasonal_period: Optional String. Specifies the seasonal period. Default - "DAYS"
                            Possible Options - "DAYS", "MONTHS"


    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "TrendAnalysis",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }
    
    if dimension_name is not None:
        template_dict["rasterFunctionArguments"]['DimensionName'] = dimension_name

    regression_type_dict = {"LINEAR":0, "HARMONIC":1, "POLYNOMIAL":2, "MANN-KENDALL":3,"SEASONAL-KENDALL":4}
    if regression_type is not None:
        if isinstance(regression_type, str):
            regression_type= regression_type_dict[regression_type.upper()]
            template_dict["rasterFunctionArguments"]['RegressionType'] = regression_type
        else:
            template_dict["rasterFunctionArguments"]['RegressionType'] = regression_type

    if cycle_length is not None:
        template_dict["rasterFunctionArguments"]['CycleLength'] = cycle_length

    if cycle_unit is not None:
        template_dict["rasterFunctionArguments"]['CycleUnit'] = cycle_unit

    if harmonic_frequency is not None:
        template_dict["rasterFunctionArguments"]['Frequency'] = harmonic_frequency

    if polynomial_order is not None:
        template_dict["rasterFunctionArguments"]['Order'] = polynomial_order

    if ignore_nodata is not None:
        template_dict["rasterFunctionArguments"]['IgnoreNoData'] = ignore_nodata

    if rmse is not None:
        template_dict["rasterFunctionArguments"]['RMSE'] = rmse

    if r2 is not None:
        template_dict["rasterFunctionArguments"]['R2'] = r2

    if slope_p_value is not None:
        template_dict["rasterFunctionArguments"]['SlopePValue'] = slope_p_value

    if seasonal_period is not None:
        seasonal_period_allowed_values = ["DAYS", "MONTHS"]
        if [element.lower() for element in seasonal_period_allowed_values].count(seasonal_period.lower()) <= 0 :
            raise RuntimeError('seasonal_period can only be one of the following:  '+str(seasonal_period_allowed_values))
        for element in seasonal_period_allowed_values:
            if seasonal_period.lower() == element.lower():
                seasonal_period = element
        template_dict["rasterFunctionArguments"]['SeasonalPeriod'] = seasonal_period

    return _clone_layer(layer, template_dict, raster_ra)



def predict_using_trend(raster, dimension_definition_type=0, dimension_values=None, start=None, end=None, interval_value=1, interval_unit='HOURS'):

    """
    Computes a forecasted multidimensional raster layer using the output trend raster from the generate_trend function.

    :param raster: The input raster.
    :param dimension_definition_type: Optional Integer or String. Specifies the method used to provide prediction dimension values.

                                      - 0 (BY_VALUE) : The prediction will be calculated for a single dimension value. For example, you want to predict yearly precipitation for the years 2050, 2100, and 2150. This is the default.
                                      - 1 (BY_INTERVAL) : The prediction will be calculated for an interval of the dimension defined by a start and an end value. For example, you want to predict yearly precipitation for every year between 2050 and 2150.

    :param dimension_values: Optional List. The dimension value or values to be used in the prediction. The format of the time, 
                             depth, and height values must match the format of the dimension values used to 
                             generate the trend raster. If the trend raster was generated for the StdTime dimension, 
                             the format should be YYYY-MM-DDTHH:MM:SS, 
                             for example, 2050-01-01T00:00:00.

    :param start: Optional String or Integer. The start date, height, or depth of the dimension interval to be used in the prediction.
                            This parameter is required when the dimension_definition_type parameter is set to 1 (By Interval).

    :param end: Optional String or Integer. The end date, height, or depth of the dimension interval to be used in the prediction.
                          This parameter is required when the dimension_definition_type parameter is set to 1 (By Interval).
    :param interval_value: Optional Integer. The number of steps between two dimension values to be included in the prediction. The default value is 1.
                                This parameter is required when the dimension_definition_type parameter is set to 1 (By Interval).
    :param interval_unit: Optional String. The unit that will be used for the value interval. This parameter only applies when the dimension of analysis is a time dimension.

                            - HOURS - The prediction will be calculated for each hour in the range of time described by the start, end, and interval_value parameters.

                            - DAYS - The prediction will be calculated for each day in the range of time described by the start, end, and interval_value parameters.

                            - WEEKS - The prediction will be calculated for each week in the range of time described by the start, end, and interval_value parameters.

                            - MONTHS - The prediction will be calculated for each month in the range of time described by the start, end, and interval_value parameters.

                            - YEARS - The prediction will be calculated for each year in the range of time described by the start, end, and interval_value parameters.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Trend",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }

    dimension_definition_type_dict = {"BY_VALUE":0, "BY_INTERVAL":1}
    if dimension_definition_type is not None:
        if isinstance(dimension_definition_type, str):
            dimension_definition_type= dimension_definition_type_dict[dimension_definition_type.upper()]
            template_dict["rasterFunctionArguments"]['DimensionDefinitionType'] = dimension_definition_type
        else:
            template_dict["rasterFunctionArguments"]['DimensionDefinitionType'] = dimension_definition_type
    

    if dimension_values is not None:
        if isinstance(dimension_values, list):
            values = ";".join(str(ele) for ele in dimension_values)
            template_dict["rasterFunctionArguments"]['DimensionValues'] = values
        else:
            template_dict["rasterFunctionArguments"]['DimensionValues'] = dimension_values

    if start is not None:
        template_dict["rasterFunctionArguments"]['DimensionStart'] = start

    if end is not None:
        template_dict["rasterFunctionArguments"]['DimensionEnd'] = end

    if interval_value is not None:
        template_dict["rasterFunctionArguments"]['DimensionInterval'] = interval_value

    if interval_unit is not None:
        template_dict["rasterFunctionArguments"]['DimensionUnit'] = interval_unit

    return _clone_layer(layer, template_dict, raster_ra)


def linear_spectral_unmixing(raster,spectral_profile_def=None, non_negative=False, sum_to_one=False):

    """
    Performs subpixel classification and calculates the fractional abundance of different land cover types for individual pixels. 

    :param raster: The input raster.
    :param spectral_profile_def: Optional dict. The class spectral profile information.
    :param non_negative: Optional bool. Specifies the options to define the output pixel values.

                         - True : There will be no negative output values.
                         - False : There can be negative values of fractional land cover. This is the default.

    :param sum_to_one: Optional bool. Specifies the options to define the output pixel values.

                        - True : Class values for each pixel are provided in decimal format with the sum of all classes equal to 1. 

                          Example: 
                            Class1 = 0.16; Class2 = 0.24; Class3 = 0.60.

                        - False : The sum of all classes in a pixel can exceed 1. This is the default.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "SpectralUnmixing",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }

    if spectral_profile_def is not None:
        template_dict["rasterFunctionArguments"]['SpectralProfileDefinition'] = spectral_profile_def

    if non_negative is not None:
        template_dict["rasterFunctionArguments"]['NonNegative'] = non_negative

    if sum_to_one is not None:
        template_dict["rasterFunctionArguments"]['SumToOne'] = sum_to_one


    return _clone_layer(layer, template_dict, raster_ra)


def multidimensional_filter(raster, 
                            variables=None,
                            dimension_definition='ALL',
                            dimension=None,
                            dimension_ranges=None,
                            dimension_values=None,
                            start_of_first_iteration=None,
                            end_of_first_iteration=None,
                            iteration_step=3,
                            iteration_unit=None,
                            dimensionless=False):

    """
    Applies a filter on a multidimensional raster. Function creates a raster layer 
    from a multidimensional raster by slicing data along defined variables and dimensions.

    :param raster: The input multidimensional raster
    :param variables: Optional List. The list of variables that will be included in the output 
                      multidimensional raster layer. If no variable is specified, the first variable will be used.

    :param dimension_definition: Optional String. Specifies the method that will be used to slice the dimension.

                                    - ALL : The full range for each dimension will be used. This is the default.

                                    - BY_RANGES : The dimension will be sliced using a range or a list of ranges.

                                    - BY_ITERATION : The dimension will be sliced over a specified interval.

                                    - BY_VALUES : The dimension will be sliced using a list of dimension values.

    :param dimension: Optional String. The dimension along which the variables will be sliced. 
                      This parameter is required when the dimension_definition is set to BY_ITERATION.

    :param dimension_ranges: Optional list of dicts. This slices the data based on the dimension name and the minimum and maximum values for the range. This parameter is required when the dimension_definition is set to BY_RANGE.
                             If dimension is StdTime, then the min value and max value must be specified in 
                             human readable time format (YYYY-MM-DDTHH:MM:SS). The input should be specified as:
                             [{"dimension":"<dimension_name>", "minValue":"<dimension_min_value>", "maxValue":"<dimension_max_value>"},{"dimension":"<dimension_name>", "minValue":"<dimension_min_value>", "maxValue":"<dimension_max_value>"}]

                             Example:
                                 [{"dimension":"StdTime", "minValue":"2013-05-17T00:00:00", "maxValue":"2013-05-17T03:00:00"},{"dimension":"StdZ", "minValue":"-5000", "maxValue":"-4000"}]

    :param dimension_values: Optional List of Dicts. This slices the data based on the dimension name and the value specified.
                             This parameter is required when the dimension_definition is set to BY_VALUE.

                             If dimension is StdTime, then the value must be specified in 
                             human readable time format (YYYY-MM-DDTHH:MM:SS). The input should be specified as:
                             [{"dimension":"<dimension_name>", "value":"<dimension_value>"},{"dimension":"<dimension_name>", "value":"<dimension_value>"}]

                             Example:
                                 [{"dimension":"StdTime", "value":"2012-01-15T03:00:00"},{"dimension":" StdZ ", "value":"-4000"}]

    :param start_of_first_iteration: Optional String. The beginning of the interval. 
                                     This parameter is required when the dimension_definition is set to BY_ITERATION
    :param end_of_first_iteration: Optional String. The end of the interval. 
                                   This parameter is required when the dimension_definition is set to BY_ITERATION
    :param iteration_step: Optional Float. The frequency with which the data will be sliced. 
                           This parameter is required when the dimension_definition is set to BY_ITERATION
                           The default is 3.
    :param iteration_unit: Optional String. Specifies the iteration unit. 
                           This parameter is required when the dimension_definition is set to BY_ITERATION
                           and the dimension parameter is set to StdTime.

                            - HOURS - Uses hours as the specified unit of time.

                            - DAYS - Uses days as the specified unit of time.

                            - WEEKS - Uses weeks as the specified unit of time.

                            - MONTHS - Uses months as the specified unit of time.

                            - YEARS -Uses years as the specified unit of time.

    :param dimensionless: Optional Boolean. Specifies whether the layer should have dimension values. This option is only available if a single slice is selected to create a layer.

                           - True - The layer will not have dimension values.
                           - False - The layer will have a dimension value. This is the default.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    dimension_definition_val = dimension_definition
    if dimension_definition is not None:
        dimension_definition_allowed_values = ['ALL', 'BY_VALUES', 'BY_RANGES', 'BY_ITERATION']
        if [element.lower() for element in dimension_definition_allowed_values].count(dimension_definition.lower()) <= 0 :
            raise RuntimeError('dimension_definition can only be one of the following: '+str(dimension_definition_allowed_values))

        for element in dimension_definition_allowed_values:
            if dimension_definition.upper() == element:
                dimension_definition_val = element

    iteration_unit_val = iteration_unit
    if iteration_unit is not None:
        iteration_unit_allowed_values = ['HOURS','DAYS', 'DAILY', 'WEEKS', 'MONTHS', 'YEARS']
        if [element.lower() for element in iteration_unit_allowed_values].count(iteration_unit.lower()) <= 0 :
            raise RuntimeError('iteration_unit can only be one of the following: '+str(iteration_unit_allowed_values))

        for element in iteration_unit_allowed_values:
            if iteration_unit.upper() == element:
                iteration_unit_val = element

    filter_dict = {}
    filter_dict.update({"definitionType": dimension_definition_val})

    if variables is not None:
        if isinstance(variables, list):
            filter_dict.update({"variables":variables})
        else:
            filter_dict.update({"variables":[variables]})


    if dimension_definition == "BY_RANGES":
        dimensions_list = []
        min_values_list = []
        max_values_list = []
        if isinstance(dimension_ranges, list):
            for ele in dimension_ranges:
                if isinstance(ele,dict):
                    min_values_list.append(ele["minValue"])
                    max_values_list.append(ele["maxValue"])
                    dimensions_list.append(ele["dimension"])
        filter_dict.update({"dimensions":dimensions_list, "minValues":min_values_list, "maxValues":max_values_list})

    elif dimension_definition == "BY_ITERATION":
        filter_dict.update({"dimension":dimension, 
                            "startValue":start_of_first_iteration, 
                            "endValue":end_of_first_iteration,
                            "stepValue":iteration_step})
        if iteration_unit is not None:
            filter_dict.update({"units":iteration_unit_val})

    elif dimension_definition == "BY_VALUES":
        dimensions_list = []
        values_list = []
        if isinstance(dimension_values, list):
            for ele in dimension_values:
                if isinstance(ele,dict):
                    values_list.append(ele["value"])
                    dimensions_list.append(ele["dimension"])
        filter_dict.update({"dimensions":dimensions_list, "values":values_list})

    template_dict = {
        "rasterFunction" : "MultidimensionalFilter",
        "rasterFunctionArguments": {
            "Raster" : raster,  
            "Filter" : _json.dumps(filter_dict)
        }
    }

    if dimensionless is not None:
        if isinstance(dimensionless, bool):
            template_dict["rasterFunctionArguments"]['Dimensionless'] = dimensionless
        else:
            raise RuntimeError('dimensionless should be an instance of bool')

    return _clone_layer(layer, template_dict, raster_ra)

def s1_radiometric_calibration(raster, calibration_type=None):

    """
    Performs different types of radiometric calibration on Sentinel-1 data.

    :param raster: The input raster.
                   The Sentinel-1 Level-1 GRD or SLC input raster you want to process.

                   The function will use the LUT file either to apply the thermal correction or to 
                   remove the correction, depending on the contents of the LUT.
    :param calibration_type: Optional string or int. one of four calibration types:
    
                             - "beta_nought" (0) - produces an output containing the radar brightness coefficient.
                             - "sigma_nought" (1) - the backscatter returned to the antenna from a unit area on the ground, related to ground range.
                             - "gamma" (2) - measurement of emitted and returned energy useful for determining antenna patterns.
                             - None - Specify None to not apply a correction. This is the default.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "S1RadiometricCalibration",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }
    
    calibration_type_dict = {"beta_nought":0, "sigma_nought":1, "gamma":2}
    if calibration_type is not None:
        if isinstance(calibration_type, str):
            calibration_type= calibration_type_dict[calibration_type.upper()]
            template_dict["rasterFunctionArguments"]['CalibrationType'] = calibration_type
        else:
            template_dict["rasterFunctionArguments"]['CalibrationType'] = calibration_type
    else:
        template_dict["rasterFunctionArguments"]['CalibrationType'] = 3

    return _clone_layer(layer, template_dict, raster_ra)


def s1_thermal_noise_removal(raster, calibration_type=None):

    """
    Removes thermal noise from Sentinel-1 data.

    :param raster: The input raster.
                   The Sentinel-1 Level-1 GRD or SLC input raster you want to process.

                   The function will use the LUT file either to apply the thermal correction or to 
                   remove the correction, depending on the contents of the LUT.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "S1ThermalNoiseRemoval",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)

def interpolate_irregular_data(point_feature, value_field=None, cell_size=0, interpolation_method = 0, radius=3, gis=None):
    """
    Interpolates from point clouds or irregular grids. (Supported from 10.8.1)

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    point_feature_class                  Required. The input point feature layer.
    --------------------------------     --------------------------------------------------------------------
    value_field                          Optional string. The name of the field that contains the value of the points to be interpolated.
    --------------------------------     --------------------------------------------------------------------
    cell_size                            Optional int. The cell size for the output raster dataset. 
    --------------------------------     --------------------------------------------------------------------
    interpolation_method                 Optional int or string. The resampling method to use for interpolation:

                                          -  0 or NEAREST_NEIGBOR : Calculates pixel value using the nearest pixel. If no source pixel 
                                             exists, no new pixel can be created in the output. This is the default.

                                          -  2 or LINEAR_TINNING : Uses a triangular irregular network from the center 
                                             points of each cell in the irregular raster to interpolate a surface 
                                             that is then converted to a regular raster.

                                          -  3 or NATURAL_NEIGHBOR : Finds the closest subset of input samples to a 
                                             query point and applies weights to them based on proportionate 
                                             areas to interpolate a value.

                                          -  4 or INVERSE_DISTANCE_WEIGHTED : Determines cell values using a 
                                             linearly weighted combination of a set of sample points or cells. 
                                             The weight is a function of the inverse of the distance from the known points or cells.
    --------------------------------     --------------------------------------------------------------------
    radius                               Optional int. The number of pixels to be included for resampling. The default value is 3 pixels.
    --------------------------------     --------------------------------------------------------------------
    gis                                  Optional gis. gis parameter can be specified to render the output raster 
                                         dynamically using the raster rendering service of the gis.
                                         If not provided, active gis will be used to do this.
                                         If gis parameter is not specified the output of interpolate_irregular_data() cannot be displayed. 
    ================================     ====================================================================

    :returns: output raster with function applied

    """
    from arcgis.geoprocessing._support import _layer_input
    point_feature = _layer_input(point_feature)
    template_dict = {
        "rasterFunction" : "InterpolateIrregularData",
        "rasterFunctionArguments": {"PointFeatureClass":point_feature,
                                    "RasterInfo":{"blockWidth" : 2048,
                                                "blockHeight" : 256,
                                                "bandCount" : 0,
                                                "pixelType" : -1,
                                                "pixelSizeX" : cell_size,
                                                "pixelSizeY" : cell_size,
                                                "format" : "",
                                                "compressionType" : "",
                                                "firstPyramidLevel" : 1,
                                                "maximumPyramidLevel" : 30,
                                                "packetSize" : 4,
                                                "compressionQuality" : 0,
                                                "pyramidResamplingType" : -1,
                                                "type" : "RasterInfo"}
                                 }
                 }

    interpolation_methods = {
        'NEAREST_NEIGBOR': 0,
        'LINEAR_TINNING': 2,
        'NATURAL_NEIGHBOR' : 3,
        'INVERSE_DISTANCE_WEIGHTED': 4
    }


    if interpolation_method is not None:
        if isinstance(interpolation_method, str):
            interpolation_method = interpolation_methods[interpolation_method.upper()]
        else:
            interpolation_method = interpolation_method
        template_dict["rasterFunctionArguments"]["InterpolationMethod"] = interpolation_method

    if value_field is not None:
        template_dict["rasterFunctionArguments"]["ValueField"] = value_field

    if radius is not None:
        template_dict["rasterFunctionArguments"]["Radius"] = radius

    if gis is not None:
        newlyr = ImageryLayer(template_dict, gis)
    else:
        newlyr = ImageryLayer(template_dict,None)
    #_LOGGER.warning("""Set the desired extent on the output Imagery Layer before viewing it""")
    newlyr._fn = template_dict
    newlyr._fnra = template_dict
    return newlyr

def _simple_collection(raster, md_info=None):
    """"
    The function is used to define the source raster as part of the default
    mosaicking behavior of the mosaic dataset. This function is a no-op function
    and takes no arguments except a raster. For more information, see
    (http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/identity-function.htm)
    :param raster: the input raster / imagery layer
    :return: the innput raster
    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "SimpleCollection",
        "rasterFunctionArguments": {
            "Raster" : raster,
        },
        "variableName": "Rasters"
    }

    if md_info is not None:
        template_dict["rasterFunctionArguments"]['MultidimensionalInfo'] = md_info


    return _clone_layer(layer, template_dict, raster_ra)

def aggregate(raster,
              dimension=None,
              aggregation_function=None,
              aggregation_definition_type="ALL",
              interval_keyword=None,
              interval_value=None,
              interval_unit=None,
              interval_ranges=None,
              ignore_nodata=False
              ):
    """
     Creates a new raster by applying an aggregation function
    :param raster: Input Raster. 
    :param aggregation_function: Optional String. Specifies the mathematical method that will be used
                               to combine the aggregated slices in an interval.

                                - MEAN : Calculates the mean of a pixel's values across all slices in the interval. This is the default.

                                - MAXIMUM : Calculates the maximum value of a pixel across all slices in the interval.

                                - MAJORITY : Calculates the value that occurred most frequently for a pixel across all slices in the interval.

                                - MINIMUM : Calculates the minimum value of a pixel across all slices in the interval.

                                - MINORITY : Calculates the value that occurred least frequently for a pixel across all slices in the interval.

                                - MEDIAN : Calculates the median value of a pixel across all slices in the interval.

                                - RANGE : Calculates the range of values for a pixel across all slices in the interval.

                                - STD : Calculates the standard deviation of a pixel's values across all slices in the interval.

                                - SUM : Calculates the sum of a pixel's values across all slices in the interval.

                                - VARIETY : Calculates the number of unique values of a pixel across all slices in the interval.

                                You may also pass custom aggregaation function. 
                                Create an RFT object out of the raster function template item on the portal and 
                                specify that as the input to aggregation_function or directly specify the RFT in JSON format as the 
                                aggregation_method.

    :param aggregation_definition_type: Optional String. Specifies the dimension interval for which the data
                                        will be aggregated.

                                        - ALL : The data values will be aggregated across all slices. This is the default.

                                        - INTERVAL_KEYWORD : The variable data will be aggregated using a commonly known interval.

                                        - INTERVAL_VALUE : The variable data will be aggregated using a user-specified interval and unit.

                                        - INTERVAL_RANGES : The variable data will be aggregated between specified pairs of values or dates.
    :param dimension: Optional String. This is the dimension along which the variables will be aggregated.
    :param interval_keyword: Optional String. Specifies the keyword interval that will be used
                                             when aggregating along the dimension. This parameter is required
                                             when the aggregation_def parameter is set to INTERVAL_KEYWORD, and
                                             the aggregation must be across time.

                                             - HOURLY : The data values will be aggregated into hourly time steps, 
                                               and the result will include every hour in the time series.

                                             - DAILY : The data values will be aggregated into daily time steps, 
                                               and the result will include every day in the time series.

                                             - WEEKLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include every week in the time series.

                                             - DEKADLY : Divides each month into 3 periods of 10 days each 
                                               (last period might have more or less than 10 days)
                                               and each month would output 3 slices.

                                             - PENTADLY : Divides each month into 6 periods of 5 days each 
                                               (last period might have more or less than 5 days)
                                               and each month would output 6 slices.

                                             - MONTHLY : The data values will be aggregated into monthly time steps, 
                                               and the result will include every month in the time series.

                                             - QUARTERLY : The data values will be aggregated into quarterly time steps, 
                                               and the result will include every quarter in the time series.

                                             - YEARLY : The data values will be aggregated into yearly time steps, 
                                               and the result will include every year in the time series.

                                             - RECURRING_DAILY : The data values will be aggregated into daily time steps, 
                                               and the result includes each one aggregated value per day. 
                                               The output will include, at most, 366 daily time slices

                                             - RECURRING_WEEKLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per week. 
                                               The output will include, at most, 53 weekly time slices.

                                             - RECURRING_MONTHLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per month. 
                                               The output will include, at most, 12 monthly time slices.

                                             - RECURRING_QUARTERLY : The data values will be aggregated into weekly time steps, 
                                               and the result will include one aggregated value per quarter. 
                                               The output will include, at most, 4 quarterly time slices.

    :param interval_value: Optional String. The size of the interval that will be used for the
                           aggregation. This parameter is required when the aggregation_def
                           parameter is set to INTERVAL_VALUE.

                           For example, to aggregate 30 years of monthly temperature data into
                           5-year increments, enter 5 as the interval_value, and specify
                           interval_unit as YEARS.

    :param interval_unit: Optional String. The unit that will be used for the interval value.
                          This parameter is required when the dimension parameter is set to a
                          time field and the aggregation_def parameter is set to INTERVAL_VALUE.

                          If you are aggregating over anything other than time, this option
                          will not be available and the unit for the interval value will match
                          the variable unit of the input multidimensional raster data.

                          - HOURS : The data values will be aggregated into hourly time slices at the interval provided.
                          - DAYS : The data values will be aggregated into daily time slices at the interval provided.
                          - WEEKS : The data values will be aggregated into weekly time slices at the interval provided.
                          - MONTHS : The data values will be aggregated into monthly time slices at the interval provided.
                          - YEARS : The data values will be aggregated into yearly time slices at the interval provided.
    :param interval_ranges: Optional List of dictionary objects. Interval ranges specified as list of dictionary objects 
                            that will be used to aggregate groups of values. 

                            This parameter is required when the aggregation_definition parameter is set to INTERVAL_RANGE.
                            If dimension is StdTime, then the value must be specified in human readable time format (YYYY-MM-DDTHH:MM:SS).

                            Syntax: 
                                [{"minValue":"<min value>","maxValue":"<max value>"},
                                {"minValue":"<min value>","maxValue":"<max value>"}]

                            Example:
                                [{"minValue":"2012-01-15T03:00:00","maxValue":"2012-01-15T09:00:00"},
                                {"minValue":"2012-01-15T12:00:00","maxValue":"2012-01-15T21:00:00"}]
    
    :param ignore_nodata: Optional Boolean. Specifies whether NoData values are ignored.

                            - True : The function will include all valid pixels and ignore any NoData pixels. This is the default.
                            - False : The function will result in NoData if there are any NoData values.
    :return: the output raster with function applied on it
    """
    from arcgis.raster._util import _local_function_template

    layer, raster, raster_ra = _raster_input(raster)
       
    template_dict = {
        "rasterFunction": "RasterCollection",
        "rasterFunctionArguments": {
            "RasterCollection": raster
        },
        "variableName": "RasterCollection"
    }
    

    if aggregation_function is not None:
        if isinstance(aggregation_function, RFT):
            template_dict["rasterFunctionArguments"]["AggregationFunction"]=aggregation_function._rft_json
        elif isinstance(aggregation_function, dict):
            template_dict["rasterFunctionArguments"]["AggregationFunction"]=aggregation_function
        elif isinstance(aggregation_function, str):
            opnum = None
            if aggregation_function.upper() == "MEAN":
                opnum = 68 if ignore_nodata else 40
            elif aggregation_function.upper() == "MAXIMUM":
                opnum = 67 if ignore_nodata else 39
            elif aggregation_function.upper() == "MAJORITY":
                opnum = 66 if ignore_nodata else 38
            elif aggregation_function.upper() == "MINIMUM":
                opnum = 70 if ignore_nodata else 42
            elif aggregation_function.upper() == "MINORITY":
                opnum = 71 if ignore_nodata else 43
            elif aggregation_function.upper() == "MEDIAN":
                opnum = 69 if ignore_nodata else 41
            elif aggregation_function.upper() == "RANGE":
                opnum = 72 if ignore_nodata else 47
            elif aggregation_function.upper() == "STD":
                opnum = 73 if ignore_nodata else 54
            elif aggregation_function.upper() == "SUM":
                opnum = 74 if ignore_nodata else 55
            elif aggregation_function.upper() == "VARIETY":
                opnum = 75 if ignore_nodata else 58

            if opnum is None:
                raise RuntimeError("Invalid aggregation_function")
            else:
                template_dict["rasterFunctionArguments"]["AggregationFunction"] = _local_function_template(opnum)
        if "type" not in template_dict["rasterFunctionArguments"]["AggregationFunction"]:
            template_dict["rasterFunctionArguments"]["AggregationFunction"].update({'type':'RasterFunctionTemplate'})

    aggregation_definition = {}
    aggregation_definition.update({"definitionType": aggregation_definition_type})

    if dimension is not None:
         aggregation_definition.update({"dimension":dimension})

    if aggregation_definition_type.upper() == "INTERVAL_VALUE":
        if interval_value is None:
            raise RuntimeError("interval_value cannot be None, if aggregation_definition_type is INTERVAL_VALUE")

        aggregation_definition.update({"intervalValue":interval_value})
        if interval_unit is not None:
            aggregation_definition.update({"intervalUnits":interval_unit})

    elif aggregation_definition_type.upper() == "INTERVAL_KEYWORD":
        if interval_keyword is None:
            raise RuntimeError("interval_keyword cannot be None, if aggregation_definition_type is INTERVAL_KEYWORD")

        interval_keyword_val=interval_keyword
        if interval_keyword is not None:
            interval_keyword_allowed_values = ['HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'YEARLY', 'RECURRING_DAILY', 'RECURRING_WEEKLY',
                                               'RECURRING_MONTHLY', 'RECURRING_QUARTERLY', 'PENTADLY', 'DEKADLY']
            if [element.lower() for element in interval_keyword_allowed_values].count(interval_keyword.lower()) <= 0 :
                raise RuntimeError('interval_keyword can only be one of the following: '+str(interval_keyword_allowed_values))
            interval_keyword_val=interval_keyword
            for element in interval_keyword_allowed_values:
                if interval_keyword.upper() == element:
                    interval_keyword_val = element
        aggregation_definition.update({"intervalKeyword":interval_keyword})

    elif aggregation_definition_type.upper() == "INTERVAL_RANGES":
        if interval_ranges is None:
            raise RuntimeError("interval_ranges cannot be None, if aggregation_definition_type is INTERVAL_RANGES")
            
        min_list=[]
        max_list=[]
        if isinstance(interval_ranges, list):
            for ele in interval_ranges:
                if isinstance(ele,dict):
                    min_list.append(ele["minValue"])
                    max_list.append(ele["maxValue"])

        if isinstance(interval_ranges, dict):
            min_list.append(interval_ranges["minValue"])
            max_list.append(interval_ranges["maxValue"])

        aggregation_definition.update({"minValues":min_list})
        aggregation_definition.update({"maxValues":max_list})

    if aggregation_definition is not None:
        template_dict["rasterFunctionArguments"]["AggregationDefinition"] = _json.dumps(aggregation_definition)

    #if where_clause is not None:
    #    template_dict["rasterFunctionArguments"]["WhereClause"] = where_clause

    return _clone_layer(layer, template_dict, raster_ra)


def compute_change(raster1,
                   raster2,
                   method=0,
                   from_class_values=[],
                   to_class_values=[],
                   filter_method=1,
                   define_transition_colors=0,
                   extent_type="IntersectionOf", 
                   cellsize_type="MaxOf"):

    """
    Produce raster outputs representing of various changes.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    raster1                                  Required ImageryLayer object. The first raster for compute change function.
                                             To evaluate change from time 1 (earlier) to time 2 (later), enter the time 1 raster here.
    ------------------------------------     --------------------------------------------------------------------
    raster2                                  Required ImageryLayer object. The second raster for compute change function.
                                             To evaluate change from time 1 (earlier) to time 2 (later), enter the time 2 raster.
    ------------------------------------     --------------------------------------------------------------------
    method                                   Optional string. Specifies the method to be used.

                                             - DIFFERENCE : The mathematical difference, or subtraction, between the pixel values in the input rasters will be calculated. This is the default.
                                             - RELATIVE_DIFFERENCE : The difference in pixel values, accounting for the magnitudes of the values being compared, will be calculated.
                                             - CATEGORICAL_DIFFERENCE : The difference between two categorical or thematic rasters will be calculated, where the output contains class transitions that occurred between the two rasters.

                                             Example:
                                                  "DIFFERENCE"
    ------------------------------------     --------------------------------------------------------------------
    from_class_values                        Optional list. List of integer values corresponding to the ClassValue 
                                             field in raster1. 
                                             Required if method is CATEGORICAL_DIFFERENCE.
    ------------------------------------     --------------------------------------------------------------------
    to_class_values                          Optional list. List of integer values corresponding to the ClassValue 
                                             field in raster2. 
                                             Required if method is CATEGORICAL_DIFFERENCE.
    ------------------------------------     --------------------------------------------------------------------
    filter_method                            Optional string. Default value is "CHANGED_PIXELS_ONLY" (1).
                                             
                                             Possible options are:

                                             - ALL
                                             - CHANGED_PIXELS_ONLY
                                             - UNCHANGED_PIXELS_ONLY
    ------------------------------------     --------------------------------------------------------------------
    define_transition_colors                 Optional string. Defines the method used to assign color for the output classes

                                             - AVERAGE : The color of the pixel will be the average of the color of its original class and the color of its final class.
                                             - FROM_COLOR : The color of the pixel will be the color of its original class.
                                             - TO_COLOR : The color of the pixel will be the color of its final class.
    ------------------------------------     --------------------------------------------------------------------
    extent_type                              Optional string.  One of "FirstOf", "IntersectionOf" "UnionOf", "LastOf"
    ------------------------------------     --------------------------------------------------------------------
    cellsize_type                            Optional string. One of "FirstOf", "MinOf", "MaxOf "MeanOf", "LastOf"
    ====================================     ====================================================================

    :return: Imagery layer

    """

    layer1, raster_1, raster_ra1 = _raster_input(raster1)
    layer2, raster_2, raster_ra2 = _raster_input(raster1, raster2)

    if layer1 is not None and (layer2 is None or ((layer2 is not None) and layer2._datastore_raster is False)):
        layer = layer1
    else:
        layer = layer2
    #layer = layer1 if layer1 is not None else layer2

    method_types = {
        'DIFFERENCE': 0,
        'RELATIVE_DIFFERENCE': 1,
        'CATEGORICAL_DIFFERENCE' : 2
    }

    if isinstance(method, str):
        method_allowed_values = ['DIFFERENCE', 'RELATIVE_DIFFERENCE', 'CATEGORICAL_DIFFERENCE']
        if [element.upper() for element in method_allowed_values].count(method.upper()) <= 0 :
            raise RuntimeError('method can only be one of the following: '+str(method_allowed_values))
        in_method = method_types[method.upper()]
    else:
        in_method = method

    template_dict = {
        "rasterFunction": "ComputeChange",
        "rasterFunctionArguments": {
            "Method": in_method,
            "Raster": raster_1,
            "Raster2": raster_2
        }
    }

    if from_class_values is not None:
        template_dict["rasterFunctionArguments"]['FromClassValues'] = from_class_values

    if to_class_values is not None:
        template_dict["rasterFunctionArguments"]['ToClassValues'] = to_class_values


    filter_method_types = {
            'ALL': 0,
            'CHANGED_PIXELS_ONLY': 1,
            'UNCHANGED_PIXELS_ONLY' : 2
        }

    if isinstance(filter_method, str):
        filter_method_allowed_values = ['ALL', 'CHANGED_PIXELS_ONLY', 'UNCHANGED_PIXELS_ONLY']
        if [element.upper() for element in filter_method_allowed_values].count(filter_method.upper()) <= 0 :
            raise RuntimeError('method can only be one of the following: '+str(filter_method_allowed_values))
        in_filter_method = filter_method_types[filter_method.upper()]
    else:
        in_filter_method = filter_method

    template_dict["rasterFunctionArguments"]['KeepMethod'] = in_filter_method


    if define_transition_colors is not None:
        define_transition_colors_types = {
        'AVERAGE': 0,
        'FROM_COLOR': 1,
        'TO_COLOR' : 2
    }

        if isinstance(define_transition_colors, str):
            define_transition_colors_allowed_values = ['AVERAGE', 'FROM_COLOR', 'TO_COLOR']
            if [element.upper() for element in define_transition_colors_allowed_values].count(define_transition_colors.upper()) <= 0 :
                raise RuntimeError('method can only be one of the following: '+str(define_transition_colors_allowed_values))
            define_transition_colors = define_transition_colors_types[define_transition_colors.upper()]
        else:
            define_transition_colors = define_transition_colors
        template_dict["rasterFunctionArguments"]['UseColorMethod'] = define_transition_colors

    extent_types = {
        "FIRSTOF": 0,
        "INTERSECTIONOF": 1,
        "UNIONOF": 2,
        "LASTOF": 3
    }

    cellsize_types = {
        "FIRSTOF": 0,
        "MINOF": 1,
        "MAXOF": 2,
        "MEANOF": 3,
        "LASTOF": 4
    }

    if extent_type.upper() not in extent_types.keys():
        raise ValueError(
            "Invalid extent_type value. Note that operation type should be one fo 'FirstOf', 'IntersectionOf', 'UnionOf', 'LastOf'")
    in_extent_type = extent_types[extent_type.upper()]

    if cellsize_type.upper() not in cellsize_types.keys():
        raise ValueError(
            "Invalid extent_type value. Note that operation type should be one fo 'FirstOf', 'MinOf', 'MaxOf', 'MeanOf', 'LastOf'")
    in_cellsize_type = cellsize_types[cellsize_type.upper()]

    if in_extent_type is not None:
        template_dict["rasterFunctionArguments"]['ExtentType'] = in_extent_type
    if in_cellsize_type is not None:
        template_dict["rasterFunctionArguments"]['CellsizeType'] = in_cellsize_type

    return _clone_layer(layer, template_dict, raster_ra1, raster_ra2)

def detect_change_using_change_analysis_raster(raster, 
                                               change_type="TIME_OF_LATEST_CHANGE", 
                                               max_number_of_changes=1,
                                               segment_date="BEGINNING_OF_SEGMENT",
                                               change_direction="ALL",
                                               filter_by_year=False,
                                               min_year=None,
                                               max_year=None,
                                               filter_by_duration=False,
                                               min_duration=None,
                                               max_duration=None,
                                               filter_by_magnitude=False,
                                               min_magnitude=None,
                                               max_magnitude=None,
                                               filter_by_start_value=False,
                                               min_start_value=None,
                                               max_start_value=None,
                                               filter_by_end_value=False,
                                               min_end_value=None,
                                               max_end_value=None):

    """
    Function generates a raster containing pixel change information using the 
    output change analysis raster from the arcgis.raster.analytics.analyze_changes_using_ccdc function
    or arcgis.raster.analytics.analyze_changes_using_landtrendr function.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    raster                                   Required ImageryLayer object. The raster generated from the analyze_changes_using_ccdc or analyze_changes_using_landtrendr.
    ------------------------------------     --------------------------------------------------------------------
    change_type                              Optional String. Specifies the change information to calculate.

                                                - TIME_OF_LATEST_CHANGE (0) - Each pixel will contain the date of the most recent change for that pixel in the time series.
                                                - TIME_OF_EARLIEST_CHANGE (1) - Each pixel will contain the date of the earliest change for that pixel in the time series.
                                                - TIME_OF_LARGEST_CHANGE (2) - Each pixel will contain the date of the most significant change for that pixel in the time series.
                                                - NUM_OF_CHANGES (3) - Each pixel will contain the total number of times the pixel changed in the time series.
                                                - TIME_OF_LONGEST_CHANGE (4) - Each pixel will contain the date of change at the end of the longest transition segment in the time series. Option available in ArcGIS Image Server 10.9 and higher.
                                                - TIME_OF_SHORTEST_CHANGE (5) - Each pixel will contain the date of change at the end of the shortest transition segment in the time series. Option available in ArcGIS Image Server 10.9 and higher.
                                                - TIME_OF_FASTEST_CHANGE (6) - Each pixel will contain the date of change at the end of the transition that occurred most quickly. Option available in ArcGIS Image Server 10.9 and higher.
                                                - TIME_OF_SLOWEST_CHANGE (7) - Each pixel will contain the date of change at the end of the transition that occurred most slowly. Option available in ArcGIS Image Server 10.9 and higher.

                                             Example:
                                                "TIME_OF_LATEST_CHANGE"
    ------------------------------------     --------------------------------------------------------------------
    max_number_of_changes                    Optional Integer. The maximum number of changes per pixel that will 
                                             be calculated. This number corresponds to the number of bands in the output raster. 
                                             The default is 1, meaning only one change date will be calculated, 
                                             and the output raster will contain only one band.

                                             This parameter is not available when the change_type parameter is set to NUM_OF_CHANGES.

                                             Example:
                                                3
    ------------------------------------     --------------------------------------------------------------------
    segment_date                             Optional string. Specifies whether to extract the date at the beginning 
                                             of a change segment, or the end. 

                                             This parameter is available only when the input change analysis raster is 
                                             the output from the arcgis.raster.analytics.analyze_changes_using_landtrendr function.

                                                - BEGINNING_OF_SEGMENT - Extract the date at the beginning of a change segment. This is the default.
                                                - END_OF_SEGMENT - Extract the date at the end of a change segment. 

                                             Example:
                                                "END_OF_SEGMENT"

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    change_direction                         Optional string. The direction of change to be included in the analysis. 
                                             For example, choose Increasing to only extract date of change information for 
                                             periods where the change is in the positive or increasing direction. 

                                             This parameter is available only when the input change analysis raster 
                                             is the output from the analyze_changes_using_landtrendr function.

                                                - ALL - All change directions will be included in the output. This is the default. 
                                                - INCREASE - Only change in the positive or increasing direction will be included in the output. 
                                                - DECREASE - Only change in the negative or decreasing direction will be included in the output. 

                                             Example:
                                                "DECREASE"
                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_year                           Optional boolean. Specifies whether to filter by a range of years. 

                                                - True - Filter results such that only changes that occurred within a specific range of years is included in the output. 
                                                - False - Do not filter results by year. This is the default.

                                             Example:
                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_year                                 Optional int. The earliest year to use to filter results. This parameter 
                                             is required if the filter_by_year parameter is set to True. 

                                             Example:
                                                2000

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_year                                 Optional int. The latest year to use to filter results. This parameter 
                                             is required if the filter_by_year parameter is set to True. 

                                             Example:
                                                2005

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_duration                       Optional boolean. Specifies whether to filter by the change duration. 
                                             This parameter is available only when the input change analysis raster 
                                             is the output from the analyze_changes_using_landtrendr function.

                                                - True - Filter results by duration such that only the changes that lasted a given amount of time will be included in the output.
                                                - False - Do not filter results by duration. This is the default. 

                                             Example:
                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_duration                             Optional float. The minimum number of consecutive years to include in 
                                             the results. This parameter is required if the filter_by_duration parameter 
                                             is set to True 

                                             Example:
                                                2

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_duration                             Optional float. The maximum number of consecutive years to include 
                                             in the results. This parameter is required if the filter_by_duration 
                                             parameter is set to True 

                                             Example:
                                                4

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_magnitude                      Optional boolean. Specifies whether to filter by change magnitude. 

                                                - True - Filter results by magnitude such that only the changes of a given magnitude will be included in the output.
                                                - False - Do not filter results by magnitude. This is the default. 

                                             Example:
                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_magnitude                            Optional float. The minimum magnitude to include in the results. 
                                             This parameter is required if the filter_by_magnitude 
                                             parameter is set to True. 

                                             Example:
                                                0.25

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_magnitude                            Optional float. The maximum magnitude to include in the results. 
                                             This parameter is required if the filter_by_magnitude parameter is set 
                                             to True.

                                             Example:
                                                3

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_start_value                    Optional boolean. Specifies whether to filter by start value. This 
                                             parameter is available only when the input change analysis raster 
                                             is the output from the arcgis.raster.analytics.analyze_changes_using_landtrendr function. 

                                                - True - Filter results by start value so that only the change that starts with value defined by a range.
                                                - False - Do not filter by start value. This is the default.

                                             Example:
                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_start_value                          Optional float. The minimum value that defines the range of start value. 
                                             This parameter is required if the filter_by_start_value parameter is set 
                                             to True.

                                             Example:
                                                0.75

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_start_value                          Optional float. The maximum value that defines the range of start value. 
                                             This parameter is required if the filter_by_start_value parameter is 
                                             set to True.

                                             Example:
                                                0.9

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_end_value                      Optional boolean. Specifies whether to filter by end value. This parameter 
                                             is available only when the input change analysis raster is the output 
                                             from the arcgis.raster.analytics.analyze_changes_using_landtrendr function.

                                                - True - Filter results by end value so that only the change that ends with value defined by a range.
                                                - False - Do not filter results by end value. This is the default. 

                                             Example:
                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_end_value                            Optional float. The minimum value that defines the range of end value. 
                                             This parameter is required if the filter_by_end_value parameter is set 
                                             to True.

                                             Example:
                                                -0.12

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_end_value                            Optional float. The maximum value that defines the range of end value. 
                                             This parameter is required if the filter_by_end_value parameter is set 
                                             to True.

                                             Example:
                                                0.35

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ====================================     ====================================================================

    :return: Imagery layer

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "DetectChange",
        "rasterFunctionArguments": {
            "Raster" : raster,
        },
        "variableName": "Raster"
    }


    if change_type is not None:
        change_types = {
            'TIME_OF_LATEST_CHANGE': 0,
            'TIME_OF_EARLIEST_CHANGE': 1,
            'TIME_OF_LARGEST_CHANGE' : 2,
            'NUM_OF_CHANGES' : 3,
            'TIME_OF_LONGEST_CHANGE' : 4,
            'TIME_OF_SHORTEST_CHANGE' : 5,
            'TIME_OF_FASTEST_CHANGE' : 6,
            'TIME_OF_SLOWEST_CHANGE' : 7
        }

        if isinstance(change_type, str):
            in_change_type = change_types[change_type.upper()]
        else:
            in_change_type = change_type

        template_dict["rasterFunctionArguments"]['ChangeType'] = in_change_type

    if max_number_of_changes is not None:
        template_dict["rasterFunctionArguments"]['MaxNumberChanges'] = max_number_of_changes

    if segment_date is not None:
        segment_date_types = {
            'BEGINNING_OF_SEGMENT': 0,
            'END_OF_SEGMENT': 1
        }

        if isinstance(segment_date, str):
            in_segment_date = segment_date_types[segment_date.upper()]
        else:
            in_segment_date = segment_date

        template_dict["rasterFunctionArguments"]['SegmentDate'] = in_segment_date

    if change_direction is not None:
        change_direction_types = {
            'ALL': 0,
            'INCREASE': 1,
            'DECREASE':2
        }

        if isinstance(change_direction, str):
            in_change_direction = change_direction_types[change_direction.upper()]
        else:
            in_change_direction = change_direction

        template_dict["rasterFunctionArguments"]['ChangeDirection'] = in_change_direction

    if isinstance(filter_by_year, bool):
        template_dict["rasterFunctionArguments"]['FilterByYear'] = filter_by_year

        if filter_by_year:
            if min_year is not None:
                template_dict["rasterFunctionArguments"]['MinimumYear'] = min_year

            if max_year is not None:
                template_dict["rasterFunctionArguments"]['MaximumYear'] = max_year

    if isinstance(filter_by_duration, bool):
        template_dict["rasterFunctionArguments"]['FilterByDuration'] = filter_by_duration

        if filter_by_duration:
            if min_duration is not None:
                template_dict["rasterFunctionArguments"]['MinimumDuration'] = min_duration

            if max_duration is not None:
                template_dict["rasterFunctionArguments"]['MaximumDuration'] = max_duration

    if isinstance(filter_by_magnitude, bool):
        template_dict["rasterFunctionArguments"]['FilterByMagnitude'] = filter_by_magnitude

        if filter_by_magnitude:
            if min_magnitude is not None:
                template_dict["rasterFunctionArguments"]['MinimumMagnitude'] = min_magnitude

            if max_magnitude is not None:
                template_dict["rasterFunctionArguments"]['MaximumMagnitude'] = max_magnitude

    if isinstance(filter_by_start_value, bool):
        template_dict["rasterFunctionArguments"]['FilterByStartValue'] = filter_by_start_value

        if filter_by_start_value:
            if min_start_value is not None:
                template_dict["rasterFunctionArguments"]['MinimumStartValue'] = min_start_value

            if max_start_value is not None:
                template_dict["rasterFunctionArguments"]['MaximumStartValue'] = max_start_value

    if isinstance(filter_by_end_value, bool):
        template_dict["rasterFunctionArguments"]['FilterByEndValue'] = filter_by_end_value

        if filter_by_end_value:
            if min_end_value is not None:
                template_dict["rasterFunctionArguments"]['MinimumEndValue'] = min_end_value

            if max_end_value is not None:
                template_dict["rasterFunctionArguments"]['MaximumEndValue'] = max_end_value

    return _clone_layer(layer, template_dict, raster_ra)

def trend_to_rgb(raster, 
                 model_type=0):

    """
    Display the generate trend raster.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    raster                                   Required ImageryLayer object. 
    ------------------------------------     --------------------------------------------------------------------
    model_type                               Optional String. Specifies the model type.

                                                - "LINEAR" (0)

                                                - "HARMONIC" (1)

                                             Example:
                                                  "LINEAR"
    ====================================     ====================================================================

    :return: Imagery layer

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "TrendToRGB",
        "rasterFunctionArguments": {
            "Raster" : raster,
        },
        "variableName": "Raster"
    }


    if model_type is not None:
        model_types = {
            'linear': 0,
            'harmonic': 1
        }

        if isinstance(model_type, str):
            in_model_type = model_types[model_type.lower()]
        else:
            in_model_type = model_type

        template_dict["rasterFunctionArguments"]['ModelType'] = in_model_type

    return _clone_layer(layer, template_dict, raster_ra)


def apparent_reflectance(raster, radiance_gain_values = None, radiance_bias_values=None, 
                         reflectance_gain_values=None, reflectance_bias_values=None,
                         sun_elevation=None, albedo=False, scale_factor=None, offset=None):

    """
    Function calibrates the digital number (DN) values of imagery from some satellite 
    sensors. The calibration uses sun elevation, acquisition date, sensor gain and 
    bias for each band to derive Top of Atmosphere reflectance, plus sun angle correction.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    raster                                   Required ImageryLayer object. 
    ------------------------------------     --------------------------------------------------------------------
    radiance_gain_values                     Optional list. The radiance gain values
    ------------------------------------     --------------------------------------------------------------------
    radiance_bias_values                     Optional list. The radiance bias values
    ------------------------------------     --------------------------------------------------------------------
    reflectance_gain_values                  Optional list. The reflectance gain values
    ------------------------------------     --------------------------------------------------------------------
    reflectance_bias_values                  Optional list. The reflectance bias values
    ------------------------------------     --------------------------------------------------------------------
    sun_elevation                            This is sun elevation value, expressed in degrees.
    ------------------------------------     --------------------------------------------------------------------
    albedo                                   Optional bool The results of the Apparent Reflectance function can also 
                                             be expressed as albedo, which is the percentage of the available 
                                             energy reflected by the planetary surface. Albedo data is used 
                                             by scientific users for complex modeling and technical 
                                             remote-sensing applications.

                                                - False - The function returns apparent reflectance values. This is the default.
                                                
                                                - True - The function returns 32-bit floating-point values, which most commonly are in the range of 0.0 to 1.0. No data clipping is performed if this option is selected.
    ------------------------------------     --------------------------------------------------------------------
    scale_factor                             Optional int. Your apparent reflectance output value can be expressed 
                                             as an integer. The scaling factor is multiplied by the albedo to 
                                             convert all floating-point values into integer values.

                                             If the scale factor is either 0 or not specified, default scaling 
                                             will be applied depending on the pixel type of the input data:

                                                 For 16-bit unsigned data types, the default scale factor is 50,000.

                                                 For 8-bit unsigned data types, default scale factor is 255.

                                                 The scaling factor is always applied when the output is apparent reflectance.
                                                 No scaling is applied when the output is albedo.
    ------------------------------------     --------------------------------------------------------------------
    offset                                   Optional int. 	
                                             Your scaled albedo value can optionally have an offset value:

                                             For 16-bit unsigned data types, the default scale offset is 5,000.
                                             For 8-bit unsigned data types, the default scale offset is 0.
                                             No scaling is applied when the output is albedo.
    ====================================     ====================================================================

    :return: Imagery layer

    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Reflectance",
        "rasterFunctionArguments": {
            "Raster" : raster,
        },
        "variableName": "Raster"
    }

    if radiance_gain_values is not None:
        template_dict["rasterFunctionArguments"]['RadianceGainValues'] = radiance_gain_values

    if radiance_bias_values is not None:
        template_dict["rasterFunctionArguments"]['RadianceBiasValues'] = radiance_bias_values

    if reflectance_gain_values is not None:
        template_dict["rasterFunctionArguments"]['ReflectanceGainValues'] = reflectance_gain_values

    if reflectance_bias_values is not None:
        template_dict["rasterFunctionArguments"]['ReflectanceBiasValues'] = reflectance_bias_values

    if sun_elevation is not None:
        template_dict["rasterFunctionArguments"]['SunElevation'] = sun_elevation

    if albedo is not None:
        template_dict["rasterFunctionArguments"]['Albedo'] = albedo

    if scale_factor is not None:
        template_dict["rasterFunctionArguments"]['ScaleFactor'] = scale_factor

    if offset is not None:
        template_dict["rasterFunctionArguments"]['Offset'] = offset

    return _clone_layer(layer, template_dict, raster_ra)

#def radar_calibration(raster, calibration_type=None):

#    """
#    The Radar Calibration function is used to calibrate RADARSAT-2 imagery in a 
#    mosaic dataset or as a raster product. Calibration is performed on radar 
#    imagery so that the pixel values are a true representation of the radar backscatter.

#    :param raster: The input raster.

#    :param calibration_type: Optional string or int. one of four calibration types: 
#                             "beta_nought" (0) - The function returns the radar reflectivity per unit area in slant range.
#                             "sigma_nought" (1) - The function returns the radar reflectivity per unit area in ground range. 
#                                                  Results are 32-bit floating-point values commonly in the range of 0.0 to 
#                                                  1.0. No data clipping is performed if this option is selected.
#                             "gamma" (2) - The function returns the radar reflectivity per unit area in the 
#                                           plane perpendicular to the direction of measurement.
#                              None - Specify None to not apply any calibration. This is the default.

#    :return: output raster 
#    """
#    layer, raster, raster_ra = _raster_input(raster)

#    template_dict = {
#        "rasterFunction" : "RadarCalibration",
#        "rasterFunctionArguments": {
#            "Raster" : raster,            
#        }
#    }
    
#    calibration_type_dict = {"beta_nought":0, "sigma_nought":1, "gamma":2}
#    if calibration_type is not None:
#        if isinstance(calibration_type, str):
#            calibration_type= calibration_type_dict[calibration_type.lower()]
#            template_dict["rasterFunctionArguments"]['CalibrationType'] = calibration_type
#        else:
#            template_dict["rasterFunctionArguments"]['CalibrationType'] = calibration_type
#    else:
#        template_dict["rasterFunctionArguments"]['CalibrationType'] = 3

#    return _clone_layer(layer, template_dict, raster_ra)

def buffered(raster):

    """
    The Buffered function is used to optimize the performance of complex function chains. 
    It stores the output from the part of the function chain that comes before it in memory.

    :param raster: The input raster.

    :return: output raster 
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "BufferedRaster",
        "rasterFunctionArguments": {
            "Raster" : raster,            
        }
    }

    return _clone_layer(layer, template_dict, raster_ra)


def rasterize_features(raster, feature_class,  class_index_field=None, resolve_overlap_method = 0):
    """
    Converts features to raster. Features are assigned pixel values based on the feature's OBJECTID (default). 
    Optionally, the pixel values can be based on a user defined value field in the input feature's attribute table.

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    raster                               Required. The input Imagery Layer object.
    --------------------------------     --------------------------------------------------------------------
    feature_class                        Required. The input point feature layer.
    --------------------------------     --------------------------------------------------------------------
    class_index_field                    Optional string.Select the field to use to identify each feature.
    --------------------------------     --------------------------------------------------------------------
    resolve_overlap_method               Optional int or string. Determine how to manage features that overlap:

                                         - FIRST - The overlapping areas will be assigned a value from the first 
                                           dataset listed.

                                         - LAST - The overlapping areas will be assigned a value from the last 
                                           dataset listed.

                                         - SMALLEST - The overlapping areas will be assigned a value from the 
                                           smaller of the features.

                                         - LARGEST - The overlapping areas will be assigned a value from the 
                                           larger of the features.
    ================================     ====================================================================

    :returns: output raster with function applied

    """
    layer, raster, raster_ra = _raster_input(raster)

    if isinstance(feature_class, _FeatureLayer):
        feature_class = feature_class.url

    template_dict = {
        "rasterFunction" : "RasterizeFeatureClass",
        "rasterFunctionArguments": {
            "Raster" : raster,
        }
    }

    if feature_class is not None:
        template_dict["rasterFunctionArguments"]['FeatureClass'] = feature_class

    if class_index_field is not None:
        template_dict["rasterFunctionArguments"]['ClassIndexField'] = class_index_field
    
    resolve_overlap_method_dict = {"first":0, "last":1, "smallest":2, 'largest':3}
    if resolve_overlap_method is not None:
        if isinstance(resolve_overlap_method, str):
            resolve_overlap_method= resolve_overlap_method_dict[resolve_overlap_method.lower()]
            template_dict["rasterFunctionArguments"]['ResolveOverlapMethod'] = resolve_overlap_method
        else:
            template_dict["rasterFunctionArguments"]['ResolveOverlapMethod'] = resolve_overlap_method

    return _clone_layer(layer, template_dict, raster_ra)


#def swath(raster, interpolation_method = 0, cell_size=0):
#    """
#    Interpolates from point clouds or irregular grids. (Supported from 10.8.1)

#    ================================     ====================================================================
#    **Argument**                         **Description**
#    --------------------------------     --------------------------------------------------------------------
#    raster                               Required Imagery Layer object. 
#    --------------------------------     --------------------------------------------------------------------
#    interpolation_method                 Optional int or string. The resampling method to use for interpolation:

#                                          -  0 or NEAREST_NEIGBOR : Calculates pixel value using the nearest pixel. If no source pixel 
#                                             exists, no new pixel can be created in the output. This is the default.

#                                          -  1 or BILINEAR : Calculates pixel value using the distance-weighted 
#                                             value of four nearest pixels.

#                                          -  2 or LINEAR_TINNING : Uses a triangular irregular network from the center 
#                                             points of each cell in the irregular raster to interpolate a surface 
#                                             that is then converted to a regular raster.

#                                          -  3 or NATURAL_NEIGHBOR : Finds the closest subset of input samples to a 
#                                             query point and applies weights to them based on proportionate 
#                                             areas to interpolate a value.
#    --------------------------------     --------------------------------------------------------------------
#    cell_size                            Optional int. The cell size for the output raster dataset. 
#    ================================     ====================================================================

#    :returns: output raster with function applied

#    """
#    layer, raster, raster_ra = _raster_input(raster)

#    template_dict = {
#        "rasterFunction" : "Swath",
#        "rasterFunctionArguments": {"RasterInfo":{"blockWidth" : 2048,
#                                                "blockHeight" : 256,
#                                                "bandCount" : 0,
#                                                "pixelType" : -1,
#                                                "pixelSizeX" : cell_size,
#                                                "pixelSizeY" : cell_size,
#                                                "format" : "",
#                                                "compressionType" : "",
#                                                "firstPyramidLevel" : 1,
#                                                "maximumPyramidLevel" : 30,
#                                                "packetSize" : 4,
#                                                "compressionQuality" : 0,
#                                                "pyramidResamplingType" : -1,
#                                                "type" : "RasterInfo"}
#                                 }
#                 }

#    interpolation_methods = {
#        'NEAREST_NEIGBOR': 0,
#        'BILINEAR':1,
#        'LINEAR_TINNING': 2,
#        'NATURAL_NEIGHBOR' : 3
#    }


#    if interpolation_method is not None:
#        if isinstance(interpolation_method, str):
#            interpolation_method = interpolation_methods[interpolation_method.upper()]
#        else:
#            interpolation_method = interpolation_method
#        template_dict["rasterFunctionArguments"]["InterpolationMethod"] = interpolation_method

#    return _clone_layer(layer, template_dict, raster_ra)


def reproject(raster, spatial_reference=None, x_cell_size=0, y_cell_size=0, x_registration_point=0, y_registration_point=0):
    """
    Modifies the projection of a raster dataset, mosaic dataset, or raster item in a 
    mosaic dataset. It can also resample the data to a new cell size and define an origin.

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    raster                               Required Imagery Layer object. The raster dataset to be reprojected or resampled.
    --------------------------------     --------------------------------------------------------------------
    spatial_reference                    Optional dict. The coordinate system used to reproject the data.

                                         Example:
                                            | {
                                            |    "wkid" : 4176,
                                            |    "latestWkid" : 4176
                                            | }
    --------------------------------     --------------------------------------------------------------------
    x_cell_size                          Optional.The x-dimension to which the data should be resampled. 
                                         This is optional. If the value is 0 or less, the output envelope 
                                         (extent and cell sizes) is calculated from the input raster.
    --------------------------------     --------------------------------------------------------------------
    y_cell_size                          Optional.The y-dimension to which the data should be resampled. 
                                         This is optional. If the value is 0 or less, the output envelope 
                                         (extent and cell sizes) is calculated from the input raster.
    --------------------------------     --------------------------------------------------------------------
    x_registration_point                 Optional. The x-coordinate used to define the upper left corner 
                                         of the dataset. This coordinate must be defined in the units of 
                                         the new spatial reference. If both the x_cell_size and y_cell_size 
                                         parameters are greater than 0,  they are used along with the
                                         x_registration_point and y_registration_point
                                         parameters to define the output envelope.
    --------------------------------     --------------------------------------------------------------------
    y_registration_point                 Optional. The y-coordinate used to define the upper left corner of the dataset. 
                                         This coordinate must be defined in the units of the new spatial reference. 
                                         If both the x_cell_size and y_cell_size parameters are greater than 0, 
                                         they are used along with the x_registration_point and y_registration_point
                                         parameters to define the output envelope.
    ================================     ====================================================================

    :returns: output raster with function applied

    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction" : "Reproject",
        "rasterFunctionArguments": {
            "Raster" : raster,
        }
    }


    if spatial_reference is not None:
        template_dict["rasterFunctionArguments"]['SpatialReference'] = spatial_reference

    if x_cell_size is not None:
        template_dict["rasterFunctionArguments"]['XCellsize'] = x_cell_size

    if y_cell_size is not None:
        template_dict["rasterFunctionArguments"]['YCellsize'] = y_cell_size

    if x_registration_point is not None:
        template_dict["rasterFunctionArguments"]['XOrigin'] = x_registration_point

    if y_registration_point is not None:
        template_dict["rasterFunctionArguments"]['YOrigin'] = y_registration_point


    return _clone_layer(layer, template_dict, raster_ra)


def heat_index(temperature_raster, relative_humidity_raster, temperature_units="Fahrenheit",
              heat_index_units="Fahrenheit"):
    """
    Calculates apparent temperature based on ambient temperature and relative humidity. The apparent temperature is often described as how hot it feels to the human body.
    
    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    temperature_raster                   The raster that represent temperature. A single-band raster where pixel 
                                         values represent ambient air temperature.
    --------------------------------     --------------------------------------------------------------------
    relative_humidity_raster             The raster that represent relative humidity. A single-band raster 
                                         where pixel values represent relative humidity as a percentage value between 0 and 100.
    --------------------------------     --------------------------------------------------------------------
    temperature_units                    Optional String. The unit of measurement associated with the input 
                                         temperature raster. Available input units are Celsius, Fahrenheit, and Kelvin.
    --------------------------------     --------------------------------------------------------------------
    heat_index_units                     Optional String. The unit of measurement associated with the output raster. 
                                         Available output units are Celsius, Fahrenheit, and Kelvin.
    ================================     ====================================================================

    :return: the output raster
    """

    layer1, raster_1, raster_ra1 = _raster_input(temperature_raster)
    layer2, raster_2, raster_ra2 = _raster_input(temperature_raster, relative_humidity_raster)

    if layer1 is not None and (layer2 is None or ((layer2 is not None) and layer2._datastore_raster is False)):
        layer = layer1
    else:
        layer = layer2

    template_dict = {
        "rasterFunction": "PythonAdapter",
        "rasterFunctionArguments": {"temperature": raster_1,
            "rh": raster_2}
    }
    if temperature_units is not None:
        template_dict["rasterFunctionArguments"]["units"] = temperature_units
    if heat_index_units is not None:
        template_dict["rasterFunctionArguments"]["outUnits"] = heat_index_units

    template_dict["rasterFunctionArguments"]["ClassName"] = "HeatIndex"
    template_dict["rasterFunctionArguments"]["PythonModule"] = "[functions]System\\HeatIndex.py"
    #template_dict["rasterFunctionArguments"]["MatchVariable"] = False

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']['temperature'] = raster_ra1
    function_chain_ra['rasterFunctionArguments']['rh'] = raster_ra2

    return _clone_layer_without_copy(layer, template_dict, function_chain_ra)

def wind_chill(temperature_raster, wind_speed_raster, temperature_units="Fahrenheit", wind_speed_units="mph",
              wind_chill_units="Fahrenheit"):

    """
    The Wind Chill function is useful for identifying dangerous winter conditions that, depending on exposure times to
    the elements, can result in frostbite or even hypothermia. Wind chill is a way to measure how cold an individual
    feels when wind is taken into account with already cold temperatures. The faster the wind speed, the more 
    quickly the body will lose heat and the colder they will feel.

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    temperature_raster                   The raster that represent temperature. A single-band raster where pixel 
                                         values represent ambient air temperature.
    --------------------------------     --------------------------------------------------------------------
    wind_speed_raster                    The raster that represent relative humidity. A single-band raster 
                                         where pixel values represent wind speed.
    --------------------------------     --------------------------------------------------------------------
    temperature_units                    Optional String. The unit of measurement associated with the input 
                                         temperature raster. Available input units are Celsius, Fahrenheit, and Kelvin.
    --------------------------------     --------------------------------------------------------------------
    wind_speed_units                     Optional String.  Defines the unit of measurement for the wind-speed raster. 
                                         Available input units are mph, km/h, ft/s and kn. Each represents

                                           - Miles Per Hour (mph)
                                           - Kilometers Per Hour (km/h)
                                           - Meters Per Second (m/s)
                                           - Feet Per Second (ft/s)
                                           - Knots (kn)
    --------------------------------     --------------------------------------------------------------------
    wind_chill_units                     Optional String. The unit of measurement associated with the output 
                                         raster. Available output units are Celsius, Fahrenheit, and Kelvin.
    ================================     ====================================================================

    :return: the output raster
    """

    layer1, raster_1, raster_ra1 = _raster_input(temperature_raster)
    layer2, raster_2, raster_ra2 = _raster_input(temperature_raster, wind_speed_raster)

    if layer1 is not None and (layer2 is None or ((layer2 is not None) and layer2._datastore_raster is False)):
        layer = layer1
    else:
        layer = layer2


    template_dict = {
        "rasterFunction": "PythonAdapter",
        "rasterFunctionArguments": {"temperature": raster_1, "ws": raster_2}
    }

    if temperature_units is not None:
        template_dict["rasterFunctionArguments"]["tunits"] = temperature_units
    if wind_speed_units is not None:
        template_dict["rasterFunctionArguments"]["wunits"] = wind_speed_units
    if wind_chill_units is not None:
        template_dict["rasterFunctionArguments"]["ounits"] = wind_chill_units

    template_dict["rasterFunctionArguments"]["ClassName"] = "Windchill"
    template_dict["rasterFunctionArguments"]["PythonModule"] = "[functions]System\\Windchill.py"
    #template_dict["rasterFunctionArguments"]["MatchVariable"] = False

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']['temperature'] = raster_ra1
    function_chain_ra['rasterFunctionArguments']['ws'] = raster_ra2

    return _clone_layer_without_copy(layer, template_dict, function_chain_ra)

def aspect_slope(raster, z_factor=1):
    """
    Creates a raster layer that simultaneously displays the aspect and slope of a surface.

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    raster                               The input raster.
    --------------------------------     --------------------------------------------------------------------
    z_factor                             A multiplication factor that converts the vertical (elevation) values 
                                         to the linear units of the horizontal (x,y) coordinate system. 
                                         "Use larger values to add vertical exaggeration.
    ================================     ====================================================================

    :return: the output raster
    """

    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "PythonAdapter",
        "rasterFunctionArguments": {"Raster" : raster}
    }

    template_dict['rasterFunctionArguments']['zf'] = z_factor

    template_dict["rasterFunctionArguments"]["ClassName"] = "AspectSlope"
    template_dict["rasterFunctionArguments"]["PythonModule"] = "[functions]System\\AspectSlope.py"

    return _clone_layer(layer, template_dict, raster_ra)

def contour(raster, adaptive_smoothing=2.5, contour_type="CONTOUR_LINES", z_base=0, number_of_contours=0,
            contour_interval=100, nth_contour_line_in_bold=5, z_factor=1):
    """
    Creates contour lines.

    :param raster: the raster from which the contour is created
    :param adaptive_smoothing(double): adaptive smooting value, e.g., 2.5
    :param contour_type(string): contour type. Available values could be "CONTOUR_LINES", "CONTOUR_FILL" or "SMOOTH_SURFACE_ONLY"
    :param z_base(double): the z-base value, e.g., 0
    :param number_of_contours(int): the number of contours, e.g., 0
    :param contour_interval(double): the contour interval, e.g., 100
    :param nth_contour_line_in_bold(int): the nth contour line that would be rendered in bold, e.g., 5
    :param z_factor(double): the z-factor, e.g., 1
    :return:
    """
    layer, raster, raster_ra = _raster_input(raster)

    template_dict = {
        "rasterFunction": "Contour",
        "rasterFunctionArguments": {"Raster" : raster}
    }

    contour_types = {
        'CONTOUR_LINES': 0,
        'CONTOUR_FILL': 1,
        'SMOOTH_SURFACE_ONLY': 2,
    }

    if adaptive_smoothing is not None:
        template_dict['rasterFunctionArguments']['SigmaGaussian'] = adaptive_smoothing
    if contour_type is not None:
        template_dict['rasterFunctionArguments']['ContourType'] = contour_types[contour_type.upper()]
    if z_base is not None:
        template_dict['rasterFunctionArguments']['ZBase'] = z_base
    if number_of_contours is not None:
        template_dict['rasterFunctionArguments']['NumberOfContours'] = number_of_contours
    if contour_interval is not None:
        template_dict['rasterFunctionArguments']['ContourInterval'] = contour_interval
    if nth_contour_line_in_bold is not None:
        template_dict['rasterFunctionArguments']['NthContourLineInBold'] = nth_contour_line_in_bold
    if z_factor is not None:
        template_dict['rasterFunctionArguments']['ZFactor'] = z_factor

    return _clone_layer(layer, template_dict, raster_ra)

class RFT:
    def __init__(self, raster_function_template,gis=None):
        try:
            self._is_public_flag = False
            self._rft=raster_function_template
            self._gis = _arcgis.env.active_gis if gis is None else gis
            key_value_dict={}
            if(".rft.xml" in self._rft.name):
                _rft_json = self.to_json(self._gis)
            else:
                file_path = self._rft.get_data()
                if isinstance(file_path, dict):
                    _rft_json=file_path
                else:
                    f=open(file_path, "r")
                    file_content = f.read()
                    file_content = file_content.replace("false", "False")
                    file_content = file_content.replace("true", "True")
                    _rft_json = eval(file_content)
            self._rft_json = _find_object_ref(_rft_json, {}, self)
            global node, end_node 
            node = 0
            end_node=0
            self._rft_dict, self._raster_dict = self._find_arguments_()

            if self._rft_dict.keys() & hidden_inputs:
                for key in hidden_inputs:
                    self._rft_dict.pop(key,None)
            self._arguments=copy.deepcopy(self._rft_dict)
            self.arguments = copy.deepcopy(self._arguments)
            self._local_raster=None
        except:
            _LOGGER.warning("Unable to find the arguments for the current raster function template. "
                  "This might be because the server could not process the template "
                  "or the template is invalid."
                  "(Ensure that the user account has access to Raster Utilities of the server. "
                  "To share the Raster utilities to all user accounts. Please refer Sharing Raster Utilities section in "
                  "https://esri.github.io/arcgis-python-api/apidoc/html/arcgis.raster.functions.RFT.html )")

    @property
    def __doc__(self):
        tab_value =-1
        help="\"\"\"\n"
        if("description" in self._rft_json):
            help=help+self._rft_json["description"]
        else:
            help=help+self._rft_json["function"]["description"]
        help = help+"\n\nParameters\n----------\n"
        for key,value in self._rft_dict.items():
            help=help+"\n"+(str(key)+" : "+str(value))

        help=help+("\n\nReturns\n-------\n")
        help=help+("Imagery Layer, on which the function chain is applied \n")
        help=help+"\"\"\""
        print(help)

    @property
    def __signature__(self):
        from inspect import Signature, Parameter
        signature_list=[]
        for name,value in self.arguments.items():
            signature_list.append(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=value))
        sig = Signature(signature_list)
        return sig


    @property
    def template(self):
        if self._template is None:
            self._template=self._rft_json
        return self._template

    def __call__(self,*args,**kwargs):
        try:
            i=0
            key_list=list(self._arguments.keys())
            for pos_arg in args:
                kwargs.update({key_list[i]:pos_arg})
                i=i+1

            for key in kwargs.keys():
                if isinstance(kwargs[key], Raster):
                    if(self._local_raster is None):
                        self._local_raster = kwargs[key]._engine_obj
                        break
            
            if(len(kwargs)==1):
                for k,v in self._raster_dict.items():
                    self._raster_dict.update({k:kwargs[list(kwargs.keys())[0]]})	
                    layer = self._apply_rft(self._raster_dict, self._gis)
                if isinstance(layer, Raster):
                    self._template = layer._engine_obj._fnra
                    if layer._engine_obj.url is None:
                        return None
                    if layer._engine==_ArcpyRaster:
                        return _clone_layer_raster_without_copy(layer._engine_obj, self._template, self._template)

                else:
                    self._template = layer._fnra
                    if layer.url is None:
                        return None
                return layer


            for key in kwargs.keys():
                for k in self._arguments.keys():
                    if(k==key):
                        self._arguments[k]=kwargs[key]
            layer = self._apply_rft(self._arguments, self._gis)
            if isinstance(layer, Raster):
                self._template = layer._engine_obj._fnra
                if layer._engine_obj.url is None:
                    return None
                if layer._engine==_ArcpyRaster:
                    return _clone_layer_raster_without_copy(layer._engine_obj, self._template, self._template)

            else:
                self._template = layer._fnra
                if layer.url is None:
                    return None
            return layer

        except:
            _LOGGER.warning("Unable to apply the current raster function template on the imagery layer. " 
                  "This might be because the server could not process the template, "
                  "the template is invalid or not populated with correct arguments. "
                  "(Make sure that Raster rendering service is turned on, inorder to display the output dynamically.)")

        
    def to_json(self, gis =None):
        """
        Converts the raster function template into a dictionary.

        =================     ====================================================================
        **Argument**          **Description**
        -----------------     --------------------------------------------------------------------
        gis                   optional, GIS on which the RFT object is based on. 
        =================     ====================================================================

        :return: dictionary
        """
        task = "ConvertRasterFunctionTemplate"
        gis = _arcgis.env.active_gis if gis is None else gis
        url = gis.properties.helperServices.rasterUtilities.url

        gptool = _arcgis.gis._GISResource(url, gis)
        params = {}
        params["inputRasterFunction"] = {"itemId": self._rft.itemid}
        params["outputFormat"]="json"
        task_url, job_info, job_id = _analysis_job(gptool, task, params)
        job_info = _analysis_job_status(gptool, task_url, job_info)
        job_values = _analysis_job_results(gptool, task_url, job_info)
        if gis._con._product == "AGOL":
            result = gptool._con.get(job_values["outputRasterFunction"]["url"],{},token=gptool._token)
        else:
            result = gptool._con.post(job_values["outputRasterFunction"]["url"],{},token=gptool._token)
        return result

    def _apply_argument(self, input_dict,arg_dict):
        if "arguments" in input_dict.keys():
            if "isDataset" in input_dict["arguments"].keys():
                if(input_dict["arguments"]["isDataset"] == False):
                    if(("value" in input_dict["arguments"]) and "elements" in input_dict["arguments"]["value"]):
                        for arg_element in input_dict["arguments"]["value"]["elements"]:
                            self._apply_argument(arg_element["arguments"],arg_dict)
                else:
                    if(("value" in input_dict["arguments"]) and "arguments" in input_dict["arguments"]["value"]):
                        self._apply_argument(input_dict["arguments"]["value"]["arguments"],arg_dict)
            self._apply_argument(input_dict["arguments"],arg_dict)
        flag_rasters = -1
        for key, value in input_dict.items():
            if isinstance(value, dict):
                if (("type" in value) and value["type"]=="RasterFunctionTemplate") and "arguments" in value.keys():
                    if "isDataset" in value["arguments"].keys():
                        if(value["arguments"]["isDataset"] == False):
                            for arg_element in value["arguments"]["value"]["elements"]:
                                self._apply_argument(arg_element["arguments"],arg_dict)
                        else:
                            if "arguments" in value["arguments"]["value"]:
                                self._apply_argument(value["arguments"]["value"]["arguments"],arg_dict)
                    self._apply_argument(value["arguments"],arg_dict)
                    flag_rasters = 1
                if(("type" in value) and value["type"]=="RasterFunctionVariable"):
                    for k,v in arg_dict.items():
                        if(value["name"]==k):
                            if isinstance(v,(ImageryLayer, Raster)) or isinstance(v, _FeatureLayer):
                                raster = _raster_input_rft(v)
                                v =_input_rft(raster)
                                if isinstance(raster,str):
                                    value["value"]=v
                                    flag_rasters=1
                                elif isinstance(raster,dict):
                                    if raster.keys() & {"mosaicRule"}:
                                        value["value"]=v
                                    else:
                                        input_dict.update({key:v})
                                break
                            else:
                                if("value" in value):
                                    if isinstance(value["value"],dict):
                                        if "type" in value["value"]:
                                            if value["value"]["type"]=="Scalar":
                                                value["value"]={"type":"Scalar","value":v}
                                                break
                                            elif value["value"]["type"]=="RasterDatasetName":
                                                value["value"]=v
                                                break
                                                
                                        else:
                                            value["value"]=v
                                            break
                                    else:
                                        value["value"]=v
                                        break
                                else:
                                    value["value"]=v
                                    if((key=="RasterInfo")) and isinstance(v, dict):
                                        v.update({"type":"RasterInfo"})
                                    if (isinstance(value["value"], numbers.Number) and value["isDataset"]==True):
                                        value["value"]={"type":"Scalar","value":v}
                                        break

                            if "name" in value and "value" in value:
                                if isinstance(value["value"], dict):
                                    if("elements" in value["value"].keys()):
                                        raster = _raster_input_rft(v)
                                        v =_input_rft(raster)
                                        if isinstance(raster,list):
                                            value["value"]=v
                                            flag_rasters=1
                                            break
                            else:
                                if("value" in value):
                                    if isinstance(value["value"],dict):
                                        if "type" in value["value"]:
                                            if value["value"]["type"]=="Scalar":
                                                value["value"]={"type":"Scalar","value":v}
                                                break
                                            elif value["value"]["type"]=="RasterDatasetName":
                                                value["value"]=v
                                                break
                                        else:
                                            value["value"]=v
                                            break
                                    else:
                                        value["value"]=v
                                        break
                                else:
                                    value["value"]=v
                                    if isinstance(value["value"], numbers.Number) and value["isDataset"]==True:
                                        value["value"]={"type":"Scalar","value":v}
                                        break
                    if(flag_rasters==-1) and "Rasters" in input_dict.keys():
                        elements_structure = []
                        if (isinstance (input_dict["Rasters"]["value"], dict)) and "elements" in input_dict["Rasters"]["value"]:
                            elements_structure = input_dict["Rasters"]["value"]["elements"]
                        elif isinstance(input_dict["Rasters"]["value"],list):
                            elements_structure = input_dict["Rasters"]["value"]
                        for element in elements_structure:
                            if isinstance(element,dict):
                                if (("type" in element) and element["type"]=="RasterFunctionTemplate") and "arguments" in element.keys():
                                    if "isDataset" in element["arguments"].keys():
                                        if(element["arguments"]["isDataset"] == False):
                                            for arg_element in element["arguments"]["value"]["elements"]:
                                                self._apply_argument(arg_element["arguments"],arg_dict)
                                        else:
                                            if(("value" in element["arguments"]) and "arguments" in element["arguments"]["value"]):
                                                self._apply_argument(element["arguments"]["value"]["arguments"],arg_dict)
                                    self._apply_argument(element["arguments"],arg_dict)
                                    flag_rasters = 1
                            for k,v in arg_dict.items():
                                if "name" in element:
                                    if(element["name"]==k):
                                        if isinstance(v,(ImageryLayer, Raster)) or isinstance(v, _FeatureLayer):
                                            raster = _raster_input_rft(v)
                                            v =_input_rft(raster)
                                            if isinstance(raster,str):
                                                element.clear()
                                                element.update(v)
                                            elif isinstance(raster,dict):
                                                if raster.keys() & {"mosaicRule"}:
                                                    element.update(v)
                                                else:
                                                    element.clear()
                                                    element.update(v)
                                                    #input_dict.update({key:v})
                                            flag_rasters=1
                                        else:
                                            if("value" in element):
                                                if isinstance(element["value"],dict):
                                                    if "type" in element["value"]:
                                                        if element["value"]["type"]=="Scalar":
                                                            element.clear()
                                                            element.update({"type":"Scalar","value":v})
                                                            break
                                                else:
                                                    element["value"]=v
                                                    break
                                            else:
                                                element["value"]=v
                                                if isinstance(element["value"], numbers.Number) and element["isDataset"]==True:
                                                    element.clear()
                                                    element.update({"type":"Scalar","value":v})
                                                    break
                        if (flag_rasters==1 and "Rasters" in input_dict.keys()):
                            if "value" in input_dict["Rasters"]:
                                if "elements" in input_dict["Rasters"]["value"]:
                                    input_dict["Rasters"]["value"]=input_dict["Rasters"]["value"]["elements"]
                if((("type" in value) and value["type"]=="RasterFunctionVariable") and ("value" in value)) and isinstance(value["value"],dict):
                    if "function" in value["value"]:
                        self._apply_argument(value["value"]["arguments"],arg_dict)

        return input_dict

    def _query_rasters(self, rft_dict, raster_dict):
        if "arguments" in rft_dict.keys():
            self._query_rasters(rft_dict["arguments"],raster_dict)
        for key, value in rft_dict.items():
            if isinstance(value, dict):
                if (value["type"]=="RasterFunctionTemplate") and "arguments" in value.keys():
                    self._query_rasters( value["arguments"],raster_dict)
                if "isDataset" in value.keys():
                    if (value["isDataset"]==True) and (value["type"]=="RasterFunctionVariable"):
                        if "name" in value and "value" not in value:
                            raster_dict.update({value["name"]:None})
                    if(value["isDataset"]==False) and (value["type"]=="RasterFunctionVariable"):
                        if "value" in value.keys():
                            if(isinstance(value["value"],dict)):
                                if "type" in value["value"].keys():
                                    if value["value"]["type"] == "ArgumentArray":
                                        if value["value"]["elements"]:
                                            for element in value["value"]["elements"]:
                                                if (element["type"]=="RasterFunctionTemplate") and "arguments" in element.keys():
                                                    self._query_rasters( element["arguments"],raster_dict)
                                                if isinstance(element, dict):
                                                    if "isDataset" in element and "type" in element:
                                                        if (element["isDataset"]==True) and (element["type"]=="RasterFunctionVariable"):
                                                            if "name" in element and "value" not in element:
                                                                raster_dict.update({element["name"]:None})
                                        else:
                                            raster_dict.update({value["name"]:None})

        return raster_dict

    def _find_arguments_(self):
        from operator import eq
        import numbers
        gdict = self._rft_json
        key_value_dict =  {}
        raster_dictionary = {}

        def _function_create(value): #Create new node for the function if it doesn't exist yet
            if "isDataset" in value["arguments"].keys():
                if(value["arguments"]["isDataset"] == False):
                    for arg_element in value["arguments"]["value"]["elements"]:
                        _function_traversal(arg_element)
                else:
                    if("value" in value["arguments"]):
                        _raster_function_traversal(value["arguments"])
            _function_traversal(value["arguments"])

        def _raster_function_traversal(raster_dict, index=1, scalar_name="Raster", ispublic=False, function_arg_type=None): #If isDataset=True
            if "value" in raster_dict.keys(): #Handling Scalar rasters
                if raster_dict["value"] is None:
                    if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                        raster_name = _python_variable_name(raster_dict["name"])
                        raster_dict.update({"name":raster_name})
                        key_value_dict.update({raster_name:None})
                elif isinstance(raster_dict["value"], dict):
                    if "value" in raster_dict["value"]:
                        if isinstance(raster_dict["value"]["value"], numbers.Number):
                            if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                                if "name" in raster_dict.keys():
                                    raster_name = _python_variable_name(raster_dict["name"])
                                    raster_dict.update({"name":raster_name})
                                    key_value_dict.update({raster_name:raster_dict["value"]["value"]})
                                    raster_dictionary.update({raster_name:raster_dict["value"]["value"]})
                                else:
                                    scalar_name = scalar_name+"_scalar_"+str(index)
                                    raster_dict.update({"name":scalar_name})
                                    key_value_dict.update({scalar_name:raster_dict["value"]["value"]}) 

                    elif "elements" in raster_dict["value"]:  #Handling Raster arrays
                        if raster_dict["value"]["elements"]:  #if elements has any value in the list
                            for e in raster_dict["value"]["elements"]:
                                index = (raster_dict["value"]["elements"].index(e))+1
                                if "name" in raster_dict:
                                    scalar_name = _python_variable_name(raster_dict["name"])
                                if ("type" in e) and e["type"]=="FunctionRasterDatasetName":
                                    if self._is_public_flag is False or  ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                                        raster_name = _python_variable_name(raster_dict["name"])
                                        raster_dict.update({"name":raster_name})
                                        key_value_dict.update({raster_name:e["arguments"]["Raster"]["datasetName"]["name"]})
                                        raster_dictionary.update({raster_name:e["arguments"]["Raster"]["datasetName"]["name"]})
                                elif "function" in e.keys(): # if function template inside
                                    _function_traversal(e)
                                else:  #if raster dataset inside raster array
                                    if function_arg_type == "LocalFunctionArguments":
                                        if self._is_public_flag is False or ispublic==True or ("isPublic" not in e.keys()) or (("isPublic" in e.keys()) and e["isPublic"] is True):
                                            _raster_function_traversal(e,  index, scalar_name, ispublic=True)
                                    elif self._is_public_flag is False or ispublic==True or ("isPublic" not in raster_dict.keys()) or (("isPublic" in raster_dict.keys()) and raster_dict["isPublic"] is True):
                                        _raster_function_traversal(e,  index, scalar_name, ispublic=True)
                        else: # If elements is empty i.e Rasters has no value when rft was created
                            if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                                raster_name = _python_variable_name(raster_dict["name"])
                                raster_dict.update({"name":raster_name})
                                key_value_dict.update({raster_name:None})
                                raster_dictionary.update({raster_name:None})
                    elif "name" in raster_dict["value"]: #if raster properties are preserved
                        if "function" in raster_dict["value"]:
                            _function_traversal(raster_dict["value"])
                        else:
                            if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                                raster_name = _python_variable_name(raster_dict["name"])
                                raster_dict.update({"name":raster_name})
                                key_value_dict.update({raster_name:raster_dict["value"]["name"]})
                                raster_dictionary.update({raster_name:raster_dict["value"]["name"]})

                    elif ("type" in raster_dict["value"]) and raster_dict["value"]["type"]=="FunctionRasterDatasetName":
                        if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                            raster_name = _python_variable_name(raster_dict["name"])
                            raster_dict.update({"name":raster_name})
                            key_value_dict.update({raster_name:raster_dict["value"]["arguments"]["Raster"]["datasetName"]["name"]})
                            raster_dictionary.update({raster_name:raster_dict["value"]["arguments"]["Raster"]["datasetName"]["name"]})
                    elif ("type" in raster_dict["value"]) and raster_dict["value"]["type"]=="RasterBandCollectionName":
                        if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                            raster_name = _python_variable_name(raster_dict["name"])
                            raster_dict.update({"name":raster_name})
                            key_value_dict.update({raster_name:raster_dict["value"]["datasetName"]["name"]})
                            raster_dictionary.update({raster_name:raster_dict["value"]["datasetName"]["name"]})
                    elif "datasetName" in raster_dict["value"]: #local image location
                        if "name" in raster_dict["value"]["datasetName"]:
                            if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict["value"]["datasetName"]) and raster_dict["value"]["datasetName"]["isPublic"] is True):
                                raster_name = _python_variable_name(raster_dict["value"]["datasetName"]["name"])
                                raster_dict["value"]["datasetName"].update({"name":raster_name})
                                key_value_dict.update({raster_name:None})

                    elif "function" in raster_dict["value"].keys(): # if function template inside
                        _function_traversal(raster_dict["value"])
                elif isinstance (raster_dict["value"], list): #raster_dict"value" does not have "value" or "elements" in it (ArcMap scalar rft case)
                        for x in raster_dict["value"]:
                            if isinstance(x, numbers.Number):  #Check if scalar float value
                                if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                                     if "name" in raster_dict.keys():
                                         raster_name = _python_variable_name(raster_dict["name"])
                                         raster_dict.update({"name":raster_name})
                                         key_value_dict.update({raster_name:x}) 
                                     #else:
                                         #time.sleep(.00000001)scalar_name+"scalar"+str(index))
                                         #scalar_name = "scalar"+''.join(e for e in str(datetime.now()) if e.isalnum())
                                         #raster_dict.update({"name":scalar_name})
                                         #key_value_dict.update({scalar_name:x}) 
                elif isinstance (raster_dict["value"], numbers.Number):
                    if self._is_public_flag is False or ispublic==True or (("isPublic" in raster_dict.keys()) and raster_dict["isPublic"] is True):
                        if "name" in raster_dict.keys():
                            raster_name = _python_variable_name(raster_dict["name"])
                            raster_dict.update({"name":raster_name})
                            key_value_dict.update({raster_name:raster_dict["value"]})
                        else:
                            scalar_name = scalar_name+"_scalar_"+str(index)
                            raster_dict.update({"name":scalar_name})
                            key_value_dict.update({scalar_name:raster_dict["value"]}) 
            else:
                if self._is_public_flag is False  or ispublic==True or  (("isPublic" in raster_dict) and raster_dict["isPublic"] is True):
                    raster_name = _python_variable_name(raster_dict["name"])
                    raster_dict.update({"name":raster_name})
                    key_value_dict.update({raster_name:None})
                    raster_dictionary.update({raster_name:None})

        def _function_traversal(dictionary):
            if "function" in dictionary.keys():
                _function_create(dictionary)
            function_arg_type=dictionary['type'] if 'type' in dictionary else None  
            for key,value in dictionary.items():
                if isinstance(value , dict):
                    if "isDataset" in value.keys():
                        if (value["isDataset"] == True) or key == "raster" or key == "Raster2" or key == "Rasters" or key == "Raster":
                            _raster_function_traversal(value,function_arg_type=function_arg_type)
                        elif (value["isDataset"] == False):  #Parameters
                            if "value" in value:                                
                                if value["value"] is not None or isinstance(value["value"],bool):
                                    if (isinstance(value["value"],dict)) and "elements" in value["value"]:
                                        if self._is_public_flag is False or (("isPublic" in value) and value["isPublic"] is True):
                                            raster_name = _python_variable_name(value["name"])
                                            value.update({"name":raster_name})
                                            key_value_dict.update({raster_name:value["value"]["elements"]})
                                    else:                                        
                                        if self._is_public_flag is False or (("isPublic" in value) and value["isPublic"] is True):
                                            raster_name = _python_variable_name(value["name"])
                                            value.update({"name":raster_name})
                                            key_value_dict.update({raster_name:value["value"]})
                            else:
                                if self._is_public_flag is False or (("isPublic" in value) and value["isPublic"] is True):
                                    raster_name = _python_variable_name(value["name"])
                                    value.update({"name":raster_name})
                                    key_value_dict.update({raster_name:None})
                    elif "datasetName" in value.keys():
                        if self._is_public_flag is False or  (("isPublic" in value["datasetName"]) and value["datasetName"]["isPublic"] is True):
                            key_value_dict.update({key:value["datasetName"]["name"]})
                    elif "function" in value.keys():  #Function Chain inside Raster
                        _function_create(value)

        if "function" in gdict.keys():
            if "isDataset" in gdict["arguments"].keys():
                if(gdict["arguments"]["isDataset"] == False):
                    if ("value" in gdict["arguments"]):
                        if "elements" in gdict["arguments"]["value"]:
                            if gdict["arguments"]["value"]["elements"]:
                                for arg_element in gdict["arguments"]["value"]["elements"]:
                                    _function_traversal(arg_element)
                            else: # when gdict["arguments"]["value"]["elements"]=[]
                                _raster_function_traversal(gdict["arguments"], function_arg_type=gdict['arguments']['type'] if 'type' in gdict['arguments'] else None)
                    else:
                        _raster_function_traversal(gdict["arguments"], function_arg_type=gdict['arguments']['type'] if 'type' in gdict['arguments'] else None)
            
                else:
                    if "value" in gdict["arguments"]:
                        _function_traversal(gdict["arguments"]["value"])
                    elif (gdict["arguments"]["isDataset"] == True): #Aspect function with only raster parameter
                        _raster_function_traversal(gdict["arguments"],  function_arg_type=gdict['arguments']['type'] if 'type' in gdict['arguments'] else None)
            _function_traversal(gdict["arguments"])
        return key_value_dict, raster_dictionary


    def _apply_rft(self, arg_dict=None, gis = None):
        rft_dict = copy.deepcopy(self._rft_json)
        arg_dict_copy=copy.copy(arg_dict)
        if arg_dict_copy is not None:
            for key in list(arg_dict_copy.keys()):
                if(arg_dict_copy[key] is None):
                    arg_dict_copy.pop(key,None)
            complete_rft_dict = self._apply_argument(rft_dict,arg_dict_copy)

        if self._local_raster is not None:
            if self._local_raster._engine == _ArcpyRaster:
                newlyr = Raster(self._local_raster._uri, is_multidimensional=self._local_raster._is_multidimensional,gis=self._local_raster._gis)
            else:
                newlyr = Raster(complete_rft_dict, is_multidimensional=self._local_raster._is_multidimensional, gis=self._gis)
            self._local_raster = None
            newlyr._engine_obj._fn = complete_rft_dict
            newlyr._engine_obj._fnra = complete_rft_dict
        else:
            newlyr = ImageryLayer(complete_rft_dict, self._gis)
        #_LOGGER.warning("""Set the desired extent on the output Imagery Layer before viewing it""")
            newlyr._fn = complete_rft_dict
            newlyr._fnra = complete_rft_dict
        return newlyr

    def draw_graph(self,show_attributes=False, graph_size="14.25, 15.25"):

        """
        Displays a structural representation of the function chain and it's raster input values. If
        show_attributes is set to True, then the draw_graph function also displays the attributes
        of all the functions in the function chain, representing the rasters in a blue rectangular
        box, attributes in green rectangular box and the raster function names in yellow.

        =================     ====================================================================
        **Argument**          **Description**
        -----------------     --------------------------------------------------------------------
        show_attributes       optional boolean. If True, the graph displayed includes all the
                              attributes of the function and not only it's function name and raster
                              inputs
                              Set to False by default, to display only he raster function name and
                              the raster inputs to it.
        -----------------     --------------------------------------------------------------------
        graph_size            optional string. Maximum width and height of drawing, in inches,
                              seperated by a comma. If only a single number is given, this is used
                              for both the width and the height. If defined and the drawing is
                              larger than the given size, the drawing is uniformly scaled down so
                              that it fits within the given size.
        =================     ====================================================================

        :return: Graph
        """
        from operator import eq
        import numbers
        try:
            from graphviz import Digraph
        except:
            print("Graphviz needs to be installed. pip install graphviz")

        G = Digraph(comment='Raster Function Chain', format = 'svg') # To declare the graph
        G.clear() #clear all previous cases of the same name
        G.attr(rankdir='LR', len='1',overlap="false",splines='ortho', nodesep='0.5',size=graph_size)   #Display graph from Left to Right
        root=0
        gdict = self._rft_json
        global nodenumber,dict_arg
        dict_arg={}
        
        def _function_create(value,childnode): #Create new node for the function if it doesn't exist yet
            global nodenumber
            dict_temp_arg={}
            list_arg=[]
            flag=0
            for k_arg, v_arg in value["arguments"].items():
                list_arg.append(k_arg+str(v_arg))
 
            list_arg.sort()
            list_arg_str=str(list_arg)
            if dict_arg is not None:
                for k_check in dict_arg.keys():
                    if k_check == list_arg_str:
                        G.edge(str(dict_arg.get(k_check)),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                        flag=1
                                    
            if flag == 0:
                nodenumber+=1
                G.node(str(nodenumber),value["function"]["name"], style=('rounded, filled'), shape='box', color='lightgoldenrod1', fillcolor='lightgoldenrod1', fontname="sans-serif")
                G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                connect = nodenumber
                dict_temp_arg={list_arg_str:connect}
                dict_arg.update(dict_temp_arg)
                if "isDataset" in value["arguments"].keys():
                    if(value["arguments"]["isDataset"] == False):
                        for arg_element in value["arguments"]["value"]["elements"]:
                            _function_graph(arg_element,connect)
                    elif (value["arguments"]["isDataset"] == True):
                        _raster_function_graph(value["arguments"],connect)
                _function_graph(value["arguments"],connect)

        def _raster_function_graph(raster_dict, childnode): #If isDataset=True
            global nodenumber,connect
            if "value" in raster_dict.keys(): #Handling Scalar rasters
                if raster_dict["value"] is not None:
                    if not (isinstance(raster_dict["value"],dict)):
                        if isinstance(raster_dict["value"], numbers.Number): 
                            nodenumber+=1
                            G.node(str(nodenumber), str(raster_dict["value"]) , style=('filled'),fontsize="12", shape='circle',fixedsize="shape",color='darkslategray2',fillcolor='darkslategray2', fontname="sans-serif")
                            G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "value" in raster_dict["value"]:
                        if isinstance(raster_dict["value"]["value"], numbers.Number): 
                            nodenumber+=1
                            G.node(str(nodenumber), str(raster_dict["value"]["value"]) , style=('filled'),fontsize="12", shape='circle',fixedsize="shape",color='darkslategray2',fillcolor='darkslategray2', fontname="sans-serif")
                            G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "elements" in raster_dict["value"]:  #Handling Raster arrays
                        if raster_dict["value"]["elements"]:  #if elements has any value in the list
                            for e in raster_dict["value"]["elements"]:
                                if "function" in e.keys(): # if function template inside
                                    _function_graph(e,childnode)
                                else:  #if raster dataset inside raster array
                                    _raster_function_graph(e, childnode)
                        else: # If elements is empty i.e Rasters has no value when rft was created
                            nodenumber+=1
                            G.node(str(nodenumber),str(raster_dict["name"]), style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                            G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "function" in raster_dict["value"]: # If function in value[]
                        _function_graph(raster_dict,childnode)

                    elif "name" in raster_dict["value"]: #if raster properties are preserved
                        if "function" in raster_dict["value"]:
                            _function_graph(raster_dict["value"],childnode)
                        else:
                            nodenumber+=1
                            G.node(str(nodenumber),str(raster_dict["value"]["name"]), style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                            G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "datasetName" in raster_dict["value"]: #local image location
                        if "name" in raster_dict["value"]["datasetName"]:
                            nodenumber+=1
                            G.node(str(nodenumber),str(raster_dict["value"]["datasetName"]["name"]), style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                            G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "function" in raster_dict["value"].keys(): # if function template inside
                        _function_graph(raster_dict["value"],childnode)
                    #raster_dict"value" does not have "value" or "elements" in it (ArcMap scalar rft case)
                    elif isinstance (raster_dict["value"], list):
                        for x in raster_dict["value"]:
                            if isinstance(x, numbers.Number):  #Check if scalar float value
                                nodenumber+=1
                                G.node(str(nodenumber), str(x), style=('filled'), fontsize="12", shape='circle',fixedsize="shape", color='darkslategray2',fillcolor='darkslategray2', fontname="sans-serif")
                                G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                        
                elif "name" in raster_dict.keys():      
                    rastername = str(raster_dict["name"]) #Handling Raster
                    nodenumber+=1
                    G.node(str(nodenumber), rastername, style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                    G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                

            elif "datasetName" in raster_dict.keys():
                rastername = str(raster_dict["datasetName"]["name"]) #Handling Raster
                nodenumber+=1
                G.node(str(nodenumber), rastername, style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")

            elif "name" in raster_dict: 
                rastername = str(raster_dict["name"]) #Handling Raster
                nodenumber+=1
                G.node(str(nodenumber), rastername, style=('filled'), shape='note',color='darkseagreen2',fillcolor='darkseagreen2', fontname="sans-serif")
                G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")


        def _function_graph(dictionary, childnode): 
            global nodenumber,connect
            count=0
            if "function" in dictionary.keys():
                _function_create(dictionary,childnode)

            for key,value in dictionary.items():
                if isinstance(value , dict):
                    if "isDataset" in value.keys():
                        if (value["isDataset"] == True) or key == "Raster" or key == "Raster2" or key == "Rasters":
                            _raster_function_graph(value, childnode)
                        elif (value["isDataset"] == False) and show_attributes == True:  #Parameters
                            nodenumber+=1
                            if "name" in value and value["name"] not in hidden_inputs:
                                if "value" in value:
                                    if value["value"] is not None or isinstance(value["value"],bool):
                                        atrr_name=str(value["name"])+" = "+str(value["value"])
                                else:
                                    atrr_name=str(value["name"])

                                G.node(str(nodenumber), atrr_name, style=('filled'), shape='rectangle',color='antiquewhite',fillcolor='antiquewhite', fontname="sans-serif")
                                G.edge(str(nodenumber),str(childnode),color="silver", arrowsize="0.9", penwidth="1")
                    
                    elif "datasetName" in value.keys():
                        _raster_function_graph(value, childnode)

                    elif "function" in value.keys():  #Function Chain inside Raster
                        _function_create(value,childnode)

        if "function" in gdict.keys():
            G.node(str(root),gdict["function"]["name"], style=('rounded, filled'), shape='box', color='lightgoldenrod1', fillcolor='lightgoldenrod1', fontname="sans-serif")
            nodenumber=root
            if "isDataset" in gdict["arguments"].keys():
                if(gdict["arguments"]["isDataset"] == False):
                    if "value" in gdict["arguments"]:
                        if "elements" in gdict["arguments"]["value"]:
                            if gdict["arguments"]["value"]["elements"]:
                                for arg_element in gdict["arguments"]["value"]["elements"]:
                                    _function_graph(arg_element,root)
                            else: # when gdict["arguments"]["value"]["elements"]=[]
                                _raster_function_graph(gdict["arguments"],root)
                    else:
                        _raster_function_graph(gdict["arguments"],root)
                else:
                    if "value" in gdict["arguments"]:
                        _function_graph(gdict["arguments"]["value"],root)
                    elif (gdict["arguments"]["isDataset"] == True):
                        _raster_function_graph(gdict["arguments"],root)
            _function_graph(gdict["arguments"],root)
        return G
  
    def _repr_svg_(self):
        graph=self.draw_graph()
        svg_graph=graph.pipe().decode('utf-8')
        return svg_graph
