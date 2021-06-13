"""
Global Raster functions.
These functions are applied to the raster data to create a
processed product on disk, using ImageryLayer.save() method or arcgis.raster.analytics.generate_raster().

Global functions cannot be used for visualization using dynamic image processing. They cannot be applied to layers that
are added to a map for on-the-fly image processing or visualized inline within the Jupyter notebook.

Functions can be applied to various rasters (or images), including the following:

* Imagery layers
* Rasters within imagery layers

"""
from arcgis.raster._layer import ImageryLayer,  Raster, _ArcpyRaster, RasterCollection
from arcgis.features import FeatureLayer
from arcgis.gis import Item
import copy
import numbers
from arcgis.raster.functions.utility import _raster_input, _get_raster, _replace_raster_url, _get_raster_url, _get_raster_ra
from arcgis.geoprocessing._support import _layer_input,_feature_input
import string as _string
import random as _random
import arcgis as _arcgis
from ..._impl.common._deprecate import deprecated

def _create_output_image_service(gis, output_name, task):
    ok = gis.content.is_service_name_available(output_name, "Image Service")
    if not ok:
        raise RuntimeError("An Image Service by this name already exists: " + output_name)

    create_parameters = {
        "name": output_name,
        "description": "",
        "capabilities": "Image",
        "properties": {
            "path": "@",
            "description": "",
            "copyright": ""
        }
    }

    output_service = gis.content.create_service(output_name, create_params=create_parameters,
                                                      service_type="imageService")
    description = "Image Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": "Analysis Image Service generated from " + task
    }
    output_service.update(item_properties)
    return output_service


def _id_generator(size=6, chars=_string.ascii_uppercase + _string.digits):
    return ''.join(_random.choice(chars) for _ in range(size))


def _gbl_clone_layer(layer, function_chain, function_chain_ra,**kwargs):
    if isinstance(layer, Raster) or isinstance(layer, RasterCollection):
        return _gbl_clone_layer_raster(layer, function_chain, function_chain_ra, **kwargs)
    if isinstance(layer, Item):
        layer = layer.layers[0]

    newlyr = ImageryLayer(layer._url, layer._gis)

    newlyr._lazy_properties = layer.properties
    newlyr._hydrated = True
    newlyr._lazy_token = layer._token

    # if layer._fn is not None: # chain the functions
    #     old_chain = layer._fn
    #     newlyr._fn = function_chain
    #     newlyr._fn['rasterFunctionArguments']['Raster'] = old_chain
    # else:
    newlyr._fn = function_chain_ra
    newlyr._fnra = function_chain_ra

    newlyr._where_clause = layer._where_clause
    newlyr._spatial_filter = layer._spatial_filter
    newlyr._temporal_filter = layer._temporal_filter
    newlyr._mosaic_rule = layer._mosaic_rule
    newlyr._filtered = layer._filtered
    newlyr._extent = layer._extent

    newlyr._uses_gbl_function = True
    for key in kwargs:
        newlyr._other_outputs.update({key:kwargs[key]})

    return newlyr

def _feature_gbl_clone_layer(layer, function_chain, function_chain_ra,**kwargs):
    if isinstance(layer, Item):
        layer = layer.layers[0]

    newlyr = ImageryLayer(layer._url, layer._gis)

    newlyr._fn = function_chain
    newlyr._fnra = function_chain_ra

    newlyr._storage = layer._storage
    newlyr._dynamic_layer = layer._dynamic_layer

    newlyr._uses_gbl_function = True
    for key in kwargs:
        newlyr._other_outputs.update({key:kwargs[key]})
    return newlyr


def _gbl_clone_layer_raster(layer, function_chain, function_chain_ra, **kwargs):

    if layer._datastore_raster:
        newlyr= Raster(layer._uri, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
    else:
        newlyr = Raster(layer._url, is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)

    if layer._engine==_ArcpyRaster:
        try:            
            import arcpy, json
            arcpylyr=arcpy.ia.Apply(layer._uri,json.dumps(function_chain_ra))
            newlyr = Raster(str(arcpylyr), is_multidimensional= layer._is_multidimensional, engine= layer._engine, gis=layer._gis)
        except:
            pass

    #newlyr.properties = layer.properties
    newlyr._engine_obj._fn = function_chain
    newlyr._engine_obj._fnra = function_chain_ra
    newlyr._engine_obj._where_clause = layer._where_clause
    newlyr._engine_obj._spatial_filter = layer._spatial_filter
    newlyr._engine_obj._temporal_filter = layer._temporal_filter
    newlyr._engine_obj._mosaic_rule = layer._mosaic_rule
    newlyr._engine_obj._filtered = layer._filtered
    newlyr._engine_obj._uses_gbl_function = layer._uses_gbl_function
    newlyr._engine_obj._do_not_hydrate = layer._do_not_hydrate

    if hasattr(layer, "_lazy_token"):
        newlyr._engine_obj._lazy_token = layer._lazy_token
    else:
        newlyr._lazy_token = layer._token

    return newlyr

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation() instead. ")
def euclidean_distance(in_source_data,
                       cell_size=None,
                       max_distance=None,
                       distance_method="PLANAR", 
                       in_barrier_data=None):
    """
    Calculates, for each cell, the Euclidean distance to the closest source.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/euclidean-distance-global-function.htm

    Parameters
    ----------
    :param in_source_data: Required. The input raster that identifies the pixels or locations to
                            which the Euclidean distance for every output pixel location is calculated.
                            The input type can be an integer or a floating-point value.
    :param cell_size:  Optional. The pixel size at which the output raster will be created. If the cell
                            size was explicitly set in Environments, that will be the default cell size.
                            If Environments was not set, the output cell size will be the same as the
                            Source Raster
    :param max_distance: Optional. The threshold that the accumulative distance values cannot exceed. If an
                            accumulative Euclidean distance exceeds this value, the output value for
                            the pixel location will be NoData. The default distance is to the edge
                            of the output raster
    :param distance_method: Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.
    :param in_barrier_data: Optional barrier raster. The input raster that defines the barriers. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 
                            
    :return: output raster with function applied
    """
    layer, in_source_data, raster_ra = _raster_input(in_source_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "EucDistance_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_distance_raster",
            "in_source_data": in_source_data,

        }
    }

    if in_barrier_data is not None:
        layer2, in_barrier_data, raster_ra2 = _raster_input(in_barrier_data)
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data
    
    if cell_size is not None:
        template_dict["rasterFunctionArguments"]["cell_size"] = cell_size

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    distance_method_list = ["PLANAR","GEODESIC"]
    if distance_method is not None:
        if distance_method.upper() not in distance_method_list:
            raise RuntimeError('distance_method should be one of the following '+ str(distance_method_list))
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra

    if in_barrier_data is not None:
        function_chain_ra["rasterFunctionArguments"]["in_barrier_data"] = raster_ra2

    return _gbl_clone_layer(layer, template_dict, function_chain_ra)


@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_allocation() instead. ")
def euclidean_allocation(in_source_data,
                         in_value_raster=None,
                         max_distance=None,
                         cell_size=None,
                         source_field=None,
                         distance_method="PLANAR",
                         in_barrier_data=None):

    """
    Calculates, for each cell, the nearest source based on Euclidean distance.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/euclidean-allocation-global-function.htm

    Parameters
    ----------
    :param in_source_data: Required; The input raster that identifies the pixels or locations to which
                            the Euclidean distance for every output pixel location is calculated.
                            The input type can be an integer or a floating-point value.
                            If the input Source Raster is floating point, the Value Raster must be set,
                            and it must be an integer. The Value Raster will take precedence over any
                            setting of the Source Field.
    :param in_value_raster: Optional. The input integer raster that identifies the zone values that should be
                            used for each input source location. For each source location pixel, the
                            value defined by the Value Raster will be assigned to all pixels allocated
                            to the source location for the computation. The Value Raster will take
                            precedence over any setting for the Source Field .
    :param max_distance: Optional. The threshold that the accumulative distance values cannot exceed. If an
                            accumulative Euclidean distance exceeds this value, the output value for
                            the pixel location will be NoData. The default distance is to the edge
                            of the output raster
    :param cell_size: Optional. The pixel size at which the output raster will be created. If the cell size
                            was explicitly set in Environments, that will be the default cell size.
                            If Environments was not set, the output cell size will be the same as the
                            Source Raster
    :param source_field: Optional. The field used to assign values to the source locations. It must be an
                            integer type. If the Value Raster has been set, the values in that input
                            will take precedence over any setting for the Source Field.
    :param distance_method: Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.
    :param in_barrier_data: Optional barrier raster. The input raster that defines the barriers. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 

    :return: output raster with function applied
    """

    layer1, in_source_data, raster_ra1 = _raster_input(in_source_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "EucAllocation_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_allocation_raster",
            "in_source_data": in_source_data

        }
    }

    if in_value_raster is not None:
        layer2, in_value_raster, raster_ra2 = _raster_input(in_value_raster)
        template_dict["rasterFunctionArguments"]["in_value_raster"] = in_value_raster

    if in_barrier_data is not None:
        layer3, in_barrier_data, raster_ra3 = _raster_input(in_barrier_data)
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data
    
    if cell_size is not None:
        template_dict["rasterFunctionArguments"]["cell_size"] = cell_size

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    if source_field is not None:
        template_dict["rasterFunctionArguments"]["source_field"] = source_field

    distance_method_list = ["PLANAR","GEODESIC"]
    if distance_method is not None:
        if distance_method.upper() not in distance_method_list:
            raise RuntimeError('distance_method should be one of the following '+ str(distance_method_list))
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_value_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_value_raster"] = raster_ra2
    if in_barrier_data is not None:
        function_chain_ra["rasterFunctionArguments"]["in_barrier_data"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation() instead. ")
def cost_distance(in_source_data,
                  in_cost_raster,
                  max_distance=None,
                  source_cost_multiplier=None,
                  source_start_cost=None,
                  source_resistance_rate=None,
                  source_capacity=None,
                  source_direction=None):
    """
    Calculates the least accumulative cost distance for each cell from or to the least-cost
    source over a cost surface.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/cost-distance-global-function.htm

    Parameters
    ----------
    :param in_source_data: Required. The input raster that identifies the pixels or locations to which the
                            least accumulated cost distance for every output pixel location is
                            calculated. The Source Raster can be an integer or a floating-point value.
    :param in_cost_raster: Required. A raster defining the cost or impedance to move planimetrically through each pixel.
                            The value at each pixel location represents the cost-per-unit distance for moving
                            through it. Each pixel location value is multiplied by the pixel resolution, while
                            also compensating for diagonal movement to obtain the total cost of passing through
                            the pixel.
    :param max_distance: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative cost
                            distance exceeds this value, the output value for the pixel location will be NoData.
                            The maximum distance defines the extent for which the accumulative cost distances are
                            calculated. The default distance is to the edge of the output raster.
    :param source_cost_multiplier: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative
                            cost distance exceeds this value, the output value for the pixel location will be
                            NoData. The maximum distance defines the extent for which the accumulative cost
                            distances are calculated. The default distance is to the edge of the output raster.
    :param source_start_cost: Optional. The starting cost from which to begin the cost calculations. This parameter allows
                            for the specification of the fixed cost associated with a source. Instead of starting
                            at a cost of 0, the cost algorithm will begin with the value set here.
                            The default is 0. The value must be 0 or greater. A numeric (double) value or a field
                            from the Source Raster can be used for this parameter.
    :param source_resistance_rate: Optional. This parameter simulates the increase in the effort to overcome costs as the
                            accumulative cost increases. It is used to model fatigue of the traveler. The growing
                            accumulative cost to reach a pixel is multiplied by the resistance rate and added to
                            the cost to move into the subsequent pixel.
                            It is a modified version of a compound interest rate formula that is used to calculate
                            the apparent cost of moving through a pixel. As the value of the resistance rate increases,
                            it increases the cost of the pixels that are visited later. The greater the resistance rate,
                            the higher the cost to reach the next pixel, which is compounded for each subsequent movement.
                            Since the resistance rate is similar to a compound rate and generally the accumulative cost
                            values are very large, small resistance rates are suggested, such as 0.005 or even smaller,
                            depending on the accumulative cost values.
                            The default is 0. The values must be 0 or greater. A numeric (double) value or a field from
                            the Source Raster can be used for this parameter.
    :param source_capacity: Optional. Defines the cost capacity for the traveler for a source. The cost calculations continue for
                            each source until the specified capacity is reached.
                            The default capacity is to the edge of the output raster. The values must be greater than 0.
                            A double numeric value or a field from the Source Raster can be used for this parameter.
    :param source_direction: Optional. Defines the direction of the traveler when applying the source resistance rate and the source
                            starting cost.

                            - FROM_SOURCE - The source resistance rate and source starting cost will be applied beginning \
                            at the input source and moving out to the nonsource cells. This is the default.

                            - TO_SOURCE - The source resistance rate and source starting cost will be applied beginning at \
                            each nonsource cell and moving back to the input source.

                            Either specify the FROM_SOURCE or TO_SOURCE keyword, which will be applied to all sources,
                            or specify a field in the Source Raster that contains the keywords to identify the direction
                            of travel for each source. That field must contain the string FROM_SOURCE or TO_SOURCE.

    :return: output raster with function applied
    """
    layer1, in_source_data, raster_ra1 = _raster_input(in_source_data)
    layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CostDistance_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_distance_raster",
            "in_source_data": in_source_data,
            "in_cost_raster": in_cost_raster

        }
    }

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    source_direction_list = ["FROM_SOURCE","TO_SOURCE"]

    if source_direction is not None:
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list))
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_cost_raster"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_allocation() instead. ")
def cost_allocation(in_source_data,
                    in_cost_raster,
                    in_value_raster=None,
                    max_distance=None,
                    source_field=None,
                    source_cost_multiplier=None,
                    source_start_cost=None,
                    source_resistance_rate=None,
                    source_capacity=None,
                    source_direction=None):
    """
    Calculates, for each cell, its least-cost source based on the least accumulative cost over a cost surface.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/cost-allocation-global-function.htm

    Parameters
    ----------
    :param in_source_data: Required. The input raster that identifies the pixels or locations to which the
                            least accumulated cost distance for every output pixel location is
                            calculated. The Source Raster can be an integer or a floating-point value.
                            If the input Source Raster is floating point, the Value Raster must be set,
                            and it must be an integer. The Value Raster will take precedence over any
                            setting of the Source Field.
    :param in_cost_raster:  Required. A raster defining the cost or impedance to move planimetrically through each pixel.
                            The value at each pixel location represents the cost-per-unit distance for moving
                            through it. Each pixel location value is multiplied by the pixel resolution, while
                            also compensating for diagonal movement to obtain the total cost of passing through
                            the pixel.
                            The values of the Cost Raster can be integer or floating point, but they cannot be
                            negative or zero.
    :param in_value_raster: Optional. The input integer raster that identifies the zone values that should be used for
                            each input source location. For each source location pixel, the value defined by
                            the Value Raster will be assigned to all pixels allocated to the source location
                            for the computation. The Value Raster will take precedence over any setting for
                            the Source Field.
    :param max_distance: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative cost
                            distance exceeds this value, the output value for the pixel location will be NoData.
                            The maximum distance defines the extent for which the accumulative cost distances are
                            calculated. The default distance is to the edge of the output raster.
    :param source_field: Optional. The field used to assign values to the source locations. It must be an integer type.
                            If the Value Raster has been set, the values in that input will take precedence over
                            any setting for the Source Field.
    :param source_cost_multiplier: Optional. This parameter allows for control of the mode of travel or the magnitude at
                            a source. The greater the multiplier, the greater the cost to move through each cell.
                            The default value is 1. The values must be greater than 0. A numeric (double) value or
                            a field from the Source Raster can be used for this parameter.
    :param source_start_cost: Optional. The starting cost from which to begin the cost calculations. This parameter allows
                            for the specification of the fixed cost associated with a source. Instead of starting
                            at a cost of 0, the cost algorithm will begin with the value set here.
                            The default is 0. The value must be 0 or greater. A numeric  (double) value or a field
                            from the Source Raster can be used for this parameter.
    :param source_resistance_rate: Optional. This parameter simulates the increase in the effort to overcome costs as the
                            accumulative cost increases. It is used to model fatigue of the traveler. The growing
                            accumulative cost to reach a pixel is multiplied by the resistance rate and added to
                            the cost to move into the subsequent pixel.
                            It is a modified version of a compound interest rate formula that is used to calculate
                            the apparent cost of moving through a pixel. As the value of the resistance rate increases,
                            it increases the cost of the pixels that are visited later. The greater the resistance rate,
                            the higher the cost to reach the next pixel, which is compounded for each subsequent movement.
                            Since the resistance rate is similar to a compound rate and generally the accumulative cost
                            values are very large, small resistance rates are suggested, such as 0.005 or even smaller,
                            depending on the accumulative cost values.
                            The default is 0. The values must be 0 or greater. A numeric (double) value or a field from
                            the Source Raster can be used for this parameter.
    :param source_capacity: Optional. Defines the cost capacity for the traveler for a source. The cost calculations continue for
                            each source until the specified capacity is reached.
                            The default capacity is to the edge of the output raster. The values must be greater than 0.
                            A double numeric value or a field from the Source Raster can be used for this parameter.
    :param source_direction: Optional. Defines the direction of the traveler when applying the source resistance rate and the source
                            starting cost.

                            - FROM_SOURCE - The source resistance rate and source starting cost will be applied beginning \
                            at the input source and moving out to the nonsource cells. This is the default.

                            - TO_SOURCE - The source resistance rate and source starting cost will be applied beginning at \
                            each nonsource cell and moving back to the input source.

                            Either specify the FROM_SOURCE or TO_SOURCE keyword, which will be applied to all sources,
                            or specify a field in the Source Raster that contains the keywords to identify the direction
                            of travel for each source. That field must contain the string FROM_SOURCE or TO_SOURCE.

    :return: output raster with function applied
    """

    layer1, in_source_data, raster_ra1 = _raster_input(in_source_data)
    layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CostAllocation_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_allocation_raster",
            "in_source_data": in_source_data,
            "in_cost_raster": in_cost_raster

        }
    }
    if in_value_raster is not None:
        layer3, in_value_raster, raster_ra3 = _raster_input(in_value_raster)
        template_dict["rasterFunctionArguments"]["in_value_raster"] = in_value_raster

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    if source_field is not None:
        template_dict["rasterFunctionArguments"]["source_field"] = source_field

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    source_direction_list = ["FROM_SOURCE","TO_SOURCE"]

    if source_direction is not None:
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list))
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_cost_raster"] = raster_ra2
    if in_value_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_value_raster"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def zonal_statistics(in_zone_data,
                     zone_field,
                     in_value_raster,
                     ignore_nodata=True,
                     statistics_type='MEAN',
                     process_as_multidimensional=None,
                     percentile_value=90,
                     percentile_interpolation_type='AUTO_DETECT'):

    """"
    Calculates statistics on values of a raster within the zones of another dataset.
    For more information,
    http://pro.arcgis.com/en/pro-app/help/data/imagery/zonal-statistics-global-function.htm

    Parameters
    ----------
    :param in_zone_data: Required raster layer. Dataset that defines the zones. The zones can be defined by an integer raster
    :param zone_field: Required string or integer. Field that holds the values that define each zone. It can be an integer or a
                            string field of the zone raster.
    :param in_value_raster: Required raster layer. Raster that contains the values on which to calculate a statistic.
    :param ignore_no_data: Optional bool. Denotes whether NoData values in the Value Raster will influence the results
                            of the zone that they fall within.

                            - True - Within any particular zone, only pixels that have a value in the Value \
                            Raster will be used in determining the output value for that zone. NoData \
                            pixels in the Value Raster will be ignored in the statistic calculation. This is the default.

                            - False - Within any particular zone, if any NoData pixels exist in the Value \
                            Raster, it is deemed that there is insufficient information to perform \
                            statistical calculations for all the pixels in that zone; therefore, the \
                            entire zone will receive the NoData value on the output raster.
    :param statistics_type: Optional str. Statistic type to be calculated. Default is MEAN

                            - MEAN-Calculates the average of all pixels in the Value Raster that belong to \
                            the same zone as the output pixel.

                            - MAJORITY-Determines the value that occurs most often of all pixels in the \
                            Value Raster that belong to the same zone as the output pixel.

                            - MAXIMUM-Determines the largest value of all pixels in the Value Raster \
                            that belong to the same zone as the output pixel.

                            - MEDIAN-Determines the median value of all pixels in the Value Raster \
                            that belong to the same zone as the output pixel.

                            - MINIMUM-Determines the smallest value of all pixels in the Value Raster \
                            that belong to the same zone as the output pixel.

                            - MINORITY-Determines the value that occurs least often of all pixels in \
                            the Value Raster that belong to the same zone as the output pixel.

                            - RANGE-Calculates the difference between the largest and smallest value \
                            of all pixels in the Value Raster that belong to the same zone as the \
                            output pixel.

                            - STD-Calculates the standard deviation of all pixels in \
                            the Value Rasterthat belong to the same zone as the output pixel.

                            - SUM-Calculates the total value of all pixels in the Value Raster that \
                            belong to the same zone as the output pixel.

                            - VARIETY-Calculates the number of unique values for all pixels in the \
                            Value Raster that belong to the same zone as the output pixel.

                            - PERCENTILE -Calculates a percentile of all cells in the value raster that \
                            belong to the same zone as the output cell. The 90th percentile \
                            is calculated by default. You can specify other values (from 0 to 100) \
                            using the percentile_value parameter.

    :param process_as_multidimensional: Optional bool, Process as multidimensional if set to True. (If the input is multidimensional raster.)
    :param percentile_value: Optional Double, The percentile to calculate. The default is 90, for the 90th percentile. The 
                             values can range from 0 to 100. The 0th percentile is essentially equivalent to the 
                             Minimum statistic, and the 100th percentile is equivalent to Maximum. 
                             A value of 50 will produce essentially the same result as the Median statistic.
                             
                             This parameter is honoured only available if the statistics_type parameter is 
                             set to PERCENTILE.
    :param percentile_interpolation_type: Optional str. Determines the type of percentile interpolation type when the 
                                          number of values from the input value raster to be calculated are even.
                                            - AUTO_DETECT - If the input value raster has integer pixel type, the NEAREST method is used. If the input value raster has floating point pixel type, then the LINEAR method is used. This is the default.
                                            - NEAREST - Nearest value to the desired percentile. In this case, the output pixel type is same as that of the input value raster.
                                            - LINEAR - Weighted average of two surrounding values from the desired percentile. In this case, the output pixel type is floating point.

                                          Parameter available in ArcGIS Image Server 10.9 and higher.
    :return: output raster with function applied

    """
    layer1, in_zone_data, raster_ra1 = _raster_input(in_zone_data)
    layer2, in_value_raster, raster_ra2 = _raster_input(in_value_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "ZonalStatistics_sa",
            "PrimaryInputParameterName" : "in_value_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_zone_data" : in_zone_data,
            "zone_field" : zone_field,
            "in_value_raster" : in_value_raster

        }
    }

    if ignore_nodata is not None:
        if not isinstance(ignore_nodata,bool):
            raise RuntimeError('ignore_nodata should be a boolean value')
        if ignore_nodata is True:
            ignore_nodata = "DATA"
        elif ignore_nodata is False:
            ignore_nodata = "NODATA"
        template_dict["rasterFunctionArguments"]["ignore_nodata"] = ignore_nodata


    statistics_type_list = ["MEAN","MAJORITY","MAXIMUM","MEDIAN","MINIMUM","MINORITY","RANGE","STD","SUM","VARIETY", "PERCENTILE"]
    if statistics_type is not None:
        if statistics_type.upper() not in statistics_type_list:
            raise RuntimeError('statistics_type should be one of the following '+ str(statistics_type_list))
        template_dict["rasterFunctionArguments"]["statistics_type"] = statistics_type

    if process_as_multidimensional is not None:
        if isinstance(process_as_multidimensional, bool):
            if process_as_multidimensional==True:
                template_dict["rasterFunctionArguments"]["process_as_multidimensional"]="ALL_SLICES"
            else:
                template_dict["rasterFunctionArguments"]["process_as_multidimensional"]="CURRENT_SLICE"

    if percentile_value is not None:
        template_dict["rasterFunctionArguments"]["percentile_value"] = percentile_value

    percentile_interpolation_type_list = ["AUTO_DETECT","NEAREST","LINEAR"]
    if percentile_interpolation_type is not None:
        if percentile_interpolation_type.upper() not in percentile_interpolation_type_list:
            raise RuntimeError('percentile_interpolation_type should be one of the following '+ str(percentile_interpolation_type_list))
        template_dict["rasterFunctionArguments"]["percentile_interpolation_type"] = percentile_interpolation_type


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_zone_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_value_raster"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def least_cost_path(in_source_data,
                    in_cost_raster,
                    in_destination_data,
                    destination_field=None,
                    path_type="EACH_CELL",
                    max_distance=None,
                    source_cost_multiplier=None,
                    source_start_cost=None,
                    source_resistance_rate=None,
                    source_capacity=None,
                    source_direction=None):
    """
    Calculates the least-cost path from a source to a destination. The least accumulative cost distance
    is calculated for each pixel over a cost surface, to the nearest source. This produces an output
    raster that records the least-cost path, or paths, from selected locations to the closest source
    pixels defined within the accumulative cost surface, in terms of cost distance.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/least-cost-path-global-function.htm

    Parameters
    ----------
    :param in_source_data: Required. The input raster that identifies the pixels or locations to which the
                            least accumulated cost distance for every output pixel location is
                            calculated. The Source Raster can be an integer or a floating-point value.
                            If the input Source Raster is floating point, the Value Raster must be set,
                            and it must be an integer. The Value Raster will take precedence over any
                            setting of the Source Field.
    :param in_cost_raster: Required. A raster defining the cost or impedance to move planimetrically through each pixel.
                            The value at each pixel location represents the cost-per-unit distance for moving
                            through it. Each pixel location value is multiplied by the pixel resolution, while
                            also compensating for diagonal movement to obtain the total cost of passing through
                            the pixel.
                            The values of the Cost Raster can be integer or floating point, but they cannot be
                            negative or zero.
    :param in_destination_data: Required. A raster that identifies the pixels from which the least-cost path is
                            determined to the least costly source. This input consists of pixels that have valid
                            values, and the remaining pixels must be assigned NoData. Values of 0 are valid.
    :param destination_field: Optional. The field used to obtain values for the destination locations.
    :param path_type: Optional. A keyword defining the manner in which the values and zones on the input destination
                            data will be interpreted in the cost path calculations:

                            - EACH_CELL-A least-cost path is determined for each pixel with valid values on the \
                            input destination data, and saved on the output raster. Each cell of the input \
                            destination data is treated separately, and a least-cost path is determined for each from cell.

                            - EACH_ZONE-A least-cost path is determined for each zone on the input destination data and \
                            saved on the output raster. The least-cost path for each zone begins at the pixel with the \
                            lowest cost distance weighting in the zone.

                            - BEST_SINGLE-For all pixels on the input destination data, the least-cost path is derived \
                            from the pixel with the minimum of the least-cost paths to source cells.
    :param max_distance: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative cost
                            distance exceeds this value, the output value for the pixel location will be NoData.
                            The maximum distance defines the extent for which the accumulative cost distances are
                            calculated. The default distance is to the edge of the output raster.
    :param source_field: Optional. The field used to assign values to the source locations. It must be an integer type.
                            If the Value Raster has been set, the values in that input will take precedence over
                            any setting for the Source Field.
    :param source_cost_multiplier: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative
                            cost distance exceeds this value, the output value for the pixel location will be
                            NoData. The maximum distance defines the extent for which the accumulative cost
                            distances are calculated. The default distance is to the edge of the output raster.
    :param source_start_cost: Optional. The starting cost from which to begin the cost calculations. This parameter allows
                            for the specification of the fixed cost associated with a source. Instead of starting
                            at a cost of 0, the cost algorithm will begin with the value set here.
                            The default is 0. The value must be 0 or greater. A numeric (double) value or a field
                            from the Source Raster can be used for this parameter.
    :param source_resistance_rate: Optional. This parameter simulates the increase in the effort to overcome costs as the
                            accumulative cost increases. It is used to model fatigue of the traveler. The growing
                            accumulative cost to reach a pixel is multiplied by the resistance rate and added to
                            the cost to move into the subsequent pixel.
                            It is a modified version of a compound interest rate formula that is used to calculate
                            the apparent cost of moving through a pixel. As the value of the resistance rate increases,
                            it increases the cost of the pixels that are visited later. The greater the resistance rate,
                            the higher the cost to reach the next pixel, which is compounded for each subsequent movement.
                            Since the resistance rate is similar to a compound rate and generally the accumulative cost
                            values are very large, small resistance rates are suggested, such as 0.005 or even smaller,
                            depending on the accumulative cost values.
                            The default is 0. The values must be 0 or greater. A numeric (double) value or a field from
                            the Source Raster can be used for this parameter.
    :param source_capacity: Optional. Defines the cost capacity for the traveler for a source. The cost calculations continue for
                            each source until the specified capacity is reached.
                            The default capacity is to the edge of the output raster. The values must be greater than 0.
                            A double numeric value or a field from the Source Raster can be used for this parameter.
    :param source_direction: Optional. Defines the direction of the traveler when applying the source resistance rate and the source
                            starting cost.

                            - FROM_SOURCE - The source resistance rate and source starting cost will be applied beginning \
                            at the input source and moving out to the nonsource cells. This is the default.

                            - TO_SOURCE - The source resistance rate and source starting cost will be applied beginning at \
                            each nonsource cell and moving back to the input source.

                            Either specify the FROM_SOURCE or TO_SOURCE keyword, which will be applied to all sources,
                            or specify a field in the Source Raster that contains the keywords to identify the direction
                            of travel for each source. That field must contain the string FROM_SOURCE or TO_SOURCE.

    :return: output raster with function applied
    """
    layer1, in_source_data, raster_ra1 = _raster_input(in_source_data)
    layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)
    layer3, in_destination_data, raster_ra3 = _raster_input(in_destination_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "ShortestPath",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_path_raster",
            "in_source_data" : in_source_data,
            "in_cost_raster" : in_cost_raster,
            "in_destination_data" : in_destination_data

        }
    }

    if destination_field is not None:
        template_dict["rasterFunctionArguments"]["destination_field"] = destination_field

    if path_type is not None:
        path_type_list = ["EACH_CELL", "EACH_ZONE", "BEST_SINGLE"]
        if path_type.upper() not in path_type_list:
            raise RuntimeError('path_type should be one of the following '+ str(path_type_list))
        template_dict["rasterFunctionArguments"]["path_type"] = path_type

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    source_direction_list = ["FROM_SOURCE","TO_SOURCE"]

    if source_direction is not None:
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_cost_raster"] = raster_ra2
    function_chain_ra["rasterFunctionArguments"]["in_destination_data"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def flow_distance(input_stream_raster,
                  input_surface_raster,
                  input_flow_direction_raster=None,
                  distance_type="VERTICAL",
                  flow_direction_type= "D8",
                  statistics_type="MINIMUM"):

    """
    This function computes, for each cell, the minimum downslope
    horizontal or vertical distance to cell(s) on a stream or
    river into which they flow. If an optional flow direction
    raster is provided, the down slope direction(s) will be
    limited to those defined by the input flow direction raster.

    Parameters
    ----------
    :param input_stream_raster: Required.  An input raster that represents a linear stream network
    :param input_surface_raster: Required. The input raster representing a continuous surface.
    :param input_flow_direction_raster: Optional. The input raster that shows the direction of flow out of each cell.
    :param distance_type: Optional. VERTICAL or HORIZONTAL distance to compute; if not specified, VERTICAL distance is computed.
    :param flow_direction_type: Optional String; Defines the type of the input flow direction raster.

                                - D8 - The input flow direction raster is of type D8. This is the default.

                                - MFD - The input flow direction raster is of type Multi Flow Direction (MFD).

                                - Dinf - The input flow direction raster is of type D-Infinity (DINF).
    :param statistics_type: Optional String; Determines the statistics type used to compute flow distance 
                            over multiple flow paths. 
                            If there is only a single flow path from each cell to a cell on the stream, 
                            all statistics types produce the same result.

                            - MINIMUM - Where multiple flow paths exist, minimum flow distance in computed. \
                            This is the default.

                            - WEIGHTED_MEAN - Where multiple flow paths exist, a weighted mean of flow distance \
                            is computed. Flow proportion from a cell to its downstream neighboring cells are \
                            used as weights for computing weighted mean.

                            - MAXIMUM - When multiple flow paths exist, maximum flow distance is computed.
    :return: output raster with function applied
    """
    layer1, input_stream_raster, raster_ra1 = _raster_input(input_stream_raster)
    layer2, input_surface_raster, raster_ra2 = _raster_input(input_surface_raster)


    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "FlowDistance_sa",
            "PrimaryInputParameterName" : "in_stream_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_stream_raster" : input_stream_raster,
            "in_surface_raster" : input_surface_raster,

        }
    }
    if input_flow_direction_raster is not None:
        layer3, input_flow_direction_raster, raster_ra3 = _raster_input(input_flow_direction_raster)
        template_dict["rasterFunctionArguments"]["in_flow_direction_raster"] = input_flow_direction_raster

    distance_type_list = ["VERTICAL","HORIZONTAL"]

    if distance_type is not None:
        if distance_type.upper() not in distance_type_list:
            raise RuntimeError('distance_type should be one of the following '+ str(distance_type_list))
        template_dict["rasterFunctionArguments"]["distance_type"] = distance_type

    flow_direction_type_list = ["D8","MFD","DINF"]
    if flow_direction_type is not None:
        if flow_direction_type.upper() not in flow_direction_type_list:
            raise RuntimeError('flow_direction_type should be one of the following D8, MFD, Dinf')
        template_dict["rasterFunctionArguments"]["flow_direction_type"] = flow_direction_type

    statistics_type_allowed_values = ["MINIMUM","WEIGHTED_MEAN","MAXIMUM"]
    if [element.lower() for element in statistics_type_allowed_values].count(statistics_type.lower()) <= 0 :
        raise RuntimeError('statistics_type can only be one of the following: '+ str(statistics_type_allowed_values))
    for element in statistics_type_allowed_values:
        if statistics_type.lower() == element.lower():
            template_dict["rasterFunctionArguments"]["statistics_type"] = statistics_type

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_stream_raster"] = raster_ra1
    function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra2
    if input_flow_direction_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def flow_accumulation(input_flow_direction_raster,
                      input_weight_raster=None,
                      data_type="FLOAT",
                      flow_direction_type= "D8"):

    """"
    Replaces cells of a raster corresponding to a mask
    with the values of the nearest neighbors.

    :param input_flow_direction_raster: Required. The input raster that shows the direction of flow out of each cell.
    :param input_weight_raster: An optional input raster for applying a weight to each cell.
    :param data_type: Optional. Choice List: INTEGER, FLOAT, DOUBLE
    :return: output raster with function applied

    """
    layer1, input_flow_direction_raster, raster_ra1 = _raster_input(input_flow_direction_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "FlowAccumulation_sa",
            "PrimaryInputParameterName" : "in_flow_direction_raster",
            "OutputRasterParameterName" : "out_accumulation_raster",
            "in_flow_direction_raster" : input_flow_direction_raster

        }
    }
    if input_weight_raster is not None:
        layer2, input_weight_raster, raster_ra2 = _raster_input(input_weight_raster)
        template_dict["rasterFunctionArguments"]["in_weight_raster"] = input_weight_raster

    data_type_list=["FLOAT","INTEGER","DOUBLE"]

    if data_type is not None:
        if data_type.upper() not in data_type_list:
            raise RuntimeError('data_type should be one of the following '+ str(data_type_list))
        template_dict["rasterFunctionArguments"]["data_type"] = data_type

    flow_direction_type_list = ["D8","MFD","DINF"]
    if flow_direction_type is not None:
        if flow_direction_type.upper() not in flow_direction_type_list:
            raise RuntimeError('flow_direction_type should be one of the following '+ str(flow_direction_type_list))
        template_dict["rasterFunctionArguments"]["flow_direction_type"] = flow_direction_type

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra1
    if input_weight_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_weight_raster"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def flow_direction(input_surface_raster,
                   force_flow="NORMAL",
                   flow_direction_type="D8",
                   generate_out_drop_raster=False):
    """
    .. image:: _static/images/flow_direction/flow_direction.png 

    The ``flow_direction`` task creates a raster of flow direction from each cell to its steepest downslope neighbor.

    This task supports three flow modeling algorithms. Those are D8, Multi Flow Direction (MFD), and D-Infinity (DINF).

    **D8 flow modeling algorithm**

    The D8 flow method models flow direction from each cell to its steepest downslope neighbor.

    The output of the FlowDirection task run with the D8 flow direction type is an integer 
    raster whose values range from 1-255. The values for each direction from the center are the following:

    .. image:: _static/images/flow_direction/D8.gif

    For example, if the direction of steepest drop was to the left of the current 
    processing cell, its flow direction would be coded at 16.

    The following are additional considerations for using the D8 flow method:

        * If a cell is lower than its eight neighbors, that cell is given the value 
          of its lowest neighbor, and flow is defined toward this cell. If multiple 
          neighbors have the lowest value, the cell is still given this value, but 
          flow is defined with one of the two methods explained below. This is used 
          to filter out one-cell sinks, which are considered noise.
        * If a cell has the same change in z-value in multiple directions and that 
          cell is part of a sink, the flow direction is referred to as undefined. In 
          such cases, the value for that cell in the output flow direction raster will 
          be the sum of those directions. For example, if the change in z-value is the 
          same both to the right (flow direction = 1) and down (flow direction = 4), 
          the flow direction for that cell is 5.
        * If a cell has the same change in z-value in multiple directions and is not 
          part of a sink, the flow directions is assigned with a lookup table defining 
          the most likely direction. See Greenlee (1987).
        * The output drop raster is calculated as the difference in z-value divided by 
          the path length between the cell centers, expressed in percentages. For adjacent 
          cells, this is analogous to the percent slop between cells. Across a flat area, 
          the distance becomes the distance to the nearest cell of lower elevation. 
          The result is a map of percent rise in the path of steepest descent from 
          each cell.
        * When calculating a drop raster in flat areas, the distance to diagonally 
          adjacent cells (1.41421 * cell size) is approximated by 1.5 * cell 
          size for improved performance.
        * With the forceFlow parameter set to the default value False, a cell 
          at the edge of the surface raster will flow towards the inner cell 
          with the steepest z-value. If the drop is less than or equal to zero, 
          the cell will flow out of the surface raster.    

    **MFD flow modeling algorithm**

    The MFD algorithm, described by Qin et al. (2007), partitions flow from a cell to all downslope neighbors. 
    A flow-partition exponent is created from an adaptive approach based on local terrain conditions and is used 
    to determine fraction of flow draining to all downslope neighbors.

    When the MFD flow direction output is added to a map, it only displays the D8 flow direction. 
    As MFD flow directions have potentially multiple values tied to each cell (each value corresponds 
    to proportion of flow to each downslope neighbor), it is not easily visualized. However, an MFD 
    flow direction output raster is an input recognized by the FlowAccumulation task that would utilize 
    the MFD flow directions to proportion and accumulate flow from each cell to all downslope neighbors.

    **DINF flow modeling algorithm**

    The DINF flow method, described by Tarboton (1997), determines flow direction as the steepest 
    downward slope on eight triangular facets formed in a 3x3 cell window centered on the cell of 
    interest. The flow direction output is a floating-point raster represented as a single angle in 
    degrees going counter-clockwise from 0 (due east) to 360 (also due east).

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_surface_raster                 Required. The input raster representing a continuous surface. 
    --------------------------------     --------------------------------------------------------------------
    force_flow                           Optional string. Specifies if edge cells will always flow outward or follow normal flow rules.

                                         Choice list: ['NORMAL', 'FORCE']

                                         The default value is 'NORMAL'.
    --------------------------------     --------------------------------------------------------------------
    flow_direction_type                  Optional string. Specifies the flow direction type to use.

                                         Choice list: ['D8', 'MFD', 'DINF']

                                         * ``D8`` is for the D8 flow direction type. This is the default.
                                         * ``MFD`` is for the Multi Flow Direction type.
                                         * ``DINF`` is for the D-Infinity type.

                                         The default value is 'D8'.
    --------------------------------     --------------------------------------------------------------------
    generate_out_drop_raster             Optional Boolean, determines whether out_drop_raster should be generated or not.
                                         Set this parameter to True, in order to generate the out_drop_raster.
                                         If set to true, the output will be a named tuple with name values being
                                         output_flow_direction_service and output_drop_service.
    ================================     ====================================================================
 
    :returns: output raster with function applied

    .. code-block:: python

            # Usage Example: To add an image to an existing image collection.
            flow_direction_output =  flow_direction(input_surface_raster=in_raster,
                                                    force_flow="NORMAL",
                                                    flow_direction_type="D8",
                                                    generate_out_drop_raster=True)

            out_var = flow_direction_output.save()

            out_var.output_flow_direction_service  # gives you the output flow direction imagery layer item

            out_var.output_drop_service # gives you the output drop raster imagery layer item

    """
    layer, input_surface_raster, raster_ra = _raster_input(input_surface_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "FlowDirection_sa",
            "PrimaryInputParameterName" : "in_surface_raster",
            "OutputRasterParameterName" : "out_flow_direction_raster",
            "in_surface_raster" : input_surface_raster
        }
    }

    force_flow_list = ["NORMAL","FORCE"]
    if force_flow is not None:
        if force_flow.upper() not in force_flow_list:
            raise RuntimeError('force_flow should be one of the following '+ str(force_flow_list))
        template_dict["rasterFunctionArguments"]["force_flow"] = force_flow


    flow_direction_type_list = ["D8","MFD","DINF"]
    if flow_direction_type is not None:
        if flow_direction_type.upper() not in flow_direction_type_list:
            raise RuntimeError('flow_direction_type should be one of the following '+ str(flow_direction_type_list))
        template_dict["rasterFunctionArguments"]["flow_direction_type"] = flow_direction_type


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_surface_raster"] = raster_ra

    if generate_out_drop_raster is True:
        return _gbl_clone_layer(layer, template_dict, function_chain_ra, out_drop_raster = generate_out_drop_raster, use_ra=True)

    return _gbl_clone_layer(layer, template_dict, function_chain_ra, out_drop_raster = generate_out_drop_raster)

def fill(input_surface_raster,
         zlimit=None):
    """
    Fills sinks in a surface raster to remove small imperfections in the data

    Parameters
    ----------
    :param input_surface_raster: Required. The input raster representing a continuous surface.
    :param zlimit: Optional. Data type - Double. Maximum elevation difference between a sink and
            its pour point to be filled.
            If the difference in z-values between a sink and its pour point is greater than the z_limit, that sink will not be filled.
            The value for z-limit must be greater than zero.
            Unless a value is specified for this parameter, all sinks will be filled, regardless of depth.

    :return: output raster with function applied

    """
    layer, input_surface_raster, raster_ra = _raster_input(input_surface_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Fill_sa",
            "PrimaryInputParameterName" : "in_surface_raster",
            "OutputRasterParameterName" : "out_surface_raster",
            "in_surface_raster" : input_surface_raster

        }
    }

    if zlimit is not None:
        template_dict["rasterFunctionArguments"]["z_limit"] = zlimit

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_surface_raster"] = raster_ra

    return _gbl_clone_layer(layer, template_dict, function_chain_ra)


def nibble(input_raster,
           input_mask_raster,
           nibble_values= "ALL_VALUES",
           nibble_no_data= "PRESERVE_NODATA",
           input_zone_raster=None):
    """
    Replaces cells of a raster corresponding to a mask
    with the values of the nearest neighbors.

    Parameters
    ----------
    :param input_raster: Required. The input rater to nibble.
                   The input raster can be either integer or floating point type.
    :param input_mask_raster: Required. The input raster to use as the mask.
    :param nibble_values: Optional. possbile options are "ALL_VALUES" and "DATA_ONLY".
        Default is "ALL_VALUES"
    :param nibble_no_data: Optional. PRESERVE_NODATA or PROCESS_NODATA possible values;
        Default is PRESERVE_NODATA.
    :param input_zone_raster: Optional. The input raster that defines the zones to use as the mask.

    :return: output raster with function applied

    """
    layer1, input_raster, raster_ra1 = _raster_input(input_raster)
    layer2, input_mask_raster, raster_ra2 = _raster_input(input_mask_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Nibble_sa",
            "PrimaryInputParameterName" : "in_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_raster" : input_raster,
            "in_mask_raster" : input_mask_raster,
            "nibble_values" : nibble_values,
            "nibble_nodata" : nibble_no_data

        }
    }

    nibble_values_list = ["ALL_VALUES","DATA_ONLY"]
    if nibble_values is not None:
        if nibble_values.upper() not in nibble_values_list:
            raise RuntimeError('nibble_values should be one of the following '+ str(nibble_values_list))
        template_dict["rasterFunctionArguments"]["nibble_values"] = nibble_values


    nibble_no_data_list = ["PRESERVE_NODATA","PROCESS_NODATA"]
    if nibble_no_data is not None:
         if nibble_no_data.upper() not in nibble_no_data_list:
             raise RuntimeError('nibble_nodata should be one of the following '+ str(nibble_no_data_list))
         template_dict["rasterFunctionArguments"]["nibble_nodata"] = nibble_no_data

    if input_zone_raster is not None:
        layer3, input_zone_raster, raster_ra3 = _raster_input(input_zone_raster)
        template_dict["rasterFunctionArguments"]["in_zone_raster"] = input_zone_raster

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_raster"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_mask_raster"] = raster_ra2
    if input_zone_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_zone_raster"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def stream_link(input_raster,
                input_flow_direction_raster):
    """
    Assigns unique values to sections of a raster linear network between intersections

    Parameters
    ----------
    :param input_raster: Required. An input raster that represents a linear stream network.
    :param input_flow_direction_raster: Required. The input raster that shows the direction of flow out of each cell
    :return: output raster with function applied

    """
    layer1, input_raster, raster_ra1 = _raster_input(input_raster)
    layer2, input_flow_direction_raster, raster_ra2 = _raster_input(input_flow_direction_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "StreamLink_sa",
            "PrimaryInputParameterName" : "in_stream_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_stream_raster" : input_raster,
            "in_flow_direction_raster" : input_flow_direction_raster
        }
    }


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_stream_raster"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def watershed(input_flow_direction_raster,
              input_pour_point_data,
              pour_point_field=None):
    """
    Replaces cells of a raster corresponding to a mask
    with the values of the nearest neighbors.

    Parameters
    ----------
    :param input_flow_direction_raster: Required raster layer. The input raster that shows the direction of flow out of each cell.
    :param input_pour_point_data: Required raster layer. This raster represents cells above
                            which the contributing area, or catchment, will be determined. All cells that
                            are not NoData will be used as source cells.
    :param pour_point_field: Optional string. Field used to assign values to the pour point locations.
                             For a raster pour point dataset, Value is used by default.
    :return: output raster with function applied

    """
    layer1, input_flow_direction_raster, raster_ra1 = _raster_input(input_flow_direction_raster)
    layer2, input_pour_point_data, raster_ra2 = _raster_input(input_pour_point_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Watershed_sa",
            "PrimaryInputParameterName" : "in_flow_direction_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_flow_direction_raster" : input_flow_direction_raster,
            "in_pour_point_data" : input_pour_point_data
        }
    }

    if pour_point_field is not None:
        template_dict["rasterFunctionArguments"]["pour_point_field"] = pour_point_field


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_pour_point_data"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
 " (or arcgis.raster.functions.gbl.distance_allocation() for allocation output), instead. ")
def calculate_travel_cost(in_source_data,
                          in_cost_raster=None,
                          in_surface_raster=None,
                          in_horizontal_raster=None,
                          in_vertical_raster=None,
                          horizontal_factor="BINARY",
                          vertical_factor="BINARY",
                          maximum_distance=None,
                          source_cost_multiplier=None,
                          source_start_cost=None,
                          source_resistance_rate=None,
                          source_capacity=None,
                          source_direction=None,
                          allocation_field=None,
                          generate_out_allocation_raster=False,
                          generate_out_backlink_raster=False):
    """

    Parameters
    ----------
    :param in_source_data:  Required. The layer that defines the sources to calculate the distance too. The layer
                            can be raster or feature.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location.

    :param in_horizontal_raster:  Optional. A raster defining the horizontal direction at each cell.

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.
                               Possible values are: "BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"

    :param vertical_factor:  Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)
                            Possible values are: "BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"

    :param maximum_distance:  Optional. The maximum distance to calculate out to. If no distance is provided, a default will
                             be calculated that is based on the locations of the input sources.

    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values.

    :param source_start_cost:  Optional. The starting cost from which to begin the cost calculations.

    :param source_resistance_rate:  Optional. This parameter simulates the increase in the effort to overcome costs
                                    as the accumulative cost increases.

    :param source_capacity:  Optional. Defines the cost capacity for the traveler for a source.

    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE

    :param allocation_field:  Optional. A field on theinputSourceRasterOrFeatures layer that holds the values that define each source.

    :param generate_out_backlink_raster:   Optional Boolean, determines whether out_backlink_raster should be generated or not.
                                           Set this parameter to True, in order to generate the out_backlink_raster.
                                           If set to true, the output will be a named tuple with name values being
                                           output_distance_service and output_backlink_service.
                                           eg,

                                           out_layer = calculate_travel_cost(in_source_data, generate_out_backlink_raster=True)
                                           out_var = out_layer.save()

                                           then,

                                           out_var.output_distance_service -> gives you the output distance imagery layer item

                                           out_var.output_backlink_service -> gives you the output backlink raster imagery layer item

    :param generate_out_allocation_raster:  Optional Boolean, determines whether out_allocation_raster should be generated or not.
                                            Set this parameter to True, in order to generate the out_backlink_raster.
                                            If set to true, the output will be a named tuple with name values being
                                            output_distance_service and output_allocation_service.
                                            eg,

                                            out_layer = calculate_travel_cost(in_source_data, generate_out_allocation_raster=False)
                                            out_var = out_layer.save()

                                            then,

                                            out_var.output_distance_service -> gives you the output distance imagery layer item

                                            out_var.output_allocation_service -> gives you the output allocation raster imagery layer item

    :param gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.

    :return: output raster with function applied
    """
    if isinstance (in_source_data, ImageryLayer):
        layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)
    else:
        raster_ra1 = _layer_input(in_source_data)
        input_source_data = raster_ra1
        layer1=raster_ra1


    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_surface_raster is not None:
        layer3, in_surface_raster, raster_ra3 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer4, in_horizontal_raster, raster_ra4 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer5, in_vertical_raster, raster_ra5 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CalculateTravelCost_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_distance_raster",
            "in_source_data" : input_source_data
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    horizontal_factor_list = ["BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"]
    if horizontal_factor.upper() not in horizontal_factor_list:
        raise RuntimeError('horizontal_factor should be one of the following '+ str(horizontal_factor_list))
    template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    vertical_factor_list = ["BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"]
    if vertical_factor.upper() not in vertical_factor_list:
        raise RuntimeError('vertical_factor should be one of the following '+ str(vertical_factor_list))
    template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if maximum_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = maximum_distance

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction

    if allocation_field is not None:
        template_dict["rasterFunctionArguments"]["allocation_field"] = allocation_field

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra3

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra4

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra5

    if isinstance(in_source_data, ImageryLayer):
        return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra, out_allocation_raster = generate_out_allocation_raster, out_backlink_raster = generate_out_backlink_raster, use_ra=True)
    else:
        return _feature_gbl_clone_layer(in_source_data, template_dict, function_chain_ra, out_allocation_raster = generate_out_allocation_raster, out_backlink_raster = generate_out_backlink_raster, use_ra=True)

def kernel_density(in_features,
                   population_field,
                   cell_size=None,
                   search_radius=None,
                   area_unit_scale_factor="SQUARE_MAP_UNITS",
                   out_cell_values="DENSITIES",
                   method="PLANAR",
                   in_barriers=None):
    """
    Calculates a magnitude-per-unit area from point or polyline features using a kernel function to
    fit a smoothly tapered surface to each point or polyline.
    For more information, see
    http://pro.arcgis.com/en/pro-app/help/data/imagery/kernel-density-global-function.htm

    Parameters
    ----------
    :param in_features:               Required. The input point or line features for which to calculate the density
    :param population_field:          Required. Field denoting population values for each feature. The Population
                                      Field is the count or quantity to be spread across the landscape to
                                      create a continuous surface. Values in the population field may be
                                      integer or floating point.
    :param cell_size:                 Optional. The pixel size for the output raster dataset. If the Cellsize has
                                      been set in the geoprocessing Environments it will be the default.
    :param search_radius:             Optional. The search radius within which to calculate density. Units are
                                      based on the linear unit of the projection.
    :param area_unit_scale_factor:    Optional. The desired area units of the output density values.

                                        - SQUARE_MAP_UNITS-For the square of the linear units of the output spatial reference.

                                        - SQUARE_MILES-For (U.S.) miles.

                                        - SQUARE_KILOMETERS-For kilometers.

                                        - ACRES For (U.S.) acres.

                                        - HECTARES-For hectares.

                                        - SQUARE_METERS-For meters.

                                        - SQUARE_YARDS-For (U.S.) yards.

                                        - SQUARE_FEET-For (U.S.) feet.

                                        - SQUARE_INCHES-For (U.S.) inches.

                                        - SQUARE_CENTIMETERS-For centimeters.

                                        - SQUARE_MILLIMETERS-For millimeters.
    :param out_cell_values: Optional. Determines what the values in the output raster represent.

                            - DENSITIES-The output values represent the predicted density value. This is the default.

                            - EXPECTED_COUNTS-The output values represent the predicted amount of the phenomenon within each
                              pixel. Since the pixel value is linked to the specified Cellsize, the resulting raster cannot be
                              resampled to a different pixel size and still represent the amount of the phenomenon.

    :param method: Optional. Determines whether to use a shortest path on a spheroid (geodesic) or a flat earth (planar) method.

                   - PLANAR-Uses planar distances between the features. This is the default.

                   - GEODESIC-Uses geodesic distances between features. This method takes into account the curvature
                     of the spheroid and correctly deals with data near the poles and the International dateline.
    :param in_barriers: Optional. The dataset that defines the barriers. The barriers can be a feature layer of polyline or polygon features. (Parameter available in ArcGIS Image Server 10.9 and higher.)
    :return: output raster
    """

    input_features = _layer_input(in_features)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "KernelDensity_sa",
            "PrimaryInputParameterName":"in_features",
            "OutputRasterParameterName":"out_raster",
            "in_features": input_features,
            "population_field":population_field,
            "RasterInfo":{"blockWidth" : 2048,
                          "blockHeight":256,
                          "bandCount":1,
                          "pixelType":9,
                          "firstPyramidLevel":1,
                          "maximumPyramidLevel":30,
                          "pixelSizeX":0,
                          "pixelSizeY" :0,
                          "type":"RasterInfo"}
        }
    }

    if search_radius is not None:
        template_dict["rasterFunctionArguments"]["search_radius"] = search_radius

    if cell_size is not None:
        template_dict["rasterFunctionArguments"]["cell_size"] = cell_size

    if area_unit_scale_factor is not None:
        area_unit_scale_factor_list = ["SQUARE_MAP_UNITS","SQUARE_MILES", "SQUARE_KILOMETERS", "ACRES","HECTARES","SQUARE_METERS","SQUARE_YARDS"
                                       "SQUARE_FEET","SQUARE_INCHES", "SQUARE_CENTIMETERS","SQUARE_MILLIMETERS"]
        if area_unit_scale_factor.upper() not in area_unit_scale_factor_list:
            raise RuntimeError('area_unit_scale_factor should be one of the following '+ str(area_unit_scale_factor_list))
        template_dict["rasterFunctionArguments"]["area_unit_scale_factor"] = area_unit_scale_factor

    out_cell_values_list = ["DENSITIES", "EXPECTED_COUNTS"]
    if out_cell_values.upper() not in out_cell_values_list:
        raise RuntimeError('out_cell_values should be one of the following '+ str(out_cell_values_list))
    template_dict["rasterFunctionArguments"]["out_cell_values"] = out_cell_values

    method_list = ["PLANAR", "GEODESIC"]
    if method.upper() not in method_list:
        raise RuntimeError('method should be one of the following '+ str(method_list))
    template_dict["rasterFunctionArguments"]["method"] = method

    if in_barriers is not None:
        input_barriers = _layer_input(in_barriers)
        template_dict["rasterFunctionArguments"]["in_barriers"] = input_barriers

    if isinstance(in_features, Item):
        in_features = in_features.layers[0]
    newlyr = ImageryLayer(in_features._url, in_features._gis)
    newlyr._fn = template_dict
    newlyr._fnra = template_dict
    newlyr._uses_gbl_function = True
    return newlyr

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.optimal_path_as_raster() instead.")
def cost_path(in_destination_data,
              in_cost_distance_raster,
              in_cost_backlink_raster,
              path_type="EACH_CELL",
              destination_field=None,
              force_flow_direction_convention=None,
             ):
    """
    Calculates the least-cost path from a source to a destination.

    Parameters
    ----------
    :param in_destination_data:     Required raster layer. A raster that identifies those cells from which the least-cost
                                    path is determined to the least costly source. The input raster layer
                                    consists of cells that have valid values (zero is a valid value), and the remaining
                                    cells must be assigned NoData.
    :param in_cost_distance_raster: Required. The name of a cost distance raster to be used to determine the least-cost path from
                                    the destination locations to a source. The cost distance raster is usually created
                                    with the Cost Distance, Cost Allocation or Cost Back Link tools. The cost distance
                                    raster stores, for each cell, the minimum accumulative cost distance over a cost
                                    surface from each cell to a set of source cells.
    :param in_cost_backlink_raster: Required. The name of a cost back link raster used to determine the path to return to a source
                                    via the least-cost path. For each cell in the back link raster, a value identifies
                                    the neighbor that is the next cell on the least accumulative cost path from the cell
                                    to a single source cell or set of source cells.
    :param path_type:               Optional. A keyword defining the manner in which the values and zones on the input destination
                                    data will be interpreted in the cost path calculations.

                                    - EACH_CELL - For each cell with valid values on the input destination data, a least-cost \
                                    path is determined and saved on the output raster. With this option, each cell of the \
                                    input destination data is treated separately, and a least-cost path is determined for \
                                    each from cell.

                                    - EACH_ZONE - For each zone on the input destination data, a least-cost path is determined \
                                    and saved on the output raster. With this option, the least-cost path for each zone \
                                    begins at the cell with the lowest cost distance weighting in the zone.

                                    - BEST_SINGLE - For all cells on the input destination data, the least-cost path is derived \
                                    from the cell with the minimum of the least-cost paths to source cells.
    :param destination_field:       Optional. The field used to obtain values for the destination locations. Input feature data must
                                    contain at least one valid field.
    :param force_flow_direction_convention: Optional boolean. Set to True to force flow direction convention for backlink raster

    :return: output raster with function applied
    """
    layer1, in_destination_data, raster_ra1 = _raster_input(in_destination_data)
    layer2, in_cost_distance_raster, raster_ra2 = _raster_input(in_cost_distance_raster)
    layer3,  in_cost_backlink_raster, raster_ra3 = _raster_input(in_cost_backlink_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CostPath_sa",
            "PrimaryInputParameterName":"in_destination_data",
            "OutputRasterParameterName":"out_raster",
            "in_destination_data": in_destination_data,
            "in_cost_distance_raster": in_cost_distance_raster,
            "in_cost_backlink_raster": in_cost_backlink_raster
        }
    }

    if path_type is not None:
        path_type_list = ["EACH_CELL", "EACH_ZONE", "BEST_SINGLE"]
        if path_type.upper() not in path_type_list:
            raise RuntimeError('path_type should be one of the following '+ str(path_type_list))
        template_dict["rasterFunctionArguments"]["path_type"] = path_type

    if destination_field is not None:
        template_dict["rasterFunctionArguments"]["destination_field"] = destination_field

    if force_flow_direction_convention is not None:
        template_dict["rasterFunctionArguments"]["force_flow_direction_convention"] = force_flow_direction_convention

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_destination_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_cost_distance_raster"] = raster_ra2
    function_chain_ra["rasterFunctionArguments"]["in_cost_backlink_raster"] = raster_ra3

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
"with value specified for output_source_direction_raster_name, instead.")
def euclidean_direction(in_source_data,
                        cell_size=None,
                        max_distance=None,
                        distance_method="PLANAR",
                        in_barrier_data=None):
    """
    Calculates, for each cell, the Euclidean distance to the closest source.

    Parameters
    ----------
    :param in_source_data:  Required raster layer. The input source locations. This is a raster that
                            identifies the cells or locations to which the Euclidean distance for
                            every output cell location is calculated. For rasters, the input type
                            can  be integer or floating point.
    :param cell_size:       Optional. Defines the threshold that the accumulative distance values cannot
                            exceed. If an accumulative Euclidean distance value exceeds this
                            value, the output value for the cell location will be NoData. The default
                            distance is to the edge of the output raster.
    :param max_distance:    Optional. Defines the threshold distance within which the direction to the 
                            closest source will be calculated. If the distance to the nearest source 
                            exceeds this, the output for that cell will be NoData. The default distance 
                            is to the extent of the output raster.
    :param distance_method: Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.
    :param in_barrier_data: Optional barrier raster. The input raster that defines the barriers. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 

    :return: output raster with function applied
    """
    layer, in_source_data, raster_ra = _raster_input(in_source_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "EucDirection_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_direction_raster",
            "in_source_data": in_source_data,

        }
    }

    if in_barrier_data is not None:
        layer2, in_barrier_data, raster_ra2 = _raster_input(in_barrier_data)
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data

    if cell_size is not None:
        template_dict["rasterFunctionArguments"]["cell_size"] = cell_size

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    distance_method_list = ["PLANAR","GEODESIC"]
    if distance_method is not None:
        if distance_method.upper() not in distance_method_list:
            raise RuntimeError('distance_method should be one of the following '+ str(distance_method_list))
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra

    if in_barrier_data is not None:
        function_chain_ra["rasterFunctionArguments"]["in_barrier_data"] = raster_ra2

    return _gbl_clone_layer(layer, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
                                            "with value specified for output_back_direction_raster_name, instead.")
def cost_backlink(in_source_data,
                  in_cost_raster,
                  max_distance=None,
                  source_cost_multiplier=None,
                  source_start_cost=None,
                  source_resistance_rate=None,
                  source_capacity=None,
                  source_direction=None):
    """
    Calculates the least accumulative cost distance for each cell from or to the least-cost
    source over a cost surface.

    Parameters
    ----------
    :param in_source_data: Required. The input raster that identifies the pixels or locations to which the
                            least accumulated cost distance for every output pixel location is
                            calculated. The Source Raster can be an integer or a floating-point value.
    :param in_cost_raster: Required. A raster defining the cost or impedance to move planimetrically through each pixel.
                            The value at each pixel location represents the cost-per-unit distance for moving
                            through it. Each pixel location value is multiplied by the pixel resolution, while
                            also compensating for diagonal movement to obtain the total cost of passing through
                            the pixel.
    :param max_distance: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative cost
                            distance exceeds this value, the output value for the pixel location will be NoData.
                            The maximum distance defines the extent for which the accumulative cost distances are
                            calculated. The default distance is to the edge of the output raster.
    :param source_cost_multiplier: Optional. The threshold that the accumulative cost values cannot exceed. If an accumulative
                            cost distance exceeds this value, the output value for the pixel location will be
                            NoData. The maximum distance defines the extent for which the accumulative cost
                            distances are calculated. The default distance is to the edge of the output raster.
    :param source_start_cost: Optional. The starting cost from which to begin the cost calculations. This parameter allows
                            for the specification of the fixed cost associated with a source. Instead of starting
                            at a cost of 0, the cost algorithm will begin with the value set here.
                            The default is 0. The value must be 0 or greater. A numeric (double) value or a field
                            from the Source Raster can be used for this parameter.
    :param source_resistance_rate: Optional. This parameter simulates the increase in the effort to overcome costs as the
                            accumulative cost increases. It is used to model fatigue of the traveler. The growing
                            accumulative cost to reach a pixel is multiplied by the resistance rate and added to
                            the cost to move into the subsequent pixel.
                            It is a modified version of a compound interest rate formula that is used to calculate
                            the apparent cost of moving through a pixel. As the value of the resistance rate increases,
                            it increases the cost of the pixels that are visited later. The greater the resistance rate,
                            the higher the cost to reach the next pixel, which is compounded for each subsequent movement.
                            Since the resistance rate is similar to a compound rate and generally the accumulative cost
                            values are very large, small resistance rates are suggested, such as 0.005 or even smaller,
                            depending on the accumulative cost values.
                            The default is 0. The values must be 0 or greater. A numeric (double) value or a field from
                            the Source Raster can be used for this parameter.
    :param source_capacity: Optional. Defines the cost capacity for the traveler for a source. The cost calculations continue for
                            each source until the specified capacity is reached.
                            The default capacity is to the edge of the output raster. The values must be greater than 0.
                            A double numeric value or a field from the Source Raster can be used for this parameter.
    :param source_direction: Optional. Defines the direction of the traveler when applying the source resistance rate and the source
                            starting cost.

                            - FROM_SOURCE - The source resistance rate and source starting cost will be applied beginning \
                            at the input source and moving out to the nonsource cells. This is the default.

                            - TO_SOURCE - The source resistance rate and source starting cost will be applied beginning at \
                            each nonsource cell and moving back to the input source.

                            Either specify the FROM_SOURCE or TO_SOURCE keyword, which will be applied to all sources,
                            or specify a field in the Source Raster that contains the keywords to identify the direction
                            of travel for each source. That field must contain the string FROM_SOURCE or TO_SOURCE.

    :return: output raster with function applied
    """
    layer1, in_source_data, raster_ra1 = _raster_input(in_source_data)
    layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CostBackLink_sa",
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_backlink_raster",
            "in_source_data": in_source_data,
            "in_cost_raster": in_cost_raster

        }
    }

    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_cost_raster"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def region_group(in_raster,
                 number_of_neighbor_cells="FOUR",
                 zone_connectivity="WITHIN",
                 add_link = "ADD_LINK",
                 excluded_value = 0):
    """
    Records, for each cell in the output, the identity of the connected region to which that cell
    belongs. A unique number is assigned to each region.

    Parameters
    ----------
    :param in_raster:  Required. The input raster for which unique connected regions of 
                       cells will be identified. It must be of integer type.

    :param number_of_neighbor_cells: Optional. The number of neighboring cells to use when evaluating 
                                     connectivity between cells that define a region.
                                     The default is FOUR.

                                      - FOUR - Connectivity is evaluated for the four nearest (orthogonal) \
                                      neighbors of each input cell.

                                      - EIGHT - Connectivity is evaluated for the eight nearest neighbors \
                                      (both orthogonal and diagonal) of each input cell.


    :param zone_connectivity:  Optional. Defines which cell values should be considered when testing for connectivity.
                               The default is WITHIN.

                                - WITHIN - Connectivity for a region is evaluated for input cells that are part of \
                                the same zone (cell value). The only cells that can be grouped are cells \
                                from the same zone that meet the spatial requirements of connectivity \
                                specified by the number_of_neighbor_cells parameter (four or eight).

                                - CROSS - Connectivity for a region is evaluated between cells of any value, \
                                except for the zone cells identified to be excluded by the \
                                excluded_value parameter, and subject to the spatial requirements \
                                specified by the number_of_neighbor_cells parameter.

    :param add_link: Optional. Specifies whether a link field will be added to the table of the output 
                     when the zone_connectivity parameter is set to WITHIN. It is ignored if that 
                     parameter is set to CROSS.

                      - ADD_LINK - A LINK field will be added to the table of the output raster. \
                      This field stores the value of the zone to which the cells of each region \
                      in the output belong, according to the connectivity rule defined in \
                      the number_of_neighbor_cells parameter. This is the default.

                      - NO_LINK - A LINK field will not be added. The attribute table for the output \
                      raster will only contain the Value and Count fields.

    :param excluded_value: Optional. A value that excludes all cells of that zone value from the 
                           connectivity evaluation. If a cell location contains the value, no 
                           spatial connectivity will be evaluated, regardless of how the number 
                           of neighbors is specified.

                           Cells with the excluded value will be treated in a similar way 
                           to NoData cells, and are eliminated from consideration in the 
                           operation. Input cells that contain the excluded value 
                           will receive 0 on the output raster. The excluded value is 
                           similar to the concept of a background value.

                           If a zone in the input raster has a value of 0, to have 
                           that zone be included in the operation, specify a value for 
                           this parameter that is not present in the input. For example, 
                           if an input raster has values of 0, 1, 2, and 3, specify an 
                           excluded_value of 99. Otherwise, all cells of value 0 in 
                           the input will be 0 in the output, and will also not have 
                           their individual regions determined.

    :return: output raster with function applied
    """
    layer, in_raster, raster_ra = _raster_input(in_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "RegionGroup_sa",
            "PrimaryInputParameterName":"in_raster",
            "OutputRasterParameterName":"out_raster",
            "in_raster": in_raster,

        }
    }

    if number_of_neighbor_cells is not None:
        if number_of_neighbor_cells.upper() == "EIGHT" or number_of_neighbor_cells.upper() == "FOUR":
            template_dict["rasterFunctionArguments"]["number_neighbors"] = number_of_neighbor_cells.upper()
        else:
            raise RuntimeError("number_of_neighbor_cells should either be 'EIGHT' or 'FOUR' ")

    if zone_connectivity is not None:
        if zone_connectivity.upper() == "WITHIN" or zone_connectivity.upper() == "CROSS":
            template_dict["rasterFunctionArguments"]["zone_connectivity"] = zone_connectivity.upper()
        else:
            raise RuntimeError("zone_connectivity should either be 'WITHIN' or 'CROSS' ")

    if add_link is not None:
        if add_link.upper() == "ADD_LINK" or add_link.upper() == "NO_LINK":
            template_dict["rasterFunctionArguments"]["add_link"] = add_link.upper()
        else:
            raise RuntimeError("add_link should either be 'ADD_LINK' or 'NO_LINK' ")

    if excluded_value is not None:
        template_dict["rasterFunctionArguments"]["excluded_value"] = excluded_value

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_raster"] = raster_ra

    return _gbl_clone_layer(layer, template_dict, function_chain_ra)


def corridor(in_distance_raster1,
             in_distance_raster2):
    """
    Calculates the sum of accumulative costs for two input accumulative cost rasters.

    Parameters
    ----------
    :param in_distance_raster1: Required. The first input distance raster.
                                It should be an accumulated cost distance output from a distance function
                                such as cost_distance or path_distance.


    :param in_distance_raster2: Required. The second input distance raster.
                                It should be an accumulated cost distance output from a distance function
                                such as cost_distance or path_distance.

    :return: output raster with function applied
    """
    layer1, in_distance_raster1, raster_ra1 = _raster_input(in_distance_raster1)
    layer2, in_distance_raster2, raster_ra2 = _raster_input(in_distance_raster2)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Corridor_sa",
            "PrimaryInputParameterName":"in_distance_raster1",
            "OutputRasterParameterName":"out_raster",
            "in_distance_raster1": in_distance_raster1,
            "in_distance_raster2": in_distance_raster2

        }
    }

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_distance_raster1"] = raster_ra1
    function_chain_ra["rasterFunctionArguments"]["in_distance_raster2"] = raster_ra2

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation() instead. ")
def path_distance(in_source_data,
                  in_cost_raster=None,
                  in_surface_raster=None,
                  in_horizontal_raster=None,
                  in_vertical_raster=None,
                  horizontal_factor="BINARY",
                  vertical_factor="BINARY",
                  maximum_distance=None,
                  source_cost_multiplier=None,
                  source_start_cost=None,
                  source_resistance_rate=None,
                  source_capacity=None,
                  source_direction=None):
    """
    Calculates, for each cell, the least accumulative cost distance from or to the least-cost source,
    while accounting for surface distance along with horizontal and vertical cost factors

    Parameters
    ----------
    :param in_source_data:  Required. The input source locations.
                            This is a raster that identifies the cells or locations from or to which the
                            least accumulated cost distance for every output cell location is calculated.

                            The raster input type can be integer or floating point.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.
                             The value at each cell location represents the cost-per-unit distance for moving through the cell.
                             Each cell location value is multiplied by the cell resolution while also
                             compensating for diagonal movement to obtain the total cost of passing through the cell.
                             The values of the cost raster can be integer or floating point,
                             but they cannot be negative or zero (you cannot have a negative or zero cost).

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location.
                               The values are used to calculate the actual surface distance covered when
                               passing between cells.

    :param in_horizontal_raster: Optional. A raster defining the horizontal direction at each cell.
                                 The values on the raster must be integers ranging from 0 to 360, with 0 degrees being north,
                                 or toward the top of the screen, and increasing clockwise. Flat areas should be given a value of -1.
                                 The values at each location will be used in conjunction with the {horizontal_factor} to determine
                                 the horizontal cost incurred when moving from a cell to its neighbors.

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell. The values are used for calculating the slope
                                used to identify the vertical factor incurred when moving from one cell to another.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.
                               Possible values are: "BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"

    :param vertical_factor: Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)
                            Possible values are: "BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"

    :param maximum_distance:  Optional. Defines the threshold that the accumulative cost values cannot exceed.
                              If an accumulative cost distance value exceeds this value, the output value for the cell
                              location will be NoData. The maximum distance defines the extent for which the accumulative cost distances are calculated.

                              The default distance is to the edge of the output raster.

    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values.

    :param source_start_cost: Optional. The starting cost from which to begin the cost calculations.

    :param source_resistance_rate:  Optional. This parameter simulates the increase in the effort to overcome costs
                                    as the accumulative cost increases.  It is used to model fatigue of the traveler.
                                    The growing accumulative cost to reach a cell is multiplied by the resistance rate
                                    and added to the cost to move into the subsequent cell.

    :param source_capacity:  Optional. Defines the cost capacity for the traveler for a source.
                             The cost calculations continue for each source until the specified capacity is reached.
                             The values must be greater than zero. The default capacity is to the edge of the output raster.

    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE

    :return: output raster with function applied
    """
    layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)

    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_surface_raster is not None:
        layer3, in_surface_raster, raster_ra3 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer4, in_horizontal_raster, raster_ra4 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer5, in_vertical_raster, raster_ra5 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "PathDistance_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_distance_raster",
            "in_source_data" : input_source_data
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    horizontal_factor_list = ["BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"]
    if horizontal_factor is not None:
        if horizontal_factor.upper() not in horizontal_factor_list:
            raise RuntimeError('horizontal_factor should be one of the following '+ str(horizontal_factor_list))
        template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    vertical_factor_list = ["BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"]
    if vertical_factor is not None:
        if vertical_factor.upper() not in vertical_factor_list:
            raise RuntimeError('vertical_factor should be one of the following '+ str(vertical_factor_list))
        template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if maximum_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = maximum_distance

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction is not None:
            if source_direction.upper() not in source_direction_list:
                raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
            template_dict["rasterFunctionArguments"]["source_direction"] = source_direction

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra3

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra4

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra5


    return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_allocation() instead.")
def path_distance_allocation(in_source_data,
                  in_cost_raster=None,
                  in_surface_raster=None,
                  in_horizontal_raster=None,
                  in_vertical_raster=None,
                  horizontal_factor="BINARY",
                  vertical_factor="BINARY",
                  maximum_distance=None,
                  in_value_raster=None,
                  source_field = None,
                  source_cost_multiplier=None,
                  source_start_cost=None,
                  source_resistance_rate=None,
                  source_capacity=None,
                  source_direction=None):

    """
    Calculates the least-cost source for each cell based on the least accumulative cost over a cost surface,
    while accounting for surface distance along with horizontal and vertical cost factors.

    Parameters
    ----------
    :param in_source_data:  Required. The input source locations.
                            This is a raster that identifies the cells or locations from or to which
                            the least accumulated cost distance for every output cell location is calculated.

                            For rasters, the input type can be integer or floating point.

                            If the input source raster is floating point, the {in_value_raster} must be set, and it must be of integer type.
                            The value raster will take precedence over any setting of the {source_field}.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location.

    :param in_horizontal_raster:  Optional. A raster defining the horizontal direction at each cell.

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.
                               Possible values are: "BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"

    :param vertical_factor: Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)
                            Possible values are: "BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"

    :param maximum_distance: Optional. Defines the threshold that the accumulative cost values cannot exceed.

    :param in_value_raster:  Optional. The input integer raster that identifies the zone values that should be
                             used for each input source location.
                             For each source location cell, the value defined by the {in_value_raster} will be
                             assigned to all cells allocated to the source location for the computation.
                             The value raster will take precedence over any setting for the {source_field}.

    :param source_field:   Optional. The field used to assign values to the source locations. It must be of integer type.
                           If the {in_value_raster} has been set, the values in that input will have precedence over any setting for the {source_field}.

    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values.
                                    Allows for control of the mode of travel or the magnitude at a source. The greater the multiplier,
                                    the greater the cost to move through each cell.

                                    The values must be greater than zero. The default is 1.

    :param source_start_cost:   Optional. The starting cost from which to begin the cost calculations.
                                Allows for the specification of the fixed cost associated with a source. Instead of starting at a cost of zero,
                                the cost algorithm will begin with the value set by source_start_cost.

                                The values must be zero or greater. The default is 0.

    :param source_resistance_rate:  Optional. This parameter simulates the increase in the effort to overcome costs as the accumulative cost increases.
                                    It is used to model fatigue of the traveler. The growing accumulative cost to reach a cell is multiplied by
                                    the resistance rate and added to the cost to move into the subsequent cell.

    :param source_capacity:  Optional. Defines the cost capacity for the traveler for a source.
                             The cost calculations continue for each source until the specified capacity is reached.

                             The values must be greater than zero. The default capacity is to the edge of the output raster.

    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE

    :return: output raster with function applied
    """

    layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)

    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_surface_raster is not None:
        layer3, in_surface_raster, raster_ra3 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer4, in_horizontal_raster, raster_ra4 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer5, in_vertical_raster, raster_ra5 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "PathAllocation_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_allocation_raster",
            "in_source_data" : input_source_data
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    horizontal_factor_list = ["BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"]
    if horizontal_factor is not None:
        if horizontal_factor.upper() not in horizontal_factor_list:
            raise RuntimeError('horizontal_factor should be one of the following '+ str(horizontal_factor_list))
        template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    vertical_factor_list = ["BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"]
    if vertical_factor is not None:
        if vertical_factor.upper() not in vertical_factor_list:
            raise RuntimeError('vertical_factor should be one of the following '+ str(vertical_factor_list))
        template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if maximum_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = maximum_distance

    if in_value_raster is not None:
        layer6, in_value_raster, raster_ra6 = _raster_input(in_value_raster)
        template_dict["rasterFunctionArguments"]["in_value_raster"] = in_value_raster

    if source_field is not None:
        template_dict["rasterFunctionArguments"]["source_field"] = source_field

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra3

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra4

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra5

    if in_value_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_value_raster"] = raster_ra6

    return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
                                            "with value specified for output_back_direction_raster_name, instead.")
def path_distance_back_link(in_source_data,
                  in_cost_raster=None,
                  in_surface_raster=None,
                  in_horizontal_raster=None,
                  in_vertical_raster=None,
                  horizontal_factor="BINARY",
                  vertical_factor="BINARY",
                  maximum_distance=None,
                  source_cost_multiplier=None,
                  source_start_cost=None,
                  source_resistance_rate=None,
                  source_capacity=None,
                  source_direction=None):
    """
    Defines the neighbor that is the next cell on the least accumulative cost path to the least-cost source,
    while accounting for surface distance along with horizontal and vertical cost factors.

    Parameters
    ----------
    :param in_source_data:  Required. The input source locations.

                            This is a raster that identifies the cells or locations from
                            or to which the least accumulated cost distance for every output cell location is calculated.

                            For rasters, the input type can be integer or floating point.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.

                             The value at each cell location represents the cost-per-unit distance for moving through the cell.
                             Each cell location value is multiplied by the cell resolution while also compensating for diagonal
                             movement to obtain the total cost of passing through the cell.

                             The values of the cost raster can be integer or floating point, but they cannot be negative or
                             zero (you cannot have a negative or zero cost).

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location. The values are used to calculate the actual
                               surface distance covered when passing between cells.

    :param in_horizontal_raster: Optional. A raster defining the horizontal direction at each cell.
                                 The values on the raster must be integers ranging from 0 to 360, with 0 degrees being north, or toward
                                 the top of the screen, and increasing clockwise. Flat areas should be given a value of -1.
                                 The values at each location will be used in conjunction with the {horizontal_factor} to determine the
                                 horizontal cost incurred when moving from a cell to its neighbors.

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell. The values are used for calculating the slope
                                used to identify the vertical factor incurred when moving from one cell to another.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.
                               Possible values are: "BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"

    :param vertical_factor: Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)
                            Possible values are: "BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"

    :param maximum_distance:  Optional. Defines the threshold that the accumulative cost values cannot exceed. If an accumulative cost distance
                              value exceeds this value, the output value for the cell location will be NoData. The maximum distance
                              defines the extent for which the accumulative cost distances are calculated.

                              The default distance is to the edge of the output raster.

    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values. Allows for control of the mode of travel or the magnitude at a source.
                                    The greater the multiplier, the greater the cost to move through each cell. The values must be greater than zero.
                                    The default is 1.

    :param source_start_cost:  Optional. The starting cost from which to begin the cost calculations. Allows for the specification of the fixed cost associated with a source.
                               Instead of starting at a cost of zero, the cost algorithm will begin with the value set by source_start_cost.

                               The values must be zero or greater. The default is 0.

    :param source_resistance_rate:  Optional. This parameter simulates the increase in the effort to overcome costs as the accumulative cost increases.
                                    It is used to model fatigue of the traveler. The growing accumulative cost to reach a cell is multiplied
                                    by the resistance rate and added to the cost to move into the subsequent cell.

    :param source_capacity:  Optional. Defines the cost capacity for the traveler for a source.
                             The cost calculations continue for each source until the specified capacity is reached.

                             The values must be greater than zero. The default capacity is to the edge of the output raster.

    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE

    :return: output raster with function applied
    """

    layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)

    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_surface_raster is not None:
        layer3, in_surface_raster, raster_ra3 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer4, in_horizontal_raster, raster_ra4 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer5, in_vertical_raster, raster_ra5 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "PathBackLink_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_backlink_raster",
            "in_source_data" : input_source_data
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    horizontal_factor_list = ["BINARY", "LINEAR", "FORWARD", "INVERSE_LINEAR"]
    if horizontal_factor is not None:
        if horizontal_factor.upper() not in horizontal_factor_list:
            raise RuntimeError('horizontal_factor should be one of the following '+ str(horizontal_factor_list))
        template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    vertical_factor_list = ["BINARY", "LINEAR", "SYMMETRIC_LINEAR", "INVERSE_LINEAR",
                            "SYMMETRIC_INVERSE_LINEAR", "COS", "SEC", "COS_SEC", "SEC_COS"]
    if vertical_factor is not None:
        if vertical_factor.upper() not in vertical_factor_list:
            raise RuntimeError('vertical_factor should be one of the following '+ str(vertical_factor_list))
        template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if maximum_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = maximum_distance


    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_start_cost is not None:
        template_dict["rasterFunctionArguments"]["source_start_cost"] = source_start_cost

    if source_resistance_rate is not None:
        template_dict["rasterFunctionArguments"]["source_resistance_rate"] = source_resistance_rate

    if source_capacity is not None:
        template_dict["rasterFunctionArguments"]["source_capacity"] = source_capacity

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra3

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra4

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra5

    return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
 " (or arcgis.raster.functions.gbl.distance_allocation() for allocation output) instead. ")
def calculate_distance(in_source_data,
                       maximum_distance=None,
                       output_cell_size=None,
                       allocation_field=None,
                       generate_out_allocation_raster=False,
                       generate_out_direction_raster=False,
                       generate_out_back_direction_raster=False,
                       in_barrier_data=None,
                       distance_method='PLANAR'):
    """
    Calculates the Euclidean distance, direction, and allocation from a single source or set of sources.

    Parameters
    ----------
    :param in_source_data:  Required. The layer that defines the sources to calculate the distance to.
                            The layer can be raster or feature. To use a raster input, it must
                            be of integer type.

    :param maximum_distance:  Optional. Defines the threshold that the accumulative distance values
                              cannot exceed. If an accumulative Euclidean distance value exceeds
                              this value, the output value for the cell location will be NoData.
                              The default distance is to the edge of the output raster.

                              Supported units: Meters | Kilometers | Feet | Miles

                              Example:

                              {"distance":"60","units":"Meters"}

    :param output_cell_size:   Optional. Specify the cell size to use for the output raster.

                               Supported units: Meters | Kilometers | Feet | Miles

                               Example:
                               {"distance":"60","units":"Meters"}

    :param allocation_field:  Optional. A field on the input_source_data layer that holds the values that
                              defines each source.

                              It can be an integer or a string field of the source dataset.

                              The default for this parameter is 'Value'.

    :param generate_out_direction_raster:  Optional Boolean, determines whether out_direction_raster should be generated or not.
                                           Set this parameter to True, in order to generate the out_direction_raster.
                                           If set to true, the output will be a named tuple with name values being
                                           output_distance_service and output_direction_service.
                                           eg,

                                           out_layer = calculate_distance(in_source_data, generate_out_direction_raster=True)
                                           out_var = out_layer.save()

                                           then,

                                           out_var.output_distance_service -> gives you the output distance imagery layer item
                                           out_var.output_direction_service -> gives you the output backlink raster imagery layer item

                                           The output direction raster is in degrees, and indicates the
                                           direction to return to the closest source from each cell center.
                                           The values on the direction raster are based on compass directions,
                                           with 0 degrees reserved for the source cells. Thus, a value of 90
                                           means 90 degrees to the East, 180 is to the South, 270 is to the west,
                                           and 360 is to the North.

    :param generate_out_allocation_raster:  Optional Boolean, determines whether out_allocation_raster should be generated or not.
                                            Set this parameter to True, in order to generate the out_backlink_raster.
                                            If set to true, the output will be a named tuple with name values being
                                            output_distance_service and output_allocation_service.
                                            eg,

                                            out_layer = calculate_distance(in_source_data, generate_out_allocation_raster=True)
                                            out_var = out_layer.save()

                                            then,

                                            out_var.output_distance_service -> gives you the output distance imagery layer item
                                            out_var.output_allocation_service -> gives you the output allocation raster imagery layer item

                                            This parameter calculates, for each cell, the nearest source based
                                            on Euclidean distance.
    :param generate_out_back_direction_raster: Optional Boolean, determines whether out_back_direction_raster should be generated or not.
                                               Set this parameter to True, in order to generate the out_back_direction_raster.
                                               If set to true, the output will be a named tuple with name values being
                                               output_distance_service and out_back_direction_service.
                                               eg,

                                               out_layer = calculate_distance(in_source_data, generate_out_back_direction_raster=True)
                                               out_var = out_layer.save()

                                               then,

                                               out_var.output_distance_service -> gives you the output distance imagery layer item
                                               out_var.out_back_direction_service -> gives you the output back direction raster imagery layer item

    :param in_barrier_data: Optional barrier raster. The input raster that defines the barriers. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 
  
    :param distance_method: Optional String. Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            Planar - Planar measurements use 2D Cartesian mathematics to calculate 
                            length and area. The option is only available when measuring in a 
                            projected coordinate system and the 2D plane of that coordinate system 
                            will be used as the basis for the measurements. This is the default.

                            Geodesic - The shortest line between two points on the earth's surface 
                            on a spheroid (ellipsoid). Therefore, regardless of input or output 
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.


    :return: output raster with function applied
    """
    if isinstance (in_source_data, ImageryLayer):
        layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)
    else:
        raster_ra1 = _layer_input(in_source_data)
        input_source_data = raster_ra1
        layer1=raster_ra1

    if in_barrier_data is not None:
        layer2, in_barrier_data, raster_ra2 = _raster_input(in_barrier_data)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "CalculateDistance_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_distance_raster",
            "in_source_data" : input_source_data
        }
    }

    if maximum_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = maximum_distance

    if output_cell_size is not None:
        template_dict["rasterFunctionArguments"]["output_cell_size"] = output_cell_size

    if allocation_field is not None:
        template_dict["rasterFunctionArguments"]["allocation_field"] = allocation_field

    if distance_method is not None:
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1

    if in_barrier_data is not None:
        function_chain_ra['rasterFunctionArguments']["in_barrier_data"] = raster_ra2

    if isinstance(in_source_data, ImageryLayer):
        return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra, out_allocation_raster = generate_out_allocation_raster, out_direction_raster = generate_out_direction_raster, out_back_direction_raster=generate_out_back_direction_raster, use_ra=True)
    else:
        return _feature_gbl_clone_layer(in_source_data, template_dict, function_chain_ra, out_allocation_raster = generate_out_allocation_raster, out_direction_raster = generate_out_direction_raster, out_back_direction_raster=generate_out_back_direction_raster, use_ra=True)

@deprecated(deprecated_in="1.8.1", details="Please use arcgis.raster.functions.gbl.distance_accumulation()"
                                           "with value specified for output_back_direction_raster_name, instead.")
def euclidean_back_direction(in_source_data,
                             cell_size=None,
                             max_distance=None,
                             distance_method="PLANAR", 
                             in_barrier_data=None):
    """
    Calculates, for each cell, the direction, in degrees, to the neighboring cell along 
    the shortest path back to the closest source while avoiding barriers.

    The direction is calculated from each cell center to the center of the source cell 
    that's nearest to it.

    The range of values is from 0 degrees to 360 degrees, with 0 reserved for the source cells.
    Due east (right) is 90 and the values increase clockwise (180 is south, 270 is west, 
    and 360 is north).

    For more information, see 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/euclidean-back-direction-function.htm

    Parameters
    ----------
    :param in_source_data: Required; The input raster that identifies the pixels or locations to
                            which the Euclidean direction for every output cell location is calculated.
                            The input type can be an integer or a floating-point value.
    :param cell_size:  Optional. The pixel size at which the output raster will be created. If the cell
                       size was explicitly set in Environments, that will be the default cell size. 
                       If Environments was not set, the output cell size will be the same as the 
                       Source Raster
    :param max_distance: Optional. The threshold that the accumulative distance values cannot exceed. If an
                         accumulative Euclidean distance exceeds this value, the output value for
                         the pixel location will be NoData. The default distance is to the edge 
                         of the output raster
    :param distance_method: Optional. Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.
    :param in_barrier_data: Optional barrier raster. The input raster that defines the barriers. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 
    :return: output raster with function applied
    """
    layer, in_source_data, raster_ra = _raster_input(in_source_data)
                  
    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "EucBackDirection_sa",           
            "PrimaryInputParameterName":"in_source_data",
            "OutputRasterParameterName":"out_back_direction_raster",
            "in_source_data": in_source_data,
                
        }
    }

    if in_barrier_data is not None:
        layer2, in_barrier_data, raster_ra2 = _raster_input(in_barrier_data)
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data
    
    if cell_size is not None:
        template_dict["rasterFunctionArguments"]["cell_size"] = cell_size
    
    if max_distance is not None:
        template_dict["rasterFunctionArguments"]["maximum_distance"] = max_distance

    distance_method_list = ["PLANAR","GEODESIC"]
    if distance_method is not None:
        if distance_method.upper() not in distance_method_list:
            raise RuntimeError('distance_method should be one of the following '+ str(distance_method_list))
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_source_data"] = raster_ra

    if in_barrier_data is not None:
        function_chain_ra["rasterFunctionArguments"]["in_barrier_data"] = raster_ra2

    return _gbl_clone_layer(layer, template_dict, function_chain_ra)

def flow_length(input_flow_direction_raster,
                direction_measurement="DOWNSTREAM",
                input_weight_raster=None):
    """
    Creates a raster layer of upstream or downstream distance, or weighted distance, 
    along the flow path for each cell.

    A primary use of the Flow Length function is to calculate the length of the longest 
    flow path within a given basin. This measure is often used to calculate the time of 
    concentration of a basin. This would be done using the Upstream option. 

    The function can also be used to create distance-area diagrams of hypothetical 
    rainfall and runoff events using the weight raster as an impedance to movement downslope.

    For more information, see 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/flow-length-function.htm

    Parameters
    ----------
    :param input_flow_direction_raster: Required. The input raster that shows the direction of flow out of each cell.
                                        The flow direction raster can be created by running the Flow Direction function.
    :param direction_measurement: Optional String. The direction of measurement along the flow path.

                                  - DOWNSTREAM - Calculates the downslope distance along the flow path, \
                                  from each cell to a sink or outlet on the edge of the raster. this is the default.

                                  - UPSTREAM - Calculates the longest upslope distance along the flow path, \
                                  from each cell to the top of the drainage divide.

    :param input_weight_raster: An optional input raster for applying a weight to each cell.
                                If no weight raster is specified, a default weight of 1 will 
                                be applied to each cell.
    :return: output raster with function applied

    """
    layer1, input_flow_direction_raster, raster_ra1 = _raster_input(input_flow_direction_raster)
            
    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "FlowLength_sa",           
            "PrimaryInputParameterName" : "in_flow_direction_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_flow_direction_raster" : input_flow_direction_raster
        }
    }

    if input_weight_raster is not None:
        layer2, input_weight_raster, raster_ra2 = _raster_input(input_weight_raster)
        template_dict["rasterFunctionArguments"]["in_weight_raster"] = input_weight_raster

    direction_measurement_list = ["DOWNSTREAM","UPSTREAM"]
    if direction_measurement is not None:
        if direction_measurement.upper() not in direction_measurement_list:
            raise RuntimeError('direction_measurement should be one of the following '+ str(direction_measurement_list))
        template_dict["rasterFunctionArguments"]["direction_measurement"] = direction_measurement

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra1
    if input_weight_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_weight_raster"] = raster_ra2
        
    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)


def sink(input_flow_direction_raster):
    """
    Creates a raster layer identifying all sinks or areas of internal drainage.

    The value type for the Sink function output raster layer is floating point.

    For more information, see 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/sink-function.htm

    Parameters
    ----------
    :param input_flow_direction_raster: Required. The input raster that shows the direction 
                                        of flow out of each cell.

                                        The flow direction raster can be created by 
                                        running the Flow Direction function.

    :return: output raster with function applied

    """
    layer, input_flow_direction_raster, raster_ra = _raster_input(input_flow_direction_raster)  
            
    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Sink_sa",
            "PrimaryInputParameterName" : "in_flow_direction_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_flow_direction_raster" : input_flow_direction_raster
             
        }
    }    
    
    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra    
    
    return _gbl_clone_layer(layer, template_dict, function_chain_ra)


def snap_pour_point(in_pour_point_data,
                    in_accumulation_raster=None,
                    snap_distance=0,
                    pour_point_field=None):
    """
    Snaps pour points to the cell of highest flow accumulation within a specified distance.

    For more information, see 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/snap-pour-point-function.htm

    Parameters
    ----------
    :param in_pour_point_data: Required. The input pour point locations that are to be snapped. 
                               For an input raster layer, all cells that are not 
                               NoData (that is, have a value) will be considered 
                               pour points and will be snapped.

    :param in_accumulation_raster: optional raster; The input flow accumulation raster layer.
    :param snap_distance: Optional. Maximum distance, in map units, to search for a cell of higher 
                          accumulated flow. Default is 0
    :param pour_point_field: Optional. Field used to assign values to the pour point locations.

    :return: output raster with function applied

    """
    layer, in_pour_point_data, raster_ra = _raster_input(in_pour_point_data)  
            
    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "SnapPourPoint_sa",
            "PrimaryInputParameterName" : "in_pour_point_data",
            "OutputRasterParameterName" : "out_raster",
            "in_pour_point_data" : in_pour_point_data
             
        }
    }

    if in_accumulation_raster is not None:
        layer2, in_accumulation_raster, raster_ra2 = _raster_input(in_accumulation_raster)
        template_dict["rasterFunctionArguments"]["in_accumulation_raster"] = in_accumulation_raster

    if snap_distance is not None:
        template_dict["rasterFunctionArguments"]["snap_distance"] = snap_distance

    if pour_point_field is not None:
        template_dict["rasterFunctionArguments"]["pour_point_field"] = pour_point_field
    
    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_pour_point_data"] = raster_ra

    if in_accumulation_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_accumulation_raster"] = raster_ra2
    
    return _gbl_clone_layer(layer, template_dict, function_chain_ra)


def stream_order(input_stream_raster,
                 input_flow_direction_raster=None,
                 order_method="STRAHLER"):
    """
    Creates a raster layer that assigns a numeric order to segments 
    of a raster representing branches of a linear network.

    For more information, see 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/stream-order-function.htm

    Parameters
    ----------
    :param input_stream_raster: Required. An input stream raster that represents a linear stream network.
    :param input_flow_direction_raster: Optional. The input raster that shows the direction of flow out of each cell
                                        The flow direction raster can be created by running the Flow 
                                        Direction function.
    :param order_method: Optional. The method used for assigning stream order.

                          - STRAHLER - The method of stream ordering proposed by Strahler in 1952. \
                          Stream order only increases when streams of the same order intersect. \
                          Therefore, the intersection of a first-order and second-order link will \
                          remain a second-order link, rather than creating a third-order link. \
                          This is the default.

                          - SHREVE - The method of stream ordering by magnitude, proposed by Shreve \
                          in 1967. All links with no tributaries are assigned a magnitude (order) \
                          of one. Magnitudes are additive downslope. When two links intersect, \
                          their magnitudes are added and assigned to the downslope link.

    :return: output raster with function applied

    """
    layer1, input_stream_raster, raster_ra1 = _raster_input(input_stream_raster)
            
    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "StreamOrder_sa",           
            "PrimaryInputParameterName" : "in_stream_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_stream_raster" : input_stream_raster
        }
    }    
    
    if input_flow_direction_raster is not None:
        layer2, input_flow_direction_raster, raster_ra2 = _raster_input(input_flow_direction_raster)
        template_dict["rasterFunctionArguments"]["in_flow_direction_raster"] = input_flow_direction_raster

    order_method_list = ["STRAHLER","SHREVE"]
    if order_method is not None:
        if order_method.upper() not in order_method_list:
            raise RuntimeError('order_method should be one of the following '+ str(order_method_list))
        template_dict["rasterFunctionArguments"]["order_method"] = order_method

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_stream_raster"] = raster_ra1
    if input_flow_direction_raster is not None:
        function_chain_ra["rasterFunctionArguments"]["in_flow_direction_raster"] = raster_ra2
        
    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

def expand(input_raster,
           number_of_cells,
           zone_values):
    """
    Expands specified zones of a raster by a specified number of cells. 
    https://pro.arcgis.com/en/pro-app/help/data/imagery/expand-function.htm

    Parameters
    ----------
    :param input_raster: Required. The input raster for which the identified zones are to 
                         be expanded.
                         It must be of integer type.

    :param number_of_cells: Required. The number of cells to expand by.
                            The value must be integer, and can be 1 or greater.

    :param zone_values: Required. The list of zones to expand. The zone values 
                        must be integer, and they can be in any order.
                        The zone values can be specified as a list or as a string. 
                        If specified as a string and if it is required to specify multiple zones, 
                        use a semicolon (";") to separate the zone values.

    :return: output raster with function applied

    """
    layer1, input_raster, raster_ra1 = _raster_input(input_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Expand_sa",
            "PrimaryInputParameterName" : "in_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_raster" : input_raster
        }
    }

    if number_of_cells is not None:
        template_dict["rasterFunctionArguments"]["number_cells"] = number_of_cells

    zone_values_str = zone_values
    if isinstance(zone_values, list):
        zone_values_str = ";".join(str(zone) for zone in zone_values)

    if zone_values_str is not None:
        template_dict["rasterFunctionArguments"]["zone_values"] = zone_values_str

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_raster"] = raster_ra1

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

def shrink(input_raster,
           number_of_cells,
           zone_values):
    """
    Shrinks the selected zones by a specified number of cells by replacing them with 
    the value of the cell that is most frequent in its neighborhood.
    https://pro.arcgis.com/en/pro-app/help/data/imagery/shrink-function.htm

    Parameters
    ----------
    :param input_raster: Required. The input raster for which the identified zones are to be shrunk.
                         It must be of integer type.

    :param number_of_cells: Required. The number of cells by which to shrink each specified zone.
                            The value must be integer, and can be 1 or greater.

    :param zone_values: Required. The list of zones to shrink. The zone values must be integer, and they can be in any order.
                        The zone values can be specified as a list or as a string. 
                        If specified as a string and if it is required to specify multiple zones, 
                        use a semicolon (";") to separate the zone values.


    :return: output raster with function applied

    """
    layer1, input_raster, raster_ra1 = _raster_input(input_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "Shrink_sa",           
            "PrimaryInputParameterName" : "in_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_raster" : input_raster
        }
    }

    if number_of_cells is not None:
        template_dict["rasterFunctionArguments"]["number_cells"] = number_of_cells


    zone_values_str = zone_values
    if isinstance(zone_values, list):
        zone_values_str = ";".join(str(zone) for zone in zone_values)

    if zone_values_str is not None:
        template_dict["rasterFunctionArguments"]["zone_values"] = zone_values_str

    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_raster"] = raster_ra1

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)

def distance_accumulation(in_source_data,
                          in_barrier_data=None,
                          in_surface_raster=None,
                          in_cost_raster=None,
                          in_vertical_raster=None,
                          vertical_factor="BINARY 1 -30 30",
                          in_horizontal_raster=None,
                          horizontal_factor="BINARY 1 45",
                          source_initial_accumulation=None,
                          source_maximum_accumulation=None,
                          source_cost_multiplier=None,
                          source_direction="FROM_SOURCE",
                          distance_method="PLANAR",
                          output_back_direction_raster_name=None, 
                          output_source_direction_raster_name=None, 
                          output_source_location_raster_name=None, 
                          ):
    """
    Calculates the least accumulative cost distance for each cell from or to the 
    least-cost source over a cost surface, preserving euclidean distance metric

    Parameters
    ----------
    :param in_source_data:  Required. The input source locations.

                            This is a layer that identifies the cells or locations from
                            or to which the least accumulated cost distance for every output cell location is calculated.
                            This parameter can have either a raster layer input or a feature layer input. 
                            For rasters, the input type can be integer or floating point.

    :param in_barrier_data: Optional. The input layer that defines the barriers. 
                            This parameter can have either a raster layer input or a feature layer input. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location. The values are used to calculate the actual
                               surface distance covered when passing between cells.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.

                             The value at each cell location represents the cost-per-unit distance for moving through the cell.
                             Each cell location value is multiplied by the cell resolution while also compensating for diagonal
                             movement to obtain the total cost of passing through the cell.

                             The values of the cost raster can be integer or floating point, but they cannot be negative or
                             zero (you cannot have a negative or zero cost).

    :param in_horizontal_raster: Optional. A raster defining the horizontal direction at each cell.
                                 The values on the raster must be integers ranging from 0 to 360, with 0 degrees being north, or toward
                                 the top of the screen, and increasing clockwise. Flat areas should be given a value of -1.
                                 The values at each location will be used in conjunction with the {horizontal_factor} to determine the
                                 horizontal cost incurred when moving from a cell to its neighbors.

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell. The values are used for calculating the slope
                                used to identify the vertical factor incurred when moving from one cell to another.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.

    :param vertical_factor: Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)

    :param maximum_distance:  Optional. Defines the threshold that the accumulative cost values cannot exceed. If an accumulative cost distance
                              value exceeds this value, the output value for the cell location will be NoData. The maximum distance
                              defines the extent for which the accumulative cost distances are calculated.

                              The default distance is to the edge of the output raster.

    :param distance_method: Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.

    :param source_initial_accumulation: Optional. The starting cost from which to begin the cost calculations. 
    
                                        Allows for the specification of the fixed 
                                        cost associated with a source. Instead of starting at a cost of zero, the cost algorithm will begin with 
                                        the value set by source_start_cost.

                                        The values must be zero or greater. The default is 0.

    :param source_maximum_accumulation: Optional. The cost capacity for the traveler for a source. 

                                        The cost calculations continue for each source until the specified capacity is reached.

                                        The values must be greater than zero. The default capacity is to the edge of the output raster.


    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values. Allows for control of the mode of travel or the magnitude at a source.
                                    The greater the multiplier, the greater the cost to move through each cell. The values must be greater than zero.
                                    The default is 1.


    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE. Default value is FROM_SOURCE.

    :param output_back_direction_raster_name: Optional string, determines whether back_direction_raster should be generated or not.
                                              Set this parameter, in order to generate the back_direction_raster.
                                              If set, the output of the function will be a named tuple.

    :param output_source_direction_raster_name: Optional string. Name of the source_direction_raster. This parameter determines 
                                                whether source_direction_raster should be generated or not.
                                                Set this parameter, in order to generate the source_direction_raster.
                                                If set, the output of the function will be a named tuple.

    :param output_source_location_raster_name: Optional string. Name of the source_location_raster.This paramter determines whether 
                                               source_location_raster should be generated or not.
                                               Set this parameter, in order to generate the source_location_raster.
                                               If set, the output of the function will be a named tuple.

    :return: output raster with function applied
    """

    if isinstance (in_source_data, ImageryLayer):
        layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)
    else:
        raster_ra1 = _layer_input(in_source_data)
        input_source_data = raster_ra1
        layer1=raster_ra1

    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_barrier_data is not None:
        if isinstance (in_barrier_data, ImageryLayer):
            layer3, in_barrier_data, raster_ra3 = _raster_input(in_barrier_data)
        else:
            raster_ra3 = _layer_input(in_barrier_data)
            in_barrier_data = raster_ra3
            layer3 = raster_ra3

    if in_surface_raster is not None:
        layer4, in_surface_raster, raster_ra4 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer5, in_horizontal_raster, raster_ra5 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer6, in_vertical_raster, raster_ra6 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "DistanceAccumulation_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_distance_accumulation_raster",
            "in_source_data" : input_source_data,
            "RasterInfo":{"blockWidth" : 2048,
                "blockHeight":256,
                "bandCount":1,
                "pixelType":9,
                "firstPyramidLevel":1,
                "maximumPyramidLevel":30,
                "pixelSizeX":1,
                "pixelSizeY" :1,
                "type":"RasterInfo"}
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_barrier_data is not None:
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    if horizontal_factor is not None:
        template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    if vertical_factor is not None:
        template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if distance_method is not None:
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    if source_initial_accumulation is not None:
        template_dict["rasterFunctionArguments"]["source_initial_accumulation"] = source_initial_accumulation

    if source_maximum_accumulation is not None:
        template_dict["rasterFunctionArguments"]["source_maximum_accumulation"] = source_maximum_accumulation

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction
    else:
        template_dict["rasterFunctionArguments"]["source_direction"] = "FROM_SOURCE"


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_barrier_data is not None:
        function_chain_ra['rasterFunctionArguments']["in_barrier_data"] = raster_ra3

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra4

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra5

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra6


    if output_back_direction_raster_name is not None or output_source_direction_raster_name is not None or output_source_location_raster_name is not None:
        if isinstance(in_source_data, ImageryLayer):
            return _gbl_clone_layer(in_source_data, 
                                    template_dict, 
                                    function_chain_ra, 
                                    output_back_direction_raster_name = output_back_direction_raster_name, 
                                    output_source_direction_raster_name = output_source_direction_raster_name,
                                    output_source_location_raster_name = output_source_location_raster_name,
                                    use_ra=True)
        else:
            return _feature_gbl_clone_layer(in_source_data, 
                                            template_dict, 
                                            function_chain_ra, 
                                            output_back_direction_raster_name = output_back_direction_raster_name, 
                                            output_source_direction_raster_name = output_source_direction_raster_name,
                                            output_source_location_raster_name = output_source_location_raster_name,
                                            use_ra=True)

    if isinstance(in_source_data, ImageryLayer):
        return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra)
    else:
        return _feature_gbl_clone_layer(in_source_data, template_dict, function_chain_ra)



def distance_allocation(in_source_data,
                        in_barrier_data=None,
                        in_surface_raster=None,
                        in_cost_raster=None,
                        in_vertical_raster=None,
                        vertical_factor="BINARY 1 -30 30",
                        in_horizontal_raster=None,
                        horizontal_factor="BINARY 1 45",
                        source_field=None,
                        source_initial_accumulation=None,
                        source_maximum_accumulation=None,
                        source_cost_multiplier=None,
                        source_direction="FROM_SOURCE",
                        distance_method="PLANAR",
                        output_distance_accumulation_raster_name=None,
                        output_back_direction_raster_name=None, 
                        output_source_direction_raster_name=None, 
                        output_source_location_raster_name=None, 
                        ):
    """
    Calculates, for each cell, its least-cost source based on the least accumulative cost over a cost surface, 
    avoiding network distance distortion.",

    Parameters
    ----------
    :param in_source_data:  Required. The input source locations.

                            This is a layer that identifies the cells or locations from
                            or to which the least accumulated cost distance for every output cell location is calculated.
                            This parameter can have either a raster layer input or a feature layer input.
                            For rasters, the input type can be integer or floating point.

    :param in_barrier_data: Optional. The input layer that defines the barriers. 
                            This parameter can have either a raster layer input or a feature layer input. The dataset must contain 
                            NoData where there are no barriers. Barriers are represented by valid values including zero. 
                            The barriers can be defined by an integer or floating-point raster. 

    :param in_surface_raster:  Optional. A raster defining the elevation values at each cell location. The values are used to calculate the actual
                               surface distance covered when passing between cells.

    :param in_cost_raster:   Optional. A raster defining the impedance or cost to move planimetrically through each cell.

                             The value at each cell location represents the cost-per-unit distance for moving through the cell.
                             Each cell location value is multiplied by the cell resolution while also compensating for diagonal
                             movement to obtain the total cost of passing through the cell.

                             The values of the cost raster can be integer or floating point, but they cannot be negative or
                             zero (you cannot have a negative or zero cost).

    :param in_vertical_raster:  Optional. A raster defining the vertical (z) value for each cell. The values are used for calculating the slope
                                used to identify the vertical factor incurred when moving from one cell to another.

    :param vertical_factor: Optional. The Vertical Factor defines the relationship between the vertical cost factor and
                            the vertical relative moving angle (VRMA)

    :param in_horizontal_raster: Optional. A raster defining the horizontal direction at each cell.
                                 The values on the raster must be integers ranging from 0 to 360, with 0 degrees being north, or toward
                                 the top of the screen, and increasing clockwise. Flat areas should be given a value of -1.
                                 The values at each location will be used in conjunction with the {horizontal_factor} to determine the
                                 horizontal cost incurred when moving from a cell to its neighbors.

    :param horizontal_factor:  Optional. The Horizontal Factor defines the relationship between the horizontal cost
                               factor and the horizontal relative moving angle.

    :param source_field: Optional. The field used to assign values to the source locations. It must be an
                         integer type. If the Value Raster has been set, the values in that input
                         will take precedence over any setting for the source field.

    :param source_initial_accumulation: Optional. The starting cost from which to begin the cost calculations. 
    
                                        Allows for the specification of the fixed 
                                        cost associated with a source. Instead of starting at a cost of zero, the cost algorithm will begin with 
                                        the value set by source_start_cost.

                                        The values must be zero or greater. The default is 0.

    :param source_maximum_accumulation: Optional. The cost capacity for the traveler for a source. 

                                        The cost calculations continue for each source until the specified capacity is reached.

                                        The values must be greater than zero. The default capacity is to the edge of the output raster.

    :param source_cost_multiplier:  Optional. Multiplier to apply to the cost values. Allows for control of the mode of travel or the magnitude at a source.
                                    The greater the multiplier, the greater the cost to move through each cell. The values must be greater than zero.
                                    The default is 1.

    :param source_direction:  Optional. Defines the direction of the traveler when applying horizontal and vertical factors,
                              the source resistance rate, and the source starting cost.
                              Possible values: FROM_SOURCE, TO_SOURCE. Default value is FROM_SOURCE.

    :param distance_method: Optional String; Determines whether to calculate the distance using a planar (flat earth) 
                            or a geodesic (ellipsoid) method.

                            - Planar - Planar measurements use 2D Cartesian mathematics to calculate \
                            length and area. The option is only available when measuring in a \
                            projected coordinate system and the 2D plane of that coordinate system \
                            will be used as the basis for the measurements. This is the default.

                            - Geodesic - The shortest line between two points on the earth's surface \
                            on a spheroid (ellipsoid). Therefore, regardless of input or output \
                            projection, the results do not change.

                            .. note::

                            One use for a geodesic line is when you want to determine the shortest 
                            distance between two cities for an airplane's flight path. This is also
                            known as a great circle line if based on a sphere rather than an ellipsoid.

    :param output_back_direction_raster_name: Optional string, determines whether back_direction_raster should be generated or not.
                                              Set this parameter, in order to generate the back_direction_raster.
                                              If set, the output of the function will be a named tuple.

    :param output_source_direction_raster_name: Optional string. Name of the source_direction_raster. This parameter determines 
                                                whether source_direction_raster should be generated or not.
                                                Set this parameter, in order to generate the source_direction_raster.
                                                If set, the output of the function will be a named tuple.

    :param output_source_location_raster_name: Optional string. Name of the source_location_raster.This paramter determines whether 
                                               source_location_raster should be generated or not.
                                               Set this parameter, in order to generate the source_location_raster.
                                               If set, the output of the function will be a named tuple.

    :return: output raster with function applied
    """

    if isinstance (in_source_data, ImageryLayer):
        layer1, input_source_data, raster_ra1 = _raster_input(in_source_data)
    else:
        raster_ra1 = _layer_input(in_source_data)
        input_source_data = raster_ra1
        layer1=raster_ra1


    if in_cost_raster is not None:
        layer2, in_cost_raster, raster_ra2 = _raster_input(in_cost_raster)

    if in_barrier_data is not None:
        if isinstance (in_barrier_data, ImageryLayer):
            layer3, in_barrier_data, raster_ra3 = _raster_input(in_barrier_data)
        else:
            raster_ra3 = _layer_input(in_barrier_data)
            in_barrier_data = raster_ra3
            layer3=raster_ra3

    if in_surface_raster is not None:
        layer4, in_surface_raster, raster_ra4 = _raster_input(in_surface_raster)
    if in_horizontal_raster is not None:
        layer5, in_horizontal_raster, raster_ra5 = _raster_input(in_horizontal_raster)
    if in_vertical_raster is not None:
        layer6, in_vertical_raster, raster_ra6 = _raster_input(in_vertical_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "DistanceAllocation_sa",
            "PrimaryInputParameterName" : "in_source_data",
            "OutputRasterParameterName":"out_distance_allocation_raster",
            "in_source_data" : input_source_data,
            "RasterInfo":{"blockWidth" : 2048,
                "blockHeight":256,
                "bandCount":1,
                "pixelType":8,
                "firstPyramidLevel":1,
                "maximumPyramidLevel":30,
                "pixelSizeX":1,
                "pixelSizeY" :1,
                "type":"RasterInfo"}
        }
    }

    if in_cost_raster is not None:
        template_dict["rasterFunctionArguments"]["in_cost_raster"] = in_cost_raster

    if in_barrier_data is not None:
        template_dict["rasterFunctionArguments"]["in_barrier_data"] = in_barrier_data

    if in_surface_raster is not None:
        template_dict["rasterFunctionArguments"]["in_surface_raster"] = in_surface_raster

    if in_horizontal_raster is not None:
        template_dict["rasterFunctionArguments"]["in_horizontal_raster"] = in_horizontal_raster

    if in_vertical_raster is not None:
        template_dict["rasterFunctionArguments"]["in_vertical_raster"] = in_vertical_raster

    if horizontal_factor is not None:
        template_dict["rasterFunctionArguments"]["horizontal_factor"] = horizontal_factor

    if vertical_factor is not None:
        template_dict["rasterFunctionArguments"]["vertical_factor"] = vertical_factor

    if source_field is not None:
        template_dict["rasterFunctionArguments"]["source_field"] = source_field

    if distance_method is not None:
        template_dict["rasterFunctionArguments"]["distance_method"] = distance_method

    if source_initial_accumulation is not None:
        template_dict["rasterFunctionArguments"]["source_initial_accumulation"] = source_initial_accumulation

    if source_maximum_accumulation is not None:
        template_dict["rasterFunctionArguments"]["source_maximum_accumulation"] = source_maximum_accumulation

    if source_cost_multiplier is not None:
        template_dict["rasterFunctionArguments"]["source_cost_multiplier"] = source_cost_multiplier

    if source_direction is not None:
        source_direction_list = ["FROM_SOURCE","TO_SOURCE"]
        if source_direction.upper() not in source_direction_list:
            raise RuntimeError('source_direction should be one of the following '+ str(source_direction_list) )
        template_dict["rasterFunctionArguments"]["source_direction"] = source_direction
    else:
        template_dict["rasterFunctionArguments"]["source_direction"] = "FROM_SOURCE"


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_source_data"] = raster_ra1
    if in_cost_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_cost_raster"] = raster_ra2

    if in_barrier_data is not None:
        function_chain_ra['rasterFunctionArguments']["in_barrier_data"] = raster_ra3

    if in_surface_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_surface_raster"] = raster_ra4

    if in_horizontal_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_horizontal_raster"] = raster_ra5

    if in_vertical_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_vertical_raster"] = raster_ra6

    if output_back_direction_raster_name is not None or \
       output_source_direction_raster_name is not None or \
       output_source_location_raster_name is not None or \
       output_distance_accumulation_raster_name is not None:
        if isinstance(in_source_data, ImageryLayer):
            return _gbl_clone_layer(in_source_data, 
                                    template_dict, 
                                    function_chain_ra, 
                                    output_distance_accumulation_raster_name=output_distance_accumulation_raster_name,
                                    output_back_direction_raster_name = output_back_direction_raster_name, 
                                    output_source_direction_raster_name = output_source_direction_raster_name,
                                    output_source_location_raster_name = output_source_location_raster_name,
                                    use_ra=True)
        else:
            return _feature_gbl_clone_layer(in_source_data, 
                                            template_dict, 
                                            function_chain_ra, 
                                            output_back_direction_raster_name = output_back_direction_raster_name, 
                                            output_source_direction_raster_name = output_source_direction_raster_name,
                                            output_source_location_raster_name = output_source_location_raster_name,
                                            use_ra=True)

    if isinstance(in_source_data, ImageryLayer):
        return _gbl_clone_layer(in_source_data, template_dict, function_chain_ra)
    else:
        return _feature_gbl_clone_layer(in_source_data, template_dict, function_chain_ra)

def optimal_path_as_raster(in_destination_data,
                           in_distance_accumulation_raster,
                           in_back_direction_raster,
                           destination_field=None,
                           path_type="EACH_ZONE"
                           ):
    """
    Calculates, for each cell, its least-cost source based on the least accumulative cost over a cost surface, 
    avoiding network distance distortion.",

    Parameters
    ----------
    :param in_destination_data: Required layer. A layer that identifies locations from which the optimal 
                                path is determined to the least costly source. 
                                This parameter can have either a raster layer input or a feature layer input.

                                If the input is a raster, it must consists of cells that have valid values 
                                (zero is a valid value), and the remaining cells must be assigned NoData.

    :param in_distance_accumulation_raster: Required raster layer. The distance accumulation raster is used 
                                            to determine the optimal path from the sources to the destinations. 

                                            The distance accumulation raster is usually created with the 
                                            distance_accumulation or distance_allocation functions.  Each cell 
                                            in the distance accumulation raster represents the minimum 
                                            accumulative cost distance over a surface from each cell to a set of source cells.

    :param in_back_direction_raster: Required raster layer. The back direction raster contains calculated directions in 
                                     degrees. The direction identifies the next cell along the optimal path back to 
                                     the least accumulative cost source while avoiding barriers.

    :param destination_field:   Optional string. The field to be used to obtain values for the destination locations. 

    :param path_type: Optional string. A keyword defining the manner in which the values and zones on the input destination
                      data will be interpreted in the cost path calculations.

                      - EACH_ZONE - For each zone on the input destination data, a least-cost path is determined \
                      and saved on the output raster. With this option, the least-cost path for each zone \
                      begins at the cell with the lowest cost distance weighting in the zone. This is the default.

                      - BEST_SINGLE - For all cells on the input destination data, the least-cost path is derived \
                      from the cell with the minimum of the least-cost paths to source cells.

                      - EACH_CELL - For each cell with valid values on the input destination data, a least-cost \
                      path is determined and saved on the output raster. With this option, each cell of the \
                      input destination data is treated separately, and a least-cost path is determined for \
                      each from cell.

    :return: output raster with function applied
    """
    if in_destination_data is not None:
        if isinstance (in_destination_data, ImageryLayer):
            layer1, input_destination_data, raster_ra1 = _raster_input(in_destination_data)
        else:
            raster_ra1 = _layer_input(in_destination_data)
            input_destination_data = raster_ra1
            layer1=raster_ra1

    layer2, in_distance_accumulation_raster, raster_ra2 = _raster_input(in_distance_accumulation_raster)

    layer3, in_back_direction_raster, raster_ra3 = _raster_input(in_back_direction_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "OptimalPathAsRaster_sa",
            "PrimaryInputParameterName" : "in_destination_data",
            "OutputRasterParameterName":"out_path_accumulation_raster",
            "in_destination_data" : input_destination_data,
            "RasterInfo":{"blockWidth" : 2048,
                "blockHeight":256,
                "bandCount":1,
                "pixelType":8,
                "firstPyramidLevel":1,
                "maximumPyramidLevel":30,
                "pixelSizeX":1,
                "pixelSizeY" :1,
                "type":"RasterInfo"}
        }
    }

    if in_distance_accumulation_raster is not None:
        template_dict["rasterFunctionArguments"]["in_distance_accumulation_raster"] = in_distance_accumulation_raster

    if in_back_direction_raster is not None:
        template_dict["rasterFunctionArguments"]["in_back_direction_raster"] = in_back_direction_raster

    if destination_field is not None:
        template_dict["rasterFunctionArguments"]["destination_field"] = destination_field

    if path_type is not None:
        path_type_list = ["EACH_CELL", "EACH_ZONE", "BEST_SINGLE"]
        if path_type.upper() not in path_type_list:
            raise RuntimeError('path_type should be one of the following '+ str(path_type_list))
        template_dict["rasterFunctionArguments"]["path_type"] = path_type


    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra['rasterFunctionArguments']["in_destination_data"] = raster_ra1

    if in_distance_accumulation_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_distance_accumulation_raster"] = raster_ra2

    if in_back_direction_raster is not None:
        function_chain_ra['rasterFunctionArguments']["in_back_direction_raster"] = raster_ra3

    if isinstance(in_destination_data, ImageryLayer):
        return _gbl_clone_layer(layer1, template_dict, function_chain_ra)
    else:
        return _feature_gbl_clone_layer(in_destination_data, template_dict, function_chain_ra)


def boundary_clean(input_raster, sort_type = "NO_SORT", number_of_runs="TWO_WAY"):
    """
    The boundary_clean function smooths the boundary between zones in a raster. 
    Function available in ArcGIS Image Server 10.9 and higher.

    ================================     ====================================================================
    **Argument**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_raster                         Required. The input raster for which the boundary between zones will 
                                         be smoothed. It must be of integer type.  
    --------------------------------     --------------------------------------------------------------------
    sort_type                            Optional string. Specifies the type of sorting to use in the smoothing process. The sorting determines the priority by which cells can expand into their neighbors. The sorting can be done based on zone value or zone area. 
                                         The available choices are: ['NO_SORT', 'DESCEND', 'ASCEND'] 
                                         The default is: 'NO_SORT'.

                                         * ``NO_SORT`` - The zones are not sorted by size. Zones with larger values 
                                           will have a higher priority to expand into zones with 
                                           smaller values in the smoothed output. This is the default. 

                                         * ``DESCEND`` - Sorts zones in descending order by size. Zones with 
                                           larger total areas have a higher priority to expand into 
                                           zones with smaller total areas. This option will tend to 
                                           eliminate or reduce the prevalence of cells from smaller 
                                           zones in the smoothed output. 

                                         * ``ASCEND`` - Sorts zones in ascending order by size. Zones with smaller 
                                           total areas have a higher priority to expand into zones 
                                           with larger total areas. This option will tend to preserve 
                                           or increase the prevalence of cells from smaller zones in 
                                           the smoothed output. 
    --------------------------------     --------------------------------------------------------------------
    number_of_runs                       Optional String or Boolean. Specifies the number of times the smoothing 
                                         process will take place, twice or once. 

                                         * ``TWO_WAY`` (true) - Performs an expansion and shrinking operation two 
                                           times.  For the first time the operation is performed according to the 
                                           specified sorting type. Then an additional  expansion and shrinking 
                                           operation is performed, but with the priority reversed. This is the default. 
                                         * ``ONE_WAY`` (false) - Performs the expansion and shrinking operation 
                                           once, according to the sorting type. 
    ================================     ====================================================================
 
    :returns: output raster with function applied

    .. code-block:: python

            # Usage Example: 
            boundary_clean_output =  boundary_clean(input_raster = imagery_layer, sort_type = "NO_SORT", number_of_runs="TWO_WAY")

            boundary_clean_item = boundary_clean_output.save()
    """
    layer1, input_raster, raster_ra1 = _raster_input(input_raster)

    template_dict = {
        "rasterFunction" : "GPAdapter",
        "rasterFunctionArguments" : {
            "toolName" : "BoundaryClean_sa",
            "PrimaryInputParameterName" : "in_raster",
            "OutputRasterParameterName" : "out_raster",
            "in_raster" : input_raster
        }
    }

    if sort_type is not None:
        sort_type_list = ["NO_SORT", "DESCEND", "ASCEND"]
        if sort_type.upper() not in sort_type_list:
            raise RuntimeError('sort_type should be one of the following '+ str(sort_type_list))
        template_dict["rasterFunctionArguments"]["sort_type"] = sort_type

    if number_of_runs is not None:
        if isinstance(number_of_runs, bool):
            template_dict["rasterFunctionArguments"]["number_of_runs"] = number_of_runs
        elif isinstance(number_of_runs, str):
            if number_of_runs.upper() == "TWO_WAY":
                number_of_runs = True
            elif number_of_runs.upper() == "ONE_WAY":
                number_of_runs = False
            else:
                 raise RuntimeError('number_of_runs should be one of the following - TWO_WAY, ONE_WAY or should be of type bool.')
            template_dict["rasterFunctionArguments"]["number_of_runs"] = number_of_runs
    function_chain_ra = copy.deepcopy(template_dict)
    function_chain_ra["rasterFunctionArguments"]["in_raster"] = raster_ra1

    return _gbl_clone_layer(layer1, template_dict, function_chain_ra)
