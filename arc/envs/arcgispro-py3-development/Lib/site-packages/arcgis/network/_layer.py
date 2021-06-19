import logging
import datetime
from arcgis.gis import Layer, _GISResource
# Supported Data Types
from arcgis.features import Feature, FeatureSet
from arcgis.features import FeatureLayer, FeatureLayerCollection, Table

from arcgis.mapping import MapImageLayer
try:

    import pandas as pd
    from arcgis.features.geo import _is_geoenabled
    HASPANDAS = True
except ImportError:
    HASPANDAS = False
    def _is_geoenabled(df):
        return False
from arcgis.gis import Item

_log = logging.getLogger(__name__)

###########################################################################
def _handle_spatial_inputs(data,
                           do_not_locate=True,
                           has_z=False,
                           where=None):
    """
    Handles the various supported inputs types
    """
    template = {
        'type' : 'features',
        'doNotLocateOnRestrictedElements' : do_not_locate,
        'hasZ' : has_z
    }
    if isinstance(data, Item) and \
       data.type in ["Feature Layer", 'Feature Service']:
        return _handle_spatial_inputs(data.layers[0])
    if HASPANDAS and \
       isinstance(data, pd.DataFrame) and \
       _is_geoenabled(df=data):
        return data.spatial.__feature_set__
    elif isinstance(data, FeatureSet):
        return data.sdf.spatial.__feature_set__
    elif isinstance(data, list):
        stops_dict = []
        for stop in data:
            if isinstance(stop, Feature):
                stops_dict.append(stop.as_dict)
            else:
                stops_dict.append(stop)
        template['features'] = stops_dict
        return template
    elif isinstance(data, (FeatureLayer, Table)):
        from urllib.parse import quote
        import json
        query = data.filter
        url = data._url
        if query and \
           len(str(query)) > 0:
            query = quote(query)
            url += "/query?where=%s&outFields=*&f=json" % query
            if data._gis._con.token:
                url += "&token=%s" % data._gis._con.token
        else:
            query = quote("1=1")
            url += "/query?where=%s&outFields=*&f=json" % query
            if data._gis._con.token:
                url += "&token=%s" % data._gis._con.token
        template['url'] = url
        return template
    else:
        return data
    return data
###########################################################################
class NAJob(object):
    """Represents a Future Job for Network Analyst Jobs"""
    _future = None
    _gis = None
    _start_time = None
    _end_time = None

    #----------------------------------------------------------------------
    def __init__(self, future, task=None, notify=False):
        """
        initializer
        """
        if task is None:
            self._task = "Network Task"
        else:
            self._task = task
        self._future = future
        self._start_time = datetime.datetime.now()
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)
    #----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        if self._end_time:
            return self._end_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time
    #----------------------------------------------------------------------
    def _set_end_time(self, future):
        """sets the finish time"""
        self._end_time = datetime.datetime.now()
    #----------------------------------------------------------------------
    def _notify(self, future):
        """prints finished method"""
        jobid = str(self).replace("<", "").replace(">", "")
        try:
            res = future.result()
            infomsg = '{jobid} finished successfully.'.format(jobid=jobid)
            _log.info(infomsg)
            print(infomsg)
        except Exception as e:
            msg = str(e)
            msg = '{jobid} failed: {msg}'.format(jobid=jobid, msg=msg)
            _log.info(msg)
            print(msg)
    #----------------------------------------------------------------------
    def __str__(self):
        task = self._task
        return f"<{task} Job>"
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the GP status

        :returns: String
        """
        return self._future.done()
    #----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :returns: boolean
        """

        if self.done():
            return False
        if self.cancelled():
            return False
        return True
    #----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :returns: boolean
        """
        return self._future.cancelled()
    #----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :returns: boolean
        """
        return self._future.running()
    #----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :returns: boolean
        """
        return self._future.done()
    #----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :returns: object
        """
        if self.cancelled():
            return None
        return self._future.result()
###########################################################################
class NetworkLayer(Layer):
    """
    NetworkLayer represents a single network layer. It provides basic
    information about the network layer such as its name, type, and network
    classes. Additionally, depending on the layer type, it provides different
    pieces of information.

    It is a base class for RouteLayer, ServiceAreaLayer, and
    ClosestFacilityLayer.
    """
    #----------------------------------------------------------------------
    def _run_async(self, fn, **inputs):
        """runs the inputs asynchronously"""
        import concurrent.futures
        tp = concurrent.futures.ThreadPoolExecutor(1)
        future = tp.submit(fn=fn, **inputs)
        tp.shutdown(False)
        return future
    #----------------------------------------------------------------------
    def retrieve_travel_modes(self):
        """identify all the valid travel modes that have been defined on the
        network dataset or in the portal if the GIS server is federated"""
        url = self._url + "/retrieveTravelModes"
        params = {"f":"json"}
        return self._con.get(path=url,
                         params=params, )


###########################################################################
class RouteLayer(NetworkLayer):
    """
    The Route Layer which has common properties of Network Layer
    as well as some attributes unique to Route Network Layer only.
    """
    def solve(self,
              stops,
              barriers=None,
              polyline_barriers=None,
              polygon_barriers=None,
              travel_mode=None,
              attribute_parameter_values=None,
              return_directions=True,
              return_routes=True,
              return_stops=False,
              return_barriers=False,
              return_polyline_barriers=True,
              return_polygon_barriers=True,
              out_sr=None,
              ignore_invalid_locations=True,
              output_lines=None,
              find_best_sequence=False,
              preserve_first_stop=True,
              preserve_last_stop=True,
              use_time_windows=False,
              start_time=None,
              start_time_is_utc=False,
              accumulate_attribute_names=None,
              impedance_attribute_name=None,
              restriction_attribute_names=None,
              restrict_u_turns=None,
              use_hierarchy=True,
              directions_language=None,
              directions_output_type=None,
              directions_style_name=None,
              directions_length_units=None,
              directions_time_attribute_name=None,
              output_geometry_precision=None,
              output_geometry_precision_units=None,
              return_z=False,
              overrides=None,
              preserve_objectid=False,
              future=False,
              time_windows_are_utc=False):
        """
        The solve operation is performed on a network layer resource.
        The solve operation is supported on a network layer whose layerType
        is esriNAServerRouteLayer. You can provide arguments to the solve
        route operation as query parameters.


        ===================================     ====================================================================
        **Argument**                            **Description**
        -----------------------------------     --------------------------------------------------------------------
        stops                                   Required Points/FeatureSet/a list of Features. The set of stops
                                                loaded as network locations during analysis. Stops can be specified
                                                using a simple comma / semi-colon based syntax or as a JSON
                                                structure. If stops are not specified, preloaded stops from the map
                                                document are used in the analysis.
        -----------------------------------     --------------------------------------------------------------------
        barriers                                Optional Point/FeatureSet. The set of barriers loaded as network
                                                locations during analysis. Barriers can be specified using a simple
                                                comma/semi-colon based syntax or as a JSON structure. If barriers
                                                are not specified, preloaded barriers from the map document are used
                                                in the analysis. If an empty json object is passed ('{}') preloaded
                                                barriers are ignored.
        -----------------------------------     --------------------------------------------------------------------
        polyline_barriers                       Optional Polyline/FeatureSet. The set of polyline barriers loaded
                                                as network locations during analysis. If polyline barriers are not
                                                specified, preloaded polyline barriers from the map document are
                                                used in the analysis. If an empty json object is passed ('{}')
                                                preloaded polyline barriers are ignored.
        -----------------------------------     --------------------------------------------------------------------
        polygon_barriers                        Optional Polygon/FeatureSet. The set of polygon barriers loaded as
                                                network locations during analysis. If polygon barriers are not
                                                specified, preloaded polygon barriers from the map document are used
                                                in the analysis. If an empty json object is passed ('{}') preloaded
                                                polygon barriers are ignored.
        -----------------------------------     --------------------------------------------------------------------
        travel_mode                             Optional string. Travel modes provide override values that help you
                                                quickly and consistently model a vehicle or mode of transportation.
                                                The chosen travel mode must be preconfigured on the network dataset
                                                that the routing service references.
        -----------------------------------     --------------------------------------------------------------------
        attribute_parameter_values              Optional string/list.  A set of attribute parameter values that can be
                                                parameterized to determine which network elements can be used by a
                                                vehicle.
        -----------------------------------     --------------------------------------------------------------------
        return_directions                       Optional boolean. If true, directions will be generated and returned
                                                with the analysis results. Default is true.
        -----------------------------------     --------------------------------------------------------------------
        return_routes                           Optional boolean. If true, routes will be returned with the analysis
                                                results. Default is true.
        -----------------------------------     --------------------------------------------------------------------
        return_stops                            Optional boolean.  If true, stops will be returned with the analysis
                                                results. Default is false.
        -----------------------------------     --------------------------------------------------------------------
        return_barriers                         Optional boolean.  If true, barriers will be returned with the
                                                analysis results. Default is false.
        -----------------------------------     --------------------------------------------------------------------
        return_polyline_barriers                Optional boolean. If true, polyline barriers will be returned with
                                                the analysis results. Default is false.
        -----------------------------------     --------------------------------------------------------------------
        return_polygon_barriers                 Optional boolean. If true, polygon barriers will be returned with
                                                the analysis results. Default is false.
        -----------------------------------     --------------------------------------------------------------------
        out_sr                                  Optional Integer. The spatial reference of the geometries returned
                                                with the analysis results.
        -----------------------------------     --------------------------------------------------------------------
        ignore_invalid_locations                Optional boolean. - If true, the solver will ignore invalid
                                                locations. Otherwise, it will raise an error. Default is true.
        -----------------------------------     --------------------------------------------------------------------
        output_lines                            The type of output lines to be generated in the result. The default
                                                is as defined in the network layer.
                                                Values: esriNAOutputLineTrueShape |
                                                        esriNAOutputLineTrueShapeWithMeasure |
                                                        esriNAOutputLineStraight | esriNAOutputLineNone
        -----------------------------------     --------------------------------------------------------------------
        find_best_sequence                      Optional boolean. If true, the solver should re-sequence the route in
                                                the optimal order. The default is as defined in the network layer.
        -----------------------------------     --------------------------------------------------------------------
        preserve_first_stop                     Optional boolean. If true, the solver should keep the first stop
                                                fixed in the sequence. The default is as defined in the network
                                                layer.
        -----------------------------------     --------------------------------------------------------------------
        preserve_last_stop                      Optional boolean. If true, the solver should keep the last stop fixed
                                                in the sequence. The default is as defined in the network layer.
        -----------------------------------     --------------------------------------------------------------------
        use_time_window                         Optional boolean. If true, the solver should consider time windows.
                                                The default is as defined in the network layer.
        -----------------------------------     --------------------------------------------------------------------
        start_time                              Optional string. The time the route begins. If not specified, the
                                                solver will use the default as defined in the network layer.
        -----------------------------------     --------------------------------------------------------------------
        start_time_is_utc                       Optional boolean. The time zone of the startTime parameter.
        -----------------------------------     --------------------------------------------------------------------
        accumulate_attribute_names              Optional string. A list of network attribute names to be accumulated
                                                with the analysis. The default is as defined in the network layer.
                                                The value should be specified as a comma separated list of attribute
                                                names. You can also specify a value of none to indicate that no
                                                network attributes should be accumulated.
        -----------------------------------     --------------------------------------------------------------------
        impedance_attribute_name                Optional string. The network attribute name to be used as the impedance
                                                attribute in analysis. The default is as defined in the network layer.
        -----------------------------------     --------------------------------------------------------------------
        restriction_attribute_names             Optional string. -The list of network attribute names to be
                                                used as restrictions with the analysis. The default is as defined in
                                                the network layer. The value should be specified as a comma
                                                separated list of attribute names. You can also specify a value of
                                                none to indicate that no network attributes should be used as
                                                restrictions.
        -----------------------------------     --------------------------------------------------------------------
        restrict_u_turns                        Optional boolean. Specifies how U-Turns should be restricted in the
                                                analysis. The default is as defined in the network layer.
                                                Values: esriNFSBAllowBacktrack | esriNFSBAtDeadEndsOnly |
                                                        esriNFSBNoBacktrack | esriNFSBAtDeadEndsAndIntersections
        -----------------------------------     --------------------------------------------------------------------
        use_hierarchy                           Optional boolean.  If true, the hierarchy attribute for the network
                                                should be used in analysis. The default is as defined in the network
                                                layer.
        -----------------------------------     --------------------------------------------------------------------
        directions_language                     Optional string. The language to be used when computing directions.
                                                The default is the language of the server's operating system. The list
                                                of supported languages can be found in REST layer description.
        -----------------------------------     --------------------------------------------------------------------
        directions_output_type                  Optional string.  Defines content, verbosity of returned directions.
                                                The default is esriDOTInstructionsOnly.
                                                Values: esriDOTComplete | esriDOTCompleteNoEvents
                                                        | esriDOTInstructionsOnly | esriDOTStandard |
                                                        esriDOTSummaryOnly
        -----------------------------------     --------------------------------------------------------------------
        directions_style_name                   Optional string. The style to be used when returning the directions.
                                                The default is as defined in the network layer. The list of
                                                supported styles can be found in REST layer description.
        -----------------------------------     --------------------------------------------------------------------
        directions_length_units                 Optional string. The length units to use when computing directions.
                                                The default is as defined in the network layer.
                                                Values: esriNAUFeet | esriNAUKilometers | esriNAUMeters |
                                                        esriNAUMiles | esriNAUNauticalMiles | esriNAUYards |
                                                        esriNAUUnknown
        -----------------------------------     --------------------------------------------------------------------
        directions_time_attribute_name          Optional string. The name of network attribute to use for the drive
                                                time when computing directions. The default is as defined in the network
                                                layer.
        -----------------------------------     --------------------------------------------------------------------
        output_geometry_precision               Optional float.  The precision of the output geometry after
                                                generalization. If 0, no generalization of output geometry is
                                                performed. The default is as defined in the network service
                                                configuration.
        -----------------------------------     --------------------------------------------------------------------
        output_geometry_precision_units         Optional string. The units of the output geometry precision. The
                                                default value is esriUnknownUnits.
                                                Values: esriUnknownUnits | esriCentimeters | esriDecimalDegrees |
                                                        esriDecimeters | esriFeet | esriInches | esriKilometers |
                                                        esriMeters | esriMiles | esriMillimeters |
                                                        esriNauticalMiles | esriPoints | esriYards
        -----------------------------------     --------------------------------------------------------------------
        return_z                                Optional boolean. If true, Z values will be included in the returned
                                                routes and compressed geometry if the network dataset is Z-aware.
                                                The default is false.
        -----------------------------------     --------------------------------------------------------------------
        overrides                               Optional dictionary. Specify additional settings that can influence
                                                the behavior of the solver.  A list of supported override settings
                                                for each solver and their acceptable values can be obtained by
                                                contacting Esri Technical Support.
        -----------------------------------     --------------------------------------------------------------------
        preserve_objectid                       Optional Boolean.  If True, all objectid values are maintained.  The
                                                default is False.
        -----------------------------------     --------------------------------------------------------------------
        future                                  Optional Boolean.  If True, the process is run asynchronously. The
                                                default is False.
        -----------------------------------     --------------------------------------------------------------------
        time_windows_are_utc                    Optional boolean. Specify whether the TimeWindowStart and TimeWindowEnd
                                                attribute values on stops are specified in coordinated universal time (UTC)
                                                or geographically local time.
        ===================================     ====================================================================


        :return: dict

        .. code-block:: python

            # USAGE EXAMPLE: Solving the routing problem by passing in a FeatureSet

            # get a FeatureSet through query
            fl = sample_cities.layers[0]
            cities_to_visit = fl.query(where="ST = 'CA' AND POP2010 > 300000",
                                       out_fields='NAME', out_sr=4326)

            type(cities_to_visit)
            >> arcgis.features.feature.FeatureSet

            # pass in the FeatureSet
            result = route_layer.solve(stops=cities_to_visit, preserve_first_stop=True,
                                       preserve_last_stop=True, find_best_sequence=True, return_directions=False,
                                       return_stops=True, return_barriers=False, return_polygon_barriers=False,
                                       return_polyline_barriers=False, return_routes=True,
                                       output_lines='esriNAOutputLineStraight')

        """

        if not self.properties.layerType == "esriNAServerRouteLayer":
            raise ValueError("The solve operation is supported on a network "
                             "layer of Route type only")

        url = self._url + "/solve"

        params = {
            "f": "json",
        }
        stops = _handle_spatial_inputs(data=stops)
        params['stops'] = stops
        if directions_output_type is None:
            directions_output_type = "esriDOTInstructionsOnly"
        if not barriers is None:
            params['barriers'] = _handle_spatial_inputs(data=barriers)
        if not polyline_barriers is None:
            params['polylineBarriers'] = _handle_spatial_inputs(data=polyline_barriers)
        if not polygon_barriers is None:
            params['polygonBarriers'] = _handle_spatial_inputs(data=polygon_barriers)
        if not travel_mode is None:
            params['travelMode'] = travel_mode
        if not attribute_parameter_values is None:
            params['attributeParameterValues'] = attribute_parameter_values
        if not return_directions is None:
            params['returnDirections'] = return_directions
        if not return_routes is None:
            params['returnRoutes'] = return_routes
        if not return_stops is None:
            params['returnStops'] = return_stops
        if not return_barriers is None:
            params['returnBarriers'] = return_barriers
        if not return_polyline_barriers is None:
            params['returnPolylineBarriers'] = return_polyline_barriers
        if not return_polygon_barriers is None:
            params['returnPolygonBarriers'] = return_polygon_barriers
        if not out_sr is None:
            params['outSR'] = out_sr
        if not ignore_invalid_locations is None:
            params['ignoreInvalidLocations'] = ignore_invalid_locations
        if not output_lines is None:
            params['outputLines'] = output_lines
        if not find_best_sequence is None:
            params['findBestSequence'] = find_best_sequence
        if not preserve_first_stop is None:
            params['preserveFirstStop'] = preserve_first_stop
        if not preserve_last_stop is None:
            params['preserveLastStop'] = preserve_last_stop
        if not use_time_windows is None:
            params['useTimeWindows'] = use_time_windows
        if not time_windows_are_utc is None:
            params['timeWindowsAreUTC'] = time_windows_are_utc
        if not start_time is None:
            if isinstance(start_time, datetime.datetime):
                start_time = f"{start_time.timestamp() * 1000}"
            params['startTime'] = start_time
        if not start_time_is_utc is None:
            params['startTimeIsUTC'] = start_time_is_utc
        if not accumulate_attribute_names is None:
            params['accumulateAttributeNames'] = accumulate_attribute_names
        if not impedance_attribute_name is None:
            params['impedanceAttributeName'] = impedance_attribute_name
        if not restriction_attribute_names is None:
            params['restrictionAttributeNames'] = restriction_attribute_names
        if not restrict_u_turns is None:
            params['restrictUTurns'] = restrict_u_turns
        if not use_hierarchy is None:
            params['useHierarchy'] = use_hierarchy
        if not directions_language is None:
            params['directionsLanguage'] = directions_language
        if not directions_output_type is None:
            params['directionsOutputType'] = directions_output_type
        if not directions_style_name is None:
            params['directionsStyleName'] = directions_style_name
        if not directions_length_units is None:
            params['directionsLengthUnits'] = directions_length_units
        if not directions_time_attribute_name is None:
            params['directionsTimeAttributeName'] = directions_time_attribute_name
        if not output_geometry_precision is None:
            params['outputGeometryPrecision'] = output_geometry_precision
        if not output_geometry_precision_units is None:
            params['outputGeometryPrecisionUnits'] = output_geometry_precision_units
        if not return_z is None:
            params['returnZ'] = return_z
        if not overrides is None:
            params['overrides'] = overrides
        if not preserve_objectid is None:
            params['preserveObjectID'] = preserve_objectid
        if future:
            f = self._run_async(self._con.post, **{'path' : url, 'postdata' : params, 'token' : self._gis._con.token})
            return NAJob(future=f, task="RouteLayer Solve")
        return self._con.post(path=url,
                              postdata=params)#,

###########################################################################
class ServiceAreaLayer(NetworkLayer):
    """
    The Service Area Layer which has common properties of Network
    Layer as well as some attributes unique to Service Area Layer
    only.
    """
    def solve_service_area(self, facilities,
                           barriers=None,
                           polyline_barriers=None,
                           polygon_barriers=None,
                           travel_mode=None,
                           attribute_parameter_values=None,
                           default_breaks=None,
                           exclude_sources_from_polygons=None,
                           merge_similar_polygon_ranges=None,
                           output_lines=None,
                           output_polygons=None,
                           overlap_lines=None,
                           overlap_polygons=None,
                           split_lines_at_breaks=None,
                           split_polygons_at_breaks=None,
                           trim_outer_polygon=None,
                           trim_polygon_distance=None,
                           trim_polygon_distance_units=None,
                           return_facilities=False,
                           return_barriers=False,
                           return_polyline_barriers=False,
                           return_polygon_barriers=False,
                           out_sr=None,
                           accumulate_attribute_names=None,
                           impedance_attribute_name=None,
                           restriction_attribute_names=None,
                           restrict_u_turns=None,
                           output_geometry_precision=None,
                           output_geometry_precision_units='esriUnknownUnits',
                           use_hierarchy=None,
                           time_of_day=None,
                           time_of_day_is_utc=None,
                           travel_direction=None,
                           return_z=False,
                           overrides=None,
                           preserve_objectid=False,
                           future=False,
                           ignore_invalid_locations=True):
        """ The solve service area operation is performed on a network layer
        resource of type service area (layerType is esriNAServerServiceArea).
        You can provide arguments to the solve service area operation as
        query parameters.
        Inputs:
            facilities - The set of facilities loaded as network locations
                         during analysis. Facilities can be specified using
                         a simple comma / semi-colon based syntax or as a
                         JSON structure. If facilities are not specified,
                         preloaded facilities from the map document are used
                         in the analysis. If an empty json object is passed
                         ('{}') preloaded facilities are ignored.
            barriers - The set of barriers loaded as network locations during
                       analysis. Barriers can be specified using a simple
                       comma/semicolon-based syntax or as a JSON structure.
                       If barriers are not specified, preloaded barriers from
                       the map document are used in the analysis. If an empty
                       json object is passed ('{}'), preloaded barriers are
                       ignored.
            polylineBarriers - The set of polyline barriers loaded as network
                               locations during analysis. If polyline barriers
                               are not specified, preloaded polyline barriers
                               from the map document are used in the analysis.
                               If an empty json object is passed ('{}'),
                               preloaded polyline barriers are ignored.
            polygonBarriers - The set of polygon barriers loaded as network
                              locations during analysis. If polygon barriers
                              are not specified, preloaded polygon barriers
                              from the map document are used in the analysis.
                              If an empty json object is passed ('{}'),
                              preloaded polygon barriers are ignored.
            travelMode - Travel modes provide override values that help you
                         quickly and consistently model a vehicle or mode of
                         transportation. The chosen travel mode must be
                         preconfigured on the network dataset that the
                         service area service references.
            attributeParameterValues - A set of attribute parameter values that
                                       can be parameterized to determine which
                                       network elements can be used by a vehicle.
            defaultBreaks - A comma-separated list of doubles. The default is
                            defined in the network analysis layer.
            excludeSourcesFromPolygons - A comma-separated list of string names.
                                         The default is defined in the network
                                         analysis layer.

            mergeSimilarPolygonRanges - If true, similar ranges will be merged
                                        in the result polygons. The default is
                                        defined in the network analysis layer.
            outputLines - The type of lines(s) generated. The default is as
                          defined in the network analysis layer.
                          Values: esriNAOutputLineNone | esriNAOutputLineTrueShape |
                          esriNAOutputLineTrueShapeWithMeasure
            outputPolygons - The type of polygon(s) generated. The default is
                             as defined in the network analysis layer.
            overlapLines - Indicates if the lines should overlap from multiple
                           facilities. The default is defined in the network
                           analysis layer.
            overlapPolygons - Indicates if the polygons for all facilities
                              should overlap. The default is defined in the
                              network analysis layer.
            splitLinesAtBreaks - If true, lines will be split at breaks. The
                                 default is defined in the network analysis
                                 layer.
            splitPolygonsAtBreaks - If true, polygons will be split at breaks.
                                    The default is defined in the network
                                    analysis layer.
            trimOuterPolygon -  If true, the outermost polygon (at the maximum
                                break value) will be trimmed. The default is
                                defined in the network analysis layer.
            trimPolygonDistance -  If polygons are being trimmed, provides the
                                   distance to trim. The default is defined in
                                   the network analysis layer.
            trimPolygonDistanceUnits - If polygons are being trimmed, specifies
                                       the units of the trimPolygonDistance. The
                                       default is defined in the network analysis
                                       layer.
            returnFacilities - If true, facilities will be returned with the
                               analysis results. Default is false.
            returnBarriers - If true, barriers will be returned with the analysis
                             results. Default is false.
            returnPolylineBarriers - If true, polyline barriers will be returned
                                     with the analysis results. Default is false.
            returnPolygonBarriers - If true, polygon barriers will be returned
                                    with the analysis results. Default is false.
            outSR - The well-known ID of the spatial reference for the geometries
                    returned with the analysis results. If outSR is not specified,
                    the geometries are returned in the spatial reference of the map.
            accumulateAttributeNames - The list of network attribute names to be
                                       accumulated with the analysis. The default
                                       is as defined in the network analysis layer.
                                       The value should be specified as a comma
                                       separated list of attribute names. You can
                                       also specify a value of none to indicate that
                                       no network attributes should be accumulated.
            impedanceAttributeName - The network attribute name to be used as the
                                     impedance attribute in analysis. The default
                                     is as defined in the network analysis layer.
            restrictionAttributeNames - The list of network attribute names to be
                                        used as restrictions with the analysis. The
                                        default is as defined in the network analysis
                                        layer. The value should be specified as a
                                        comma separated list of attribute names.
                                        You can also specify a value of none to
                                        indicate that no network attributes should
                                        be used as restrictions.
            restrictUTurns - Specifies how U-Turns should be restricted in the
                             analysis. The default is as defined in the network
                             analysis layer. Values: esriNFSBAllowBacktrack |
                             esriNFSBAtDeadEndsOnly | esriNFSBNoBacktrack |
                             esriNFSBAtDeadEndsAndIntersections
            outputGeometryPrecision - The precision of the output geometry after
                                      generalization. If 0, no generalization of
                                      output geometry is performed. The default is
                                      as defined in the network service configuration.
            outputGeometryPrecisionUnits - The units of the output geometry precision.
                                           The default value is esriUnknownUnits.
                                           Values: esriUnknownUnits | esriCentimeters |
                                           esriDecimalDegrees | esriDecimeters |
                                           esriFeet | esriInches | esriKilometers |
                                           esriMeters | esriMiles | esriMillimeters |
                                           esriNauticalMiles | esriPoints | esriYards
            useHierarchy - If true, the hierarchy attribute for the network should be
                           used in analysis. The default is as defined in the network
                           layer. This cannot be used in conjunction with outputLines.
            timeOfDay - The date and time at the facility. If travelDirection is set
                        to esriNATravelDirectionToFacility, the timeOfDay value
                        specifies the arrival time at the facility. if travelDirection
                        is set to esriNATravelDirectionFromFacility, the timeOfDay
                        value is the departure time from the facility. The time zone
                        for timeOfDay is specified by timeOfDayIsUTC.
            timeOfDayIsUTC - The time zone or zones of the timeOfDay parameter. When
                             set to false, which is the default value, the timeOfDay
                             parameter refers to the time zone or zones in which the
                             facilities are located. Therefore, the start or end times
                             of the service areas are staggered by time zone.
            travelDirection - Options for traveling to or from the facility. The
                              default is defined in the network analysis layer.
                              Values: esriNATravelDirectionFromFacility |
                                      esriNATravelDirectionToFacility
            returnZ - If true, Z values will be included in saPolygons and saPolylines
                      geometry if the network dataset is Z-aware. The default is false.
            overrides - Optional dictionary. Specify additional settings that can
                        influence the behavior of the solver.  A list of supported
                        override settings for each solver and their acceptable values
                        can be obtained by contacting Esri Technical Support.
            preserve_objectid - Optional Boolean.  If True, all objectid values are
                                maintained.  The default is False.
            future - Optional Boolean.  If True, the process is run asynchronously.
                     The default is False. If True, a NAJob is returned instead of the
                     results.
            ignoreInvalidLocations - If true, the solver will ignore invalid
                                     locations. Otherwise, it will raise an error.
                                     Default is true.


    """
        if not self.properties.layerType == "esriNAServerServiceAreaLayer":
            raise TypeError("The solveServiceArea operation is supported on a network "
                             "layer of Service Area type only")

        url = self._url + "/solveServiceArea"
        params = {
                "f" : "json",
                "facilities": _handle_spatial_inputs(facilities)
                }

        if not barriers is None:
            params['barriers'] = _handle_spatial_inputs(barriers)
        if not polyline_barriers is None:
            params['polylineBarriers'] = _handle_spatial_inputs(polyline_barriers)
        if not polygon_barriers is None:
            params['polygonBarriers'] = _handle_spatial_inputs(polygon_barriers)
        if not travel_mode is None:
            params['travelMode'] = travel_mode
        if not attribute_parameter_values is None:
            params['attributeParameterValues'] = attribute_parameter_values
        if not default_breaks is None:
            params['defaultBreaks'] = default_breaks
        if not exclude_sources_from_polygons is None:
            params['excludeSourcesFromPolygons'] = exclude_sources_from_polygons
        if not merge_similar_polygon_ranges is None:
            params['mergeSimilarPolygonRanges'] = merge_similar_polygon_ranges
        if not output_lines is None:
            params['outputLines'] = output_lines
        if not output_polygons is None:
            params['outputPolygons'] = output_polygons
        if not overlap_lines is None:
            params['overlapLines'] = overlap_lines
        if not overlap_polygons is None:
            params['overlapPolygons'] = overlap_polygons
        if not split_lines_at_breaks is None:
            params['splitLinesAtBreaks'] = split_lines_at_breaks
        if not split_polygons_at_breaks is None:
            params['splitPolygonsAtBreaks'] = split_polygons_at_breaks
        if not trim_outer_polygon is None:
            params['trimOuterPolygon'] = trim_outer_polygon
        if not trim_polygon_distance is None:
            params['trimPolygonDistance'] = trim_polygon_distance
        if not trim_polygon_distance_units is None:
            params['trimPolygonDistanceUnits'] = trim_polygon_distance_units
        if not return_facilities is None:
            params['returnFacilities'] = return_facilities
        if not return_barriers is None:
            params['returnBarriers'] = return_barriers
        if not return_polyline_barriers is None:
            params['returnPolylineBarriers'] = return_polyline_barriers
        if not return_polygon_barriers is None:
            params['returnPolygonBarriers'] = return_polygon_barriers
        if not out_sr is None:
            params['outSR'] = out_sr
        if not ignore_invalid_locations is None:
            params['ignoreInvalidLocations'] = ignore_invalid_locations
        if not accumulate_attribute_names is None:
            params['accumulateAttributeNames'] = accumulate_attribute_names
        if not impedance_attribute_name is None:
            params['impedanceAttributeName'] = impedance_attribute_name
        if not restriction_attribute_names is None:
            params['restrictionAttributeNames'] = restriction_attribute_names
        if not restrict_u_turns is None:
            params['restrictUTurns'] = restrict_u_turns
        if not output_geometry_precision is None:
            params['outputGeometryPrecision'] = output_geometry_precision
        if not output_geometry_precision_units is None:
            params['outputGeometryPrecisionUnits'] = output_geometry_precision_units
        if not use_hierarchy is None:
            params['useHierarchy'] = use_hierarchy
        if not time_of_day is None:
            if isinstance(time_of_day, datetime.datetime):
                time_of_day = f"{time_of_day.timestamp() * 1000}"
            params['timeOfDay'] = time_of_day
        if not time_of_day_is_utc is None:
            params['timeOfDayIsUTC'] = time_of_day_is_utc
        if not travel_direction is None:
            params['travelDirection'] = travel_direction
        if not return_z is None:
            params['returnZ'] = return_z
        if not overrides is None:
            params['overrides'] = overrides
        if not preserve_objectid is None:
            params['preserveObjectID'] = preserve_objectid
        if future:
            f = self._run_async(self._con.post, **{'path' : url, 'postdata' : params, 'token' : self._gis._con.token})
            return NAJob(future=f, task='Solve Service Area')
        return self._con.post(path=url,
                              postdata=params)


###########################################################################
class ClosestFacilityLayer(NetworkLayer):
    """
    The Closest Facility Network Layer which has common properties of Network
    Layer as well as some attributes unique to Closest Facility Layer
    only.
    """
    def solve_closest_facility(self, incidents, facilities,
                               barriers=None,
                               polyline_barriers=None,
                               polygon_barriers=None,
                               travel_mode=None,
                               attribute_parameter_values=None,
                               return_directions=False,
                               directions_language=None,
                               directions_style_name=None,
                               directions_length_units=None,
                               directions_time_attribute_name=None,
                               return_cf_routes=True,
                               return_facilities=False,
                               return_incidents=False,
                               return_barriers=False,
                               return_polyline_barriers=False,
                               return_polygon_barriers=False,
                               output_lines=None,
                               default_cutoff=None,
                               default_target_facility_count=None,
                               travel_direction=None,
                               out_sr=None,
                               accumulate_attribute_names=None,
                               impedance_attribute_name=None,
                               restriction_attribute_names=None,
                               restrict_u_turns=None,
                               use_hierarchy=True,
                               output_geometry_precision=None,
                               output_geometry_precision_units=None,
                               time_of_day=None,
                               time_of_day_is_utc=None,
                               time_of_day_usage=None,
                               return_z=False,
                               overrides=None,
                               preserve_objectid=False,
                               future=False,
                               ignore_invalid_locations=True,
                               directions_output_type=None):
        """The solve operation is performed on a network layer resource of
        type closest facility (layerType is esriNAServerClosestFacilityLayer).
        You can provide arguments to the solve route operation as query
        parameters.
        Inputs:
            facilities  - The set of facilities loaded as network locations
                          during analysis. Facilities can be specified using
                          a simple comma / semi-colon based syntax or as a
                          JSON structure. If facilities are not specified,
                          preloaded facilities from the map document are used
                          in the analysis.
            incidents - The set of incidents loaded as network locations
                        during analysis. Incidents can be specified using
                        a simple comma / semi-colon based syntax or as a
                        JSON structure. If incidents are not specified,
                        preloaded incidents from the map document are used
                        in the analysis.
            barriers - The set of barriers loaded as network locations during
                       analysis. Barriers can be specified using a simple comma
                       / semi-colon based syntax or as a JSON structure. If
                       barriers are not specified, preloaded barriers from the
                       map document are used in the analysis. If an empty json
                       object is passed ('{}') preloaded barriers are ignored.
            polylineBarriers - The set of polyline barriers loaded as network
                               locations during analysis. If polyline barriers
                               are not specified, preloaded polyline barriers
                               from the map document are used in the analysis.
                               If an empty json object is passed ('{}')
                               preloaded polyline barriers are ignored.
            polygonBarriers - The set of polygon barriers loaded as network
                              locations during analysis. If polygon barriers
                              are not specified, preloaded polygon barriers
                              from the map document are used in the analysis.
                              If an empty json object is passed ('{}') preloaded
                              polygon barriers are ignored.
            travelMode - Travel modes provide override values that help you
                         quickly and consistently model a vehicle or mode of
                         transportation. The chosen travel mode must be
                         preconfigured on the network dataset that the routing
                         service references.
            attributeParameterValues - A set of attribute parameter values that
                                       can be parameterized to determine which
                                       network elements can be used by a vehicle.
            returnDirections - If true, directions will be generated and returned
                               with the analysis results. Default is false.
            directionsLanguage - The language to be used when computing directions.
                                 The default is the language of the server's operating
                                 system. The list of supported languages can be found
                                 in REST layer description.
            directionsOutputType -  Defines content, verbosity of returned
                                    directions. The default is esriDOTStandard.
                                    Values: esriDOTComplete | esriDOTCompleteNoEvents
                                    | esriDOTInstructionsOnly | esriDOTStandard |
                                    esriDOTSummaryOnly
            directionsStyleName - The style to be used when returning the directions.
                                  The default is as defined in the network layer. The
                                  list of supported styles can be found in REST
                                  layer description.
            directionsLengthUnits - The length units to use when computing directions.
                                    The default is as defined in the network layer.
                                    Values: esriNAUFeet | esriNAUKilometers |
                                    esriNAUMeters | esriNAUMiles |
                                    esriNAUNauticalMiles | esriNAUYards |
                                    esriNAUUnknown
            directionsTimeAttributeName - The name of network attribute to use for
                                          the drive time when computing directions.
                                          The default is as defined in the network
                                          layer.
            returnCFRoutes - If true, closest facilities routes will be returned
                             with the analysis results. Default is true.
            returnFacilities -  If true, facilities  will be returned with the
                                analysis results. Default is false.
            returnIncidents - If true, incidents will be returned with the
                              analysis results. Default is false.
            returnBarriers -  If true, barriers will be returned with the analysis
                              results. Default is false.
            returnPolylineBarriers -  If true, polyline barriers will be returned
                                      with the analysis results. Default is false.
            returnPolygonBarriers - If true, polygon barriers will be returned with
                                    the analysis results. Default is false.
            outputLines - The type of output lines to be generated in the result.
                          The default is as defined in the network layer.
                          Values: esriNAOutputLineTrueShape |
                          esriNAOutputLineTrueShapeWithMeasure |
                          esriNAOutputLineStraight | esriNAOutputLineNone
            defaultCutoff - The default cutoff value to stop traversing.
            defaultTargetFacilityCount - The default number of facilities to find.
            travelDirection - Options for traveling to or from the facility.
                              The default is defined in the network layer.
                              Values: esriNATravelDirectionFromFacility |
                              esriNATravelDirectionToFacility
            outSR - The spatial reference of the geometries returned with the
                    analysis results.
            accumulateAttributeNames - The list of network attribute names to be
                                       accumulated with the analysis. The default is
                                       as defined in the network layer. The value
                                       should be specified as a comma separated list
                                       of attribute names. You can also specify a
                                       value of none to indicate that no network
                                       attributes should be accumulated.
            impedanceAttributeName - The network attribute name to be used as the
                                     impedance attribute in analysis. The default is
                                     as defined in the network layer.
            restrictionAttributeNames -The list of network attribute names to be
                                       used as restrictions with the analysis. The
                                       default is as defined in the network layer.
                                       The value should be specified as a comma
                                       separated list of attribute names. You can
                                       also specify a value of none to indicate that
                                       no network attributes should be used as
                                       restrictions.
            restrictUTurns -  Specifies how U-Turns should be restricted in the
                              analysis. The default is as defined in the network
                              layer. Values: esriNFSBAllowBacktrack |
                              esriNFSBAtDeadEndsOnly | esriNFSBNoBacktrack |
                              esriNFSBAtDeadEndsAndIntersections
            useHierarchy -  If true, the hierarchy attribute for the network should
                            be used in analysis. The default is as defined in the
                            network layer.
            outputGeometryPrecision -  The precision of the output geometry after
                                       generalization. If 0, no generalization of
                                       output geometry is performed. The default is
                                       as defined in the network service
                                       configuration.
            outputGeometryPrecisionUnits - The units of the output geometry
                                           precision. The default value is
                                           esriUnknownUnits. Values: esriUnknownUnits
                                           | esriCentimeters | esriDecimalDegrees |
                                           esriDecimeters | esriFeet | esriInches |
                                           esriKilometers | esriMeters | esriMiles |
                                           esriMillimeters | esriNauticalMiles |
                                           esriPoints | esriYards
            timeOfDay - Arrival or departure date and time. Values: specified by
                        number of milliseconds since midnight Jan 1st, 1970, UTC.
            timeOfDayIsUTC - The time zone of the timeOfDay parameter. By setting
                             timeOfDayIsUTC to true, the timeOfDay parameter refers
                             to Coordinated Universal Time (UTC). Choose this option
                             if you want to find what's nearest for a specific time,
                             such as now, but aren't certain in which time zone the
                             facilities or incidents will be located.
            timeOfDayUsage - Defines the way timeOfDay value is used. The default
                             is as defined in the network layer.
                             Values: esriNATimeOfDayUseAsStartTime |
                             esriNATimeOfDayUseAsEndTime
            returnZ - If true, Z values will be included in the returned routes and
                       compressed geometry if the network dataset is Z-aware.
                       The default is false.
            overrides - Optional dictionary. Specify additional settings that can influence
                        the behavior of the solver.  A list of supported override settings
                        for each solver and their acceptable values can be obtained by
                        contacting Esri Technical Support.
            preserve_objectid - Optional Boolean.  If True, all objectid values are
                                maintained.  The default is False.
            future - Optional Boolean.  If True, the process is run asynchronously.
                     The default is False. If True, a NAJob is returned instead of the
                     results.
            ignoreInvalidLocations - If true, the solver will ignore invalid
                                     locations. Otherwise, it will raise an error.
                                     Default is true.

    """

        if not self.properties.layerType == "esriNAServerClosestFacilityLayer":
            raise TypeError("The solveClosestFacility operation is supported on a network "
                             "layer of Closest Facility type only")

        url = self._url + "/solveClosestFacility"
        params = {
                "f" : "json",
                "facilities": _handle_spatial_inputs(facilities),
                "incidents": _handle_spatial_inputs(incidents)
                }

        if not barriers is None:
            params['barriers'] = _handle_spatial_inputs(barriers)
        if not polyline_barriers is None:
            params['polylineBarriers'] = _handle_spatial_inputs(polyline_barriers)
        if not polygon_barriers is None:
            params['polygonBarriers'] = _handle_spatial_inputs(polygon_barriers)
        if not travel_mode is None:
            params['travelMode'] = travel_mode
        if not attribute_parameter_values is None:
            params['attributeParameterValues'] = attribute_parameter_values
        if not return_directions is None:
            params['returnDirections'] = return_directions
        if not directions_language is None:
            params['directionsLanguage'] = directions_language
        if not directions_style_name is None:
            params['directionsStyleName'] = directions_style_name
        if not directions_length_units is None:
            params['directionsLengthUnits'] = directions_length_units
        if not directions_time_attribute_name is None:
            params['directionsTimeAttributeName'] = directions_time_attribute_name
        if not directions_output_type is None:
            params['directionsOutputType'] = directions_output_type
        if not return_cf_routes is None:
            params['returnCFRoutes'] = return_cf_routes
        if not return_facilities is None:
            params['returnFacilities'] = return_facilities
        if not return_incidents is None:
            params['returnIncidents'] = return_incidents
        if not return_barriers is None:
            params['returnBarriers'] = return_barriers
        if not return_polyline_barriers is None:
            params['returnPolylineBarriers'] = return_polyline_barriers
        if not return_polygon_barriers is None:
            params['returnPolygonBarriers'] = return_polygon_barriers
        if not output_lines is None:
            params['outputLines'] = output_lines
        if not default_cutoff is None:
            params['defaultCutoff'] = default_cutoff
        if not default_target_facility_count is None:
            params['defaultTargetFacilityCount'] = default_target_facility_count
        if not travel_direction is None:
            params['travelDirection'] = travel_direction
        if not out_sr is None:
            params['outSR'] = out_sr
        if not ignore_invalid_locations is None:
            params['ignoreInvalidLocations'] = ignore_invalid_locations
        if not accumulate_attribute_names is None:
            params['accumulateAttributeNames'] = accumulate_attribute_names
        if not impedance_attribute_name is None:
            params['impedanceAttributeName'] = impedance_attribute_name
        if not restriction_attribute_names is None:
            params['restrictionAttributeNames'] = restriction_attribute_names
        if not restrict_u_turns is None:
            params['restrictUTurns'] = restrict_u_turns
        if not use_hierarchy is None:
            params['useHierarchy'] = use_hierarchy
        if not output_geometry_precision is None:
            params['outputGeometryPrecision'] = output_geometry_precision
        if not output_geometry_precision_units is None:
            params['outputGeometryPrecisionUnits'] = output_geometry_precision_units
        if not time_of_day is None:
            if isinstance(time_of_day, datetime.datetime):
                time_of_day = f"{time_of_day.timestamp() * 1000}"
            params['timeOfDay'] = time_of_day
        if not time_of_day_is_utc is None:
            params['timeOfDayIsUTC'] = time_of_day_is_utc
        if not time_of_day_usage is None:
            params['timeOfDayUsage'] = time_of_day_usage
        if not return_z is None:
            params['returnZ'] = return_z
        if not overrides is None:
            params['overrides'] = overrides
        if not preserve_objectid is None:
            params['preserveObjectID'] = preserve_objectid
        if future:
            f = self._run_async(self._con.post, **{'path' : url,
                                                   'postdata' : params})
            return NAJob(future=f, task="Solve Closest Facility")
        return self._con.post(path=url,
                              postdata=params)
###########################################################################
class ODCostMatrixLayer(NetworkLayer):
    """
    OD Cost Matrix Layer is part of the Network Layer services.  It allows users
    to generate cost matrix data for a given set of input.
    """
    def solve_od_cost_matrix(self,
                             origins,
                             destinations,
                             default_cutoff=None,
                             default_target_destination_count=None,
                             travel_mode=None,
                             output_type='Sparse Matrix',
                             time_of_day=None,
                             time_of_day_is_utc=None,
                             barriers=None,
                             polyline_barriers=None,
                             polygon_barriers=None,
                             impedance_attribute_name=None,
                             accumulate_attribute_names=None,
                             restriction_attribute_names=None,
                             attribute_parameter_values=None,
                             restrict_u_turns=None,
                             use_hierarchy=True,
                             return_origins=False,
                             return_destinations=False,
                             return_barriers=False,
                             return_polyline_barriers=False,
                             return_polygon_barriers=False,
                             out_sr=None,
                             ignore_invalid_locations=True,
                             return_z=False,
                             overrides=None,
                             future=False):
        """

        The Origin Destination Cost Matrix service helps you to create an
        origin-destination (OD) cost matrix from multiple origins to
        multiple destinations. An Origin Destination Cost Matrix is a table
        that contains the cost, such as the travel time or travel distance,
        from every origin to every destination. Additionally, it ranks the
        destinations that each origin connects to in ascending order based
        on the minimum cost required to travel from that origin to each
        destination. When generating an OD Cost Matrix, you can optionally
        specify the maximum number of destinations to find for each origin
        and the maximum time or distance to travel when searching for
        destinations.

        The results from the Origin Destination Cost Matrix service often
        become input for other spatial analyses where the cost to travel on
        the street network is more appropriate than straight-line cost.

        The travel time and/or distance for each origin-destination pair is
        stored in the output matrix (default) or as part of the attributes
        of the output lines, which can have no shapes or a straight line
        shape. Even though the lines are straight, they always store the
        travel time and/or travel distance based on the street network, not
        based on Euclidean distance.

        ====================================     ====================================================================
        **Argument**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        origins                                  Required FeatureLayer/SeDF/FeatureSet. Specifies the starting points from which to travel to the destinations.
        ------------------------------------     --------------------------------------------------------------------
        destinations                             Required FeatureLayer/SeDF/FeatureSet. Specifies the ending point locations to travel to from the origins.
        ------------------------------------     --------------------------------------------------------------------
        default_cutoff                           Optional Float. Specify the travel time or travel distance value at
                                                 which to stop searching for destinations. The default value is
                                                 `None` which means to search until all destinations are found for
                                                 every origin. The units are the same as the impedance attribute
                                                 units.
        ------------------------------------     --------------------------------------------------------------------
        default_target_destination_count         Optional Integer. Specify the number of destinations to find per
                                                 origin. The default value is `None` which means to search until all
                                                 destinations are found for every origin.
        ------------------------------------     --------------------------------------------------------------------
        travel_mode                              Optional String. Choose the mode of transportation for the analysis.
        ------------------------------------     --------------------------------------------------------------------
        output_type                              Optional String. Specify the type of output returned by the service.
                                                 Allowed value: `Sparse Matrix` (default), `Straight Lines`, or
                                                 `No Lines`.
        ------------------------------------     --------------------------------------------------------------------
        time_of_day                              Optional Datetime. The `time_of_day` value represents the time at which
                                                 the travel begins from the input origins.
                                                 If a value of `now` is passed, the travel begins at current time.
        ------------------------------------     --------------------------------------------------------------------
        time_of_day_is_utc                       Optional Boolean. Specify the time zone or zones of the
                                                 `time_of_day` parameter. The default is as defined
                                                 in the network layer.
        ------------------------------------     --------------------------------------------------------------------
        barriers                                 Optional FeatureLayer/SeDF/FeatureSet. Specify one or more points that act as
                                                 temporary restrictions or represent additional time or distance that
                                                 may be required to travel on the underlying streets.
        ------------------------------------     --------------------------------------------------------------------
        polyline_barriers                        Optional FeatureLayer/SeDF/FeatureSet. Specify one or more lines that prohibit
                                                 travel anywhere the lines intersect the streets.
        ------------------------------------     --------------------------------------------------------------------
        polygon_barriers                         Optional FeatureLayer/SeDF/FeatureSet. Specify polygons that either prohibit
                                                 travel or proportionately scale the time or distance required to
                                                 travel on the streets intersected by the polygons.
        ------------------------------------     --------------------------------------------------------------------
        impedance_attribute_name                 Optional String. Specify the impedance. The default is as defined
                                                 in the network layer.
        ------------------------------------     --------------------------------------------------------------------
        accumulate_attribute_names               Optional String. Specify whether the service should accumulate
                                                 values other than the value specified for `impedance_attribute_names`.

                                                 The default is as defined in the network layer. The parameter value
                                                 should be specified as a comma-separated list of names.
        ------------------------------------     --------------------------------------------------------------------
        restriction_attribute_names              Optional String. Specify which restrictions should be honored by the service.
        ------------------------------------     --------------------------------------------------------------------
        attribute_parameter_values               Optional String. Specify additional values required by an attribute or restriction.
        ------------------------------------     --------------------------------------------------------------------
        restrict_u_turns                         Optional String. Restrict or permit the route from making U-turns at
                                                 junctions. The default is as defined in the network layer.

                                                 Values: esriNFSBAllowBacktrack | esriNFSBAtDeadEndsOnly |
                                                         esriNFSBNoBacktrack | esriNFSBAtDeadEndsAndIntersections
        ------------------------------------     --------------------------------------------------------------------
        use_hierarchy                            Optional Boolean. Specify whether hierarchy should be used when finding the shortest paths. The default value is true.
        ------------------------------------     --------------------------------------------------------------------
        return_origins                           Optional Boolean. Specify whether origins will be returned by the service. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        return_destinations                      Optional Boolean. Specify whether origins will be returned by the service. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        return_barriers                          Optional Boolean. Specify whether barriers will be returned by the service. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        return_polyline_barriers                 Optional Boolean. Specify whether polyline barriers will be returned by the service. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        return_polygon_barriers                  Optional Boolean. Specify whether polygon barriers will be returned by the service. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        out_sr                                   Optional Integer. Specify the spatial reference of the geometries.
        ------------------------------------     --------------------------------------------------------------------
        ignore_invalid_locations                 Optional Boolean. Specify whether invalid input locations should be
                                                 ignored when finding the best solution. The default is True.
        ------------------------------------     --------------------------------------------------------------------
        return_z                                 Optional Boolean. Include z values for the returned geometries if supported by the underlying network. The default value is false.
        ------------------------------------     --------------------------------------------------------------------
        overrides                                Optional Dict. Specify additional settings that can influence the behavior of the solver.
        ------------------------------------     --------------------------------------------------------------------
        future                                   Optional boolean. If True, the result will be a `SolveJob` object and results will be returned asynchronously.
        ====================================     ====================================================================

        :returns: Dictionary or `NAJob` when `future=True`

        """
        if not self.properties.layerType == "esriNAServerODCostMatrixLayer":
            raise TypeError("The solveODCostMatrix operation is supported on a network "
                             "layer of OD Cost Matrix type only")
        url = f"{self._url}/solveODCostMatrix"
        params = {
            "f" : "json",
            "origins": _handle_spatial_inputs(origins),
            "destinations": _handle_spatial_inputs(destinations)
        }
        allowed_output_types = {
            "esriNAODOutputSparseMatrix" : "esriNAODOutputSparseMatrix",
            "Sparse Matrix" : "esriNAODOutputSparseMatrix",
            "esriNAODOutputStraightLines" : "esriNAODOutputStraightLines",
            "Straight Lines" : "esriNAODOutputStraightLines",
            "esriNAODOutputNoLines" : "esriNAODOutputNoLines",
            "No Lines" : "esriNAODOutputNoLines"
        }
        if not default_cutoff is None:
            params['defaultCutoff'] = default_cutoff
        if not default_target_destination_count is None:
            params['defaultTargetDestinationCount'] = default_target_destination_count
        if not travel_mode is None:
            params['travelMode'] = travel_mode
        if not output_type is None:
            assert output_type in allowed_output_types
            params['outputType'] = allowed_output_types[output_type]
        if not time_of_day is None:
            if isinstance(time_of_day, datetime.datetime):
                time_of_day = f"{time_of_day.timestamp() * 1000}"
            params['timeOfDay'] = time_of_day
        if not time_of_day_is_utc is None:
            params['timeOfDayIsUTC'] = time_of_day_is_utc
        if not barriers is None:
            params['barriers'] = _handle_spatial_inputs(barriers)
        if not polyline_barriers is None:
            params['polylineBarriers'] = _handle_spatial_inputs(polyline_barriers)
        if not polygon_barriers is None:
            params['polygonBarriers'] = _handle_spatial_inputs(polygon_barriers)
        if not impedance_attribute_name is None:
            params['impedanceAttributeName'] = impedance_attribute_name
        if not accumulate_attribute_names is None:
            params['accumulateAttributeNames'] = accumulate_attribute_names
        if not restriction_attribute_names is None:
            params['restrictionAttributeNames'] = restriction_attribute_names
        if not attribute_parameter_values is None:
            params['attributeParameterValues'] = attribute_parameter_values
        if not restrict_u_turns is None:
            params['restrictUTurns'] = restrict_u_turns
        if not use_hierarchy is None:
            params['useHierarchy'] = use_hierarchy
        if not return_origins is None:
            params['returnOrigins'] = return_origins
        if not return_destinations is None:
            params['returnDestinations'] = return_destinations
        if not return_barriers is None:
            params['returnBarriers'] = return_barriers
        if not return_polyline_barriers is None:
            params['returnPolylineBarriers'] = return_polyline_barriers
        if not return_polygon_barriers is None:
            params['returnPolygonBarriers'] = return_polygon_barriers
        if not out_sr is None:
            params['outSR'] = out_sr
        if not ignore_invalid_locations is None:
            params['ignoreInvalidLocations'] = ignore_invalid_locations
        if not return_z is None:
            params['returnZ'] = return_z
        if not overrides is None:
            params['overrides'] = overrides
        if future:
            f = self._run_async(self._con.post, **{'path' : url,
                                                   'postdata' : params})
            return NAJob(future=f, task="Solve OD Cost Matrix")
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def retrieve_travel_modes(self):
        """
        Identify all the valid travel modes that have been defined on the
        network dataset or in the portal if the GIS server is federated

        :returns: Dictionary

        """
        from arcgis._impl.common._isd import InsensitiveDict
        url = self._url + "/retrieveTravelModes"
        params = {"f":"json"}
        return InsensitiveDict(self._con.get(path=url,
                                             params=params))
###########################################################################
class NetworkDataset(_GISResource):
    """
    A network dataset containing a collection of network layers including route layers,
    service area layers and closest facility layers.
    """
    def __init__(self, url, gis=None):
        super(NetworkDataset, self).__init__(url, gis)
        try:
            from ..gis.server._service._adminfactory import AdminServiceGen
            self.service = AdminServiceGen(service=self, gis=gis)
        except: pass
        self._closestFacilityLayers = []
        self._routeLayers = []
        self._serviceAreaLayers = []
        self._odCostMatrix = []
        self._load_layers()

    @classmethod
    def fromitem(cls, item):
        """Creates a network dataset from a 'Network Analysis Service' Item in the GIS"""
        if not item.type == 'Network Analysis Service':
            raise TypeError("item must be a type of Network Analysis Service, not " + item.type)

        return cls(item.url, item._gis)

    #----------------------------------------------------------------------
    def _load_layers(self):
        """loads the various layer types"""
        self._closestFacilityLayers = []
        self._routeLayers = []
        self._serviceAreaLayers = []
        self._odCostMatrix = []
        params = {
            "f" : "json",
        }
        json_dict = self._con.get(path=self._url, params=params)
        for k,v in json_dict.items():
            if k == "routeLayers" and json_dict[k]:
                self._routeLayers = []
                for rl in v:
                    self._routeLayers.append(
                        RouteLayer(url=self._url + "/%s" % rl,
                                   gis=self._gis))
            elif k == "serviceAreaLayers" and json_dict[k]:
                self._serviceAreaLayers = []
                for sal in v:
                    self._serviceAreaLayers.append(
                        ServiceAreaLayer(url=self._url + "/%s" % sal,
                                         gis=self._gis))
            elif k == "closestFacilityLayers" and json_dict[k]:
                self._closestFacilityLayers = []
                for cf in v:
                    self._closestFacilityLayers.append(
                        ClosestFacilityLayer(url=self._url + "/%s" % cf,
                                             gis=self._gis))
            elif k == "odCostMatrixLayers" and json_dict[k]:
                self._odCostMatrix = []
                for cf in v:
                    self._odCostMatrix.append(
                        ODCostMatrixLayer(url=self._url + "/%s" % cf,
                                             gis=self._gis))

    #----------------------------------------------------------------------
    @property
    def route_layers(self):
        """List of route layers in this network dataset"""
        if self._routeLayers is None:
            self._load_layers()
        return self._routeLayers
    #----------------------------------------------------------------------
    @property
    def service_area_layers(self):
        """List of service area layers in this network dataset"""
        if self._serviceAreaLayers is None:
            self._load_layers()
        return self._serviceAreaLayers
    #----------------------------------------------------------------------
    @property
    def closest_facility_layers(self):
        """List of closest facility layers in this network dataset"""
        if self._closestFacilityLayers is None:
            self._load_layers()
        return self._closestFacilityLayers

    @property
    def od_cost_matrix_layers(self):
        """
        List of OD Cost Matrix Layers

        :returns: List
        """
        if self._odCostMatrix is None:
            self._load_layers()
        return self._odCostMatrix
