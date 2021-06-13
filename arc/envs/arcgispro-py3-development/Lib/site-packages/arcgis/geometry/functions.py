"""
Functions which take geometric types as parameters and return geometric type results.
"""

import arcgis.env

# https://utility.arcgisonline.com/ArcGIS/rest/services/Geometry/GeometryServer.

def areas_and_lengths(polygons,
                      length_unit,
                      area_unit,
                      calculation_type,
                      spatial_ref=4326,
                      gis=None, 
                      future=False):
    """
       The areas_and_lengths function calculates areas and perimeter lengths
       for each polygon specified in the input array.

       Inputs:
          polygons - The array of polygons whose areas and lengths are
                     to be computed.
          length_unit - The length unit in which the perimeters of
                       polygons will be calculated. If calculation_type
                       is planar, then length_unit can be any esriUnits
                       constant. If lengthUnit is not specified, the
                       units are derived from spatial_ref. If calculationType is
                       not planar, then lengthUnit must be a linear
                       esriUnits constant, such as esriSRUnit_Meter or
                       esriSRUnit_SurveyMile. If length_unit is not
                       specified, the units are meters. For a list of
                       valid units, see esriSRUnitType Constants and
                       esriSRUnit2Type Constant.
          area_unit - The area unit in which areas of polygons will be
                     calculated. If calculation_type is planar, then
                     area_unit can be any esriUnits constant. If
                     area_unit is not specified, the units are derived
                     from spatial_ref. If calculation_type is not planar, then
                     area_unit must be a linear esriUnits constant such
                     as esriSRUnit_Meter or esriSRUnit_SurveyMile. If
                     area_unit is not specified, then the units are
                     meters. For a list of valid units, see
                     esriSRUnitType Constants and esriSRUnit2Type
                     constant.
                     The list of valid esriAreaUnits constants include,
                     esriSquareInches | esriSquareFeet |
                     esriSquareYards | esriAcres | esriSquareMiles |
                     esriSquareMillimeters | esriSquareCentimeters |
                     esriSquareDecimeters | esriSquareMeters | esriAres
                     | esriHectares | esriSquareKilometers.
          calculation_type -  The type defined for the area and length
                             calculation of the input geometries. The
                             type can be one of the following values:
                             planar - Planar measurements use 2D
                                      Euclidean distance to calculate
                                      area and length. Th- should
                                      only be used if the area or
                                      length needs to be calculated in
                                      the given spatial reference.
                                      Otherwise, use preserveShape.
                             geodesic - Use this type if you want to
                                      calculate an area or length using
                                      only the vertices of the polygon
                                      and define the lines between the
                                      points as geodesic segments
                                      independent of the actual shape
                                      of the polygon. A geodesic
                                      segment is the shortest path
                                      between two points on an ellipsoid.
                             preserveShape - This type calculates the
                                      area or length of the geometry on
                                      the surface of the Earth
                                      ellipsoid. The shape of the
                                      geometry in its coordinate system
                                      is preserved.
        future - boolean. This operation determines if the job is run asynchronously or not.
       Output:
          JSON as dictionary
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.areas_and_lengths(
        polygons,
        length_unit,
        area_unit,
        calculation_type,
        spatial_ref, future=future)


def auto_complete(polygons=None,
                  polylines=None,
                  spatial_ref=None,
                  gis=None, future=False):
    """
       The auto_complete function simplifies the process of
       constructing new polygons that are adjacent to other polygons.
       It constructs polygons that fill in the gaps between existing
       polygons and a set of polylines.

       Inputs:
        polygons -
         array of Polygon objects
        polylines -
         list of Polyline objects
        spatial_ref -
         spatial reference of the input geometries WKID
        future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.auto_complete(
        polygons,
        polylines,
        spatial_ref, future=future)


def buffer(geometries,
           in_sr,
           distances,
           unit,
           out_sr=None,
           buffer_sr=None,
           union_results=None,
           geodesic=None,
           gis=None, future=False):
    """
       The buffer function is performed on a geometry service resource
       The result of this function is buffered polygons at the
       specified distances for the input geometry array. Options are
       available to union buffers and to use geodesic distance.

       Inputs:

         geometries -
          The array of geometries to be buffered.
         in_sr -
          The well-known ID of the spatial reference or a spatial
          reference JSON object for the input geometries.
         distances -
          The distances that each of the input geometries is
          buffered.
         unit - The units for calculating each buffer distance. If unit
          is not specified, the units are derived from bufferSR. If
          bufferSR is not specified, the units are derived from in_sr.
         out_sr - The well-known ID of the spatial reference or a
          spatial reference JSON object for the input geometries.
         buffer_sr - The well-known ID of the spatial reference or a
          spatial reference JSON object for the input geometries.
         union_results -  If true, all geometries buffered at a given
          distance are unioned into a single (gis,possibly multipart)
          polygon, and the unioned geometry is placed in the output
          array. The default is false
         geodesic - Set geodesic to true to buffer the input geometries
          using geodesic distance. Geodesic distance is the shortest
          path between two points along the ellipsoid of the earth. If
          geodesic is set to false, the 2D Euclidean distance is used
          to buffer the input geometries. The default value depends on
          the geometry type, unit and bufferSR.
        future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.buffer(
        geometries,
        in_sr,
        distances,
        unit,
        out_sr,
        buffer_sr,
        union_results,
        geodesic, future=future)

def convex_hull(geometries,
                spatial_ref=None,
                gis=None, future=False):
    """
    The convex_hull function is performed on a geometry service
    resource. It returns the convex hull of the input geometry. The
    input geometry can be a point, multipoint, polyline, or polygon.
    The convex hull is typically a polygon but can also be a polyline
    or point in degenerate cases.

    Inputs:
       geometries - The geometries whose convex hull is to be created.
       spatial_ref - The well-known ID or a spatial reference JSON object for
            the output geometry.
       future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.convex_hull(
        geometries,
        spatial_ref, future=future)

def cut(cutter,
        target,
        spatial_ref=None,
        gis=None, future=False):
    """
    The cut function is performed on a geometry service resource. This
    function splits the target polyline or polygon where it's crossed
    by the cutter polyline.
    At 10.1 and later, this function calls simplify on the input
    cutter and target geometries.

    Inputs:
       cutter - The polyline that will be used to divide the target
        into pieces where it crosses the target.The spatial reference
        of the polylines is specified by spatial_ref. The structure of the
        polyline is the same as the structure of the JSON polyline
        objects returned by the ArcGIS REST API.
       target - The array of polylines/polygons to be cut. The
        structure of the geometry is the same as the structure of the
        JSON geometry objects returned by the ArcGIS REST API. The
        spatial reference of the target geometry array is specified by
        spatial_ref.
       spatial_ref - The well-known ID or a spatial reference JSON object for
        the output geometry.
       future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.cut(
        cutter,
        target,
        spatial_ref, future=future)

def densify(geometries,
            spatial_ref,
            max_segment_length,
            length_unit,
            geodesic=False,
            gis=None, future=False):
    """
    The densify function is performed using the GIS's geometry engine.
    This function densifies geometries by plotting points between
    existing vertices.

    Inputs:
       geometries - The array of geometries to be densified. The
        structure of each geometry in the array is the same as the
        structure of the JSON geometry objects returned by the ArcGIS
        REST API.
       spatial_ref - The well-known ID or a spatial reference JSON object for
        the input polylines. For a list of valid WKID values, see
        Projected coordinate systems and Geographic coordinate systems.
       max_segment_length - All segments longer than maxSegmentLength are
        replaced with sequences of lines no longer than
        max_segment_length.
       length_unit - The length unit of max_segment_length. If geodesic is
        set to false, then the units are derived from spatial_ref, and
        length_unit is ignored. If geodesic is set to true, then
        length_unit must be a linear unit. In a case where length_unit is
        not specified and spatial_ref is a PCS, the units are derived from spatial_ref.
        In a case where length_unit is not specified and spatial_ref is a GCS,
        then the units are meters.
       geodesic - If geodesic is set to true, then geodesic distance is
        used to calculate max_segment_length. Geodesic distance is the
        shortest path between two points along the ellipsoid of the
        earth. If geodesic is set to false, then 2D Euclidean distance
        is used to calculate max_segment_length. The default is false.
       future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.densify(
        geometries,
        spatial_ref,
        max_segment_length,
        length_unit,
        geodesic, future=future)

def difference(geometries,
               spatial_ref,
               geometry,
               gis=None, future=False):
    """
    The difference function is performed on a geometry service
    resource. This function constructs the set-theoretic difference
    between each element of an array of geometries and another geometry
    the so-called difference geometry. In other words, let B be the
    difference geometry. For each geometry, A, in the input geometry
    array, it constructs A-B.

    Inputs:
      geometries -  An array of points, multipoints, polylines or
       polygons. The structure of each geometry in the array is the
       same as the structure of the JSON geometry objects returned by
       the ArcGIS REST API.
      geometry - A single geometry of any type and of a dimension equal
       to or greater than the elements of geometries. The structure of
       geometry is the same as the structure of the JSON geometry
       objects returned by the ArcGIS REST API. The use of simple
       syntax is not supported.
      spatial_ref - The well-known ID of the spatial reference or a spatial
       reference JSON object for the input geometries.
      future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.difference(
        geometries,
        spatial_ref,
        geometry, future=future)

def distance(spatial_ref,
             geometry1,
             geometry2,
             distance_unit="",
             geodesic=False,
             gis=None, future=False):
    """
    The distance function is performed on a geometry service resource.
    It reports the 2D Euclidean or geodesic distance between the two
    geometries.

    Inputs:
     spatial_ref - The well-known ID or a spatial reference JSON object for
      input geometries.
     geometry1 - The geometry from which the distance is to be
      measured. The structure of the geometry is same as the structure
      of the JSON geometry objects returned by the ArcGIS REST API.
     geometry2 - The geometry from which the distance is to be
      measured. The structure of the geometry is same as the structure
      of the JSON geometry objects returned by the ArcGIS REST API.
     distanceUnit - specifies the units for measuring distance between
      the geometry1 and geometry2 geometries.
     geodesic - If geodesic is set to true, then the geodesic distance
      between the geometry1 and geometry2 geometries is returned.
      Geodesic distance is the shortest path between two points along
      the ellipsoid of the earth. If geodesic is set to false or not
      specified, the planar distance is returned. The default value is
      false.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.distance(
        spatial_ref,
        geometry1,
        geometry2,
        distance_unit,
        geodesic, future=future)

def find_transformation(in_sr, out_sr, extent_of_interest=None, num_of_results=1, gis=None, future=False):
    """
    The find_transformations function is performed on a geometry
    service resource. This function returns a list of applicable
    geographic transformations you should use when projecting
    geometries from the input spatial reference to the output spatial
    reference. The transformations are in JSON format and are returned
    in order of most applicable to least applicable. Recall that a
    geographic transformation is not needed when the input and output
    spatial references have the same underlying geographic coordinate
    systems. In this case, findTransformations returns an empty list.
    Every returned geographic transformation is a forward
    transformation meaning that it can be used as-is to project from
    the input spatial reference to the output spatial reference. In the
    case where a predefined transformation needs to be applied in the
    reverse direction, it is returned as a forward composite
    transformation containing one transformation and a transformForward
    element with a value of false.

    Inputs:
       in_sr - The well-known ID (gis,WKID) of the spatial reference or a
         spatial reference JSON object for the input geometries
       out_sr - The well-known ID (gis,WKID) of the spatial reference or a
         spatial reference JSON object for the input geometries
       extent_of_interest -  The bounding box of the area of interest
         specified as a JSON envelope. If provided, the extent of
         interest is used to return the most applicable geographic
         transformations for the area. If a spatial reference is not
         included in the JSON envelope, the in_sr is used for the
         envelope.
       num_of_results - The number of geographic transformations to
         return. The default value is 1. If num_of_results has a value of
         -1, all applicable transformations are returned.
       future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.find_transformation(in_sr, out_sr,
                                                    extent_of_interest, num_of_results, future=future)


def from_geo_coordinate_string(spatial_ref, strings,
                               conversion_type, conversion_mode=None, gis=None, future=False):
    """
    The from_geo_coordinate_string function is performed on a geometry
    service resource. The function converts an array of well-known
    strings into xy-coordinates based on the conversion type and
    spatial reference supplied by the user. An optional conversion mode
    parameter is available for some conversion types.

    Inputs:
     spatial_ref - The well-known ID of the spatial reference or a spatial
      reference json object.
     strings - An array of strings formatted as specified by
      conversion_type.
      Syntax: [<string1>,...,<stringN>]
      Example: ["01N AA 66021 00000","11S NT 00000 62155",
                "31U BT 94071 65288"]
     conversion_type - The conversion type of the input strings.
      Valid conversion types are:
       MGRS - Military Grid Reference System
       USNG - United States National Grid
       UTM - Universal Transverse Mercator
       GeoRef - World Geographic Reference System
       GARS - Global Area Reference System
       DMS - Degree Minute Second
       DDM - Degree Decimal Minute
       DD - Decimal Degree
     conversion_mode - Conversion options for MGRS, UTM and GARS
      conversion types.
      Conversion options for MGRS and UTM conversion types.
      Valid conversion modes for MGRS are:
       mgrsDefault - Default. Uses the spheroid from the given spatial
        reference.
       mgrsNewStyle - Treats all spheroids as new, like WGS 1984. The
        180 degree longitude falls into Zone 60.
       mgrsOldStyle - Treats all spheroids as old, like Bessel 1841.
        The 180 degree longitude falls into Zone 60.
       mgrsNewWith180InZone01 - Same as mgrsNewStyle except the 180
        degree longitude falls into Zone 01.
       mgrsOldWith180InZone01 - Same as mgrsOldStyle except the 180
        degree longitude falls into Zone 01.
      Valid conversion modes for UTM are:
       utmDefault - Default. No options.
       utmNorthSouth - Uses north/south latitude indicators instead of
        zone numbers. Non-standard. Default is recommended
      future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.from_geo_coordinate_string(spatial_ref, strings,
                                                           conversion_type, conversion_mode, future=future)


def generalize(spatial_ref,
               geometries,
               max_deviation,
               deviation_unit,
               gis=None, future=False):
    """
    The generalize function is performed on a geometry service
    resource. The generalize function simplifies the input geometries
    using the Douglas-Peucker algorithm with a specified maximum
    deviation distance. The output geometries will contain a subset of
    the original input vertices.

    Inputs:
     spatial_ref - The well-known ID or a spatial reference JSON object for the
      input geometries.
     geometries - The array of geometries to be generalized.
     max_deviation - max_deviation sets the maximum allowable offset,
      which will determine the degree of simplification. This value
      limits the distance the output geometry can differ from the input
      geometry.
     deviation_unit - A unit for maximum deviation. If a unit is not
      specified, the units are derived from spatial_ref.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.generalize(
        spatial_ref,
        geometries,
        max_deviation,
        deviation_unit, future=future)

def intersect(spatial_ref,
              geometries,
              geometry,
              gis=None, future=False):
    """
    The intersect function is performed on a geometry service
    resource. This function constructs the set-theoretic intersection
    between an array of geometries and another geometry. The dimension
    of each resultant geometry is the minimum dimension of the input
    geometry in the geometries array and the other geometry specified
    by the geometry parameter.

    Inputs:
     spatial_ref - The well-known ID or a spatial reference JSON object for the
      input geometries.
     geometries - An array of points, multipoints, polylines, or
      polygons. The structure of each geometry in the array is the same
      as the structure of the JSON geometry objects returned by the
      ArcGIS REST API.
     geometry - A single geometry of any type with a dimension equal to
      or greater than the elements of geometries.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.intersect(spatial_ref,
                                          geometries,
                                          geometry, future=future)

def label_points(spatial_ref,
                 polygons,
                 gis=None, future=False):
    """
    The label_points function is performed on a geometry service
    resource. The labelPoints function calculates an interior point
    for each polygon specified in the input array. These interior
    points can be used by clients for labeling the polygons.

    Inputs:
     spatial_ref - The well-known ID of the spatial reference or a spatial
      reference JSON object for the input polygons.
     polygons - The array of polygons whose label points are to be
      computed. The spatial reference of the polygons is specified by
      spatial_ref.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.label_points(spatial_ref, polygons, future=future)


def lengths(spatial_ref,
            polylines,
            length_unit,
            calculation_type,
            gis=None, future=False):
    """
    The lengths function is performed on a geometry service resource.
    This function calculates the 2D Euclidean or geodesic lengths of
    each polyline specified in the input array.

    Inputs:
     spatial_ref - The well-known ID of the spatial reference or a spatial
      reference JSON object for the input polylines.
     polylines - The array of polylines whose lengths are to be
      computed.
     length_unit - The unit in which lengths of polylines will be
      calculated. If calculation_type is planar, then length_unit can be
      any esriUnits constant. If calculation_type is planar and
      length_unit is not specified, then the units are derived from spatial_ref.
      If calculation_type is not planar, then length_unit must be a
      linear esriUnits constant such as esriSRUnit_Meter or
      esriSRUnit_SurveyMile. If calculation_type is not planar and
      length_unit is not specified, then the units are meters.
     calculation_type - calculation_type defines the length calculation
      for the geometry. The type can be one of the following values:
        planar - Planar measurements use 2D Euclidean distance to
         calculate length. This type should only be used if the length
         needs to be calculated in the given spatial reference.
         Otherwise, use preserveShape.
        geodesic - Use this type if you want to calculate a length
         using only the vertices of the polygon and define the lines
         between the vertices as geodesic segments independent of the
         actual shape of the polyline. A geodesic segment is the
         shortest path between two points on an earth ellipsoid.
        preserveShape - This type calculates the length of the geometry
         on the surface of the earth ellipsoid. The shape of the
         geometry in its coordinate system is preserved.
        future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.lengths(
        spatial_ref,
        polylines,
        length_unit,
        calculation_type, future=future)

def offset(geometries,
           offset_distance,
           offset_unit,
           offset_how="esriGeometryOffsetRounded",
           bevel_ratio=10,
           simplify_result=False,
           spatial_ref=None,
           gis=None, future=False):
    """
    The offset function is performed on a geometry service resource.
    This function constructs geometries that are offset from the
    given input geometries. If the offset parameter is positive, the
    constructed offset will be on the right side of the geometry. Left
    side offsets are constructed with negative parameters. Tracing the
    geometry from its first vertex to the last will give you a
    direction along the geometry. It is to the right and left
    perspective of this direction that the positive and negative
    parameters will dictate where the offset is constructed. In these
    terms, it is simple to infer where the offset of even horizontal
    geometries will be constructed.

    Inputs:
     geometries -  The array of geometries to be offset.
     offset_distance - Specifies the distance for constructing an offset
      based on the input geometries. If the offset_distance parameter is
      positive, the constructed offset will be on the right side of the
      curve. Left-side offsets are constructed with negative values.
     offset_unit - A unit for offset distance. If a unit is not
      specified, the units are derived from spatial_ref.
     offset_how - The offset_how parameter determines how outer corners
      between segments are handled. The three options are as follows:
       esriGeometryOffsetRounded - Rounds the corner between extended
        offsets.
       esriGeometryOffsetBevelled - Squares off the corner after a
        given ratio distance.
       esriGeometryOffsetMitered - Attempts to allow extended offsets
        to naturally intersect, but if that intersection occurs too far
        from the corner, the corner is eventually bevelled off at a
        fixed distance.
     bevel_ratio - bevel_ratio is multiplied by the offset distance, and
      the result determines how far a mitered offset intersection can
      be located before it is bevelled. When mitered is specified,
      bevel_ratio is ignored and 10 is used internally. When bevelled is
      specified, 1.1 will be used if bevel_ratio is not specified.
      bevel_ratio is ignored for rounded offset.
     simplify_result - if simplify_result is set to true, then self
      intersecting loops will be removed from the result offset
      geometries. The default is false.
     spatial_ref - The well-known ID or a spatial reference JSON object for the
      input geometries.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.offset(
        geometries,
        offset_distance,
        offset_unit,
        offset_how,
        bevel_ratio,
        simplify_result,
        spatial_ref, future=future)


def project(geometries,
            in_sr,
            out_sr,
            transformation="",
            transform_forward=False,
            gis=None, future=False):
    """
    The project function is performed on a geometry service resource.
    This function projects an array of input geometries from the input
    spatial reference to the output spatial reference.

    Inputs:
     geometries - The list of geometries to be projected.
     in_sr - The well-known ID (gis,WKID) of the spatial reference or a
      spatial reference JSON object for the input geometries.
     out_sr - The well-known ID (gis,WKID) of the spatial reference or a
      spatial reference JSON object for the input geometries.
     transformation - The WKID or a JSON object specifying the
      geographic transformation (gis,also known as datum transformation) to
      be applied to the projected geometries. Note that a
      transformation is needed only if the output spatial reference
      contains a different geographic coordinate system than the input
      spatial reference.
     transform_forward - A Boolean value indicating whether or not to
      transform forward. The forward or reverse direction of
      transformation is implied in the name of the transformation. If
      transformation is specified, a value for the transformForward
      parameter must also be specified. The default value is false.
     future - boolean. This operation determines if the job is run asynchronously or not.

    Example:
     input_geom = [{"x": -17568824.55, "y": 2428377.35}, {"x": -17568456.88, "y": 2428431.352}]
     result = project(geometries = input_geom, in_sr = 3857, out_sr = 4326)

    returns:
     a list of geometries in the out_sr coordinate system, for instance:
     [{"x": -157.82343617279275, "y": 21.305781607280093}, {"x": -157.8201333369876, "y": 21.306233559873714}]
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.project(
        geometries,
        in_sr,
        out_sr,
        transformation,
        transform_forward, future=future)


def relation(geometries1,
             geometries2,
             spatial_ref,
             spatial_relation="esriGeometryRelationIntersection",
             relation_param="",
             gis=None, future=False):
    """
    The relation function is performed on a geometry service resource.
    This function determines the pairs of geometries from the input
    geometry arrays that participate in the specified spatial relation.
    Both arrays are assumed to be in the spatial reference specified by
    spatial_ref, which is a required parameter. Geometry types cannot be mixed
    within an array. The relations are evaluated in 2D. In other words,
    z coordinates are not used.

    Inputs:
     geometries1 - The first array of geometries used to compute the
      relations.
     geometries2 -The second array of geometries used to compute the
     relations.
     spatial_ref - The well-known ID of the spatial reference or a spatial
      reference JSON object for the input geometries.
     spatial_relation - The spatial relationship to be tested between the two
      input geometry arrays.
      Values: esriGeometryRelationCross | esriGeometryRelationDisjoint |
      esriGeometryRelationIn | esriGeometryRelationInteriorIntersection |
      esriGeometryRelationIntersection | esriGeometryRelationLineCoincidence |
      esriGeometryRelationLineTouch | esriGeometryRelationOverlap |
      esriGeometryRelationPointTouch | esriGeometryRelationTouch |
      esriGeometryRelationWithin | esriGeometryRelationRelation
     relation_param - The Shape Comparison Language string to be
      evaluated.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.relation(
        geometries1,
        geometries2,
        spatial_ref,
        spatial_relation,
        relation_param, future=future)


def reshape(spatial_ref,
            target,
            reshaper,
            gis=None, future=False):
    """
    The reshape function is performed on a geometry service resource.
    It reshapes a polyline or polygon feature by constructing a
    polyline over the feature. The feature takes the shape of the
    reshaper polyline from the first place the reshaper intersects the
    feature to the last.

    Input:
     spatial_ref - The well-known ID of the spatial reference or a spatial
      reference JSON object for the input geometries.
     target -  The polyline or polygon to be reshaped.
     reshaper - The single-part polyline that does the reshaping.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.reshape(
        spatial_ref,
        target,
        reshaper, future=future)


def simplify(spatial_ref,
             geometries,
             gis=None, future=False):
    """
    The simplify function is performed on a geometry service resource.
    Simplify permanently alters the input geometry so that the geometry
    becomes topologically consistent. This resource applies the ArcGIS
    simplify function to each geometry in the input array.

    Inputs:
    spatial_ref - The well-known ID of the spatial reference or a spatial
      reference JSON object for the input geometries.
    geometries - The array of geometries to be simplified.
    future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.simplify(spatial_ref, geometries, future=future)


def to_geo_coordinate_string(spatial_ref,
                             coordinates,
                             conversion_type,
                             conversion_mode="mgrsDefault",
                             num_of_digits=None,
                             rounding=True,
                             add_spaces=True,
                             gis=None, future=False):
    """
    The to_geo_coordinate_string function is performed on a geometry
    service resource. The function converts an array of
    xy-coordinates into well-known strings based on the conversion type
    and spatial reference supplied by the user. Optional parameters are
    available for some conversion types. Note that if an optional
    parameter is not applicable for a particular conversion type, but a
    value is supplied for that parameter, the value will be ignored.

    Inputs:
      spatial_ref -  The well-known ID of the spatial reference or a spatial
       reference json object.
      coordinates - An array of xy-coordinates in JSON format to be
       converted. Syntax: [[x1,y2],...[xN,yN]]
      conversion_type - The conversion type of the input strings.
       Allowed Values:
        MGRS - Military Grid Reference System
        USNG - United States National Grid
        UTM - Universal Transverse Mercator
        GeoRef - World Geographic Reference System
        GARS - Global Area Reference System
        DMS - Degree Minute Second
        DDM - Degree Decimal Minute
        DD - Decimal Degree
      conversion_mode - Conversion options for MGRS and UTM conversion
       types.
       Valid conversion modes for MGRS are:
        mgrsDefault - Default. Uses the spheroid from the given spatial
         reference.
        mgrsNewStyle - Treats all spheroids as new, like WGS 1984. The
         180 degree longitude falls into Zone 60.
        mgrsOldStyle - Treats all spheroids as old, like Bessel 1841.
         The 180 degree longitude falls into Zone 60.
        mgrsNewWith180InZone01 - Same as mgrsNewStyle except the 180
         degree longitude falls into Zone 01.
        mgrsOldWith180InZone01 - Same as mgrsOldStyle except the 180
         degree longitude falls into Zone 01.
       Valid conversion modes for UTM are:
        utmDefault - Default. No options.
        utmNorthSouth - Uses north/south latitude indicators instead of
         zone numbers. Non-standard. Default is recommended.
      num_of_digits - The number of digits to output for each of the
       numerical portions in the string. The default value for
       num_of_digits varies depending on conversion_type.
      rounding - If true, then numeric portions of the string are
       rounded to the nearest whole magnitude as specified by
       numOfDigits. Otherwise, numeric portions of the string are
       truncated. The rounding parameter applies only to conversion
       types MGRS, USNG and GeoRef. The default value is true.
      addSpaces - If true, then spaces are added between components of
       the string. The addSpaces parameter applies only to conversion
       types MGRS, USNG and UTM. The default value for MGRS is false,
       while the default value for both USNG and UTM is true.
      future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.to_geo_coordinate_string(
        spatial_ref,
        coordinates,
        conversion_type,
        conversion_mode,
        num_of_digits,
        rounding,
        add_spaces, future=future)


def trim_extend(spatial_ref,
                polylines,
                trim_extend_to,
                extend_how=0,
                gis=None, future=False):
    """
    The trim_extend function is performed on a geometry service
    resource. This function trims or extends each polyline specified
    in the input array, using the user-specified guide polylines. When
    trimming features, the part to the left of the oriented cutting
    line is preserved in the output, and the other part is discarded.
    An empty polyline is added to the output array if the corresponding
    input polyline is neither cut nor extended.

    Inputs:
     spatial_ref - The well-known ID of the spatial reference or a spatial
       reference json object.
     polylines - An array of polylines to be trimmed or extended.
     trim_extend_to - A polyline that is used as a guide for trimming or
      extending input polylines.
     extend_how - A flag that is used along with the trimExtend
      function.
      0 - By default, an extension considers both ends of a path. The
       old ends remain, and new points are added to the extended ends.
       The new points have attributes that are extrapolated from
       adjacent existing segments.
      1 - If an extension is performed at an end, relocate the end
       point to the new position instead of leaving the old point and
       adding a new point at the new position.
      2 - If an extension is performed at an end, do not extrapolate
       the end-segment's attributes for the new point. Instead, make
       its attributes the same as the current end. Incompatible with
       esriNoAttributes.
      4 - If an extension is performed at an end, do not extrapolate
       the end-segment's attributes for the new point. Instead, make
       its attributes empty. Incompatible with esriKeepAttributes.
      8 - Do not extend the 'from' end of any path.
      16 - Do not extend the 'to' end of any path.
     future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.trim_extend(spatial_ref,
                                            polylines,
                                            trim_extend_to,
                                            extend_how, future=future)


def union(spatial_ref,
          geometries,
          gis=None, future=False):
    """
    The union function is performed on a geometry service resource.
    This function constructs the set-theoretic union of the geometries
    in the input array. All inputs must be of the same type.

    Inputs:
    spatial_ref - The well-known ID of the spatial reference or a spatial
     reference json object.
    geometries - The array of geometries to be unioned.
    future - boolean. This operation determines if the job is run asynchronously or not.
    """
    if gis is None:
        gis = arcgis.env.active_gis
    return gis._tools.geometry.union(spatial_ref, geometries, future=future)
