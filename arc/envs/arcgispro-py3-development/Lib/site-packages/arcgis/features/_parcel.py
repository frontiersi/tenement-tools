import time
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features import FeatureLayer, FeatureLayerCollection
from arcgis.features._version import Version, VersionManager

########################################################################
class ParcelFabricManager(object):
    """
    The Parcel Fabric Server is responsible for exposing parcel management
    capabilities to support a variety of workflows from different clients
    and systems.

    ====================     ====================================================================
    **Argument**             **Description**
    --------------------     --------------------------------------------------------------------
    url                      Required String. The URI to the service endpoint.
    --------------------     --------------------------------------------------------------------
    gis                      Required GIS. The enterprise connection.
    --------------------     --------------------------------------------------------------------
    version                  Required Version. This is the version object where the modification
                             will occur.
    --------------------     --------------------------------------------------------------------
    flc                      Required FeatureLayerCollection. This is the parent container for
                             ParcelFabricManager.
    ====================     ====================================================================

    """
    _con = None
    _flc = None
    _gis = None
    _url = None
    _version = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self,
                 url,
                 gis,
                 version,
                 flc):
        """Constructor"""
        self._url = url
        self._gis = gis
        self._con = gis._portal.con
        self._version = version
        self._flc = flc
    #----------------------------------------------------------------------
    def __str__(self):
        return "<ParcelFabricManager @ %s>" % self._url
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    def __enter__(self):
        return self
    #----------------------------------------------------------------------
    def __exit__(self, type, value, traceback):
        return
    #----------------------------------------------------------------------
    @property
    def layer(self):
        """returns the Parcel Layer for the service"""
        if "controllerDatasetLayers" in self._flc.properties and \
           "parcelLayerId" in self._flc.properties.controllerDatasetLayers:
            url = "%s/%s" % (self._flc.url,
                             self._flc.properties.controllerDatasetLayers.parcelLayerId)
            return FeatureLayer(url=url, gis=self._gis)
        return None
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the service"""
        if self._properties is None:

            res = self._con.get(self._url, {'f':'json'})
            self._properties = PropertyMap(res)
        return self._properties
    #----------------------------------------------------------------------
    def assign_to_record(self,
                         features,
                         record,
                         write_attribute,
                         moment=None):
        """
        Assigns the specified parcel features to the specified record. If
        parcel polygons are assigned, the record polygon will be updated to
        match the cumulative geometry of all the parcels associated to it.
        The Created By Record or Retired By Record attribute field of the
        parcel features is updated with the global ID of the assigned
        record.

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        features                 Required List. The parcel features to assign to the specified record.
                                 Can be parcels, parcel polygons, parcel points, and parcel lines.

                                 Syntax: parcelFeatures=[{"id":"<guid>","layerId":"<layerID>"},{...}]

        --------------------     --------------------------------------------------------------------
        record                   Required String. The record that will be assigned to the specified
                                 parcel features.
        --------------------     --------------------------------------------------------------------
        write_attribute          Required String. Represents the record field to update on the parcel
                                 features. Either the Created By Record or Retired By Record field is
                                 to be updated with the global ID of the assigned record.

                                 Allowed Values: `CreatedByRecord` or `RetiredByRecord`

        --------------------     --------------------------------------------------------------------
        moment                   Optional Integer. This should only be specified by the client when
                                 they do not want to use the current moment
        ====================     ====================================================================

        :returns: Boolean

        """
        url = "{base}/assignFeaturesToRecord".format(base=self._url)
        if moment is None:
            moment = int(time.time())
        params = {
            "gdbVersion" : self._version.properties.versionName,
            "sessionId" : self._version._guid,
            "moment" : moment,
            "parcelFeatures" : features,
            "record" : record,
            "writeAttribute" : write_attribute,
            "f": "json"
        }
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def build(self,
              extent=None,
              moment=None,
              return_errors=False,
              record=None):
        """
        A `build` will fix known parcel fabric errors.

        For example, if a parcel polygon exists without lines, then build will
        construct the missing lines. If lines are missing, the polygon row(s)
        are created. When constructing this objects, build will attribute the
        related keys as appropriate. Build also maintains `lineage` and `record`
        features. The parcel fabric must have sufficient information for build
        to work correctly. Ie, source reference document, and connected lines.

        Build provides options to increase performance. The process can just
        work on specific parcels, geometry types or only respond to parcel point
        movement in the case of an adjustment.

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        extent                   Optional Envelope. The extent to build.

                                 Syntax: {"xmin":X min,"ymin": y min, "xmax": x max, "ymax": y max,
                                         "spatialReference": <wkt of spatial reference>}

        --------------------     --------------------------------------------------------------------
        moment                   Optional String. This should only be specified by the client when
                                 they do not want to use the current moment
        --------------------     --------------------------------------------------------------------
        return_errors            Optional Boolean. If True, a verbose response will be given if errors
                                 occured.  The default is False.  **Deprecated**
        --------------------     --------------------------------------------------------------------
        record                   Optional String. Represents the record identifier (guid).  If a
                                 record guid is provided, only parcels associated to the record are
                                 built, regardless of the build extent.
        ====================     ====================================================================


        :return: Boolean

        """
        url = "{base}/build".format(base=self._url)
        if moment is None:
            moment = int(time.time())
        params = {
            "gdbVersion" : self._version.properties.versionName,
            "sessionId" : self._version._guid,
            "moment" : moment,
            "buildExtent" : extent,
            "record" : record,
            "async" : False,
            #"returnErrors" : return_errors,
            "f": "json"
        }
        res =  self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def clip(self,
             parent_parcels,
             clip_record=None,
             clipping_parcels=None,
             geometry=None,
             moment=None,
             option=None,
             area_unit=None):
        """

        Clip cuts a new child parcel into existing parent parcels. Commonly
        it retires the parent parcel(s) it cuts into to generate a reminder
        child parcel. This type of split is often part of a `parcel split
        metes and bounds` record driven workflow.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        parent_parcels              parent parcels that will be clipped into.
                                    Syntax:  parentParcels= <parcel (guid)+layer (name)...>
        -----------------------     --------------------------------------------------------------------
        clip_record                 Optional String. It is the GUID for the active legal record.
        -----------------------     --------------------------------------------------------------------
        clipping_parcels            Optional List. A list of child parcels that will be used to clip
                                    into the parent parcels. Parcel lineage is created if the child
                                    'clipping_parcels' and the parcels being clipped are of the same
                                    parcel type.

                                    Syntax: clippingParcels= < id : parcel guid, layered: <layer id>...>

                                    Example:

                                    [{"id":"{D01D3F47-5FE2-4E39-8C07-E356B46DBC78}","layerId":"16"}]

                                    **Either clipping_parcels or geometry is required.**
        -----------------------     --------------------------------------------------------------------
        geometry                    Optional Polygon. Allows for the clipping a parcel based on geometry instead of
                                    'clippingParcels' geometry. No parcel lineage is created.

                                    **Either clipping_parcels or geometry is required.**
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This should only be specified by the client when
                                    they do not want to use the current moment
        -----------------------     --------------------------------------------------------------------
        option                      Optional String. Represents the type of clip to perform:

                                      -  PreserveArea - Preserve the areas that intersect and discard the remainder areas. (default)
                                      -  DiscardArea - Discard the areas that intersect and preserve the remainder areas.
                                      -  PreserveBothAreasSplit - Preserve both the intersecting and remainder areas.
        -----------------------     --------------------------------------------------------------------
        area_unit                   Optional String. Area units to be used when calculating the stated
                                    areas of the clipped parcels. The stated area of the clipped parcels
                                    will be calculated if the stated areas exist on the parent parcels
                                    being clipped.
        =======================     ====================================================================

        :returns: Dictionary


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/clip".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parentParcels": parent_parcels,
            "moment" : moment,
            "record" : clip_record,
            "clippingParcels" : clipping_parcels,
            "clippingGeometry" : geometry,
            "clipOption" : option,
            "defaultAreaUnit" : area_unit,
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def merge(self,
              parent_parcels,
              target_parcel_type,
              attribute_overrides=None,
              child_name=None,
              default_area_unit=None,
              merge_record=None,
              merge_into=None,
              moment=None):
        """
        Merge combines 2 or more parent parcels into onenew child parcel. Merge
        sums up legal areas of parent parcels to the new child parcel legal
        area (using default area units as dictated by client). The child parcel
        lines arecomposed from the outer boundaries of the parent parcels.
        Merge can create multipart parcels as well as proportion lines (partial
        overlap of parent parcels). Record footprint is updated to match the
        child parcel.

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        parent_parcels           Required String. It is the parcel(guid)+layer(name) identifiers to
                                 merge.
        --------------------     --------------------------------------------------------------------
        target_parcel_type       Required String. Layer where parcel is merged to.  History is
                                 created when parents and child are of the same parcel type
        --------------------     --------------------------------------------------------------------
        attribute_overrides      Optional List. A list of attributes to set on the child parcel, if
                                 they exist. Pairs of field name and value.

                                 Syntax: attributeOverrides= [{ "type":"PropertySet","propertySetItems":[<field name>,<field value>]}]

                                 * to set subtype, include subtype value in this list.
        --------------------     --------------------------------------------------------------------
        child_name               Optional String. A descript of the child layer. **DEPRECATED**
        --------------------     --------------------------------------------------------------------
        default_area_unit        Optional String. The area units of the child parcel.
        --------------------     --------------------------------------------------------------------
        merge_record             Optional String. Record identifier (guid).  If missing, no history
                                 is created.
        --------------------     --------------------------------------------------------------------
        merge_into               Optional String. A parcel identifier (guid). Invalid to have a
                                 record id.
        --------------------     --------------------------------------------------------------------
        moment                   Optional String. This parameter represents the session moment (the
                                 default is the version current moment). This should only be
                                 specified by the client when they do not want to use the current
                                 moment.
        --------------------     --------------------------------------------------------------------
        area_unit                Optional Integer. Represents the default area units to be used when
                                 calculating the stated area of the merged parcel. The stated area of
                                 the merged parcel will be calculated if the stated areas exist on
                                 the parcels being merged.
        --------------------     --------------------------------------------------------------------
        attribute_overrides      Optional Dict. Represents a list of attributes to set on the new
                                 merged parcel.

                                 Syntax: attribute_overrides={"type":"PropertySet",
                                 "propertySetItems":["<FieldName>",<value>,
                                                    "<FieldName>",<value>,.....,"IsSeed",0]}

        ====================     ====================================================================


        :return: Dictionary

        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/merge".format(base=self._url)
        params = {
            "gdbVersion" : gdb_version, #
            "sessionId" : session_id, #
            "parentParcels" : parent_parcels, #
            "record" : merge_record, #
            "moment" : moment, #
            "targetParcelType" : target_parcel_type,#
            "mergeInto" : merge_into, #
            #"childName" : child_name,
            "defaultAreaUnit" : default_area_unit,#
            "attributeOverrides" : attribute_overrides,#
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def copy_lines_to_parcel_type(self,
                                  parent_parcels,
                                  record,
                                  target_type,
                                  moment=None,
                                  mark_historic=False,
                                  use_source_attributes=False,
                                  attribute_overrides=None,
                                  use_polygon_attributes=False,
                                  parcel_subtype=None):
        """

        Copy lines to parcel type is used when the construction of the
        child parcel is based on parent parcel geometry. It creates a
        copy of the parent parcels lines that the user can modify (insert,
        delete, update) before they build the child parcels. If the source
        parcel type and the target parcel type are identical (common)
        parcel lineage is created.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        parent_parcels              Required String. Parcel parcels from which lines are copied.
        -----------------------     --------------------------------------------------------------------
        record                      Required String. The unique identifier (guid) of the active legal
                                    record.
        -----------------------     --------------------------------------------------------------------
        target_type                 Required String. The target parcel layer to which the lines will be
                                    copied to.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        mark_historic               Optional Boolean. Mark the parent parcels historic. The default is
                                    False.
        -----------------------     --------------------------------------------------------------------
        use_source_attributes       Optional Boolean. If the source and the target line schema match,
                                    attributes from the parent parcel lines will be copied to the new
                                    child parcel lines when it is set to  True. The default is False.
        -----------------------     --------------------------------------------------------------------
        use_polygon_attributes      Optional Boolean. Parameter representing whether to preserve and
                                    transfer attributes of the parent parcels to the generated seeds.
        -----------------------     --------------------------------------------------------------------
        attribute_overrides         Optional Dictionary. To set fields on the child parcel lines with a
                                    specific value. Uses a key/value pair of FieldName/Value.

                                    Example:

                                    {'type' : "PropertySet", "propertySetItems" : []}
        -----------------------     --------------------------------------------------------------------
        parcel_subtype              Optional Integer. Represents the target parcel subtype.
        =======================     ====================================================================

        :returns: boolean


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/copyLinesToParcelType".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parentParcels": parent_parcels,
            "record" : record,
            "markParentAsHistoric" : mark_historic,
            "useSourceLineAttributes": use_source_attributes,
            "useSourcePolygonAttributes" : use_polygon_attributes,
            "targetParcelType" : target_type,
            "targetParcelSubtype" : parcel_subtype,
            "attributeOverrides": attribute_overrides,
            "moment" : moment,
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def change_type(self,
                    parcels,
                    target_type,
                    parcel_subtype=0,
                    moment=None):
        """

        Changes a set of parcels to a new parcel type. It creates new
        polygons and lines and deletes them from the source type. This
        is used when a parcel was associated in the wrong parcel type subtype
        and/or when creating multiple parcels as part of a build process.
        Example: when lot parcels are created as part of a subdivision, the
        road parcel is moved to the encumbrance (easement) parcel type.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. Parcels list that will change type
        -----------------------     --------------------------------------------------------------------
        target_type                 Required String. The target parcel layer
        -----------------------     --------------------------------------------------------------------
        target_subtype              Optional Integer. Target parcel subtype. The default is 0 meaning
                                    no subtype required.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        =======================     ====================================================================

        :returns: Dictionary


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/changeParcelType".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parcels" : parcels,
            "targetParcelType" : target_type,
            "targetParcelSubtype" : parcel_subtype,
            "moment" : moment,
            "f": "json"
        }
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def delete(self, parcels, moment=None):
        """

        Delete a set of parcels, removing associated or unused lines, and
        connected points.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. The parcels to erase.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        =======================     ====================================================================

        :returns: Boolean


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/deleteParcels".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parcels" : parcels,
            "moment" : moment,
            "f": "json"
        }
        return self._con.post(url, params)['success']
    #----------------------------------------------------------------------
    def update_history(self, features, record,
                       moment=None, set_as_historic=False):
        """
        Sets the specified parcel features to current or historic using the
        specified record. If setting current parcels as historic, the
        Retired By Record field of the features is updated with the Global
        ID of the specified record. If setting historic parcels as current,
        the Created By Record field of the features is updated with the
        Global ID of the specified record.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        features                    Required List. The parcel features to be set as historic or current.
                                    Can be parcels, parcel polygons, parcel points, and parcel lines.

                                    Syntax: ```features=[{"id":"<guid>","layerId":"<layerID>"},{...}]```
        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        set_as_historic             Optional Boolean.  Boolean parameter representing whether to set the
                                    features as historic (true). If false, features will be set as
                                    current.
        =======================     ====================================================================

        :returns: Dictionary

        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/updateParcelHistory".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment" : moment,
            'record' : record,
            'setAsHistoric' : set_as_historic,
            'parcelFeatures' : features,
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def create_seeds(self,
                     record,
                     moment=None,
                     extent=None):
        """

        Create seeds creates parcel seeds for closed loops of lines that
        are associated with the specified record.

        When building parcels from lines, parcel seeds are used. A parcel
        seed is the initial state or seed state of a parcel. A parcel seed
        indicates to the build process that a parcel can be built from the
        lines enclosing the seed.

        A parcel seed is a minimized polygon feature and is stored in the
        parcel type polygon feature class.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        extent                      Optional Dict/arcgis.Geometry.Envelope. The envelope of the extent
                                    in which to create seeds.
        =======================     ====================================================================

        :returns: Dictionary

        """
        from arcgis.geometry import Envelope
        if isinstance(extent, (dict, Envelope)):
            extent = dict(extent)
        elif extent is None:
            pass
        elif not extent is None:
            raise ValueError("Parameter `extent` must be None, Envelope or dict.")
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/createSeeds".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment" : moment,
            'record' : record,
            'extent' : extent,
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def duplicate(self,
                  parcels,
                  parcel_type,
                  record,
                  parcel_subtype=None,
                  moment=None):
        """
        `duplicate` allows for the cloning of parcels from a specific record.

        Parcels can be duplicated in the following ways:

          -  Duplicate to a different parcel type.
          -  Duplicate to a different subtype in the same parcel type.
          -  Duplicate to a different subtype in a different parcel type.

        Similarly, parcel seeds can be duplicated to subtypes and different parcel types.

        =======================     ====================================================================
        **Argument**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. A list of parcels to duplicate.

                                    :Syntax:

                                    ```python
                                    [{"id":"<parcelguid>","layerId":"16"},{...}]
                                    ```

        -----------------------     --------------------------------------------------------------------
        parcel_type                 Required Integer. The target parcel type.
        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        parcel_subtype              Optional Integer. The target parcel subtype.  The default is 0.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        =======================     ====================================================================

        :returns: Dictionary

        """
        if parcel_type is None:
            parcel_subtype = 0
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/duplicateParcels".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment" : moment,
            'record' : record,
            'moment' : moment,
            'parcels' : parcels,
            'targetParcelType' : parcel_type,
            'targetParcelSubtype' : parcel_subtype,
            "f": "json"
        }
        return self._con.post(url, params)
    # ----------------------------------------------------------------------
    def analyze_least_squares_adjustment(self,
                                    analysis_type="CONSISTENCY_CHECK",
                                    convergence_tolerance=0.05,
                                    parcel_features=None,
                                    future=False):
        """
        Note: Least Squares Adjustment functionality introduced at version 10.8.1

        Analyzes the parcel fabric measurement network by running a least squares adjustment on the
        input parcels. A least-squares adjustment is a mathematical procedure that uses statistical
        analysis to estimate the most likely coordinates for connected points in a measurement network.

        Use apply_least_squares_adjustment to apply the results of a least squares adjustment to parcel fabric feature classes.
        ====================    ====================================================================
        **Argument**            **Description**
        --------------------    --------------------------------------------------------------------
        analysis_type           Optional string. Represents the type of least squares analysis that will be run on the input parcels.

                                    CONSISTENCY_CHECK - A free-network least-squares adjustment will be run to check dimensions on
                                    parcel lines for inconsistencies and mistakes. Fixed or weighted control points will not be
                                    used by the adjustment.

                                    WEIGHTED_LEAST_SQUARES - A weighted least-squares adjustment will be run to compute updated
                                    coordinates for parcel points. The parcels being adjusted should connect to at least two fixed
                                    or weighted control points.

                                The default value is CONSISTENCY_CHECK.
        --------------------    --------------------------------------------------------------------
        convergence_tolerance   Optional float. Represents the maximum coordinate shift expected after iterating the least squares adjustment. A least
                                squares adjustment is run repeatedly (in iterations) until the solution converges. The solution is
                                considered converged when maximum coordinate shift encountered becomes less than the specified convergence
                                tolerance.

                                The default value is 0.05 meters or 0.164 feet.
        --------------------    --------------------------------------------------------------------
        parcel_features         Optional list. Represents the input parcels that will be analyzed by a least squares adjustment.

                                    Syntax: parcel_features = [{"id":"<guid>","layerId":"<layerID>"},{...}]

                                If None, the method will analyze the entire parcel fabric.
        --------------------    --------------------------------------------------------------------
        future                  Optional boolean. If true, the request is processed as an asynchronous job and a URL is returned that points a location
                                displaying the status of the job.

                                The default is False.
        ====================    ====================================================================
        :return: Dictionary

        """
        url = "{base}/analyzeByLeastSquaresAdjustment".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "analysisType": analysis_type,
            "convergenceTolerance": convergence_tolerance,
            "parcelFeatures": parcel_features,
            "async": future,
            "f": "json"
            }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(self._status_via_url, con=self._con,
                                url=res["statusUrl"], params={"f": "json"})
            return f
        else:
            res = self._con.post(url, params)
            if 'success' in res:
                return res
            return res
    # ----------------------------------------------------------------------
    def apply_least_squares_adjustment(self,
              movement_tolerance=0.05,
              update_attributes=True,
              future=False):
        """
        Note: Least Squares Adjustment functionality introduced at version 10.8.1

        Applies the results of a least squares adjustment to parcel fabric feature classes. Least squares adjustment results stored
        in the AdjustmentLines and AdjustmentPoints feature classes are applied to the corresponding parcel line, connection line,
        and parcel fabric point feature classes.

        Use analyze_least_squares_adjustment to run a least-squares analysis on parcels and store the results in adjustment feature classes.
        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        movement_tolerance        Optional float. Represents the minimum allowable coordinate shift when updating parcel fabric points. If the distance between
                                  the adjustment point and the parcel fabric point is greater than the specified tolerance, the parcel fabric
                                  point is updated to the location of the adjustment point.

                                  The default tolerance is 0.05 meters or 0.164 feet.
        --------------------     --------------------------------------------------------------------
        update_attributes         Optional boolean. Specifies whether attribute fields in the parcel fabric Points feature class will be updated with
                                  statistical metadata. The XY Uncertainty, Error Ellipse Semi Major, Error Ellipse Semi Minor, and
                                  Error Ellipse Direction fields will be updated with the values stored in the same fields in the AdjustmentPoints
                                  feature class.

                                  The default is True
        --------------------     --------------------------------------------------------------------
        future                   Optional boolean. If true, the request is processed as an asynchronous job and a URL is returned that points a location
                                 displaying the status of the job.

                                 The default is False.
        ====================     ====================================================================

        :return: Dictionary

        """

        url = "{base}/applyLeastSquaresAdjustment".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "movementTolerance": movement_tolerance,
            "updateAttributes": update_attributes,
            "async": future,
            "f": "json"
            }
        if future:
            res = self._con.post(path=url, postdata=params)
            future = self._run_async(self._status_via_url, con=self._con,
                                url=res["statusUrl"], params={"f": "json"})
            return future
        else:
            res = self._con.post(url, params)
            if 'success' in res:
                return res
            return res
    # ----------------------------------------------------------------------
    def _run_async(self, fn, **inputs):
        """runs the inputs asynchronously"""
        import concurrent.futures
        tp = concurrent.futures.ThreadPoolExecutor(1)
        future = tp.submit(fn=fn, **inputs)
        tp.shutdown(False)
        return future
    # ----------------------------------------------------------------------
    def _status_via_url(self, con, url, params):
        """
        performs the asynchronous check to see if the operation finishes
        """
        status_allowed = ['esriJobSubmitted', 'esriJobWaiting', 'esriJobExecuting', 'esriJobSucceeded',
                          'esriJobFailed', 'esriJobTimedOut', 'esriJobCancelling', 'esriJobCancelled']
        status = con.get(url, params)
        while status['status'] in status_allowed and status["status"] != "esriJobSucceeded":
            if status['status'] == 'esriJobSucceeded':
                return status
            elif status['status'] in ['esriJobFailed', 'esriJobTimedOut', 'esriJobCancelled']:
                break
            status = con.get(url, params)
        return status