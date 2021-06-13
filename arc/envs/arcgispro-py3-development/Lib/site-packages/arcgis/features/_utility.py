from arcgis.gis import GIS
from arcgis import env
from arcgis._impl.common._mixins import PropertyMap

########################################################################
class UtilityNetworkManager(object):
    """
    The Utility Network Service exposes analytic capabilities (tracing)
    as well as validation of network topology and management of
    subnetworks (managing sources, updating subnetworks, exporting
    subnetworks, and so on). The Utility Network Service is conceptually
    similar to the Network Analysis Service for transportation networks.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    url                     Required String. The web endpoint to the utility service.
    ---------------------   -------------------------------------------
    version                 Required Version. The `Version` class where the branch version will take place.
    ---------------------   -------------------------------------------
    gis                     Optional GIS. The `GIS` connection object.
    =====================   ===========================================


    """
    _con = None
    _gis = None
    _url = None
    _version = None
    _property = None
    _version_guid = None
    _version_name = None
    #----------------------------------------------------------------------
    def __init__(self, url, version, gis=None):
        """Constructor"""
        if gis is None:
            gis = env.active_gis
        self._gis = gis
        self._con = gis._portal.con
        self._url =  url
        self._version = version
        self._version_guid = version._guid
        self._version_name = version.properties.versionName
    #----------------------------------------------------------------------
    def _init(self):
        """initializer"""
        try:
            res = self._con.get(self._url, {'f':'json'})
            self._property = PropertyMap(res)
        except Exception as e:
            self._property = PropertyMap({})

    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties for the service"""
        if self._property is None:
            self._init()
        return self._property
    #----------------------------------------------------------------------
    def trace(self,
              locations,
              trace_type,
              fields=None,
              moment=None,
              configuration=None,
              result_type=None
              ):
        """
        A trace refers to a pre-configured algorithm that systematically
        travels a network to return results. Generalized traces allow you to
        trace across multiple types of domain networks. For example, running
        a Connected trace from your electric network through to your gas
        network. An assortment of options is provided with trace to support
        various analytic work flows. All traces use the network topology to
        read cached information about network features. This can improve
        performance of complex traces on large networks. Trace results are
        not guaranteed to accurately represent a utility network when dirty
        areas are present. The network topology must be validated to ensure
        it reflects the most recent edits or updates made to the network.


        """
        url = "%s/trace" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "traceType" : trace_type,
            "moment" : moment,
            "traceLocations" : locations,
            "traceConfiguration" : configuration,
            "resultFields" : fields,
            "resultType" : result_type
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def disable_topology(self):
        """
        Disables the network topology for a utility network. When the 
        topology is disabled, feature and association edits do not generate
        dirty areas. Analytics and diagram generation can't be performed if 
        the topology is not present.

        When the topology is disabled, the following happens:

             - All current rows in the topology tables are deleted.
             - No dirty areas are generated from edits.
             - Remaining error features still exist and can be cleaned up without the overhead of dirty areas.
        
        To perform certain network configuration tasks, the network 
        topology must be disabled.

             - This operation must be executed by the portal utility network owner.
             - The topology can be disabled in the default version or in a named version. 
             
        :returns: Dictionary
        """
        url = "%s/disableTopology" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid
        }        
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def enable_topology(self, error_count=10000):
        """
        Enabling the network topology for a utility network is done on the 
        **DEFAULT** version. Enabling is **not** supported in named versions. 
        When the topology is enabled, all feature and association edits 
        generate dirty areas, which are then consumed when the network 
        topology is updated.
        
        
        ====================================     ====================================================================
        **Argument**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        error_count                              Optional Integer. Sets the threshold when the `enable_topology` will 
                                                 stop if the maximum number of errors is met. The default value is 
                                                 10,000.
        ====================================     ====================================================================
        
        :returns: Dictionary
        
        """
        if self._version_name.lower().find("default") == -1:
            raise Exception("Current version is not the `DEFAULT` version.")
        
        params = {'f' : 'json',
                  'maxErrorCount' : error_count}
        url = "%s/enableTopology" % self._url
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def disable_subnetwork_controller(self,
                                      network_source_id,
                                      global_id,
                                      terminal_id):
        """
        A subnetwork controller (or simply, a source or a sink) is the
        origin (or destination) of resource flow for a subpart of the
        network. Examples of subnetwork controllers are circuit breakers in
        electric networks, or town border stations in gas networks.
        Subnetwork controllers correspond to devices that have the
        Subnetwork Controller network capability set. A source is removed
        with `disable_subnetwork_controller`.


        """

        url = "%s/disableSubnetworkController" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            'networkSourceId' : network_source_id,
            'featureGlobalId' : global_id,
            'terminalId' : terminal_id
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def enable_subnetwork_controller(self,
                                     network_source_id,
                                     global_id,
                                     terminal_id,
                                     subnetwork_controller_name,
                                     tier_name,
                                     subnetwork_name=None,
                                     description=None,
                                     notes=None
                                     ):
        """
        A subnetwork controller is the origin (or destination) of resource
        flow for a subpart of the network (e.g., a circuit breaker in
        electric networks, or a town border station in gas networks).
        Controllers correspond to Devices that have the Subnetwork
        Controller network capability set.


        """

        url = "%s/enableSubnetworkController" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            'networkSourceId' : network_source_id,
            'featureGlobalId' : global_id,
            'terminalID' : terminal_id,
            'subnetworkControllerName' : subnetwork_controller_name,
            'subnetworkName' : subnetwork_name,
            'tierName' : tier_name,
            'description' : description,
            'notes' : notes
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def export_subnetwork(self,
                          domain_name,
                          tier_name,
                          subnetwork_name,
                          trace_configuration=None,
                          export_acknowlegement=False,
                          fields=None,
                          result_type=None,
                          moment=None):
        """
        The `export_subnetwork` operation is used to export information
        about a subnetwork into a JSON file. That information can then be
        consumed by outside systems such as outage management and asset
        tracking. The exportSubnetwork operation allows you to delete
        corresponding rows in the Subnetwork Sources table as long as the
        IsDeleted attribute is set to True. This indicates a source feeding
        the subnetwork has been removed.

        """

        url = "%s/exportSubnetwork" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "moment" : moment,
            "domainNetworkName" : domain_name,
            "tierName" : tier_name,
            "subnetworkName" : subnetwork_name,
            "exportAcknowledgement" : export_acknowlegement,
            "traceConfiguration" : trace_configuration,
            "resultFields" : fields,
            "resultType" : result_type
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def query_network_moments(self,
                              moments_to_return="fullValidateTopology",
                              moment=None
                             ):
        """
        The `query_network_moments` operation returns the moments related
        to the network topology and operations against the topology. This
        includes when the topology was initially enabled, when it was last
        validated, when the topology was last disabled (and later enabled),
        and when the definition of the utility network was last modified.
        """
        url = "%s/queryNetworkMoments" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "momentsToReturn" : moments_to_return,
            "moment" : moment
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def query_overrides(self,
                        attribute_ids=None,
                        all_attributes=False,
                        all_connectivity=False):
        """
        Network attributes support the ability to have their values
        overridden without having to edit features and validate the network
        topology (build the index). The utility network also supports the
        ability to place ephemeral connectivity (e.g., jumpers in an
        electrical network) between two devices or junctions without having
        to edit features or connectivity associations and validate the
        network topology (build the index). This operation allows the
        client to query all the overrides associated with the network
        attributes (by network attribute id). In addition, all connectivity
        overrides are returned.
        """
        url = "%s/queryOverrides" % self._url
        params = {
            "f" : "json",
            "attributeIDs" : attribute_ids,
            "allAttributes" : all_attributes,
            "allConnectivity" : all_connectivity
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def synthesize_association_geometries(self,
                                          attachment_associations=False,
                                          connectivity_associations=False,
                                          containment_associations=False,
                                          count=200,
                                          extent=False,
                                          out_sr=None,
                                          moment=None):
        """
        The `synthesize_association_geometries` operation is used to export
        geometries representing associations that are synthesized as line
        segments corresponding to the geometries of the devices at the
        endpoints. All features associated with an association must be in
        the specified extent in order for the geometry to be synthesized.
        If only zero or one of the devices/junctions intersects the extent,
        then no geometry will be synthesized.



        """
        url = "%s/synthesizeAssociationGeometries"
        params = {
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "moment": moment,
            "attachmentAssociations": attachment_associations,
            "connectivityAssociations": connectivity_associations,
            "containmentAssociations": containment_associations,
            "maxGeometryCount":  count,
            "extent": extent,
            "outSR":  out_sr,
            "f" : "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def update_is_connected(self):
        """

        Utility network features have an attribute called IsConnected that
        lets you know if a feature is connected to a source or sink, and
        therefore it could potentially be part of an existing subnetwork.
        The `update_is_connected` operation updates this attribute on
        features in the specified utility network. This operation can only
        be executed on the default version by the portal utility network
        owner.
        """
        url = "%s/updateIsConnected" % self._url
        params = {"f" : "json"}
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def update_subnetwork(self,
                          domain_name,
                          tier_name,
                          subnetwork_name=None,
                          all_subnetwork_tier=False,
                          continue_on_failure=False,
                          trace_configuration=None):
        """
        A subnetwork is updated by calling the `update_subnetwork` operation.
        With this operation, one or all of the subnetworks in a single tier
        can be updated. When a subnetwork is updated, four things can occur;
        the Subnetwork Name attribute is updated for all features in the
        subnetwork, the record representing the subnetwork inside the
        SubnetLine class is refreshed, the Subnetworks table is updated and
        finally diagrams are generated or updated for the subnetwork.

        :returns: Boolean

        """
        url = "%s/updateSubnetwork" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "domainNetworkName" : domain_name,
            "tierName" : tier_name,
            "subnetworkName" : subnetwork_name,
            "allSubnetworksInTier" : all_subnetwork_tier,
            "continueOnFailure" : continue_on_failure,
            "traceConfiguration" : trace_configuration
        }
        return self._con.post(url, params)['success']
    #----------------------------------------------------------------------
    def validate_topology(self,
                          envelope,
                          run_async=False,
                          return_edits=False):
        """
        Validating the network topology for a utility network maintains
        consistency between feature editing space and network topology space.
        Validating a network topology may include all or a subset of the
        dirty areas present in the network. Validation of network topology
        is supported synchronously and asynchronously.
        
        :returns: Dictionary
        
        """
        url = "%s/validateNetworkTopology" % self._url
        params = {
            "f" : "json",
            "gdbVersion" : self._version_name,
            "sessionId" : self._version_guid,
            "validateArea" : envelope,
            "async" : run_async,
            'returnEdits' : return_edits
            
        }
        if run_async == False:
            return self._con.post(url, params)['success']
        else:
            return self._con.post(url, params)
    #----------------------------------------------------------------------
    def apply_overrides(self, adds=None, deletes=None):
        """
        Network attributes support the ability to have their values
        overridden without having to edit features and validate the network
        topology (build the index). The utility network also supports the
        ability to place ephemeral connectivity (for example, jumpers in an
        electrical network) between two devices or junctions without having
        to edit features or connectivity associations and validate the
        network topology (build the index). When specified by the client, a
        trace operation may optionally incorporate the network attribute
        and connectivity override values when the trace is run on.


        """
        url = "%s/applyOverrides"
        params = {'f' : 'json',
                  'adds' : adds,
                  'deletes' : deletes}
        return self._con.post(url, params)
