"""
Classes and objects used to manage published services.
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import json
import tempfile
from .._common import BaseServer
from .parameters import Extension
from arcgis._impl.common._mixins import PropertyMap
########################################################################
class ServiceManager(BaseServer):
    """
    Helper class for managing services. This class is not created by users directly. An instance of this class,
    called 'services', is available as a property of the Server object. Users call methods on this 'services' object to
    managing services.
    """
    _currentURL = None
    _url = None
    _con = None
    _json_dict = None
    _currentFolder = None
    _folderName = None
    _folders = None
    _foldersDetail = None
    _folderDetail = None
    _webEncrypted = None
    _description = None
    _isDefault = None
    _services = None
    _json = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis,
                 initialize=False,
                 sm=None):
        """Constructor

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required string. The administration url endpoint.
        ---------------     --------------------------------------------------------------------
        gis                 Required GIS or Server object. This handles the credential management.
        ===============     ====================================================================
        """
        if sm is not None:
            self._sm = sm
        super(ServiceManager, self).__init__(gis=gis,
                                             url=url, sm=sm)
        self._con = gis
        self._url = url
        self._currentURL = url
        self._currentFolder = '/'
        if initialize:
            self._init(gis)
    #----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f":"json"}
        try:
            result = connection.get(path=self._currentURL,
                                    params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    @property
    def _folder(self):
        """ returns current folder """
        return self._folderName
    #----------------------------------------------------------------------
    @_folder.setter
    def _folder(self, folder):
        """gets/set the current folder"""

        if folder == "" or\
             folder == "/" or \
             folder is None:
            self._currentURL = self._url
            self._services = None
            self._description = None
            self._folderName = None
            self._webEncrypted = None
            self._init()
            self._folderName = folder
        elif folder.lower() in [f.lower() for f in self.folders]:
            self._currentURL = self._url + "/%s" % folder
            self._services = None
            self._description = None
            self._folderName = None
            self._webEncrypted = None
            self._init()
            self._folderName = folder
    #----------------------------------------------------------------------
    @property
    def folders(self):
        """ returns a list of all folders """
        if self._folders is None:
            self._init()
            self._folders = self.properties['folders']
        if "/" not in self._folders:
            self._folders.append("/")
        return self._folders
    #----------------------------------------------------------------------
    def list(self, folder=None, refresh=True):
        """
        returns a list of services in the specified folder

         ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Required string. The name of the folder to list services from.
        ---------------     --------------------------------------------------------------------
        refresh             Optional boolean. Default is False. If True, the list of services will be
                            requested to the server, else the list will be returned from cache.
        ===============     ====================================================================


        :return: list

        """
        if folder is None:
            folder = '/'
        if folder != self._currentFolder or \
           self._services is None or refresh:
            self._currentFolder = folder
            self._folder = folder
            return self._services_list()

        return self._services_list()
    #----------------------------------------------------------------------
    def _export_services(self, folder):
        """
        Export services allows for the backup and storage of non-hosted services.

        =================   ====================================================
        **Argument**        **Description**
        -----------------   ----------------------------------------------------
        folder              required string.  This is the path to the save folder.
                            The ArcGIS Account must have access to the location
                            to write the backup file.
        =================   ====================================================

        :returns: string to the save location.

        """
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        url = self._url + "/exportServices"
        params = {
            "f" : "json",
            "location" : folder,
            "csrfPreventToken" : self._con.token
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'location' in res:
            return res['location']
        return None
    #----------------------------------------------------------------------
    def _import_services(self, file_path):
        """
        Import services allows for the backup and storage of non-hosted services.

        =================   ====================================================
        **Argument**        **Description**
        -----------------   ----------------------------------------------------
        file_path           required string.  File path with extension
                            .agssiteservices.
        =================   ====================================================

        :returns: boolean

        """
        folder = os.path.dirname(file_path)
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        url = self._url + "/importServices"
        params = {
            "f" : "json",
            "csrfPreventToken" : self._con.token
        }
        files = {'location' : file_path}
        res = self._con.post(path=url,
                             files=files,
                             postdata=params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def _services_list(self):
        """ returns the services in the current folder """
        self._services = []
        params = {
            "f" : "json"
        }
        json_dict = self._con.get(path=self._currentURL,
                                  params=params)
        if "services" in json_dict.keys():
            for s in json_dict['services']:
                from urllib.parse import quote, quote_plus, urlparse, urljoin
                u_url = self._currentURL + "/%s.%s" % (s['serviceName'], s['type'])
                parsed = urlparse(u_url)
                u_url = "{scheme}://{netloc}{path}".format(scheme=parsed.scheme,
                                                           netloc=parsed.netloc,
                                                           path=quote(parsed.path))
                self._services.append(
                    Service(url=u_url,
                            gis=self._con)
                )
        return self._services
    #----------------------------------------------------------------------
    @property
    def _extensions(self):
        """
        This resource is a collection of all the custom server object
        extensions that have been uploaded and registered with the server.
        You can register new server object extensions using the register
        extension operation. When updating an existing extension, you need
        to use the update extension operation. If an extension is no longer
        required, you can use the unregister operation to remove the
        extension from the site.
        """
        url = self._url + "/types/extensions"
        params = {'f' : 'json'}
        return self._con.get(path=url,
                             params=params)
    #----------------------------------------------------------------------
    def publish_sd(self,
                   sd_file,
                   folder=None):
        """
        publishes a service definition file to arcgis server

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        sd_file             Required string. File path to the .sd file
        ---------------     --------------------------------------------------------------------
        folder              Optional string. This parameter allows for the override of the
                            folder option set in the SD file.
        ===============     ====================================================================


        :return: boolean

        """
        return self._sm.publish_sd(sd_file, folder)
    #----------------------------------------------------------------------
    def _find_services(self, service_type="*"):
        """
            returns a list of a particular service type on AGS

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        service_type        Required string. Type of service to find.  The allowed types
                             are: ("GPSERVER", "GLOBESERVER", "MAPSERVER",
                             "GEOMETRYSERVER", "IMAGESERVER",
                             "SEARCHSERVER", "GEODATASERVER",
                             "GEOCODESERVER", "*").  The default is *
                             meaning find all service names.
        ===============     ====================================================================


        :return: list of service name as folder/name.type

        """
        allowed_service_types = ("GPSERVER", "GLOBESERVER", "MAPSERVER",
                                 "GEOMETRYSERVER", "IMAGESERVER",
                                 "SEARCHSERVER", "GEODATASERVER",
                                 "GEOCODESERVER", "*")
        lower_types = [l.lower() for l in service_type.split(',')]
        for v in lower_types:
            if v.upper() not in allowed_service_types:
                return {"message" : "%s is not an allowed service type." % v}
        params = {
            "f" : "json"
        }
        type_services = []
        folders = self.folders
        folders.append("")
        baseURL = self._url
        for folder in folders:
            if folder == "":
                url = baseURL
            else:
                url = baseURL + "/%s" % folder
            res = self._con.get(path=url, params=params)
            if res.has_key("services"):
                for service in res['services']:
                    if service['type'].lower() in lower_types:
                        service['URL'] = url + "/%s.%s" % (service['serviceName'],
                                                           service_type)
                        type_services.append(service)
                    del service
            del res
            del folder
        return type_services
    #----------------------------------------------------------------------
    def _examine_folder(self, folder=None):
        """
        A folder is a container for GIS services. ArcGIS Server supports a
        single level hierarchy of folders.
        By grouping services within a folder, you can conveniently set
        permissions on them as a single unit. A folder inherits the
        permissions of the root folder when it is created, but you can
        change those permissions at a later time.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Required string. name of folder to examine.
        ===============     ====================================================================


        :return: dict

        """
        params = {'f': 'json'}
        if folder:
            url = self._url + "/" + folder
        else:
            url = self._url
        return self._con.get(path=url, params=params)
    #----------------------------------------------------------------------
    def _can_create_service(self,
                           service,
                           options=None,
                           folder_name=None,
                           service_type=None):
        """
        Use canCreateService to determine whether a specific service can be
        created on the ArcGIS Server site.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service configuration in JSON format. For more
                            information about the service configuration options, see
                            createService. This is an optional parameter, though either the
                            service_type or service parameter must be used.
        ---------------     --------------------------------------------------------------------
        options             optional dict. This is an optional parameter that provides additional
                            information about the service, such as whether it is a hosted
                            service.
        ---------------     --------------------------------------------------------------------
        folder_name         Optional string. This is an optional parameter to indicate the folder
                            where can_create_service will check for the service.
        ---------------     --------------------------------------------------------------------
        service_type        Optional string. The type of service that can be created. This is an
                            optional parameter, though either the service type or service
                            parameter must be used.
        ===============     ====================================================================


        :return: boolean

        """
        url = self._url + "/canCreateService"
        params = {"f" : "json",
                  'service' : service}
        if options:
            params['options'] = options
        if folder_name:
            params['folderName'] = folder_name
        if service_type:
            params['serviceType'] = service_type
        return self._con.post(path=url,
                              postdata=params)

    #----------------------------------------------------------------------
    def _add_folder_permission(self, principal, is_allowed=True, folder=None):
        """
           Assigns a new permission to a role (principal). The permission
           on a parent resource is automatically inherited by all child
           resources

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of role to assign/disassign accesss.
        ---------------     --------------------------------------------------------------------
        is_allowed          Optional boolean. True means grant access, False means revoke.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of folder to assign permissions to.
        ===============     ====================================================================


        :return: dict

        """
        if folder is not None:
            u_url = self._url + "/%s/%s" % (folder, "/permissions/add")
        else:
            u_url = self._url + "/permissions/add"
        params = {
            "f" : "json",
            "principal" : principal,
            "isAllowed" : is_allowed
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _folder_permissions(self, folder_name):
        """
        Lists principals which have permissions for the folder.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name              Optional string. Name of folder to examine permissions.
        ===============     ====================================================================


        :return: dict

        """
        u_url = self._url + "/%s/permissions" % folder_name
        params = {
            "f" : "json",
        }
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _clean_permissions(self, principal):
        """
        Cleans all permissions that have been assigned to a role
        (principal). This is typically used when a role is deleted.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of role to dis-assign all accesss.
        ===============     ====================================================================


        :return: boolean
        """
        u_url = self._url + "/permissions/clean"
        params = {
            "f" : "json",
            "principal" : principal
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def create_folder(self, folder_name, description=""):
        """
        Creates a unique folder name on AGS

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. Name of the new folder.
        ---------------     --------------------------------------------------------------------
        description         Optional string. Description of what the folder is.
        ===============     ====================================================================

        :return: boolean
        """
        params = {
            "f" : "json",
            "folderName" : folder_name,
            "description" : description
        }
        u_url = self._url + "/createFolder"
        res = self._con.post(path=u_url, postdata=params)
        self._init()
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def delete_folder(self, folder_name):
        """
        Removes a folder on ArcGIS Server

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. Name of the folder.
        ===============     ====================================================================

        :return: boolean
        """
        params = {
            "f" : "json"
        }
        if folder_name in self.folders:
            u_url = self._url + "/%s/deleteFolder" % folder_name
            res = self._con.post(path=u_url, postdata=params)
            self._init()
            if 'status' in res:
                return res['status'] == 'success'
            return res
        else:
            return False
    #----------------------------------------------------------------------
    def _delete_service(self, name, service_type, folder=None):
        """
        Deletes a service from ArcGIS Server

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. Name of the service
        ---------------     --------------------------------------------------------------------
        service_type        Required string. Name of the service type.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of the service on ArcGIS Server.
        ===============     ====================================================================

        :return: boolean

        """
        if folder is None:
            u_url = self._url + "/%s.%s/delete" % (name,
                                                   service_type)
        else:
            u_url = self._url + "/%s/%s.%s/delete" % (folder,
                                                      name,
                                                      service_type)
        params = {
            "f" : "json"
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _service_report(self, folder=None):
        """
        Provides a report on all items in a given folder.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of the service on ArcGIS Server.
        ===============     ====================================================================

        :return: boolean
        """
        items = ["description", "status",
                 "instances", "iteminfo",
                 "properties"]
        if folder is None:
            u_url = self._url + "/report"
        else:
            u_url = self._url + "/%s/report" % folder
        params = {
            "f" : "json",
            "parameters" : items
        }
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    @property
    def _types(self):
        """ returns the allowed services types """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/types"
        return self._con.get(path=u_url,
                             params=params)
    #----------------------------------------------------------------------
    def _federate(self):
        """
        This operation is used when federating ArcGIS Server with Portal
        for ArcGIS. It imports any services that you have previously
        published to your ArcGIS Server site and makes them available as
        items with Portal for ArcGIS. Beginning at 10.3, services are
        imported automatically as part of the federate process.
        If the automatic import of services fails as part of the federation
        process, the following severe-level message will appear in the
        server logs:
           Failed to import GIS services as items within portal.
        If this occurs, you can manually re-run the operation to import
        your services as items in the portal. Before you do this, obtain a
        portal token and then validate ArcGIS Server is federated with
        your portal using the portal website. This is done in
        My Organization > Edit Settings > Servers.
        After you run the Federate operation, specify sharing properties to
        determine which users and groups will be able to access each service.
        """
        params = {'f' : 'json'}
        url = self._url + "/federate"
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _unfederate(self):
        """
        This operation is used when unfederating ArcGIS Server with Portal
        for ArcGIS. It removes any items from the portal that represent
        services running on your federated ArcGIS Server. You typically run
        this operation in preparation for a full unfederate action. For
        example, this can be performed using
               My Organization > Edit Settings > Servers
        in the portal website or the Unregister Server operation in the
        ArcGIS REST API.
        Beginning at 10.3, services are removed automatically as part of
        the unfederate process. If the automatic removal of service items
        fails as part of the unfederate process, you can manually re-run
        the operation to remove the items from the portal.
        """
        params = {'f' : 'json'}
        url = self._url + "/unfederate"
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _unregister_extension(self, extension_filename):
        """
        Unregisters all the extensions from a previously registered server
        object extension (.SOE) file.

        ======================     ====================================================================
        **Argument**               **Description**
        ----------------------     --------------------------------------------------------------------
        extension_filename         Required string. Name of the previously registered .SOE file.
        ======================     ====================================================================

        :return: boolean

        """
        params = {
            "f" : "json",
            "extensionFilename" : extension_filename
        }
        url = self._url + "/types/extensions/unregister"
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _update_extension(self, item_id):
        """
        Updates extensions that have been previously registered with the
        server. All extensions in the new .SOE file must match with
        extensions from a previously registered .SOE file.
        Use this operation to update your implementations or extension
        configuration properties.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. Id of the uploaded .SOE file
        ===============     ====================================================================

        :return: boolean


        """
        params = {'f':'json',
                  'id': item_id}
        url = self._url + "/types/extensions/update"
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _rename_service(self, name, service_type,
                        new_name, folder=None):
        """
        Renames a published AGS Service

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string.  Old service name.
        ---------------     --------------------------------------------------------------------
        service_type        Required string. The type of service.
        ---------------     --------------------------------------------------------------------
        new_name            Required string. The new service name.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of where the service lives, none means
                            root folder.
        ===============     ====================================================================

        :return: boolean

        """
        params = {
            "f" : "json",
            "serviceName" : name,
            "serviceType" : service_type,
            "serviceNewName" : new_name
        }
        if folder is None:
            u_url = self._url + "/renameService"
        else:
            u_url = self._url + "/%s/renameService" % folder
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        self._init()
        return res
    #----------------------------------------------------------------------
    def create_service(self, service):
        """
        Creates a new GIS service in the folder. A service is created by
        submitting a JSON representation of the service to this operation.

        The JSON representation of a service contains the following four
        sections:
         - Service Description Properties-Common properties that are shared
          by all service types. Typically, they identify a specific service.
         - Service Framework Properties-Properties targeted towards the
          framework that hosts the GIS service. They define the life cycle
          and load balancing of the service.
         - Service Type Properties -Properties targeted towards the core
          service type as seen by the server administrator. Since these
          properties are associated with a server object, they vary across
          the service types. The Service Types section in the Help
          describes the supported properties for each service.
         - Extension Properties-Represent the extensions that are enabled
          on the service. The Extension Types section in the Help describes
          the supported out-of-the-box extensions for each service type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service is the properties to create a service.
        ===============     ====================================================================

        :return: dict

        Output:
         dictionary status message
        """
        url = self._url + "/createService"
        params = {
            "f" : "json"
        }
        if isinstance(service, str):
            params['service'] = service
        elif isinstance(service, dict):
            params['service'] = json.dumps(service)
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _stop_services(self, services):
        """
        Stops serveral services on a single server.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        services            Required list.  A list of dictionary objects. Each dictionary object
                            is defined as:
                              folder_name - The name of the folder containing the
                                service, for example, "Planning". If the service
                                resides in the root folder, leave the folder
                                property blank ("folder_name": "").
                              serviceName - The name of the service, for example,
                                "FireHydrants".
                              type - The service type, for example, "MapServer".
                                Example:
                                [{
                                  "folder_name" : "",
                                  "serviceName" : "SampleWorldCities",
                                  "type" : "MapServer"
                                }]
        ===============     ====================================================================

        :return: boolean


        """
        url = self._url + "/stopServices"
        if isinstance(services, dict):
            services = [services]
        elif isinstance(services, (list, tuple)):
            services = list(services)
        else:
            Exception("Invalid input for parameter services")
        params = {
            "f" : "json",
            "services" : {
                "services":services
            }
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _start_services(self, services):
        """
        starts serveral services on a single server

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        services            Required list.  A list of dictionary objects. Each dictionary object
                            is defined as:
                              folder_name - The name of the folder containing the
                                service, for example, "Planning". If the service
                                resides in the root folder, leave the folder
                                property blank ("folder_name": "").
                              serviceName - The name of the service, for example,
                                "FireHydrants".
                              type - The service type, for example, "MapServer".
                                Example:
                                [{
                                  "folder_name" : "",
                                  "serviceName" : "SampleWorldCities",
                                  "type" : "MapServer"
                                }]
        ===============     ====================================================================

        :return: boolean

        """
        url = self._url + "/startServices"
        if isinstance(services, dict):
            services = [services]
        elif isinstance(services, (list, tuple)):
            services = list(services)
        else:
            Exception("Invalid input for parameter services")
        params = {
            "f" : "json",
            "services" : {
                "services":services
            }
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _edit_folder(self, description, web_encrypted=False):
        """
        This operation allows you to change the description of an existing
        folder or change the web encrypted property.
        The web encrypted property indicates if all the services contained
        in the folder are only accessible over a secure channel (SSL). When
        setting this property to true, you also need to enable the virtual
        directory security in the security configuration.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        description         Required string. A description of the folder.
        ---------------     --------------------------------------------------------------------
        web_encrypted       Optional boolean. The boolean to indicate if the services are
                            accessible over SSL only.
        ===============     ====================================================================

        :return: boolean

        """
        url = self._url + "/editFolder"
        params = {
            "f" : "json",
            "webEncrypted" : web_encrypted,
            "description" : "%s" % description
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def exists(self, folder_name, name=None, service_type=None):
        """
        This operation allows you to check whether a folder or a service
        exists. To test if a folder exists, supply only a folder_name. To
        test if a service exists in a root folder, supply both serviceName
        and service_type with folder_name=None. To test if a service exists
        in a folder, supply all three parameters.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. The folder name to check for.
        ---------------     --------------------------------------------------------------------
        name                Optional string. The service name to check for.
        ---------------     --------------------------------------------------------------------
        service_type        Optional string. A service type. Allowed values:
                             GeometryServer | ImageServer | MapServer | GeocodeServer |
                             GeoDataServer | GPServer | GlobeServer | SearchServer
        ===============     ====================================================================

        :return: boolean

        """
        if folder_name and \
           name is None and \
           service_type is None:
            for folder in self.folders:
                if folder.lower() == folder_name.lower():
                    return True
                del folder
            return False
        url = self._url + "/exists"
        params = {
            "f" : "json",
            "folderName" : folder_name,
            "serviceName" : name,
            "type" : service_type
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        elif 'exists' in res:
            return res['exists']
        return res
########################################################################
class Service(BaseServer):
    """
    Represents a GIS administrative service

    **(This should not be created by a user)**

    """
    _ii = None
    _con = None
    _frameworkProperties = None
    _recycleInterval = None
    _instancesPerContainer = None
    _maxWaitTime = None
    _minInstancesPerNode = None
    _maxIdleTime = None
    _maxUsageTime = None
    _allowedUploadFileTypes = None
    _datasets = None
    _properties = None
    _recycleStartTime = None
    _clusterName = None
    _description = None
    _isDefault = None
    _type = None
    _serviceName = None
    _isolationLevel = None
    _capabilities = None
    _loadBalancing = None
    _configuredState = None
    _maxStartupTime = None
    _private = None
    _maxUploadFileSize = None
    _keepAliveInterval = None
    _maxInstancesPerNode = None
    _json = None
    _json_dict = None
    _interceptor = None
    _provider = None
    _portalProperties = None
    _jsonProperties = None
    _url = None
    _extensions = None
    _jm = None
    #----------------------------------------------------------------------
    def __init__(self,
                 url,
                 gis,
                 initialize=False,
                 **kwargs):
        """
        Constructor

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required string. The administration URL.
        ---------------     --------------------------------------------------------------------
        gis                 Required GIS. GIS or Server object
        ---------------     --------------------------------------------------------------------
        initialize          Optional boolean. fills all the properties at object creation is
                            true.
        ===============     ====================================================================


        """
        from arcgis.gis import GIS
        if isinstance(gis, GIS):
            con = gis._con
        else:
            con = gis
        super(Service, self)

        self._service_manager = kwargs.pop('service_manager', None)
        self._url = url
        self._currentURL = url
        self._con = con
        #if url.lower().find('gpserver') > -1:
        #    self.jobs = self._jobs
        if initialize:
            self._init(self._con)
    #----------------------------------------------------------------------
    def _init(self, connection=None):
        """ populates server admin information """
        from .parameters import Extension
        params = {
            "f" : "json"
        }
        if connection:
            json_dict = connection.get(path=self._url,
                                       params=params)
        else:
            json_dict = self._con.get(path=self._currentURL,
                                      params=params)
        self._json = json.dumps(json_dict)
        self._json_dict = json_dict
        attributes = [attr for attr in dir(self)
                      if not attr.startswith('__') and \
                      not attr.startswith('_')]
        self._properties = PropertyMap(json_dict)
        for k, v in json_dict.items():
            if k.lower() == "extensions":
                self._extensions = []
                for ext in v:
                    self._extensions.append(Extension.fromJSON(ext))
                    del ext

            del k
            del v
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def _refresh(self):
        """refreshes the object's values by re-querying the service"""
        self._init()
    #----------------------------------------------------------------------
    def _json_properties(self):
        """returns the jsonProperties"""
        if self._jsonProperties is None:
            self._init()
        return self._jsonProperties
    #----------------------------------------------------------------------
    def change_provider(self, provider):
        """
        Allows for the switching of the service provide and how it is hosted on the ArcGIS Server instance.

        Values:

           + 'ArcObjects' means the service is running under the ArcMap runtime i.e. published from ArcMap
           + 'ArcObjects11': means the service is running under the ArcGIS Pro runtime i.e. published from ArcGIS Pro
           + 'DMaps': means the service is running in the shared instance pool (and thus running under the ArcGIS Pro provider runtime)

        :returns: Boolean

        """
        allowed_providers = ['ArcObjects',  'ArcObjects11', 'DMaps']
        url = self._url + "/changeProvider"
        params = {'f' : 'json',
                  'provider' : provider}
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    @property
    def extensions(self):
        """lists the :class:`extensions <arcgis.gis.server.Extension>` on a service"""
        if self._extensions is None:
            self._init()
        return self._extensions
    #----------------------------------------------------------------------
    def modify_extensions(self,
                          extension_objects=None):
        """
        enables/disables a service extension type based on the name

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        extension_objects      Required list. A list of new extensions.
        ==================     ====================================================================


        :return: boolean

        """
        if extension_objects is None:
            extension_objects = []
        if len(extension_objects) > 0 and \
           isinstance(extension_objects[0], Extension):
            self._extensions = extension_objects
            self._json_dict['extensions'] = [x.value for x in extension_objects]
            res = self.edit(str(self._json_dict))
            self._json = None
            self._init()
            return res
        return False
    #----------------------------------------------------------------------
    def _has_child_permissions_conflict(self, principal, permission):
        """
        You can invoke this operation on the resource (folder or service)
        to determine if this resource has a child resource with opposing
        permissions. This operation is typically invoked before adding a
        new permission to determine if the new addition will overwrite
        existing permissions on the child resources.
        For more information, see the section on the Continuous Inheritance
        Model.
        Since this operation basically checks if the permission to be added
        will cause a conflict with permissions on a child resource, this
        operation takes the same parameters as the Add Permission operation.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of the role for whom the permission is being
                            assigned.
        ---------------     --------------------------------------------------------------------
        permission          Required dict. The permission dict. The format is described below.
                            Format:
                                {
                                "isAllowed": <true|false>,
                                "constraint": ""
                                }
        ===============     ====================================================================


        :return: dict

        """
        params = {
            "f" : "json",
            "principal" : principal,
            "permission" : permission
        }
        url = self._url + "/permissions/hasChildPermissionsConflict"
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def start(self):
        """ starts the specific service """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/start"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def stop(self):
        """ stops the current service """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/stop"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def restart(self):
        """ restarts the current service """
        self.stop()
        self.start()
        return True

    #----------------------------------------------------------------------
    def rename(self, new_name):
        """
        Renames this service to the new name

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        new_name            Required string. New name of the current service.
        ===============     ====================================================================


        :return: boolean

        """
        params = {
            "f": "json",
            "serviceName": self.properties.serviceName,
            "serviceType": self.properties.type,
            "serviceNewName": new_name
        }

        u_url = self._url[:self._url.rfind('/')] + "/renameService"

        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def delete(self):
        """deletes a service from arcgis server"""
        params = {
            "f" : "json",
        }
        u_url = self._url + "/delete"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def status(self):
        """ returns the status of the service """
        params = {
            "f" : "json",
        }
        u_url = self._url + "/status"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    @property
    def statistics(self):
        """ returns the stats for the service """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/statistics"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    @property
    def _permissions(self):
        """ returns the permissions for the service """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/permissions"
        return self._con.get(path=u_url, param_dict=params)
    #----------------------------------------------------------------------
    @property
    def _iteminfo(self):
        """ returns the item information """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/iteminfo"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    def _register_extension(self, item_id):
        """
        Registers a new server object extension file with the server.
        Before you register the file, you need to upload the .SOE file to
        the server using the Upload Data Item operation. The item_id
        returned by the upload operation must be passed to the register
        operation.
        This operation registers all the server object extensions defined
        in the .SOE file.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. unique ID of the item
        ===============     ====================================================================


        :return: dict

        """
        params = {
            "id" : item_id,
            "f" : "json"
        }
        url = self._url + "/types/extensions/register"
        return self._con.post(path=url,
                              postdata=params)

    #----------------------------------------------------------------------
    def _delete_item_info(self):
        """
        Deletes the item information.
        """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/iteminfo/delete"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    def _upload_item_info(self, folder, path):
        """
        Allows for the upload of new itemInfo files such as metadata.xml

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Required string. Folder on ArcGIS Server.
        ---------------     --------------------------------------------------------------------
        path                Required string. Full path of the file to upload.
        ===============     ====================================================================


        :return: dict


        """
        files = {}
        url = self._url + "/iteminfo/upload"
        params = {
            "f" : "json",
            "folder" : folder
        }
        files['file'] = path
        return self._con.post(path=url,
                              postdata=params,
                              files=files)
    #----------------------------------------------------------------------
    def _edit_item_info(self, json_dict):
        """
        Allows for the direct edit of the service's item's information.
        To get the current item information, pull the data by calling
        iteminfo property.  This will return the default template then pass
        this object back into the editItemInfo() as a dictionary.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict.  Item information dictionary.
        ===============     ====================================================================


        :return: dict

        """
        url = self._url + "/iteminfo/edit"
        params = {
            "f" : "json",
            "serviceItemInfo" : json.dumps(json_dict)
        }
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _service_manifest(self, file_type="json"):
        """
        The service manifest resource documents the data and other
        resources that define the service origins and power the service.
        This resource will tell you underlying databases and their location
        along with other supplementary files that make up the service.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        file_type           Required string.  This value can be json or xml.  json return the
                            manifest.json file.  xml returns the manifest.xml file.
        ===============     ====================================================================


        :return: string


        """

        url = self._url + "/iteminfo/manifest/manifest.%s" % file_type
        params = {
        }
        f = self._con.get(path=url,
                          params=params,
                          out_folder=tempfile.gettempdir(),
                          file_name=os.path.basename(url))
        return open(f, 'r').read()
    #----------------------------------------------------------------------
    def _add_permission(self, principal, is_allowed=True):
        """
        Assigns a new permission to a role (principal). The permission
        on a parent resource is automatically inherited by all child resources.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. The role to be assigned.
        ---------------     --------------------------------------------------------------------
        is_allowed          Optional boolean. Access of resource by boolean.
        ===============     ====================================================================


        :return: boolean

        """
        u_url = self._url + "/permissions/add"
        params = {
            "f" : "json",
            "principal" : principal,
            "isAllowed" : is_allowed
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def edit(self, service):
        """
        To edit a service, you need to submit the complete JSON
        representation of the service, which includes the updates to the
        service properties. Editing a service causes the service to be
        restarted with updated properties.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service JSON as a dictionary.
        ===============     ====================================================================


        :return: boolean


        """
        url = self._url + "/edit"
        params = {
            "f" : "json"
        }
        if isinstance(service, str):
            params['service'] = service
        elif isinstance(service, dict):
            params['service'] = json.dumps(service)
        res = self._con.post(path=url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def iteminformation(self):
        """ returns the item information

        :returns: ItemInformationManager

        """
        if self._ii is None:
            u_url = self._url + "/iteminfo"
            self._ii = ItemInformationManager(url=u_url,
                                       con=self._con)
        return self._ii
    #----------------------------------------------------------------------
    @property
    def jobs(self):
        """returns a `JobManager` to manage asynchronous geoprocessing tasks"""
        if self._jm is None:
            url = "%s/jobs" % self._url
            self._jm = JobManager(url=url,
                                  con=self._con)
        return self._jm
    #----------------------------------------------------------------------
    @property
    def _jobs(self):
        """returns a `JobManager` to manage asynchronous geoprocessing tasks"""
        if self._jm is None:
            url = "%s/jobs" % self._url
            self._jm = JobManager(url=url,
                                  con=self._con)
        return self._jm
###########################################################################
class JobManager(BaseServer):
    """
    The `JobManager` provides operations to locate, monitor, and intervene
    in current asynchronous jobs being run by the geoprocessing service.
    """
    _con = None
    _gis = None
    _url = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, con):
        """Constructor"""
        self._url = url
        self._con = con
    #----------------------------------------------------------------------
    def search(self,
               start_time=None,
               end_time=None,
               status=None,
               username=None,
               machine=None):
        """
        This operation allows you to query the current jobs for a
        geoprocessing service, with a range of parameters to find jobs that
        meet specific conditions.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        start_time          Optional Datetime. The start date/time of the geoprocessing job.
        ---------------     --------------------------------------------------------------------
        end_time            Optional Datetime. The end date/time of the geoprocessing job.
        ---------------     --------------------------------------------------------------------
        status              Optional String. The current status of the job. The possible
                            statuses are as follows:

                            - esriJobNew
                            - esriJobSubmitted
                            - esriJobExecuting
                            - esriJobSucceeded
                            - esriJobFailed
                            - esriJobCancelling
                            - esriJobCancelled
                            - esriJobWaiting
        ---------------     --------------------------------------------------------------------
        username            Optional String. The ArcGIS Server user who submitted the job. If
                            the service is anonymous, this parameter will be unavailable.
        ---------------     --------------------------------------------------------------------
        machine             Optional String. The machine running the job.
        ===============     ====================================================================


        :returns: List of `Job`

        """
        url = "{base}/query".format(base=self._url)
        import datetime as _datetime
        if start_time and end_time is None:
            end_time = int(_datetime.datetime.now().timestamp() * 1000)
        params = {
            'f' : 'json',
            'start' : 1,
            'number' : 10,
            'startTime' : "",
            'endTime' : "",
            'userName' : "",
            'machineName' : ""
        }
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        if status:
            params['status'] = status
        if username:
            params['userName'] = username
        if machine:
            params['machineName'] = machine
        results = []
        res = self._con.get(url, params)
        results = [Job(url="%s/%s" % (self._url, key), con=self._con) for key in res['results'].keys()]
        while res['nextStart'] > -1:
            params['start'] = res['nextStart']
            res = self._con.get(url, params)
            results += [Job(url="%s/%s" % (self._url, key), con=self._con) for key in res['results'].keys()]
        return results
    #----------------------------------------------------------------------
    def purge(self):
        """
        The method `purge` cancels all asynchronous jobs for the
        geoprocessing service that currently carry a status of NEW,
        SUBMITTED, or WAITING.

        :returns: Boolean

        """
        url = "{base}/purgeQueue".format(base=self._url)
        params = {'f' : 'json'}
        return self._con.post(url, params)

###########################################################################
class Job(BaseServer):
    """
    A `Job` represents the asynchronous execution of an operation by a
    geoprocessing service.
    """
    _con = None
    _url = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, con):
        """Constructor"""
        self._con = con
        self._url = url
    #----------------------------------------------------------------------
    def cancel(self):
        """
        Cancels the current job from the server

        :returns: Boolean

        """
        url = "{base}/cancel".format(base=self._url)
        params = {'f' : 'json'}
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status']
        return res
    #----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the current job from the server

        :returns: Boolean

        """
        url = "{base}/cancel".format(base=self._url)
        params = {'f' : 'json'}
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status']
        return res
###########################################################################
class ItemInformationManager(BaseServer):
    """
    The item information resource stores metadata about a service.
    Typically, this information is available to clients that want to index
    or harvest information about the service.

    Item information is represented in JSON. The property `properties` allows
    users to access the schema and see the current format of the JSON.


    """
    _url = None
    _properties = None
    _con = None
    def __init__(self, url, con):
        """Constructor"""
        self._url = url
        self._con = con
    #----------------------------------------------------------------------
    def delete(self):
        """Deletes the item information.

        :returns: Boolean

        """
        url = "{base}/delete".format(base=self._url)
        params = {'f': 'json'}
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def upload(self, info_file, folder=None):
        """Uploads a file associated with the item information to the server.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        info_file           Required String. The file to upload to the server.
        ---------------     --------------------------------------------------------------------
        folder              Optional String. The name of the folder on the server to which the
                            file must be uploaded.
        ===============     ====================================================================

        :returns: Dict

        """
        f = {'file' : info_file}
        params = {
            'f' : 'json',

        }
        if folder:
            params['folder'] = folder
        url = "{base}/upload".format(base=self._url)
        res = self._con.post(url,
                             params,
                             files=f)
        return res
    #----------------------------------------------------------------------
    @property
    def manifest(self):
        """
        The service manifest resource documents the data and other resources
        that define the service origins and power the service. This resource
        will tell you underlying databases and their location along with
        other supplementary files that make up the service.


        The JSON representation of the manifest has the following two sections:

        Databases

           + byReference - Indicates whether the service data is referenced
                           from a registered folder or database (true) or
                           if it was copied to the server at the time the
                           service was published (false).
           + onPremiseConnectionString - Path to publisher data location.
           + onServerConnectionString - Path to data location after
                                        publishing completes.


        When both the server machine and the publisher's machine are using
        the same folder or database, byReference is true and the
        onPremiseConnectionString and onServerConnectionString properties
        have the same value.

        When the server machine and the publisher machine are using
        different folders or databases, byReference is true and the
        onPremiseConnectionString and onServerConnectionString properties
        have different values.

        When the data is copied to the server automatically at publish time,
        byReference is false.

        Resources

           + clientName - Machine where ArcGIS for Desktop was used to
                          publish the service.
           + onPremisePath - Path, relative to the 'clientName'
                             machine, where the source resource (.mxd,
                             .3dd, .tbx files, geodatabases, and so on)
                             originated.
           + serverPath - Path to the document after publishing
                          completes.

        :returns: Dict

        """
        url = "{base}/manifest/manifest.json".format(base=self._url)
        params = {'f' : 'json'}

        return self._con.get(url, params)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Gets/Sets the Item Information for a serivce.

        :returns: Dict

        """
        url = "{base}".format(base=self._url)
        params = {'f': 'json'}
        return self._con.get(url, params)
    #----------------------------------------------------------------------
    @properties.setter
    def properties(self, value):
        """
        Gets/Sets the Item Information for a serivce.

        :returns: Dict

        """
        url = "{base}/edit".format(base=self._url)
        params = {'f': 'json'}
        return self._con.post(url, params)
###########################################################################
class ItemInforamtionManager(ItemInformationManager):
    """
    The item information resource stores metadata about a service.
    Typically, this information is available to clients that want to index
    or harvest information about the service.

    Item information is represented in JSON. The property `properties` allows
    users to access the schema and see the current format of the JSON.


    """
    pass













