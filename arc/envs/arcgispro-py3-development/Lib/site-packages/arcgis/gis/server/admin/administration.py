"""
   Adminstration.py allows users to control ArcGIS for Server 10.1+
   through the Administration REST API

"""
from __future__ import absolute_import
from .._common import BaseServer
from . import _machines, _clusters
from . import _data, _info
from . import _kml, _logs
from . import _security, _services
from . import _system
from . import _uploads, _usagereports
from . import _mode
from .. import  ServicesDirectory
from ..._impl._con import Connection
########################################################################
class Server(BaseServer):
    """
    An ArcGIS Enterprise Server site used for hosting GIS Web services. This class is not
    created by users directly to access server instances in an Enterprise configuration. Use the
    :class:`ServerManager` :func:`~ServerManager.list` or :func:`~ServerManager.get` methods.

    .. code-block:: python

        # Usage Example 1: Get a GIS server federated with Enterprise

        gis = GIS(profile="your_ent_admin_profile")

        hosting_server = gis.servers.get(role="HOSTING_SERVER")


    For stand alone servers, directly create a :class:`Server` instance.

    .. code-block:: python

        # Usage Example 2: Get a stand alone server that has Web Adaptor installed

        server_base_url = "https://example.site.com"

        gis_server = Server(url=f"{server_base_url}/web_adaptor/admin",
                            token_url=f"{server_base_url}/web_adaptor/tokens/generateToken",
                            username="admin_user",
                            password="admin_password")


    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string. The URL to the ArcGIS Server administration
                           end point for the ArcGIS Server Site.

                           Example: https://gis.mysite.com/arcgis/admin

                           The URL should be formatted as follows:
                           <scheme>://<fully_qualified_domain_name>:<port (optional)>/<web adaptor>/admin

                           Note: Using the fully-qualified domain name to the server, also known as the
                           Web Context URL, is recommended as generally the SSL Certificate binding for
                           the web server uses this hostname.
    ------------------     --------------------------------------------------------------------
    gis                    Optional string. The GIS object representing the Portal which thi
                           Server is federated with. The GIS object should be logged in with a username
                           in the publisher or administrator Role in order to administer the Server
    ==================     ====================================================================

    =====================     ====================================================================
    **Optional Argument**     **Description**
    ---------------------     --------------------------------------------------------------------
    baseurl                   Optional string. The root URL to a site.
                              Example: https://mysite.com/arcgis
    ---------------------     --------------------------------------------------------------------
    tokenurl                  Optional string. Used when a site is federated or when the token
                              URL differs from the site's baseurl.  If a site is federated, the
                              token URL will return as the Portal token and ArcGIS Server users
                              will not validate correctly.
    ---------------------     --------------------------------------------------------------------
    username                  Optional string. The login username for BUILT-IN GIS Server security.
    ---------------------     --------------------------------------------------------------------
    password                  Optional string. A secret word or phrase that must be used to gain
                              access to the account above.
    ---------------------     --------------------------------------------------------------------
    key_file                  Optional string. The path to a PKI key file used to authenticate the
                              user to the Web Server in front of the ArcGIS Server site.
    ---------------------     --------------------------------------------------------------------
    cert_file                 Optional string. The path to PKI cert file used to authenticate the
                              user to the Web Server in front of the ArcGIS Server site.
    ---------------------     --------------------------------------------------------------------
    proxy_host                Optional string. The web address to the proxy host if the environment
                              where the Python API is running requires a proxy host for access to the
                              Site URL or GIS URL.

                              Example: proxy.mysite.com
    ---------------------     --------------------------------------------------------------------
    proxy_port                Optional integer. The port which the proxy is accessed through,
                              default is 80.
    ---------------------     --------------------------------------------------------------------
    expiration                Optional integer. This is the length of time in minutes that a token
                              requested through this login process will be valid for.
                              Example: 1440 is one day. The Default is 60.
    ---------------------     --------------------------------------------------------------------
    all_ssl                   Optional boolean. If True, all calls will be made over HTTPS instead
                              of HTTP. The default is False.
    ---------------------     --------------------------------------------------------------------
    portal_connection         Optional string. This is used when a site is federated. It is the
                              ArcGIS Online or Portal GIS object used.
    ---------------------     --------------------------------------------------------------------
    initialize                Optional boolean. If True, the object will attempt to reach out to
                              the URL resource and populate at creation time. The default is False.
    =====================     ====================================================================

    """

    _url = None
    _con = None
    _json_dict = None
    _json = None
    _catalog = None
    _sitemanager = None
    #----------------------------------------------------------------------
    def __init__(self,
                 url,
                 gis=None,
                 **kwargs):
        """Constructor"""
        if gis is None and len(kwargs) > 0:
            if 'baseurl' not in kwargs:
                kwargs['baseurl'] = url
            gis = Connection(**kwargs)
        initialize = kwargs.pop('initialize', False)
        super(Server, self).__init__(gis=gis,
                                     url=url,
                                     initialize=initialize,
                                     **kwargs)
        if url.endswith("/"):
            url = url[:-1]
        self._catalog = kwargs.pop('servicesdirectory', None)
        if not url.lower().endswith('/admin'):
            url = "%s/admin" % url
        self._url = url

        #else:
        #    raise ValueError("You must provide either a GIS or login credentials to use this object.")
        if hasattr(gis, '_con'):
            self._con = gis._con
        elif hasattr(gis, '_portal'):
            self._con = gis._portal._con
        elif isinstance(gis, Connection):
            self._con = gis
        else:
            raise ValueError("Invalid gis Type: Must be GIS/ServicesDirectory Object")
        if initialize:
            self._init(self._con)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def publish_sd(self,
                    sd_file,
                    folder=None):
        """
        Publishes a service definition file to ArcGIS Server.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        sd_file                Required string. The service definition file to be uploaded and published.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. The folder in which to publish the service definition
                               file to.  If this folder is not present, it will be created.  The
                               default is None in which case the service definition will be published
                               to the System folder.
        ==================     ====================================================================

        :return:
           A boolean indicating success (True) or failure (False).
        """
        import json
        if sd_file.lower().endswith('.sd') == False:
            return False
        catalog = self.content
        if 'System' not in self.services.folders:
            return False
        if folder and \
           folder.lower() not in [f.lower() for f in self.services.folders]:
            self.services.create_folder(folder)
        service = catalog.get(name="PublishingTools", folder='System')
        if service is None:
            service = catalog.get(name="PublishingToolsEx", folder='System')
        if service is None:
            return False
        status, res = self._uploads.upload(path=sd_file, description="sd file")
        if status:
            uid = res['item']['itemID']
            if folder:
                config = self._uploads._service_configuration(uid)
                if 'folderName' in config:
                    config['folderName'] = folder
                res = service.publish_service_definition(in_sdp_id=uid,
                                                         in_config_overwrite=json.dumps(config))
            else:
                res = service.publish_service_definition(in_sdp_id=uid)
            return True
        return False
    #----------------------------------------------------------------------
    @staticmethod
    def _create(url,
                username,
                password,
                config_store_connection,
                directories,
                cluster=None,
                logs_settings=None,
                run_async=False,
                **kwargs):
        """
        This is the first operation that you must invoke when you install
        ArcGIS Server for the first time. Creating a new site involves:

          -Allocating a store to save the site configuration
          -Configuring the server machine and registering it with the site
          -Creating a new cluster configuration that includes the server
           machine
          -Configuring server directories
          -Deploying the services that are marked to auto-deploy

        Because of the sheer number of tasks, it usually takes some time
        for this operation to complete. Once a site has been created,
        you can publish GIS services and deploy them to your server
        machines.

        ======================     ====================================================================
        **Argument**               **Description**
        ----------------------     --------------------------------------------------------------------
        connection
        ----------------------     --------------------------------------------------------------------
        url                        Required string. URI string to the site.
        ----------------------     --------------------------------------------------------------------
        username                   Required string. The name of the administrative account to be used by
                                   the site. This can be changed at a later stage.
        ----------------------     --------------------------------------------------------------------
        password                   Required string. The password to the administrative account.
        ----------------------     --------------------------------------------------------------------
        configStoreConnection      Required string. A JSON object representing the connection to the
                                   configuration store. By default, the configuration store will be
                                   maintained in the ArcGIS Server installation directory.
        ----------------------     --------------------------------------------------------------------
        directories                Required string. A JSON object representing a collection of server
                                   directories to create. By default, the server directories will be
                                   created locally.
        ----------------------     --------------------------------------------------------------------
        cluster                    Optional string. An optional cluster configuration. By default, the
                                   site will create a cluster called 'default' with the first available
                                   port numbers starting from 4004.
        ----------------------     --------------------------------------------------------------------
        logsSettings               Optional string. Optional log settings, see http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Log_Settings/02r3000001t6000000/ .
        ----------------------     --------------------------------------------------------------------
        runAsync                   Optional boolean. A flag to indicate if the operation needs to be run
                                   asynchronously.
        ======================     ====================================================================


        =====================     ====================================================================
        **Optional Argument**     **Description**
        ---------------------     --------------------------------------------------------------------
        baseurl                   Optional string. The root URL to a site.
                                  Example: https://mysite.com/arcgis
        ---------------------     --------------------------------------------------------------------
        tokenurl                  Optional string. Used when a site is federated or when the token
                                  URL differs from the site's baseurl.  If a site is federated, the
                                  token URL will return as the Portal token and ArcGIS Server users
                                  will not validate correctly.
        ---------------------     --------------------------------------------------------------------
        username                  Optional string. The login username for BUILT-IN security.
        ---------------------     --------------------------------------------------------------------
        password                  Optional string. A secret word or phrase that must be used to gain
                                  access to the account above.
        ---------------------     --------------------------------------------------------------------
        key_file                  Optional string. The path to PKI key file.
        ---------------------     --------------------------------------------------------------------
        cert_file                 Optional string. The path to PKI cert file.
        ---------------------     --------------------------------------------------------------------
        proxy_host                Optional string. The web address to the proxy host.

                                  Example: proxy.mysite.com
        ---------------------     --------------------------------------------------------------------
        proxy_port                Optional integer. The port where the proxy resides on, default is 80.
        ---------------------     --------------------------------------------------------------------
        expiration                Optional integer. This is the length of time a token is valid for.
                                  Example 1440 is one week. The Default is 60.
        ---------------------     --------------------------------------------------------------------
        all_ssl                   Optional boolean. If True, all calls will be made over HTTPS instead
                                  of HTTP. The default is False.
        ---------------------     --------------------------------------------------------------------
        portal_connection         Optional string. This is used when a site is federated. It is the
                                  ArcGIS Online or Portal GIS object used.
        ---------------------     --------------------------------------------------------------------
        initialize                Optional boolean. If True, the object will attempt to reach out to
                                  the URL resource and populate at creation time. The default is False.
        =====================     ====================================================================

        :return:
           Success statement.
        """
        url = url + "/createNewSite"
        params = {
            "f" : "json",
            "cluster" : cluster,
            "directories" : directories,
            "username" : username,
            "password" : password,
            "configStoreConnection" : config_store_connection,
            "logSettings" : logs_settings,
            "runAsync" : run_async
        }
        con = Connection(**kwargs)
        return con.post(path=url,
                        postdata=params)
    #----------------------------------------------------------------------
    def _join(self, admin_url, username, password):
        """
        The Join Site operation is used to connect a server machine to an
        existing site. This is considered a 'push' mechanism, in which a
        server machine pushes its configuration to the site. For the
        operation to be successful, you need to provide an account with
        administrative privileges to the site.
        When an attempt is made to join a site, the site validates the
        administrative credentials, then returns connection information
        about its configuration store back to the server machine. The
        server machine then uses the connection information to work with
        the configuration store.
        If this is the first server machine in your site, use the Create
        Site operation instead.


        ======================     ====================================================================
        **Argument**               **Description**
        ----------------------     --------------------------------------------------------------------
        admin_url                  Required string. The site URL of the currently live site. This is
                                   typically the Administrator Directory URL of one of the server
                                   machines of a site.
        ----------------------     --------------------------------------------------------------------
        username                   Required string. The name of the administrative account for this site.
        ----------------------     --------------------------------------------------------------------
        password                   Required string. The password to the administrative account.
        ======================     ====================================================================


        :return:
           Success statement.
        """
        url = self._url + "/joinSite"
        params = {
            "f" : "json",
            "adminURL" : admin_url,
            "username" : username,
            "password" : password
        }
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _delete(self):
        """
        Deletes the site configuration and releases all server resources.
        This is an unrecoverable operation. This operation is well suited
        for development or test servers that need to be cleaned up
        regularly. It can also be performed prior to uninstall. Use caution
        with this option because it deletes all services, settings, and
        other configurations.
        This operation performs the following tasks:
          - Stops all server machines participating in the site. This in
            turn stops all GIS services hosted on the server machines.
          - All services and cluster configurations are deleted.
          - All server machines are unregistered from the site.
          - All server machines are unregistered from the site.
          - The configuration store is deleted.

        :return:
           Success statement.
        """
        url = self._url + "/deleteSite"
        params = {
            "f" : "json"
        }
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _export(self, location=None):
        """
        Exports the site configuration to a location you specify as input
        to this operation.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Optional string. A path to a folder accessible to the server where
                               the exported site configuration will be written. If a location is
                               not specified, the server writes the exported site configuration
                               file to directory owned by the server and returns a virtual path
                               (an HTTP URL) to that location from where it can be downloaded.
        ==================     ====================================================================


        :return:
           Success statement.
        """
        url = self._url + "/exportSite"
        params = {
            "f" : "json"
        }
        if location is not None:
            params['location'] = location
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _import_site(self, location):
        """
        This operation imports a site configuration into the currently
        running site. Importing a site means replacing all site
        configurations (including GIS services, security configurations,
        and so on) of the currently running site with those contained in
        the site configuration file you supply as input. The input site
        configuration file can be obtained through the exportSite
        operation.
        This operation will restore all information included in the backup,
        as noted in exportSite. When it is complete, this operation returns
        a report as the response. You should review this report and fix any
        problems it lists to ensure your site is fully functioning again.
        The importSite operation lets you restore your site from a backup
        that you created using the exportSite operation.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Required string. A file path to an exported configuration or an ID
                               referencing the stored configuration on the server.
        ==================     ====================================================================


        :return:
           A report.
        """
        url = self._url + "/importSite"
        params = {
            "f" : "json",
            "location" : location
        }
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _upgrade(self, run_async=False):
        """
        This is the first operation that must be invoked during an ArcGIS
        Server upgrade. Once the new software version has been installed
        and the setup has completed, this operation will be available. A
        successful run of this operation will complete the upgrade of
        ArcGIS Server.

        .. note::
            **caution** If errors are returned with the upgrade operation,
            you must address the errors before you may continue. For example,
            if you encounter an error about an invalid license, you will need
            to re-authorize the software using a valid license and you may
            then retry this  operation.

            This operation is available only when a server machine is
            currently being upgraded. It will not be available after a
            successful upgrade of a server machine.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        run_async              Optional boolean. A flag to indicate if the operation needs to be run
                               asynchronously. The default value is False.
        ==================     ====================================================================


        :return:
           Success statement.
        """
        url = self._url + "/upgrade"
        params = {
            "f" : "json",
            "runAsync" : run_async
        }
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    @property
    def _public_key(self):
        """Gets the public key."""
        url = self._url + "/publicKey"
        params = {
            "f" : "json",
        }
        return self._con.get(path=url,
                             params=params)
    #----------------------------------------------------------------------
    @property
    def machines(self):
        """
        Gets the list of server machines registered with the site.
        This resource represents a collection of all the server machines that
        have been registered with the site. It other words, it represents
        the total computing power of your site. A site will continue to run
        as long as there is one server machine online.
        For a server machine to start hosting GIS services, it must be
        grouped (or clustered). When you create a new site, a cluster called
        'default' is created for you.
        The list of server machines in your site can be dynamic. You can
        register additional server machines when you need to increase the
        computing power of your site or unregister them if you no longer
        need them.
        """
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           'machines' in self.resources:
            url = self._url + "/machines"
            return _machines.MachineManager(url,
                                            gis=self._con,
                                            initialize=False)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def datastores(self):
        """
        Gets the information about the data holdings of the server.
        Data items are used by ArcGIS for Desktop and other clients
        to validate data paths referenced by GIS services.
        You can register new data items with the server by using the
        Register Data Item operation. Use the Find Data Items operation to
        search through the hierarchy of data items.
        A relational data store type represents a database platform that
        has been registered for use on a portal's hosting server by the
        ArcGIS Server administrator. Each relational data store type
        describes the properties ArcGIS Server requires in order to connect
        to an instance of a database for a particular platform. At least
        one registered relational data store type is required before client
        applications such as Insights for ArcGIS can create Relational
        Database Connection portal items.
        The Compute Ref Count operation counts and lists all references to
        a specific data item. This operation helps you determine if a
        particular data item can be safely deleted or refreshed.
        """
        if self.properties is None:
            self._init()
        if isinstance(self.resources, list) and \
           "data" in self.resources:
            url = self._url + "/data"
            return _data.DataStoreManager(url=url,
                                          gis=self._con)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def _info(self):
        """
        A read-only resource that returns meta information about the server.
        """
        url = self._url + "/info"
        return _info.Info(url=url,
                          gis=self._con,
                          initialize=True)
    #----------------------------------------------------------------------
    @property
    def site(self):
        """
        Gets the site's collection of server resources. This collection
        includes server machines that are installed with ArcGIS Server,
        including GIS services, data and so on. The site resource also
        lists the current version of the software.
        When you install ArcGIS Server on a server machine for the first
        time, you must create a new site. Subsequently, newer server
        machines can join your site and increase its computing power. Once
        a site is no longer required, you can delete the site, which will
        cause all of the resources to be cleaned up.
        """
        if self._sitemanager is None:
            self._sitemanager = SiteManager(self)
        return self._sitemanager
    #----------------------------------------------------------------------
    @property
    def _clusters(self):
        """Gets the clusters functions if supported in resources."""
        if self.properties is None:
            self._init()
        if isinstance(self.resources, list) and \
           "clusters" in self.resources:
            url = self._url + "/clusters"
            return _clusters.Cluster(url=url,
                                     gis=self._con,
                                     initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def services(self):
        """
        Gives the administrator access to the services on ArcGIS Server as a
        ServerManager Object.
        """
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           'services' in self.resources:
            url = self._url + "/services"
            return _services.ServiceManager(url=url,
                                            gis=self._con,
                                            initialize=True,
                                            sm=self)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def usage(self):
        """
        Gets the collection of all the usage reports created
        within your site. The Create Usage Report operation lets you define
        a new usage report.
        """
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           'usagereports' in self.resources:
            url = self._url + "/usagereports"
            return _usagereports.ReportManager(url=url,
                                              gis=self._con,
                                              initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def _kml(self):
        """Gets the KML functions for a server."""
        url = self._url + "/kml"
        return _kml.KML(url=url,
                        gis=self._con,
                        initialize=True)
    #----------------------------------------------------------------------
    @property
    def logs(self):
        """
        Gives users access the ArcGIS Server's logs and lets
        administrators query and find errors and/or problems related to
        the server or a service.

        Logs are the records written by the various components of ArcGIS
        Server. You can query the logs and change various log settings.
        **Note**
        ArcGIS Server Only
        """
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           'logs' in self.resources:
            url = self._url + "/logs"
            return _logs.LogManager(url=url,
                                    gis=self._con,
                                    initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def _security(self):
        """Gets an object to work with the site security."""
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           "security" in self.resources:
            url = self._url + "/security"
            return _security.Security(url=url,
                                      gis=self._con,
                                      initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def users(self):
        """Gets operations to work with users."""
        return self._security.users
    #----------------------------------------------------------------------
    @property
    def content(self):
        """
        Gets the Services Directory which can help you discover information about
        services available on a particular server. A service represents a
        local GIS resource whose functionality has been made available on
        the server to a wider audience. For example, an ArcGIS Server
        administrator can publish an ArcMap document (.mxd) as a map
        service. Developers and clients can display the map service and
        query its contents.

        The Services Directory is available as part of the REST services
        infrastructure available with ArcGIS Server installations. It
        enables you to list the services available, including secured
        services when you provide a proper login. For each service, a set
        of general properties are displayed. For map services, these
        properties include the spatial extent, spatial reference
        (coordinate system) and supported operations. Layers are also
        listed, with links to details about layers, which includes layer
        fields and extent. The Services Directory can execute simple
        queries on layers.

        The Services Directory is also useful for finding information about
        non-map service types. For example, you can use the Services
        Directory to determine the required address format for a geocode
        service, or the necessary model inputs for a geoprocessing service.
        """
        from .. import ServicesDirectory
        if self._catalog is None:
            url = self._url.lower().replace("/admin", "")
            self._catalog = ServicesDirectory(url=url,
                                              con=self._con)
        return self._catalog
    #----------------------------------------------------------------------
    @property
    def system(self):
        """
        Provides access to common system configuration settings.
        """
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           "system" in self.resources:
            url = self._url + "/system"
            return _system.SystemManager(url=url,
                                         gis=self._con,
                                         initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def _uploads(self):
        """Gets an object to work with the site uploads."""
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           "uploads" in self.resources:
            url = self._url + "/uploads"
            return _uploads.Uploads(url=url,
                                    gis=self._con,
                                    initialize=True)
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def _mode(self):
        """Gets the class that works with Mode."""
        if self.resources is None:
            self._init()
        if isinstance(self.resources, list) and \
           'mode' in self.resources:
            url = self._url + "/mode"
            return _mode.Mode(url=url,
                              gis=self._con,
                              initialize=True)
        return None


########################################################################
class SiteManager(object):
    """
    A site is a collection of server resources. This collection includes
    server machines that are installed with ArcGIS Server, including GIS
    services, data and so on. The site resource also lists the current
    version of the software.
    When you install ArcGIS Server on a server machine for the first time,
    you must create a new site. Subsequently, newer server machines can
    join your site and increase its computing power. Once a site is no
    longer required, you can delete the site, which will cause all of
    the resources to be cleaned up.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    server                 Required string. The arcgis.gis.server object.
    ==================     ====================================================================


    """
    _sm = None
    #----------------------------------------------------------------------
    def __init__(self, server, initialize=False):
        """Constructor"""
        self._sm = server
        isinstance(self._sm, SiteManager)

        if initialize:
            self._sm._init()
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._sm._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._sm._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """Gets the site properties. """
        return self._sm.properties
    #----------------------------------------------------------------------
    @staticmethod
    def create(username,
               password,
               config_store_connection,
               directories,
               cluster=None,
               logs_settings=None,
               run_async=False,
               **kwargs):
        """
        This is the first operation that you must invoke when you install
        ArcGIS Server for the first time. Creating a new site involves:

          -Allocating a store to save the site configuration
          -Configuring the server machine and registering it with the site
          -Creating a new cluster configuration that includes the server
           machine
          -Configuring server directories
          -Deploying the services that are marked to auto-deploy

        Because of the sheer number of tasks, it usually takes some time
        for this operation to complete. Once a site has been created,
        you can publish GIS services and deploy them to your server
        machines.

        ======================     ====================================================================
        **Argument**               **Description**
        ----------------------     --------------------------------------------------------------------
        connection
        ----------------------     --------------------------------------------------------------------
        url                        Required string. URI string to the site.
        ----------------------     --------------------------------------------------------------------
        username                   Required string. The name of the administrative account to be used by
                                   the site. This can be changed at a later stage.
        ----------------------     --------------------------------------------------------------------
        password                   Required string. The password to the administrative account.
        ----------------------     --------------------------------------------------------------------
        configStoreConnection      Required string. A JSON object representing the connection to the
                                   configuration store. By default, the configuration store will be
                                   maintained in the ArcGIS Server installation directory.
        ----------------------     --------------------------------------------------------------------
        directories                Required string. A JSON object representing a collection of server
                                   directories to create. By default, the server directories will be
                                   created locally.
        ----------------------     --------------------------------------------------------------------
        cluster                    Optional string. An optional cluster configuration. By default, the
                                   site will create a cluster called 'default' with the first available
                                   port numbers starting from 4004.
        ----------------------     --------------------------------------------------------------------
        logsSettings               Optional string. Optional log settings, see http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Log_Settings/02r3000001t6000000/ .
        ----------------------     --------------------------------------------------------------------
        runAsync                   Optional boolean. A flag to indicate if the operation needs to be run
                                   asynchronously.
        ======================     ====================================================================


        =====================     ====================================================================
        **Optional Argument**     **Description**
        ---------------------     --------------------------------------------------------------------
        baseurl                   Optional string. The root URL to a site.
                                  Example: https://mysite.com/arcgis
        ---------------------     --------------------------------------------------------------------
        tokenurl                  Optional string. Used when a site is federated or when the token
                                  URL differs from the site's baseurl.  If a site is federated, the
                                  token URL will return as the Portal token and ArcGIS Server users
                                  will not validate correctly.
        ---------------------     --------------------------------------------------------------------
        username                  Optional string. The login username for BUILT-IN security.
        ---------------------     --------------------------------------------------------------------
        password                  Optional string. A secret word or phrase that must be used to gain
                                  access to the account above.
        ---------------------     --------------------------------------------------------------------
        key_file                  Optional string. The path to PKI key file.
        ---------------------     --------------------------------------------------------------------
        cert_file                 Optional string. The path to PKI cert file.
        ---------------------     --------------------------------------------------------------------
        proxy_host                Optional string. The web address to the proxy host.

                                  Example: proxy.mysite.com
        ---------------------     --------------------------------------------------------------------
        proxy_port                Optional integer. The port where the proxy resides on, default is 80.
        ---------------------     --------------------------------------------------------------------
        expiration                Optional integer. This is the length of time a token is valid for.
                                  Example 1440 is one week. The Default is 60.
        ---------------------     --------------------------------------------------------------------
        all_ssl                   Optional boolean. If True, all calls will be made over HTTPS instead
                                  of HTTP. The default is False.
        ---------------------     --------------------------------------------------------------------
        portal_connection         Optional string. This is used when a site is federated. It is the
                                  ArcGIS Online or Portal GIS object used.
        ---------------------     --------------------------------------------------------------------
        initialize                Optional boolean. If True, the object will attempt to reach out to
                                  the URL resource and populate at creation time. The default is False.
        =====================     ====================================================================

        :return:
           Success statement.

        """
        return Server._create(username,
                                password,
                                config_store_connection,
                                directories,
                                cluster,
                                logs_settings,
                                run_async)
    #----------------------------------------------------------------------
    def join(self, admin_url, username, password):
        """
        The Join Site operation is used to connect a server machine to an
        existing site. This is considered a 'push' mechanism, in which a
        server machine pushes its configuration to the site. For the
        operation to be successful, you need to provide an account with
        administrative privileges to the site.
        When an attempt is made to join a site, the site validates the
        administrative credentials, then returns connection information
        about its configuration store back to the server machine. The
        server machine then uses the connection information to work with
        the configuration store.
        If this is the first server machine in your site, use the Create
        Site operation instead.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        admin_url              Required string. The site URL of the currently live site. This is
                               typically the Administrator Directory URL of one of the server
                               machines of a site.
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of an administrative account for the site.
        ------------------     --------------------------------------------------------------------
        password               Required string. The password of the administrative account.
        ==================     ====================================================================


        :return:
           A status indicating success or failure.

        """
        return self._sm._join(admin_url, username, password)
    #----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the site configuration and releases all server resources.
        This operation is well suited
        for development or test servers that need to be cleaned up
        regularly. It can also be performed prior to uninstall. Use caution
        with this option because it deletes all services, settings, and
        other configurations.

        This operation performs the following tasks:

          - Stops all server machines participating in the site. This in
            turn stops all GIS services hosted on the server machines.
          - All services and cluster configurations are deleted.
          - All server machines are unregistered from the site.
          - All server machines are unregistered from the site.
          - The configuration store is deleted.

        .. note::
            This is an unrecoverable operation!

        :return:
           A status indicating success or failure.
        """
        return self._sm._delete()
    #----------------------------------------------------------------------
    def export(self, location=None):
        """
        Exports the site configuration to a location you specify as input
        to this operation.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Optional string. A path to a folder accessible to the server
                               where the exported site configuration will be written. If a location
                               is not specified, the server writes the exported site configuration
                               file to directory owned by the server and returns a virtual path
                               (an HTTP URL) to that location from where it can be downloaded.
        ==================     ====================================================================

        :return:
           A status indicating success (along with the folder location) or failure.


        """
        return self._sm._export(location)
    #----------------------------------------------------------------------
    def import_site(self, location):
        """
        This operation imports a site configuration into the currently
        running site. Importing a site means replacing all site
        configurations (including GIS services, security configurations,
        and so on) of the currently running site with those contained in
        the site configuration file you supply as input. The input site
        configuration file can be obtained through the exportSite
        operation.

        This operation will restore all information included in the backup,
        as noted in exportSite. When it is complete, this operation returns
        a report as the response. You should review this report and fix any
        problems it lists to ensure your site is fully functioning again.
        The importSite operation lets you restore your site from a backup
        that you created using the exportSite operation.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Required string. A file path to an exported configuration or an ID
                               referencing the stored configuration on the server.
        ==================     ====================================================================


        :return:
           A status indicating success (along with site details) or failure.
        """
        return self._sm._import_site(location=location)
    #----------------------------------------------------------------------
    def upgrade(self, run_async=False):
        """
        This is the first operation that must be invoked during an ArcGIS
        Server upgrade. Once the new software version has been installed
        and the setup has completed, this operation will be available. A
        successful run of this operation will complete the upgrade of
        ArcGIS Server.


        .. note::
            If errors are returned with the upgrade operation, you must address
            the errors before you can continue. For example, if you encounter
            an error about an invalid license, you will need to re-authorize
            the software using a valid license and you may then retry this
            operation.

            This operation is available only when a server machine is currently
            being upgraded. It will not be available after a successful upgrade
            of a server machine.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        run_async              Required string. A flag to indicate if the operation needs to be run
                               asynchronously. The default value is False.
        ==================     ====================================================================


        :return:
           A status indicating success or failure.

        """
        return self._sm._upgrade(run_async)
    #----------------------------------------------------------------------
    @property
    def public_key(self):
        """Gets the public key."""
        return self._sm._public_key
