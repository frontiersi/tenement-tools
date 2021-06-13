import os
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap

########################################################################
class SystemManager(object):
    """
    The System resource is a collection of server-wide resources in your
    ArcGIS Notebook Server site. Within this resource, you can access
    information and perform operations pertaining to licenses, Web
    Adaptors, containers, server properties, directories, Jobs, and the
    configuration store.
    """
    _url = None
    _con = None
    _gis = None
    _dir = None
    _wam = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            url = self._url + "/properties"
            params = {'f': 'json'}
            res = self._con.get(url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<SystemManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<SystemManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """
        ArcGIS Notebook Server has configuration properties that govern
        some of its intricate behavior. This resource is a container for
        these properties. The properties are available to all server
        objects and extensions through the server environment interface.


        The available properties are as follows:

            + WebContextURL - Defines the web front-end as seen by your users. Example: https://mydomain.com/gis
            + maxContainersPerNode - The default maximum number of containers that can be opened on a notebook server machine assuming the machine has the necessary CPU/Memory resources to support the containers.
            + idleNotebookThreshold - Specifies the time (in minutes) after which idle notebooks are closed automatically.
            + containerCreatedThreshold - Specifies the time (in minutes) after which an empty container is closed automatically.
            + webSocketSize - Specifies the amount of memory (in MB) available to ArcGIS Notebooks for WebSocket communication

        :returns: PropertyMap
        """
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    @property
    def recent_statistics(self):
        """
        returns statistics about the current state of the notebook server
        
        :returns: Dictionary
        """
        try:
            url = self._url + "/statistics/mostRecent"
            params = {'f' : 'json' }
            return self._con.get(url, params)
        except:
            raise Exception(("Recent Statistics is not supported on your cur"
                            "rent version of Notebook Server, please use v10.8.1+"))
    #----------------------------------------------------------------------
    @properties.setter
    def properties(self, value):
        """
        Sets the ArcGIS Notebook Server has configuration properties that govern
        some of its intricate behavior. This resource is a container for
        these properties. The properties are available to all server
        objects and extensions through the server environment interface.


        The available properties are as follows:

            + WebContextURL - Defines the web front-end as seen by your users. Example: https://mydomain.com/gis
            + maxContainersPerNode - The default maximum number of containers that can be opened on a notebook server machine assuming the machine has the necessary CPU/Memory resources to support the containers.
            + idleNotebookThreshold - Specifies the time (in minutes) after which idle notebooks are closed automatically.
            + containerCreatedThreshold - Specifies the time (in minutes) after which an empty container is closed automatically.
            + webSocketSize - Specifies the amount of memory (in MB) available to ArcGIS Notebooks for WebSocket communication

        :returns: PropertyMap
        """
        properties: {
          "dockerConnectionPort": 2375,
          "webSocketSize": 16,
          "maxContainersPerNode": 20,
          "containersStartPort": 30001,
          "containersStopPort": 31000,
          "idleNotebookThreshold": 1440,
          "containerCreatedThreshold": 60,
          "dockerConnectionHost": "localhost"
        }
        props = {}
        url = self._url + "/properties/update"
        params = {'f' : 'json',
                  'properties' : {}}
        current = dict(self.properties)
        for k in current.keys():
            if k in value:
                params['properties'][k] = value[k]
            else:
                params['properties'][k] = current[k]
        res = self._con.post(url, params)
        if not 'status' in res:
            raise Exception(res)
    #----------------------------------------------------------------------
    @property
    def containers(self):
        """
        Returns a list of active containers.

        :returns: List of Container classes
        """
        container = []
        url = self._url + "/containers"
        params = {'f' : 'json'}
        res = self._con.get(url, params)
        if "containers" in res:
            for c in res["containers"]:
                cid = c['id']
                curl = self._url + "/containers/{cid}".format(cid=cid)
                container.append(Container(url=curl, gis=self._gis))
        return container

    #----------------------------------------------------------------------
    @property
    def licenses(self):
        """
        Gets the license resource list.  The licenses resource lists the
        current license level of ArcGIS Notebook Sever and all authorized
        extensions. Contact Esri Customer Service if you have questions
        about license levels or expiration properties.
        """
        url = self._url + "/licenses"
        params = {
            "f" : "json"
        }
        return self._con.get(url,
                             params)
    #----------------------------------------------------------------------
    @property
    def web_adaptors(self):
        """
        returns a list of web adapters

        :return: List
        """
        if self._wam is None:
            url = self._url + "/webadaptors"
            self._wam = WebAdaptorManager(url=url, gis=self._gis)
        return self._wam
    #----------------------------------------------------------------------
    @property
    def jobs(self):
        """
        This resource is a collection of all the administrative jobs
        (asynchronous operations) created within your site. When operations
        that support asynchronous execution are run, ArcGIS Notebook Server
        creates a new job entry that can be queried for its current status
        and messages.

        :returns: list

        """
        url = self._url + "/jobs"
        params = {'f' : 'json'}
        res = self._con.get(url, params)
        if "asyncJobs" in res:
            return res["asyncJobs"]
        return res
    #----------------------------------------------------------------------
    @property
    def directories(self):
        """Provides access to registering directories"""
        if self._dir is None:
            url = self._url + "/directories"
            self._dir = DirectoryManager(url=url, gis=self._gis)
        return self._dir
    #----------------------------------------------------------------------
    def job_details(self, job_id):
        """
        A job represents the asynchronous execution of an operation in
        ArcGIS Notebook Server. You can acquire progress information by
        periodically querying the job.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        job_id                 Required String. The unique identifier of the job.
        ==================     ====================================================================

        :returns: dict

        """
        url = self._url + "/jobs/{jid}".format(jid=job_id)
        params = {'f':'json'}
        return self._con.get(url, params)
    #----------------------------------------------------------------------
    @property
    def config_store(self):
        """
        The configuration store maintains configurations for ArcGIS Notebook
        Server. Typical configurations include all the resources such as
        machines and security rules that are required to power the site. In
        a way, the configuration store is a physical representation of a site.

        Every ArcGIS Notebook Server machine, when it joins the site, is
        provided with a connection to the configuration store and it can
        thereafter participate in the management of the site. You can change
        the store's properties during runtime using the edit operation.

        The Administrator API that runs on every server machine is capable
        of reading and writing to the store. As a result, the store must be
        accessible to every server machine within the site. The default
        implementation is built on top of a file system and stores all the
        configurations in a hierarchy of folders and files.

        :returns: dict

        """
        url = self._url + "/configStore"
        params = {'f' : 'json'}
        return self._con.get(url, params)

########################################################################
class DirectoryManager(object):
    """
    A manages and maintains a collection of all server directories.
    """
    _url = None
    _con = None
    _gis = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            url = self._url
            params = {'f': 'json'}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<DirectoryManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<DirectoryManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    def list(self):
        """
        returns the current registered directories

        :return: List

        """
        self._properties = None
        val = dict(self.properties)
        return val['directories']
    #----------------------------------------------------------------------
    def register(self, name, path, directory_type):
        """
        This operation registers a new data directory from your local
        machine with the ArcGIS Notebook Server site. Registering a local
        folder as a data directory allows your notebook authors to work with
        files in the folder.

        ==================     ====================================================================
        **Parameter**          **Description**
        ------------------     --------------------------------------------------------------------
        name	               The name of the directory.
        ------------------     --------------------------------------------------------------------
        path	               The full path to the directory on your machine.
        ------------------     --------------------------------------------------------------------
        directory_type	       The type of directory. Values: DATA | WORKSPACE | OUTPUT
        ==================     ====================================================================

        :returns: boolean

        """
        params = {
            'f' : 'json',
            'name' : name,
            'path' : path,
            'type' : directory_type
        }
        url = self._url + "/register"
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def unregister(self, directory_id):
        """
        This operation unregisters an existing directory from the ArcGIS
        Notebook Server site.

        ==================     ====================================================================
        **Parameter**          **Description**
        ------------------     --------------------------------------------------------------------
        directory_id           Required String.  The directory ID to remove.
        ==================     ====================================================================

        :returns: boolean

        """
        params = {'f' : 'json'}
        url = self._url + "/{uid}/unregister" .format(uid=directory_id)
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
########################################################################
class WebAdaptorManager(object):
    """
    Manages and configures web adaptors for the ArcGIS Notebook Server.
    """
    _url = None
    _con = None
    _gis = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {'f': 'json'}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<WebAdapterManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<WebAdapterManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    def register(self,
                 name,
                 ip,
                 webadapter_url,
                 http_port,
                 https_port,
                 description=""):
        """
        Registers a new web adapter.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required String. The name of the web adapter
        ------------------     --------------------------------------------------------------------
        ip                     Required String. The IP of the web adapter.
        ------------------     --------------------------------------------------------------------
        webadapter_url         Required String. The URI endpoint of the web adpater.
        ------------------     --------------------------------------------------------------------
        http_port              Required Integer. The port number of the web adapter
        ------------------     --------------------------------------------------------------------
        https_port             Required Integer. The secure port of the web adapter.
        ------------------     --------------------------------------------------------------------
        description            Optional String. The optional web adapter description.
        ==================     ====================================================================

        :returns: Boolean

        """
        params = {
            "f" : "json",
            "machineName" : name,
            "machineIP" : ip,
            "webAdaptorURL" : webadaptor_url,
            "description" : description,
            "httpPort" : http_port,
            "httpsPort" : https_port
        }
        url = self._url + "/register"
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def config(self):
        """
        Gets the Web Adaptors configuration which is a resource of all the
        configuration parameters shared across all the Web Adaptors in the
        site. Most importantly, this resource lists the shared key that is
        used by all the Web Adaptors to encrypt key data bits for the
        incoming requests to the server.
        """
        url = self._url + "/config"
        params = {
            "f" : "json"
        }
        return self._con.get(url,
                             params)
    #----------------------------------------------------------------------
    @config.setter
    def config(self, config):
        """
        This is a property that allows for the retreival and manipulation of web adaptors.

        You can use this operation to change the Web Adaptor configuration
        and the sharedkey attribute. The sharedkey attribute must be present
        in the request.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required dict. The configuration items to be updated for this web
                               adaptor. Always include the web adaptor's sharedkey attribute.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True), else a Python dictionary containing an error message.

        """
        url = self._url + "/config/update"
        params = {
            "f" : "json",
            "webAdaptorConfig" : config
        }
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def list(self):
        """
        Returns all registered Web Adapters

        :return: List
        """
        url = self._url
        params = {'f' : 'json'}
        res = self._con.get(url, params)
        if "webAdaptors" in res:
            return [WebAdaptor(self._url + "/{wa}".format(wa=wa['id']),
                               gis=self._gis) for wa in res["webAdaptors"]]
        return res
########################################################################
class WebAdaptor(object):
    """
    This resource provides information about the ArcGIS Web Adaptor
    configured with your ArcGIS Notebook Server site. ArcGIS Web Adaptor
    is a web application that runs in a front-end web server. One of the
    Web Adaptor's primary responsibilities is to forward HTTP requests
    from end users to ArcGIS Notebook Server in a round-robin fashion.
    The Web Adaptor acts a reverse proxy, providing the end users with
    an entry point into the system, hiding the server itself, and
    providing some degree of immunity from back-end failures.

    The front-end web server could authenticate incoming requests against
    your enterprise identity stores and provide specific authentication
    schemes like Integrated Windows Authentication (IWA), HTTP Basic or
    Digest.

    Most importantly, ArcGIS Web Adaptor provides your end users with a
    well-defined entry point into your system without exposing the internal
    details of your server site. ArcGIS Notebook Server will trust requests
    being forwarded by ArcGIS Web Adaptor and will not challenge the user
    for any credentials. However, the authorization of the request (by
    looking up roles and permissions) is still enforced by the server site.

    ArcGIS Notebooks use the WebSocket protocol for communication. You can
    update the maximum size of the file sent using WebSocket by updating your
    site's webSocketMaxHeapSize property.
    """
    _url = None
    _con = None
    _gis = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {'f': 'json'}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<WebAdapter @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<WebAdapter @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def unregister(self):
        """
        Unregisters a WebAdapter for the Notebook Server
        :returns: boolean
        """
        url = self._url + "/unregister"
        params = {'f':'json'}
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
########################################################################
class Container(object):
    """
    This represents a single hosted notebook container.
    """
    _url = None
    _con = None
    _gis = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {'f': 'json'}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<Container @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<Container @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    @property
    def sessions(self):
        """
        When an ArcGIS Notebook is opened in a container, a computational
        engine called a kernel is launched; the kernel runs while the notebook
        is active and executes all the work done by the notebook. This
        resource tracks the active kernels running in the specified container
        to provide information about current notebook sessions.


        :returns: list of dict

        ==================     ====================================================================
        **Response**           **Description**
        ------------------     --------------------------------------------------------------------
        ID	               The unique ID string for the container session.
        ------------------     --------------------------------------------------------------------
        path	               The working path to the running ArcGIS Notebook, ending in .ipynb.
        ------------------     --------------------------------------------------------------------
        kernel	               Properties describing the kernel. They are as follows:

                                    + last_activity: The date and time of the most recent action performed by the kernel.
                                    + name: The name of the kernel. At 10.7, this value is python3.
                                    + id: The unique ID string for the kernel.
                                    + execution_state: Whether the kernel is currently executing an action or is idle.
                                    + connections: The number of users currently accessing the notebook.
        ==================     ====================================================================
        """
        url = self._url + "/sessions"
        params = {'f' : 'json'}
        res = self._con.get(url, params)
        if 'sessions' in res:
            return res['sessions']
        return res
    #----------------------------------------------------------------------
    def shutdown(self):
        """
        Terminates the current container

        :returns: boolean
        """
        url = self._url + "/terminateContainer"
        params = {'f' : 'json'}
        res = self._con.post(url, params)
        if 'status' in res:
            return res['status'] == 'success'
        return res



