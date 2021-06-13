"""
This is the ArcGIS Notebook Server API Framework
"""

import os
from urllib.parse import urlparse

from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap

from ._logs import LogManager
from ._system import SystemManager
from ._security import SecurityManager
from ._machines import MachineManager
from ._nbm import NotebookManager

########################################################################
class NotebookServer(object):
    """
    Provides access to the ArcGIS Notebook Server administration API.
    """
    _gis = None
    _url = None
    _properties = None
    _logs = None
    _system = None
    _machine = None
    _notebook = None
    _security = None
    _version = None
    _sitemanager = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        if url.lower().endswith("/admin") == False:
            url += "/admin"
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
        else:
            raise ValueError("Invalid GIS object")
    #----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {'f': 'json'}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})
    #----------------------------------------------------------------------
    def __str__(self):
        return "<NotebookServer @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<NotebookServer @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """Properties of the object"""
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    @property
    def version(self):
        """
        Returns the notebook server version

        :returns: List
        """
        if self._version is None:
            self._version = [int(i) for i in self.properties.version.split('.')]
        return self._version
    #----------------------------------------------------------------------
    @property
    def site(self):
        """
        Provides access to the notebook server's site management operations

        :returns: SiteManager
        """
        if self._sitemanager is None:
            from ._site import SiteManager
            self._sitemanager = SiteManager(url=self._url, notebook=self, gis=self._gis)
        return self._sitemanager
    #----------------------------------------------------------------------
    @property
    def info(self):
        """
        Returns information about the server site itself

        :returns: PropertyMap

        """
        url = self._url + "/info"
        params = {'f' : 'json'}
        res = self._gis._con.get(url, params)
        return PropertyMap(res)
    #----------------------------------------------------------------------
    @property
    def health_check(self):
        """

        The `health_check` verifies that your ArcGIS Notebook Server site
        has been created, and that its Docker environment has been
        correctly configured.

        **This is only avaible if the site can be accessed around the web adapter**

        :returns: boolean

        """
        netloc = urlparse(self._url).netloc
        url = "https://{base}:11443/arcgis/rest/info/healthcheck".format(base=netloc)
        params = {'f' : 'json'}
        res = self._gis._con.get(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    @property
    def logs(self):
        """
        Provides access to the notebook server's logging system

        :returns: LogManager

        """
        if self._logs is None:
            url = self._url + "/logs"
            self._logs = LogManager(url=url, gis=self._gis)
        return self._logs
    #----------------------------------------------------------------------
    @property
    def system(self):
        """
        returns access to the system properties of the ArcGIS Notebook Server

        :return: SystemManager

        """
        if self._system is None:
            url = self._url + "/system"
            self._system = SystemManager(url=url, gis=self._gis)
        return self._system
    #----------------------------------------------------------------------
    @property
    def machine(self):
        """
        Provides access to managing the registered machines with ArcGIS
        Notebook Server

        :returns: MachineManager

        """
        if self._machine is None:
            url = self._url + "/machines"
            self._machine = MachineManager(url=url, gis=self._gis)
        return self._machine
    #----------------------------------------------------------------------
    @property
    def security(self):
        """
        Provides access to managing the ArcGIS Notebook Server's security
        settings.

        :return: SecurityManager

        """
        if self._security is None:
            url = self._url + "/security"
            self._security = SecurityManager(url=url, gis=self._gis)
        return self._security
    #----------------------------------------------------------------------
    @property
    def notebooks(self):
        """
        Provides access to managing the ArcGIS Notebook Server's
        Notebooks

        :return: NotebookManager
        """
        if self._notebook is None:
            url = self._url + "/notebooks"
            self._notebook = NotebookManager(url=url, gis=self._gis, nbs=self)
        return self._notebook
    #----------------------------------------------------------------------
    @property
    def url(self):
        """The URL of the notebook server."""
        return self._url
