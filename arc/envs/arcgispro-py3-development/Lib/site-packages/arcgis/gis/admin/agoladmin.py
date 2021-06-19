"""
Entry point to working with local enterprise GIS functions
"""
from .._impl._con import Connection
from ...gis import GIS
from ._resources import PortalResourceManager
from ._base import BasePortalAdmin
from ...apps.tracker._location_tracking import LocationTrackingManager
########################################################################
class AGOLAdminManager(object):
    """
    This is the root resource for administering your online GIS. Starting from
    this root, all of the GIS's environment is organized into a
    hierarchy of resources and operations.

    Parameter:
    :param gis: GIS object containing Administrative credentials
    :param ux: the UX object (optional)
    :param metadata: the metadata manager object (optional)
    :param collaborations: the CollaborationManager object (optional)
    """
    _gis = None
    _ux = None
    _idp = None
    _pp = None
    _credits = None
    _metadata = None
    _collaborations = None
    _ur = None
    _sp = None
    _license = None
    _usage = None
    _category_schema = None
    _certificates = None
    #----------------------------------------------------------------------
    def __init__(self,
                 gis,
                 ux=None,
                 metadata=None,
                 collaborations=None):
        """initializer"""
        self._gis = gis
        self._ux = ux
        self._collaborations = collaborations
        self._metadata = metadata
        self.resources = PortalResourceManager(gis=self._gis)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._gis._portal.resturl)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._gis._portal.resturl)
    #----------------------------------------------------------------------
    @property
    def ux(self):
        """returns a UX/UI manager"""
        if self._ux is None:
            from ._ux import UX
            self._ux = UX(gis=self._gis)
        return self._ux
    #----------------------------------------------------------------------
    @property
    def _user_experience_program(self):
        """
        ArcGIS Online works continuously to improve our products and one of
        the best ways to find out what needs improvement is through
        customer feedback. The Esri User Experience Improvement program
        (EUEI) allows your organization to contribute to the design and
        development of ArcGIS Online. The program collects information
        about the usage of ArcGIS Online including hardware and browser
        characteristics, without interrupting work. The program is
        completely optional and anonymous; none of the information
        collected is used to identify or contact members of your
        organization.
        """
        return self._gis.properties['eueiEnabled']
    #----------------------------------------------------------------------
    @_user_experience_program.setter
    def _user_experience_program(self, value):
        """
        ArcGIS Online works continuously to improve our products and one of
        the best ways to find out what needs improvement is through
        customer feedback. The Esri User Experience Improvement program
        (EUEI) allows your organization to contribute to the design and
        development of ArcGIS Online. The program collects information
        about the usage of ArcGIS Online including hardware and browser
        characteristics, without interrupting work. The program is
        completely optional and anonymous; none of the information
        collected is used to identify or contact members of your
        organization.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required boolean. True means that the organization will be enrolled
                            in the Esri User Experience Improvement Program. False means the
                            organization will not be part of the program.
        ===============     ====================================================================

        """
        if value != self.user_experience_program:
            self._gis.update_properties({
                "clearEmptyFields" : True,
                "eueiEnabled" : value
            })
            self._gis._get_properties(True)
    #----------------------------------------------------------------------
    @property
    def collaborations(self):
        """
        The collaborations resource lists all collaborations in which a
        portal participates
        """
        if self._collaborations is None:
            from ._collaboration import CollaborationManager
            self._collaborations = CollaborationManager(gis=self._gis)
        return self._collaborations
    #----------------------------------------------------------------------
    @property
    def category_schema(self):
        """
        This resource allows for the setting and manipulating of catagory
        schemas.
        """
        if self._category_schema is None:
            from ._catagoryschema import CategoryManager
            self._category_schema = CategoryManager(gis=self._gis)
        return self._category_schema
    #----------------------------------------------------------------------
    @property
    def idp(self):
        """
        This resource allows for the setting and configuration of the identity provider
        """
        if self._idp is None:
            from ._idp import IdentityProviderManager
            self._idp = IdentityProviderManager(gis=self._gis)
        return self._idp
    #----------------------------------------------------------------------
    @property
    def location_tracking(self):
        """
        The manager for Location Tracking. See :class:`~arcgis.apps.tracker.LocationTrackingManager'.
        """
        return LocationTrackingManager(self._gis)
    @property
    #----------------------------------------------------------------------
    def social_providers(self):
        """
        This resource allows for the setting and configuration of the social providers
        for a GIS.
        """
        if self._sp is None:
            from ._socialproviders import SocialProviders
            self._sp = SocialProviders(gis=self._gis)
        return self._sp
    #----------------------------------------------------------------------
    @property
    def credits(self):
        """
        manages the credits on a ArcGIS Online
        """
        if self._credits is None:
            from ._creditmanagement import CreditManager
            self._credits = CreditManager(gis=self._gis)
        return self._credits
    #----------------------------------------------------------------------
    @property
    def metadata(self):
        """
        resources to work with metadata on GIS
        """
        if self._metadata is None:
            from ._metadata import MetadataManager
            self._metadata = MetadataManager(gis=self._gis)
        return self._metadata
    #----------------------------------------------------------------------
    @property
    def password_policy(self):
        """tools to manage a Site's password policy"""
        if self._pp is None:
            from ._security import PasswordPolicy
            url = "%s/portals/self/securityPolicy" % (self._gis._portal.resturl)
            self._pp = PasswordPolicy(url=url,
                                      gis=self._gis)
        return self._pp
    #----------------------------------------------------------------------
    @property
    def usage_reports(self):
        """
        provides access to the usage reports of the AGOL organization
        """
        if self._ur is None:
            from ._usage import AGOLUsageReports
            url = "%sportals/%s/usage" % (self._gis._portal.resturl,
                                          self._gis.properties.id)
            self._ur = AGOLUsageReports(url=url, gis=self._gis)
        return self._ur
    #----------------------------------------------------------------------
    @property
    def license(self):
        """
        provides a set of tools to access and manage user licenses and
        entitlements.
        """
        if self._license is None:
            from ._license import LicenseManager
            url = self._gis._portal.resturl + "portals/self/purchases"
            self._license = LicenseManager(url=url, gis=self._gis)
        return self._license
    #----------------------------------------------------------------------
    @property
    def urls(self):
        """
        returns the URLs to the Hosting and Tile Server for ArcGIS Online
        """
        res = self._gis._con.get(path="%s/portals/%s/urls" % (self._gis._portal.resturl,
                                                              self._gis.properties.id),
                                      params={'f': 'json'})
        return res
    #----------------------------------------------------------------------
    def history(self, 
                start_date, 
                to_date=None, 
                num=100, 
                all_events=True, 
                event_ids=None,
                event_types=None, 
                actors=None, 
                owners=None,
                actions=None, 
                ips=None,
                sort_order='asc',
                data_format='csv',
                save_folder=None):
        """
        Returns a CSV file or Pandas's DataFrame containing the login history from a start_date to the present.

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        start_date        Required datetime.datetime object. The beginning date to start with.
        ----------------  -------------------------------------------------------------------------------
        to_date           Optional datetime.datetime object. The ending date.  If not provided, the query
                          will attempt to obtain all records till the current date.  
        ----------------  -------------------------------------------------------------------------------
        num               Optional Integer. The maximum number of records to return.  The maximum value 
                          is 10,000 set by the ArcGIS REST API.  If the value of -1 is provided it will 
                          attempt to get all records for the date range.  The default is **100**.
        ----------------  -------------------------------------------------------------------------------
        all_events        Optional Boolean. If `True`, all types of events are included.  If `False`, only 
                          actions targeted by the organization are included.  When exporting as `csv` this
                          parameter is `True`.
        ----------------  -------------------------------------------------------------------------------
        event_id          Optional String. Filter events by specific target user name or target ID in a batch result set. 
                          It can be the ID of an item, a group, a role, a collaboration, an identity 
                          provider, and so on.
        ----------------  -------------------------------------------------------------------------------
        event_types       Optional String.  Filter events by a comma-separated list of target types in a 
                          batch result set.

                          Values: a (organization), c (collaboration), cp (collaboration participate), 
                                  cpg (collaboration participate group), cw (collaboration workspace), 
                                  cwp (collaboration workspace participate), g (group), i (item), 
                                  idp (identity provider), inv (invitation), r (role), u (user)
        ----------------  -------------------------------------------------------------------------------
        actors            Optional String. Comma seperated list of usernames.
        ----------------  -------------------------------------------------------------------------------
        owners            Optional String. Filter events by a comma-separated list of user names who own 
                          the action targets in a batch result set.
        ----------------  -------------------------------------------------------------------------------
        actions           Optional String. Comma seperated list of actions to query for.  
                     
                          Values: `add`, `addusers`, `create`, `delete`, `removeusers`, `share`, `unshare`, 
                          `update`, `failedlogin`, `login`, and `updateUsers`.
        ----------------  -------------------------------------------------------------------------------
        ips               Optional String. Filter events by a comma-separated list of IP addresses in a batch result set.
        ----------------  -------------------------------------------------------------------------------
        sort_order        Optional String.  Describes whether the results return in ascending or 
                          descending chronological order. The default is ascending.

                          Values: `asc` or `desc`
        ----------------  -------------------------------------------------------------------------------
        data_format       Optional String.  The way the data is returned to the user.  The response can 
                          be a `df` or `csv`.
                          
                          Values: `df` or `csv`
        ----------------  -------------------------------------------------------------------------------
        save_folder       Optional String. The save location of the CSV file.
        ================  ===============================================================================

        :returns: string or pd.DataFrame

        """
        import tempfile, json
        from arcgis._impl.common._utils import _date_handler
        if save_folder is None:
            save_folder = tempfile.gettempdir()
        if num == 0:
            raise ValueError("`num` cannot be zero.")
        url = "{url}portals/self/history".format(url=self._gis._portal.resturl)
        params = {
            'f' : data_format,
            'num' : num,
            #'start' : "",
            'all' : all_events,
            'id' : event_ids,
            'types' : event_types,
            'actors' : actors,
            'owners' : owners,
            'actions' : actions,
            'fromDate' : json.dumps(start_date, default=_date_handler),
            
            'ips' : ips,
            'sortOrder' : sort_order,
            'ips' : ips
        }
        
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        if to_date:
            params['toDate'] = json.dumps(to_date, 
                                          default=_date_handler)
            
        if data_format == 'csv':
            params['f'] = 'csv'
            params['num'] = 10000
            params['all'] = True
            return self._gis._con.post(url, params,
                                       file_name="history.csv",
                                       out_folder=save_folder,
                                       try_json=False)
        elif data_format in ['json', 'df']:
            import pandas as _pd
            params['f'] = 'json'
            data = []
            
            res = self._gis._con.post(url, params)
            data.extend(res['items'])
            while len(res['items']) > 0 and res['nextKey']:
                params['start'] = res['nextKey']
                res = self._gis._con.post(url, params)
                data.extend(res['items'])
                if num > 0 and len(data) >= num:
                    data = data[:num]
                    break
            return _pd.DataFrame(data)
    #----------------------------------------------------------------------
    @property
    def certificates(self):
        """
        Provides access to managing the organization's certificates.
        """
        if self._certificates is None:
            from .._impl import CertificateManager
            self._certificates = CertificateManager(gis=self._gis)
        return self._certificates

