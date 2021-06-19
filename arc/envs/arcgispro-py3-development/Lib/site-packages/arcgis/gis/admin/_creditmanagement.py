########################################################################
class CreditManager(object):
    """
    Manages an AGOL Site's Credits for users and sites

    **Example Usage**

    .. code-block:: python

        from arcgis.gis import GIS
        gis = GIS(profile='agol_account')
        cm = gis.admin.credits
        cm.allocate("user1", 100)



    """
    _gis = None
    _con = None
    _portal = None
    #----------------------------------------------------------------------
    def __init__(self, gis):
        """Constructor"""
        self._gis = gis
        self._portal = gis._portal
        self._con = self._portal.con
    #----------------------------------------------------------------------
    @property
    def credits(self):
        """returns the current number of credits on the GIS"""
        try:
            return self._gis.properties.availableCredits
        except:
            return 0
    #----------------------------------------------------------------------
    @property
    def is_enabled(self):
        """
        boolean that show is credit credit assignment
        """
        return self._gis.properties.creditAssignments == 'enabled'
    #----------------------------------------------------------------------
    def enable(self):
        """
        enables credit allocation on AGOL
        """
        return self._gis.update_properties(
            {"creditAssignments" : 'enabled'})
    #----------------------------------------------------------------------
    def disable(self):
        """
        disables credit allocation on AGOL
        """
        return self._gis.update_properties(
            {"creditAssignments" : 'disabled'})
    #----------------------------------------------------------------------
    @property
    def default_limit(self):
        """
        Gets/Sets the default credit allocation for AGOL
        """
        return self._gis.properties.defaultUserCreditAssignment
    #----------------------------------------------------------------------
    @default_limit.setter
    def default_limit(self, value):
        """
        Gets/Sets the default credit allocation for AGOL
        """
        params = {"defaultUserCreditAssignment" : value}
        self._gis.update_properties(params)
    #----------------------------------------------------------------------
    def allocate(self, username, credits=None):
        """
        Allows organization administrators to allocate credits for
        organizational users in ArcGIS Online

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        username                        Required string.The name of the user to assign credits to.
        ---------------------------     --------------------------------------------------------------------
        credits                         Optional float. The number of credits to assign to a user. If None
                                        is provided, it sets user to unlimited credits.
        ===========================     ====================================================================

        :returns: boolean

        """
        if hasattr(username, 'username'):
            username = getattr(username, "username")
        if not credits is None:
            params = {
                "f" : "json",
                "userAssignments" : [{"username" : username, "credits" : credits}]
            }
            path = "portals/self/assignUserCredits"
            res =  self._con.post(path, params)
            if 'success' in res:
                return res['success']
            return res
        else:
            return self.deallocate(username=username)
    #----------------------------------------------------------------------
    def deallocate(self, username):
        """
        Allows organization administrators to set credit limit to umlimited for
        organizational users in ArcGIS Online

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        username                        Required string.The name of the user to set to unlimited credits.
        ===========================     ====================================================================

        :returns: boolean

        """
        if hasattr(username, 'username'):
            username = getattr(username, "username")
        params = {"usernames" : [username],
                  "f" : 'json'}
        path = "portals/self/unassignUserCredits"
        res = self._con.post(path, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def credit_usage(self, start_time=None, end_time=None):
        """
        returns the total credit consumption for a given time period.

        ===================   ===============================================
        **arguements**        **description**
        -------------------   -----------------------------------------------
        start_time            datetime.datetime object. This is the date to
                              start at.
        -------------------   -----------------------------------------------
        end_time              datetime.datetime object. This is the stop time
                              to look for credit consumption. It needs to be
                              at least 1 day previous than then start_time.
        ===================   ===============================================

        returns: dictionary
        """
        import datetime
        if isinstance(start_time, datetime.datetime):
            start_time =int(start_time.timestamp() * 1000)
        else:
            start_time = int(datetime.datetime.now().timestamp() * 1000)
        if isinstance(end_time, datetime.datetime):
            end_time = int(end_time.timestamp() * 1000)
        else:
            end_time = int((datetime.datetime.now() - datetime.timedelta(days=5)).timestamp() * 1000)
        path = "portals/self/usage"
        params = {
        'f' : 'json',
        'startTime' : end_time,
        'endTime' : start_time,
        'period' : '1d',
        'groupby' : 'stype,etype',
        'vars' : 'credits,num'
        }
        data = self._con.get(path, params)
        res = {}
        for d in data['data']:
            if d['stype'] in res:
                res[d['stype']] += sum([float(a[1]) for a in d['credits']])
            else:
                res[d['stype']] = sum([float(a[1]) for a in d['credits']])
        return res
