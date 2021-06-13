"""
Entry point to working with licensing on Portal or ArcGIS Online
"""
from .._impl._con import Connection
from ..._impl.common._mixins import PropertyMap
from ...gis import GIS
from ._base import BasePortalAdmin
########################################################################
class LicenseManager(BasePortalAdmin):
    """
    Provides tools to work and manage licenses in ArcGIS Online and
    ArcGIS Enterprise (Portal)

    ===============     ====================================================
    **Argument**        **Description**
    ---------------     ----------------------------------------------------
    url                 required string, the web address of the site to
                        manage licenses.
                        example:
                        https://<org url>/<wa>/sharing/rest/portals/self/purchases
    ---------------     ----------------------------------------------------
    gis                 required GIS, the gis connection object
    ===============     ====================================================

    :returns:
       LicenseManager Object
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None
    def __init__(self, url, gis=None, initialize=True, **kwargs):
        """class initializer"""
        super(LicenseManager, self).__init__(url=url, gis=gis)
        self._url = url
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError(
                "connection must be of type GIS or Connection")
        if initialize:
            self._init(connection=self._con)
    #----------------------------------------------------------------------
    def __str__(self):
        return "<License Manager at {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    def get(self, name):
        """
        retrieves a license by it's name (title)
        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        name                required string, name of the entitlement to locate
                            on the organization.
                            example:
                            name="arcgis pro"
        ===============     ====================================================

        :returns:
           License Object
        """
        licenses = self.all()
        for l in licenses:
            if 'listing' in l.properties and \
               'title' in l.properties['listing'] and \
                l.properties['listing']['title'].lower() == name.lower():
                return l
            del l
        del licenses
        return None
    #----------------------------------------------------------------------
    def all(self):
        """
        Returns all Licenses registered with an organization

        :returns:
           list of License objects
        """
        licenses = []
        if self._properties is None:
            self._init()
        if 'purchases' in self.properties:
            purchases = self.properties['purchases']
            for purchase in purchases:
                licenses.append(License(gis=self._gis, info=purchase))
        return licenses
    #----------------------------------------------------------------------
    @property
    def bundles(self):
        """
        Returns a list of Application Bundles for an Organization

        :returns:
           list of Bundle objects

        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("`bundles` not implemented before version 6.4")

        url = "{base}/portals/self/appBundles".format(base=self._gis._portal.resturl)
        params = {
            'f' : 'json',
            'num' : 100,
            'start' : 1
        }

        res = self._con.get(url, params)
        buns = res['appBundles']
        while res['nextStart'] != -1:
            params['start'] = res['nextStart']
            res = self._gis._con.get(url, params)
            buns += res['appBundles']
            if res['nextStart'] == -1:
                break
        return [Bundle(url="{base}content/listings/{id}".format(base=self._gis._portal.resturl,
                                                id=b["appBundleItemId"]),
                       properties=b,
                       gis=self._gis)
                for b in buns]
    #----------------------------------------------------------------------
    @property
    def offline_pro(self):
        """
        Administrators can get/set the disconnect settings for the ArcGIS Pro licensing.
        A value of True means that a user can check out a license from the enterprise
        inorder to use it in a disconnected setting.  By setting `offline_pro` to False,
        the enterprise users cannot check out licenses to work in a disconnected setting
        for ArcGIS Pro.

        :returns: Boolean

        """
        lic = self.get("arcgis pro")
        return lic.properties.provision.canDisconnect
    #----------------------------------------------------------------------
    @offline_pro.setter
    def offline_pro(self, value):
        """
        Administrators can get/set the disconnect settings for the ArcGIS Pro licensing.
        A value of True means that a user can check out a license from the enterprise
        inorder to use it in a disconnected setting.  By setting `offline_pro` to False,
        the enterprise users cannot check out licenses to work in a disconnected setting
        for ArcGIS Pro.

        """
        import json
        lic = self.get("arcgis pro")
        url = "{base}content/listings/{itemid}/setDisconnectSettings".format(
            base=self._gis._portal.resturl,
            itemid=lic.properties.provision.itemId
        )
        params = {
            "f" : "json",
            "canDisconnect" : json.dumps(value),
            "maxDisconnectDuration": -1
        }
        res = self._con.post(url, params)
        if 'success' in res and \
           res['success'] == False:
            raise Exception("Could not update the dicconnect settings.")
        elif 'success' not in res:
            raise Exception("Could not update the dicconnect settings: %s" % res)


########################################################################
class Bundle(object):
    """
    This represents a single instance of an application bundle
    """
    _users = None
    _con = None
    _url = None
    _properties = None
    _gis = None
    _id = None
    #----------------------------------------------------------------------
    def __init__(self, url, properties=None, gis=None):
        """Constructor"""
        import os
        if gis is None:
            import arcgis
            gis = arcgis.env.active_gis
        self._gis = gis
        self._url = url
        self._con = gis._con
        self._id = os.path.basename(url)
        self._properties = properties
    #----------------------------------------------------------------------
    def _find(self, appid):
        """if properties are missing, populate the app bundle's properties"""
        url = "{base}/portals/self/appBundles".format(base=self._gis._portal.resturl)
        params = {
            'f' : 'json',
            'num' : 100,
            'start' : 1
        }

        res = self._con.get(url, params)
        buns = res['appBundles']
        while res['nextStart'] != -1:
            params['start'] = res['nextStart']
            res = self._gis._con.get(url, params)
            buns += res['appBundles']
            if res['nextStart'] == -1:
                break
        for b in buns:
            if b['appBundleItemId'] == self._id:
                return b
        raise ValueError("Invalid Application Bundle.")
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the application bundles properties"""
        if self._properties is None:
            try:
                params = {'f': 'json'}
                r = self._con.get(self._url, params)
                self._properties = PropertyMap(r)
            except:
                self._properties = PropertyMap({})
        elif isinstance(self._properties, dict):
            self._properties = PropertyMap(self._properties)
        return self._properties
    #----------------------------------------------------------------------
    @property
    def users(self):
        """returns a list of users assigned the application bundle"""
        if self._users is None:
            self._users = self._gis.users.search("appbundle:%s" % self.properties.appBundleItemId)
        return self._users
    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.users)
    #----------------------------------------------------------------------
    def __str__(self):
        """"""
        return "<AppBundle: %s >" % self.properties['name']
    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        return "<AppBundle: %s >" % self.properties['name']
    #----------------------------------------------------------------------
    def assign(self, users):
        """
        Assigns the current application bundle to a list of users

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        users               Required List. A list of user names or User objects
                            to assign the current application bundle to.
        ===============     ====================================================


        :returns: boolean

        """
        if isinstance(users, (tuple, set, list)) == False:
            users = [users]
        from arcgis.gis import User
        url = "%s%s" % (self._url, "/provisionUserAppBundle")
        params = {
            'f' : 'json',
            'users' : None,
            'revoke' : False
        }
        us = []
        for user in users:
            if isinstance(users, str):
                us.append(user)
            elif isinstance(user, User):
                us.append(user.username)
        params['users'] = ",".join(us)
        res = self._con.post(url, params)
        self._users = None
        self._properties = None
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def revoke(self, users):
        """
        Revokes the current application bundle to a list of users

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        users               Required List. A list of user names or User objects
                            to remove the current application bundle to.
        ===============     ====================================================


        :returns: boolean

        """
        if isinstance(users, (tuple, set, list)) == False:
            users = [users]
        from arcgis.gis import User
        url = "%s%s" % (self._url, "/provisionUserAppBundle")
        params = {
            'f' : 'json',
            'users' : None,
            'revoke' : True
        }
        us = []
        self._users = None
        self._properties = None
        for user in users:
            if isinstance(users, str):
                us.append(user)
            elif isinstance(user, User):
                us.append(user.username)
        params['users'] = ",".join(us)
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res


########################################################################
class License(object):
    """
    Represents a single entitlement for a given organization.


    ===============     ====================================================
    **Argument**        **Description**
    ---------------     ----------------------------------------------------
    gis                 required GIS, the gis connection object
    ---------------     ----------------------------------------------------
    info                required dictionary, the information provided by
                        the organization's site containing the provision
                        and listing information.
    ===============     ====================================================

    :returns:
       License Object
    """
    _properties = None
    _gis = None
    _con = None
    #----------------------------------------------------------------------
    def __init__(self, gis, info):
        """Constructor"""
        self._gis = gis
        self._con = gis._con
        self._properties = PropertyMap(info)
    #----------------------------------------------------------------------
    def __str__(self):
        try:
            return '<%s %s at %s>' % (self.properties['listing']['title'],type(self).__name__, self._gis._portal.resturl)
        except:
            return '<%s at %s>' % (type(self).__name__, self._gis._portal.resturl)
    #----------------------------------------------------------------------
    def __repr__(self):
        try:
            return '<%s %s at %s>' % (self.properties['listing']['title'],type(self).__name__, self._gis._portal.resturl)
        except:
            return '<%s at %s>' % (type(self).__name__, self._gis._portal.resturl)
    #----------------------------------------------------------------------
    @property
    def properties(self):
        return self._properties
    #----------------------------------------------------------------------
    @property
    def report(self):
        """
        returns a Panda's Dataframe of the licensing count.
        """
        import pandas as pd
        data = []
        columns = ['Entitlement', 'Total', 'Assigned', 'Remaining']
        if 'provision' in self.properties:
            for k,v in self.properties['provision']['orgEntitlements']['entitlements'].items():
                counter = 0
                for u in self.all():
                    if k in u['entitlements']:
                        counter += 1
                row = [k, v['num'], counter, v['num'] - counter]
                data.append(row)
                del k, v
        return pd.DataFrame(data=data, columns=columns)
    #----------------------------------------------------------------------
    def plot(self):
        """returns a simple bar chart of assigned and remaining entitlements"""
        report = self.report
        try:
            return report.plot(x=report["Entitlement"],
                               y=['Assigned', 'Remaining'],
                               kind='bar',stacked=True).legend(loc='best')
        except:
            report.set_index("Entitlement", drop=True, append=False,
                             inplace=True, verify_integrity=False)
            return report.plot(y=['Assigned', 'Remaining'],
                               kind='bar',stacked=True).legend(loc='best')
    #----------------------------------------------------------------------
    def all(self):
        """
        returns a list of all usernames and their entitlements for this license
        """
        item_id = self.properties['listing']['itemId']
        url = "%scontent/listings/%s/userEntitlements" % (self._gis._portal.resturl, item_id)
        start = 1
        num = 100
        params = {
            'start' : start,
            'num' : num
        }
        user_entitlements = []
        res = self._con.get(url, params)
        user_entitlements += res['userEntitlements']
        if 'nextStart' in res:
            while res['nextStart'] > 0:
                start += num
                params = {
                    'start' : start,
                    'num' : num
                }
                res = self._con.get(url, params)
                user_entitlements += res['userEntitlements']
        return user_entitlements
    #----------------------------------------------------------------------
    def user_entitlement(self, username):
        """
        checks if a user has the entitlement assigned to them

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, the name of the user you want to
                            examine the entitlements for.
        ===============     ====================================================

        :returns:
           dictionary
        """
        item_id = self.properties['listing']['itemId']
        url = "%scontent/listings/%s/userEntitlements" % (
            self._gis._portal.resturl,
            item_id)
        start = 1
        num = 100
        params = {
            'start' : start,
            'num' : num
        }
        user_entitlements = []
        res = self._con.get(url, params)
        for u in res['userEntitlements']:
            if u['username'].lower() == username.lower():
                return u
        if 'nextStart' in res:
            while res['nextStart'] > 0:
                start += num
                params = {
                    'start' : start,
                    'num' : num
                }
                res = self._con.get(url, params)
                for u in res['userEntitlements']:
                    if u['username'].lower() == username.lower():
                        return u
        return {}
    #----------------------------------------------------------------------
    def assign(self, username, entitlements, suppress_email=True):
        """
        grants a user an entitlement.

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, the name of the user you wish to
                            assign an entitlement to.
        ---------------     ----------------------------------------------------
        entitlments         required list, a list of entitlements values
        ---------------     ----------------------------------------------------
        suppress_email       optional boolean, if True, the org will not notify
                            a user that their entitlements has changed (default)
                            If False, the org will send an email notifying a
                            user that their entitlements have changed.
        ===============     ====================================================

        :returns:
           boolean
        """
        item_id = self.properties['listing']['itemId']
        if isinstance(entitlements, str):
            entitlements = entitlements.split(',')
        params = {
            "f" : "json",
            "userEntitlements" : {"users":[username],
                                  "entitlements":entitlements},

        }
        if suppress_email is not None:
            params["suppressCustomerEmail"] = suppress_email
        url = "%scontent/listings/%s/provisionUserEntitlements" % (self._gis._portal.resturl, item_id)
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success'] == True
        return res
    #----------------------------------------------------------------------
    def revoke(self, username, entitlements, suppress_email=True):
        """
        removes a specific license from a given entitlement

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, the name of the user you wish to
                            assign an entitlement to.
        ---------------     ----------------------------------------------------
        entitlments         required list, a list of entitlements values,
                            if * is given, all entitlements will be revoked
        ---------------     ----------------------------------------------------
        suppress_email      optional boolean, if True, the org will not notify
                            a user that their entitlements has changed (default)
                            If False, the org will send an email notifying a
                            user that their entitlements have changed.
        ===============     ====================================================

        :returns:
           boolean
        """
        if entitlements == "*":
            return self.assign(username=username,
                                  entitlements=[],
                                  suppress_email=suppress_email)
        elif isinstance(entitlements, list):
            es = self.user_entitlement(username=username)

            if 'entitlements' in es:
                lookup = {e.lower() : e for e in es['entitlements']}
                es = [e.lower() for e in es['entitlements']]
                if isinstance(entitlements, str):
                    entitlements = [entitlements]
                entitlements = list(set(es) - set([e.lower() for e in entitlements]))
                es2 = []
                for e in entitlements:
                    es2.append(lookup[e])
                return self.assign(username=username,
                                   entitlements=es2,
                                   suppress_email=suppress_email)
        return False
