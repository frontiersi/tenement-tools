class InvitationManager(object):
    """

    The `InvitationManager` provides functionality to see the existing invitations
    set out via email to your organization.  The manager has the ability to delete
    any invitation sent out by an organization.

    """
    _gis = None
    _url = None
    _invites = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis):
        self._url = url
        self._gis = gis
    #----------------------------------------------------------------------
    def __str__(self):
        return "<InvitationManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<InvitationManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __len__(self):
        url = self._url
        params = {
            'f' : 'json',
            'num':100,
            'start':1
        }
        res = self._gis._con.get(url, params)
        return res['total']
    #----------------------------------------------------------------------
    def list(self):
        """
        Returns all the organizations invitations

        :returns: List

        """
        invites = []
        from arcgis.gis import GIS
        isinstance(self._gis, GIS)
        url = self._url
        params = {
            'f' : 'json',
            'num':100,
            'start':1
        }
        res = self._gis._con.get(url, params)
        invites = res["invitations"]
        while res['nextStart'] > -1:
            params['start'] += 100
            res = self._gis._con.get(url, params)
            invites.extend(res["invitations"])
        return invites
    def get(self, invite_id):
        """
        Returns information about a single invitation

        :returns: Dict

        """
        url = self._url + "/{id}".format(id=invite_id)
        params = {
            'f' : 'json'
        }
        return self._gis._con.get(url, params)

    #----------------------------------------------------------------------
    def delete(self, invite_id):
        """
        deletes an invitation by ID

        :returns: Boolean
        """
        url = self._url + "/{id}/delete".format(id=invite_id)
        params = {
            'f' : 'json'
        }
        res = self._gis._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res