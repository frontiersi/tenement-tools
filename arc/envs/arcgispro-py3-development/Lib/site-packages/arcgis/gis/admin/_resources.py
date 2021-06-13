"""
Provides class, methods and functions to manage resources for a given GIS.
"""
import os
import json
import tempfile


class PortalResourceManager(object):
    """
    Helper class to manage a GIS' resources

    ================  ===============================================================
    **Argument**      **Description**
    ----------------  ---------------------------------------------------------------
    gis               required GIS, connection to ArcGIS Online or ArcGIS Enterprise
    ================  ===============================================================


    """
    _workdir = None

    def __init__(self, gis):
        """Creates helper object to manage custom roles in the GIS"""
        self._gis = gis
        self._portal = gis._portal
        self._is_portal = self._gis.properties.isPortal
        self._workdir = tempfile.gettempdir()


    def add(self, key=None, path=None, text=None, **kwargs):
        """
        The add resource operation allows the administrator to add a file
        resource, for example, the organization's logo or custom banner.
        The resource can be used by any member of the organization. File
        resources use storage space from your quota and are scanned for
        viruses.


        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        key               optional string, look up key for file
        ----------------  ---------------------------------------------------------------
        path              optional string, file path to the local resource to upload
        ----------------  ---------------------------------------------------------------
        text              optional string, text value to add to the site's resources
        ----------------  ---------------------------------------------------------------
        access            optional string, sets the access level for the resource. The
                          default is 'public'.
                          Values: public, org, orgprivate
        ================  ===============================================================

        :returns:
           boolean
        """
        access = kwargs.pop("access", None)
        files = None
        if key is None and path:
            key = os.path.basename(path)
        elif key is None and path is None:
            raise ValueError("key must be populated is path is null")
        url = "portals/self/addResource"
        postdata = {
            "f" : "json",
            "key" : key,
        }
        if path:
            files = {
                'file' : path
            }
        if text:
            if isinstance(text, dict):
                postdata['text'] = json.dumps(text)
            elif isinstance(text, str):
                from arcgis._impl.common._utils import _to_utf8
                postdata['text'] = _to_utf8(text)
        else:
            if self._portal.is_arcgisonline == False:
                postdata['text'] = ""
        if self._is_portal == False:
            url = "portals/%s/addResource" % self._gis.properties.id
            if text is None:
                postdata['text'] = ""
            if access:
                postdata['access'] = access
            else:
                postdata['access'] = 'public'

        resp = self._portal.con.post(url,
                                     postdata, files=files)
        if 'success' in resp:
            return resp['success']
        return resp

    def delete(self, key):
        """
        The Remove Resource operation allows the administrator to remove
        a file resource.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        key               optional string, look up key for file to delete
        ================  ===============================================================

        :returns:
           boolean
        """
        postdata = {
            "f" : "json",
            "key" : key,
            }
        resp = self._portal.con.post('portals/self/removeresource',
                                     postdata)
        if 'success' in resp:
            return resp['success']
        return resp

    #----------------------------------------------------------------------
    def list(self, start=1, num=100):
        """
        returns a list of resources uploaded to portal.  The items can be
        images, files and other content used to stylize and modify a
        portal's appearance.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        start             optional int, start location of the search. The default is
                          a value of 1
        ----------------  ---------------------------------------------------------------
        num               optional int, the number of search results to return at one
                          time. The value ranges between 1-100 (max).
                          Default: 100
        ================  ===============================================================

        :returns:
           boolean
        """
        postdata = {
            "f" : "json",
            'start' : start,
            'num' : num
        }
        resp = self._portal.con.post('portals/self/resources',
                                     postdata)
        if 'resources' in resp:
            return resp['resources']
        return resp

    def get(self, resource_name, download_path=None):
        """
        Download or get a portal resource item

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        resource_name     optional string, key/name of data
        ----------------  ---------------------------------------------------------------
        download_path     optional string, save folder location
        ================  ===============================================================

        :return:
           path to data or raw data if not file.
        """
        data_path = 'portals/self/resources/' + resource_name
        if not download_path:
            download_path = self._workdir

        download_path = self._portal.con.get(path=data_path,
                                             file_name=resource_name,
                                             out_folder=download_path,
                                             try_json=False,
                                             force_bytes=False)
        return download_path
