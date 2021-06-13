import os
import ssl
import logging
from urllib.parse import urlparse
from ._common import BaseServer
from .._impl._con import Connection
from ._service import Service
from arcgis.gis import GIS

_log = logging.getLogger()
########################################################################
class ServicesDirectory(BaseServer):
    """
    Provides ArcGIS Server Services Directory access.

    The Services Directory allows you to browse the contents of an ArcGIS Server and obtain
    information that can be useful to endusers for developing applications, performing analysis, or
    cataloging services. The ServicesDirectory is a view of the ArcGIS Server REST API in a python
    class.

    With the Services Directory, you can do the following:
     - Browse the contents of the GIS Server and get service-level metadata
     - Navigate a series of links to view information about the services on your GIS Server.
     - Get information to help you develop applications

    When you develop applications with the Python API, you must provide URLs to services and the
    layers and functionality they expose. The Services Directory provides an interactive way for
    you to construct those URLs.

    The Services Directory works using REST. REST is an architectural style that allows ArcGIS
    Server to reveal a hierarchy of information about itself through endpoints, or URLs. When you
    use the Services Directory, you navigate through a series of links to discover information
    about the server. Each time you click a link, you see a new page that reveals additional
    information about what's available on the server. The information that you see on the page is
    retrieved through REST using the page's URL.


    =====================     ====================================================================
    **Arguments**             **Description**
    ---------------------     --------------------------------------------------------------------
    url                       string required. The web address to the ArcGIS Server administration
                              end point.

                              Example: https://mysite.com/arcgis

                              The URL should be formatted as follows:
                              <scheme>://<host>:<port (optional)>/<web adapter>
    ---------------------     --------------------------------------------------------------------
    baseurl                   optional string, the root URL to a site.
                              Example: https://mysite.com/arcgis
    ---------------------     --------------------------------------------------------------------
    tokenurl                  optional string. Used when a site if federated or when the token
                              URL differs from the site's baseurl.  If a site is federated, the
                              token URL will return as the Portal token and ArcGIS Server users
                              will not validate correctly.
    ---------------------     --------------------------------------------------------------------
    username                  optional string, login username for BUILT-IN security
    ---------------------     --------------------------------------------------------------------
    password                  optional string, a secret word or phrase that must be used to gain
                              access to the account above.
    ---------------------     --------------------------------------------------------------------
    key_file                  optional string, path to PKI ket file
    ---------------------     --------------------------------------------------------------------
    cert_file                 optional string, path to PKI cert file
    ---------------------     --------------------------------------------------------------------
    proxy_host                optional string, web address to the proxy host

                              Example: proxy.mysite.com
    ---------------------     --------------------------------------------------------------------
    proxy_port                optional integer, default is 80. The port where the proxy resided on
    ---------------------     --------------------------------------------------------------------
    expiration                optional integer. The Default is 60. This is the length of time a
                              token is valid for.
                              Example 1440 is one week.
    ---------------------     --------------------------------------------------------------------
    all_ssl                   optional boolean. The default is False. If True, all calls will be
                              made over HTTPS instead of HTTP.
    ---------------------     --------------------------------------------------------------------
    portal_connection         optional GIS. This is used when a site is federated. It is the
                              ArcGIS Online or Portal GIS object used.
    ---------------------     --------------------------------------------------------------------
    initialize                optional boolean.  The default is False.  If True, the object will
                              attempt to reach out to the URL resource and populate at creation
                              time.
    ---------------------     --------------------------------------------------------------------
    verify_cert               optional booean.  The defaut is True.  If False, any certificate
                              errors will be ignored.  This is useful for new sites or where site
                              that have invalid certificates.
    =====================     ====================================================================

    """
    _con = None
    _gis = None
    _url = None
    _adminurl = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url,
                 username=None,
                 password=None,
                 key_file=None,
                 cert_file=None,
                 verify_cert=False,
                 **kwargs):
        """Constructor"""
        super(ServicesDirectory, self)

        if url.lower().find('/rest') == -1 and \
           url.endswith('/rest') == False:
            url = "%s/rest/services" % url
        if url.lower().find('/services') == -1 and \
           url.lower().endswith('/services') == False:
            url = "%s/services" % url
        self._url = url
        self._username = username
        self._password = password
        self._key_file = key_file
        self._cert_file = cert_file
        self._portal_connection = kwargs.pop('portal_connection', None)
        self._is_agol = kwargs.pop("is_agol", False)
        con = kwargs.pop('con', None)
        if verify_cert == False:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
        aurl = None
        if 'admin_url' in kwargs:
            aurl = kwargs.pop('admin_url', None)
        if aurl is None:
            parsed = urlparse(url)
            wa = parsed.path[1:].split('/')[0]
            self._adminurl = "%s://%s/%s/admin" % (parsed.scheme, parsed.netloc, wa)
        else:
            self._adminurl = aurl

        if self._is_agol and self._portal_connection:
            if isinstance(self._portal_connection, GIS):
                self._con = self._portal_connection._portal.con
            elif hasattr(self._portal_connection, 'post'):
                self._con = self._portal_connection
        elif con:
            self._con = con
        else:
            self._con = Connection(baseurl=url,
                                   username=username,
                                   password=password,
                                   key_file=key_file,
                                   cert_file=cert_file,
                                   portal_connection=self._portal_connection,
                                   verify_cert=verify_cert,
                                   **kwargs)
        self._gis = kwargs.pop('gis', None)
        if self._is_agol == False and self._con._auth.lower() != "anon":
            try:
                from .admin.administration import Server
                self.admin = Server(gis=self._con,
                                    url=self._adminurl,
                                    servicesdirectory=self,
                                    initialize=False)
            except: pass
        self._init(self._con)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self.url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self.url)
    #----------------------------------------------------------------------
    def report(self, as_html=True, folder=None):
        """
        Generates a table of Services in the given folder, as a Pandas dataframe.


        """
        import pandas as pd
        pd.set_option('display.max_colwidth', -1)
        data = []
        a_template = """<a href="%s?token=%s">URL Link</a>"""
        columns = ['Service Name', 'Service URL']
        if folder is None:
            res = self._con.get(self._url, {"f" : 'json'})
        elif folder.lower() in [f.lower() for f in self.folders]:
            res = self._con.get("%s/%s" % (self._url, folder), {"f" : 'json'})
        if 'services' in res:
            for s in res['services']:
                #if s['name'].split('/')[-1].lower() == name.lower():
                url = "%s/%s/%s" % (self._url,
                                  s['name'],
                                  s['type'])
                data.append([s['name'].split('/')[-1], """<a href="%s">Service</a>""" % url])
        #for service in self.list(folder=folder):
            #name = os.path.basename(os.path.dirname(service._url))
            #data.append([name, a_template % (service._url, self._con.token)])
            #del service
        df = pd.DataFrame(data=data, columns=columns)
        if as_html:
            table = \
                """<div class="9item_container" style="height: auto; overflow: hidden; """ + \
                """border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; """ + \
                """line-height: 1.21429em; padding: 10px;">%s</div>""" % df.to_html(escape=False,
                                                                                    index=False)
            return table.replace('\n', '')
        else:
            return df

    #----------------------------------------------------------------------
    def get(self, name, folder=None):
        """returns a single service in a folder"""
        if folder is None:
            res = self._con.get(self._url, {"f" : 'json'})
        elif folder.lower() in [f.lower() for f in self.folders]:
            res = self._con.get("%s/%s" % (self._url, folder), {"f" : 'json'})
        if 'services' in res:
            for s in res['services']:
                if s['name'].split('/')[-1].lower() == name.lower():
                    return Service(url="%s/%s/%s" % (self._url,
                                                     s['name'],
                                                     s['type']),
                                   server=self._con)
                del s
        return None
    #----------------------------------------------------------------------
    def list(self, folder=None):
        """
        returns a list of services at the given folder
        """
        services = []
        if folder is None:
            res = self._con.get(self._url, {"f" : 'json'})
        elif folder.lower() in [f.lower() for f in self.folders]:
            res = self._con.get("%s/%s" % (self._url, folder), {"f" : 'json'})
        if 'services' in res:
            for s in res['services']:
                try:
                    services.append(Service(url="%s/%s/%s" % (self._url,
                                                              s['name'],
                                                              s['type']),
                                            server=self._con))

                except:
                    url ="%s/%s/%s" % (self._url, s['name'], s['type'])
                    _log.warning("Could not load service: %s" % url)
        return services
    #----------------------------------------------------------------------
    def find(self, service_name, folder=None):
        """
        finds a service based on it's name in a given folder
        """
        return self.get(name=service_name, folder=folder)
    #----------------------------------------------------------------------
    @property
    def folders(self):
        """
        returns a list of server folders
        """
        self._init()
        if self._is_agol:
            return ['/']
        else:
            return self.properties['folders']
        return []