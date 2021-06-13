"""
The **gis** module provides an information model for GIS hosted
within ArcGIS Online or ArcGIS Enterprise.
This module provides functionality to manage
(create, read, update and delete) GIS users, groups and content. This module
is the most important and provides the entry point into the GIS.
"""
from __future__ import absolute_import
import base64
import json
import locale
import sys
import os
import re
import tempfile
import zipfile
import configparser
from contextlib import contextmanager
import functools
from datetime import datetime
import logging
from typing import Tuple
from urllib.error import  HTTPError
from urllib.parse import urlparse
import concurrent.futures

import arcgis.env
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._utils import _DisableLogger
from arcgis.gis._impl._con._helpers import _is_http_url
from arcgis._impl.common._deprecate import deprecated
from ._impl import _portalpy

from ._impl._jb import StatusJob
_log = logging.getLogger(__name__)

class Error(Exception): pass

@contextmanager
def _tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write((bytes(data, 'UTF-8')))
    temp.close()
    yield temp.name
    os.unlink(temp.name)

def _lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = '_lazy_' + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

try:
    from arcgis.features.geo import _is_geoenabled
except:
    def _is_geoenabled(o):
        return False

class GIS(object):
    """
    .. _gis:

    A GIS is representative of a single ArcGIS Online organization or an ArcGIS Enterprise deployment. The GIS object
    provides helper objects to manage (search, create, retrieve) GIS resources such as content, users, and groups.

    Additionally, the GIS object has properties to query its state, which is accessible using the properties attribute.

    The GIS provides a mapping widget that can be used in the Jupyter Notebook environment for visualizing GIS content
    as well as the results of your analysis. To create a new map, call the map() method.

    The constructor constructs a GIS object given a url and user credentials to ArcGIS Online
    or an ArcGIS Enterprise Portal. User credentials can be passed in using username/password
    pair, or key_file/cert_file pair (in case of PKI). Supports built-in users, LDAP, PKI, Integrated Windows Authentication
    (using NTLM and Kerberos) and Anonymous access.

    If no url is provided, ArcGIS Online is used. If username/password
    or key/cert files are not provided, the currently logged-in user's credentials (IWA) or anonymous access is used.

    Persisted profiles for the GIS can be created by giving the GIS authorization credentials and
    specifying a profile name. The profile stores all of the authorization credentials (except the password) in the
    user's home directory in an unencrypted config file named .arcgisprofile. The profile securely stores the password
    in an O.S. specific password manager through the `keyring <https://pypi.python.org/pypi/keyring>`_ python module.
    (Note: Linux systems may need additional software installed and configured for proper security) Once a profile has
    been saved, passing the profile parameter by itself uses the authorization credentials saved in the configuration
    file/password manager by that profile name. Multiple profiles can be created and used in parallel.

    See https://developers.arcgis.com/python/guide/working-with-different-authentication-schemes/ for examples.


    ================    ===============================================================
    **Argument**        **Description**
    ----------------    ---------------------------------------------------------------
    url                 Optional string. If URL is None, then the URL will be ArcGIS
                        Online.  This should be a web address to either a local Portal
                        or to ArcGIS Online in the form:
                        <scheme>://<fully_qualified_domain_name>/<web_adaptor> (Portal Example)
                        https://gis.example.com/portal
    ----------------    ---------------------------------------------------------------
    username            Optional string. The login user name (case-sensitive).
    ----------------    ---------------------------------------------------------------
    password            Optional string. If a username is provided, a password is
                        expected.  This is case-sensitive. If the password is not
                        provided, the user is prompted in the interactive dialog.
    ----------------    ---------------------------------------------------------------
    key_file            Optional string. The file path to a user's key certificate for PKI
                        authentication
    ----------------    ---------------------------------------------------------------
    cert_file           Optional string. The file path to a user's certificate file for PKI
                        authentication. If a PFX or P12 certificate is used, a password is required.
                        If a PEM file is used, the key_file is required.
    ----------------    ---------------------------------------------------------------
    verify_cert         Optional boolean. If a site has an invalid SSL certificate or is
                        being accessed via the IP or hostname instead of the name on the
                        certificate, set this value to False.  This will ensure that all
                        SSL certificate issues are ignored.
                        The default is True.
                        **Warning** Setting the value to False can be a security risk.
    ----------------    ---------------------------------------------------------------
    set_active          Optional boolean. The default is True.  If True, the GIS object
                        will be used as the default GIS object throughout the whole
                        scripting session.
    ----------------    ---------------------------------------------------------------
    client_id           Optional string. Used for OAuth authentication.  This is the
                        client ID value.
    ----------------    ---------------------------------------------------------------
    profile             Optional string. the name of the profile that the user wishes to use
                        to authenticate, if set, the identified profile will be used to login
                        to the specified GIS.
    ================    ===============================================================

    In addition to explicitly named parameters, the GIS object supports optional key word
    arguments:

    ================    ===============================================================
    **kwargs**          **Description**
    ----------------    ---------------------------------------------------------------
    proxy_host          Optional string. The host name of the proxy server used to allow HTTP/S
                        access in the network where the script is run.

                        ex: 127.0.0.1
    ----------------    ---------------------------------------------------------------
    proxy_port          Optional integer. The proxy host port.  The default is 80.
    ----------------    ---------------------------------------------------------------
    token               Optional string. This is the Enterprise token for built-in
                        logins. This parameter is only honored if the username/password
                        is None and the security for the site uses BUILT-IN security.
    ----------------    ---------------------------------------------------------------
    api_key             Optional string.  This is a key generated by the developer site
                        to allow for a limited subset of the REST API functionality.
    ----------------    ---------------------------------------------------------------
    trust_env           Optional Boolean. Trust environment settings for proxy
                        configuration, default authentication and similar. If `False`
                        the GIS class will ignore the `netrc` files defined on the
                        system.
    ================    ===============================================================




    .. code-block:: python

        # Usage Example 1: Anonymous Login to ArcGIS Online

        gis = GIS()

    .. code-block:: python

        # Usage Example 2: Built-in Login to ArcGIS Online

        gis = GIS(username="someuser", password="secret1234")

    .. code-block:: python

        # Usage Example 3: Built-in Login to ArcGIS Enterprise

        gis = GIS(url="http://pythonplayground.esri.com/portal",
              username="user1", password="password1")

    .. code-block:: python

        # Usage Example 4: Built-in Login to ArcGIS Enterprise, ignoring SSL errors

        gis = GIS(url="http://pythonplayground.esri.com/portal", username="user1",
                  password="password1", verify_cert=False)

    .. code-block:: python

        # Usage Example 5: Anonymous ArcGIS Online Login with Proxy

        gis = GIS(proxy_host='127.0.0.1', proxy_port=8888)

    .. code-block:: python

        # Usage Example 6: PKI Login to ArcGIS Enterprise, using PKCS12 user certificate

        gis = GIS(url="https://pkienterprise.esri.com/portal",
                  cert_file="C:\\users\\someuser\\mycert.pfx", password="password1")

    """
    _server_list = None
    _is_hosted_nb_home = False
    _product_version = None
    _is_agol = None
    _pds = None
    """If 'True', the GIS instance is a GIS('home') from hosted nbs"""
    # admin = None
    # oauth = None
    def __init__(self, url=None, username=None, password=None, key_file=None, cert_file=None,
                 verify_cert=True, set_active=True, client_id=None, profile=None, **kwargs):
        """
        Constructs a GIS object given a url and user credentials to ArcGIS Online
        or an ArcGIS Portal. User credentials can be passed in using username/password
        pair, or key_file/cert_file pair (in case of PKI). Supports built-in users, LDAP,
        PKI, Integrated Windows Authentication (using NTLM and Kerberos) and Anonymous access.

        If no url is provided, ArcGIS Online is used. If username/password
        or key/cert files are not provided, logged in user credentials (IWA) or anonymous access is used.

        Persisted profiles for the GIS can be created by giving the GIS authorization credentials and
        specifying a profile name. The profile stores all of the authorization credentials (except the password) in the
        user's home directory in an unencrypted config file named .arcgisprofile. The profile securely stores the password
        in an O.S. specific password manager through the `keyring <https://pypi.python.org/pypi/keyring>`_ python module.
        (Note: Linux systems may need additional software installed and configured for proper security) Once a profile has
        been saved, passing the profile parameter by itself uses the authorization credentials saved in the configuration
        file/password manager by that profile name. Multiple profiles can be created and used in parallel.

        If the GIS uses a secure (https) url, certificate verification is performed. If you are using self signed certificates
        in a testing environment and wish to disable certificate verification, you may specify verify_cert=False to disable
        certificate verification in the Python process. However, this should not be done in production environments and is
        strongly discouraged.
        """
        self._proxy_host = kwargs.pop('proxy_host', None)
        self._proxy_port = kwargs.pop('proxy_port', 80)
        self._referer = kwargs.pop('referer', None)
        custom_auth = kwargs.pop('custom_auth', None)
        self._expiration = kwargs.pop('expiration', None)
        from arcgis._impl.tools import _Tools
        if profile is not None and \
           len(profile) == 0:
            raise ValueError("A `profile` name must not be an empty string.")
        elif profile is not None:
            # Load config
            pm = ProfileManager()

            cfg_file_path = pm._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
                # Check if config file is in the old format
                if not self._config_is_in_new_format(config):
                    answer = input("Warning: profiles in {} appear to be in "\
                                   "the <v1.3 format, and must be deleted before "\
                        "continuing. Delete? [y/n]".format(cfg_file_path))
                    if "y" in answer.lower():
                        os.remove(cfg_file_path)
                        config = configparser.ConfigParser()
                    else:
                        raise RuntimeError("{} not deleted, exiting"\
                                           "".format(cfg_file_path))

            # Add any __init__() args to config/keyring store
            if profile not in pm.list():
                _log.info("Adding new profile {} to config...".format(profile))
                pm.create(profile=profile, url=url, username=username, password=password,
                          key_file=key_file, cert_file=cert_file, client_id=client_id)
            elif profile in pm.list():
                # run an update to be safe.
                pm.update(profile, url=url, username=username, password=password,
                          key_file=key_file, cert_file=cert_file, client_id=client_id)
            if profile in pm.list(): # check if the profile name was successfully added, if so, use the profile credentials
                url, username, password, key_file, cert_file, client_id = pm._retrieve(profile)
            else:
                _log.info(f"Profile {profile} was not saved, using user provided credentials for the `GIS` object.")

        if url is None:
            url = "https://www.arcgis.com"
        if (self._uri_validator(url) == False) and \
           (str(url).lower() not in ['pro', 'home']):
            raise Exception("Malformed url provided: %s" % url)
        if username is not None and password is None:
            from getpass import getpass
            password = getpass('Enter password: ')
        # Assumes PFX is being passed in cert_file parameter and no key_file is specified
        if (cert_file is not None) and (key_file is None):
            if (cert_file.lower().endswith(".pfx") or cert_file.lower().endswith(".p12")):
                if password is None:
                    from getpass import getpass
                    password = getpass('Enter PFX password: ')
                key_file, cert_file = self._pfx_to_pem(cert_file, password)
            else:
                raise Exception("key_file parameter is required along with cert_file when using PKI authentication.")

        self._url = url
        self._username = username
        self._password = password
        self._key_file = key_file
        self._cert_file = cert_file
        self._portal = None
        self._con = None
        self._verify_cert = verify_cert
        self._client_id = client_id
        self._datastores_list = None
        self._utoken = kwargs.pop('token', None)
        client_secret = kwargs.pop('client_secret', None)
        if self._username is None:
            if "ESRI_API_KEY" in os.environ and self._utoken is None:
                self._utoken = os.environ.get("ESRI_API_KEY", None)
            elif self._utoken is None and not "ESRI_API_KEY" in os.environ:
                self._utoken = kwargs.pop('api_key', None)

        if self._url.lower() == "home" and \
           not os.getenv('NB_AUTH_FILE', None) is None:
            #configuring for hosted notebooks need to happen before portalpy
            self._try_configure_for_hosted_nb()
            if self._expiration is None:
                self._expiration = 10080
        elif self._url.lower() == "home" and \
             os.getenv('NB_AUTH_FILE', None) is None:
            self._url = "pro"
            url = "pro"
        elif self._expiration is None: # Keep Default Value
            self._expiration = 60
        try:
            self._portal = _portalpy.Portal(self._url, self._username,
                                           self._password, self._key_file,
                                           self._cert_file,
                                           proxy_host=self._proxy_host,
                                           proxy_port=self._proxy_port,
                                           verify_cert=self._verify_cert,
                                           client_id=self._client_id,
                                           expiration=self._expiration,
                                           referer=self._referer,
                                           custom_auth=custom_auth, #token=self._utoken,
                                           client_secret=client_secret,
                                           trust_env=kwargs.get("trust_env", None))
            if self._is_hosted_nb_home:
                # For GIS("home") objects, force no referer passed in
                self._portal.con._referer = ""
                self._portal.con._session.headers.pop("Referer", None)
            if not (self._utoken is None):
                self._portal.con._token = self._utoken
                self._portal.con.token = self._utoken
                self._portal.con._auth = "HOME"

        except Exception as e:
            if len(e.args) > 0 and str(type(e.args[0])) == "<class 'ssl.SSLError'>":
                raise RuntimeError("An untrusted SSL error occurred when attempting to connect to the provided GIS.\n"
                                   "If you trust this server and want to proceed, add 'verify_cert=False' as an "
                                   "argument when connecting to the GIS.")
            else:
                raise e
        try:
            if url.lower().find("arcgis.com") > -1 and \
               self._portal.is_logged_in and \
               self._portal.con._auth.lower() == 'oauth':
                from urllib.parse import urlparse
                props = self._portal.get_properties(force=False)
                url = "%s://%s.%s" % (urlparse(self._url).scheme,
                                      props['urlKey'],
                                      props['customBaseUrl'])
                self._url = url
                self._portal.resturl = self._portal.resturl.replace(self._portal.url,
                                                                    url)
                self._portal.url = url
                self._portal.con.baseurl = self._portal.resturl
                if self._portal.con._auth != "OAUTH":
                    self._portal.con._token = None
            elif url.lower().find("arcgis.com") > -1 and \
                 self._portal.is_logged_in:
                from urllib.parse import urlparse
                props = self._portal.get_properties(force=False)
                url = "%s://%s.%s" % (urlparse(self._url).scheme,
                                      props['urlKey'],
                                      props['customBaseUrl'])
                self._url = url
                pp =  _portalpy.Portal(url,
                                      self._username,
                                      self._password,
                                      self._key_file,
                                      self._cert_file,
                                      verify_cert=self._verify_cert,
                                      client_id=self._client_id,
                                      proxy_port=self._proxy_port,
                                      proxy_host=self._proxy_host,
                                      expiration=self._expiration,
                                      referer=self._referer,
                                      custom_auth=custom_auth,
                                      #token=self._utoken,
                                      trust_env=kwargs.get("trust_env", None),
                                      client_secret=client_secret)
                self._portal = pp
        except: pass

        force_refresh = False
        if not (self._utoken is None) and self._portal.con._auth != "HOME":
            self._portal.con._token = self._utoken
            self._portal.con._auth = "BUILTIN"
            force_refresh = True
        elif self._portal.con._auth == "HOME":
            force_refresh = True

        # If a token was injected, then force refresh to get updated properties
        self._lazy_properties = PropertyMap(self._portal.get_properties(force=force_refresh))

        self._con = self._portal.con

        if self._url.lower() == "pro":
            self._url = self._portal.url
            self._con._auth = "PRO"

        if self._con._auth != 'anon':
            me = self.users.me

        if self._con._auth.lower() != 'anon' and \
           self._con._auth is not None and \
           hasattr(me, 'role') and \
           me.role == "org_admin":
            try:
                if self._is_hosted_nb_home:
                    import warnings
                    warnings.warn("You are logged on as %s with an administrator role, proceed with caution." % \
                                  self.users.me.username)
                if self.properties.isPortal == True:
                    from arcgis.gis.admin.portaladmin import PortalAdminManager
                    self.admin = PortalAdminManager(url="%s/portaladmin" % self._portal.url,
                                                    gis=self)
                else:
                    from .admin.agoladmin import AGOLAdminManager
                    self.admin = AGOLAdminManager(gis=self)
            except Exception as e:
                pass
        elif self._con._auth.lower() != 'anon' and \
             self._con._auth is not None and\
             hasattr(me, 'role') and \
             me.role == 'org_publisher' and \
             self._portal.is_arcgisonline == False:
            try:
                from .admin.portaladmin import PortalAdminManager
                self.admin = PortalAdminManager(url="%s/portaladmin" % self._portal.url,
                                                gis=self, is_admin=False)
            except:
                pass
        elif self._con._auth.lower() != 'anon' and \
             self._con._auth is not None and\
             hasattr(me, 'privileges') and \
             self._portal.is_arcgisonline == False:
            privs = ['portal:publisher:publishFeatures',
                     'portal:publisher:publishScenes',
                     'portal:publisher:publishServerGPServices',
                     'portal:publisher:publishServerServices',
                     'portal:publisher:publishTiles']
            for priv in privs:
                if priv in me.privileges:
                    can_publish = True
                    break
                else:
                    can_publish = False
            if can_publish:
                try:
                    from .admin.portaladmin import PortalAdminManager
                    self.admin = PortalAdminManager(url="%s/portaladmin" % self._portal.url,
                                                    gis=self, is_admin=False)
                except:
                    pass
        if self._con._auth.lower() != 'anon' and \
           self._con._auth is not None and\
           hasattr(me, 'role') and \
           me.role == 'org_publisher' and \
           self._portal.is_arcgisonline == False:
            try:
                from .admin.portaladmin import PortalAdminManager
                self.admin = PortalAdminManager(url="%s/portaladmin" % self._portal.url,
                                                gis=self, is_admin=False)
            except:
                pass
        self._tools = _Tools(self)
        if set_active:
            arcgis.env.active_gis = self
        if self._product_version is None:
            self._is_agol = self._portal.is_arcgisonline
            self._product_version = [int(i) for i in self._portal.get_version().split('.')]
    #----------------------------------------------------------------------
    @_lazy_property
    def api_keys(self):
        """
        Returns the `APIKeyManager` which allows User to generate, manage
        and modify API Keys for controlled application access.

        **The API Key manager is only available for ArcGIS Online**

        :returns: APIKeyManager

        """
        if self._portal.is_arcgisonline and self.version >= [8,2]:
            from arcgis.gis._impl._apikeys import APIKeyManager
            return APIKeyManager(self)
        return None
    #----------------------------------------------------------------------
    def _private_service_url(self, service_url):
        """
        returns the public and private URL for a given registered service

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        service_url         Required string.  The URL to the service.
        ===============     ====================================================================

        :return: dict

        """
        if self.version < [5,3]:
            return { "serviceUrl" : service_url }
        url = ("{base}portals/self"
               "/servers/computePrivateServiceUrl").format(
                   base=self._portal.resturl)
        params = {
            'f' : 'json',
            'serviceUrl' : service_url
        }

        return self._con.post(url, params)

    #----------------------------------------------------------------------
    def _pfx_to_pem(self, pfx_path, pfx_password):
        """ Decrypts the .pfx file to be used with requests.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        pfx_path            Required string.  File pathname to .pfx file to parse.
        ---------------     --------------------------------------------------------------------
        pfx_password        Required string.  Password to open .pfx file to extract key/cert.
        ===============     ====================================================================

        :return:
           File path to key_file located in a tempfile location
           File path to cert_file located in a tempfile location
        """
        try:
            import OpenSSL.crypto
        except:
            raise RuntimeError("OpenSSL.crypto library is not installed.  You must install this in order "+\
                               "to use a PFX for connecting to a PKI protected portal.")
        key_file = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
        cert_file = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
        k = open(key_file.name, 'wb')
        c = open(cert_file.name, 'wb')
        try:
            pfx = open(pfx_path, 'rb').read()
            p12 = OpenSSL.crypto.load_pkcs12(pfx, pfx_password)
        except OpenSSL.crypto.Error:
            raise RuntimeError("Invalid PFX password.  Unable to parse file.")
        k.write(OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, p12.get_privatekey()))
        c.write(OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, p12.get_certificate()))
        k.close()
        c.close()
        return key_file.name, cert_file.name

    def _config_is_in_new_format(self, config):
        """ Any version <= 1.3.0 of the API used a different config file
        formatting that, among other things, did not store the last time
        a profile was modified. Thus, if 'date_modified' is found in at least
        one profile, it is in the new format
        """
        return any([profile_data for profile_data in config.values() \
                    if "date_modified" in profile_data])

    def _try_configure_for_hosted_nb(self):
        """If 'home' is specified as the 'url' argument, this func is called"""
        try:
            #Set relevant properties and overrides
            self._is_hosted_nb_home = True
            self._verify_cert = False

            # Get the auth file from environment variables
            nb_auth_file_path = os.getenv('NB_AUTH_FILE', None)
            if not nb_auth_file_path:
                raise RuntimeError("Environment variable 'NB_AUTH_FILE' "\
                                   "must be defined.")
            elif not os.path.isfile(nb_auth_file_path):
                raise RuntimeError("'{}' file needed for "\
                                   "authentication not found.".format(nb_auth_file_path))
            # Open that auth file,
            with open(nb_auth_file_path) as nb_auth_file:
                required_json_keys = set(["privatePortalUrl",
                                          "publicPortalUrl", "referer"])
                json_data = json.load(nb_auth_file)
                assert required_json_keys.issubset(json_data)
                self._url = json_data["privatePortalUrl"]
                self._public_portal_url = json_data["publicPortalUrl"]
                if "token" in json_data:
                    self._utoken = json_data["token"]
                self._expiration = json_data.get("expiration", None)
                if "encryptedToken" in json_data:
                    from arcgis.gis._impl._decrypt_nbauth import get_token
                    self._utoken = get_token(nb_auth_file_path)

        # Catch errors and re-throw in with more human readable messages
        except json.JSONDecodeError as e:
            self._raise_hosted_nb_error("'{}' file is not "\
                                        "valid JSON.".format(nb_auth_file.name))
        except AssertionError as e:
            self._raise_hosted_nb_error("Authentication file doesn't contain "\
                                        "required keys {}".format(required_json_keys))
        except Exception as e:
            self._raise_hosted_nb_error("Unexpected exception when authenticating "\
                                        "through 'home' mode: {}".format(e))

    def _raise_hosted_nb_error(self, err_msg):
        """In the event a user can't authenticate in 'home' mode, raise
        an error while also giving a simple mitigation technique of connecting
        to your portal in the standard GIS() way.
        """
        mitigation_msg =  "You can still connect to your portal by creating "\
            "a GIS() object with the standard user/password, cert_file, etc. "\
            "See https://bit.ly/2DT1156 for more information."
        _log.warning('Authenticating in GIS("home") mode failed.'\
                    '{}'.format(mitigation_msg))
        raise RuntimeError("{}\n-----\n{}".format(err_msg, mitigation_msg))

    def _uri_validator(self, x):
        from urllib.parse import urlparse
        if x is None:
            return False
        try:
            result = urlparse(x)
            return result.scheme != "" and result.netloc != ""
        except:
            return False

    @_lazy_property
    def users(self):
        """
        The resource manager for GIS users. See :class:`~arcgis.gis.UserManager`.
        """
        return UserManager(self)


    @_lazy_property
    def groups(self):
        """
        The resource manager for GIS groups. See :class:`~arcgis.gis.GroupManager`.
        """
        return GroupManager(self)

    @_lazy_property
    def content(self):
        """
        The resource manager for GIS content. See :class:`~arcgis.gis.ContentManager`.
        """
        return ContentManager(self)

    @_lazy_property
    def hub(self):
        """
        The resource manager for GIS hub. See :class:`~arcgis.apps.hub.Hub`.
        """
        if self._portal.is_arcgisonline:
            return arcgis.apps.hub.Hub(self)
        else:
            raise Exception("Hub is currently only compatible with ArcGIS Online.")

    @property
    def datastore(self):
        """
        The resource managers for GIS datastores. This is only available with Enterprises version 10.7+.
        See :class:`~arcgis.gis._impl._datastores.PortalDataStore` for more information.

        :return: :class:`~arcgis.gis._impl._datastores.PortalDataStore`

        """
        if self.version >= [7,1] and not self._portal.is_arcgisonline:
            from arcgis.gis._impl._datastores import PortalDataStore
            url = self._portal.resturl + "portals/self/datastores"
            self._pds = PortalDataStore(url=url, gis=self)
        return self._pds

    @_lazy_property
    def _datastores(self):
        """
        The list of datastores resource managers for sites federated with the GIS.
        """
        if self._datastores_list is not None:
            return self._datastores_list

        self._datastores_list = []
        try:
            res = self._portal.con.post("portals/self/servers", {"f": "json"})

            servers = res['servers']
            admin_url = None
            for server in servers:
                admin_url = server['adminUrl'] + '/admin'
                self._datastores_list.append(DatastoreManager(self, admin_url, server))
        except:
            pass
        return self._datastores_list

    @_lazy_property
    def properties(self):
        """
        The properties of the GIS.
        """
        return PropertyMap(self._get_properties(force=True))

    def update_properties(self, properties_dict):
        """Updates the GIS's properties from those in properties_dict. This method can be useful
        for updating the utility services used by the GIS.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        properties_dict     Required dictionary. A dictionary of just those properties and
                            values that are to be updated.
        ===============     ====================================================================

        :return:
           True if successfully updated, False if unsuccessful.


        .. note::
            For examples of the property names and key/values to use when updating utility services,
            refer to the Portal parameters section at https://developers.arcgis.com/rest/users-groups-and-items/common-parameters.htm

        .. code-block:: python

            # Usage Example: Update the geocode service

            gis = GIS(profile='xyz')
            upd = {'geocodeService': [{
              "singleLineFieldName": "Single Line Input",
              "name": "AtlantaLocator",
              "url": "https://some.server.com/server/rest/services/GeoAnalytics/AtlantaLocator/GeocodeServer",
              "itemId": "abc6e1fc691542938917893c8944606d",
              "placeholder": "",
              "placefinding": "true",
              "batch": "true",
              "zoomScale": 10000}]}

            gis.update_properties(upd)

        """
        postdata = self._portal._postdata()
        postdata.update(properties_dict)

        resp = self._portal.con.post('portals/self/update', postdata)
        if resp:
            self._lazy_properties = PropertyMap(self._portal.get_properties(force=True))
            # delattr(self, '_lazy_properties') # force refresh of properties when queried next
            return resp.get('success')

    @property
    def url(self):
        """Readonly URL of the GIS you are connected to."""
        if self._is_hosted_nb_home:
            return self._public_portal_url
        else:
            return self._url

    @property
    def _public_rest_url(self):
        return self.url + "/sharing/rest/"
    #----------------------------------------------------------------------
    @property
    def _subscription_information(self):
        """
        Returns the ArcGIS Online Subscription Information for a Site.

        :returns: dictionary
        """
        if self.version > [6,4] and \
           self._portal.is_arcgisonline:
            url = "%sportals/self/subscriptionInfo" % self._portal.resturl
            params = {'f': 'json'}
            return self._con.get(url, params)
        return None
    #----------------------------------------------------------------------
    @property
    def version(self):
        """returns the GIS version number"""
        self._is_agol = self._portal.is_arcgisonline
        self._product_version = [int(i) for i in self._portal.get_version().split('.')]
        return self._product_version
    #----------------------------------------------------------------------
    @property
    def org_settings(self):
        """
        The portal settings resource is used to return a view of the
        portal's configuration as seen by the current users, either
        anonymous or logged in. Information returned by this resource
        includes helper services, allowed redirect URIs, and the current
        configuration for any access notices or information banners.

        ======================     ===============================================================
        **Parameters**             **Description**
        ----------------------     ---------------------------------------------------------------
        settings                   Required Dict.  A dictionary of the settings

                                    ==========================    =============================================
                                    **Fields**                    **Description**
                                    --------------------------    ---------------------------------------------
                                    anonymousAccessNotice         Dict. A JSON object representing a notice that is shown to your organization's anonymous users.
                                                                  Ex: {'title': 'Anonymous Access Notice Title', 'text': 'Anonymous Access Notice Text', 'buttons': 'acceptAndDecline', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    authenticatedAccessNotice     Dict. A JSON object representing a notice that is shown to your organization's authenticated users.
                                                                  Ex: {'title': 'Authenticated Access Notice Title', 'text': 'Authenticated Access Notice Text', 'buttons': 'okOnly', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    informationalBanner           Dict. A JSON object representing the informational banner that is shown at the top of your organization's page.
                                                                  Ex: {'text': 'Header Text', 'bgColor': 'grey', 'fontColor': 'blue', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    clearEmptyFields              Bool.  If True, any empty dictionary will be set to null.
                                    ==========================    =============================================

        ======================     ===============================================================

        :returns: Dictionary

        """
        if self.version >= [7,4]:
            url = "portals/self/settings"
            params = {'f' : 'json'}
            return self._con.post(url, params)
        return
    #----------------------------------------------------------------------
    @org_settings.setter
    def org_settings(self, settings):
        """
        This operation allows you to enable and customize an access notice
        and informational banner for your organization. The access notice,
        for authenticated and anonymous access, acts as a terms of service
        that users must agree to before being able to access the portal
        site. The informational banner allows you to alert members of your
        organization about your site's current status and content, such as
        a notice that the site is currently in read-only mode or
        containing content of a specific classification level.

        ======================     ===============================================================
        **Parameters**             **Description**
        ----------------------     ---------------------------------------------------------------
        settings                   Required Dict.  A dictionary of the settings

                                    ==========================    =============================================
                                    **Fields**                    **Description**
                                    --------------------------    ---------------------------------------------
                                    anonymousAccessNotice         Dict. A JSON object representing a notice that is shown to your organization's anonymous users.
                                                                  Ex: {'title': 'Anonymous Access Notice Title', 'text': 'Anonymous Access Notice Text', 'buttons': 'acceptAndDecline', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    authenticatedAccessNotice     Dict. A JSON object representing a notice that is shown to your organization's authenticated users.
                                                                  Ex: {'title': 'Authenticated Access Notice Title', 'text': 'Authenticated Access Notice Text', 'buttons': 'okOnly', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    informationalBanner           Dict. A JSON object representing the informational banner that is shown at the top of your organization's page.
                                                                  Ex: {'text': 'Header Text', 'bgColor': 'grey', 'fontColor': 'blue', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    clearEmptyFields              Bool.  If True, any empty dictionary will be set to null.
                                    ==========================    =============================================

        ======================     ===============================================================

        """
        if self.version >= [7,4] and \
           isinstance(settings, dict):
            url = "portals/self/settings/update"
            params = {'f' : 'json'}
            params.update(settings)
            self._con.post(url, params)
    #----------------------------------------------------------------------
    def __str__(self):
        return 'GIS @ {url} version:{version}'.format(url=self.url,
                                    version=".".join([str(i) for i in self._product_version]))
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    def _repr_html_(self):
        """
        HTML Representation for IPython Notebook
        """
        return 'GIS @ <a href="' + self.url + '">' + self.url + '</a>'
    #----------------------------------------------------------------------
    def _get_properties(self, force=False):
        """ Returns the portal properties (using cache unless force=True). """
        return self._portal.get_properties(force)

    def map(self, location=None, zoomlevel=None, mode="2D", geocoder=None):
        """
        Creates a map widget centered at the declared location with the specified
        zoom level. If an address is provided, it is geocoded
        using the GIS's configured geocoders and if a match is found, the geographic
        extent of the matched address is used as the map extent. If a zoomlevel is also
        provided, the map is centered at the matched address instead and the map is zoomed
        to the specified zoomlevel. See :class:`~arcgis.widgets.MapView` for more information.

        Note: The map widget is only supported within a Jupyter Notebook.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Optional string. The address or lat-long tuple of where the map is to be centered.
        ------------------     --------------------------------------------------------------------
        zoomlevel              Optional integer. The desired zoom level.
        ------------------     --------------------------------------------------------------------
        mode                   Optional string of either '2D' or '3D' to specify map mode. Defaults to '2D'.
        ------------------     --------------------------------------------------------------------
        geocoder               Optional Geocoder. Allows users to specify a geocoder to find a given location.
        ==================     ====================================================================


        :return:
          The map widget (displayed in Jupyter Notebook when queried).

        .. note::
            Note: If the Jupyter Notebook server is running over http, you need to
            configure your portal/organization to allow your host and port; or else
            you will run into CORs issues when displaying this map widget.

            This can be accomplished by signing into your portal/organization in a
            browser, then navigating to:

            `Organization` > `Settings` > `Security` > `Allow origins` > `Add` > http://localhost:8888 (replace with the host/port you are running on)
        """
        try:
            from arcgis.widgets import MapView
            from arcgis.geocoding import get_geocoders, geocode, Geocoder
        except Error as err:
            _log.error("ipywidgets packages is required for the map widget.")
            _log.error("Please install it:\n\tconda install ipywidgets")

        if isinstance(location, Item) and location.type == 'Web Map':
            mapwidget = MapView(gis=self, item=location, mode=mode)
        else:
            mapwidget = MapView(gis=self, mode=mode)

            # Geocode the location
            if isinstance(location, str):
                if geocoder and \
                   isinstance(geocoder, Geocoder):
                    locations = geocode(location, out_sr=4326, max_locations=1, geocoder=geocoder)
                    if len(locations) > 0:
                        if zoomlevel is not None:
                            loc = locations[0]['location']
                            mapwidget.center = loc['y'], loc['x']
                            mapwidget.zoom = zoomlevel
                        else:
                            mapwidget.extent = locations[0]['extent']
                else:
                    for geocoder in get_geocoders(self):
                        locations = geocode(location, out_sr=4326, max_locations=1, geocoder=geocoder)
                        if len(locations) > 0:
                            if zoomlevel is not None:
                                loc = locations[0]['location']
                                mapwidget.center = loc['y'], loc['x']
                                mapwidget.zoom = zoomlevel
                            else:
                                if 'extent' in locations[0]:
                                    mapwidget.extent = locations[0]['extent']
                            break

            # Center the map at the location
            elif isinstance(location, (tuple, list)):
                if all(isinstance(el, list) for el in location):
                    extent = {
                        'xmin': location[0][0],
                        'ymin': location[0][1],
                        'xmax': location[1][0],
                        'ymax': location[1][1]
                    }
                    mapwidget.extent = extent
                else:
                    mapwidget.center = location

            elif isinstance(location, dict): # geocode result
                if 'extent' in location and zoomlevel is None:
                    mapwidget.extent = location['extent']
                elif 'location' in location:
                    mapwidget.center = location['location']['y'], location['location']['x']
                    if zoomlevel is not None:
                        mapwidget.zoom = zoomlevel

            elif location is not None:
                print("location must be an address(string) or (lat, long) pair as a tuple")

        if zoomlevel is not None:
            mapwidget.zoom = zoomlevel

        return mapwidget

###########################################################################
class Datastore(dict):
    """
    Represents a datastore (folder, database or bigdata fileshare) within the GIS's data store.
    """
    def __init__(self, datastore, path):
        dict.__init__(self)
        self._datastore = datastore
        self._portal = datastore._portal
        self._admin_url = datastore._admin_url

        self.datapath = path


        params = { "f" : "json" }
        path = self._admin_url + "/data/items" + self.datapath

        datadict = self._portal.con.post(path, params, verify_cert=False)

        if datadict:
            self.__dict__.update(datadict)
            super(Datastore, self).update(datadict)

    def __getattr__(self, name): # support group attributes as group.access, group.owner, group.phone etc
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

    def __getitem__(self, k): # support group attributes as dictionary keys on this object, eg. group['owner']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            params = { "f" : "json" }
            path = self._admin_url + "/data/items" + self.datapath

            datadict = self._portal.con.post(path, params, verify_cert=False)
            super(Datastore, self).update(datadict)
            self.__dict__.update(datadict)
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    def __repr__(self):
        return '<%s title:"%s" type:"%s">' % (type(self).__name__, self.path, self.type)

    @property
    def manifest(self):
        """
        Gets or sets the manifest resource for bigdata fileshares, as a dictionary.
        """
        data_item_manifest_url = self._admin_url + '/data/items' + self.datapath + "/manifest"

        params = {
            'f': 'json',
        }
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)
        return res

    @manifest.setter
    def manifest(self, value):
        """
        Updates the manifest resource for bigdata file shares.
        """
        manifest_upload_url =  self._admin_url + '/data/items' + self.datapath + '/manifest/update'

        with _tempinput(json.dumps(value)) as tempfilename:
            # Build the files list (tuples)
            files = []
            files.append(('manifest', tempfilename, os.path.basename(tempfilename)))

            postdata = {
                'f' : 'pjson'
            }

            resp = self._portal.con.post(manifest_upload_url, postdata, files, verify_cert=False)

            if resp['status'] == 'success':
                return True
            else:
                print(str(resp))
                return False

    @property
    def ref_count(self):
        """
        Gets the total number of references to this data item that exists on the server. You can use this
        property to determine if this data item can be safely deleted or taken down for maintenance.
        """
        data_item_manifest_url = self._admin_url + '/data/computeTotalRefCount'

        params = {
            'f': 'json',
            'itemPath': self.datapath
        }
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)
        return res["totalRefCount"]

    def delete(self):
        """
        Unregisters this data item from the data store.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {
            "f" : "json" ,
            "itempath" : self.datapath,
            "force": True
        }
        path = self._admin_url + "/data/unregisterItem"

        resp = self._portal.con.post(path, params, verify_cert=False)
        if resp:
            return resp.get('success')
        else:
            return False

    def update(self, item):
        """
        Edits this data item to update its connection information.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required dictionary. The representation of the updated item.
        ===============     ====================================================================


        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = {
            "f" : "json" ,
            "item" : item
        }
        path = self._admin_url +  "/data/items" + self.datapath +  "/edit"

        resp = self._portal.con.post(path, params, verify_cert=False)
        if resp ['status'] == 'success':
            return True
        else:
            return False
    #----------------------------------------------------------------------
    def regenerate(self):
        """
        This regenerates the manifest for a big data file share. You can
        regenerate a manifest if you have added new data or if you have
        uploaded a hints file using the edit resource.

        :returns: Boolean. True = Success, False = Failure

        """
        url = self._admin_url + '/data/items' + self.datapath + "/manifest/regenerate"
        params = {'f' : 'json'}
        res = self._portal.con.post(url, params)
        if isinstance(res, dict):
            if 'success' in res:
                return res['success']
            if 'status' in res:
                return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def validate(self):
        """
        Validates that this data item's path (for file shares) or connection string (for databases)
        is accessible to every server node in the site.

        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = { "f" : "json" }
        path = self._admin_url + "/data/items" + self.datapath

        datadict = self._portal.con.post(path, params, verify_cert=False)

        params = {
            "f" : "json",
            "item": datadict
        }
        path = self._admin_url + "/data/validateDataItem"

        res = self._portal.con.post(path, params, verify_cert=False)
        if isinstance(res, dict):
            if 'success' in res:
                return res['success']
            if 'status' in res:
                return res['status'] == 'success'
        return res

    @property
    def datasets(self):
        """
        Gets the datasets in the data store, as a dictionary (currently implemented for big data file shares).
        """
        data_item_manifest_url = self._admin_url + '/data/items' + self.datapath + "/manifest"

        params = {
            'f': 'json',
        }
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)

        return res['datasets']
###########################################################################
class GroupMigrationManager(object):
    """
    This manager class allows groups to export and import data to and from EPK files.
    """
    _con = None
    _gis = None
    _group = None

    def __init__(self, group):
        """initializer"""
        assert isinstance(group, Group)
        self._group = group
        self._gis = group._gis
        self._con = group._gis._con
    #----------------------------------------------------------------------
    def _from_package(self,
                      item,
                      item_id_list=None,
                      preview_only=False,
                      run_async=False,
                      overwrite=False):
        """
        Imports an EPK Item to a Group.  This will import items associated with this group.

        :returns: Boolean
        """
        if self._gis.users.me.role == 'org_admin':
            try_json = True
            if preview_only:
                try_json = False
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/import"
            if isinstance(item, Item):
                item = item.itemid
            params = {
                'f' : 'json',
                "itemId": item,
            }
            if item_id_list:
                params['itemIdList'] = item_id_list
            if overwrite is not None:
                params['overwriteExistingItems'] = overwrite
            if preview_only:
                params['previewOnly'] = preview_only
            if run_async:
                params['async'] = run_async

            return self._con.post(url,
                                  params,
                                  try_json=try_json)

        else:
            raise Exception("Must be an administror to perform this action")
        pass
    #----------------------------------------------------------------------
    def _status(self, job_id, key=None):
        """
        Checks the status of an export job
        """
        import time
        params = {}
        if job_id:
            url = f"{self._gis._portal.resturl}portals/self/jobs/{job_id}"
            params['f'] = 'json'
            res = self._con.post(url, params)
            while res["status"] not in ["completed", "complete"]:
                res = self._con.post(url, params)
                if res['status'] == "failed":
                    raise Exception(res)
                time.sleep(2)
            return res
        else:
            raise Exception(res)
    #----------------------------------------------------------------------
    def create(self,
               items=None,
               future:bool=True):
        """
        Exports a `Group` content to a **EPK Package Item**.

        `EPK Items` are intended to migrate content from an enterprise deployment to a new
        enterprise. Once an `EPK Item` is created using this method, you can use the `load`
        to ingest the package's content into the target enterprise. If your package
        contains web maps, web-mapping applications, and/or associated web layers, during
        the import operation, the method will takes care of swizzling the service URLs and
        item IDs correctly.

        There are some limits to this functionality. Packages should be under 10 GB in size
        and only hosted feature layers, web maps, web-mapping apps, and other text-based
        items are supported. You need to have **administrative** privileges to run this
        operation.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        items                  Optional List<Item>. A set of items to export from the group.  If nothing is given, all items will be attempted to be exported.
        ------------------     --------------------------------------------------------------------
        future                 Optional Boolean.  When True, the operation will return a Job object and return the results asynchronously.
        ==================     ====================================================================

        :returns: Item --or-- Job when future=True

        """
        if self._gis.users.me.role == 'org_admin':
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/export"
            if items and isinstance(items, (list, tuple)):
                items = ",".join([i.id for i in items])
            else:
                items = None
            params = {
                      'itemIdList' : items
                      }

            params['async'] = json.dumps(True)
            res = self._gis._con.post(url, params)

            executor =  concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(self._status, **{"job_id" : res['jobId'], "key": res['key']})
            executor.shutdown(False)
            job = StatusJob(future=futureobj, op='Export Group Content', jobid=res['jobId'], gis=self._gis, notify=arcgis.env.verbose)
            if future:
                return job
            else:
                return job.result()
        else:
            raise Exception("Must be an administror to perform this action")
    #----------------------------------------------------------------------
    def load(self,
             epk_item,
             item_ids:list=None,
             overwrite:bool=True,
             future:bool=True):
        """
        Imports the EPK content into the current `Group`.

        Administrative privileges are required to run this operation.
        Once imported, items will be owned by the importer, and will have
        to be manually reassigned to the proper owner if needed.


        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        epk_item          Required Item. A report on the content of the EPK Item.  This allows administrators
                          to view the contents inside a EPK.
        ----------------  -------------------------------------------------------------------------------
        item_ids          Optional list. A list of item IDs to import to the organization.
        ----------------  -------------------------------------------------------------------------------
        overwrite         Optional bool. If the Items import exist, or the Item ID that is in use
                          already, it will delete the old item and replace it with this one.
        ----------------  -------------------------------------------------------------------------------
        future            Optional bool. When True, the `load` will return a `Job` object and will not
                          pause the current thread.  When `False` `load` will occur in a synchronous
                          fashion pausing the thread.  If you are loading large amounts of data, set
                          future to `True` to reduce time.
        ================  ===============================================================================

        :returns: dict --or-- Job when future=True

        """
        assert isinstance(epk_item, Item)
        if isinstance(epk_item, Item) and \
           epk_item.type == 'Export Package':
            res = self._from_package(item=epk_item,
                                      item_id_list=item_ids,
                                      preview_only=False,
                                      run_async=True,
                                      overwrite=overwrite)
            executor =  concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(self._status, **{"job_id" : res['jobId'], "key": res['key']})
            executor.shutdown(False)
            job = StatusJob(future=futureobj,
                            op='Export Group Content',
                            jobid=res['jobId'],
                            gis=self._gis, notify=arcgis.env.verbose)
            if future:
                return job
            else:
                return job.result()
        else:
            raise Exception(f"Invalid Item {epk_item.type}")
        return None
    #----------------------------------------------------------------------
    def inspect(self, epk_item) -> dict:
        """
        Returns the contents of the EPK Package

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        epk_item          Required Item. A report on the content of the EPK Item.  This allows administrators
                          to view the contents inside a EPK.

        ================  ===============================================================================

        :returns: dict

        """
        if isinstance(epk_item, Item) and epk_item.type == 'Export Package':
            try:
                import time
                self._from_package(epk_item.itemid, preview_only=True, run_async=False)
                time.sleep(2)
            except:
                pass
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/importPreview/{epk_item.itemid}"
            params = {"f" : "json", "start" : 1, "num" : 25}
            res = self._con.post(url, params)
            results = res['results']
            while res['nextStart'] > 0:
                params['start'] = res['nextStart']
                res = self._con.post(url, params)
                results.extend(res['results'])
                if res['nextStart'] == -1:
                    break
            res['results'] = results
            return res

        else:
            raise Exception("Invalid Item Type.")
        return None

###########################################################################
class DatastoreManager(object):
    """
    Helper class for managing the GIS data stores in on-premises ArcGIS Portals.
    This class is not created by users directly.
    Instances of this class are returned from arcgis.geoanalytics.get_datastores() and
    arcgis.raster.analytics.get_datastores() functions to get the corresponding datastores.
    Users call methods on this 'datastores' object to manage the datastores in a site
    federated with the portal.
    """
    def __init__(self, gis, admin_url, server):
        self._gis = gis
        self._portal = gis._portal
        self._admin_url = admin_url
        self._server = server

    def __str__(self):
        return '<%s for %s>' % (type(self).__name__, self._admin_url)

    def __repr__(self):
        return '<%s for %s>' % (type(self).__name__, self._admin_url)

    @property
    def config(self):
        """
        Gets or sets the data store configuration properties, which affect the behavior of the data holdings of the server. The properties include:
        blockDataCopy. When this property is False, or not set at all, copying data to the site when publishing services from a client application is allowed. This is the default behavior.
        When this property is True, the client application is not allowed to copy data to the site when publishing. Rather, the publisher is required to register data items through which the service being published can reference data. Values: True | False
        Note:
        If you specify the property as True, users will not be able to publish geoprocessing services and geocode services from composite locators. These service types require data to be copied to the server. As a workaround, you can temporarily set the property to False, publish the service, and then set the property back to True.
        """
        params = {"f" : "json"}
        path = self._admin_url + "/data/config"
        res = self._portal.con.post(path, params, verify_cert=False)
        return res

    @config.setter
    def config(self, value):
        """
        The data store configuration properties affect the behavior of the data holdings of the server. The properties include:
        blockDataCopy When this property is False, or not set at all, copying data to the site when publishing services from a client application is allowed. This is the default behavior.
        When this property is True, the client application is not allowed to copy data to the site when publishing. Rather, the publisher is required to register data items through which the service being published can reference data. Values: True | False
        Note:
        If you specify the property as True, users will not be able to publish geoprocessing services and geocode services from composite locators. These service types require data to be copied to the server. As a workaround, you can temporarily set the property to False, publish the service, and then set the property back to True.
        """
        params = {"f" : "json"}
        params['datastoreConfig'] = value
        path = self._admin_url + "/data/config/update"
        res = self._portal.con.post(path, params)
        return res

    def add_folder(self,
                   name,
                   server_path,
                   client_path=None):

        """
        Registers a folder with the data store.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique fileshare name on the server.
        ---------------     --------------------------------------------------------------------
        server_path         Required string. The path to the folder from the server (and client, if shared path).
        ---------------     --------------------------------------------------------------------
        client_path         Optional string. If folder is replicated, the path to the folder from the client.
        ===============     ====================================================================


        :return:
           The folder if registered successfully, None otherwise.
        """
        conn_type = "shared"
        if client_path is not None:
            conn_type = "replicated"

        item = {
            "type" : "folder",
            "path" : "/fileShares/" + name,
            "info" : {
                "path" : server_path,
                "dataStoreConnectionType" : conn_type
            }
        }

        if client_path is not None:
            item['clientPath'] = client_path

        params = {
            "f" : "json",
            "item" : item
        }
        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res['status'] == 'success' or res['status'] == 'exists':
            return Datastore(self, "/fileShares/" + name)
        else:
            print(str(res))
            return None

    def add_bigdata(self,
                    name,
                    server_path=None,
                    connection_type="fileShare"):
        """
        Registers a bigdata fileshare with the data store.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique bigdata fileshare name on the server.
        ---------------     --------------------------------------------------------------------
        server_path         Optional string. The path to the folder from the server.
        ---------------     --------------------------------------------------------------------
        connection_type     Optional string. Allows for the setting of the types of big data store.
                            The value 'fileShare' is used for local big data stores, and for
                            cloud stores, the connection_type should be 'dataStore'. The value
                            'fileShare' is the default value.
        ===============     ====================================================================


        :return:
           The big data fileshare if registered successfully, None otherwise.
        """
        output = None
        path = self._admin_url + "/data/registerItem"

        pattern = r'\\\\[a-zA-Z]+'
        if re.match(pattern, server_path) is not None:  # starts with double backslash, double the backslashes
            server_path = server_path.replace('\\', '\\\\')

        path_str = '{"path":"' + server_path + '"}'
        params = {
            'f': 'json',
            'item' : json.dumps({
                "path": "/bigDataFileShares/" + name,
                "type": "bigDataFileShare",

                "info": {
                    "connectionString": path_str,
                    "connectionType": connection_type
                }
            })
        }

        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res['status'] == 'success' or res['status'] == 'exists':
            output = Datastore(self, "/bigDataFileShares/" + name)

        if res['success']:
            print("Created Big Data file share for " + name)
        elif res['success'] == False and res['status'] != 'exists':
            raise Exception("Could not create Big Data file share: %s" % name)
        elif res['status'] == 'exists':
            print("Big Data file share exists for " + name)

        return output
    #----------------------------------------------------------------------
    def add_amazon_s3(self,
                      name,
                      bucket_name,
                      access_key,
                      access_secret,
                      region,
                      folder=None,
                      default_protocal="https"):
        """

        Allows administrators to registered Amazon S3 Buckets as Datastores.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the Amazon S3 instance.
        ------------------     --------------------------------------------------------------------
        bucket_name            Required String. The name of the S3 bucket.
        ------------------     --------------------------------------------------------------------
        access_key             Required String. The key value for the S3 Bucket.
        ------------------     --------------------------------------------------------------------
        access_secret          Required String. The access secret value for the S3 bucket.
        ------------------     --------------------------------------------------------------------
        region                 Required String. The Amazon region as a string.
        ------------------     --------------------------------------------------------------------
        folder                 Optional String. The S3 folder within the S3 Bucket.
        ------------------     --------------------------------------------------------------------
        default_protocal       Optional String. The URL scheme to contact the S3 bucket.
        ==================     ====================================================================

        :return: DataStore

        """
        if folder is not None:
            bucket_name = f"{bucket_name}/{folder}"
        path = self._admin_url + "/data/registerItem"
        template = {
            "path": f"/cloudStores/{name}",
            "type": "cloudStore",
            "provider": "amazon",
            "info": {
                "isManaged": False,
                "objectStore" : bucket_name,
                "connectionString": {
                    "accessKeyId": f"{access_key}",
                    "secretAccessKey": f"{access_secret}",
                    "region": region,
                    "defaultEndpointsProtocol": default_protocal,
                    "credentialType": "accesskey"
                }
            }

        }
        params = {
            'f' : 'json',
            'item' : json.dumps(template)
        }

        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res['status'] == 'success' or res['status'] == 'exists':
            output = Datastore(self, "/cloudStores/" + name)

        if res['success']:
            print("Created cloud store for " + name)
        elif res['success'] == False and res['status'] != 'exists':
            raise Exception("Could not create cloud store: %s" % name)
        elif res['status'] == 'exists':
            print("Cloud store exists for exists for " + name)
        return output
    #----------------------------------------------------------------------
    def add_ms_azure_storage(self,
                             cloud_storage_name,
                             account_key,
                             account_name,
                             container_name,
                             folder=None
                             ):
        """
        Creates a cloud store for an Amazon or Microsoft Azure store.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        cloud_storage_name     Required string. The name of the storage entry.
        ------------------     --------------------------------------------------------------------
        access_key             Required String. The key value for the Azure storage.
        ------------------     --------------------------------------------------------------------
        access_secret          Required String. The access secret value for the Azure storage.
        ------------------     --------------------------------------------------------------------
        coontainer_name        Required String. The container holding the data.
        ------------------     --------------------------------------------------------------------
        folder                 Optional String. The Azure folder within the datastore item.
        ==================     ====================================================================

        :return: DataStore

        """
        path = self._admin_url + "/data/registerItem"
        object_store = ""
        if folder:
            object_store = f"{container_name}/{folder}"
        else:
            object_store = f"{container_name}"
        template = {
            "type":"cloudStore",
            "info":{
                "isManaged":False,
                "connectionString":{
                    "accountKey": account_key,
                    "accountName": account_name,
                    "defaultEndpointsProtocol":"https",
                    "accountEndpoint": "core.windows.net",
                    "credentialType" : "accessKey"},
                "objectStore": object_store},
            "path": f"/cloudStores/{cloud_storage_name}",
            "provider" : "azure"
        }
        params = {
            'f' : 'json',
            'item' : json.dumps(template)
        }

        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res['status'] == 'success' or res['status'] == 'exists':
            output = Datastore(self, "/cloudStores/" + cloud_storage_name)

        if res['success']:
            print("Created cloud store for " + cloud_storage_name)
        elif res['success'] == False and res['status'] != 'exists':
            raise Exception("Could not create cloud store: %s" % cloud_storage_name)
        elif res['status'] == 'exists':
            print("Cloud store exists for exists for " + cloud_storage_name)
        return output
    #----------------------------------------------------------------------
    def add_cloudstore(self, name, conn_str, object_store,
                       provider, managed=False, folder=None):
        """
        Cloud Store data item represents a connection to a Amazon or Microsoft Azure store.
        Connection information for the data store item is stored within conn_str as a
        stringified JSON. ArcGIS Server encrypts connection string for storage. Connection
        strings that are encrypted will include a {crypt} prefix. You can get a data store
        item with decrypted connection string by passing a decrypt=true parameter in the request
        for a data store item. Data store with decrypted connection string will be returned only for
        requests made with https. The examples below show data stores with decrypted conn_str.
        A valid object_store (S3 bucket or Azure Blob store) is required. Folders within an object
        store are optional.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The name of the cloud store.
        ---------------     --------------------------------------------------------------------
        conn_str            Required string. The connection information for the cloud storage
                            product.
        ---------------     --------------------------------------------------------------------
        object_store        Required string. This is the amazon bucket path or Azuze path.
        ---------------     --------------------------------------------------------------------
        provider            Required string. Values must be azuredatalakestore, amazon,
                            Alibaba, or azure.
        ---------------     --------------------------------------------------------------------
        managed             Optional boolean. When the data store is server only, the database
                            is entirely managed and owned by the server and cannot be accessed
                            by the publisher directly. When this option is chosen, the
                            managed property should be set to true. Otherwise it is false.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. For some Azure cloud stores, an optional folder
                            can be specified.
        ===============     ====================================================================


        :return: DataStore

        """
        path = self._admin_url + "/data/registerItem"
        cs = {
            "path": "/cloudStores/%s" % name,
            "type": "cloudStore",
            "provider": provider,
            "info": {
                "isManaged": managed,
                "connectionString": conn_str,
                "objectStore": object_store
            }
        }
        if folder is not None:
            cs['info']['folder'] = folder
        params = {
            'f' : 'json',
        'item' : json.dumps(cs)
        }

        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res['status'] == 'success' or res['status'] == 'exists':
            output = Datastore(self, "/cloudStores/" + name)

        if res['success']:
            print("Created cloud store for " + name)
        elif res['success'] == False and res['status'] != 'exists':
            raise Exception("Could not create cloud store: %s" % name)
        elif res['status'] == 'exists':
            print("Cloud store exists for exists for " + name)


        return output


    def add_database(self,
                     name,
                     conn_str,
                     client_conn_str=None,
                     conn_type="shared"):
        """
        Registers a database with the data store.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique database name on the server.
        ---------------     --------------------------------------------------------------------
        conn_str            Required string. the path to the folder from the server (and client, if shared or serverOnly database).
        ---------------     --------------------------------------------------------------------
        client_conn_str     Optional string. The connection string for client to connect to replicated enterprise database.
        ---------------     --------------------------------------------------------------------
        conn_type           Optional string. Choice of "<shared|replicated|serverOnly>", shared is the default.
        ===============     ====================================================================


        :return:
           The database if registered successfully, None otherwise.
        """

        item = {
            "type" : "egdb",
            "path" : "/enterpriseDatabases/" + name,
            "info" : {
                "connectionString" : conn_str,
                "dataStoreConnectionType" : conn_type
            }
        }

        if client_conn_str is not None:
            item['info']['clientConnectionString'] = client_conn_str

        is_managed = False
        if conn_type == "serverOnly":
            is_managed = True

        item['info']['isManaged'] = is_managed

        params = {
            "f" : "json",
            "item" : item
        }
        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res['status'] == 'success' or res['status'] == 'exists':
            return Datastore(self, "/enterpriseDatabases/" + name)
        else:
            print(str(res))
            return None

    def add(self,
            name,
            item):
        """
        Registers a new data item with the data store.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The name of the item to be added on the server.
        ---------------     --------------------------------------------------------------------
        item                Required dictionary. The dictionary representing the data item.  See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000001s9000000
        ===============     ====================================================================


        :return:
           The new data item if registered successfully, None otherwise.
        """
        params = {
            "f" : "json"
        }

        params['item'] = item
        status, msg = self._validate_item(item=params['item'])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res['status'] == 'success' or res['status'] == 'exists':
            return Datastore(self, "/enterpriseDatabases/" + name)
        else:
            print(str(res))
            return None

    def get(self, path):
        """
        Returns the data item object at the given path.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        path                Required string. The path for the data item.
        ===============     ====================================================================


        :return:
           The data item object if found, None otherwise.
        """
        params = { "f" : "json" }
        urlpath = self._admin_url + "/data/items" + path

        datadict = self._portal.con.post(urlpath, params, verify_cert=False)
        if 'status' not in datadict:
            return Datastore(self, path)
        else:
            print(datadict['messages'])
            return None

    def search(self, parent_path=None, ancestor_path=None,
               types=None, id=None):
        """
           You can use this operation to search through the various data
           items registered in the server's data store. Searching without
           specifying the parent path and other parameters returns a list
           of all registered data items.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        parentPath          Optional string. The path of the parent under which to find items.
                            Pass '/' to get the root data items.
        ---------------     --------------------------------------------------------------------
        ancestorPath        Optional string. The path of the ancestor under which to find items.
        ---------------     --------------------------------------------------------------------
        types               Optional string. A comma separated filter for the type of the items.
                            Types include folder, egdb, bigDataFileShare, datadir.
        ---------------     --------------------------------------------------------------------
        id                  Optional string. A filter to search by the ID of the item.
        ===============     ====================================================================


        :return:
           A list of data items matching the specified query.
        """
        params = {
            "f" : "json",
        }
        if parent_path is None and ancestor_path is None and types is None and id is None:
            ancestor_path = '/'
        if parent_path is not None:
            params['parentPath'] = parent_path
        if ancestor_path is not None:
            params['ancestorPath'] = ancestor_path
        if types is not None:
            params['types'] = types
        if id is not None:
            params['id'] = id


        path = self._admin_url + "/data/findItems"


        dataitems = []

        res = self._portal.con.post(path, params, verify_cert=False)
        for item in res['items']:
            dataitems.append(Datastore(self, item['path']))
        return dataitems

    def _validate_item(self, item):
        """validates a BDS connection"""
        msg = ""
        url = self._admin_url + "/data/validateDataItem"
        params = {
            'f' : 'json',
            'item' : item
        }
        res = self._portal.con.post(url, params, verify_cert=False)
        try:
            return res['status'] == 'success', ""
        except:
            return False, res

    def validate(self):
        """
        Validates all items in the datastore. In order for a data item to be registered and
        used successfully within the GIS's data store, you need to make sure that the path
        (for file shares) or connection string (for databases) is accessible to every server
        node in the site. To validate all registered data items all
        at once, you can invoke this operation.

        :return:
           True if the data store items were validated, False if not.
        """
        params = {"f" : "json"}
        path = self._admin_url + "/data/validateAllDataItems"
        res = self._portal.con.post(path, params, verify_cert=False)
        return res['status'] == 'success'
###########################################################################
class UserManager(object):
    """
    Helper class for managing GIS users. This class is not created by users directly.
    An instance of this class, called 'users', is available as a property of the Gis object.
    Users call methods on this 'users' object to manipulate (create, get, search, etc) users.
    """
    _me = None
    #----------------------------------------------------------------------
    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal
    #----------------------------------------------------------------------
    def __str__(self):
        return "<UserManager @ {url}>".format(url=self._gis._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def user_settings(self):
        """
        Gets/sets the user's settings

        The `user_settings` allows administrators to set, and edit, new
        member defaults. Members who create their own built-in accounts and
        members added by an administrator or through automatic account
        creation will be automatically assigned the new member defaults.

        Passing in `None` to the property will delete all the user settings.

        **Settings Key/Value Dictionary**

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        role	          String/Role. The role ID. To assign a custom role as the new member default,
                          provide a Role object.

                          Values: `administrator`, `publisher`, `editor`, `viewer` or custom `Role` object
        ----------------  -------------------------------------------------------------------------------
        userLicenseType   String. The ID of a user type licensed with your organization. To see which
                          user types are included with your organization's licensing, see the License
                          resource in the Portal Admin API.

                          Values: `creator`, `editor`, `Advanced GIS`, `Basic GIS`, `Standard GIS`,
                          `viewer`, or `fieldWorker`
        ----------------  -------------------------------------------------------------------------------
        groups            List of String/Groups. An array of group ID numbers or `Group` objects that
                          specify the groups new members will be added to.
        ----------------  -------------------------------------------------------------------------------
        userType          String.  This key only applies to `ArcGIS Online`. If new members will have
                          Esri access (both) or if Esri access will be disabled (arcgisonly). The default
                          value is `arcgisonly`.

                          Values: `arcgisonly` or `both`
        ----------------  -------------------------------------------------------------------------------
        apps              List of dictionaries.  An array of an app's itemID and, when applicable, entitlement.
                          Example: `{"apps" :[{"itemId": "f761dd0f298944dcab22d1e888c60293","entitlements": ["Insights"]}]}`
        ----------------  -------------------------------------------------------------------------------
        appBundles        List of dictionaries. An array of an app bundle's ID.

                          Example: `{"appBundles":[{"itemId": "99d7956c7e824ff4ab27422e2a26c2b7}]}`
        ================  ===============================================================================

        :returns: Dictionary

        """
        if self._gis.version >= [7,3]:
            url = f"{self._gis._portal.resturl}portals/self/userDefaultSettings"
            params = {'f' : 'json'}
            return self._gis._con.get(url, params)
        return None
    #----------------------------------------------------------------------
    @user_settings.setter
    def user_settings(self, settings):
        """
        Gets/sets the user's settings

        The `user_settings` allows administrators to set, and edit, new
        member defaults. Members who create their own built-in accounts and
        members added by an administrator or through automatic account
        creation will be automatically assigned the new member defaults.

        Passing in `None` to the property will delete all the user settings.

        **Settings Key/Value Dictionary**

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        role	          String/Role. The role ID. To assign a custom role as the new member default,
                          provide a Role object.

                          Values: `administrator`, `publisher`, `editor`, `viewer` or custom `Role` object
        ----------------  -------------------------------------------------------------------------------
        userLicenseType   String. The ID of a user type licensed with your organization. To see which
                          user types are included with your organization's licensing, see the License
                          resource in the Portal Admin API.

                          Values: `creator`, `editor`, `Advanced GIS`, `Basic GIS`, `Standard GIS`,
                          `viewer`, or `fieldWorker`
        ----------------  -------------------------------------------------------------------------------
        groups            List of String/Groups. An array of group ID numbers or `Group` objects that
                          specify the groups new members will be added to.
        ----------------  -------------------------------------------------------------------------------
        userType          String.  This key only applies to `ArcGIS Online`. If new members will have
                          Esri access (both) or if Esri access will be disabled (arcgisonly). The default
                          value is `arcgisonly`.

                          Values: `arcgisonly` or `both`
        ----------------  -------------------------------------------------------------------------------
        apps              List of dictionaries.  An array of an app's itemID and, when applicable, entitlement.
                          Example: `{"apps" :[{"itemId": "f761dd0f298944dcab22d1e888c60293","entitlements": ["Insights"]}]}`
        ----------------  -------------------------------------------------------------------------------
        appBundles        List of dictionaries. An array of an app bundle's ID.

                          Example: `{"appBundles":[{"itemId": "99d7956c7e824ff4ab27422e2a26c2b7}]}`
        ================  ===============================================================================

        :returns: Dictionary

        """
        user_li_lu = {
            "creatorUT" : "creatorUT",
            "creator" : "creatorUT",
            "editor" : "editorUT",
            "editorUT" : "editorUT",
            "GISProfessionalAdvUT" : "GISProfessionalAdvUT",
            "Advanced GIS" : "GISProfessionalAdvUT",
            "Basic GIS" : "GISProfessionalBasicUT",
            "GISProfessionalBasicUT" : "GISProfessionalBasicUT",
            "Standard GIS" : "GISProfessionalStdUT",
            "GISProfessionalStdUT" : "GISProfessionalStdUT",
            "viewer" : "viewerUT",
            "viewerUT" : "viewerUT",
            "fieldworker" : "fieldWorkerUT",
            "fieldWorkerUT" : "fieldWorkerUT"
        }
        role_lu = {
             "administrator" : "org_admin",
             "org_admin" : "org_admin",
             "publisher" : "org_publisher",
             "org_publisher" : "org_publisher",
             "user" : "org_user",
             "iBBBBBBBBBBBBBBB" : "iBBBBBBBBBBBBBBB",
             "editor" : "iBBBBBBBBBBBBBBB",
             "viewer" : "iAAAAAAAAAAAAAAA",
             "iAAAAAAAAAAAAAAA" : "iAAAAAAAAAAAAAAA"
        }
        if self._gis.version > [7, 3]:
            if settings is None or \
               (isinstance(settings, dict) and \
               len(settings) == 0):
                cs = self.user_settings
                if cs and len(cs) > 0:
                    self._delete_user_settings()
            else:
                url = f"{self._gis._portal.resturl}portals/self/setUserDefaultSettings"
                params = {'f' : 'json'}
                if 'role' in settings:
                    if settings['role'] in role_lu:
                        settings['role'] = role_lu[settings['role'].lower()]
                    elif isinstance(settings['role'], Role):
                        settings['role'] = settings['role'].role_id
                if 'userLicenseType' in settings:
                    if settings['userLicenseType'].lower() in user_li_lu:
                        settings['userLicenseType'] = user_li_lu[settings['userLicenseType'].lower()]
                if 'userType' in settings and self._gis._portal.is_arcgisonline == False:
                    del settings['userType']
                if 'groups' in settings:
                    settings['groups'] = [grp.groupid for grp in settings['groups'] if isinstance(grp, Group)] + \
                        [grp for grp in settings['groups'] if isinstance(grp, str)]
                params.update(settings)
                res = self._gis._con.post(url, params)
                if 'success' in res and res['success'] == False:
                    raise Exception(res)
    #----------------------------------------------------------------------
    def _delete_user_settings(self):
        """
        This operation allows administrators to clear the previously
        configured new member defaults set either through the Set User
        Default Settings operation or from the New Member Defaults tab in
        the Organization Settings of the portal.

        :returns: Boolean

        """
        if self._gis.version > [7, 3]:
            url = f"{self._gis._portal.resturl}portals/self/userDefaultSettings/delete"
            params = {'f' : 'json'}
            res = self._portal.con.post(url, params)
            if 'success' in res:
                return res['success']
            return res
        return None
    #----------------------------------------------------------------------
    @property
    def license_types(self):
        """
        Returns a list of available licenses associated with a given GIS.
        The information returned can help administrators determine what type
        of a user should me based on the bundles associated with each user
        type.

        **This is only available on 10.7+.**

        :returns: list
        """

        if self._gis.version < [6,4]:
            return []

        url = "portals/self/userLicenseTypes"
        params = {
            'f': "json",
            'start' : 1,
            'num' : 255
        }

        res = self._gis._con.get(url, params)
        results = res['userLicenseTypes']
        while res['nextStart'] > -1:
            params['start'] += 255
            res = self._gis._con.get(url, params)
            results += res['userLicenseTypes']
        return results
    #----------------------------------------------------------------------
    def counts(self, type='bundles', as_df=True):
        """
        This method returns a simple report on the number of licenses currently used
        for a given `type`.  A `type` can be a role, app, bundle or user license type.

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        type              Required String. The type of data to return.  The following values are valid:

                            + role - returns counts on user roles
                            + app - returns counts on registered applications
                            + bundles - returns counts on application bundles
                            + user_type - returns counts on the user license types
        ----------------  -------------------------------------------------------------------------------
        as_df             Optional boolean. If true, the results are returned as a pandas DataFrame, else
                          it is returned as a list of dictionaries.
        ================  ===============================================================================

        :returns: Pandas DataFrame if as_df is True. If False, the result is a list of dictionaries.


        **Example as_df=True**

        >>> df = gis.um.counts('user_type', True)
        >>> df
            count        key
         0     12  creatorUT
         1      2   viewerUT



        **Example as_df=False**


        >>> df = gis.um.counts('user_type', False)
        >>> df
        [{'key': 'creatorUT', 'count': 12}, {'key': 'viewerUT', 'count': 2}]


        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("`counts` is not implemented at version %s of Enterprise" % \
                                 ".".join([str(i) for i in self._gis.version]))

        url = "portals/self/users/counts"
        lu = {
            'roles' : 'role',
            'role' : 'role',
            'app' : 'app',
            'bundles' : 'appBundle',
            'user_type' : 'userLicenseType',
            'usertype' : 'userLicenseType'
        }
        results = []
        params = {
            'f' : 'json',
            'type' : lu[type.lower()],
            'num' : 100,
            'start' : 1
        }
        res = self._portal.con.get(url, params, ssl=True)
        results += res['results']
        while res['nextStart'] != -1:
            if res['nextStart'] == -1:
                break
            params['start'] = res['nextStart']
            res = self._portal.con.get(url, params, ssl=True)
            results += res['results']
        if as_df:
            import pandas as pd
            return pd.DataFrame(data=results)
        return results
    #----------------------------------------------------------------------
    def send_notification(self,
                          users,
                          subject,
                          message,
                          type='builtin',
                          client_id=None):
        """
        Creates a user notifcations for a list of users.


        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        users             Required List. A list of strings or User objects to send notifcations to.
        ----------------  -------------------------------------------------------------------------------
        subject           Required String. The notifcation subject line.
        ----------------  -------------------------------------------------------------------------------
        message           Required String. The notifcation content. This should be in plain text.
        ----------------  -------------------------------------------------------------------------------
        type              Optional String.  The notification can be sent various ways. These include:

                             - builtin - The enterprise built-in system
                             - push - The push notifcation to send a message to
                             - email - a notification sent to the user's email account
        ----------------  -------------------------------------------------------------------------------
        client_id         Optional String. The client id for push notification.
        ================  ===============================================================================

        :returns: Boolean

        """
        if self._gis.version >= [6,4]:
            susers = []
            for u in users:
                if isinstance(u, str):
                    susers.append(u)
                elif isinstance(u, User):
                    susers.append(u.username)
                del u
            url = "{base}portals/self/createNotification".format(base=self._gis._portal.resturl)
            params = {
                "f" : 'json',
                "notificationChannelType": type,
                "subject" : subject,
                "message" : message,
                "users" : ",".join(susers),
                "clientId" : client_id
            }
            return self._portal.con.post(url, params)['success']
        else:
            raise NotImplementedError("The current version of the enterprise does not support `send_notification`")
        return False
    #----------------------------------------------------------------------
    def create(self, username, password, firstname, lastname, email, description=None, role=None,
               provider='arcgis', idp_username=None, level=2, thumbnail=None, user_type=None, credits=-1,
               groups=None):
        """
        This operation is used to pre-create built-in or enterprise accounts within the portal,
        or built-in users in an ArcGIS Online organization account. Only an administrator
        can call this method.

        To create a viewer account, choose role='org_viewer' and level='viewer'

        .. note:
            When Portal for ArcGIS is connected to an enterprise identity store, enterprise users sign
            into portal using their enterprise credentials. By default, new installations of Portal for
            ArcGIS do not allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Portal, and
                          6-24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string. The password for the user.  It must be at least 8 characters.
                          This is a required parameter only if
                          the provider is arcgis; otherwise, the password parameter is ignored.
                          If creating an account in an ArcGIS Online org, it can be set as None to let
                          the user set their password by clicking on a link that is emailed to him/her.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to user's image.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The :class:`role <arcgis.gis.Role>` for the user account. The
                          default value is ``org_user``. Other possible values are ``org_publisher``,
                          ``org_admin``, ``viewer``, ``viewplusedit`` or a custom :class:`role_id <arcgis.gis.Role>`
                          value obtained from the :func:`~RoleManager.all` method of the :class:`RoleManager` class.

                           .. code-block:: python
                              :emphasize-lines: 10,18

                              # Usage Example: Assign custom role to a new user

                              >>> role_mgr = gis.users.roles

                              >>> for role in role_mgr.all():
                              >>>     print(f"{role.name}  {role.role_id}")

                                  Viewer              iAAAAAAAAAAAAAAA
                                  Data Editor         iBBBBBBBBBBBBBBB
                                  CustomRole          bKrTCjFF9tKbaFk8

                              >>> gis.users.create(username='new_user_1',
                                                   password='<strong_password>',
                                                   firstname='New',
                                                   lastname='User',
                                                   email='namee@organization.com',
                                                   description='User with custom role assigned',
                                                   role='bKrTCjFF9tKbaFk8',
                                                   user_type='Creator')
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        idp_username      Optional string. The name of the user as stored by the enterprise user store.
                          This parameter is only required if the provider parameter is enterprise.
        ----------------  -------------------------------------------------------------------------------
        level             Optional string. The account level. (Pre 10.7 Portal)
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ----------------  -------------------------------------------------------------------------------
        user_type         Required string. The account user type. This can be creator or viewer.  The
                          type effects what applications a user can use and what actions they can do in
                          the organization. (10.7+)
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ----------------  -------------------------------------------------------------------------------
        credits           Optional Float. The number of credits to assign a user.  The default is None,
                          which means unlimited. (10.7+)
        ----------------  -------------------------------------------------------------------------------
        groups            Optional List. An array of Group objects to provide access to for a given
                          user. (10.7+)
        ================  ===============================================================================

        :return:
            The :class:`user <arcgis.gis.User>` if successfully created, None if unsuccessful.

        """
        kwargs = locals()
        if self._gis.version >= [6,4]:
            allowed_keys = {'username', 'password', 'firstname', 'lastname',
                            'email', 'description', 'role', 'provider', 'idp_username',
                            'user_type', 'thumbnail', 'credits', 'groups', 'level'}
            params = {}
            for k,v in kwargs.items():
                if k in allowed_keys:
                    params[k] = v
            return self._create64plus(**params)
        else:
            allowed_keys = {'username', 'password', 'firstname', 'lastname',
                            'email', 'description', 'role', 'provider', 'idp_username',
                            'level', 'thumbnail'}
            params = {}
            for k,v in kwargs.items():
                if k in allowed_keys:
                    params[k] = v
            return self._createPre64(**params)
        return None
    #----------------------------------------------------------------------
    def _createPre64(self, username, password, firstname, lastname, email, description=None, role='org_user',
                     provider='arcgis', idp_username=None, level=2, thumbnail=None):
        """
        This operation is used to pre-create built-in or enterprise accounts within the portal,
        or built-in users in an ArcGIS Online organization account. Only an administrator
        can call this method.

        To create a viewer account, choose role='org_viewer' and level=1

        .. note:
            When Portal for ArcGIS is connected to an enterprise identity store, enterprise users sign
            into portal using their enterprise credentials. By default, new installations of Portal for
            ArcGIS do not allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Portal, and
                          6-24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string. The password for the user.  It must be at least 8 characters.
                          This is a required parameter only if
                          the provider is arcgis; otherwise, the password parameter is ignored.
                          If creating an account in an ArcGIS Online org, it can be set as None to let
                          the user set their password by clicking on a link that is emailed to him/her.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to user's image.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The role for the user account. The default value is org_user.
                          Other possible values are org_publisher, org_admin, org_viewer.
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        idp_username      Optional string. The name of the user as stored by the enterprise user store.
                          This parameter is only required if the provider parameter is enterprise.
        ----------------  -------------------------------------------------------------------------------
        level             Optional string. The account level.
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ================  ===============================================================================

        :return:
            The user if successfully created, None if unsuccessful.

        """
        #map role parameter of a viewer to the internal value for org viewer.
        if role == 'org_viewer':
            role = 'iAAAAAAAAAAAAAAA'

        if self._gis._portal.is_arcgisonline:
            email_text = '''<html><body><p>''' + self._gis.properties.user.fullName + \
                ''' has invited you to join an ArcGIS Online Organization, ''' + self._gis.properties.name + \
                         '''</p>
<p>Please click this link to finish setting up your account and establish your password: <a href="https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@">https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@</a></p>
<p>Note that your account has already been created for you with the username, <strong>@@touser.username@@</strong>.  </p>
<p>If you have difficulty signing in, please contact ''' + self._gis.properties.user.fullName + \
                                                         '(' + self._gis.properties.user.email + '''). Be sure to include a description of the problem, the error message, and a screenshot.</p>
<p>For your reference, you can access the home page of the organization here: <br>''' + self._gis.properties.user.fullName + '''</p>
<p>This link will expire in two weeks.</p>
<p style="color:gray;">This is an automated email. Please do not reply.</p>
</body></html>'''
            params = {
                'f': 'json',
                'invitationList' : {'invitations' : [ {
                    'username': username,
                    'firstname': firstname,
                    'lastname': lastname,
                    'fullname': firstname + ' ' + lastname,
                    'email': email,
                    'role': role,
                    'level': level
                    } ] },
                'message' : email_text
            }
            if idp_username is not None:
                if provider is None:
                    provider = 'enterprise'
                params['invitationList']['invitations'][0]['targetUserProvider'] = provider
                params['invitationList']['invitations'][0]['idpUsername'] = idp_username
            if password is not None:
                params['invitationList']['invitations'][0]['password'] = password

            resp = self._portal.con.post('portals/self/invite', params, ssl=True)
            if resp and resp.get('success'):
                if username in resp['notInvited']:
                    print('Unable to create ' + username)
                    _log.error('Unable to create ' + username)
                    return None
                else:
                    return self.get(username)
        else:
            createuser_url = self._portal.url + "/portaladmin/security/users/createUser"
            #print(createuser_url)
            params = {
                'f': 'json',
                'username' : username,
                'password' : password,
                'firstname' : firstname,
                'lastname' : lastname,
                'email' : email,
                'description' : description,
                'role' : role,
                'provider' : provider,
                'idpUsername' : idp_username,
                'level' : level
            }
            self._portal.con.post(createuser_url, params)
            user = self.get(username)
            if thumbnail is not None:
                ret = user.update(thumbnail=thumbnail)
                if not ret:
                    _log.error('Unable to update the thumbnail for  ' + username)
            return user
    #----------------------------------------------------------------------
    def _create64plus(self, username, password, firstname, lastname, email, description=None, role='org_user',
                      provider='arcgis', idp_username=None, user_type='creator',
                      thumbnail=None, credits=None, groups=None, level=None):
        """
        This operation is used to pre-create built-in or enterprise accounts within the portal,
        or built-in users in an ArcGIS Online organization account. Only an administrator
        can call this method.

        To create a viewer account, choose role='org_viewer' and level='viewer'

        .. note:
            When Portal for ArcGIS is connected to an enterprise identity store, enterprise users sign
            into portal using their enterprise credentials. By default, new installations of Portal for
            ArcGIS do not allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Portal, and
                          6-24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string. The password for the user.  It must be at least 8 characters.
                          This is a required parameter only if the provider is arcgis; otherwise, the
                          password parameter is ignored.
                          If creating an account in an ArcGIS Online org, it can be set as None to let
                          the user set their password by clicking on a link that is emailed to him/her.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to user's image.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The role for the user account. The default value is org_user.
                          Other possible values are org_user, org_publisher, org_admin, viewer,
                          view_only, viewplusedit or a custom role object (from gis.users.roles).
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        idp_username      Optional string. The name of the user as stored by the enterprise user store.
                          This parameter is only required if the provider parameter is enterprise.
        ----------------  -------------------------------------------------------------------------------
        user_type         Required string. The account user type. This can be creator or viewer.  The
                          type effects what applications a user can use and what actions they can do in
                          the organization.
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ----------------  -------------------------------------------------------------------------------
        credits           Optional Float. The number of credits to assign a user.  The default is None,
                          which means unlimited.
        ----------------  -------------------------------------------------------------------------------
        groups            Optional List. An array of Group objects to provide access to for a given user.
        ================  ===============================================================================

        :return:
            The user if successfully created, None if unsuccessful.

        """
        #map role parameter of a viewer to the internal value for org viewer.
        if self._gis.version >= [7,2]:
            if self._gis._is_agol:
                if user_type is None and role is None:
                    if self.user_settings and \
                       'userLicenseType' in self.user_settings:
                        user_type = self.user_settings['userLicenseType']
                        role = self.user_settings['role']
        else:
            if self._gis.version >= [7,1]:
                if user_type is None and role is None:
                    if 'defaultUserTypeIdForUser' in self._gis.admin.security.config:
                        user_type = self._gis.admin.security.config['defaultUserTypeIdForUser']
                        role = self._gis.admin.security.config['defaultRoleForUser']
        if level == 2 and user_type is None and role is None:
            user_type = "creator"
            role = 'publisher'
        elif level == 1 and user_type is None and role is None:
            user_type = "viewer"
            role = 'viewer'
        elif level == 1 and user_type is None:
            user_type = "viewer"
        elif level == 1 and role is None:
            role = "viewer"
        elif level == 2 and user_type is None:
            user_type = "creator"
        elif level == 2 and role is None:
            role = "publisher"

        levels = {'creator' : 'creatorUT',
                  'viewer' : 'viewerUT'}
        role_lookup = {
            'admin' : 'org_admin',
            'user' : 'org_user',
            'publisher' : 'org_publisher',
            'creator' : 'org_publisher',
            'view_only' : 'tLST9emLCNfFcejK',
            'org_viewer' : 'iAAAAAAAAAAAAAAA',
            'viewer' : 'iAAAAAAAAAAAAAAA',
            'viewplusedit' : 'iBBBBBBBBBBBBBBB'
        }

        if groups is None:
            groups = []

        if user_type.lower() in levels:
            user_type = levels[user_type.lower()]

        if isinstance(role, Role):
            role = role.role_id
        elif role.lower() in role_lookup:
            role = role_lookup[role.lower()]

        if self._gis._portal.is_arcgisonline:
            email_text = '''<html><body><p>''' + self._gis.properties.user.fullName + \
                ''' has invited you to join an ArcGIS Online Organization, ''' + self._gis.properties.name + \
                '''</p>
<p>Please click this link to finish setting up your account and establish your password: <a href="https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@">https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@</a></p>
<p>Note that your account has already been created for you with the username, <strong>@@touser.username@@</strong>.  </p>
<p>If you have difficulty signing in, please contact ''' + self._gis.properties.user.fullName + \
                                                         '(' + self._gis.properties.user.email + '''). Be sure to include a description of the problem, the error message, and a screenshot.</p>
<p>For your reference, you can access the home page of the organization here: <br>''' + self._gis.properties.user.fullName + '''</p>
<p>This link will expire in two weeks.</p>
<p style="color:gray;">This is an automated email. Please do not reply.</p>
</body></html>'''
            if credits == -1 and self._gis.version >= [7,2] and \
                self._gis.properties['defaultUserCreditAssignment'] != -1:
                credits = self._gis.properties['defaultUserCreditAssignment']
            if not groups and \
               self.user_settings and \
               'groups' in self.user_settings and \
               self.user_settings['groups']:
                groups = [g for g in self.user_settings['groups']]

            params = {
                'f': 'json',
                'invitationList': {'invitations': [
                    {
                    'username': username,
                    'firstname': firstname,
                    'lastname': lastname,
                    'fullname': firstname + ' ' + lastname,
                    'email': email,
                    'role': role,
                    "userLicenseType": user_type,
                    "groups":",".join(groups),
                    "userCreditAssignment": credits,

                    }
                    ],
                        "apps":[],
                        "appBundles":[]
                },
                #'message' : email_text
            }
            if idp_username is not None:
                if provider is None:
                    provider = 'enterprise'
                params['invitationList']['invitations'][0]['targetUserProvider'] = provider
                params['invitationList']['invitations'][0]['idpUsername'] = idp_username
            if password is not None:
                params['invitationList']['invitations'][0]['password'] = password

            resp = self._portal.con.post('portals/self/invite', params, ssl=True)
            if resp and resp.get('success'):
                if username in resp['notInvited']:
                    print('Unable to create ' + username)
                    _log.error('Unable to create ' + username)
                    return None
                else:
                    new_user = self.get(username)
                    if self.user_settings and \
                    'userType' in self.user_settings and \
                       not self.user_settings['userType'] == 'arcgisonly':
                        update_url = "community/users/" + username + "/update"
                        user_params = {"f":"json",
                                       "token":"token",
                                       "userType": self.user_settings['userType']}
                        self._portal.con.post(update_url, user_params, ssl=True)
                        return new_user
                    else:
                        return new_user
        else:
            createuser_url = self._portal.url + "/portaladmin/security/users/createUser"
            params = {
                'f': 'json',
                'username' : username,
                'password' : password,
                'firstname' : firstname,
                'lastname' : lastname,
                'email' : email,
                'description' : description,
                'role' : role,
                'provider' : provider,
                'idpUsername' : idp_username,
                "userLicenseTypeId": user_type
            }
            self._portal.con.post(createuser_url, params)
            user = self.get(username)
            for grp in groups:
                grp.add_users([username])
            if thumbnail is not None:
                ret = user.update(thumbnail=thumbnail)
                if not ret:
                    _log.error('Unable to update the thumbnail for  ' + username)
            return user

    #----------------------------------------------------------------------
    def invite(self,
               email, role='org_user',
               level=2, provider=None,
               must_approve=False, expiration='1 Day',
               validate_email=True):
        """
        Invites a user to an organization by email

        ================  ===============================================================================
        **Argument**      **Description**
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The user's email that will be invited to the organization.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The role for the user account. The default value is org_user.
                          Other possible values are org_publisher, org_admin, org_viewer.
        ----------------  -------------------------------------------------------------------------------
        level             Optional string. The account level. The default is 2.
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        must_approve      Optional boolean. After a user accepts the invite, if True, and administrator
                          must approve of the individual joining the organization. The default is False.
        ----------------  -------------------------------------------------------------------------------
        expiration        Optional string.  The default is '1 Day'. This is the time the emailed user has
                          to accept the invitation request until it expires.
                          The values are: 1 Day (default), 3 Days, 1 Week, or 2 Weeks.
        ----------------  -------------------------------------------------------------------------------
        validate_email    Optional boolean. If True (default) the Enterprise will ensure that the email
                          is properly formatted. If false, no check will occur
        ================  ===============================================================================

        :returns: boolean

        """
        if self._gis._portal.is_arcgisonline == False:
            raise Exception("This method is only for ArcGIS Online.")
        time_lookup = {
            '1 Day'.upper() : 1440,
            '3 Days'.upper() : 4320,
            '1 Week'.upper() : 10080,
            '2 Weeks'.upper() : 20160
        }
        if expiration.upper() in time_lookup:
            expiration = time_lookup[expiration.upper()]
        elif not isinstance(expiration, int):
            raise ValueError("Invalid expiration.")

        url = self._portal.resturl + "/portals/self/inviteByEmail"
        msg = "You have been invited you to join an ArcGIS Online Organization, %s" % (self._gis.properties['name'])
        params = {
            "f" : "json",
            "emails" : email,
            "message" : msg,
            "role" : role,
            "level" : level,
            "targetUserProvider" : provider or "arcgis",
            "mustApprove" : must_approve,
            "expiration" : expiration,
            "validateEmail" : validate_email
        }
        res = self._portal.con.post(url, params)
        if 'success' in res:
            return res['success']
        return False
    #----------------------------------------------------------------------
    @property
    def invitations(self):
        """
        Provides access to invitations sent to users using the `invite` method

        **Note** : this is only supported by ArcGIS Online

        :returns: InvitationManager

        """

        if self._gis._portal.is_arcgisonline == False:
            raise Exception("This property is only for ArcGIS Online.")
        from ._impl._invitations import InvitationManager
        url = self._portal.resturl + "portals/self/invitations"
        return InvitationManager(url, gis=self._gis)
    #----------------------------------------------------------------------
    def signup(self, username, password, fullname, email):
        """
        Signs up a user to an instance of Portal for ArcGIS.

        .. note:
            This method only applies to Portal and not ArcGIS
            Online.  This method can be called anonymously, but
            keep in mind that self-signup can also be disabled
            in a Portal.  It also only creates built-in
            accounts, it does not work with enterprise
            accounts coming from ActiveDirectory or your
            LDAP.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        username          Required string. The desired username, which must be unique in the Portal,
                          and at least 4 characters.
        ----------------  --------------------------------------------------------
        password          Required string. The passowrd, which must be at least 8 characters.
        ----------------  --------------------------------------------------------
        fullname          Required string. The full name of the user.
        ----------------  --------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ================  ========================================================

        :return:
            The user if successfully created, None if unsuccessful.

        """
        success = self._portal.signup(username, password, fullname, email)
        if success:
            return User(self._gis, username)
        else:
            return None
    #----------------------------------------------------------------------
    def get(self, username):
        """
        Returns the user object for the specified username.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The user to get as a string. This can be the
                               user's login name or the user's ID.
        ==================     ====================================================================


        :return:
            The user object if successfully found, None if unsuccessful.
        """
        try:
            with _DisableLogger():
                user = self._portal.get_user(username)

        except RuntimeError as re:
            if re.args[0].__contains__("User does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as e:
            if e.args[0].__contains__("User does not exist or is inaccessible"):
                return None
            else:
                raise e
        if user is not None:
            return User(self._gis, user['username'], user)
        return None
    #----------------------------------------------------------------------
    def enable_users(self, users):
        """
        This is a bulk operation that allows administrators to quickly enable large number of users
        in a single call.  It is useful to do this operation if you have multiple users that need
        to be enabled. Supported on ArcGIS REST API 6.4+.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List. List of User or UserNames to enable
        ==================     ====================================================================

        :returns: Boolean

        """
        if self._gis.version >= [6,4]:
            url = "{base}/portals/self/enableUsers".format(base=self._portal.resturl)
            params = {
                'f' : 'json',
                'users' : None
            }
            if isinstance(users, User) or \
               isinstance(users, str):
                users = [users]
            if isinstance(users, (list, tuple)):
                ul = []
                for user in users:
                    if isinstance(user, User):
                        ul.append(user.username)
                    else:
                        ul.append(user)
                params['users'] = ",".join(ul)
                res = self._portal.con.post(url, params)
                return any([r['status'] for r in res['results']])
            else:
                raise ValueError('Invalid input: must be of type list.')
        return False
    #----------------------------------------------------------------------
    def disable_users(self, users):
        """
        This is a bulk disables user operation that allows administrators to quickly disable large
        number of users in a single call.  It is useful to do this operation if you have multiple
        users that need to be disabled.  Supported on ArcGIS REST API 6.4+.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List. List of User or UserNames to disable
        ==================     ====================================================================

        :returns: Boolean

        """
        if self._gis.version >= [6,4]:
            url = "{base}/portals/self/disableUsers".format(base=self._portal.resturl)
            params = {
                'f' : 'json',
                'users' : None
            }
            if isinstance(users, User) or \
               isinstance(users, str):
                users = [users]
            if isinstance(users, (list, tuple)):
                ul = []
                for user in users:
                    if isinstance(user, User):
                        ul.append(user.username)
                    else:
                        ul.append(user)
                params['users'] = ",".join(ul)
                res = self._portal.con.post(url, params)
                return any([r['status'] for r in res['results']])
            else:
                raise ValueError('Invalid input: must be of type list.')
        return False
    #----------------------------------------------------------------------
    def advanced_search(self, query,
                        return_count=False, max_users=10,
                        start=1, sort_field="username",
                        sort_order="asc", as_dict=False):
        """
        The `advanced_search` method allows for the full control of the query operations
        by any given user.  The searches are performed against a high performance
        index that indexes the most popular fields of an user. See the Search
        reference page for information on the fields and the syntax of the query.

        The search index is updated whenever users is added, updated, or deleted. There
        can be a lag between the time that the user is updated and the time when it's
        reflected in the search results.

        The results of a search only contain items that the user has permission to access.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        query                  Required String.  The search query.
        ------------------     --------------------------------------------------------------------
        return_count           Optional Boolean.  If True, the number of users found by the query
                               string is returned.
        ------------------     --------------------------------------------------------------------
        max_users              Optional Integer. Limits the total number of users returned in a
                               a query.  The default is `10` users.  If all users is needed, `-1`
                               should be used.
        ------------------     --------------------------------------------------------------------
        start                  Optional Int. The starting position to search from.  This is
                               only required if paging is needed.
        ------------------     --------------------------------------------------------------------
        sort_field             Optional String. Responses from the `search` operation can be
                               sorted on various fields. `avgrating` is the default.
        ------------------     --------------------------------------------------------------------
        sort_order             Optional String. The sequence into which a collection of
                               records are arranged after they have been sorted. The allowed
                               values are: asc for ascending and desc for descending.
        ------------------     --------------------------------------------------------------------
        as_dict                Required Boolean. If True, the response comes back as a dictionary.
        ==================     ====================================================================

        :returns: dictionary if `return_count` is False, else an integer
        """
        from arcgis.gis._impl import _search
        stype = "users"
        max_items = max_users
        group_id = None
        if max_items == -1:
            max_items = _search(gis=self._gis, query=query, stype=stype,
                          max_items=0, start=start, sort_field=sort_field,
                          sort_order=sort_order, group_id=group_id, as_dict=as_dict)['total']
        so = {
            'asc' : 'asc',
            'desc' : 'desc',
            'ascending' : 'asc',
            'descending' : 'desc'
        }
        if sort_order:
            sort_order = so[sort_order]

        if return_count:
            max_items = 0
        if max_items <= 10:
            res = _search(gis=self._gis, query=query, stype=stype,
                          max_items=max_items,
                          start=start, sort_field=sort_field,
                          sort_order=sort_order, group_id=group_id, as_dict=as_dict)
            if 'total' in res and \
               return_count:
                return res['total']
            elif 'aggregations' in res:
                return res['aggregations']
            return res
        else:
            allowed_keys = [ 'query', 'return_count', 'max_users',
                             'bbox','categories', 'category_filter',
                             'start', 'sort_field', 'sort_order',
                             'count_fields','count_size', 'as_dict']
            inputs = locals()
            kwargs = {}
            for k,v in inputs.items():
                if k in allowed_keys:
                    kwargs[k] = v
            import concurrent.futures
            import math, copy
            num = 10
            steps = range(math.ceil(max_items / num))
            params = [ ]
            for step in steps:
                new_start = start + num*step
                kwargs['max_users'] = num
                kwargs['start'] = new_start
                params.append(copy.deepcopy(kwargs))
            items = {
                'results' : [],
                'start' : start,
                'num' : 10,
                'total' : max_items
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(self.advanced_search, **param): param for param in params}
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future_to_url[future]
                    data = future.result()
                    if 'results' in data:
                        items['results'].extend(data['results'])
            if len(items['results']) > max_items:
                items['results'] = items['results'][:max_items]
            return items
        return None
    #----------------------------------------------------------------------
    def search(self, query=None, sort_field='username', sort_order='asc',
               max_users=100, outside_org=False, exclude_system=False,
               user_type=None, role=None):
        """
        Searches portal users.

        Returns a list of users matching the specified query

        .. note::
            A few things that will be helpful to know.

            1. The query syntax has quite a few features that can't
               be adequately described here.  The query syntax is
               available in ArcGIS help.  A short version of that URL
               is http://bitly.com/1fJ8q31.

            2. Searching without specifying a query parameter returns
               a list of all users in your organization.

            3. Most of the time when searching users you want to
               search within your organization in ArcGIS Online
               or within your Portal.  As a convenience, the method
               automatically appends your organization id to the query by
               default.  If you don't want the API to append to your query
               set outside_org to True.  If you use this feature with an
               OR clause such as field=x or field=y you should put this
               into parenthesis when using outside_org.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        query             Optional string. The query string.  See notes above. Pass None
                          to get list of all users in the organization.
        ----------------  --------------------------------------------------------
        sort_field        Optional string. Valid values can be username (the default) or created.
        ----------------  --------------------------------------------------------
        sort_order        Optional string. Valid values are asc (the default) or desc.
        ----------------  --------------------------------------------------------
        max_users         Optional integer. The maximum number of users to be returned. The default is 100.
        ----------------  --------------------------------------------------------
        outside_org       Optional boolean. This controls whether to search outside
                          your organization. The default is False (search only
                          within your organization).
        ----------------  --------------------------------------------------------
        exclude_system    Optional boolean. Controls if built-in system accounts are
                          returned or not.  True means built-in account are not
                          returned, where as False means that they are.
        ----------------  --------------------------------------------------------
        user_type         Optional String. This parameters allows for the filtering
                          of the users by their assigned type.
        ----------------  --------------------------------------------------------
        role              Optional String.  Specify the roleId. This parameter
                          allows for the filting of the users based on a roleId.
        ================  ========================================================

        :return:
            A list of users.
        """
        ut = {
            'creator' : 'creatorUT',
            'viewer' : 'viewerUT'
        }
        if user_type and \
           user_type.lower() in ut:
            user_type = ut[user_type.lower()]
        if query is None:
            users = self._portal.get_org_users(max_users,
                                               exclude_system=json.dumps(exclude_system),
                                               user_type=user_type,
                                               role=role)
            gis = self._gis
            user_storage = []
            for u in users:
                if 'id' in u and \
                   (u['id'] is None or u['id'] == 'null'):
                    un = u['username']
                elif 'id' not in u:
                    un = u['username']
                else:
                    un = u['id']
                if not 'roleId' in u:
                    u['roleId'] = u.pop('role', None)
                user_storage.append(User(gis, un, u))
            return user_storage
        else:
            userlist = []
            users = self._portal.search_users(query, sort_field, sort_order,
                                              max_users, outside_org, json.dumps(exclude_system),
                                              user_type=user_type, role=role)
            for user in users:
                if 'id' in user and \
                   (user['id'] is None or user['id'] == 'null'):
                    un = user['username']
                elif self._gis.version <= [6,4]:
                    un = user['username']
                elif 'id' not in user:
                    un = user['username']
                else:
                    un = user['id']
                userlist.append(User(self._gis, un, userdict=user))
            return userlist
    #----------------------------------------------------------------------
    @property
    def me(self):
        """ Gets the logged in user.
        """
        if self._me is None:
            meuser = self._portal.logged_in_user()
            if meuser is not None:
                self._me = User(self._gis, meuser['username'], meuser)
            else:
                self._me = None
        return self._me
    #----------------------------------------------------------------------
    @_lazy_property
    def roles(self):
        """Helper object to manage custom roles for users"""
        return RoleManager(self._gis)

    #----------------------------------------------------------------------
    def user_groups(self, users, max_results=-1):
        """
        Givens a List of Users, the `user_groups` will report back all group ids
        that each user belongs to.  This method is designed to be a reporting
        tool for administrators so they can easily manage a user or users groups.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        users             Required List. An array of User objects or usernames.
        ----------------  --------------------------------------------------------
        max_results       Optional Integer. A limitor on the number of groups
                          returned for each user.
        ----------------  --------------------------------------------------------

        :returns: List of dictionaries.

        """
        if max_results == -1 or\
           max_results is None:
            max_results = None
        else:
            max_results = int(max_results)

        us = []
        for user in users:
            if isinstance(user, User):
                us.append(user.username)
            else:
                us.append(user)
        params = {
            'f' : 'json',
            "users": ",".join(us),
            "limit": max_results
        }
        url = "{base}/portals/self/usersGroups".format(base=self._portal.resturl)
        res = res = self._portal.con.get(url, params)
        if 'results' in res:
            return res['results']
        return res

class RoleManager(object):
    """Helper class to manage custom :class:`roles <arcgis.gis.Role>` for users in a GIS."""

    def __init__(self, gis):
        """Creates helper object to manage custom roles in the GIS"""
        self._gis = gis
        self._portal = gis._portal


    def create(self, name, description, privileges=None):
        """Creates a custom role with the specified parameters.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The custom role's name.
        ------------------     --------------------------------------------------------------------
        description            Required string. The custom role's description.
        ------------------     --------------------------------------------------------------------
        privileges             Optional string. An array of strings with predefined permissions within
                               each privilege.  For supported privileges see
                               http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Privileges/02r3000002wq000000/
        ==================     ====================================================================


        :return:
           The custom :class:`role <arcgis.gis.Role>` if successfully created, None if unsuccessful.
        """
        if self.exists(role_name=name) == False:
            role_id = self._portal.create_role(name, description)
            if role_id is not None:
                role_data = {
                    "id": role_id,
                    "name": name,
                  "description": description
                }
                role = Role(self._gis, role_id, role_data)
                role.privileges = privileges
                return role
            else:
                return None
        else:
            n = str(name.lower())
            roles = [r for r in self.all() \
                     if r.name.lower() == n]
            return roles[0]
        return None

    def exists(self, role_name):
        """
        Checks to see if a role exists given the declared role name.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        role_name              Required string. The name of the role to determine if it exists or not.
        ==================     ====================================================================

        :return:
           True if the :class:`role <arcgis.gis.Role>` exists, and False if it does not.
        """
        for role in self.all():
            if role.name.lower() == role_name.lower():
                return True
        return False

    def all(self, max_roles=1000):
        """
        Provides the list of all non-default roles in the GIS. (The ``org_admin``, ``org_user``,
        and ``org_publisher`` roles are not returned)

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        max_roles              Required integer. The maximum number of roles to be returned, defaults to 1000.
        ==================     ====================================================================

        :return:
           The list of all non-default :class:`roles <arcgis.gis.Role>` in the GIS.
        """
        roles = self._portal.get_org_roles(max_roles)
        return [Role(self._gis, role['id'], role) for role in roles]


    def get_role(self, role_id):
        """
        Retrieves the role with the specified custom roleId.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        role_id                Required string. The role ID of the custom role to get.
        ==================     ====================================================================

        :return:
           The :class:`Role <arcgis.gis.Role>` object associated with the specified role ID
        """
        role = self._portal.con.post('portals/self/roles/' + role_id, self._portal._postdata())
        return Role(self._gis, role['id'], role)


class Role(object):
    """A custom role in the GIS."""
    def __init__(self, gis, role_id, role):
        """Create a custom role"""
        self._gis = gis
        self._portal = gis._portal
        self.role_id = role_id
        if role is not None:
            self._name = role['name']
            self._description = role['description']

    def __repr__(self):
        return '<Role name: ' + self.name + ', description: ' + self.description + '>'

    def ___str___(self):
        return 'Custom Role name: ' + self.name + ', description: ' + self.description

    @property
    def name(self):
        """Gets and sets the name of the custom role."""
        return self._name

    @name.setter
    def name(self, value):
        """Name of the custom role"""
        self._name = value
        self._update_role()

    @property
    def description(self):
        """Gets and sets the description of the custom role."""
        return self._description

    @description.setter
    def description(self, value):
        """Description of the custom role"""
        self._description = value
        self._update_role()

    def _update_role(self):
        """Updates the name or description of this role"""
        postdata = self._portal._postdata()
        postdata['name'] = self._name
        postdata['description'] = self._description

        resp = self._portal.con.post('portals/self/roles/' + self.role_id + '/update', postdata)
        if resp:
            return resp.get('success')

    @property
    def privileges(self):
        """
        Get or sets the privileges for the custom role as a list of strings.

        Supported privileges with predefined permissions are:

        *Administrative Privileges:*

        Members

        - portal:admin:viewUsers: grants the ability to view full member account information within organization.
        - portal:admin:updateUsers: grants the ability to update member account information within organization.
        - portal:admin:deleteUsers: grants the ability to delete member accounts within organization.
        - portal:admin:inviteUsers: grants the ability to invite members to organization. (This privilege is only applicable to ArcGIS Online.)
        - portal:admin:disableUsers: grants the ability to enable and disable member accounts within organization.
        - portal:admin:changeUserRoles: grants the ability to change the role a member is assigned within organization; however, it does not grant the ability to promote a member to, or demote a member from, the Administrator role. That privilege is reserved for the Administrator role alone.
        - portal:admin:manageLicenses: grants the ability to assign licenses to members of organization.
        - portal:admin:reassignUsers: grants the ability to assign all groups and content of a member to another within organization.

        Groups

        - portal:admin:viewGroups: grants the ability to view all groups within organization.
        - portal:admin:updateGroups: grants the ability to update groups within organization.
        - portal:admin:deleteGroups: grants the ability to delete groups within organization.
        - portal:admin:reassignGroups: grants the ability to reassign groups to other members within organization.
        - portal:admin:assignToGroups: grants the ability to assign members to, and remove members from, groups within organization.
        - portal:admin:manageEnterpriseGroups: grants the ability to link group membership to an enterprise group. (This privilege is only applicable to Portal for ArcGIS.)

        Content

        - portal:admin:viewItems: grants the ability to view all content within organization.
        - portal:admin:updateItems: grants the ability to update content within organization.
        - portal:admin:deleteItems: grants the ability to delete content within organization.
        - portal:admin:reassignItems: grants the ability to reassign content to other members within organization.
        - portal:admin:shareToGroup: grants the ability to share other member's content to groups the user belongs to.
        - portal:admin:shareToOrg: grants the ability to share other member's content to organization.
        - portal:admin:shareToPublic: grants the ability to share other member's content to all users of the portal.

        ArcGIS Marketplace Subscriptions

        - marketplace:admin:purchase: grants the ability to request purchase information about apps and data in ArcGIS Marketplace. (This privilege is only applicable to ArcGIS Online.)
        - marketplace:admin:startTrial: grants the ability to start trial subscriptions in ArcGIS Marketplace. (This privilege is only applicable to ArcGIS Online.)
        - marketplace:admin:manage: grants the ability to create listings, list items and manage subscriptions in ArcGIS Marketplace. (This privilege is only applicable to ArcGIS Online.)

        *Publisher Privileges:*

        Content

        - portal:publisher:publishFeatures: grants the ability to publish hosted feature layers from shapefiles, CSVs, etc.
        - portal:publisher:publishTiles: grants the ability to publish hosted tile layers from tile packages, features, etc.
        - portal:publisher:publishScenes: grants the ability to publish hosted scene layers.

        *User Privileges:*

        Groups

        - portal:user:createGroup: grants the ability for a member to create, edit, and delete their own groups.
        - portal:user:joinGroup: grants the ability to join groups within organization.
        - portal:user:joinNonOrgGroup: grants the ability to join groups external to the organization. (This privilege is only applicable to ArcGIS Online.)

        Content

        - portal:user:createItem: grants the ability for a member to create, edit, and delete their own content.

        Sharing

        - portal:user:shareToGroup: grants the ability to share content to groups.
        - portal:user:shareToOrg: grants the ability to share content to organization.
        - portal:user:shareToPublic: grants the ability to share content to all users of portal.
        - portal:user:shareGroupToOrg: grants the ability to make groups discoverable by the organization.
        - portal:user:shareGroupToPublic: grants the ability to make groups discoverable by all users of portal.

        Premium Content

        - premium:user:geocode: grants the ability to perform large-volume geocoding tasks with the Esri World Geocoder such as publishing a CSV of addresses as hosted feature layer.
        - premium:user:networkanalysis: grants the ability to perform network analysis tasks such as routing and drive-time areas.
        - premium:user:geoenrichment: grants the ability to geoenrich features.
        - premium:user:demographics: grants the ability to make use of premium demographic data.
        - premium:user:spatialanalysis: grants the ability to perform spatial analysis tasks.
        - premium:user:elevation: grants the ability to perform analytical tasks on elevation data.

        Features

        - features:user:edit: grants the ability to edit features in editable layers, according to the edit options enabled on the layer.
        - features:user:fullEdit: grants the ability to add, delete, and update features in a hosted feature layer regardless of the editing options enabled on the layer.

        Open Data

        - opendata:user:openDataAdmin: grants the ability to manage Open Data Sites for the organization. (This privilege is only applicable to ArcGIS Online.)
        - opendata:user:designateGroup: grants the ability to designate groups within organization as being available for use in Open Data. (This privilege is only applicable to ArcGIS Online.)

        """
        resp = self._portal.con.post('portals/self/roles/' + self.role_id + '/privileges', self._portal._postdata())
        if resp:
            return resp.get('privileges')
        else:
            return None

    @privileges.setter
    def privileges(self, value):
        """Privileges for the custom role as a list of strings"""
        postdata = self._portal._postdata()
        postdata['privileges'] = { 'privileges' : value }

        resp = self._portal.con.post('portals/self/roles/' + self.role_id + '/setPrivileges', postdata)
        if resp:
            return resp.get('success')

    def delete(self):
        """Deletes this role.

        :return:
           A boolean indicating success (True) or failure (False).
        """
        resp = self._portal.con.post('portals/self/roles/' + self.role_id + '/delete', self._portal._postdata())
        if resp:
            return resp.get('success')


class GroupManager(object):
    """
    Helper class for managing GIS groups. This class is not created by users directly.
    An instance of this class, called 'groups', is available as a property of the Gis object.
    Users call methods on this 'groups' object to manipulate (create, get, search, etc) users.
    """
    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal

    def create(self, title, tags, description=None,
               snippet=None, access='public', thumbnail=None,
               is_invitation_only=False, sort_field='avgRating',
               sort_order='desc', is_view_only=False, auto_join=False,
               provider_group_name=None, provider=None,
               max_file_size=None, users_update_items=False,
               display_settings=None, is_open_data=False,
               leaving_disallowed=False):
        """
        Creates a group with the values for any particular arguments that are specified.
        Only title and tags are required.


        ====================  =========================================================
        **Argument**          **Description**
        --------------------  ---------------------------------------------------------
        title                 Required string. The name of the group.
        --------------------  ---------------------------------------------------------
        tags                  Required string. A comma-delimited list of tags, or
                              list of tags as strings.
        --------------------  ---------------------------------------------------------
        description           Optional string. A detailed description of the group.
        --------------------  ---------------------------------------------------------
        snippet               Optional string.  A short snippet (<250 characters)
                              that summarizes the group.
        --------------------  ---------------------------------------------------------
        access                Optional string. Choices are private, public, or org.
        --------------------  ---------------------------------------------------------
        thumbnail             Optional string. URL or file location to a group image.
        --------------------  ---------------------------------------------------------
        is_invitation_only    Optional boolean. Defines whether users can join by
                              request. Default is False meaning users can ask to join
                              by request or join by invitation.
        --------------------  ---------------------------------------------------------
        sort_field            Optional string. Specifies how shared items with
                              the group are sorted.
        --------------------  ---------------------------------------------------------
        sort_order            Optional string.  Choices are asc or desc for ascending
                              or descending, respectively.
        --------------------  ---------------------------------------------------------
        is_view_only          Optional boolean. Defines whether the group is searchable.
                              Default is False meaning the group is searchable.
        --------------------  ---------------------------------------------------------
        auto_join             Optional boolean. Only applies to org accounts. If True,
                              this group will allow joining without requesting
                              membership approval. Default is False.

        --------------------  ---------------------------------------------------------
        provider_group_name   Optional string. The name of the domain group.
        --------------------  ---------------------------------------------------------
        provider              Optional string. Name of the provider.
        --------------------  ---------------------------------------------------------
        max_file_size         Optional integer.  This is the maximum file size allowed
                              be uploaded/shared to a group. Default value is: 1024000
        --------------------  ---------------------------------------------------------
        users_update_items    Optional boolean.  Members can update all items in this
                              group.  Updates to an item can include changes to the
                              item's description, tags, metadata, as well as content.
                              This option can't be disabled once the group has
                              been created. Default is False.
        --------------------  ---------------------------------------------------------
        display_settings      Optional String. Defines the default display for the
                              group page to show a certain type of items. The allowed
                              values are: `apps, all, files, maps, layers, scenes, tools`.
                              The default value is `all`.
        --------------------  ---------------------------------------------------------
        is_open_data          Optional Boolean. Defines whether the group can be used
                              in the Open Data capabilities of ArcGIS Hub. The default
                              is False.
        --------------------  ---------------------------------------------------------
        leaving_disallowed    Optional boolean. Defines whether users are restricted
                              from choosing to leave the group. If True, only an
                              administrator can remove them from the group. The default
                              is False.
        ====================  =========================================================

        :return:
            The group if successfully created, None if unsuccessful.
        """
        display_settings_lu = {
            "apps" : {"itemTypes":"Application"},
            "all" : {"itemTypes":""},
            "files" : {"itemTypes":"CSV"},
            None : {"itemTypes":""},
            "maps" : {"itemTypes":"Web Map"},
            "layers" : {"itemTypes":"Layer"},
            "scenes" : {"itemTypes":"Web Scene"},
            "tools" : {"itemTypes":"Locator Package"}
        }
        if max_file_size is None:
            max_file_size = 1024000
        if users_update_items is None:
            users_update_items = False

        if type(tags) is list:
            tags = ",".join(tags)
        params = {
            'title' : title, 'tags' : tags, 'description' : description,
            'snippet' : snippet, 'access' : access, 'sortField' : sort_field,
            'sortOrder' : sort_order, 'isViewOnly' : is_view_only,
            'isinvitationOnly' : is_invitation_only,
            'autoJoin': auto_join,
            'leavingDisallowed': leaving_disallowed
        }
        if provider_group_name:
            params['provider'] = provider
            params['providerGroupName'] = provider_group_name
        if users_update_items == True:
            params['capabilities'] = "updateitemcontrol"
        else:
            params['capabilities'] = ""
        params['isOpenData'] = is_open_data
        params['MAX_FILE_SIZE'] = max_file_size
        if isinstance(display_settings, str) and display_settings.lower() in display_settings_lu:
            params['displaySettings'] = display_settings_lu[display_settings.lower()]
        elif display_settings is None:
            params['displaySettings'] = display_settings_lu[display_settings]
        else:
            raise ValueError("Display settings must be set to a valid value.")
        #if self._gis.version >= [8,2] and display_settings:
        #    params["itemTypes"] = display_settings
        group = self._portal.create_group_from_dict(params, thumbnail)

        if group is not None:
            return Group(self._gis, group['id'], group)
        else:
            return None

    def create_from_dict(self, dict):
        """
        Creates a group via a dictionary with the values for any particular arguments that are specified.
        Only title and tags are required.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        dict                   Required dictionary. A dictionary of entries to create/define the
                               group.  See help of the create() method for parameters.
        ==================     ====================================================================


        :return:
            The group if successfully created, None if unsuccessful.
        """
        thumbnail = dict.pop("thumbnail", None)

        if 'tags' in dict:
            if type(dict['tags']) is list:
                dict['tags'] = ",".join(dict['tags'])

        group = self._portal.create_group_from_dict(dict, thumbnail)
        if group is not None:
            return Group(self._gis, group['id'], group)
        else:
            return None

    def get(self, groupid):
        """
        Returns the group object for the specified groupid.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        groupid                Required string. The group identifier.
        ==================     ====================================================================


        :return:
           The group object if the group is found, None if it is not found.
        """
        try:
            group = self._portal.get_group(groupid)
        except RuntimeError as re:
            if re.args[0].__contains__("Group does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as re:
            if re.args[0].__contains__("Group does not exist or is inaccessible"):
                return None
            else:
                raise re

        if group is not None:
            return Group(self._gis, groupid, group)
        return None

    def search(self, query='', sort_field='title', sort_order='asc',
               max_groups=1000, outside_org=False, categories=None):
        """
        Searches for portal groups.

        .. note::
            A few things that will be helpful to know.

            1. The query syntax has many features that can't
                be adequately described here.  The query syntax is
               available in ArcGIS Help.  A short version of that URL
                is http://bitly.com/1fJ8q31.

            2. Searching without specifying a query parameter returns
               a list of all groups in your organization.

            3. Most of the time when searching for groups, you'll want to
                search within your organization in ArcGIS Online
                or within your Portal.  As a convenience, the method
                automatically appends your organization id to the query by
                default.  If you don't want the API to append to your query
                set outside_org to True.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        query             Optional string on Portal, or required string for ArcGIS Online.
                          If not specified, all groups will be searched. See notes above.
        ----------------  --------------------------------------------------------
        sort_field        Optional string. Valid values can be title, owner,
                          created.
        ----------------  --------------------------------------------------------
        sort_order        Optional string. Valid values are asc or desc.
        ----------------  --------------------------------------------------------
        max_groups        Optional integer. Maximum number of groups returned, default is 1,000.
        ----------------  --------------------------------------------------------
        outside_org       Optional boolean. Controls whether to search outside
                          your org. Default is False, do not search ourside your org.
        ----------------  --------------------------------------------------------
        categories        Optional string or list. A string of category values.
        ================  ========================================================


        :return:
           A list of groups matching the specified query.
        """
        grouplist = []
        groups = self._portal.search_groups(query, sort_field, sort_order, max_groups, outside_org, categories)
        for group in groups:
            grouplist.append(Group(self._gis, group['id'], group))
        return grouplist


def _is_shapefile(data):
    try:
        if zipfile.is_zipfile(data):
            zf = zipfile.ZipFile(data, 'r')
            namelist = zf.namelist()
            for name in namelist:
                if name.endswith('.shp') or name.endswith('.SHP'):
                    return True
        return False
    except:
        return False


class ContentManager(object):
    """
    Helper class for managing content in ArcGIS Online or ArcGIS Enterprise.
    This class is not created by users directly. An instance of this class,
    called 'content', is available as a property of the GIS object. Users
    call methods on this 'content' object to manipulate (create, get, search,
    etc) items.
    """
    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal

    def _add_by_part(self, file_path, itemid, item_properties, size=1e7, owner=None, folder=None):
        """
        Performs a special add operation that chunks up a file and loads it piece by piece.
        This is an internal method used by `add`

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        file_path           Required String.  The path to the file to load into Portal.
        ---------------     --------------------------------------------------------------------
        itemid              Required String. The unique ID of the Item to load the data to.
        ---------------     --------------------------------------------------------------------
        item_properties     Required Dict.  The properties for the item.
        ---------------     --------------------------------------------------------------------
        size                Optional Integer.  The chunk size off the parts in bytes.  The
                            smallest size allowed is 5 MB or 5e6.
        ---------------     --------------------------------------------------------------------
        multipart           Optional Boolean.  Loads a file by chunks to the Enterprise. The
                            default is False.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder where placing item.
        ===============     ====================================================================


        """
        if size < 5e6:
            size = 5e6
        def read_in_chunks(file_object, chunk_size=10000000):
            """Generate file chunks of 10MB"""
            while True:
                data = file_object.read(chunk_size)
                if not data:
                    break
                yield data

        owner_name = owner
        if isinstance(owner, User):
            owner_name = owner.username

        # If owner isn't specified, use the logged in user
        if not owner_name:
            owner_name = self._gis.users.me.username

        # Setup the item path, including the folder
        path = 'content/users/' + owner_name
        if folder and folder != '/':
            folder_id = self._portal.get_folder_id(owner_name, folder)
            path += '/' + folder_id

        url = "{base}{path}/items/{itemid}/addPart".format(base=self._gis._portal.resturl,
                                                                          path=path,
                                                                          itemid=itemid)
        file = {'file': None}
        params = {
            'f': 'json',
            'partNum' : None
        }
        messages = []
        future_files = []
        with open(file_path, 'rb') as f:
            import copy
            import uuid
            import concurrent.futures
            nthreads = 5
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as tp:
                future_parts = {}
                base_file = uuid.uuid4().hex[:4]
                for part_num, piece in enumerate(read_in_chunks(f), start=1):
                    params['partNum'] = part_num
                    temp_file = os.path.join(tempfile.gettempdir(), "split%s.part%s" % (base_file,part_num))
                    kwargs = {
                        "path" : url,
                        "postdata" : copy.copy(params),
                        'files' : {'file' : copy.copy(temp_file)}
                    }
                    with open(temp_file, 'wb') as writer:
                        writer.write(piece)
                        del writer
                    future_parts[tp.submit(self._gis._con.post, **kwargs)] = part_num
                    future_files.append(copy.copy(temp_file))
                concurrent.futures.wait(list(future_parts.keys()), None, concurrent.futures.ALL_COMPLETED)
                for future in concurrent.futures.as_completed(future_parts):
                    part_num = future_parts[future]
                    try:
                        if future.done():
                            data = future.result()
                            if 'success' in data:
                                messages.append(data['success'])
                            else:
                                messages.append(False)
                    except Exception as exc:
                        _log.error('%r generated an exception: %s' % (url, exc))
                    else:
                        _log.debug('%r page is %s' % (url, data))
                for ffile in future_files:
                    if os.path.isfile(ffile):
                        os.remove(ffile)

        if all(messages):
            # commit the addition
            url = "{base}{path}/items/{itemid}/commit".format(base=self._gis._portal.resturl,
                                                                             path=path,
                                                                             itemid=itemid)
            params = {
                'f' : "json",
                'id' : itemid,
                'type' : item_properties['type'],
                'async' : True
            }
            params.update(item_properties)
            res = self._gis._con.post(url, params)
            if 'success' in res:
                url = "{base}{path}/items/{itemid}/status".format(base=self._gis._portal.resturl,
                                                                                path=path,
                                                                                itemid=itemid)
                import time
                params = {'f' : 'json'}
                res = self._gis._portal.con.post(url, params)
                while res["status"] != "completed":
                    if 'fail' in res['status']:
                        return False
                    time.sleep(1)
                    res = self._gis._portal.con.post(url, {'f': 'json'})

                return res['status']
            else:
                return False
        return False

    #----------------------------------------------------------------------
    def can_delete(self, item):
        """
        The 'can_delete' Item indicates whether an item can be erased or
        not. When the returned response from 'can_delete' Item is true, the
        item can be safely removed. When the returned response is false,
        the item cannot be deleted due to a dependency or protection
        setting.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required `Item`. The `Item` to be erased.
        ===============     ====================================================================

        :returns: Dict


        ===============     ====================================================================
        **Status**          **Response**
        ---------------     --------------------------------------------------------------------
        success             {
                            "itemId": "e03f626be86946f997c29d6dfc7a9666",
                            "success": True
                            }

        ---------------     --------------------------------------------------------------------
        failure             {
                            "itemId": "a34c2e6711494e62b3b8d7452d4d6235",
                            "success": false,
                            "reason": {
                            "message": "Unable to delete item. Delete protection is turned on."
                            }
                            }

        ===============     ====================================================================

        """
        params = {'f' : 'json'}
        url = "{resturl}content/users/{username}/items/{itemid}/canDelete".format(resturl=self._portal.resturl,
                                                                           username=item.owner,
                                                                           itemid=item.itemid)
        try:
            res = self._portal.con.post(url, params)
            return res
        except Exception as e:
            return {
                "itemId": item.itemid,
                "success": False,
                "reason": {
                    "message": "{msg}".format(msg=e.args[0])
                }
            }
        return False
    #----------------------------------------------------------------------
    def add(self, item_properties, data=None, thumbnail=None,
            metadata=None, owner=None, folder=None, item_id=None):
        """ Adds content to the GIS by creating an item.

        .. note::
            Content can be a file (such as a service definition, shapefile,
            CSV, layer package, file geodatabase, geoprocessing package,
            map package) or it can be a URL (to an ArcGIS Server service,
            WMS service, or an application).

            If you are uploading a package or other file, provide a path or
            URL to the file in the data argument.

            From a technical perspective, none of the item_properties (see
            table below *Key:Value Dictionary Options for Argument
            item_properties*) are required.  However, it is strongly
            recommended that arguments title, type, typeKeywords, tags,
            snippet, and description be provided.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        data                Optional string. Either a path or URL to the data.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder where placing item.
        ---------------     --------------------------------------------------------------------
        item_id             Optionl String. **Available in Enterprise/AGOL 10.8.1+**.  A string
                            of 32 character UID without any special characters.

                            If the `item_id` is already being used, an error will be raised
                            during the `add` process.

                            Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        ===============     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*


        ==========================  =====================================================================
        **Key**                     **Value**
        --------------------------  ---------------------------------------------------------------------
        type                        Optional string. Indicates type of item, see URL 1 below for valid values.
        --------------------------  ---------------------------------------------------------------------
        dataUrl                     Optional string. The Url of the data stored on cloud storage. If given, filename is required.
        --------------------------  ---------------------------------------------------------------------
        filename                    Optional string. The name of the file on cloud storage.  This is required is dataUrl is used.
        --------------------------  ---------------------------------------------------------------------
        typeKeywords                Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        --------------------------  ---------------------------------------------------------------------
        description                 Optional string. Description of the item.
        --------------------------  ---------------------------------------------------------------------
        title                       Optional string. Name label of the item.
        --------------------------  ---------------------------------------------------------------------
        url                         Optional string. URL to item that are based on URLs.
        --------------------------  ---------------------------------------------------------------------
        text                        Optional string. For text based items such as Feature Collections & WebMaps
        --------------------------  ---------------------------------------------------------------------
        tags                        Optional string. Tags listed as comma-separated values, or a list of strings.
                                    Used for searches on items.
        --------------------------  ---------------------------------------------------------------------
        snippet                     Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        --------------------------  ---------------------------------------------------------------------
        extent                      Optional string. Provide comma-separated values for min x, min y, max x, max y.
        --------------------------  ---------------------------------------------------------------------
        spatialReference            Optional string. Coordinate system that the item is in.
        --------------------------  ---------------------------------------------------------------------
        accessInformation           Optional string. Information on the source of the content.
        --------------------------  ---------------------------------------------------------------------
        licenseInfo                 Optional string.  Any license information or restrictions regarding the content.
        --------------------------  ---------------------------------------------------------------------
        culture                     Optional string. Locale, country and language information.
        --------------------------  ---------------------------------------------------------------------
        commentsEnabled             Optional boolean. Default is true, controls whether comments are allowed (true)
                                    or not allowed (false).
        --------------------------  ---------------------------------------------------------------------
        culture                     Optional string. Language and country information.
        --------------------------  ---------------------------------------------------------------------
        overwrite                   Optional boolean. Default is `false`. Controls whether item can be overwritten.
        ==========================  =====================================================================


        URL 1:  `Item and Item Types <https://developers.arcgis.com/rest/users-groups-and-items/items-and-item-types.htm>`_

        :return:
           The item if successfully added, None if unsuccessful.
        """
        import os
        filetype = None
        if not isinstance(item_properties, dict):
            raise ValueError("`item_properties` must be  dictionary.")
        if item_id and isinstance(item_id, str) and len(item_id) == 32:
            item_properties['itemIdToCreate'] = item_id

        if data is not None:
            title = os.path.splitext(os.path.basename(data))[0]
            extn = os.path.splitext(os.path.basename(data))[1].upper()

            filetype = None
            if (extn == 'GPKG'):
                filetype = "GeoPackage"
            elif (extn == '.CSV'):
                filetype = 'CSV'
            elif (extn == '.SD'):
                filetype = 'Service Definition'
            elif title.upper().endswith('.GDB'):
                filetype = 'File Geodatabase'
            elif (extn in ('.SLPK', '.SPK')):
                filetype = 'Scene Package'
            elif (extn in ('.LPK', '.LPKX')):
                filetype = 'Layer Package'
            elif (extn in ('.GPK', '.GPKX')):
                filetype = 'Geoprocessing Package'
            elif (extn == '.GCPK'):
                filetype = 'Locator Package'
            elif (extn == '.TPK'):
                filetype = 'Tile Package'
            elif (extn in ('.MPK', '.MPKX')):
                filetype = 'Map Package'
            elif (extn == '.MMPK'):
                filetype = 'Mobile Map Package'
            elif (extn == '.APTX'):
                filetype = 'Project Template'
            elif (extn == '.VTPK'):
                filetype = 'Vector Tile Package'
            elif (extn == '.PPKX'):
                filetype = 'Project Package'
            elif (extn == '.RPK'):
                filetype = 'Rule Package'
            elif (extn == '.MAPX'):
                filetype = 'Pro Map'

            if _is_shapefile(data):
                filetype = 'Shapefile'

            if not 'type' in item_properties:
                if filetype is not None:
                    item_properties['type'] = filetype
                else:
                    raise RuntimeError('Specify type in item_properties')
            if not 'title' in item_properties:
                item_properties['title'] = title
        if 'type' in item_properties and \
           item_properties['type'] == "WMTS" and \
           'text' not in item_properties:
            from arcgis.mapping.ogc import WMTSLayer
            item_properties['text'] = json.dumps(WMTSLayer(item_properties['url'], gis=self._gis).__text__)

        owner_name = owner
        if isinstance(owner, User):
            owner_name = owner.username

        if 'tags' in item_properties:
            if type(item_properties['tags']) is list:
                item_properties['tags'] = ",".join(item_properties['tags'])
        try:
            from arcgis._impl.common._utils import bytesto
            is_file = os.path.isfile(data)
            if is_file and \
               bytesto(os.stat(data).st_size) < 7:
                multipart = False
                item_properties.pop('multipart', None)
            else:
                if 'multipart' in item_properties:
                    item_properties['multipart'] = True
                multipart = True
        except:
            is_file = False
            multipart = False
            item_properties.pop('multipart', None)
        if multipart and \
           is_file:
            import copy
            item_properties['multipart'] = True
            params = {}
            params.update(item_properties)
            params['fileName'] = os.path.basename(data)
            # Create an empty Item
            itemid = self._portal.add_item(params, None,
                                           thumbnail, metadata,
                                           owner_name, folder)
            # check the status and commit the final result
            status = self._add_by_part(
                file_path=data,
                itemid=itemid,
                item_properties=item_properties,
                size=1e7,
                owner=owner_name,
                folder=folder)

            # Update the thumbnail and return the item
            item = Item(gis=self._gis, itemid=itemid)
            if item.type == "KML":
                item.update({'url' : f"{self._gis._portal.resturl}content/items/{item.itemid}/data"})
            item.update(thumbnail=thumbnail)
            return item
        else:
            if filetype:
                item_properties['fileName'] = os.path.basename(data)

            itemid = self._portal.add_item(item_properties, data,
                                           thumbnail, metadata,
                                           owner_name, folder)

        if itemid is not None:
            item = Item(self._gis, itemid)
            if item.type == "KML":
                item.update({'url' : f"{self._gis._portal.resturl}content/items/{item.itemid}/data"})
            return item
        else:
            return None
    #----------------------------------------------------------------------
    def analyze(self,
                url=None,
                item=None,
                file_path=None,
                text=None,
                file_type=None,
                source_locale='en',
                geocoding_service=None,
                location_type=None,
                source_country='world',
                country_hint=None
                ):
        """
        The Analyze call helps a client analyze a CSV or Excel file (.xlsx, .xls) prior to publishing or generating features using the Publish or Generate operation, respectively.

        Analyze returns information about the file including the fields present as well as sample records. Analyze attempts to detect the presence of location fields that may be present as either X,Y fields or address fields.

        Analyze packages its result so that publishParameters within the JSON response contains information that can be passed back to the server in a subsequent call to Publish or Generate. The publishParameters subobject contains properties that describe the resulting layer after publishing, including its fields, the desired renderer, and so on. Analyze will suggest defaults for the renderer.

        In a typical workflow, the client will present portions of the Analyze results to the user for editing before making the call to Publish or Generate.

        If the file to be analyzed currently exists in the portal as an item, callers can pass in its itemId. Callers can also directly post the file. In this case, the request must be a multipart post request pursuant to IETF RFC1867. The third option for text files is to pass the text in as the value of the text parameter.

        =======================    =============================================================
        **Argument**               **Description**
        -----------------------    -------------------------------------------------------------
        url                        optional string. The URL of the csv file.
        -----------------------    -------------------------------------------------------------
        item                       optional string/Item. The ID or Item of the item to be
                                   analyzed.
        -----------------------    -------------------------------------------------------------
        file_path                  optional string. The file to be analyzed.
        -----------------------    -------------------------------------------------------------
        text                       optional string. The text in the file to be analyzed.
        -----------------------    -------------------------------------------------------------
        file_type                  optional string. The type of the input file: shapefile, csv, excel,
                                   or geoPackage (Added ArcGIS API for Python 1.8.3+).
        -----------------------    -------------------------------------------------------------
        source_locale              optional string. The locale used for the geocoding service source.
        -----------------------    -------------------------------------------------------------
        geocoding_service          optional string/geocoder. The URL of the service.
        -----------------------    -------------------------------------------------------------
        location_type              optional string. Indicates the type of spatial information stored in the dataset.

                                   Values for CSV: coordinates | address | lookup | none
                                   Values for Excel: coordinates | address | none
        -----------------------    -------------------------------------------------------------
        source_country             optional string. The two character country code associated with the geocoding service, default is "world".
        -----------------------    -------------------------------------------------------------
        country_hint               optional string. If first time analyzing, the hint is used. If source country is already specified than sourcecountry is used.
        =======================    =============================================================

        :returns: dictionary

        """
        surl = "%s/sharing/rest/content/features/analyze" % self._gis._url
        params = {
            'f' : 'json',
            'analyzeParameters' : {}
        }
        files = None
        if not (text or file_path or item or url):
            return Exception("Must provide an itemid, file_path or text to analyze data.")
        if item:
            if isinstance(item, str):
                params['itemid'] = item
            elif isinstance(item, Item):
                params['itemid'] = item.itemid
        elif file_path and os.path.isfile(file_path):
            files = {'file' : file_path}
        elif text:
            params['text'] = text
        elif url:
            params['sourceUrl'] = url

        params['analyzeParameters']['sourcelocale'] = source_locale
        if geocoding_service:
            from arcgis.geocoding._functions import Geocoder
            if isinstance(geocoding_service, Geocoder):
                params['analyzeParameters']['geocodeServiceUrl'] = geocoding_service.url
            else:
                params['analyzeParameters']['geocodeServiceUrl'] = geocoding_service
        if location_type:
            params['analyzeParameters']['locationType'] = location_type

        if file_type is None and \
           (url or file_path):
            d = url or file_path
            if d:
                if str(d).lower().endswith('.csv'):
                    params['fileType'] = 'csv'
                elif str(d).lower().endswith('.xls') or \
                     str(d).lower().endswith('.xlsx'):
                    params['fileType'] = 'excel'
                elif str(d).lower().endswith('gpkg'):
                    params['fileType'] = 'geoPackage'

        elif str(file_type).lower() in ['excel', 'csv']:
            params['fileType'] = file_type
        if source_country:
            params['analyzeParameters']['sourceCountry'] = source_country
        if country_hint:
            params['analyzeParameters']['sourcecountryhint'] = country_hint

        gis = self._gis
        params['analyzeParameters'] = json.dumps(params['analyzeParameters'])
        return gis._con.post(path=surl, postdata=params, files=files)
    #----------------------------------------------------------------------
    def create_service(self, name,
                       service_description="",
                       has_static_data=False,
                       max_record_count = 1000,
                       supported_query_formats = "JSON",
                       capabilities = None,
                       description = "",
                       copyright_text = "",
                       wkid=102100,
                       create_params=None,
                       service_type="featureService",
                       owner=None, folder=None,
                       item_properties=None,
                       is_view=False,
                       tags=None,
                       snippet=None,
                       item_id=None):
        """ Creates a service in the Portal.


        =======================    =============================================================
        **Argument**               **Description**
        -----------------------    -------------------------------------------------------------
        name                       Required string. The unique name of the service.
        -----------------------    -------------------------------------------------------------
        service_description        Optional string. Description of the service.
        -----------------------    -------------------------------------------------------------
        has_static_data            Optional boolean. Indicating whether the data can change.  Default is True, data is not allowed to change.
        -----------------------    -------------------------------------------------------------
        max_record_count           Optional integer. Maximum number of records in query operations.
        -----------------------    -------------------------------------------------------------
        supported_query_formats    Optional string. Formats in which query results are returned.
        -----------------------    -------------------------------------------------------------
        capabilities               Optional string. Specify service capabilities.
                                   If left unspecified, 'Image,Catalog,Metadata,Download,Pixels'
                                   are used for image services, and 'Query'
                                   is used for feature services, and 'Query' otherwise
        -----------------------    -------------------------------------------------------------
        description                Optional string. A user-friendly description for the published dataset.
        -----------------------    -------------------------------------------------------------
        copyright_text             Optional string. The copyright information associated with the dataset.
        -----------------------    -------------------------------------------------------------
        wkid                       Optional integer. The well known id (WKID) of the spatial reference for the service.
                                   All layers added to a hosted feature service need to have the same spatial
                                   reference defined for the feature service. When creating a new
                                   empty service without specifying its spatial reference, the spatial
                                   reference of the hosted feature service is set to the first layer added to that feature service.
        -----------------------    -------------------------------------------------------------
        create_params              Optional dictionary. Add all create_service parameters into a dictionary. If this parameter is used,
                                   all the parameters above are ignored.
        -----------------------    -------------------------------------------------------------
        service_type               Optional string. The type of service to be created.  Currently the options are imageService or featureService.
        -----------------------    -------------------------------------------------------------
        owner                      Optional string. The username of the owner of the service being created.
        -----------------------    -------------------------------------------------------------
        folder                     Optional string. The name of folder in which to create the service.
        -----------------------    -------------------------------------------------------------
        item_properties            Optional dictionary. See below for the keys and values
        -----------------------    -------------------------------------------------------------
        is_view                    Optional boolean. Indicating if the service is a hosted feature layer view
        -----------------------    -------------------------------------------------------------
        item_id                    Optionl String. **Available in Enterprise/AGOL 10.8.1+**.  A string
                                   of 32 character UID without any special characters.

                                   If the `item_id` is already being used, an error will be raised
                                   during the `add` process.

                                   Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        -----------------------    -------------------------------------------------------------
        tags                       Optional string. Tags listed as comma-separated values, or a list of strings.
                                   Used for searches on items.
        -----------------------    -------------------------------------------------------------
        snippet                    Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        =======================    =============================================================


        *Key:Value Dictionary Options for Argument item_properties*


        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        type               Optional string. Indicates type of item, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        url                Optional string. URL to item that are based on URLs.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        extent             Optional string. Provide comma-separated values for min x, min y, max x, max y.
        -----------------  ---------------------------------------------------------------------
        spatialReference   Optional string. Coordinate system that the item is in.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        -----------------  ---------------------------------------------------------------------
        commentsEnabled    Optional boolean. Default is true, controls whether comments are allowed (true)
                           or not allowed (false).
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Language and country information.
        =================  =====================================================================

        :return:
             The item for the service if successfully created, None if unsuccessful.
        """
        if capabilities is None:
            if service_type == 'imageService':
                capabilities = 'Image,Catalog,Metadata,Download,Pixels'
            elif service_type == 'featureService':
                capabilities = 'Query'
            else:
                capabilities = 'Query'
        if self._gis.version <= [7,1] and item_id:
            item_id = None
            import warnings
            warnings.warn("Item ID is not Support at this version. Please use version >=10.8.1 Enterprise.")
        itemid = self._portal.create_service(name,
                                             service_description,
                                             has_static_data,
                                             max_record_count,
                                             supported_query_formats,
                                             capabilities,
                                             description,
                                             copyright_text,
                                             wkid,
                                             service_type,
                                             create_params,
                                             owner, folder, item_properties,
                                             is_view, item_id, tags, snippet)
        if itemid is not None:
            item = Item(self._gis, itemid)
            if item_properties is None:
                item_properties = {}
            else:
                item.update(item_properties=item_properties)
            if 'access' in item_properties.keys():
                if item_properties['access'] == 'public':
                    item.share(everyone=True)
                elif item_properties['access'] == 'org':
                    item.share(org=True)
            return item
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def categories(self):
        """
        The category manager for items. See :class:`~arcgis.gis.CategorySchemaManager`.
        """

        base_url = "{base}portals/self".format(base=self._gis._portal.resturl)
        return CategorySchemaManager(base_url=base_url, gis=self._gis)
    #----------------------------------------------------------------------
    def get(self, itemid):
        """ Returns the item object for the specified itemid.


        =======================    =============================================================
        **Argument**               **Description**
        -----------------------    -------------------------------------------------------------
        itemid                     Required string. The item identifier.
        =======================    =============================================================

        :return:
            The item object if the item is found, None if the item is not found.
        """
        try:
            item = self._portal.get_item(itemid)
        except RuntimeError as re:
            if re.args[0].__contains__("Item does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as e:
            if e.args[0].__contains__("Item does not exist or is inaccessible"):
                return None
            else:
                raise e

        if item is not None:
            return Item(self._gis, itemid, item)
        return None

    def advanced_search(self, query, return_count=False, max_items=100, bbox=None,
                        categories=None, category_filter=None,
                        start=1, sort_field="title",
                        sort_order="asc", count_fields=None,
                        count_size=None, as_dict=False):
        """
        This method allows the ability to fully customize  the search experience.
        The `advanced_search` method allows users to control of the finer grained parameters
        not exposed by the 'search' method.  Additionally, it allows for the manual paging of
        information and how the data is returned.

        ================    ===============================================================
        **Argument**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String.  The search query.
        ----------------    ---------------------------------------------------------------
        bbox                Optional String/List. This is the xmin,ymin,xmax,ymax bounding
                            box to limit the search in.  Items like documents do not have
                            bounding boxes and will not be included in the search.
        ----------------    ---------------------------------------------------------------
        categories          Optional String. A comma separated list of up to 8 org content
                            categories to search items. Exact full path of each category is
                            required, OR relationship between the categories specified.

                            Each request allows a maximum of 8 categories parameters with
                            AND relationship between the different categories parameters
                            called.
        ----------------    ---------------------------------------------------------------
        category_filters    Optional String. A comma separated list of up to 3 category
                            terms to search items that have matching categories. Up to 2
                            `category_filters` parameter are allowed per request. It can
                            not be used together with categories to search in a request.
        ----------------    ---------------------------------------------------------------
        start               Optional Int. The starting position to search from.  This is
                            only required if paging is needed.
        ----------------    ---------------------------------------------------------------
        sort_field          Optional String. Responses from the `search` operation can be
                            sorted on various fields. `avgrating` is the default.
        ----------------    ---------------------------------------------------------------
        sort_order          Optional String. The sequence into which a collection of
                            records are arranged after they have been sorted. The allowed
                            values are: asc for ascending and desc for descending.
        ----------------    ---------------------------------------------------------------
        count_fields        Optional String. A comma separated list of fields to count.
                            Maximum count fields allowed per request is 3. Supported count
                            fields: `tags`, `type`, `access`, `contentstatus`, and
                            `categories`.
        ----------------    ---------------------------------------------------------------
        count_size          Optional Int. The maximum number of field values to count for
                            each `count_fields`. The default value is None, and maximum size
                            allowed is 200.
        ----------------    ---------------------------------------------------------------
        as_dict             Required Boolean. If True, the response comes back as a dictionary.
        ================    ===============================================================

        :returns: Depends on the inputs.
                  - Dictionary for a standard search
                  - `return_count`=True an integer is returned
                  - `count_fields` is specified a list of dicts for each field specified

        """
        from arcgis.gis._impl import _search
        stype = "content"
        group_id = None
        if max_items == -1:
            max_items = _search(gis=self._gis, query=query, stype=stype,
                          max_items=0, bbox=bbox,
                          categories=categories, category_filter=category_filter,
                          start=start, sort_field=sort_field,
                          sort_order=sort_order, count_fields=count_fields,
                          count_size=count_size, group_id=group_id, as_dict=as_dict)['total']
        so = {
            'asc' : 'asc',
            'desc' : 'desc',
            'ascending' : 'asc',
            'descending' : 'desc'
        }
        if sort_order:
            sort_order = so[sort_order]

        if count_fields or return_count:
            max_items = 0
        if max_items <= 100:
            res = _search(gis=self._gis, query=query, stype=stype,
                          max_items=max_items, bbox=bbox,
                          categories=categories, category_filter=category_filter,
                          start=start, sort_field=sort_field,
                          sort_order=sort_order, count_fields=count_fields,
                          count_size=count_size, group_id=group_id, as_dict=as_dict)
            if 'total' in res and \
               return_count:
                return res['total']
            elif 'aggregations' in res:
                return res['aggregations']
            return res
        else:
            allowed_keys = [ 'query', 'return_count', 'max_items', 'bbox','categories', 'category_filter',
                             'start', 'sort_field', 'sort_order', 'count_fields','count_size', 'as_dict']
            inputs = locals()
            kwargs = {}
            for k,v in inputs.items():
                if k in allowed_keys:
                    kwargs[k] = v
            import concurrent.futures
            import math, copy
            num = 100
            steps = range(math.ceil(max_items / num))
            params = [ ]
            for step in steps:
                new_start = start + num*step
                kwargs['max_items'] = num
                kwargs['start'] = new_start
                params.append(copy.deepcopy(kwargs))
            items = {
                'results' : [],
                'start' : start,
                'num' : 100,
                'total' : -999
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(self.advanced_search, **param): param for param in params}
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future_to_url[future]
                    data = future.result()
                    if 'results' in data:
                        items['results'].extend(data['results'])
            if len(items['results']) > max_items:
                items['results'] = items['results'][:max_items]
            return items

    def search(self,
               query, item_type=None,
               sort_field='avgRating', sort_order='desc',
               max_items=10, outside_org=False,
               categories=None,
               category_filters=None):
        """ Searches for portal items.

        .. note::
            A few things that will be helpful to know...

            1. The query syntax has many features that can't be adequately
               described here.  The query syntax is available in ArcGIS Help.
               A short version of that URL is http://bitly.com/1fJ8q31.

            2. Most of the time when searching for items, you'll want to
               search within your organization in ArcGIS Online
               or within your Portal.  As a convenience, the method
               automatically appends your organization id to the query by
               default.  If you want content from outside your organization
               set outside_org to True.

        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        query             Required string. A query string.  See notes above.
        ----------------  --------------------------------------------------------------------------
        item_type         Optional string. Set type of item to search.
                          http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000000ms000000
        ----------------  --------------------------------------------------------------------------
        sort_field        Optional string. Valid values can be title, uploaded, type, owner, modified,
                          avgRating, numRatings, numComments, and numViews.
        ----------------  --------------------------------------------------------------------------
        sort_order        Optional string. Valid values are asc or desc.
        ----------------  --------------------------------------------------------------------------
        max_items         Optional integer. Maximum number of items returned, default is 10.
        ----------------  --------------------------------------------------------------------------
        outside_org       Optional boolean. Controls whether to search outside your org (default is False, do not search ourside your org).
        ----------------  --------------------------------------------------------------------------
        categories        Optional string or list. A string of category values.
        ----------------  --------------------------------------------------------------------------
        category_filters  Optional string. A comma separated list of up to 3 category terms to
                          search items that have matching categories.

                          Up to 2 category_filters parameter are allowed per request. It can not be
                          used together with categories to search in a request.

        ================  ==========================================================================

        :return:
            A list of items matching the specified query.
        """
        if max_items > 10000:
            raise Exception(("Use `advanced_search` fo"
                             "r item queries over 10,000 Items."))
        itemlist = []
        if query is not None and query != '' and item_type is not None:
            query += ' AND '

        if item_type is not None:
            item_type = item_type.lower()
            if item_type == "web map":
                query += ' (type:"web map" NOT type:"web mapping application")'
            elif item_type == "web scene":
                query += ' (type:"web scene" NOT type:"CityEngine Web Scene")'
            elif item_type == "feature layer":
                query += ' (type:"feature service")'
            elif item_type == "geoprocessing tool":
                query += ' (type:"geoprocessing service")'
            elif item_type == "geoprocessing toolbox":
                query += ' (type:"geoprocessing service")'
            elif item_type == "feature layer collection":
                query += ' (type:"feature service")'
            elif item_type == "image layer":
                query += ' (type:"image service")'
            elif item_type == "imagery layer":
                query += ' (type:"image service")'
            elif item_type == "map image layer":
                query += ' (type:"map service")'
            elif item_type == "vector tile layer":
                query += ' (type:"vector tile service")'
            elif item_type == "scene layer":
                query += ' (type:"scene service")'
            elif item_type == "layer":
                query += ' (type:"layer" NOT type:"layer package" NOT type:"Explorer Layer")'
            elif item_type == "feature collection":
                query += ' (type:"feature collection" NOT type:"feature collection template")'
            elif item_type == "desktop application":
                query += ' (type:"desktop application" NOT type:"desktop application template")'
            else:
                query += ' (type:"' + item_type +'")'
        if isinstance(categories, list):
            categories = ",".join(categories)
        if not outside_org:
            accountid = self._gis.properties.get('id')
            if accountid and query:
                query += ' accountid:' + accountid
            elif accountid:
                query = 'accountid:' + accountid
        itemlist = self.advanced_search(query=query, max_items=max_items,
                             categories=categories,
                             start=1, sort_field=sort_field,
                             sort_order=sort_order)['results']
        return itemlist

    def create_folder(self, folder, owner=None):
        """
        Creates a folder with the given folder name, for the given owner. Does
        nothing if the folder already exists. If owner is not specified, owner
        is set as the logged in user.


        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        folder            Required string. The name of the folder to create for the owner.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user.
        ================  ==========================================================================

        :return:
            A json object like the following if the folder was created:
            {"username" : "portaladmin","id" : "bff13218991c4485a62c81db3512396f","title" : "testcreate"}; None otherwise.
        """
        if folder != '/': # we don't create root folder
            if owner is None:
                owner = self._portal.logged_in_user()['username']
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            if self._portal.get_folder_id(owner_name, folder) is None:
                return self._portal.create_folder(owner_name, folder)
            else:
                print('Folder already exists.')
        return None

    def rename_folder(self, old_folder, new_folder, owner=None):

        """
        Renames an existing folder from it's existing name to a new name.
        If owner is not specified, owner is set as the logged in user.


        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        old_folder        Required string. The name of the folder to rename for the owner.
        ----------------  --------------------------------------------------------------------------
        new_folder        Required string. The new name of the folder.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user.
        ================  ==========================================================================

        :return: Boolean

        """
        params = {
            'f' : 'json',
            'newTitle' : new_folder
        }
        if old_folder != '/': # we don't rename the root folder
            if owner is None:
                owner = self._portal.logged_in_user()['username']
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            folderid = self._portal.get_folder_id(owner_name, old_folder)
            if folderid is None:
                raise ValueError("Folder: %s does not exist." % old_folder)
            url = "{base}content/users/{user}/{folderid}/updateFolder".format(
                base=self._gis._portal.resturl,
                user=owner_name,
                folderid=folderid
            )
            res = self._gis._con.post(url, params)
            if 'success' in res:
                return res['success']
        return False


    def delete_items(self, items):
        """
        Deletes a collection of items from a users content.

        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        items             list of Item or Item Ids.  This is an array of items to be deleted from
                          the current user's content
        ================  ==========================================================================

        Returns: boolean. True on
        """
        if self._gis._portal.con.baseurl.endswith("/"):
            url = "%s/%s/%s/deleteItems" % (self._gis._portal.con.baseurl[:-1],
                                            "content/users",
                                            self._gis.users.me.username)
        else:
            url = "%s/%s/%s/deleteItems" % (self._gis._portal.con.baseurl,
                                            "content/users",
                                            self._gis.users.me.username)
        params = {
            'f' : 'json',
        'items' : ""
        }
        ditems = []
        for item in items:
            if isinstance(item, str):
                ditems.append(item)
            elif isinstance(item, Item):
                ditems.append(item.id)
            del item
        if len(ditems) > 0:
            params['items'] = ",".join(ditems)
            res = self._gis._con.post(path=url, postdata=params)
            return all([r['success'] for r in res['results']])
        return False


    def delete_folder(self, folder, owner=None):
        """
        Deletes a folder for the given owner (logged in user by default) with
        the given folder name.


        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        folder            Required string. The name of the folder to delete.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user is the default.
        ================  ==========================================================================

        :return:
            True if folder deletion succeeded, False if folder deletion failed.
        """
        if folder != '/':
            if owner is None:
                owner = self._portal.logged_in_user()['username']
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            return self._portal.delete_folder(owner_name, folder)
    #----------------------------------------------------------------------
    def _generate(self, gurl, params, files, gis):
        """
        private async logic for `generate`.
        """
        res = gis._con.post(gurl, params, files=files)
        if 'success' in res and res['success'] == False:
            raise Exception("Failed to Generate Features")

        if res['status']:
            item = gis.content.get(res['outputItemId'])
            status = item.status(res['jobId'], "generateFeatures")
            while status['status'] != 'completed':
                if status['status'] == 'failed':
                    try:
                        item.delete()
                        return status
                    except:
                        return status
                status = item.status(res['jobId'], "generateFeatures")
            item.update(item_properties={'title' : f"Generate Features: {res['jobId']}"})
            return item
        return res
    #----------------------------------------------------------------------
    def generate(self,
                 item=None,
                 file_path=None,
                 url=None,
                 text=None,
                 publish_parameters=None,
                 future=False):
        """
        The Generate call helps a client generate features from a CSV file, shapefile,
        GPX, or GeoJson file types.

        ===================  ==========================================================================
        **Argument**         **Description**
        -------------------  --------------------------------------------------------------------------
        item                 Optional Item. An `Item` on the current portal.
        -------------------  --------------------------------------------------------------------------
        file_path            Optional String. The file resource location on local disk.
        -------------------  --------------------------------------------------------------------------
        url                  Optional String. A web resource of a 'shapefile', 'csv', 'gpx' or 'geojson' file.
        -------------------  --------------------------------------------------------------------------
        text                 Optional String. The source text.
        -------------------  --------------------------------------------------------------------------
        publish_parameters   Optional Dict. A Python dictionary describing the layer and service to be created
                             as part of the `publish` operation. The appropriate key-value pairs
                             depend on the file type being published. For a complete
                             description, see the `Publish Item <https://developers.arcgis.com/rest/users-groups-and-items/publish-item.htm>`_
                             documentation in the REST API. Also see the :class:`~arcgis.gis.Item`
                             :meth:`~arcgis.gis.Item.publish` method. ``CSV``, ``Shapefiles`` and
                             ``GeoJSON`` file types must have publish parameters provided.
        -------------------  --------------------------------------------------------------------------
        future               Optional Boolean.  This allows the operation to run asynchronously allowing
                             the user to not pause the thread and continue to perform multiple operations.
                             The default is `True`.  When `True` the result of the method will be a
                             concurrent `Future` object.  The `result` of the method can be obtained
                             using the `result()` on the `Future` object.  When `False`, and Item is
                             returned. Future == True is only supported for 'shapefiles' and 'gpx' files.
        ===================  ==========================================================================

        :return: `Future` object when `future==True`,
                 `Item` when `future==False`,
                 `dict` of error messages on Exceptions

        """
        if item is None and \
           file_path is None and \
           text is None and \
           url is None:
            raise Exception("You must provide an item, file_path, text or url.")
        gurl = f"{self._gis._portal.resturl}content/features/generate"
        params = {
            "f" : "json",
            "itemid": "",
            "sourceUrl" : "",
            "text" : "",
            "filetype" : "",
            "publishParameters" : publish_parameters or "",
            'async' : True
        }
        files = None
        file_types = {
            '.gpx' : 'gpx',
            '.csv' : 'csv',
            '.zip' : 'shapefile',
            '.json' : 'geojson'
        }
        if item and item.type.lower() in ['shapefile', 'csv', 'gpx', 'geojson']:
            params['itemid'] = item.itemid
            if item.type.lower() == 'shapefile':
                params['filetype'] = 'shapefile'
                if publish_parameters in (None, "") and \
                   self._gis._portal.is_arcgisonline == False:
                    raise ValueError("A publish parameter is needed for this data type.")
            elif item.type.lower() == 'gpx':
                params['filetype'] = 'gpx'
            elif item.type.lower() == 'csv':
                params['filetype'] = 'csv'
                if publish_parameters is None:
                    raise ValueError("A publish parameter is needed for this data type.")
            elif item.type.lower() == 'geojson':
                params['filetype'] = 'geojson'
                if publish_parameters is None:
                    raise ValueError("A publish parameter is needed for this data type.")
            else:
                raise Exception(f"Invalid Item Type {item.type}")

        elif url:
            params['sourceUrl'] = url
            part = os.path.splitext(url)[-1]
            if part in file_types:
                params['filetype'] = file_types[part]
            else:
                raise Exception(f"Invalid file extension: {part}")
        elif file_path and os.path.isfile(file_path):
            files = []
            part = os.path.splitext(file_path)[-1]
            if part in file_types:
                params['filetype'] = file_types[part]
            else:
                raise Exception(f"Invalid file extension: {part}")
            if params['filetype'] in ['shapefile', 'csv', 'geojson'] and \
               self._gis._portal.is_arcgisonline == False and \
               params['publishParameters'] in (None, ""):
                raise ValueError("A publish parameter is needed for this data type.")
            files.append(('file', file_path, os.path.basename(file_path)))
        elif text:
            params['text'] = text
            params['fileType'] = 'csv'

        if future == True:
            executor =  concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(self._generate,
                                        **{"gurl" : gurl, "params":params,
                                           "files" : files, "gis" : self._gis})
            executor.shutdown(False)
            return futureobj
        else:
            params['async'] = False
            res = self._gis._con.post(gurl, params, files=files)
            return res
    #----------------------------------------------------------------------
    def import_data(self, df, address_fields=None, folder=None, item_id=None, **kwargs):
        """
        Imports a Pandas data frame (that has an address column), or an arcgis
        spatial dataframe into the GIS.

        Spatial dataframes are imported into the GIS and published as feature
        layers. Pandas dataframes that have an address column are imported as
        an in-memory feature collection.
        Note: By default, there is a limit of 1,000 rows/features for Pandas
        dataframes. This limit isn't there for spatial dataframes.

        ================  ==========================================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------------------------
        df                Required string. Pandas dataframe or arcgis.SpatialDataFrame
        ----------------  --------------------------------------------------------------------------
        address_fields    Optional dictionary. Dictionary containing mapping of df columns to address fields, eg: { "CountryCode" : "Country"} or { "Address" : "Address" }.
        ----------------  --------------------------------------------------------------------------
        folder            Optional string. Name of the folder where imported data would be stored.
        ----------------  --------------------------------------------------------------------------
        title             Optional string. Title of the item. This is used for spatial dataframe objects.
        ----------------  --------------------------------------------------------------------------
        tags              Optional string. Tags listed as comma-separated values, or a list of strings. Provide tags when publishing a spatial dataframe to the the GIS.
        ----------------  --------------------------------------------------------------------------
        item_id           Optionl String. **Available in Enterprise/AGOL 10.8.1+**.  A string
                          of 32 character UID without any special characters.

                          If the `item_id` is already being used, an error will be raised
                          during the `add` process.

                          Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        ================  ==========================================================================

        In addition to the parameters aboce, you can specify additional information to help publish CSV
        data.

        =====================  ==========================================================================
        **Optional Argument**  **Description**
        ---------------------  --------------------------------------------------------------------------
        location_type          Optional string. Indicates the type of spatial information stored in the
                               dataset.

                               Values for CSV:

                                  + coordinates
                                  + address (default)
                                  + lookup
                                  + none

                               Values for Excel:

                                  + coordinates
                                  + address (default)
                                  + none

                               When location_type = coordinates, the CSV or Excel data contains x,y
                               information.
                               When location_type = address, the CSV or Excel data contains address
                               fields that will be geocoded to a single point.
                               When location_type = lookup, the CSV or Excel data contains fields that
                               can be mapped to well-known sets of geographies.
                               When location_type = none, the CSV or Excel data contains no spatial
                               content and data will be loaded and subsequently queried as tabular data.

                               Based on this parameter, additional parameters will be required, for
                               example, when specifying location_type = coordinates, the latitude and
                               longitude field names must be specified.
        ---------------------  --------------------------------------------------------------------------
        latitude_field         Optional string. If location_type = coordinates, the name of the field that
                               contains the y coordinate.
        ---------------------  --------------------------------------------------------------------------
        longitude_field        Optional string. If location_type = coordinates, the name of the field that
                               contains the x coordinate.
        ---------------------  --------------------------------------------------------------------------
        coordinate_field_type  Optional string. Specify the type of coordinates that contain location
                               information. Values: LatitudeAndLongitude (default), MGRS, USNG
        ---------------------  --------------------------------------------------------------------------
        coordinate_field_name  Optional string. The name of the field that contains the coordinates
                               specified in coordinate_field_type
        ---------------------  --------------------------------------------------------------------------
        lookup_type            Optional string. The type of place to look up.
        ---------------------  --------------------------------------------------------------------------
        lookup_fields          Optional string. A JSON object with name value pairs that define the
                               fields used to look up the location.
        ---------------------  --------------------------------------------------------------------------
        geocode_url            Optional string. The URL of the geocoding service that supports batch
                               geocoding.
        ---------------------  --------------------------------------------------------------------------
        source_locale          Optional string. The locale used for the geocoding service source.
        ---------------------  --------------------------------------------------------------------------
        source_country         Optional string. The two character country code associated with the
                               geocoding service, default is 'world'.
        ---------------------  --------------------------------------------------------------------------
        country_hint           Optional string. If first time analyzing, the hint is used. If source
                               country is already specified than source_country is used.
        =====================  ==========================================================================


        When publishing a Spatial Dataframe, additional options can be given:

        =====================  ==========================================================================
        **Optional Argument**  **Description**
        ---------------------  --------------------------------------------------------------------------
        target_sr              optional integer.  WKID of the output data.  This is used when publishing
                               Spatial Dataframes to Hosted Feature Layers. The default is: 102100
        ---------------------  --------------------------------------------------------------------------
        title                  optional string. Name of the layer. The default is a random string.
        ---------------------  --------------------------------------------------------------------------
        tags                   optional string. Comma seperated strings that provide metadata for the
                               items. The default is FGDB.
        ---------------------  --------------------------------------------------------------------------
        capabilities           optional string. specifies the operations that can be performed on the
                               feature layer service. The default is Query.
        =====================  ==========================================================================


        :return:
           A feature collection or feature layer that can be used for analysis,
           visualization, or published to the GIS as an item.
        """
        if item_id and self._gis.version <= [7,1]:
            item_id = None
            import warnings
            warnings.warn("`item_id` is not allowed at this version of Portal, please use Enterprise 10.8.1+")
        from arcgis.features import FeatureCollection, SpatialDataFrame, FeatureSet

        from arcgis._impl.common._utils import zipws

        import shutil
        from uuid import uuid4
        import pandas as pd
        try:
            import arcpy
            has_arcpy = True
        except ImportError:
            has_arcpy = False
        except RuntimeError:
            has_arcpy = False
        try:
            import shapefile
            has_pyshp = True
        except ImportError:
            has_pyshp = False
        if isinstance(df, FeatureSet):
            df = df.sdf
        if has_arcpy == False and \
           has_pyshp == False and \
           (isinstance(df, SpatialDataFrame) or _is_geoenabled(df)):
            raise Exception("Spatially enabled DataFrame's must have either pyshp or" + \
                            " arcpy available to use import_data")
        elif (isinstance(df, SpatialDataFrame) or _is_geoenabled(df)):
            import random
            import string
            temp_dir = os.path.join(tempfile.gettempdir(), "a" + uuid4().hex[:7])
            title = kwargs.pop("title", uuid4().hex)
            tags = kwargs.pop('tags', 'FGDB')
            target_sr = kwargs.pop('target_sr', 102100)
            capabilities = kwargs.pop('capabilities', "Query")
            os.makedirs(temp_dir)
            temp_zip = os.path.join(temp_dir, "%s.zip" % ("a" + uuid4().hex[:5]))
            if has_arcpy:
                name = "%s%s.gdb" % (random.choice(string.ascii_lowercase),
                                     uuid4().hex[:5])
                from arcgis.features.geo._tools._utils import run_and_hide
                result = run_and_hide(fn=arcpy.CreateFileGDB_management,
                                      **{
                                          "out_folder_path" : temp_dir,
                                          "out_name" : name
                                      })
                fgdb = result[0]
                if isinstance(df, SpatialDataFrame) :
                    ds = df.to_featureclass(out_location=fgdb,
                                            out_name=os.path.basename(temp_dir))
                else:
                    ds =\
                        df.spatial.to_featureclass(location=os.path.join(fgdb,
                                                                         os.path.basename(temp_dir)))

                zip_fgdb = zipws(path=fgdb, outfile=temp_zip, keep=True)
                item = self.add(
                    item_properties={
                        "title" : title,
                        "type" : "File Geodatabase",
                        "tags" : tags},
                    data=zip_fgdb,
                    folder=folder)
                shutil.rmtree(temp_dir,
                              ignore_errors=True)
                publish_parameters =  {"hasStaticData":True, "name": os.path.splitext(item['name'])[0],
                                       "maxRecordCount":2000, "layerInfo":{"capabilities":capabilities}}
                if target_sr is not None:
                    publish_parameters['targetSR'] = { 'wkid' : target_sr }
                return item.publish(publish_parameters=publish_parameters, item_id=item_id)
            elif has_pyshp:
                import random
                import string
                name = "%s%s.shp" % (random.choice(string.ascii_lowercase),
                                     uuid4().hex[:5])
                if isinstance(df, SpatialDataFrame) :
                    ds = df.to_featureclass(out_location=temp_dir,
                                            out_name=name)
                else:
                    ds = df.spatial.to_featureclass(
                        location=os.path.join(temp_dir,
                                              name)
                    )
                zip_shp = zipws(path=temp_dir, outfile=temp_zip, keep=False)
                item = self.add(
                    item_properties={
                        "title":title,
                        "tags":tags},
                    data=zip_shp,
                    folder=folder)
                shutil.rmtree(temp_dir,
                              ignore_errors=True)
                publish_parameters =  {"hasStaticData":True, "name": os.path.splitext(item['name'])[0],
                                       "maxRecordCount":2000, "layerInfo":{"capabilities":capabilities}}
                if target_sr is not None:
                    publish_parameters['targetSR'] = { 'wkid' : target_sr }
                return item.publish(publish_parameters=publish_parameters, item_id=item_id)
            return
        elif isinstance(df, pd.DataFrame) and \
             'location_type' not in kwargs:
            # CSV WORKFLOW
            path = "content/features/analyze"

            postdata = {
                "f": "pjson",
                "text" : df.to_csv(),
                "filetype" : "csv",

                "analyzeParameters" : {
                    "enableGlobalGeocoding": "true",
                    "sourceLocale":"en-us",
                    #"locationType":"address",
                    "sourceCountry":"",
                    "sourceCountryHint":"",
                    "geocodeServiceUrl":self._gis.properties.helperServices.geocode[0]['url']
                }
            }

            if address_fields is not None:
                postdata['analyzeParameters']['locationType'] = 'address'

            res = self._portal.con.post(path, postdata)
            #import json
            #json.dumps(res)
            if address_fields is not None:
                res['publishParameters'].update({"addressFields":address_fields})

            path = "content/features/generate"
            postdata = {
                "f": "pjson",
                "text" : df.to_csv(),
                "filetype" : "csv",
                "publishParameters" : json.dumps(res['publishParameters'])
            }
            if item_id:
                postdata['itemIdToCreate'] = item_id
            res = self._portal.con.post(path, postdata)#, use_ordered_dict=True) - OrderedDict >36< PropertyMap

            fc = FeatureCollection(res['featureCollection']['layers'][0])
            return fc
        elif isinstance(df, pd.DataFrame) and \
             'location_type' in kwargs:
            path = "content/features/analyze"

            postdata = {
                "f": "pjson",
                "text" : df.to_csv(),
                "filetype" : "csv",

                "analyzeParameters" : {
                    "enableGlobalGeocoding": "true",
                    "sourceLocale":kwargs.pop("source_locale", "us-en"),

                    "sourceCountry": kwargs.pop("source_country", ""),
                    "sourceCountryHint": kwargs.pop("country_hint", ""),
                    "geocodeServiceUrl": kwargs.pop("geocode_url",
                                                    self._gis.properties.helperServices.geocode[0]['url']),
                    #"locationType": kwargs.pop('location_type', None),
                    #"latitudeFieldName" : kwargs.pop("latitude_field", None),
                    #"longitudeFieldName" : kwargs.pop("longitude_field", None),
                    #"coordinateFieldName" : kwargs.pop("coordinate_field_name", None),

                    #"coordinateFieldType" : kwargs.pop("coordinate_field_type", None)

                }
            }
            update_dict = {}
            update_dict["locationType"] = kwargs.pop('location_type', "")
            update_dict["latitudeFieldName"] = kwargs.pop("latitude_field", "")
            update_dict["longitudeFieldName"] = kwargs.pop("longitude_field", "")
            update_dict["coordinateFieldName"] = kwargs.pop("coordinate_field_name", "")
            update_dict["coordinateFieldType"] = kwargs.pop("coordinate_field_type", "")
            rk = []
            for k,v in update_dict.items():
                if v == "":
                    rk.append(k)
            for k in rk:
                del update_dict[k]

            res = self._portal.con.post(path, postdata)
            res['publishParameters'].update(update_dict)
            path = "content/features/generate"
            postdata = {
                "f": "pjson",
                "text" : df.to_csv(),
                "filetype" : "csv",
                "publishParameters" : json.dumps(res['publishParameters'])
            }
            if item_id:
                postdata['itemIdToCreate'] = item_id
            res = self._portal.con.post(path, postdata)#, use_ordered_dict=True) - OrderedDict >36< PropertyMap

            fc = FeatureCollection(res['featureCollection']['layers'][0])
            return fc
            #return
        return None

    def is_service_name_available(self, service_name, service_type):
        """ For a desired service name, determines if that service name is
            available for use or not.

            ================  ======================================================================
            **Argument**      **Description**
            ----------------  ----------------------------------------------------------------------
            service_name      Required string. A desired service name.
            ----------------  ----------------------------------------------------------------------
            service_type      Required string. The type of service to be created.  Currently the options are imageService or featureService.
            ================  ======================================================================

            :return:
                 True if the specified service_name is available for the
               specified service_type, False if the service_name is unavailable.

        """
        path = "portals/self/isServiceNameAvailable"

        postdata = {
            "f": "pjson",
            "name" : service_name,
            "type" : service_type
        }

        res = self._portal.con.post(path, postdata)
        return res['available']

    def clone_items(self, items, folder=None, item_extent=None, use_org_basemap=False, copy_data=True, copy_global_ids=False, search_existing_items=True, item_mapping=None, group_mapping=None, owner=None):
        """ Clone content to the GIS by creating new items.

        .. note::
        Cloning an item will create a copy of the item and for certain
        item types a copy of the item dependencies in the GIS.

        For example a web application created using Web AppBuilder
        or a Configurable App Template which is built from a web map
        that references one or more hosted feature layers. This function
        will clone all of these items to the GIS and swizzle the paths
        in the web map and web application to point to the new layers.

        This creates an exact copy of the application, map, and layers
        in the GIS.

        =====================     ====================================================================
        **Argument**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required list. Collection of Items to clone.
        ---------------------     --------------------------------------------------------------------
        folder                    Optional string. Name of the folder where placing item.
        ---------------------     --------------------------------------------------------------------
        item_extent               Optional Envelope. Extent set for any cloned items. Default is None,
                                  extent will remain unchanged. Spatial reference of the envelope will be
                                  used for any cloned feature layers.
        ---------------------     --------------------------------------------------------------------
        use_org_basemap           Optional boolean. Indicating whether the basemap in any cloned web maps
                                  should be updated to the organizations default basemap. Default is False,
                                  basemap will not change.
        ---------------------     --------------------------------------------------------------------
        copy_data                 Optional boolean. Indicating whether the data should be copied with any
                                  feature layer or feature collections. Default is True, data will be copied.
        ---------------------     --------------------------------------------------------------------
        copy_global_ids           Optional boolean. Assumes previous parameter is set to True. If True,
                                  features copied will preserve their global IDs. Default is False
        ---------------------     --------------------------------------------------------------------
        search_existing_items     Optional boolean. Indicating whether items that have already been cloned
                                  should be searched for in the GIS and reused rather than cloned again.
        ---------------------     --------------------------------------------------------------------
        item_mapping              Optional dictionary. Can be used to associate an item id in the source
                                  GIS (key) to an item id in the target GIS (value). The target item will
                                  be used rather than cloning the source item.
        ---------------------     --------------------------------------------------------------------
        group_mapping             Optional dictionary. Can be used to associate a group id in the source
                                  GIS (key) to a group id in the target GIS (value). The target group will
                                  be used rather than cloning the source group.
        ---------------------     --------------------------------------------------------------------
        owner                     Optional string. Defaults to the logged in user.
        =====================     ====================================================================

        :return:
           A list of items created during the clone.

        """

        import arcgis._impl.common._clone as clone
        wgs84_extent = None
        service_extent = item_extent
        if service_extent:
            wgs84_extent = clone._wgs84_envelope(service_extent)
        owner_name = owner
        if owner_name is None:
            owner_name = self._gis.users.me.username
        if isinstance(owner, User):
            owner_name = owner.username
        deep_cloner = clone._DeepCloner(self._gis, items, folder, wgs84_extent, service_extent, use_org_basemap, copy_data, copy_global_ids, search_existing_items, item_mapping, group_mapping, owner_name)
        return deep_cloner.clone()

    def bulk_update(self, itemids, properties):
        """
        Updates a collection of items' properties.

        Example:

        >>> itemsids = gis.content.search("owner: TestUser12399")
        >>> properties = {'categories' : ["clothes","formal_wear/socks"]}
        >>> gis.content._bulk_update(itemids, properties)
        [{'results' : [{'itemid' : 'id', 'success' : "True/False" }]}]

        .. :Note: bulk_update only works with content categories at this time.

        ================  ======================================================================
        **Argument**      **Description**
        ----------------  ----------------------------------------------------------------------
        itemids           Required list of string or Item. The collection of Items to update.
        ----------------  ----------------------------------------------------------------------
        properties        Required dictionary. The Item's properties to update.
        ================  ======================================================================

        :returns: list of results

        """
        path = "content/updateItems"
        params = {'f' : 'json',
                  'items' : []}
        updates = []
        results = []
        for item in itemids:
            if isinstance(item, Item):
                updates.append({
                    item.itemid : properties
                })
            elif isinstance(item, str):
                updates.append({
                    item : properties
                })
            else:
                raise ValueError("Invalid Item or ItemID, must be string or Item")
        def _chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i+n]
        for i in _chunks(l=updates, n=100):
            params['items'] = i

            res = self._gis._con.post(path=path, postdata=params)
            results.append(res)
            del i
        return results
    #----------------------------------------------------------------------
    def replace_service(self,
                        replace_item,
                        new_item,
                        replaced_service_name=None,
                        replace_metadata=False):
        """
        The replace_service operation allows you to replace your production vector tile layers with staging ones. This
        operation allows you to perform quality control on a staging tile layer and to then replace the production tile
        layer with the staging with minimal downtime. This operation has the option to keep a backup of the production
        tile layer.

        *Note*: This functionality is only available for hosted vector tile layers, hosted tile layers and hosted scene
        layers based on packages.

        *Note*: If you are looking to clone services, use the `clone_items()` method instead.

        Workflow for replace_service:

        1. Publish the staging service to the same system as the production service. Both services are active at
        the same time. Share the staging service with a smaller set of users and QA the staging service.

        2. The item properties (ex: thumbnail, iteminfo, metadata) of the production item will be preserved.
        If you need to update them use the `Item.update()` method.

        3. Call the replace_service operation. The service running on the hosting server gets replaced
        (for example, its cache).

        *Note:
        It is the responsibility of the user to ensure both services are functionally equivalent for clients
        consuming them. For example, when replacing a hosted feature service, ensure the new service is constructed
        with the anticipated layers and fields for its client application.

        If you want to retain the replaced production service, for example, to keep an archive of the evolution of the
        service you can do so by omitting a value for "Replaced Service Name" . If replaced service name is not provided,
        the production service being replaced will be archived with a time stamp when replace service was executed.
        You can provide any name for the replaced service as long as it is not pre-existing on your portal content.

        ======================  ======================================================================
        **Argument**            **Description**
        ----------------------  ----------------------------------------------------------------------
        replace_item            Required Item or Item's Id as string. The service to be replaced
        ----------------------  ----------------------------------------------------------------------
        new_item                Required Item or Item's Id as string. The replacement service.
        ----------------------  ----------------------------------------------------------------------
        replaced_service_name   Optional string. The name of the replacement service.
        ----------------------  ----------------------------------------------------------------------
        replace_metadata        Optional Boolean. When set to `True`, the item info {"thumbnail", "tag",
                                "description", "summary"} of the current service is updated to that of
                                the replacement service. The Credits, Terms of use, and Created from
                                details will not be replaced. This option is set to `False` by default.

        ======================  ======================================================================

        :returns: boolean
        """
        user = self._gis.users.me
        if 'id' in user:
            user = user.username
        else:
            user = user.username
        url = "%s/content/users/%s/replaceService" % (self._portal.resturl, user)

        if isinstance(replace_item, Item):
            replace_item = replace_item.itemid

        if isinstance(new_item, Item):
            new_item = new_item.itemid

        create_new_item = False
        if replaced_service_name:
            create_new_item = True

        params = {
            'toReplaceItemId': replace_item,
            'replacementItemId': new_item,
            'replaceMetadata' : replace_metadata,
            'createNewItem': create_new_item,
            'f': 'json'
        }
        if replaced_service_name is not None:
            params['replacedServiceName'] = replaced_service_name
        res = self._gis._con.post(path=url, postdata=params)
        if 'success' in res:
            return res['success']
        return False
    #----------------------------------------------------------------------
    def share_items(self, items, everyone=False, org=False,
                    groups=None, allow_members_to_edit=False):
        """
        Shares a batch of items with everyone, members of the organization, or specified list of groups.
        Users can only share items with groups to which they belong.

        =====================     ====================================================================
        **Argument**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required List. A list of Item or item ids to modify sharing on.
        ---------------------     --------------------------------------------------------------------
        everyone                  Optional boolean. Default is False, don't share with everyone.
        ---------------------     --------------------------------------------------------------------
        org                       Optional boolean. Default is False, don't share with the
                                  organization.
        ---------------------     --------------------------------------------------------------------
        groups                    Optional list of group names as strings, or a list of
                                  arcgis.gis.Group objects, or a comma-separated list of group IDs.
        ---------------------     --------------------------------------------------------------------
        allow_members_to_edit     Optional boolean. Default is False, to allow item to be shared with
                                  groups that allow shared update
        =====================     ====================================================================

        :returns: dict

        """
        url = "{base}content/users/{username}/shareItems".format(
            base=self._portal.resturl,
            username=self._gis.users.me.username
        )
        params = {
            'f' : 'json'
        }
        if groups is None:
            groups = []
        elif isinstance(groups, (tuple, list)):
            grps = []
            for grp in groups:
                if isinstance(grp, str):
                    grps.append(grp)
                elif isinstance(grp, Group):
                    grps.append(grp.groupid)
            groups = grps
        if isinstance(items, Item):
            sitems = [items.itemid]
            items = [items]
        elif isinstance(items, str):
            sitems = [items]
            items = [Item(gis=self._gis, itemid=items)]
        else:
            sitems = []
            for i in items:
                if isinstance(i, Item):
                    sitems.append(i.itemid)
                else:
                    sitems.append(i)
            if not isinstance(sitems[0], Item):
                items = [Item(gis=self._gis, itemid=i) for i in sitems]
        params['items'] = ",".join(sitems)
        params['everyone'] = everyone
        params['org'] = org
        params['confirmItemControl'] = allow_members_to_edit
        params['groups'] = ",".join(groups)
        res = self._gis._con.post(url, params)
        for i in items:
            i._hydrated = False
        return res
    #----------------------------------------------------------------------
    def unshare_items(self, items, groups=None, everyone=None, org=None):
        """
        Unshares a batch of items with the specified list of groups, everyone, or organization.
        Each item's current sharing will be overwritten with this method.

        =====================     ====================================================================
        **Argument**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required List. A list of Item or item ids to modify sharing on.
        ---------------------     --------------------------------------------------------------------
        everyone                  Required Boolean. If provided, the everyone sharing property will be
                                  updated.  True means it will share the items with anyone. False means
                                  the item will not be shared with all users.
        ---------------------     --------------------------------------------------------------------
        org                       Required Boolean. A true value means that the items will be shared
                                  with all members of the organization.  A false value means that the
                                  item will not be shared with all organization users.
        ---------------------     --------------------------------------------------------------------
        groups                    Required list of group names as strings, or a list of
                                  arcgis.gis.Group objects, or a list of group IDs.
        =====================     ====================================================================

        :returns: dict

        """
        res = True
        if groups is None and \
           everyone is None and\
           org is None:
            return True
        if groups:
            url = "{base}content/users/{username}/unshareItems".format(
                base=self._portal.resturl,
                username=self._gis.users.me.username
            )
            params = {
                'f' : 'json'
            }
            if isinstance(groups, (list, tuple)) == False:
                groups = [groups]
            if isinstance(items, (list,tuple)) == False:
                items = [items]
            if isinstance(groups, (tuple, list)):
                grps = []
                for grp in groups:
                    if isinstance(grp, str):
                        grps.append(grp)
                    elif isinstance(grp, Group):
                        grps.append(grp.groupid)
                groups = grps
            if isinstance(items, (tuple, list)):
                sitems = []
                for i in items:
                    if isinstance(i, str):
                        sitems.append(i)
                    elif isinstance(i, Item):
                        sitems.append(i.itemid)
                    #items = sitems
            params['groups'] = ",".join(groups)
            params['items'] = ",".join(sitems)
            res = self._gis._con.post(url, params)
        if everyone is not None and \
            org is not None:
            for item in items:
                if isinstance(item, Item):
                    item.share(everyone=everyone, org=org)
                elif isinstance(item, str):
                    Item(gis=self._gis, itemid=item).share(everyone=everyone, org=org)
        elif everyone is not None and \
            org is None:
            for item in items:
                if isinstance(item, Item):
                    org = item.shared_with['org']
                    item.share(everyone=everyone, org=org)
                if isinstance(item, str):
                    usitem = Item(gis=self._gis, itemid=item)
                    org = usitem.shared_with['org']
                    usitem.share(everyone=everyone, org=org)
        elif everyone is None and \
            org is not None:
            for item in items:
                if isinstance(item, Item):
                    everyone = item.shared_with['everyone']
                    item.share(everyone=everyone, org=org)
                if isinstance(item, str):
                    usitem = Item(gis=self._gis, itemid=item)
                    everyone = usitem.shared_with['everyone']
                    usitem.share(everyone=everyone, org=org)
        for item in items:
            if isinstance(item, Item):
                item._hydrated = False
            if isinstance(item, str):
                Item(gis=self._gis, itemid=item)._hydrated = False
        return res

########################################################################
class CategorySchemaManager(object):
    """
    Helper class for managing category schemas. This class is not created
    by users directly. An instance of this class, called `categories`, is
    available as a property on `gis.content` or on `gis.groups`.
    """
    _gis = None
    _url = None

    #----------------------------------------------------------------------
    def __init__(self, base_url, gis=None):
        """Constructor"""
        self._url = base_url
        if gis is None:
            import arcgis
            gis = arcgis.env.active_gis
        self._gis = gis
    #----------------------------------------------------------------------
    def __str__(self):
        return "<CategorySchemaManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """Returns the properties of the schema."""
        from arcgis._impl.common._mixins import PropertyMap
        return PropertyMap(self.schema)
    #----------------------------------------------------------------------
    @property
    def schema(self):
        """
        This property allows group owners/managers to manage the content
        categories for a group. These content categories are a hierarchical
        set of classes to help organize and browse group content.

        Each group can have a maximum of 5 category trees with each
        category schema can have up to 4 hierarchical levels. The maximum
        number of categories a group can have in total is 200 with each
        category of less than 100 characters title and 300 characters
        description.

        When getting this property, returns the content category schema
        set on a group.

        When setting this property, will update the group category schema
        based on the `dict` this property is set to. See below.

        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        categories          Required Dict. A category schema object consists of an
                            array of dict objects representing top level categories.
                            Each object has title, description and categories
                            properties where categories consists of an array of
                            objects with each having the same properties and
                            represents the descendant categories or subcategories and
                            so on.
        ==================  =========================================================
        """
        params = {'f' : 'json'}
        url = "{base}/categorySchema".format(base=self._url)
        res = self._gis._con.get(url, params)
        if 'categorySchema' in res:
            return res['categorySchema']
        return res
    #----------------------------------------------------------------------
    @schema.setter
    def schema(self, categories):
        """See main `schema` property docstring"""
        if categories is None:
            return self.delete()
        elif len(categories) == 0:
            return self.delete()
        params = {
            'f' : 'json',
            "categorySchema" : {"categorySchema": categories}
        }
        url = "{base}/assignCategorySchema".format(base=self._url)
        res = self._gis._con.post(url, params)
        if 'success' in res:
            return res['success']
        return

    #----------------------------------------------------------------------
    def delete(self):
        """
        This function allows group owner or managers to remove the
        category schema set on a group.

        :returns: Boolean
        """
        params = {'f' : 'json'}
        url = "{base}/deleteCategorySchema".format(base=self._url)
        try:
            if self.schema == []:
                return True
            res = self._gis._con.post(url, params)
            if 'success' in res:
                return res['success']
            return False
        except:
            return False
    #----------------------------------------------------------------------
    def assign_to_items(self, items):
        """
        This function adds group content categories to the portal items
        specified in the `items` argument (see below). For assigning categories
        to items in a group, you must be the group owner/manager. For assigning
        organization content categories on items, you must be the item owner
        or an administrator who has the `portal:admin:updateItems` privilege.
        A maximum of 100 items can be bulk updated per request.

        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        items               Required List. A JSON array of item objects. Each is
                            specified with the item ID that consists of a categories
                            object. categories is specified with an array that lists
                            all content categories to update on the item, each with
                            full hierarchical path prefixed with /.

                            Each item can be categorized to a maximum of 20
                            categories.

                            :Example:

                                [{
                                        "2678d3002eea4e4a825e3bdf10016e61": {
                                                "categories": ["/Categories/Geology", "/Categories/Elevation"]
                                        }
                                }, {
                                        "c3ad4ed8bcf04d619537cfe252a1760d": {
                                                "categories": ["/Categories/Geology", "/Categories/Land cover/Forest/Deciduous Forest"]
                                        }
                                }, {
                                        "9ced00fdce3e4b20bb4b05155acbe817": {
                                                "categories": []
                                        }
                                }]

        ==================  =========================================================


        :returns: A `dict` of `item_id` : `status`, with `status` being
        whether the content categories were successfully added

        """
        params = {'f' : 'json',
                  'items' : json.dumps(items)}
        if self._url.lower().find("/portals/") == -1:
            # If this SchemaManager is attached to a GroupManager
            url = "{base}/updateCategories".format(base=self._url)
        else:
            # else this SchemaManager is attached to a ContentManager
            url = "{base}content/updateItems".format(base=self._gis._portal.resturl)
        response = self._gis._con.post(url, params)
        output = {}
        if 'results' in response:
            for res in response['results']:
                if 'success' in res and 'itemId' in res:
                    output[res['itemId']] = res['success']
        return output

class ResourceManager(object):
    """
    Helper class for managing resource files of an item. This class is not created by users directly.
    An instance of this class, called 'resources', is available as a property of the Item object.
    Users call methods on this 'resources' object to manage (add, remove, update, list, get) item resources.
    """
    _user_id = None
    def __init__(self, item, gis):
        self._gis = gis
        self._portal = gis._portal
        self._item = item

        owner = self._item.owner
        user = gis.users.get(owner)
        if (hasattr(user, 'id')) and (user.id != 'null'):
            self._user_id = user.username
            #self._user_id = user.id
        else:
            self._user_id = user.username

    def export(self, save_path=None, file_name=None):
        """Export's the data's resources as a zip file"""
        url = 'content/users/'+ self._user_id +\
                    '/items/' + self._item.itemid + "/resources/export"
        if save_path is None:
            save_path = tempfile.gettempdir()
        if file_name is None:
            import uuid
            file_name = f"{uuid.uuid4().hex[:6]}.zip"
        params = {'f' : 'zip'}
        #from arcgis.gis._impl._con import Connection
        con = self._portal.con
        #isinstance(con, Connection)
        resources = con.get(url, params=params,out_folder=save_path, file_name=file_name, try_json=False)
        return resources

    def add(self, file=None, folder_name=None, file_name=None, text=None, archive=False, access=None):
        """The add resources operation adds new file resources to an existing item. For example, an image that is
        used as custom logo for Report Template. All the files are added to 'resources' folder of the item. File
        resources use storage space from your quota and are scanned for viruses. The item size is updated to
        include the size of added resource files. Each file added should be no more than 25 Mb.

        Supported item types that allow adding file resources are: Vector Tile Service, Vector Tile Package,
        Style, Code Attachment, Report Template, Web Mapping Application, Feature Service, Web Map,
        Statistical Data Collection, Scene Service, and Web Scene.

        Supported file formats are: JSON, XML, TXT, PNG, JPEG, GIF, BMP, PDF, MP3, MP4, and ZIP.
        This operation is only available to the item owner and the organization administrator.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Optional string. The path to the file that needs to be added.
        ----------------  ---------------------------------------------------------------
        folder_name       Optional string. Provide a folder name if the file has to be
                          added to a folder under resources.
        ----------------  ---------------------------------------------------------------
        file_name         Optional string. The file name used to rename an existing file
                          resource uploaded, or to be used together with text as file name for it.
        ----------------  ---------------------------------------------------------------
        text              Optional string. Text input to be added as a file resource,
                          used together with file_name. If this resource is used, then
                          file_name becomes required.
        ----------------  ---------------------------------------------------------------
        archive           Optional boolean. Default is False.  If True, file resources
                          added are extracted and files are uploaded to respective folders.
        ----------------  ---------------------------------------------------------------
        access            Optional String. Set file resource to be private regardless of
                          the item access level, or revert it by setting it to `inherit`
                          which makes the item resource have the same access as the item.

                          Supported values: `private` or `inherit`.
        ================  ===============================================================

        :return:
            Python dictionary like the following if it succeeded:
            {
                "success": True,
                "itemId": "<item id>",
                "owner": "<owner username>",
                "folder": "<folder id>"}

            else like the following if it failed:
            {"error": {
                        "code": 400,
                        "messageCode": "CONT_0093",
                        "message": "File type not allowed for addResources",
                        "details": []
                        }}
        """
        if not file and (not text or not file_name):
            raise ValueError("Please provide a valid file or text/file_name.")
        query_url = 'content/users/'+ self._user_id +\
            '/items/' + self._item.itemid + '/addResources'

        files = [] #create a list of named tuples to hold list of files
        if file and os.path.isfile(os.path.abspath(file)):
            files.append(('file',file, os.path.basename(file)))
        elif file and os.path.isfile(os.path.abspath(file)) == False:
            raise RuntimeError("File(" + file + ") not found.")

        params = {}
        params['f'] = 'json'

        if folder_name is not None:
            params['resourcesPrefix'] = folder_name
        if file_name is not None:
            params['fileName'] = file_name
        if text is not None:
            params['text'] = text
        params['archive'] = 'true' if archive else 'false'
        if access and str(access) in ['inherit', 'private']:
            params['access'] = access
        resp = self._portal.con.post(query_url, params,
                                     files=files, compress=False)
        return resp

    def update(self, file, folder_name=None, file_name=None, text=None):
        """The update resources operation allows you to update existing file resources of an item.
        File resources use storage space from your quota and are scanned for viruses. The item size
        is updated to include the size of updated resource files.

        Supported file formats are: JSON, XML, TXT, PNG, JPEG, GIF, BMP, PDF, and ZIP.
        This operation is only available to the item owner and the organization administrator.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Required string. The path to the file on disk to be used for
                          overwriting an existing file resource.
        ----------------  ---------------------------------------------------------------
        folder_name       Optional string. Provide a folder name if the file resource
                          being updated resides in a folder.
        ----------------  ---------------------------------------------------------------
        file_name         Optional string. The destination name for the file used to update
                          an existing resource, or to be used together with the text parameter
                          as file name for it.

                          For example, you can use fileName=banner.png to update an existing
                          resource banner.png with a file called billboard.png without
                          renaming the file locally.
        ----------------  ---------------------------------------------------------------
        text              Optional string. Text input to be added as a file resource,
                          used together with file_name.
        ================  ===============================================================

        :return:
            Python dictionary like the following if it succeeded:
            {
                "success": True,
                "itemId": "<item id>",
                "owner": "<owner username>",
                "folder": "<folder id>" }

            else like the following if it failed:
            {"error": {
                        "code": 404,
                        "message": "Resource does not exist or is inaccessible.",
                        "details": []
                        } }
        """

        query_url = 'content/users/' + self._user_id + \
            '/items/' + self._item.itemid + '/updateResources'

        files = []  # create a list of named tuples to hold list of files
        if not os.path.isfile(os.path.abspath(file)):
            raise RuntimeError("File(" + file + ") not found.")
        files.append(('file', file, os.path.basename(file)))

        params = {}
        params['f'] = 'json'

        if folder_name is not None:
            params['resourcesPrefix'] = folder_name
        if file_name is not None:
            params['fileName'] = file_name
        if text is not None:
            params['text'] = text

        resp = self._portal.con.post(query_url, params, files=files)
        return resp

    def list(self):
        """
        Provides a lists all file resources of an existing item. This resource is only available to
        the item owner and the organization administrator.

        :return:
            A Python list of dictionaries of the form:
            [
                {
                  "resource": "<resource1>"
                },
                {
                  "resource": "<resource2>"
                },
                {
                  "resource": "<resource3>"
                }
            ]
        """
        query_url = 'content/items/' + self._item.itemid + '/resources'
        params = {'f':'json',
                  'num': 1000}
        resp = self._portal.con.get(query_url, params)
        resp_resources = resp.get('resources')
        count = int(resp.get('num'))
        next_start = int(resp.get('nextStart', -999)) # added for back support for portal (10.4.1)

        # loop through pages
        while next_start > 0:
            params2 = {'f':'json',
                       'num':1000,
                       'start':next_start + 1}

            resp2 = self._portal.con.get(query_url, params2)
            resp_resources.extend(resp2.get('resources'))
            count += int(resp2.get('num'))
            next_start = int(resp2.get('nextStart', -999))# added for back support for portal (10.4.1)
            if next_start == -999:
                break

        return resp_resources

    def get(self, file, try_json = True, out_folder = None, out_file_name = None):
        """
        Gets a specific file resource of an existing item.  This operation is only
        available to the item owner and the organization administrator.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Required string. The path to the file to be downloaded.
                          For files in the root, just specify the file name. For files in
                          folders (prefixes), specify using the format
                          <foldername>/<foldername>./../<filename>
        ----------------  ---------------------------------------------------------------
        try_json          Optional boolean. If True, will attempt to convert JSON files to
                          Python dictionary objects. Default is True.
        ----------------  ---------------------------------------------------------------
        out_folder        Optional string. Specify the folder into which the file has to
                          be saved. Default is user's temporary directory.
        ----------------  ---------------------------------------------------------------
        out_file_name     Optional string. Specify the name to use when downloading the
                          file. Default is the resource file's name.
        ================  ===============================================================


        :return:
           Path to the downloaded file if getting a binary file (like a jpeg or png file) or if
           try_jon = False when getting a JSON file.

           If file is a JSON, returns as a Python dictionary.
        """

        safe_file_format = file.replace(r'\\','/')
        safe_file_format = safe_file_format.replace('//', '/')

        query_url = 'content/items/' + self._item.itemid + '/resources/' + safe_file_format

        return self._portal.con.get(query_url, try_json = try_json, out_folder=out_folder,
                                    file_name = out_file_name)

    def remove(self, file = None):
        """
        Removes a single resource file or all resources. The item size is updated once
        resource files are deleted. This operation is only available to the item owner
        and the organization administrator.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Optional string. The path to the file to be removed.
                          For files in the root, just specify the file name. For files in
                          folders (prefixes), specify using the format
                          <foldername>/<foldername>./../<filename>

                          If not specified, all resource files will be removed.
        ================  ===============================================================


        :return:
            If succeeded a boolean of True will be returned,

            else a dictionary with error info
            {"error": {"code": 404,
                        "message": "Resource does not exist or is inaccessible.",
                        "details": []
                      }
            }
        """
        safe_file_format = ""
        delete_all = 'false'
        if file:
            safe_file_format = file.replace(r'\\','/')
            safe_file_format = safe_file_format.replace('//', '/')
        else:
            delete_all = 'true'

        query_url = 'content/users/'+ self._user_id +\
            '/items/' + self._item.itemid + '/removeResources'
        params = {'f':'json',
                  'resource': safe_file_format if safe_file_format else "",
                  'deleteAll':delete_all}
        res = self._portal.con.post(query_url, postdata=params)
        if 'success' in res:
            return res['success']
        return res

class Group(dict):
    """
    Represents a group within the GIS (ArcGIS Online or Portal for ArcGIS).
    """
    def __init__(self, gis, groupid, groupdict=None):
        dict.__init__(self)
        self._gis = gis
        self._migrate = None
        self._portal = gis._portal
        self.groupid = groupid
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        # groupdict = self._portal.get_group(self.groupid)
        self._hydrated = False
        if groupdict:
            self.__dict__.update(groupdict)
            super(Group, self).update(groupdict)

    def _hydrate(self):
        groupdict = self._portal.get_group(self.groupid)
        self._hydrated = True
        super(Group, self).update(groupdict)
        self.__dict__.update(groupdict)

    def __getattr__(self, name): # support group attributes as group.access, group.owner, group.phone etc
        if not self._hydrated and not name.startswith('_'):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))



    def __getitem__(self, k): # support group attributes as dictionary keys on this object, eg. group['owner']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith('_'):
                self._hydrate()
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    def __repr__(self):
        return '<%s title:"%s" owner:%s>' % (type(self).__name__, self.title, self.owner)

    def get_thumbnail_link(self):
        """ URL to the thumbnail image """
        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            return self._gis.url + '/home/images/group-no-image.png'
        else:
            thumbnail_url_path = self._gis._public_rest_url + 'community/groups/' + self.groupid + '/info/' + thumbnail_file
            return thumbnail_url_path

    def search(self,
               query,
               return_count=False,
               max_items=100,
               bbox=None,
               categories=None,
               category_filter=None,
               start=1,
               sort_field="title",
               sort_order="ASC",
               as_dict=False):
        """
        The `search` operation allows users to find content within the specific group.

        ================    ===============================================================
        **Argument**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String.  The search query.
        ----------------    ---------------------------------------------------------------
        bbox                Optional String/List. This is the xmin,ymin,xmax,ymax bounding
                            box to limit the search in.  Items like documents do not have
                            bounding boxes and will not be included in the search.
        ----------------    ---------------------------------------------------------------
        categories          Optional String. A comma separated list of up to 8 org content
                            categories to search items. Exact full path of each category is
                            required, OR relationship between the categories specified.

                            Each request allows a maximum of 8 categories parameters with
                            AND relationship between the different categories parameters
                            called.
        ----------------    ---------------------------------------------------------------
        category_filters    Optional String. A comma separated list of up to 3 category
                            terms to search items that have matching categories. Up to 2
                            `category_filters` parameter are allowed per request. It can
                            not be used together with categories to search in a request.
        ----------------    ---------------------------------------------------------------
        start               Optional Int. The starting position to search from.  This is
                            only required if paging is needed.
        ----------------    ---------------------------------------------------------------
        sort_field          Optional String. Responses from the `search` operation can be
                            sorted on various fields. `avgrating` is the default.
        ----------------    ---------------------------------------------------------------
        sort_order          Optional String. The sequence into which a collection of
                            records are arranged after they have been sorted. The allowed
                            values are: asc for ascending and desc for descending.
        ----------------    ---------------------------------------------------------------
        as_dict             Required Boolean. If True, the response comes back as a dictionary.
        ================    ===============================================================


        :returns: List of Items
        """
        from ._impl._search import _search
        if return_count:
            return _search(gis=self._gis,
                       query=query, stype="group_content",
                       max_items=max_items,
                       bbox=bbox,
                       categories=categories,
                       category_filter=category_filter,
                       start=start,
                       sort_field=sort_field,
                       sort_order=sort_order,
                       group_id=self.id,
                       as_dict=True)['total']
        return _search(gis=self._gis,
                       query=query, stype="group_content",
                       max_items=max_items,
                       bbox=bbox,
                       categories=categories,
                       category_filter=category_filter,
                       start=start,
                       sort_field=sort_field,
                       sort_order=sort_order,
                       group_id=self.id,
                       as_dict=as_dict)

    #----------------------------------------------------------------------
    @property
    def categories(self):
        """
        The category manager for groups. See :class:`~arcgis.gis.CategorySchemaManager`.
        """
        base_url = "{base}community/groups/{groupid}".format(
            base=self._gis._portal.resturl,
            groupid=self.groupid)
        return CategorySchemaManager(base_url=base_url, gis=self._gis)

    @property
    def homepage(self):
        """Gets the URL to the HTML page for the group."""
        return "{}{}{}".format(self._gis.url,
                               "/home/group.html?id=",
                               self.groupid)

    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        else:
            b64 = base64.b64encode(self.get_thumbnail())
            thumbnail = "data:image/png;base64," + str(b64,"utf-8") + "' "

        title = 'Not Provided'
        snippet = 'Not Provided'
        description = 'Not Provided'
        owner = 'Not Provided'
        try:
            title = self.title
        except:
            title = 'Not Provided'

        try:
            description = self.description
        except:
            description = 'Not Provided'

        try:
            snippet = self.snippet
        except:
            snippet = 'Not Provided'

        try:
            owner = self.owner
        except:
            owner = 'Not available'

        url = self.homepage

        return """<div class="9item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='""" + str(url) + """' target='_blank'>
                        <img src='""" + str(thumbnail) + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right" style="float: none; width: auto; overflow: hidden;">
                        <a href='""" + str(url) + """' target='_blank'><b>""" + str(title) + """</b>
                        </a>
                        <br/>
                        <br/><b>Summary</b>: """ + str(snippet) + """
                        <br/><b>Description</b>: """ + str(description)  + """
                        <br/><b>Owner</b>: """ + str(owner)  + """
                        <br/><b>Created</b>: """ + str(datetime.fromtimestamp(self.created/1000).strftime("%B %d, %Y")) + """

                    </div>
                </div>
                """

    def content(self, max_items=1000):
        """
        Gets the list of items shared with this group.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        max_items              Required integer. The maximum number of items to be returned, defaults to 1000.
        ==================     ====================================================================


        :return:
           The list of items that are shared.

        """
        itemlist = []
        items = self._portal.search('group:' + self.groupid, max_results=max_items, outside_org=True)
        for item in items:
            itemlist.append(Item(self._gis, item['id'], item))
        return itemlist

    def delete(self):
        """
        Deletes this group.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        return self._portal.delete_group(self.groupid)

    def get_thumbnail(self):
        """
        Gets the bytes that make up the thumbnail for this group.


        :return:
            Bytes that represent the image.

        Example

        .. code-block:: python

            response = group.get_thumbnail()
            f = open(filename, 'wb')
            f.write(response)
        """
        return self._portal.get_group_thumbnail(self.groupid)

    @property
    def migration(self):
        """provides to to migrate content of a `Group` to a new Organaization or Portal"""
        isinstance(self._gis, GIS)

        if self._gis.version > [7,3] and \
           self._gis._portal.is_arcgisonline == False:
            self._migrate = GroupMigrationManager(group=self)
        return self._migrate

    def download_thumbnail(self, save_folder=None):
        """
        Downloads the group thumbnail for this group.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        save_folder            Optional string. The file path to where the group thumbnail will be downloaded.
        ==================     ====================================================================


        :return:
           The file path to which the group thumbnail is downloaded.
        """
        if self.thumbnail is None:
            self._hydrate()
        thumbnail_file = self.thumbnail
        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = 'community/groups/' + self.groupid + '/info/' + thumbnail_file
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50: #If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                self._portal.con.get(path=thumbnail_url_path, try_json=False,
                                     out_folder=save_folder,
                                     file_name=file_name)
                return file_path

        else:
            return None

    def add_users(self, usernames=None, admins=None):
        """ Adds users to this group.

        .. note::
            This method will only work if the user for the
            Portal object is either an administrator for the entire
            Portal or the owner of the group.

        ============  ======================================
        **Argument**  **Description**
        ------------  --------------------------------------
        usernames     Optional list of strings or single string.
                      The list of usernames or single username to be added.
        ------------  --------------------------------------
        admins        Optional List of String, or Single String.  This is a list of users to be an administrator of the group.
        ============  ======================================

        :return:
           A dictionary which contains the users that were not added to the group.
        """
        if usernames is None and admins is None:
            return {
                "notAdded": [ ]
            }

        users = None
        ladmins = None

        if isinstance(usernames, (list, tuple)) == False:
            usernames = [usernames]

        if admins and \
           isinstance(admins, (list, tuple)) == False:
            admins = [admins]

        if admins:
            ladmins = []
            for u in admins:
                if isinstance(u, str):
                    ladmins.append(u)
                elif isinstance(u, User):
                    ladmins.append(u.username)
        if usernames:
            users = []
            for u in usernames:
                if isinstance(u, str):
                    users.append(u)
                elif isinstance(u, User):
                    users.append(u.username)
        return self._portal.add_group_users(users, self.groupid, ladmins)

    def delete_group_thumbnail(self):
        """
        Deletes the group's thumbnail

        :returns: Boolean

        """
        return self._portal.delete_group_thumbnail(self.groupid)

    def remove_users(self, usernames):
        """
        Remove users from this group.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        usernames         Required string.  A comma-separated list of users to be removed.
        ================  ========================================================

        :return:
            A dictionary with a key notRemoved that is a list of users not removed.
        """
        users = []
        if isinstance(usernames, (list, tuple)) == False:
            usernames = [usernames]
        for u in usernames:
            if isinstance(u, str):
                users.append(u)
            elif isinstance(u, User):
                users.append(u.username)

        return self._portal.remove_group_users(users, self.groupid)

    def invite_users(self, usernames, role='group_member', expiration=10080):
        """
        Invites existing users to this group. The user executing this command must be the group owner.

        .. note::
            A user who is invited to this group will see a list of invitations
            in the "Groups" tab of Portal listing invitations.  The user
            can either accept or reject the invitation.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        usernames         Required list of strings. The users to invite.
        ----------------  --------------------------------------------------------
        role              Optional string. Either group_member (the default) or group_admin.
        ----------------  --------------------------------------------------------
        expiration        Optional integer. Specifies how long the invitation is
                          valid for in minutes.  Default is 10,080 minutes (7 days).
        ================  ========================================================

        :return:
           A boolean indicating success (True) or failure (False).
        """
        return self._portal.invite_group_users(usernames, self.groupid, role, expiration)

    #----------------------------------------------------------------------
    @deprecated(deprecated_in="v1.5.1", removed_in=None,
                current_version=None,
                details="Use `Group.invite` instead.")
    def invite_by_email(self, email, message, role='member', expiration='1 Day'):
        """
        ** Deprecated: This function is not supported **

        Invites a user by email to the existing group.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        email             Required string. The user to send join email to.
        ----------------  --------------------------------------------------------
        message           Required string. The message to send to the user.
        ----------------  --------------------------------------------------------
        role              Optional string. Either member (the default) or admin.
        ----------------  --------------------------------------------------------
        expiration        Optional string.  The is the time out of the invite.
                          The values are: 1 Day (default), 3 Days, 1 Week, or
                          2 Weeks.
        ================  ========================================================

        :returns: boolean
        """

        if self._gis.version >= [6,4]:
            return False

        time_lookup = {
            '1 Day'.upper() : 1440,
            '3 Days'.upper() : 4320,
            '1 Week'.upper() : 10080,
            '2 Weeks'.upper() : 20160
        }
        role_lookup = {
            'member' : 'group_member',
            'admin' : 'group_admin'
        }
        url = 'community/groups/' + self.groupid + '/inviteByEmail'
        params = {
            "f" : "json",
            "emails" : email,
            "message" : message,
            "role" : role_lookup[role.lower()],
            'expiration' : time_lookup[expiration.upper()]
        }
        return self._portal.con.post(url, params)


    def reassign_to(self, target_owner):
        """
        Reassigns this group to another owner.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        target_owner      Required string or User.  The username of the new group owner.
        ================  ========================================================

        :return:
            A boolean indicating success (True) or failure (False).
        """
        params = {'f' : 'json'}
        if isinstance(target_owner, User):
            params['targetUsername'] = target_owner.username
        else:
            params['targetUsername'] = target_owner
        res = self._gis._con.post('community/groups/' + self.groupid + '/reassign', params)
        if res:
            self._hydrated = False
            self._hydrate()
            return res.get('success')
        return False

    #----------------------------------------------------------------------
    def notify(self, users, subject, message, method="email", client_id=None):
        """
        Creates a group notification that sends a message to all users within
        the group.

        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        users               Required List. A list of users or user names.
        ------------------  ---------------------------------------------------------
        subject             Required String. The subject of the notification.
        ------------------  ---------------------------------------------------------
        message             Required String. The message body that will be sent to
                            the group's users.
        ------------------  ---------------------------------------------------------
        method              Optional String. This is the form for which users will be
                            contacted.  The allowed values are: email, push, and
                            builtin.

                            + email - sent a message via smtp.
                            + push - pushes a message out.
                            + builtin - creates a user notification.


        ------------------  ---------------------------------------------------------
        client_id           Optional String. The client id of the application for the
                            push operation.
        ==================  =========================================================

        :return: Boolean

        """
        from arcgis.gis import User
        cusers = []
        for user in users:
            if isinstance(user, User):
                cusers.append(user.username)
            else:
                cusers.append(user)
            del user
        url = "community/groups/{groupid}/createNotification".format(groupid=self.groupid)
        params = {
            "notificationChannelType": method,
            "subject": subject,
            "message" : message,
            "users" : ",".join(cusers),
            "clientId" : client_id,
            "f": "json"
        }
        return self._gis._con.post(url, params)

    def get_members(self):
        """
        Gets the members of this group.


        *Key:Value Dictionary Return Values*

            ================  ========================================================
            **Key**           **Value**
            ----------------  --------------------------------------------------------
            owner             The group's owner (string).
            ----------------  --------------------------------------------------------
            admins            The group's admins (list of strings). Typically this is the same as the owner.
            ----------------  --------------------------------------------------------
            users             The members of the group (list of strings).
            ================  ========================================================


        :return:
            A dictionary with keys: owner, admins, and users.


        .. code-block:: python

            # Usage Example: To print users in a group

            response = group.get_members()
            for user in response['users'] :
                print(user)

        """
        url = '%s/community/groups/%s/users' % (self._gis._portal.resturl,
                                                self.groupid)
        params = {'f': 'json'}
        return self._gis._con.post(url, params)

    def update(self, title=None, tags=None, description=None, snippet=None, access=None,
               is_invitation_only=None, sort_field=None, sort_order=None, is_view_only=None,
               thumbnail=None, max_file_size=None, users_update_items=False, clear_empty_fields=False,
               display_settings=None, is_open_data=False, leaving_disallowed=False):
        """
        Updates this group with only values supplied for particular arguments.


        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        title               Optional string. The new name of the group.
        ------------------  ---------------------------------------------------------
        tags                Optional string. A comma-delimited list of new tags, or
                            a list of tags as strings.
        ------------------  ---------------------------------------------------------
        description         Optional string. The new description for the group.
        ------------------  ---------------------------------------------------------
        snippet             Optional string. A new short snippet (<250 characters)
                            that summarizes the group.
        ------------------  ---------------------------------------------------------
        access              Optional string. Choices are private, public, or org.
        ------------------  ---------------------------------------------------------
        is_invitation_only  Optional boolean. Defines whether users can join by
                            request. True means an invitation is required.
        ------------------  ---------------------------------------------------------
        sort_field          Optional string. Specifies how shared items with the
                            group are sorted.
        ------------------  ---------------------------------------------------------
        sort_order          Optional string. Choices are asc or desc for ascending
                            or descending, respectively.
        ------------------  ---------------------------------------------------------
        is_view_only        Optional boolean. Defines whether the group is searchable.
                            True means the group is searchable.
        ------------------  ---------------------------------------------------------
        thumbnail           Optional string. URL or file location to a new group image.
        ------------------  ---------------------------------------------------------
        max_file_size       Optional integer.  This is the maximum file size allowed
                            be uploaded/shared to a group. Default value is: 1024000
        ------------------  ---------------------------------------------------------
        users_update_items  Optional boolean.  Members can update all items in this
                            group.  Updates to an item can include changes to the
                            item's description, tags, metadata, as well as content.
                            This option can't be disabled once the group has
                            been created. Default is False.
        ------------------  ---------------------------------------------------------
        clear_empty_fields  Optional Boolean. If True, the user can set values to
                            empty string, else, None values will be ignored.
        ------------------  ---------------------------------------------------------
        display_settings    Optional String. Defines the default display for the
                            group page to show a certain type of items. The allowed
                            values are: `apps, all, files, maps, layers, scenes, tools`.
                            The default value is `all`.
        ------------------  ---------------------------------------------------------
        is_open_data        Optional Boolean. Defines whether the group can be used
                            in the Open Data capabilities of ArcGIS Hub. The default
                            is False.
        ------------------  ---------------------------------------------------------
        leaving_disallowed  Optional boolean. Defines whether users are restricted
                            from choosing to leave the group. If True, only an
                            administrator can remove them from the group. The default
                            is False.
        ==================  =========================================================


        :return:
            A boolean indicating success (True) or failure (False).
        """
        display_settings_lu = {
            "apps" : {"itemTypes":"Application"},
            "all" : {"itemTypes":""},
            "files" : {"itemTypes":"CSV"},
            None : {"itemTypes":""},
            "maps" : {"itemTypes":"Web Map"},
            "layers" : {"itemTypes":"Layer"},
            "scenes" : {"itemTypes":"Web Scene"},
            "tools" : {"itemTypes":"Locator Package"}
        }
        if max_file_size is None:
            max_file_size = 1024000
        if users_update_items is None:
            users_update_items = False
        if tags is not None:
            if type(tags) is list:
                tags = ",".join(tags)
        if isinstance(display_settings, str) and display_settings.lower() in display_settings_lu:
            display_settings = display_settings_lu[display_settings.lower()]
        elif display_settings is None:
            display_settings = display_settings_lu[display_settings]
        else:
            raise ValueError("Display settings must be set to a valid value.")
        resp = self._portal.update_group(self.groupid, title, tags,
                                         description, snippet, access,
                                         is_invitation_only, sort_field,
                                         sort_order, is_view_only, thumbnail,
                                         max_file_size, users_update_items,
                                         clear_empty_fields=clear_empty_fields,
                                         display_settings=display_settings,
                                         is_open_data=is_open_data,
                                         leaving_disallowed=leaving_disallowed)
        if resp:
            self._hydrate()
        return resp

    def leave(self):
        """
        Removes the logged in user from this group.  It is required
        that the user be logged in.


        :return:
           A boolean indicating success (True) or failure (False).
        """
        return self._portal.leave_group(self.groupid)

    def join(self):
        """
        Users apply to join a group using the Join Group operation. This
        creates a new group application, which the group administrators
        accept or decline. This operation also creates a notification for
        the user indicating that they have applied to join this group.
        Available only to authenticated users.
        Users can only apply to join groups to which they have access. If
        the group is private, users will not be able to find it to ask to
        join it.
        Information pertaining to the applying user, such as their full
        name and username, can be sent as part of the group application.

        :return:
             A boolean indicating success (True) or failure (False).
        """
        url = "community/groups/%s/join" % (self.groupid)
        params = {"f" : "json"}
        res = self._portal.con.post(url, params)
        if 'success' in res:
            return res['success'] == True
        return res
    #----------------------------------------------------------------------
    @property
    def applications(self):
        """
        Gets the group applications for the given group as a list. Available to
        administrators of the group or administrators of an organization if
        the group is part of one.
        """
        apps = []
        try:
            path = "%scommunity/groups/%s/applications" % (self._portal.resturl, self.groupid)
            params = {"f" : "json"}
            res = self._portal.con.post(path, params)
            if 'applications' in res:
                for app in res['applications']:
                    url = "%s/%s" % (path, app['username'])
                    apps.append(GroupApplication(url=url, gis=self._gis))
        except:
            print()
        return apps
    #----------------------------------------------------------------------
    @property
    def protected(self):
        """
        Indicates if the group is protected from deletion. Set it to `True`
        to protect the group and `False` to unprotect it.
        """
        return self['protected']
    #----------------------------------------------------------------------
    @protected.setter
    def protected(self, value):
        """
        If set to True, the group will be prevented from being deleted.
        If false, a group can be deleted.
        """
        params = {'f' : 'json'}
        if value == True and self.protected == False:
            url = "%s/community/groups/%s/protect" % (self._portal.resturl, self.groupid)
            res = self._portal.con.post(url, params)
            self._hydrated = False
            self._hydrate()
        elif value == False and self.protected == True:
            url = "%s/community/groups/%s/unprotect" % (self._portal.resturl, self.groupid)
            res = self._portal.con.post(url, params)
            self._hydrated = False
            self._hydrate()




class GroupApplication(object):
    """
    Represents a single group application on the GIS (ArcGIS Online or
    Portal for ArcGIS).
    """
    _con = None
    _portal =  None
    _gis = None
    _url = None
    _properties = None
    def __init__(self, url, gis, **kwargs):
        initialize = kwargs.pop('initialize', False)
        self._url = url
        self._gis = gis
        self._portal = gis._portal
        self._con = self._portal.con
        if initialize:
            self._init()

    def _init(self):
        """Loads the properties."""
        try:
            res = self._con.get(self._url, {'f':'json'})
            self._properties = PropertyMap(res)
            self._json_dict = res
        except:
            self._properties = PropertyMap({})
            self._json_dict = {}

    @property
    def properties(self):
        """Gets the properties of the Group application."""
        if self._properties is None:
            self._init()
        return self._properties

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s for %s>' % (type(self).__name__, self.properties.username)

    def accept(self):
        """
        When a user applies to join a group, a group application is
        created. Group administrators choose to accept this application
        using the Accept Group Application operation. This operation adds
        the applying user to the group then deletes the application. This
        operation also creates a notification for the user indicating that
        the user's group application was accepted. Available only to group
        owners and admins.

        :return:
           A boolean indicating success (True) or failure (False).
        """
        url = "%s/accept" % self._url
        params = {"f" : "json"}
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success'] == True
        return res

    def decline(self):
        """
        When a user applies to join a group, a group application is
        created. Group administrators can decline this application using
        this method. This method
        deletes the application and creates a notification for the user
        indicating that the user's group application was declined. The
        applying user will not be added to the group. Available only to
        group owners and admins.

        :return:
           A boolean indicating success (True) or failure (False).
        """
        url = "%s/decline" % self._url
        params = {"f" : "json"}
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success'] == True
        return res

class User(dict):
    """
    Represents a registered user of the GIS (ArcGIS Online, or Portal for ArcGIS).

    =====================    =========================================================
    **Property**             **Details**
    ---------------------    ---------------------------------------------------------
    username                 The username of the user.
    ---------------------    ---------------------------------------------------------
    fullName                 The user's full name
    ---------------------    ---------------------------------------------------------
    availableCredits         The number of credits available to the user.
    ---------------------    ---------------------------------------------------------
    assignedCredits          The number of credits allocated to the user.
    ---------------------    ---------------------------------------------------------
    firstName                The user's first name.
    ---------------------    ---------------------------------------------------------
    lastName                 The user's last name.
    ---------------------    ---------------------------------------------------------
    preferredView            The user's preferred view for content, either web or GIS.
    ---------------------    ---------------------------------------------------------
    description              A description of the user.
    ---------------------    ---------------------------------------------------------
    email                    The user's e-mail address.
    ---------------------    ---------------------------------------------------------
    idpUsername              The original username if using enterprise logins.
    ---------------------    ---------------------------------------------------------
    favGroupId               The user's favorites group and is created automatically for each user.
    ---------------------    ---------------------------------------------------------
    lastLogin                The last login date of the user in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    mfaEnabled               Indicates if the user's account has multifactor authentication set up.
    ---------------------    ---------------------------------------------------------
    access                   Indicates the level of access of the user: private, org, or public. If private, the user descriptive information will not be available to others nor will the username be searchable.
    ---------------------    ---------------------------------------------------------
    storageUsage             | The amount of storage used for the entire organization.

                             **NOTE:** This value is an estimate for the organization, not the specific user.
                             For storage estimate of a user's items, see code example in the :attr:`items` method.
    ---------------------    ---------------------------------------------------------
    storageQuota             Applicable to public users as it sets the total amount of storage available for a subscription. The maximum quota is 2GB.
    ---------------------    ---------------------------------------------------------
    orgId                    The ID of the organization the user belongs to.
    ---------------------    ---------------------------------------------------------
    role                     | Defines the user's role in the organization.
                             Values:
                               * ``org_admin`` - administrator or custom role with administrative privileges
                               * ``org_publisher`` - publisher or custom role with publisher privileges
                               * ``org_user`` - user or custom role with user privileges)
    ---------------------    ---------------------------------------------------------
    privileges               A JSON array of strings with predefined permissions in each. For a complete listing, see Privileges.
    ---------------------    ---------------------------------------------------------
    roleId                   (Optional) The ID of the user's role if it is a custom one.
    ---------------------    ---------------------------------------------------------
    level                    The level of the user.
    ---------------------    ---------------------------------------------------------
    disabled                 Disables access to the organization by the user.
    ---------------------    ---------------------------------------------------------
    units                    User-defined units for measurement.
    ---------------------    ---------------------------------------------------------
    tags                     User-defined tags that describe the user.
    ---------------------    ---------------------------------------------------------
    culture                  The user locale information (language and country).
    ---------------------    ---------------------------------------------------------
    cultureFormat            The user preferred number and date format defined in CLDR (only applicable for English and Spanish, i.e. when culture is en or es).<br><br>See Languages for supported formats. It will inherit from organization cultureFormat if undefined.
    ---------------------    ---------------------------------------------------------
    region                   The user preferred region, used to set the featured maps on the home page, content in the gallery, and the default extent of new maps in the Viewer.
    ---------------------    ---------------------------------------------------------
    thumbnail                The file name of the thumbnail used for the user.
    ---------------------    ---------------------------------------------------------
    created                  The date the user was created. Shown in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    modified                 The date the user was last modified. Shown in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    groups                   A JSON array of groups the user belongs to. See Group for properties of a group.
    ---------------------    ---------------------------------------------------------
    provider                 The identity provider for the organization.<br>Values: arcgis (for built-in users) ,enterprise (for external users managed by an enterprise identity store), facebook (for public accounts in ArcGIS Online), google (for public accounts in ArcGIS Online)
    ---------------------    ---------------------------------------------------------
    id                       (optional) The unique identifier of the user used on AGOL/ArcGIS Enterprise 10.7+
    =====================    =========================================================



    """
    def __init__(self, gis, username, userdict=None):
        dict.__init__(self)
        self._gis = gis
        self._portal = gis._portal
        self._user_id = username
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        # userdict = self._portal.get_user(self.username)
        self._hydrated = False
        if userdict:
            if 'groups' in userdict and len(userdict['groups']) == 0: # groups aren't set unless hydrated
                del userdict['groups']
            if 'role' in userdict and \
               'roleId' not in userdict:
                userdict['roleId'] = userdict['role']
            elif 'roleId' in userdict and 'role' not in userdict:
                # try getting role name - only needed for custom roles
                try:
                    role_obj = self._gis.users.roles.get_role(userdict['roleId'])
                    userdict['role'] = role_obj.name
                except Exception as ex:
                    userdict['role'] = userdict['roleId']
            self.__dict__.update(userdict)
            super(User, self).update(userdict)
        if hasattr(self, 'id') and \
           self.id !='null':
            #self._user_id = self.id
            self._user_id = self.username
        else:
            self._user_id = self.username

    # Using http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991

    def _hydrate(self):
        userdict = self._portal.get_user(self._user_id)
        if not 'roleId' in userdict and \
           'role' in userdict:
            userdict['roleId'] = userdict['role']
        self._hydrated = True
        super(User, self).update(userdict)
        self.__dict__.update(userdict)

    def __getattr__(self, name): # support user attributes as user.access, user.email, user.role etc
        if not self._hydrated and not name.startswith('_'):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))


    def __getitem__(self, k): # support user attributes as dictionary keys on this object, eg. user['role']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith('_'):
                self._hydrate()
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s username:%s>' % (type(self).__name__, self.username)

    def user_types(self):
        """
        Notes: Available in 10.7+
        returns the user type and assigned applications
        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("`user_types` is not implemented at version %s" % \
                                 ".".join([str(i) for i in self._gis.version]))

        url = "%s/community/users/%s/userLicenseType" % (self._portal.resturl, self.username)
        params = {'f' : 'json'}
        return self._portal.con.post(url, params)
    #----------------------------------------------------------------------
    @property
    def tasks(self):
        """The resource manager for user's tasks. See :class:`~arcgis.gis.tasks.TaskManager`."""
        if str(self.role).lower() == 'org_admin' or \
           self._gis.properties['user']:
            url = f"{self._gis._portal.resturl}community/users/{self.username}/tasks"
            from .tasks import TaskManager
            return TaskManager(url=url,
                               user=self,
                               gis=self._gis)
        return None
    #----------------------------------------------------------------------
    def generate_direct_access_url(self, store_type:str) -> str:
        """
        Creates a direct access URL for uploading large files to datafile share, notebook workspaces or raster stores.

        **Available in ArcGIS Online Only**

        =====================  =========================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------
        store_type             Optional String. The type of upload URL to generate.
                               Types: `big_data_file`, 'notebook', or 'raster`.
        =====================  =========================================================

        :returns: str

        """
        if self._gis._portal.is_arcgisonline == False:
            return None
        _lu = {
            'big_data_file' : 'bigDataFileShare',
            'notebook' : 'notebookWorkspace',
            'raster' : 'rasterStore'
        }
        url = f"{self._gis._portal.resturl}content/users/{self.username}/generateDirectAccessUrl"
        params = {
            'f' : 'json',
            'expiration' : 1440,
            'storeType' : _lu[store_type.lower()]
        }
        return self._portal.con.get(url, params)
    #----------------------------------------------------------------------
    @property
    def provisions(self):
        """
        Returns a list of all provisioned licenses for the current user.

        Available in 10.7+

        :returns: List

        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("Provisions is not implemented at version %s" % \
                                 ".".join([str(i) for i in self._gis.version]))

        provs = []
        url = "%s/community/users/%s/provisionedListings" % (self._portal.resturl, self.username)
        params = {
            'f': 'json',
            'start' :1,
            'num' : 255
        }
        res = self._portal.con.post(url, params)
        provs = [Item(gis=self._gis, itemid=i["itemId"])for i in res["provisionedListings"]]
        while res['nextStart'] > -1:
            params = {
                'f': 'json',
                'start' : res['nextStart'],
                'num' : 255
            }
            res = self._portal.con.post(url, params)
            provs += [Item(gis=self._gis, itemid=i["itemId"])for i in res["provisionedListings"]]
        return provs
    #----------------------------------------------------------------------
    @property
    def bundles(self):
        """

        Provides the current user's assigned application bundles.

        Available in ArcGIS Online and Portal 10.7+

        :returns: List of Bundle objects
        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("`bundles` is not implemented at version %s" % \
                                 ".".join([str(i) for i in self._gis.version]))

        from arcgis.gis.admin._license import Bundle
        url = "%s/community/users/%s/appBundles" % (self._portal.resturl, self.username)
        params = {
            'f' : 'json',
            "start" : 1,
            "num" : 10
        }
        bundles = []
        res = self._portal.con.post(url, params)
        bundles = res["appBundles"]
        while res["nextStart"] > -1:
            params['start'] = res["nextStart"]
            res = self._portal.con.post(url, params)
            bundles += res["appBundles"]
        return [Bundle(url="{base}content/listings/{id}".format(base=self._gis._portal.resturl,
                                                                id=b["id"]),
                       properties=b,
                    gis=self._gis)
                for b in bundles]
    #----------------------------------------------------------------------
    def get_thumbnail_link(self):
        """ Retrieves the URL to the thumbnail image.

        :return:
           The thumbnail's URL.
        """
        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            return self._gis.url + '/home/js/arcgisonline/css/images/no-user-thumb.jpg'
        else:
            thumbnail_url_path = self._gis._public_rest_url + 'community/users/' + self._user_id + '/info/' + thumbnail_file
            return thumbnail_url_path

    @property
    def homepage(self):
        """Gets the URL to the HTML page for the user."""
        return "{}{}{}".format(self._gis.url,
                               "/home/user.html?user=",
                               self._user_id)

    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        else:
            b64 = base64.b64encode(self.get_thumbnail())
            thumbnail = "data:image/png;base64," + str(b64,"utf-8") + "' width='200' height='133"

        firstName = 'Not Provided'
        lastName = 'Not Provided'
        fullName = 'Not Provided'
        description = "This user has not provided any personal information."

        try:
            firstName = self.firstName
        except:
            firstName = 'Not Provided'

        try:
            lastName = self.lastName
        except:
            firstName = 'Not Provided'

        try:
            fullName = self.fullName
        except:
            fullName = 'Not Provided'

        try:
            description = self.description
        except:
            description = "This user has not provided any personal information."

        url = self.homepage

        return """<div class="9item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='""" + str(url) + """' target='_blank'>
                        <img src='""" + str(thumbnail) + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right" style="float: none; width: auto; overflow: hidden;">
                        <a href='""" + str(url) + """' target='_blank'><b>""" + str(fullName) + """</b>
                        </a>
                        <br/><br/><b>Bio</b>: """ + str(description) + """
                        <br/><b>First Name</b>: """ + str(firstName) + """
                        <br/><b>Last Name</b>: """ + str(lastName)  + """
                        <br/><b>Username</b>: """ + str(self.username)  + """
                        <br/><b>Joined</b>: """ + str(datetime.fromtimestamp(self.created/1000).strftime("%B %d, %Y")) + """

                    </div>
                </div>
                """
    @property
    def groups(self):
        """Gets a list of Group objects the current user belongs to."""
        return [Group(self._gis, group['id']) for group in self['groups']]
    #----------------------------------------------------------------------
    def update_license_type(self, user_type):
        """

        Allows for the updating of the user's licensing type. This allows
        administrators to change a user from a creator to a viewer or any
        other custom user license type.

        **Available in ArcGIS Online and Portal 10.7+**

        =====================  =========================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------
        user_type              Required string. The user license type to assign a user.

                               Built-in Types: creator or viewer
        =====================  =========================================================

        :returns: Boolean

        """
        if self._gis.version < [6,4]:
            raise NotImplementedError("`update_license_type` is not implemented at version %s" % \
                                      ".".join([str(i) for i in self._gis.version]))

        builtin = {
            'creator' : 'creatorUT',
            'viewer' : 'viewerUT'
        }
        if user_type.lower() in builtin:
            user_type = builtin[user_type.lower()]

        url = "%s/portals/self/updateUserLicenseType" % self._portal.resturl
        params = {
            'users' : [self.username],
            'userLicenseType' : user_type,
            'f' : 'json'
        }
        res = self._gis._con.post(url, params)
        status = [r['status'] for r in res['results']]
        self._hydrated = False
        self._hydrate()
        if all(status):
            return all(status)
        return res
    #----------------------------------------------------------------------
    def delete_thumbnail(self):
        """
        Removes the thumbnail from the user's profile.

        :returns: Boolean

        """
        if self._gis.version >= [7,3]:
            url = self._gis._portal.resturl + "community/users/%s/deleteThumbnail" % self.username
            params = {'f' : 'json'}
            res = self._gis._con.post(url, params)
            if 'success' in res:
                return res['success']
            return res
        else:
            raise Exception("The operation delete_thumbnail is not supported on this portal.")
    #----------------------------------------------------------------------
    def reset(self,
              password=None,
              new_password=None,
              new_security_question=None,
              new_security_answer=None,
              reset_by_email=False):
        """ Resets a user's password, security question, and/or security answer.

        .. note::
            This function does not apply to those using enterprise accounts
            that come from an enterprise such as ActiveDirectory, LDAP, or SAML.
            It only has an effect on built-in users.

            If a new security question is specified, a new security answer should
            be provided.

        .. note::
            To reset the password by email, set `reset_by_email` to True and `password`
            to `None`.

        =====================  =========================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------
        password               Required string. The current password.
        ---------------------  ---------------------------------------------------------
        new_password           Optional string. The new password if resetting password.
        ---------------------  ---------------------------------------------------------
        new_security_question  Optional string. The new security question if desired.
        ---------------------  ---------------------------------------------------------
        new_security_answer    Optional string. The new security question answer if desired.
        ---------------------  ---------------------------------------------------------
        reset_by_email         Optional Boolean.  If True, the `user` will be reset by email. The default is False.
        =====================  =========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        postdata = {'f' : 'json'}
        if password:
            postdata['password'] = password
        if new_password:
            postdata['newPassword'] = new_password
        if new_security_question:
            postdata['newSecurityQuestionIdx'] = new_security_question
        if new_security_answer:
            postdata['newSecurityAnswer'] = new_security_answer
        if reset_by_email:
            postdata["email"] = reset_by_email
        url = self._gis._portal.resturl + 'community/users/' + self.username + '/reset'
        resp = self._gis._con.post(url,
                                   postdata,
                                   ssl=True)
        if resp:
            return resp.get('success')
        return False

    def update(self, access=None, preferred_view=None, description=None, tags=None,
               thumbnail=None, fullname=None, email=None, culture=None, region=None,
               first_name=None, last_name=None, security_question=None, security_answer=None):
        """ Updates this user's properties.

        .. note::
            Only pass in arguments for properties you want to update.
            All other properties will be left as they are.  If you
            want to update the description, then only provide
            the description argument.

        .. note::
            When updating the security question, you must provide a
            security_answer as well.

        ==================  ==========================================================
        **Argument**        **Description**
        ------------------  ----------------------------------------------------------
        access              Optional string. The access level for the user, values
                            allowed are private, org, public.
        ------------------  ----------------------------------------------------------
        preferred_view      Optional string. The preferred view for the user, values allowed are Web, GIS, null.
        ------------------  ----------------------------------------------------------
        description         Optional string. A description of the user.
        ------------------  ----------------------------------------------------------
        tags                Optional string. Tags listed as comma-separated values, or a list of strings.
        ------------------  ----------------------------------------------------------
        thumbnail           Optional string. The path or url to a file of type PNG, GIF,
                            or JPEG. Maximum allowed size is 1 MB.
        ------------------  ----------------------------------------------------------
        fullname            Optional string. The full name of this user, only for built-in users.
        ------------------  ----------------------------------------------------------
        email               Optional string. The e-mail address of this user, only for built-in users.
        ------------------  ----------------------------------------------------------
        culture             Optional string. The two-letter language code, fr for example.
        ------------------  ----------------------------------------------------------
        region              Optional string. The two-letter country code, FR for example.
        ------------------  ----------------------------------------------------------
        first_name          Optional string. User's first name.
        ------------------  ----------------------------------------------------------
        last_name           Optional string. User's first name.
        ------------------  ----------------------------------------------------------
        security_question   Optional integer.  The is a number from 1-14.  The
                            questions are as follows:

                            1. What city were you born in?
                            2. What was your high school mascot?
                            3. What is your mother's maden name?
                            4. What was the make of your first car?
                            5. What high school did you got to?
                            6. What is the last name of your best friend?
                            7. What is the middle name of your youngest sibling?
                            8. What is the name of the street on which your grew up?
                            9. What is the name of your favorite fictional character?
                            10. What is the name of your favorite pet?
                            11. What is the name of your favorite restaurant?
                            12. What is the title of your facorite book?
                            13. What is your dream job?
                            14. Where did you go on your first date?

                            Usage Example:

                            security_question=13
        ------------------  ----------------------------------------------------------
        security_answer     Optional string.  This is the answer to security querstion.
                            If you are changing a user's question, an answer must be
                            provided.

                            Usage example:

                            security_answer="Working on the Python API"
        ==================  ==========================================================

        :return:
           A boolean indicating success (True) or failure (False).

        """
        user_type = None
        if tags is not None and \
           isinstance(tags, list):
            tags = ",".join(tags)
        import copy
        params = {"f" : "json",
                  'access' : access,
                  'preferredView' : preferred_view,
                  'description' : description,
                  'tags' : tags,
                  'password' : None,
                  'fullname' : fullname,
                  'email' : email,
                  'securityQuestionIdx' : None,
                  'securityAnswer' : None,
                  'culture' : culture,
                  'region' : region,
                  'firstName' : first_name,
                  'lastName' : last_name,
                  "clearEmptyFields" : True
                  }
        if security_answer and security_question:
            params['securityQuestionIdx'] = security_question
            params['securityAnswer'] = security_answer
        for k,v in copy.copy(params).items():
            if v is None:
                del params[k]

        if thumbnail:
            files = {'thumbnail' : thumbnail}
        else:
            files = None
        url = "%s/sharing/rest/community/users/%s/update" % (self._gis._url,
                                                             self._user_id)
        ret = self._gis._con.post(path=url,
                                  postdata=params,
                                  files=files)
        if ret['success'] == True:
            self._hydrate()
        return ret['success']
    #----------------------------------------------------------------------
    def disable(self):
        """
        Disables login access for the
        user. It is only available to the administrator of the organization.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {"f" : "json"}
        url = "%s/sharing/rest/community/users/%s/disable" % (self._gis._url, self._user_id)
        res = self._gis._con.post(url, params)
        if 'status' in res:
            self._hydrate()
            return res['status'] == 'success'
        elif 'success' in res:
            self._hydrate()
            return res['success']
        return False
    #----------------------------------------------------------------------
    def enable(self):
        """
        Enables login access for the user.
        It is only available to the administrator of the organization.
        """
        params = {"f" : "json"}
        url = "%s/sharing/rest/community/users/%s/enable" % (self._gis._url, self._user_id)
        res = self._gis._con.post(url, params)
        if 'status' in res:
            self._hydrate()
            return res['status'] == 'success'
        elif 'success' in res:
            self._hydrate()
            return res['success']
        return False
    #----------------------------------------------------------------------
    @property
    def esri_access(self):
        """
        When getting, will return a string describing the current user's esri access
        When setting, supply a *bool* to enable or disable esri_access for that user (Administrator privileges required)

        A member whose account has Esri access enabled can use My Esri and
        Community and Forums (GeoNet), access e-Learning on the Training
        website, and manage email communications from Esri. The member
        cannot enable or disable their own access to these Esri resources.

        **Trial** accounts cannot modify esri_access property.

        Please see: http://doc.arcgis.com/en/arcgis-online/administer/manage-members.htm#ESRI_SECTION1_7CE845E428034AE8A40EF8C1085E2A23
        or https://bit.ly/2JsJV1i for more information.


        """
        if self._portal.is_arcgisonline:
            self._hydrate()
            return self['userType']
        else:
            return False
    #----------------------------------------------------------------------
    @esri_access.setter
    def esri_access(self, value):
        """

        Enable or disable 'Esri access'. Administrator privileges required.
        A member whose account has Esri access enabled can use My Esri and
        Community and Forums (GeoNet), access e-Learning on the Training
        website, and manage email communications from Esri. The member
        cannot enable or disable their own access to these Esri resources.

        **Trial** accounts cannot modify esri_access property.

        Please see: http://doc.arcgis.com/en/arcgis-online/administer/manage-members.htm#ESRI_SECTION1_7CE845E428034AE8A40EF8C1085E2A23
        for more information.


        ================  ==========================================================
        **Argument**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required boolean. The current user will be allowed to use
                          the username for other Esri/ArcGIS logins when the value
                          is set to True. If false, the account can only be used to
                          access a given individual's organization.
        ================  ==========================================================
        """
        if self._portal.is_arcgisonline:
            if value == True:
                ret = self._portal.update_user(self._user_id,
                                               user_type="both")
            else:
                ret = self._portal.update_user(self._user_id,
                                               user_type="arcgisonly")
            self._hydrate()
    #----------------------------------------------------------------------
    @property
    def linked_accounts(self):
        """returns all linked account for the current user as User objects"""
        if self._gis._portal.is_arcgisonline == False:
            return []
        url = "%s/sharing/rest/community/users/%s/linkedUsers" % (self._gis._url,
                                                                  self._user_id)
        start = 1
        params = {
            'f' : 'json',
            'num' : 10,
            'start' : start
        }
        users = []
        res = self._gis._con.get(url, params)
        users = res["linkedUsers"]
        if len(users) == 0:
            return users
        else:
            while (res["nextStart"] > -1):
                start += 10
                params['start'] = start
                res = self._gis._con.get(url, params)
                users += res['linkedUsers']
        users = [self._gis.users.get(user['username']) for user in users]
        return users
    #----------------------------------------------------------------------
    def link_account(self, username, user_gis):
        """
        If you use multiple accounts for ArcGIS Online and Esri websites,
        you can link them so you can switch between accounts and share your
        Esri customer information with My Esri, e-Learning, and GeoNet. You
        can link your organizational, public, enterprise, and social login
        accounts. Your content and privileges are unique to each account.
        From Esri websites, only Esri access-enabled accounts appear in
        your list of linked accounts.

        See: http://doc.arcgis.com/en/arcgis-online/reference/sign-in.htm for
        addtional information.

        ================  ==========================================================
        **Argument**      **Description**
        ----------------  ----------------------------------------------------------
        username          required string/User. This is the username or User object
                          that a user wants to link to.
        ----------------  ----------------------------------------------------------
        user_gis          required GIS.  This is the GIS object for the username.
                          In order to link an account, a user must be able to login
                          to that account.  The GIS object is the entry into that
                          account.
        ================  ==========================================================

        returns: Boolean. True for success, False for failure.

        """
        userToken = user_gis._con.token
        if isinstance(username, User):
            username = username.username
        params = {
            'f' : 'json',
            'user' : username,
            'userToken' : userToken
        }
        url = "%s/sharing/rest/community/users/%s/linkUser" % (self._gis._url, self._user_id)
        res = self._gis._con.post(url, params)
        if 'success' in res:
            return res['success']
        return False
    #----------------------------------------------------------------------
    def unlink_account(self, username):
        """
        When a user wishes to no longer have a linked account, the unlink method
        allows for the removal if linked accounts.

        See: http://doc.arcgis.com/en/arcgis-online/reference/sign-in.htm for
        addtional information.

        ================  ==========================================================
        **Argument**      **Description**
        ----------------  ----------------------------------------------------------
        username          required string/User. This is the username or User object
                          that a user wants to unlink.
        ================  ==========================================================

        returns: boolean.
        """
        if isinstance(username, User):
            username = username.username
        params = {
            'f' : 'json',
            'user' : username
        }
        url = "%s/sharing/rest/community/users/%s/unlinkUser" % (self._gis._url, self._user_id)
        res = self._gis._con.post(url, params)
        if 'success' in res:
            return res['success']
        return False
    #----------------------------------------------------------------------
    def update_level(self, level):
        """
        Allows only administrators
        of an organization to update the level of a user. Administrators can
        leverage two levels of membership when assigning roles and
        privileges to members. Membership levels allow organizations to
        control access to some ArcGIS capabilities for some members while
        granting more complete access to other members. Level 1 membership
        is designed for members who need privileges to view and interact
        with existing content, while Level 2 membership is for those who
        contribute, create, and share content and groups, in addition to
        other tasks.

        Maximum user quota of an organization at the given level is checked
        before allowing the update.

        Built-in roles including organization administrator, publisher, and
        user are assigned as Level 2, members with custom roles can be
        assigned as Level 1, 1PlusEdit, or Level 2.

        Level 1 membership allows for limited capabilities given through a
        maximum of 8 privileges: `portal:user:joinGroup,
        portal:user:viewOrgGroups, portal:user:viewOrgItems,
        portal:user:viewOrgUsers, premium:user:geocode,
        premium:user:networkanalysis, premium:user:demographics, and
        premium:user:elevation`. If updating the role of a Level 1 user with
        a custom role that has more privileges than the eight, additional
        privileges will be disabled for the user to ensure restriction.

        Level 1 users are not allowed to own any content or group which can
        be reassigned to other users through the Reassign Item and Reassign
        Group operations before downgrading them. The operation will also
        fail if the user being updated has got licenses assigned to premium
        apps that are not allowed at the targeting level.

        =====================  =========================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------
        level                  Required string. The values of 1 or 2. This
                               is the user level for the given user.


                                    + 1 - View only
                                    + 2 - Content creator


        =====================  =========================================================

        :returns:
           A boolean indicating success (True) or failure (False).
        """
        if self._gis.version >= [6,4]:
            raise NotImplementedError("`update_level` is not applicable at version %s" % \
                                      ".".join([str(i) for i in self._gis.version]))
        if 'roleId' in self and \
           self['roleId'] != 'iAAAAAAAAAAAAAAA':
            self.update_role('iAAAAAAAAAAAAAAA')
            self._hydrated = False
            self._hydrate()
        elif not ('roleId' in self) and level == 1:
            self.update_role('iAAAAAAAAAAAAAAA')
            self._hydrated = False
            self._hydrate()

        allowed_roles = {'1', '2', '11'}

        if level not in allowed_roles:
            raise ValueError("level must be in %s" % ",".join(allowed_roles))

        url = "%s/portals/self/updateUserLevel" % self._portal.resturl
        params = {
            'user' : self.username,
            'level' : level,
            'f' : 'json'
        }
        res = self._gis._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def update_role(self, role):
        """
        Updates this user's role to org_user, org_publisher, org_admin, viewer, view_only,
        viewplusedit, or a custom role.

        .. note::
            There are four types of roles in Portal - user, publisher, administrator and custom roles.
            A user can share items, create maps, create groups, etc.  A publisher can
            do everything a user can do and additionally create hosted services.  An administrator can
            do everything that is possible in Portal. A custom roles privileges can be customized.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        role              Required string. Value must be either org_user,
                          org_publisher, org_admin, viewer, view_only, viewplusedit
                          or a custom role object (from gis.users.roles).
        ================  ========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        lookup = {
            'admin' : 'org_admin',
            'user' : 'org_user',
            'publisher' : 'org_publisher',
            'view_only' : 'tLST9emLCNfFcejK',
            'viewer' : 'iAAAAAAAAAAAAAAA',
            'viewplusedit' : 'iBBBBBBBBBBBBBBB'
        }

        if isinstance(role, Role):
            role = role.role_id
        elif isinstance(role, str):
            if role.lower() in lookup:
                role = lookup[role.lower()]
        passed = self._portal.update_user_role(self.username, role)
        if passed:
            self._hydrated = False
            self._hydrate()
            self.role = role
        return passed

    def delete(self, reassign_to=None):
        """
        Deletes this user from the portal, optionally deleting or reassigning groups and items.

        .. note::
            You can not delete a user in Portal if that user owns groups or items and/or is
            assigned an application bundle.  If you specify someone in the reassign_to
            argument, then items and groups will be transferred to that user.  If that
            argument is not set then the method will fail if the user has items or groups
            that need to be reassigned.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        reassign_to       Optional string. The new owner of the items and groups
                          that belong to the user being deleted.
        ================  ========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        if isinstance(reassign_to, User):
            reassign_to = reassign_to.username

        for l in self._gis.admin.license.all():
            entitle = l.user_entitlement(username=self.username)
            if 'entitlements' in entitle:
                l.revoke(username=self.username,
                         entitlements=entitle['entitlements'],
                         suppress_email=True)
        return self._portal.delete_user(self._user_id, reassign_to)

    def reassign_to(self, target_username):
        """
        Reassigns all of this user's items and groups to another user.

        Items are transferred to the target user into a folder named
        <user>_<folder> where user corresponds to the user whose items were
        moved and folder corresponds to the folder that was moved.

        .. note::
            This method must be executed as an administrator.  This method also
            can not be undone.  The changes are immediately made and permanent.

        ================  ===========================================================
        **Argument**      **Description**
        ----------------  -----------------------------------------------------------
        target_username   Required string. The user who will be the new owner of the
                          items and groups from which these are being reassigned from.
        ================  ===========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        if isinstance(target_username, User):
            target_username = target_username.username
        return self._portal.reassign_user(self._user_id, target_username)

    def get_thumbnail(self):
        """
        Returns the bytes that make up the thumbnail for this user.

        :return:
            Bytes that represent the image.

        .. code-block:: python

            Usage Example:

            response = user.get_thumbnail()
            f = open(filename, 'wb')
            f.write(response)

        """
        thumbnail_file = self.thumbnail
        if thumbnail_file:
            thumbnail_url_path = 'community/users/' + self._user_id + '/info/' + thumbnail_file
            if thumbnail_url_path:
                return self._portal.con.get(thumbnail_url_path, try_json=False, force_bytes=True)

    def download_thumbnail(self, save_folder=None):
        """
        Downloads the item thumbnail for this user.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        save_folder            Optional string. The desired folder name to download the thumbnail to.
        ==================     ====================================================================


        :return:
           The file path of the downloaded thumbnail.
        """
        thumbnail_file = self.thumbnail

        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = 'community/users/' + self._user_id + '/info/' + thumbnail_file
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50: #If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                return self._portal.con.get(path=thumbnail_url_path, try_json=False,
                                            out_folder=save_folder,
                                            file_name=file_name)
        else:
            return None

    @property
    def folders(self):
        """Gets the list of the user's folders"""
        return self._portal.user_folders(self._user_id)

    def items(self, folder=None, max_items=100):
        """
        Provides a list of items in the specified folder. For content in the root folder, use
        the default value of None for the folder argument. For other folders, pass in the folder
        name as a string, or as a dictionary containing
        the folder ID, such as the dictionary obtained from the folders property.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. The specifc folder (as a string or dictionary)
                               to get a list of items in.
        ------------------     --------------------------------------------------------------------
        max_items              Optional integer. The maximum number of items to be returned. The default is 100.
        ==================     ====================================================================


        :return:
           The list of items in the specified folder.

        .. code-block:: python

            # Example to **estimate** storage for a user's items

            storage = 0
            for item in user.items():
                storage += item.size
            try:
                for f in user.folders:
                    for f_item in user.folders(folder=f):
                        storage += f_item.size
                print(f"{user.username} using {storage} bytes")
            except Exception as e:
                print(f"{user.username} using {storage} bytes")
        """

        items = []
        folder_id = None
        if folder is not None:
            if isinstance(folder, str):
                folder_id = self._portal.get_folder_id(self._user_id, folder)
                if folder_id is None:
                    msg = "Could not locate the folder: %s" % folder
                    raise ValueError("%s. Please verify that this folder exists and try again." % msg)
            elif isinstance(folder, dict):
                folder_id = folder['id']
            else:
                print("folder should be folder name as a string"
                      "or a dict containing the folder 'id'")

        resp = self._portal.user_items(self._user_id, folder_id, max_items)
        for item in resp:
            items.append(Item(self._gis, item['id'], item))

        return items
    #----------------------------------------------------------------------
    @property
    def notifications(self):
        """
        Gets the list of notifications available for the given user.
        """
        from .._impl.notification import Notification
        result = []
        url = "%s/community/users/%s/notifications" % (self._portal.resturl, self._user_id)
        params = {"f" : "json"}
        ns = self._portal.con.get(url, params)
        if "notifications" in ns:
            for n in ns["notifications"]:
                result.append(Notification(url="%s/%s" % (url, n['id']),
                                           user=self,
                                           data=n,
                                           initialize=False)
                              )
                del n
            return result
        return result

class Item(dict):
    """
    An item (a unit of content) in the GIS. Each item has a unique identifier and a well
    known URL that is independent of the user owning the item.
    An item can have associated binary or textual data that's available via the item data resource.
    For example, an item of type Map Package returns the actual bits corresponding to the
    map package via the item data resource.

    Items that have layers (eg FeatureLayerCollection items and ImageryLayer items) and tables have
    the dynamic `layers` and `tables` properties to get to the individual layers/tables in this item.
    """

    _uid = None

    def __init__(self, gis, itemid, itemdict=None):
        dict.__init__(self)
        self._portal = gis._portal
        self._gis = gis
        self.itemid = itemid
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        if itemdict:
            self._hydrated = False
            if 'size' in itemdict and itemdict['size'] == -1:
                del itemdict['size'] # remove nonsensical size
            self.__dict__.update(itemdict)
            super(Item, self).update(itemdict)
        else:
            self._hydrated = False
        try:
            self._depend = ItemDependency(item=self)
        except: pass

        if self._has_layers():
            self.layers = None
            self.tables = None
            self['layers'] = None
            self['tables'] = None

    @_lazy_property
    def resources(self):
        """
        Returns the Item's Resource Manager

        :returns: ResourceManager
        """
        return ResourceManager(self, self._gis)
    #----------------------------------------------------------------------
    @property
    def _user_id(self):
        """gets/sets the _user_id property"""
        if self._uid is None:
            user = self._gis.users.get(self.owner)
            if hasattr(user, 'id') and \
               getattr(user, 'id') != 'null':
                #self._uid = user.id
                self._uid = user.username
            else:
                self._uid = user.username
        return self._uid
    #----------------------------------------------------------------------
    @_user_id.setter
    def _user_id(self, value):
        """gets/sets the user id property"""
        self._uid = value
    #----------------------------------------------------------------------
    def _has_layers(self):
        return self.type ==  'Feature Collection' or \
               self.type == 'Feature Service' or \
               self.type == 'Big Data File Share' or \
            self.type == 'Image Service' or \
            self.type == 'Map Service' or \
            self.type == 'Globe Service' or \
            self.type == 'Scene Service' or \
            self.type == 'Network Analysis Service' or \
            self.type == 'Vector Tile Service'

    def _populate_layers(self):
        from arcgis.features import FeatureLayer, FeatureCollection, FeatureLayerCollection, Table
        from arcgis.mapping import VectorTileLayer, MapImageLayer, SceneLayer
        from arcgis.network import NetworkDataset
        from arcgis.raster import ImageryLayer

        if self._has_layers():
            layers = []
            tables = []

            params = {"f" : "json"}

            if self.type == 'Image Service': # service that is itself a layer
                layers.append(ImageryLayer(self.url, self._gis))

            elif self.type == 'Feature Collection':
                lyrs = self.get_data()['layers']
                for layer in lyrs:
                    layers.append(FeatureCollection(layer))

            elif self.type == 'Big Data File Share':
                serviceinfo = self._portal.con.post(self.url, params)
                for lyr in serviceinfo['children']:
                    lyrurl = self.url + '/' + lyr['name']
                    layers.append(Layer(lyrurl, self._gis))


            elif self.type == 'Vector Tile Service':
                layers.append(VectorTileLayer(self.url, self._gis))

            elif self.type == 'Network Analysis Service':
                svc = NetworkDataset.fromitem(self)

                # route laters, service area layers, closest facility layers
                for lyr in svc.route_layers:
                    layers.append(lyr)
                for lyr in svc.service_area_layers:
                    layers.append(lyr)
                for lyr in svc.closest_facility_layers:
                    layers.append(lyr)

            elif self.type == 'Feature Service':
                m = re.search(r'\d+$', self.url)
                if m is not None:  # ends in digit - it's a single layer from a Feature Service
                    layers.append(FeatureLayer(self.url, self._gis))
                else:
                    svc = FeatureLayerCollection.fromitem(self)
                    for lyr in svc.layers:
                        layers.append(lyr)
                    for tbl in svc.tables:
                        tables.append(tbl)

            elif self.type == 'Map Service':
                svc = MapImageLayer.fromitem(self)
                for lyr in svc.layers:
                    layers.append(lyr)
            else:
                m = re.search(r'\d+$', self.url)
                if m is not None: # ends in digit
                    layers.append(FeatureLayer(self.url, self._gis))
                else:
                    svc = _GISResource(self.url, self._gis)
                    for lyr in svc.properties.layers:
                        if self.type == 'Scene Service':
                            lyr_url = svc.url + '/layers/' + str(lyr.id)
                            lyr = SceneLayer(lyr_url, self._gis)
                        else:
                            lyr_url = svc.url+'/'+str(lyr.id)
                            lyr = Layer(lyr_url, self._gis)
                        layers.append(lyr)
                    try:
                        for lyr in svc.properties.tables:
                            lyr = Table(svc.url+'/'+str(lyr.id), self._gis)
                            tables.append(lyr)
                    except:
                        pass

            self.layers = layers
            self.tables = tables
            self['layers'] = layers
            self['tables'] = tables

    def _hydrate(self):
        itemdict = self._portal.get_item(self.itemid)
        self._hydrated = True
        super(Item, self).update(itemdict)
        self.__dict__.update(itemdict)
        try:
            with _DisableLogger():
                self._populate_layers()
        except:
            pass
        user = self._gis.users.get(self.owner)
        if hasattr(user, 'id') and \
           user.id != 'null':
            self._user_id = user.username
            #self._user_id = user.id
        else:
            self._user_id = user.username


    def __getattribute__ (self, name):
        if name == 'layers':
            if self['layers'] == None or self['layers'] == []:
                try:
                    with _DisableLogger():
                        self._populate_layers()
                except Exception as e:
                    if str(e).lower().find("token required") >-1 and self._gis._con._auth.lower() == "pki":
                        with _DisableLogger():
                            self._populate_layers()
                    else:
                        print(e)
                    pass
                return self['layers']
        elif name == 'tables':
            if self['tables'] == None or self['tables'] == []:
                try:
                    with _DisableLogger():
                        self._populate_layers()
                except:
                    pass
                return self['tables']
        return super(Item, self).__getattribute__(name)

    def __getattr__(self, name): # support item attributes
        if not self._hydrated and not name.startswith('_'):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

    def __getitem__(self, k): # support item attributes as dictionary keys on this object
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith('_'):
                self._hydrate()
            return dict.__getitem__(self, k)
    #----------------------------------------------------------------------
    @property
    def content_status(self):
        """
        The content_status property states if an Item is authoritative or deprecated.  This
        givens owners and administrators of Item the ability to warn users that they
        should be either this information or not.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Optional string or None.  Defines if an Item is deprecated or
                               authoritative.
                               If a value of None is given, then the value will be reset.

                               Allowed Values: authoritative, deprecated, or None
        ==================     ====================================================================
        """
        try:
            return self.contentStatus
        except:
            return ""

    #----------------------------------------------------------------------
    @content_status.setter
    def content_status(self, value):
        """
        The content_status property states if an Item is authoritative or deprecated.  This
        givens owners and administrators of Item the ability to warn users that they
        should be either this information or not.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Optional string or None.  Defines if an Item is deprecated or
                               authoritative.
                               If a value of None is given, then the value will be reset.

                               Allowsed Values: authoritative, deprecated, or None
        ==================     ====================================================================
        """
        status_values = ['authoritative',
                         'org_authoritative',
                         'public_authoritative',
                         'deprecated']

        if value is None:
            pass
        elif str(value).lower() not in status_values:
            raise ValueError("%s is not valid value of: authoritative or deprecated" % value)

        if str(value).lower() == 'authoritative':
            value = 'org_authoritative'

        params = {
            'f' : 'json',
            'status' : value
        }
        url = 'content/items/' + self.itemid + '/setContentStatus'

        if value is None:
            value = ""
            params['status'] = ""
            params['clearEmptyFields'] = True
        else:
            params['clearEmptyFields'] = False
        res = self._portal.con.get(url,
                                   params)
        if 'success' in res:
            self.contentStatus = value
            self._hydrate()

    @property
    def homepage(self):
        """Gets the URL to the HTML page for the item."""
        return "{}{}{}".format(self._gis.url,
                               "/home/item.html?id=",
                               self.itemid)

    def copy_feature_layer_collection(self, service_name, layers=None, tables=None, folder=None,
                                      description=None, snippet=None, owner=None):
        """
        This operation allows users to copy existing Feature Layer Collections and select the
        layers/tables that the user wants in the service.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        service_name           Required string. It is the name of the service.
        ------------------     --------------------------------------------------------------------
        layers                 Optional list/string.  This is a either a list of integers or a comma
                               seperated list of integers as a string.  Each index value represents
                               a layer in the feature layer collection.
        ------------------     --------------------------------------------------------------------
        tables                 Optional list/string. This is a either a list of integers or a comma
                               seperated list of integers as a string.  Each index value represents
                               a table in the feature layer collection.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. This is the name of the folder to place in.  The
                               default is None, which means the root folder.
        ------------------     --------------------------------------------------------------------
        description            Optional string. This is the Item description of the service.
        ------------------     --------------------------------------------------------------------
        snippet                Optional string. This is the Item's snippet of the service. It is
                               no longer than 250 characters.
        ------------------     --------------------------------------------------------------------
        owner                  Optional string/User. The default is the current user, but if you
                               want the service to be owned by another user, pass in this value.
        ==================     ====================================================================


        :return:
           Item on success. None on failure

        """
        from ..features import FeatureLayerCollection
        if self.type != "Feature Service" and \
           self.type != "Feature Layer Collection":
            return
        if layers is None and tables is None:
            raise ValueError("An index of layers or tables must be provided")
        content = self._gis.content
        if isinstance(owner, User):
            owner = owner.username
        idx_layers = []
        idx_tables = []
        params = {}
        allowed = ['description', 'allowGeometryUpdates', 'units', 'syncEnabled',
                   'serviceDescription', 'capabilities', 'serviceItemId',
                   'supportsDisconnectedEditing', 'maxRecordCount',
                   'supportsApplyEditsWithGlobalIds', 'name', 'supportedQueryFormats',
                   'xssPreventionInfo', 'copyrightText', 'currentVersion',
                   'syncCapabilities', '_ssl', 'hasStaticData', 'hasVersionedData',
                   'editorTrackingInfo', 'name']
        parent = None
        if description is None:
            description = self.description
        if snippet is None:
            snippet = self.snippet
        i = 1
        is_free = content.is_service_name_available(service_name=service_name,
                                                    service_type="Feature Service")
        if is_free == False:
            while is_free == False:
                i += 1
                s = service_name + "_%s" % i
                is_free = content.is_service_name_available(service_name=s,
                                                            service_type="Feature Service")
                if is_free:
                    service_name = s
                    break
        if len(self.tables) > 0 or len(self.layers) > 0:
            parent = FeatureLayerCollection(url=self.url, gis=self._gis)
        else:
            raise Exception("No tables or layers found in service, cannot copy it.")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                for idx in layers:
                    idx_layers.append(self.layers[idx])
                    del idx
            elif isinstance(layers, (str)):
                for idx in layers.split(','):
                    idx_layers.append(self.layers[idx])
                    del idx
            else:
                raise ValueError("layers must be a comma seperated list of integers or a list")
        if tables is not None:
            if isinstance(tables, (list, tuple)):
                for idx in tables:
                    idx_tables.append(self.tables[idx])
                    del idx
            elif isinstance(tables, (str)):
                for idx in tables.split(','):
                    idx_tables.append(self.tables[idx])
                    del idx
            else:
                raise ValueError("tables must be a comma seperated list of integers or a list")
        for k, v in dict(parent.properties).items():
            if k in allowed:
                if k.lower() == 'name':
                    params[k] = service_name
                if k.lower() == '_ssl':
                    params['_ssl'] = False
                params[k] = v
            del k, v
        if 'name' not in params.keys():
            params['name'] = service_name
        params['_ssl'] = False
        copied_item = content.create_service(name=service_name,
                                             create_params=params,
                                             folder=folder,
                                             owner=owner,
                                             item_properties={'description':description,
                                                              'snippet': snippet,
                                                              'tags' : self.tags,
                                                              'title' : service_name
                                                              })

        fs = FeatureLayerCollection(url=copied_item.url, gis=self._gis)
        fs_manager = fs.manager
        add_defs = {'layers' : [], 'tables' : []}
        for l in idx_layers:
            v = dict(l.manager.properties)
            if 'indexes' in v:
                del v['indexes']
            if 'adminLayerInfo' in v:
                del v['adminLayerInfo']
            add_defs['layers'].append(v)
            del l
        for l in idx_tables:
            v = dict(l.manager.properties)
            if 'indexes' in v:
                del v['indexes']
            if 'adminLayerInfo' in v:
                del v['adminLayerInfo']
            add_defs['tables'].append(v)
            del l
        res = fs_manager.add_to_definition(json_dict=add_defs)
        if res['success'] ==  True:
            return copied_item
        else:
            try:
                copied_item.delete()
            except: pass
        return None

    def download(self, save_path=None, file_name=None):
        """
        Downloads the data to the specified folder or a temporary folder if a folder is not provided.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        save_path           Optional string. Folder location to download the file to.
        ---------------     --------------------------------------------------------------------
        file_name           Optional string. The name of the file.
        ===============     ====================================================================


        :return:
           The download path if data was available, otherwise None.
        """
        data_path = 'content/items/' + self.itemid + '/data'
        if file_name is None:
            import re
            file_name = self.name or self.title
            file_name = re.sub('[^a-zA-Z0-9 \n\.]', '', file_name)
        if not save_path:
            save_path = self._workdir
        if data_path:
            import re
            name = self.name or self.title
            name = re.sub('[^a-zA-Z0-9 \n\.]', '', name)
            download_path = self._portal.con.get(path=data_path, file_name=file_name,
                                                 out_folder=save_path, try_json=False, force_bytes=False)
            if download_path == '':
                return None
            else:
                return download_path

    def export(self, title, export_format,
               parameters=None, wait=True, enforce_fld_vis=None,
               tags=None, snippet=None, overwrite=False):
        """
        Exports a service item to the specified export format.
        Available only to users with an organizational subscription.
        Invokable only by the service item owner or an administrator, unless a Location Tracking
        Service or Location Tracking View.
        This is useful for long running exports that could hold up a script.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The desired name of the exported service item.
        ---------------     --------------------------------------------------------------------
        export_format       Required string. The format to export the data to. Allowed types: 'Shapefile',
                            'CSV', 'File Geodatabase', 'Feature Collection', 'GeoJson', 'Scene Package', 'KML',
                             'Excel', 'geoPackage', or 'Vector Tile Package'.
        ---------------     --------------------------------------------------------------------
        parameters          Optional string. A JSON object describing the layers to be exported
                            and the export parameters for each layer.  See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Export_Item/02r30000008s000000/
                            for guidance.
        ---------------     --------------------------------------------------------------------
        wait                Optional boolean. Default is True, which forces a wait for the
                            export to complete; use False for when it is okay to proceed while
                            the export continues to completion.
        ---------------     --------------------------------------------------------------------
        enforce_fld_vis     Optional boolean. Be default when you are the owner of an item and
                            the `export` operation is called, the data provides all the columns.
                            If the export is being perform on a view, to ensure the view's
                            column definition is honor, then set the value to True. When the
                            owner of the service and the value is set to False, all data and
                            columns will be exported.
        ---------------     --------------------------------------------------------------------
        tags                Optional String.  A comma seperated value of item descriptors.
        ---------------     --------------------------------------------------------------------
        snippet             Optional String. A short descriptive piece of text.
        ---------------     --------------------------------------------------------------------
        overwrite           Optional Boolean. If the export Item exists, the item will be
                            replaced with the new one.
        ===============     ====================================================================


        :return:
           Item or dictionary.  Item is returned when wait=True. A dictionary describing the status of
           the item is returned when wait=False.
        """
        import time
        formats = ['Shapefile',
                   'CSV',
                   'File Geodatabase',
                   'Feature Collection',
                   'GeoJson',
                   'GeoPackage', # geoPackage
                   'geoPackage',
                   'Scene Package',
                   'KML',
                   'Excel',
                   'Vector Tile Package']
        if export_format == 'GeoPackage':
            export_format = 'geoPackage'
        user_id = self._user_id
        # allow exporting of LTS / LTV even if not owner for ArcGIS Online
        if not self._gis.properties['isPortal'] and 'Location Tracking Service' in self.typeKeywords:
            user_id = self._gis.users.me.username
        data_path = 'content/users/%s/export' % user_id
        params = {
            "f" : "json",
            "itemId" : self.itemid,
            "exportFormat" : export_format,
            "title" : title
        }
        if tags and isinstance(tags, (list, tuple)):
            tags = ",".join([str(t) for t in tags])
        if tags and isinstance(tags, str):
            params['tags'] = tags
        if snippet:
            params['snippet'] = snippet
        if parameters:
            params.update({'exportParameters': parameters})
        if not enforce_fld_vis is None and \
           'View Service' in self.typeKeywords:
            if 'exportParameters' in params:
                params['exportParameters']["enforceFieldVisibility"] = enforce_fld_vis
            else:
                params['exportParameters'] = {"enforceFieldVisibility" : enforce_fld_vis }
        res = self._portal.con.post(data_path, params)
        export_item = Item(gis=self._gis, itemid=res['exportItemId'])
        if wait == True:
            status = "partial"
            while status != "completed":
                status = export_item.status(job_id=res['jobId'],
                                            job_type="export")
                if status['status'] == 'failed':
                    raise Exception("Could not export item: %s" % self.itemid)
                elif status['status'].lower() == "completed":
                    return export_item
                time.sleep(2)
        return res
    #----------------------------------------------------------------------
    def status(self, job_id=None, job_type=None):
        """
        Provides the status when publishing an item, adding an item in
        async mode, or adding with a multipart upload. "Partial" is
        available for Add Item Multipart, when only a part is uploaded
        and the item is not committed.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Optional string. The job ID returned during publish, generateFeatures,
                            export, and createService calls.
        ---------------     --------------------------------------------------------------------
        job_type            Optional string. The type of asynchronous job for which the status
                            has to be checked. Default is none, which checks the item's status.
                            This parameter is optional unless used with the operations listed
                            below. Values: `publish`, generateFeatures, export, and createService
        ===============     ====================================================================


        :return:
           The status of a publishing item.
        """
        params = {
            "f" : "json"
        }
        data_path = 'content/users/%s/items/%s/status' % (self._user_id,
                                                          self.itemid)
        if job_type is not None:
            params['jobType'] = job_type
        if job_id is not None:
            params["jobId"] = job_id
        return self._portal.con.get(data_path,
                                    params)
    #----------------------------------------------------------------------
    def get_thumbnail(self):
        """
        Retrieves the bytes that make up the thumbnail for this item.

        :return:
           Bytes that represent the item.

        Example

        .. code-block:: python

            response = item.get_thumbnail()
            f = open(filename, 'wb')
            f.write(response)

        """
        thumbnail_file = self.thumbnail
        if thumbnail_file:
            thumbnail_url_path = 'content/items/' + self.itemid + '/info/' + thumbnail_file
            if thumbnail_url_path:
                return self._portal.con.get(thumbnail_url_path, try_json=False, force_bytes=True)

    def download_thumbnail(self, save_folder=None):
        """
        Downloads the thumbnail for this item.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        save_folder          Optional string. Folder location to download the item's thumbnail to.
        ===============     ====================================================================


        :return:
           For a successful download of the thumbnail, a file path. None if the item does not have a thumbnail.
        """
        if self.thumbnail is None:
            self._hydrate()
        thumbnail_file = self.thumbnail

        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = 'content/items/' + self.itemid  + '/info/' + thumbnail_file
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50: #If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                self._portal.con.get(path=thumbnail_url_path, try_json=False,
                                     out_folder=save_folder,
                                     file_name=file_name)
                return file_path
        else:
            return None

    def get_thumbnail_link(self):
        """ URL to the thumbnail image. """
        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            if self._gis.properties.portalName == 'ArcGIS Online':
                return 'http://static.arcgis.com/images/desktopapp.png'
            else:
                return self._gis.url + '/portalimages/desktopapp.png'
        else:
            thumbnail_url_path = self._gis._public_rest_url + '/content/items/' + self.itemid + '/info/' + thumbnail_file
            return thumbnail_url_path

    @property
    def metadata(self):
        """ Gets and sets the item metadata for the specified item.
            Returns None if the item does not have metadata.
            Items with metadata have 'Metadata' in their typeKeywords.
        """
        metadataurlpath = 'content/items/' + self.itemid  + '/info/metadata/metadata.xml'
        try:
            return self._portal.con.get(metadataurlpath, try_json=False)

        # If the get operation returns a 400 HTTP Error then the metadata simply
        # doesn't exist, let's just return None in this case
        except HTTPError as e:
            if e.code == 400 or e.code == 500:
                return None
            else:
                raise e

    #----------------------------------------------------------------------
    @metadata.setter
    def metadata(self, value):
        """
        For metadata enabled site, users can get/set metadata from a file
        or XML text.
        """
        import shutil
        from six import string_types
        xml_file = os.path.join(tempfile.gettempdir(), 'metadata.xml')
        if os.path.isfile(xml_file) == True:
            os.remove(xml_file)
        if os.path.isfile(value) == True and \
           str(value).lower().endswith('.xml'):
            if os.path.basename(value).lower() != 'metadata.xml':
                shutil.copy(value, xml_file)
            else:
                xml_file = value
        elif isinstance(value, string_types):
            with open(xml_file, mode='w') as writer:
                writer.write(value)
                writer.close()
        else:
            raise ValueError("Input must be XML path file or XML Text")
        return self.update(metadata=xml_file)

    def download_metadata(self, save_folder=None):
        """
        Downloads the item metadata for the specified item id. Items with metadata have 'Metadata'
        in their typeKeywords.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        save_folder          Optional string. Folder location to download the item's metadata to.
        ===============     ====================================================================


        :return:
           For a successful download of metadata, a file path. None if the item does not have metadata.
        """
        metadataurlpath = 'content/items/' + self.itemid + '/info/metadata/metadata.xml'
        if not save_folder:
            save_folder = self._workdir
        try:
            file_name="metadata.xml"
            file_path = os.path.join(save_folder, file_name)
            self._portal.con.get(path=metadataurlpath,
                                 out_folder=save_folder,
                                 file_name=file_name, try_json=False)
            return file_path

        # If the get operation returns a 400 HTTP/IO Error then the metadata
        # simply doesn't exist, let's just return None in this case
        except HTTPError as e:
            if e.code == 400 or e.code == 500:
                return None
            else:
                raise e

    def _get_icon(self):
        icon = "layers16.png"
        if self.type.lower() == "web map":
            icon = "maps16.png"
        elif self.type.lower() == "web scene":
            icon = "websceneglobal16.png"
        elif self.type.lower() == "cityengine web scene":
            icon = "websceneglobal16.png"
        elif self.type.lower() == "pro map":
            icon = "mapsgray16.png"
        elif self.type.lower() == "feature service" and "Table" in self.typeKeywords:
            icon = "table16.png"
        elif self.type.lower() == "feature service":
            icon = "featureshosted16.png"
        elif self.type.lower() == "map service":
            icon = "mapimages16.png"
        elif self.type.lower() == "image service":
            icon = "imagery16.png"
        elif self.type.lower() == "kml":
            icon = "features16.png"
        elif self.type.lower() == "wms":
            icon = "mapimages16.png"
        elif self.type.lower() == "feature collection":
            icon = "features16.png"
        elif self.type.lower() == "feature collection template":
            icon = "maps16.png"
        elif self.type.lower() == "geodata service":
            icon = "layers16.png"
        elif self.type.lower() == "globe service":
            icon = "layers16.png"
        elif self.type.lower() == "shapefile":
            icon = "datafiles16.png"
        elif self.type.lower() == "web map application":
            icon = "apps16.png"
        elif self.type.lower() == "map package":
            icon = "mapsgray16.png"
        elif self.type.lower() == "feature layer":
            icon = "featureshosted16.png"
        elif self.type.lower() == "map service":
            icon = "maptiles16.png"
        elif self.type.lower() == "map document":
            icon = "mapsgray16.png"
        else:
            icon = "layers16.png"

        icon = self._gis.url + '/home/js/jsapi/esri/css/images/item_type_icons/' + icon
        return icon

    def _ux_item_type(self):
        item_type= self.type
        if self.type == 'Geoprocessing Service':
            item_type = 'Geoprocessing Toolbox'
        elif self.type.lower() == 'feature service' and 'Table' in self.typeKeywords:
            item_type = 'Table Layer'
        elif self.type.lower() == 'feature service':
            item_type = 'Feature Layer Collection'
        elif self.type.lower() == 'map service':
            item_type = 'Map Image Layer'
        elif self.type.lower() == 'image service':
            item_type = 'Imagery Layer'
        elif self.type.lower().endswith('service'):
            item_type = self.type.replace('Service', 'Layer')
        return item_type

    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        else:
            try:
                b64 = base64.b64encode(self.get_thumbnail())
                thumbnail = "data:image/png;base64," + str(b64,"utf-8") + "' width='200' height='133"
            except:
                if self._gis.properties.portalName == 'ArcGIS Online':
                    thumbnail = 'http://static.arcgis.com/images/desktopapp.png'
                else:
                    thumbnail = self._gis.url + '/portalimages/desktopapp.png'

        snippet = self.snippet
        if snippet is None:
            snippet = ""

        portalurl = self.homepage

        locale.setlocale(locale.LC_ALL, '')
        numViews = locale.format("%d", self.numViews, grouping=True)
        return """<div class="item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='""" + portalurl + """' target='_blank'>
                        <img src='""" + thumbnail + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right"     style="float: none; width: auto; overflow: hidden;">
                        <a href='""" + portalurl + """' target='_blank'><b>""" + self.title + """</b>
                        </a>
                        <br/>""" + snippet + """<img src='""" + self._get_icon() +"""' style="vertical-align:middle;">""" + self._ux_item_type() + """ by """ + self.owner + """
                        <br/>Last Modified: """ + datetime.fromtimestamp(self.modified/1000).strftime("%B %d, %Y") + """
                        <br/>""" + str(self.numComments) + """ comments, """ +  str(numViews) + """ views
                    </div>
                </div>
                """

    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    def __repr__(self):
        return '<%s title:"%s" type:%s owner:%s>' % (type(self).__name__, self.title, self._ux_item_type(), self.owner)

    def reassign_to(self, target_owner, target_folder=None):
        """
        Allows the administrator to reassign a single item from one user to another.

        .. note::
            If you wish to move all of a user's items (and groups) to another user then use the
            user.reassign_to() method.  This method only moves one item at a time.

        ================  ========================================================
        **Argument**      **Description**
        ----------------  --------------------------------------------------------
        target_owner      Required string. The new desired owner of the item.
        ----------------  --------------------------------------------------------
        target_folder     Optional string. The folder to move the item to.
        ================  ========================================================

        :return:
            A boolean indicating success (True) with the ID of the reassigned item, or failure (False).

        """
        try:
            current_folder = self.ownerFolder
        except:
            current_folder = None
        resp = self._portal.reassign_item(self.itemid, self._user_id, target_owner, current_folder, target_folder)
        if resp is True:
            self._hydrate() # refresh
            return resp

    @property
    def shared_with(self):
        """
        Reveals the privacy or sharing status of the current item. An item can be private or shared with one or more of
        the following: A specified list of groups, to all members in the organization or to everyone (including
        anonymous users). If the return is False for `org`, `everyone` and contains an empty list of `groups`, then the
        item is private and visible only to the owner.

        :return:
            Dictionary of the following kind
            {
            'groups': [],  # one or more Group objects
            'everyone': True | False,
            'org': True | False
            }
        """
        if not self._hydrated:
            self._hydrate()  # hydrated properties needed below

        # find if portal is ArcGIS Online
        if self._gis._portal.is_arcgisonline:
            # Call with owner info
            if self._user_id != self._gis.users.me.username:
                url = "{resturl}content/items/{itemid}/groups".format(
                    resturl=self._gis._portal.resturl,
                    itemid=self.itemid
                )
                resp = self._portal.con.post(url, {'f': 'json'})
                ret_dict = {'everyone': self.access == 'public',
                            'org': (self.access == 'public' or self.access == 'org'),
                            'groups': []}
                for grpid in resp.get('admin', []) + resp.get("other", []) + resp.get("member", []):
                    try:
                        grp = Group(gis=self._gis, groupid=grpid['id'])
                        ret_dict['groups'].append(grp)
                    except:
                        pass
                return ret_dict
            else:
                resp = self._portal.con.get('content/users/' + self._user_id + "/items/" + self.itemid)

        else:  # gis is a portal, find if item resides in a folder
            if self._user_id != self._gis.users.me.username:
                url = "{resturl}content/items/{itemid}/groups".format(
                    resturl=self._gis._portal.resturl,
                    itemid=self.itemid
                )
                resp = self._portal.con.post(url, {'f': 'json'})
                ret_dict = {'everyone': self.access == 'public',
                            'org': (self.access == 'public' or self.access == 'org'),
                            'groups': []}
                for grpid in resp.get('admin', []) + resp.get("other", []) + resp.get("member", []):
                    try:
                        grp = Group(gis=self._gis, groupid=grpid['id'])
                        ret_dict['groups'].append(grp)
                    except:
                        pass
                return ret_dict
            if self.ownerFolder is not None:
                resp = self._portal.con.get('content/users/' + self._user_id + '/' + self.ownerFolder + "/items/" +
                                            self.itemid)
            else:
                resp = self._portal.con.get('content/users/' + self._user_id + "/items/" + self.itemid)

        # Get the sharing info
        sharing_info = resp['sharing']
        ret_dict = {'everyone': False,
                    'org': False,
                    'groups': []}

        if sharing_info['access'] == 'public':
            ret_dict['everyone'] = True
            ret_dict['org'] = True

        if sharing_info['access'] == 'org':
            ret_dict['org'] = True

        if len(sharing_info['groups']) > 0:
            grps = []
            for g in sharing_info['groups']:
                try:
                    grps.append(Group(self._gis, g))
                except: # ignore groups you can't access
                    pass
            ret_dict['groups'] = grps

        return ret_dict

    def share(self, everyone=False, org=False, groups=None, allow_members_to_edit=False):
        """
        Shares an item with the specified list of groups.

        ======================  ========================================================
        **Argument**            **Description**
        ----------------------  --------------------------------------------------------
        everyone                Optional boolean. Default is False, don't share with
                                everyone.
        ----------------------  --------------------------------------------------------
        org                     Optional boolean. Default is False, don't share with
                                the organization.
        ----------------------  --------------------------------------------------------
        groups                  Optional list of group ids as strings, or a list of
                                arcgis.gis.Group objects, or a comma-separated list of
                                group IDs.
        ----------------------  --------------------------------------------------------
        allow_members_to_edit   Optional boolean. Default is False, to allow item to be
                                shared with groups that allow shared update
        ======================  ========================================================

        :return:
            A dictionary with key "notSharedWith" containing array of groups with which the item could not be shared.

        """
        if everyone:
            org = True
        try:
            folder = self.ownerFolder
        except:
            folder = None

        # get list of group IDs
        group_ids = ''
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, Group):
                    group_ids = group_ids + "," + group.id

                elif isinstance(group, str):
                    # search for group using id
                    search_result = self._gis.groups.search(query='id:' + group, max_groups=1)
                    if len(search_result) > 0:
                        group_ids = group_ids + "," + search_result[0].id
                    else:
                        raise Exception("Cannot find group with id: " + group)
                else:
                    raise Exception("Invalid group(s)")

        elif isinstance(groups, str):
            #old API - groups sent as comma separated group ids
            group_ids = groups
        if self.owner == self._gis.users.me.username:

            url = "{resturl}content/users/{owner}/shareItems".format(resturl=self._gis._portal.resturl,
                                                                      owner=self.owner)
            params = {
                'f' : 'json',
                'items' : self.id,
                "groups": group_ids,
                "everyone": everyone,
                "account": org
            }
            if allow_members_to_edit:
                params['owner'] = self.owner
                params['confirmItemControl'] = allow_members_to_edit  # True
        else:
            url = "{resturl}/content/items/{itemid}/share".format(resturl=self._gis._portal.resturl,
                                                                  itemid=self.itemid)
            params = {
                'f' : 'json',
                "groups": group_ids,
                "everyone": everyone,
                "account": org
            }

        res = self._portal.con.post(url, params)
        self._hydrated = False
        self._hydrate()
        return res

    def unshare(self, groups):
        """
        Stops sharing of the item with the specified list of groups.


        ================  =========================================================================================
        **Argument**      **Description**
        ----------------  -----------------------------------------------------------------------------------------
        groups            Optional list of group names as strings, or a list of arcgis.gis.Group objects,
                          or a comma-separated list of group IDs.
        ================  =========================================================================================


        :return:
            Dictionary with key "notUnsharedFrom" containing array of groups from which the item could not be unshared.
        """
        try:
            folder = self.ownerFolder
        except:
            folder = None

        # get list of group IDs
        group_ids = ''
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, Group):
                    group_ids = group_ids + "," + group.id

                elif isinstance(group, str):
                    # search for group using id
                    search_result = self._gis.groups.search(query='id:' + group, max_groups=1)
                    if len(search_result) > 0:
                        group_ids = group_ids + "," + search_result[0].id
                    else:
                        raise Exception("Cannot find group with id: " + group)
                else:
                    raise Exception("Invalid group(s)")

        elif isinstance(groups, str):
            # old API - groups sent as comma separated group ids
            group_ids = groups

        if self.access == 'public':
            return self._portal.unshare_item_as_group_admin(self.itemid, group_ids)
        else:
            owner = self._user_id
            return self._portal.unshare_item(self.itemid, owner, folder, group_ids)

    def delete(self, force=False, dry_run=False):
        """
        Deletes the item. If unable to delete, raises a RuntimeException. To know if you can safely delete the item,
        use the optional parameter 'dry_run'

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        force               Optional bool. Available in ArcGIS Enterprise 10.6.1 and higher.
                            Force deletion is applicable only to items that were orphaned when
                            a server federated to the ArcGIS Enterprise was removed accidentally
                            before properly unfederating it. When called on other items, it has
                            no effect.
        ---------------     --------------------------------------------------------------------
        dry_run             Optional bool. Available in ArcGIS Enterprise 10.6.1 and higher.If
                            True, checks if the item can be safely deleted and gives you back
                            either a dictionary with details. If dependent items are preventing
                            deletion, a list of such Item objects are provided.
        ===============     ====================================================================

        :return:
            A bool containing True (for success) or False (for failure). When dry_run is used, a dictionary with
            details is returned.

        .. code-block:: python

            USAGE EXAMPLE: Successful deletion of an item

            item1 = gis.content.get('itemId12345')
            item1.delete()

            >> True

        .. code-block:: python

            USAGE EXAMPLE: Failed deletion of an item

            item1 = gis.content.get('itemId12345')
            item1.delete()

            >> RuntimeError: Unable to delete item. This service item has a related Service item
            >> (Error Code: 400)

        .. code-block:: python

            USAGE EXAMPLE: Dry run to check deletion of an item

            item1 = gis.content.get('itemId12345abcde')
            item1.delete(dry_run=True)

            >> {'can_delete': False,
            >> 'details': {'code': 400,
            >> 'message': 'Unable to delete item. This service item has a related Service item',
            >> 'offending_items': [<Item title:"Chicago_accidents_WFS" type:WFS owner:sharing1>]}}

        .. note::
            During the dry run, if you receive a list of offending items, attempt to delete them first before deleting
            the current item. You can in turn call 'dry_run' on those items to ensure they can be deleted safely.
        """

        try:
            folder = self.ownerFolder
        except:
            folder = None

        if dry_run:
            can_delete_resp = self._portal.can_delete(self.itemid, self._user_id, folder)
            if can_delete_resp[0]:
                return {'can_delete':True}
            else:
                error_dict = {'code': can_delete_resp[1].get('code'),
                              'message': can_delete_resp[1].get('message'),
                              'offending_items': [Item(self._gis, e['itemId']) for e in
                                                  can_delete_resp[1].get('offendingItems')]}

                return {'can_delete':False, 'details': error_dict}
        else:
            return self._portal.delete_item(self.itemid, self._user_id, folder, force)

    def create_thumbnail(self, update=True):
        """
        Creates a Thumbnail for a feature service portal item using the service's symbology
        and the print service registered for the enterprise.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        update              Optional boolean. When set to True, the item will be updated with the
                            thumbnail generated in this call, else it will not update the item.
                            The default is True.
        ===============     ====================================================================

        :returns: DataFile

        """
        from arcgis.geoprocessing._tool import Toolbox
        from arcgis.features import FeatureLayer, FeatureLayerCollection
        from arcgis.gis.server._service import Service
        props = self._gis.properties
        gp_url = os.path.dirname(self._gis.properties.helperServices.printTask.url)

        if self.type == 'Feature Service':
            layers = []
            container = self.layers[0].container
            extent = container.properties.initialExtent
            for lyr in self.layers:
                layers.append(
                    {
                        "id":"%s_%s" % (lyr.properties.serviceItemId, lyr.properties.id),
                        "title": lyr.properties.name,
                        "opacity":1,
                        "minScale": lyr.properties.minScale,
                        "maxScale": lyr.properties.maxScale,
                        "layerDefinition": {
                            "drawingInfo": dict(lyr.properties.drawingInfo)},
                        "token": self._gis._con.token,
                        "url": lyr._url
                    }
                )
                del lyr
            wmjs = {
                "mapOptions":{
                    "showAttribution":False,
                    "extent": dict(extent),
                    "spatialReference":dict(container.properties.spatialReference)
                    },
                "operationalLayers": layers,
                "exportOptions":{
                    "outputSize":[600,400],
                    "dpi":96
                }
            }

        elif self.type == 'Web Map':
            import json
            layers = []
            mapjson = self.get_data()
            container = None
            for lyr in mapjson['baseMap']['baseMapLayers']:
                del lyr['layerType']
                layers.append(lyr)
            for lyr in mapjson['operationalLayers']:
                flyr = Service(url=lyr['url'], server=self._gis._con)
                if container is None and isinstance(flyr, FeatureLayer):
                    container = FeatureLayerCollection(url=os.path.dirname(flyr._url), gis=self._gis)
                layers.append(
                    {
                        "id":"%s" % lyr['id'],
                        "title": lyr['title'],
                        "opacity": lyr['opacity'],
                        "minScale": flyr.properties.minScale,
                        "maxScale": flyr.properties.maxScale,
                        "layerDefinition": {
                            "drawingInfo": dict(flyr.properties.drawingInfo)},
                        "token": self._gis._con.token,
                        "url": lyr['url']
                    }
                )
                del lyr
            wmjs = {
                "mapOptions":{
                    "showAttribution":False,
                    "extent": dict(container.properties.initialExtent),
                    "spatialReference": dict(container.properties.spatialReference)
                    },
                "operationalLayers": layers,
                "exportOptions":{
                    "outputSize":[600,400],
                    "dpi":96
                }
            }
            print()
        else:
            return None
        if isinstance(self._gis._portal, _portalpy.Portal) and \
           self._gis._portal.is_arcgisonline:
            tbx = Toolbox(url=gp_url)
        else:
            tbx = Toolbox(url=gp_url, gis=self._gis)
        res = tbx.export_web_map_task(web_map_as_json=wmjs,format="png32")
        if update:
            self.update(item_properties={'thumbnailUrl' : res.url})
        return res

    def update(self, item_properties=None, data=None, thumbnail=None, metadata=None):
        """ Updates an item in a Portal.


        .. note::
            Content can be a file (such as a layer package, geoprocessing package,
            map package) or a URL (to an ArcGIS Server service, WMS service,
            or an application).

            To upload a package or other file, provide a path or URL
            to the file in the data argument.

            For item_properties, pass in arguments for only the properties you want to be updated.
            All other properties will be untouched.  For example, if you want to update only the
            item's description, then only provide the description argument in item_properties.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        data                Optional string. Either a path or URL to the data.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ===============     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*


        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        type               Optional string. Indicates type of item, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        url                Optional string. URL to item that are based on URLs.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        text               Optional string. For text based items such as Feature Collections & WebMaps
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        extent             Optional string. Provide comma-separated values for min x, min y, max x, max y.
        -----------------  ---------------------------------------------------------------------
        spatialReference   Optional string. Coordinate system that the item is in.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        -----------------  ---------------------------------------------------------------------
        commentsEnabled    Optional boolean. Default is true, controls whether comments are allowed (true)
                           or not allowed (false).
        =================  =====================================================================


        URL 1: http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#//02r3000000ms000000

        :return:
           A boolean indicating success (True) or failure (False).
        """
        #owner = self._gis.users.get(self.owner)
        owner = self._user_id
        #if hasattr(owner, 'id') and \
        #   owner.id != 'null':
        #    owner = owner.id
        #else:
        #    owner = owner.username
        try:
            folder = self.ownerFolder
        except:
            folder = None

        if item_properties:
            large_thumbnail = item_properties.pop("largeThumbnail", None)
        else:
            large_thumbnail = None

        if item_properties is not None:
            if 'tags' in item_properties:
                if type(item_properties['tags']) is list:
                    item_properties['tags'] = ",".join(item_properties['tags'])

        ret = self._portal.update_item(self.itemid, item_properties, data,
                                       thumbnail, metadata, owner, folder,
                                       large_thumbnail)
        if ret:
            self._hydrate()
        return ret

    def usage(self, date_range='7D', as_df=True):
        """

        ArcGIS Online Only

        For item owners and administrators, usage provides usage details about an item that help you
        gauge its popularity. Usage details show how many times the item has been used for the time
        period you select. Historical usage information is available for the past year. Depending on
        the item type, usage details can include the number of views, requests, or downloads, and
        the average number of views, requests, or downloads per day.

        Views refers to the number of times the item has been viewed or opened. For maps, scenes,
        nonhosted layers, and web apps, the view count is increased by one when you open the item
        page or open the item in Map Viewer. For example, if you opened the item page for a map
        image layer and clicked Open in Map Viewer, the count would increase by two. For other items
        such as mobile apps, KML, and so on, the view count is increased by one when you open the
        item; the count does not increase when you open the item details page.

        For hosted web layers (feature, tile, and scene), the number of requests is provided instead
        of views. Requests refers to the number of times a request is made for the data within the
        layer. For example, someone might open an app that contains a hosted feature layer. Opening
        the app counts as one view for the application, but multiple requests may be necessary to
        draw all the features in the hosted layer and are counted as such.

        For downloadable file item types such as CSV, SHP, and so on, the number of downloads is
        displayed. For registered apps, the Usage tab also displays the number of times users have
        logged in to the app. Apps that allow access to subscriber content through the organization
        subscription show usage by credits. You can change the time frame for the credit usage
        reporting period.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        date_range          Optional string.  The default is 7d.  This is the period to query
                            usage for a given item.

                            =======  =========================
                            24H      Past 24 hours
                            -------  -------------------------
                            7D       Past 7 days (default)
                            -------  -------------------------
                            14D      Past 14 days
                            -------  -------------------------
                            30D      Past 30 days
                            -------  -------------------------
                            60D      Past 60 days
                            -------  -------------------------
                            6M       Past 6 months
                            -------  -------------------------
                            1Y       Past 12 months
                            =======  =========================
        ---------------     --------------------------------------------------------------------
        as_df               Optional boolean.  Returns a Pandas DataFrame when True, returns data
                            as a dictionary when False
        ===============     ====================================================================

        :returns: Pandas DataFrame or Dictionary

        """
        if not self._portal.is_arcgisonline:
            raise ValueError("Usage() only supported for ArcGIS Online items.")
        end_date = None
        if end_date is None:
            end_date = datetime.now()
        params = {
            'f' : 'json',
            'startTime': None,
            'endTime': int(end_date.timestamp() * 1000),
            "period": '',
            'vars': 'num',
            'groupby': 'name',
            'etype': 'svcusg',
            'name': self.itemid,

        }
        from datetime import timedelta
        if self.type == 'Feature Service':
            params['stype'] = 'features'
            if not self.layers[0].container:
                params['name'] = os.path.basename(os.path.abspath(os.path.join(self.layers[0]._url, ".." + os.sep + "..")))
            else:
                params['name'] = os.path.basename(os.path.dirname(self.layers[0].container._url))
        if date_range.lower() in ['24h', '1d']:
            params['period'] = '1h'
            params['startTime'] = int((end_date - timedelta(days=1)).timestamp() * 1000)
        elif date_range.lower() == '7d':
            params['period'] = '1d'
            params['startTime'] = int((end_date - timedelta(days=7)).timestamp() * 1000)
        elif date_range.lower() == '14d':
            params['period'] = '1d'
            params['startTime'] = int((end_date - timedelta(days=14)).timestamp() * 1000)
        elif date_range.lower() == '30d':
            params['period'] = '1d'
            params['startTime'] = int((end_date - timedelta(days=30)).timestamp() * 1000)
        elif date_range.lower() == '60d':
            params['period'] = '1d'
            params['startTime'] = int((end_date - timedelta(days=60)).timestamp() * 1000)
        elif date_range.lower() == '6m':
            sd = end_date - timedelta(days=int(365/2))
            ranges = {
                "1" : [sd, sd + timedelta(days=60)],
                "2" : [sd + timedelta(days=61), sd + timedelta(days=120)],
                "3" : [sd + timedelta(days=121), sd + timedelta(days=180)],
                "4" : [sd + timedelta(days=181), end_date + timedelta(days=1)]
            }
            params['period'] = '1d'
            if self._gis._portal.is_logged_in:
                url = "%s/portals/%s/usage" % (self._portal.resturl, self._gis.properties.id)
            else:
                url = "%s/portals/%s/usage" % (self._portal.resturl, "self")
            results = []
            for k,v in ranges.items():
                sd = int(v[0].timestamp() * 1000)
                ed = int(v[1].timestamp() * 1000)
                params['startTime'] = sd
                params['endTime'] = ed
                res = self._portal.con.post(url, params)
                if as_df:
                    import pandas as pd

                    if 'data' not in res or len(res['data']) == 0:
                        res = pd.DataFrame([],
                                           columns=['Date', 'Usage'])
                    elif len(res['data']):
                        res = pd.DataFrame(res['data'][0]['num'],
                                           columns=['Date', 'Usage'])
                        res.Date = res.astype(float) / 1000
                        res.Date = res.Date.apply(lambda x: datetime.fromtimestamp(x))
                        res.Usage = res.Usage.astype(int)

                results.append(res)
                del k,v
            if as_df:
                if len(results):
                    return (pd.concat(results)
                            .reset_index(drop=True)
                            .drop_duplicates(keep='first',
                                             inplace=False))
                else:
                    return (pd.DataFrame([],
                                         columns=['Date', 'Usage']))
            else:
                return results
        elif date_range.lower() in ['12m', '1y']:
            sd = end_date - timedelta(days=int(365))
            ranges = {
                "1" : [sd, sd + timedelta(days=60)],
                "2" : [sd + timedelta(days=61), sd + timedelta(days=120)],
                "3" : [sd + timedelta(days=121), sd + timedelta(days=180)],
                "4" : [sd + timedelta(days=181), sd + timedelta(days=240)],
                "5" : [sd + timedelta(days=241), sd + timedelta(days=320)],
                "6" : [sd + timedelta(days=321), sd + timedelta(days=366)]
            }
            params['period'] = '1d'
            url = "%s/portals/%s/usage" % (self._portal.resturl, self._gis.properties.id)
            results = []
            for k,v in ranges.items():
                sd = int(v[0].timestamp() * 1000)
                ed = int(v[1].timestamp() * 1000)
                params['startTime'] = sd
                params['endTime'] = ed
                res = self._portal.con.post(url, params)
                if as_df:
                    import pandas as pd

                    if 'data' not in res or len(res['data']) == 0:
                        res = pd.DataFrame([],
                                           columns=['Date', 'Usage'])
                    elif len(res['data']):
                        res = pd.DataFrame(res['data'][0]['num'],
                                           columns=['Date', 'Usage'])
                        res.Date = res.astype(float) / 1000
                        res.Date = res.Date.apply(lambda x: datetime.fromtimestamp(x))
                        res.Usage = res.Usage.astype(int)

                results.append(res)
                del k,v

            if as_df:
                if len(results):
                    return (pd.concat(results)
                            .reset_index(drop=True)
                            .drop_duplicates(keep='first',
                                             inplace=False))
                else:
                    return (pd.DataFrame([],
                                         columns=['Date', 'Usage']))
            else:
                return results
        else:
            raise ValueError("Invalid date range.")
        if self._gis._portal.is_logged_in:
            url = "%sportals/%s/usage" % (self._portal.resturl, self._gis.properties.id)
        else:
            url = "%sportals/%s/usage" % (self._portal.resturl, "self")

        try:
            res = self._portal.con.post(url, params)
            if as_df:
                import pandas as pd
                if 'data' not in res or len(res['data']) == 0:
                    df = pd.DataFrame([],
                                      columns=['Date', 'Usage'])
                elif len(res['data']):
                    df = pd.DataFrame(res['data'][0]['num'],
                                      columns=['Date', 'Usage'])
                    df.Date = df.astype(float) / 1000
                    df.Date = df.Date.apply(lambda x : datetime.fromtimestamp(x))
                    df.Usage = df.Usage.astype(int)
                return df
            return res
        except:
            return None

    def get_data(self, try_json=True):
        """
        Retrieves the data associated with an item. Note that this call may
        return different results for different item types: some item types may
        even return *None*. See
        `this REST API page <https://developers.arcgis.com/rest/users-groups-and-items/working-with-users-groups-and-items.htm>`_
        for more information.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        try_json            Optional string. Default is True. For JSON/text files, if try_json
                            is True, the method tries to convert the data to a Python dictionary
                            (use json.dumps(data) to convert the dictionary to a string),
                            otherwise the data is returned as a string.
        ===============     ====================================================================


        :return:
           Dependent on the content type of the data.
           For non-JSON/text data, binary files are returned and the path to the downloaded file.
           For JSON/text files, a Python dictionary or a string.  All others will be a byte array,
           that can be converted to string using data.decode('utf-8'). Zero byte files will return None.
        """
        folder = None
        item_data = self._portal.get_item_data(self.itemid, try_json, folder)

        if item_data == '':
            return None
        elif type(item_data) == bytes:
            try:
                item_data_str = item_data.decode('utf-8')
                if item_data_str == '':
                    return None
                else:
                    return item_data
            except:
                return item_data
        else:
            return item_data

    def dependent_upon(self):
        """ Returns items, urls, etc that this item is dependent on. This capability (item dependencies)
        is not yet available on ArcGIS Online. Currently it is available only with an ArcGIS Enterprise."""
        return self._portal.get_item_dependencies(self.itemid)

    def dependent_to(self):
        """ Returns items, urls, etc that are dependent to this item. This capability (item dependencies)
        is not yet available on ArcGIS Online. Currently it is available only with an ArcGIS Enterprise."""
        return self._portal.get_item_dependents_to(self.itemid)

    _RELATIONSHIP_TYPES = frozenset(['Area2CustomPackage', 'Service2Layer', 'Map2Area',
                                     'Area2Package', 'Service2Route', 'Survey2Data',
                                     'Survey2Service', 'Service2Style', 'Style2Style',
                                     'Listed2Provisioned', 'Item2Report', 'Item2Attachment',
                                     'Map2AppConfig', 'Map2Service', 'WMA2Code',
                                     'Map2FeatureCollection', 'MobileApp2Code',
                                     'Service2Data', 'Service2Service', 'WorkforceMap2FeatureService',
                                     'TrackView2Map', 'SurveyAddIn2Data', 'Theme2Story',
                                     'WebStyle2DesktopStyle', 'Map2FeatureCollectionMobileApp2Code'])

    _RELATIONSHIP_DIRECTIONS = frozenset(['forward', 'reverse'])

    def related_items(self, rel_type, direction="forward"):
        """
        Retrieves the items related to this item. Relationships can be added and deleted using
        item.add_relationship() and item.delete_relationship(), respectively.

        .. note::
            With WebMaps items, relationships are only available on local enterprises.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_. in the REST API help
                            for more information on this parameter.
        ---------------     --------------------------------------------------------------------
        direction           Required string. One of ['forward', 'reverse']
        ===============     ====================================================================


        :return:
           The list of related items.
        """

        if rel_type not in self._RELATIONSHIP_TYPES:
            raise Error('Unsupported relationship type: ' + rel_type)
        if not direction in self._RELATIONSHIP_DIRECTIONS:
            raise Error('Unsupported direction: ' + direction)

        related_items = []

        postdata = { 'f' : 'json' }
        postdata['relationshipType'] = rel_type
        postdata['direction'] = direction
        resp = self._portal.con.post('content/items/' + self.itemid + '/relatedItems', postdata)
        for related_item in resp['relatedItems']:
            related_items.append(Item(self._gis, related_item['id'], related_item))
        return related_items

    def add_relationship(self, rel_item, rel_type):
        """ Adds a relationship from this item to rel_item.

        .. note::
            Relationships are not tied to an item. They are directional links from an origin item
            to a destination item and have a type. The type defines the valid origin and destination
            item types as well as some rules. See Relationship types in REST API help for more information.
            Users don't have to own the items they relate unless so defined by the rules of the relationship
            type.

            Users can only delete relationships they create.

            Relationships are deleted automatically if one of the two items is deleted.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_item            Required Item object corresponding to the related item.
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_. in the REST API help
                            for more information on this parameter.
        ===============     ====================================================================


        :return:
           Returns True if the relationship was added, False if the add failed.
        """
        if not rel_type in self._RELATIONSHIP_TYPES:
            raise Error('Unsupported relationship type: ' + rel_type)

        postdata = { 'f' : 'json' }
        postdata['originItemId'] = self.itemid
        postdata['destinationItemId'] = rel_item.itemid
        postdata['relationshipType'] = rel_type
        path = 'content/users/{uid}/addRelationship'.format(uid=self._user_id)

        resp = self._portal.con.post(path, postdata)
        if resp:
            return resp.get('success')

    def delete_relationship(self, rel_item, rel_type):
        """
        Deletes a relationship between this item and the rel_item.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_item            Required Item object corresponding to the related item.
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_. in the REST API help
                            for more information on this parameter.
        ===============     ====================================================================


        :return:
           Returns True if the relationship was deleted, False if the deletion failed.
        """
        if not rel_type in self._RELATIONSHIP_TYPES:
            raise Error('Unsupported relationship type: ' + rel_type)
        postdata = { 'f' : 'json' }
        postdata['originItemId'] =  self.itemid
        postdata['destinationItemId'] = rel_item.itemid
        postdata['relationshipType'] = rel_type
        path = 'content/users/{uid}/deleteRelationship'.format(uid=self._user_id)

        resp = self._portal.con.post(path, postdata)
        if resp:
            return resp.get('success')

    def publish(self, publish_parameters=None, address_fields=None, output_type=None, overwrite=False,
                file_type=None, build_initial_cache=False, item_id=None):
        """
        Publishes a hosted service based on an existing source item (this item).
        Publishers can create feature, tiled map, vector tile and scene services.

        Feature services can be created using input files of type csv, shapefile, serviceDefinition, featureCollection, and fileGeodatabase.
        CSV files that contain location fields (i.e. address fields or XY fields) are spatially enabled during the process of publishing.
        Shapefiles and file geodatabases should be packaged as *.zip files.

        Tiled map services can be created from service definition (*.sd) files, tile packages, and existing feature services.

        Vector tile services can be created from vector tile package (*.vtpk) files.

        Scene services can be created from scene layer package (*.spk, *.slpk) files.

        Service definitions are authored in ArcGIS for Desktop and contain both the cartographic definition for a map
        as well as its packaged data together with the definition of the geo-service to be created.

        .. note::
            ArcGIS does not permit overwriting if you published multiple hosted feature layers from the same data item.


        ===================    ===============================================================
        **Argument**           **Description**
        -------------------    ---------------------------------------------------------------
        publish_parameters     Optional dictionary. containing publish instructions and customizations.
                               Cannot be combined with overwrite.  See http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Publish_Item/02r300000080000000/ for details.
        -------------------    ---------------------------------------------------------------
        address_fields         Optional dictionary. containing mapping of df columns to address fields,
                               eg: { "CountryCode" : "Country"} or { "Address" : "Address" }
        -------------------    ---------------------------------------------------------------
        output_type            Optional string.  Only used when a feature service is published as a tile service.
                               eg: output_type='Tiles'
        -------------------    ---------------------------------------------------------------
        overwrite              Optional boolean.   If True, the hosted feature service is overwritten.
                               Only available in ArcGIS Online and Portal for ArcGIS 10.5 or later.
        -------------------    ---------------------------------------------------------------
        file_type              Optional string.  Some formats are not automatically detected, when this occurs, the
                               file_type can be specified: serviceDefinition,shapefile,csv,
                               tilePackage, featureService, featureCollection, fileGeodatabase,
                               geojson, scenepackage, vectortilepackage, imageCollection,
                               mapService, and sqliteGeodatabase are valid entries. This is an
                               optional parameter.
        -------------------    ---------------------------------------------------------------
        build_initial_cache    Optional boolean.  The boolean value (default False), if true
                               and applicable for the file_type, the value will built cache
                               for the service.
        -------------------    ---------------------------------------------------------------
        item_id                Optionl String. **Available in Enterprise/AGOL 10.8.1+**.  A string
                               of 32 character UID without any special characters.

                               If the `item_id` is already being used, an error will be raised
                               during the `publish` process.

                               Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        ===================    ===============================================================


        :return:
            An arcgis.gis.Item object corresponding to the published web layer.

        For publish_parameters, see http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Publish_Item/02r300000080000000/
        """

        import time
        params = {
            "f" : "json"
        }

        buildInitialCache = build_initial_cache
        if file_type is None:
            if self['type'] == "GeoPackage":
                fileType = "gpkg"
            elif self['type'] == 'Compact Tile Package':
                fileType = 'compactTilePackage'
            elif self['type'] == 'Service Definition':
                fileType = 'serviceDefinition'
            elif self['type'] == 'Microsoft Excel':
                fileType = 'excel'
            elif self['type'] == 'Feature Collection':
                fileType = 'featureCollection'
            elif self['type'] == 'CSV':
                fileType = 'CSV'
            elif self['type'] == 'Shapefile':
                fileType = 'shapefile'
            elif self['type'] == 'File Geodatabase':
                fileType = 'fileGeodatabase'
            elif self['type'] == 'Vector Tile Package':
                fileType = 'vectortilepackage'
            elif self['type'] == 'Scene Package':
                fileType = 'scenePackage'
            elif self['type'] == 'Tile Package':
                fileType = 'tilePackage'
            elif self['type'] == 'SQLite Geodatabase':
                fileType = 'sqliteGeodatabase'
            elif self['type'] in ['GeoJson', 'geojson']:
                fileType = 'geojson'
            elif self['type'] == 'Feature Service' and \
                 'Spatiotemporal' in self['typeKeywords']:
                fileType = 'featureService'
            else:
                raise ValueError("A file_type must be provide, data format not recognized")
        else:
            fileType = file_type
        try:
            folder = self.ownerFolder
        except:
            folder = None

        if publish_parameters is None:
            if fileType == 'shapefile' and not overwrite:
                publish_parameters =  {"hasStaticData":True, "name": os.path.splitext(self['name'])[0].replace(" ", "_"),
                                       "maxRecordCount":2000, "layerInfo":{"capabilities":"Query"} }

            elif fileType in ['CSV', 'excel'] and not overwrite:
                path = "content/features/analyze"

                postdata = {
                    "f": "pjson",
                    "itemid" : self.itemid,
                    "filetype" : "csv",

                    "analyzeParameters" : {
                        "enableGlobalGeocoding": "true",
                        "sourceLocale":"en-us",
                        #"locationType":"address",
                        "sourceCountry":"",
                        "sourceCountryHint":""
                    }
                }

                if address_fields is not None:
                    postdata['analyzeParameters']['locationType'] = 'address'

                res = self._portal.con.post(path, postdata)
                publish_parameters =  res['publishParameters']
                if address_fields is not None:
                    publish_parameters.update({"addressFields":address_fields})

                # use csv title for service name, after replacing non-alphanumeric characters with _
                service_name = re.sub(r'[\W_]+', '_', self['title'])
                publish_parameters.update({"name": service_name})

            elif fileType in ['CSV', 'shapefile', 'fileGeodatabase'] and overwrite: #need to construct full publishParameters
                #find items with relationship 'Service2Data' in reverse direction - all feature services published using this data item
                related_items = self.related_items('Service2Data', 'reverse')

                return_item_list = []
                if len (related_items) == 1: #simple 1:1 relationship between data and service items
                    r_item = related_items[0]
                    #construct a FLC manager
                    from arcgis.features import FeatureLayerCollection
                    flc = FeatureLayerCollection.fromitem(r_item)
                    flc_mgr = flc.manager

                    # get the publish parameters from FLC manager
                    publish_parameters, update_params = flc_mgr._gen_overwrite_publishParameters(r_item)
                    if update_params:  # when overwriting file on portals, need to update source item metadata
                        self.update(item_properties=update_params)

                    # if source file type is CSV or Excel, blend publish parameters with analysis results
                    if fileType == 'CSV':
                        publish_parameters_orig = publish_parameters
                        path = "content/features/analyze"

                        postdata = {
                            "f": "pjson",
                            "itemid": self.itemid,
                            "filetype": "csv",

                            "analyzeParameters": {
                                "enableGlobalGeocoding": "true",
                                "sourceLocale": "en-us",
                                # "locationType":"address",
                                "sourceCountry": "",
                                "sourceCountryHint": ""
                            }
                        }

                        if address_fields is not None:
                            postdata['analyzeParameters']['locationType'] = 'address'

                        res = self._portal.con.post(path, postdata)
                        publish_parameters = res['publishParameters']
                        publish_parameters.update(publish_parameters_orig)

                elif len(related_items) == 0:
                    # the CSV item was never published. Hence overwrite should work like first time publishing - analyze csv
                    path = "content/features/analyze"
                    postdata = {
                        "f": "pjson",
                        "itemid" : self.itemid,
                        "filetype" : "csv",

                        "analyzeParameters" : {
                            "enableGlobalGeocoding": "true",
                            "sourceLocale":"en-us",
                            #"locationType":"address",
                            "sourceCountry":"",
                            "sourceCountryHint":""
                        }
                    }

                    if address_fields is not None:
                        postdata['analyzeParameters']['locationType'] = 'address'

                    res = self._portal.con.post(path, postdata)
                    publish_parameters =  res['publishParameters']
                    if address_fields is not None:
                        publish_parameters.update({"addressFields":address_fields})

                    # use csv title for service name, after replacing non-alphanumeric characters with _
                    service_name = re.sub(r'[\W_]+', '_', self['title'])
                    publish_parameters.update({"name": service_name})

                elif len(related_items) > 1:
                    # length greater than 1, then 1:many relationship
                    raise RuntimeError("User cant overwrite this service, using this data, as this data is already referring to another service.")

            elif fileType == 'vectortilepackage':
                name = re.sub(r'[\W_]+', '_', self['title'])
                publish_parameters = {'name': name, 'maxRecordCount':2000}
                output_type = 'VectorTiles'
                buildInitialCache = True

            elif fileType == 'scenePackage':
                name = re.sub(r'[\W_]+', '_', self['title'])
                buildInitialCache = True
                publish_parameters = {'name': name, 'maxRecordCount':2000}
                output_type = 'sceneService'
            elif fileType == 'featureService':
                name = re.sub(r'[\W_]+', '_', self['title'])
                c = self._gis.content
                is_avail = c.is_service_name_available(name, 'featureService')
                i = 1
                while is_avail == False:
                    sname = name + "_%s" % i
                    is_avail = c.is_service_name_available(sname, 'featureService')
                    if is_avail:
                        name = sname
                        break
                    i += 1
                ms = self.layers[0].container.manager
                publish_parameters = ms._generate_mapservice_definition()
                output_type = "bdsMapService"
                buildInitialCache = True
                if 'serviceName' in publish_parameters:
                    publish_parameters['serviceName'] = name

            elif fileType == 'tilePackage':
                name = re.sub(r'[\W_]+', '_', self['title'])
                publish_parameters = {'name': name, 'maxRecordCount':2000}
                buildInitialCache = True
            elif fileType == 'sqliteGeodatabase':
                name = re.sub(r'[\W_]+', '_', self['title'])
                publish_parameters = {"name":name,
                                      'maxRecordCount':2000,
                                      "capabilities":"Query, Sync"}
            else: #sd files
                name = re.sub(r'[\W_]+', '_', self['title'])
                publish_parameters =  {"hasStaticData":True, "name": name, "maxRecordCount":2000, "layerInfo":{"capabilities":"Query"} }

        elif fileType == 'CSV' or \
             fileType == 'excel': # merge users passed-in publish parameters with analyze results
            publish_parameters_orig = publish_parameters
            path = "content/features/analyze"

            postdata = {
                "f": "pjson",
                "itemid" : self.itemid,
                "filetype" : "csv",

                "analyzeParameters" : {
                    "enableGlobalGeocoding": "true",
                    "sourceLocale":"en-us",
                    #"locationType":"address",
                    "sourceCountry":"",
                    "sourceCountryHint":""
                }
            }

            if address_fields is not None:
                postdata['analyzeParameters']['locationType'] = 'address'

            res = self._portal.con.post(path, postdata)
            publish_parameters =  res['publishParameters']
            publish_parameters.update(publish_parameters_orig)
        #params['overwrite'] = json.dumps(overwrite)
        ret = self._portal.publish_item(self.itemid, None,
                                        None, fileType,
                                        publish_parameters, output_type,
                                        overwrite, self.owner,
                                        folder, buildInitialCache, item_id=item_id)

        #Check publishing job status

        if buildInitialCache and \
           self._gis._portal.is_arcgisonline and \
           fileType.lower() in  ['tilepackage', 'compacttilepackage']:
            from ..mapping._types import MapImageLayer
            from ..raster._layer import ImageryLayer
            if len(ret) > 0 and \
               'success' in ret[0] and \
               ret[0]['success'] == False:
                raise Exception(ret[0]['error'])
            ms_url = self._gis.content.get(ret[0]['serviceItemId']).url
            if ms_url.lower().find("mapserver") > -1:
                ms = MapImageLayer(url=ms_url, gis=self._gis)
                manager = ms.manager
            elif ms_url.lower().find("imageserver") > -1:
                ms = ImageryLayer(url=ms_url, gis=self._gis)
                manager = ms.cache_manager
                if not self._gis._portal.is_arcgisonline:
                    return Item(self._gis, ret[0]['serviceItemId'])
            serviceitem_id = ret[0]['serviceItemId']
            try:
                # first edit the tile service to set min, max scales
                if not ms.properties.minScale:
                    min_scale = ms.properties.tileInfo.lods[0]['scale']
                    max_scale = ms.properties.tileInfo.lods[-1]['scale']
                else:
                    min_scale = ms.properties.minScale
                    max_scale = ms.properties.maxScale

                edit_result = manager.edit_tile_service(min_scale=min_scale, max_scale=max_scale)

                # Get LoD from Map Image Layer
                full_extent = dict(ms.properties.fullExtent)
                lod_dict = ms.properties.tileInfo['lods']
                lod = [current_lod['level'] for current_lod in lod_dict
                       if (min_scale <= current_lod['scale'] <= max_scale)]
                ret = manager.update_tiles(levels=lod, extent=full_extent)
            except Exception as tiles_ex:
                raise Exception('Error unpacking tiles :' + str(tiles_ex))
        elif not buildInitialCache and output_type is not None and output_type.lower() in ['sceneservice']:
            return Item(self._gis, ret[0]['serviceItemId'])
        elif not buildInitialCache and ret[0]['type'].lower() == 'image service':
            return Item(self._gis, ret[0]['serviceItemId'])
        else:
            serviceitem_id = self._check_publish_status(ret, folder)
        return Item(self._gis, serviceitem_id)

    def move(self, folder, owner=None):
        """
        Moves this item to the folder with the given name.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        folder            Required string. The name of the folder to move the item to.
                          Use '/' for the root folder. For other folders, pass in the
                          folder name as a string, or a dictionary containing the folder ID,
                          such as the dictionary obtained from the folders property.
        ----------------  ---------------------------------------------------------------
        owner             Optional string or Owner object.  The name of the user to
                          move to.
        ================  ===============================================================

        :return:
            A json object like the following:
            {"success": true | false,
               "itemId": "<item id>",
               "owner": "<owner username>",
               "folder": "<folder id>"}

        """
        owner_name = self._user_id
        folder_id = None
        if folder is not None:
            if isinstance(folder, str):
                if folder == '/':
                    folder_id = '/'
                else:
                    folder_id = self._portal.get_folder_id(owner_name, folder)
            elif isinstance(folder, dict):
                folder_id = folder['id']
            else:
                print("folder should be folder name as a string, or dict with id")

        if folder_id is not None:
            ret = self._portal.move_item(self.itemid, owner_name, self.ownerFolder, folder_id)
            self._hydrate()
            return ret
        else:
            print('Folder not found for given owner')
            return None

    #----------------------------------------------------------------------
    def create_tile_service(self,
                            title,
                            min_scale,
                             max_scale,
                             cache_info=None,
                             build_cache=False):
        """
        Allows publishers and administrators to publish hosted feature
        layers and hosted feature layer views as a tile service.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        title             Required string. The name of the new service.
                          Example: "SeasideHeightsNJTiles"
        ----------------  ---------------------------------------------------------------
        min_scale         Required float. The smallest scale at which to view data.
                          Example: 577790.0
        ----------------  ---------------------------------------------------------------
        max_scale         Required float. The largest scale at which to view data.
                          Example: 80000.0
        ----------------  ---------------------------------------------------------------
        cache_info        Optional dictionary. If not none, administrator provides the
                          tile cache info for the service. The default is the AGOL scheme.
        ----------------  ---------------------------------------------------------------
        build_cache       Optional boolean. Default is False; if True, the cache will be
                          built at publishing time.  This will increase the time it takes
                          to publish the service.
        ================  ===============================================================

        :return:
           The item if successfully added, None if unsuccessful.

        """

        if self.type.lower() == 'Feature Service'.lower():
            p = self.layers[0].container
            if cache_info is None:
                cache_info = {'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
                              'rows': 256, 'preciseDpi': 96, 'cols': 256, 'dpi': 96,
                              'origin': {'y': 20037508.342787, 'x': -20037508.342787},
                              'lods': [{'level': 0, 'scale': 591657527.591555, 'resolution': 156543.033928},
                                       {'level': 1, 'scale': 295828763.795777, 'resolution': 78271.5169639999},
                                       {'level': 2, 'scale': 147914381.897889, 'resolution': 39135.7584820001},
                                       {'level': 3, 'scale': 73957190.948944, 'resolution': 19567.8792409999},
                                       {'level': 4, 'scale': 36978595.474472, 'resolution': 9783.93962049996},
                                       {'level': 5, 'scale': 18489297.737236, 'resolution': 4891.96981024998},
                                       {'level': 6, 'scale': 9244648.868618, 'resolution': 2445.98490512499},
                                       {'level': 7, 'scale': 4622324.434309, 'resolution': 1222.99245256249},
                                       {'level': 8, 'scale': 2311162.217155, 'resolution': 611.49622628138},
                                       {'level': 9, 'scale': 1155581.108577, 'resolution': 305.748113140558},
                                       {'level': 10, 'scale': 577790.554289, 'resolution': 152.874056570411},
                                       {'level': 11, 'scale': 288895.277144, 'resolution': 76.4370282850732},
                                       {'level': 12, 'scale': 144447.638572, 'resolution': 38.2185141425366},
                                       {'level': 13, 'scale': 72223.819286, 'resolution': 19.1092570712683},
                                       {'level': 14, 'scale': 36111.909643, 'resolution': 9.55462853563415},
                                       {'level': 15, 'scale': 18055.954822, 'resolution': 4.77731426794937},
                                       {'level': 16, 'scale': 9027.977411, 'resolution': 2.38865713397468},
                                       {'level': 17, 'scale': 4513.988705, 'resolution': 1.19432856685505},
                                       {'level': 18, 'scale': 2256.994353, 'resolution': 0.597164283559817},
                                       {'level': 19, 'scale': 1128.497176, 'resolution': 0.298582141647617},
                                       {'level': 20, 'scale': 564.248588, 'resolution': 0.14929107082380833},
                                       {'level': 21, 'scale': 282.124294, 'resolution': 0.07464553541190416},
                                       {'level': 22, 'scale': 141.062147, 'resolution': 0.03732276770595208}]
                              }
            pp = {"minScale":min_scale,"maxScale":max_scale,"name":title,
                  "tilingSchema":{"tileCacheInfo": cache_info,
                                  "tileImageInfo":{"format":"PNG32","compressionQuality":0,"antialiasing":True},
                                  "cacheStorageInfo":{"storageFormat":"esriMapCacheStorageModeExploded",
                                      "packetSize":128}},"cacheOnDemand":True,
                  "cacheOnDemandMinScale":144448,
                  "capabilities":"Map,ChangeTracking"}
            params = {
                "f" : "json",
                "outputType" : "tiles",
                "buildInitialCache" : build_cache,
                "itemid" : self.itemid,
                "filetype" : "featureService",
                "publishParameters" : json.dumps(pp)
            }
            url = "%s/content/users/%s/publish" % (self._portal.resturl,
                                                   self._user_id)
            res = self._gis._con.post(url, params)
            serviceitem_id = self._check_publish_status(res['services'], folder=None)
            if self._gis._portal.is_arcgisonline:
                from ..mapping._types import MapImageLayer
                ms_url = self._gis.content.get(serviceitem_id).url
                ms = MapImageLayer(url=ms_url, gis=self._gis)
                extent = ",".join([str(ms.properties['fullExtent']['xmin']),
                                   str(ms.properties['fullExtent']['ymin']),
                                   str(ms.properties['fullExtent']['xmax']),
                                   str(ms.properties['fullExtent']['ymax'])])
                lods = []
                for lod in cache_info['lods']:
                    if lod['scale'] <= min_scale and \
                       lod['scale'] >= max_scale:
                        lods.append(str(lod['level']))
                ms.manager.update_tiles(levels=",".join(lods), extent=extent)
            return self._gis.content.get(serviceitem_id)
        else:
            raise ValueError("Input must of type FeatureService")
        return

    def protect(self, enable=True):
        """
        Enables or disables delete protection on this item.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            Optional boolean. Default is True which enables delete
                          protection, False to disable delete protection.
        ================  ===============================================================

        :return:
            A json object like the following:
            {"success": true | false}

        """

        try:
            folder = self.ownerFolder
        except:
            folder = None
        res = self._portal.protect_item(self.itemid, self._user_id, folder, enable)
        self._hydrated = False
        self._hydrate()
        return res

    def _check_publish_status(self, ret, folder):
        """ Internal method to check the status of a publishing job.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        ret                 Required dictionary. Represents the result of a publish REST call.
                            This dict should contain the `serviceItemId` and `jobId` of the publishing job.
        ---------------     --------------------------------------------------------------------
        folder              Required string. Obtained from self.ownerFolder
        ===============     ====================================================================


        :return:
           The status.
        """

        import time
        try:
            serviceitem_id = ret[0]['serviceItemId']
        except KeyError as ke:
            raise RuntimeError(ret[0]['error']['message'])

        if 'jobId' in ret[0]:
            job_id = ret[0]['jobId']
            path = 'content/users/' + self.owner
            if folder is not None:
                path = path + '/' + folder + '/'

            path = path + '/items/' + serviceitem_id + '/status'
            params = {
                "f" : "json",
                "jobid" : job_id
            }
            job_response = self._portal.con.post(path, params)

            # Query and report the Analysis job status.
            #
            num_messages = 0
            #print(str(job_response))
            if "status" in job_response:
                while not job_response.get("status") == "completed":
                    time.sleep(5)

                    job_response = self._portal.con.post(path, params)

                    #print(str(job_response))
                    if job_response.get("status") in ("esriJobFailed","failed"):
                        raise Exception("Job failed.")
                    elif job_response.get("status") == "esriJobCancelled":
                        raise Exception("Job cancelled.")
                    elif job_response.get("status") == "esriJobTimedOut":
                        raise Exception("Job timed out.")

            else:
                raise Exception("No job results.")
        else:
            raise Exception("No job id")

        return serviceitem_id
    #----------------------------------------------------------------------
    @property
    def comments(self):
        """
        Gets a list of comments for a given item.
        """
        from .._impl.comments import Comment
        cs = []
        start = 1
        num = 100
        nextStart = 0
        url = "%s/sharing/rest/content/items/%s/comments" % (self._portal.url, self.id)
        while nextStart != -1:
            params = {
                "f" : "json",
                "start" : start,
                "num" : num
            }
            res = self._portal.con.post(url, params)
            for c in res['comments']:
                cs.append(Comment(url="%s/%s" % (url, c['id']),
                                  item=self, initialize=True))
            start += num
            nextStart = res['nextStart']
        return cs
    #----------------------------------------------------------------------
    def add_comment(self, comment):
        """
        Adds a comment to an item. Available only to authenticated users
        who have access to the item.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        comment             Required string. Text to be added as a comment to a specific item.
        ===============     ====================================================================


        :return:
           Comment ID if successful, None on failure.
        """
        params = {
            "f" : "json",
            "comment" : comment
        }
        url = "%s/sharing/rest/content/items/%s/addComment" % (self._portal.url, self.id)
        res = self._portal.con.post(url, params)
        if 'commentId' in res:
            return res['commentId']
        return None
    #----------------------------------------------------------------------
    @property
    def rating(self):
        """
        Gets or sets the rating given by the current user to the item.
        """
        url = "%s/sharing/rest/content/items/%s/rating" % (self._portal.url, self.id)
        params = {"f" : "json"}
        res = self._portal.con.get(url, params)
        if 'rating' in res:
            return res['rating']
        return None
    #----------------------------------------------------------------------
    @rating.setter
    def rating(self, value):
        """
        Adds a rating to an item to which you have access. Only one rating
        can be given to an item per user. If this call is made on a
        currently rated item, the new rating will overwrite the existing
        rating. A user cannot rate their own item. Available only to
        authenticated users.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required float. The rating to be applied for the item. The value
                            must be a floating point number between 1.0 and 5.0.
        ===============     ====================================================================


        """
        url = "%s/sharing/rest/content/items/%s/addRating" % (self._portal.url,
                                                              self.id)
        params = {"f" : "json",
                  'rating' : float(value)}
        self._portal.con.post(url, params)
    #----------------------------------------------------------------------
    def delete_rating(self):
        """
        Removes the rating the calling user added for the specified item.
        """
        url = "%s/sharing/rest/content/items/%s/deleteRating" % (self._portal.url,
                                                                 self.id)
        params = {"f" : "json"}
        res = self._portal.con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    @property
    def proxies(self):
        """
        Gets the ArcGIS Online hosted proxy services set on a registered app
        item with the Registered App type keyword. This resource is only
        available to the item owner and the organization administrator.
        """
        url = "%s/sharing/rest/content/users/%s/items/%s/proxies" % (self._portal.url,
                                                                     self._user_id,
                                                        self.id)
        params = {"f" : "json"}
        ps = []
        try:
            res = self._portal.con.get(url, params)
            if 'appProxies' in res:
                for p in res['appProxies']:
                    ps.append(p)
        except:
            return []
        return ps
    #----------------------------------------------------------------------
    def _create_proxy(self,
                      url:str=None,
                     hit_interval:int=None,
                     interval_length:int=60,
                     proxy_params:dict=None) -> dict:
        """
        A service proxy creates a new endpoint for a service that is
        specific to your application. Only allowed domains that you
        specify will be able to access the service.

        ===================    ===============================================================
        **Argument**           **Description**
        -------------------    ---------------------------------------------------------------
        url                    Optional string. Represents the hosted service URLs to proxy.
        -------------------    ---------------------------------------------------------------
        hit_interval           Optional Integer. Number of times a service can be used in the
                               given interval_length.
        -------------------    ---------------------------------------------------------------
        interval_length        Optional Integer. The time gap for the total hit_interval that
                               a service can be used.  The number is in seconds.
        -------------------    ---------------------------------------------------------------
        proxy_params           Optional dict. Dictionary that provides referrer checks when
                               accessing the premium content and optionally rate limiting if
                               it is not set for each service in proxies.
                                Example:

                                {
                                   "referrers": ["http://foo.example.com", "http://bar.example.com"],
                                   "hitsPerInterval": 1000,
                                   "intervalSeconds": 60
                                }
        ===================    ===============================================================


        :return: Item

        """
        url = "%s/sharing/rest/content/users/%s/items/%s/createProxies" % (self._portal.url,
                                                                           self._user_id,
                                                                           self.id)
        params = {
            'f' : 'json',
            'proxies' : [],
            'serviceProxyParams': {}
        }
        if url and hit_interval and interval_length:
            params['proxies'].append({
                "sourceUrl": url,
                "hitPerInterval" : hit_interval,
                "intervalSeconds" : interval_length
            })
        if proxy_params is not None:
            params['serviceProxyParams'] = proxy_params
        res = self._portal.con.post(url, params)
        return Item(gis=self._gis, itemid=res['id'])
    #----------------------------------------------------------------------
    def _delete_proxy(self, proxy_id:str) -> dict:
        """
        The delete proxies removes a hosted proxies set on an item. The
        operation can only be made by the item owner or the organization
        administrator.

        ===================    ===============================================================
        **Argument**           **Description**
        -------------------    ---------------------------------------------------------------
        proxy_id               Required string. This is a comma seperated list of proxy ids.
        ===================    ===============================================================


        :return: dict

        """
        params = {'f': 'json',
                  'proxies': proxy_id}
        url = "%s/sharing/rest/content/users/%s/items/%s/deleteProxies" % (self._portal.url,
                                                                           self._user_id,
                                                                           self.id)
        return self._portal.con.post(url, params)
    #----------------------------------------------------------------------
    def copy(self, title=None, tags=None, snippet=None, description=None, layers=None):
        """
        Copy allows for the creation of an item that is derived from the current item.

        For layers, `copy` will create a new item that uses the URL as a reference.
        For non-layer based items, these will be copied and the exact same data will be
        provided.


        If title, tags, snippet of description is not provided the values from `item` will be used.

        Copy use example:

            + Vector tile service sprite customization
            + Limiting feature service exposure
            + Sharing content by reference with groups
            + Creating backup items.

        **Usage Example**

        >>> item.copy()
        <Item title:"gisslideshow - Copy 94452b" type:Microsoft Powerpoint owner:geoguy>
        >>> item.copy(title="GIS_Tutorial")
        <Item title:"GIS_Tutorial" type:Microsoft Powerpoint owner:geoguy>
        >>> item.copy()
        <Item title:"NZTiles - Copy 021a06" type:Vector Tile Layer owner:geoguy>


        =======================    =============================================================
        **Argument**               **Description**
        -----------------------    -------------------------------------------------------------
        title                      Optional string. The name of the new item.
        -----------------------    -------------------------------------------------------------
        tags                       Optional list of string. Descriptive words that help in the
                                   searching and locating of the published information.
        -----------------------    -------------------------------------------------------------
        snippet                    Optional string. A brief summary of the information being
                                   published.
        -----------------------    -------------------------------------------------------------
        description                Optional string. A long description of the Item being
                                   published.
        -----------------------    -------------------------------------------------------------
        layers                     Optional list of integers.  If you have a layer with multiple
                                   and you only want specific layers, an index can be provided
                                   those layers.  If nothing is provided, all layers will be
                                   visible.

                                   Example: layers=[0,3]
                                   Example 2: layers=[9]
        =======================    =============================================================

        :returns: Item

        """
        TEXT_BASED_ITEM_TYPES = ['Web Map', 'Web Scene','360 VR Experience',
                                 'Operation View', 'Workforce Project',
                                 'Insights Model', 'Insights Page', 'Dashboard',
                                 'Feature Collection', 'Insights Workbook',
                                 'Feature Collection Template', 'Hub Initiative',
                                 'Hub Site Application', 'Hub Page',
                                 'Web Mapping Application', 'Mobile Application',
                                 'Symbol Set', 'Color Set', 'Content Category Set',
                                 'Windows Viewer Configuration']
        FILE_BASED_ITEM_TYPES = ['Notebook',
                                 'CityEngine Web Scene','Pro Map', 'Map Area', 'KML Collection',
                                 'Code Attachment', 'Operations Dashboard Add In',
                                 'Native Application', 'Native Application Template', 'KML',
                                 'Native Application Installer', 'Form', 'AppBuilder Widget Package',
                                 'File Geodatabase','CSV', 'Image',  'Locator Package',
                                 'Map Document', 'Shapefile', 'Microsoft Word', 'PDF',
                                 'CAD Drawing', 'Service Definition', 'Image',
                                 'Visio Document', 'iWork Keynote', 'iWork Pages',
                                 'iWork Numbers', 'Report Template', 'Statistical Data Collection',
                                 'SQLite Geodatabase', 'Mobile Basemap Package', 'Project Package',
                                 'Task File', 'ArcPad Package', 'Explorer Map', 'Globe Document',
                                 'Scene Document', 'Published Map', 'Map Template', 'Windows Mobile Package',
                                 'Layout', 'Project Template', 'Layer', 'Explorer Package',
                                 'Image Collection', 'Desktop Style', 'Geoprocessing Sample',
                                 'Locator Package', 'Rule Package', 'Raster function template',
                                 'ArcGIS Pro Configuration', 'Workflow Manager Package',
                                 'Desktop Application', 'Desktop Application Template',
                                 'Code Sample', 'Desktop Add In', 'Explorer Add In', 'ArcGIS Pro Add In',
                                 'Microsoft Powerpoint', 'Microsoft Excel', 'Layer Package',
                                 'Mobile Map Package', 'Geoprocessing Package', 'Scene Package',
                                 'Tile Package', 'Vector Tile Package']
        SERVICE_BASED_ITEM_TYPES = ["Vector Tile Service","Scene Service", 'WMS', 'WFS', 'WMTS',
                                    'Geodata Service', 'Globe Service', 'Scene Service',
                                    'Relational Database Connection',
                                    'AppBuilder Extension', 'Document Link',
                                    'Geometry Service', 'Geocoding Service',
                                    'Network Analysis Service', 'Geoprocessing Service',
                                    'Workflow Manager Service', "Image Service",
                                    "Map Service", "Feature Service"]
        item = self
        from datetime import timezone
        from uuid import uuid4
        now = datetime.now(timezone.utc)
        if title is None:
            title = item.title + " - Copy %s" % uuid4().hex[:6]
        if tags is None:
            tags = item.tags
        if snippet is None:
            snippet = item.snippet
        if description is None:
            description = item.description

        if item.type in SERVICE_BASED_ITEM_TYPES or \
           item.type == 'KML' and item.url is not None:
            params = {
                'f' : 'json',
                'item' : item.title.replace(" ", "_") + "-_copy_%s" % int(now.timestamp() * 1000),
                'type' :item.type,
                'url' : item.url,
                'typeKeywords' : ",".join(item.typeKeywords)
            }

            params['title'] = title
            params['tags'] = ",".join(tags)
            params['snippet'] = snippet
            params['description'] = description
            if not layers is None:
                text = {
                    "layers": []
                }
                lyrs = item.layers
                for idx, lyr in enumerate(lyrs):
                    if idx in layers:
                        text['layers'].append({
                            "layerDefinition": {
                                "defaultVisibility": True
                                },
                            "id": idx
                        })
                params['text'] = text
            url = "%s/content/users/%s/addItem" % (self._gis._portal.resturl,
                                                   self._user_id)
            res = self._gis._con.post(url,
                                      params)
            if 'id' in res:
                itemid = res['id']
            else:
                return None

            if itemid is not None:
                return Item(self._gis, itemid)
            else:
                return None
        elif item.type.lower() == 'notebook':
            with tempfile.TemporaryDirectory() as d:
                import shutil

                fp = item.download(save_path=d)
                if fp.find("..") > -1:
                    shutil.copy(fp, fp.replace("..", "."))
                    fp = fp.replace("..", ".")
                sfp = os.path.split(fp)
                fname, ext = os.path.splitext(sfp[1])
                ext = ext.replace(".", "")
                nfp = os.path.join(sfp[0],
                                   "%s_%s.%s" % (fname, uuid4().hex[:5], ext))
                os.rename(fp, nfp)
                ip = {
                    'type' : item.type,
                    'tags' : ",".join(item.tags),
                    'snippet' : snippet,
                    'description' : description,
                    'typeKeywords' : ",".join(item.typeKeywords),
                    'title' : title
                }

                item = self._gis.content.add(item_properties=ip, data=nfp)
                return item
        elif item.type in FILE_BASED_ITEM_TYPES:
            fp = self.get_data()
            sfp = os.path.split(fp)
            fname, ext = os.path.splitext(sfp[1])
            nfp = os.path.join(sfp[0],
                               "%s_%s.%s" % (fname, uuid4().hex[:5], ext))
            os.rename(fp, nfp)
            ip = {
                'type' : item.type,
                'tags' : ",".join(item.tags),
                'snippet' : snippet,
                'description' : description,
                'typeKeywords' : ",".join(item.typeKeywords),
                'title' : title
            }
            item = self._gis.content.add(item_properties=ip, data=nfp)
            os.remove(nfp)
            return item
        elif item.type in TEXT_BASED_ITEM_TYPES:
            data = self.get_data()
            ip = {
                'type' : item.type,
                'tags' : ",".join(item.tags),
                'typeKeywords' : ",".join(item.typeKeywords),
                'snippet' : snippet,
                'description' : description,
                'text' : data,
                'title' : title
            }
            if item.type == "Notebook":
                ip['properties'] = item.properties
            new_item = self._gis.content.add(item_properties=ip)
            if item.url and item.url.find(item.id) > -1:
                new_item.update({"url" : item.url.replace(item.id, new_item.id)})
            return new_item

        else:
            raise ValueError("Item of type: %s is not supported by copy" % (item.type))
        return
    #----------------------------------------------------------------------
    @property
    def dependencies(self):
        """returns a class to management Item dependencies"""
        if self._depend is None:
            self._depend = ItemDependency(self)
        return self._depend
    #----------------------------------------------------------------------
    def register(self, app_type, redirect_uris=None, http_referers=None, privileges=None):
        """

        The register method registers an app item with the enterprise. App
        registration results in an APPID and APPSECRET (also known as
        client_id and client_secret in OAuth speak, respectively) being
        generated for that app. Upon successful registration, a Registered
        App type keyword gets appended to the app item.

        **Available to the item owner.**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        app_type            Required string. The type of app that was registered indicating
                            whether it's a browser app, native app, server app, or a multiple
                            interface app.
                            Values: browser, native, server, or multiple
        ---------------     --------------------------------------------------------------------
        redirect_uris       Optional list.  The URIs where the access_token or authorization
                            code will be delivered upon successful authorization. The
                            redirect_uri specified during authorization must match one of the
                            registered URIs, otherwise authorization will be rejected.

                            A special value of urn:ietf:wg:oauth:2.0:oob can also be specified
                            for authorization grants. This will result in the authorization
                            code being delivered to a portal URL (/oauth2/approval). This
                            value is typically used by apps that don't have a web server or a
                            custom URI scheme where the code can be delivered.

                            The value is a JSON string array.

                            Example:

                            [
                                "https://app.example.com",
                                "urn:ietf:wg:oauth:2.0:oob"
                            ]
        ---------------     --------------------------------------------------------------------
        http_referers       Optional List. A list of the http referrers for which usage of the
                            API Key will be restricted to.

                            **Example**

                            ```
                            [
                            "https://foo.com",
                            "https://bar.com"
                            ]
                            ```

                            Note: Http Referrers can be configured for non apiKey type apps as
                            well. The list configured here will be used to validate the app
                            tokens sent in while accessing the sharing API. The referrer checks
                            will not be applied to user tokens.
        ---------------     --------------------------------------------------------------------
        privileges          Optional List. A list of the privileges that will be available for
                            this API key.

                            **Example**

                            ```

                            [
                                 "portal:apikey:basemaps",
                                 "portal:app:access:item:itemId",
                                 "premium:user:geocode",
                                 "premium:user:networkanalysis"
                            ]

                            ```
                            Note: Privileges can be configured for non  `API Key` type apps as
                            well. The list configured here will be used to grant access to items
                            when item endpoint is accessed with app tokens. The checks will not
                            be applied to user tokens and they can continue accessing items
                            based on the current item sharing model. With app tokens, all items
                            of app owner can be accessed if the privileges list is not
                            configured.
        ===============     ====================================================================


        :return: dict

        """
        if not self.type.lower() in ['application', 'api key']:
            return None
        if redirect_uris is None:
            redirect_uris = []
        if str(app_type).lower() not in ["browser", "native", "server", "multiple", "apikey"]:
            raise ValueError(("Invalid app_type of : %s. Allowed values"
                              ": browser, native, server or multiple." % app_type))
        params = {
            "f" : 'json',
            "itemId" : self.id,
            "appType" : app_type,
            "redirect_uris" : redirect_uris
        }
        if http_referers:
            params['httpReferrers'] = http_referers
        if privileges:
            params['privileges'] = privileges
        url = "%soauth2/registerApp" % self._portal.resturl
        res = self._portal.con.post(url, params)
        self._hydrated = False
        self._hydrate()
        return res
    #----------------------------------------------------------------------
    def unregister(self):
        """

        The unregister app removes the application registration from an app
        item along with the Registered App type keyword.

        The operation is available to item owner and organization administrators.

        **Available to the item owner.**

        :return: boolean


        """
        appinfo = self.app_info
        if 'Registered App' not in self.typeKeywords:
            return False
        if appinfo == {} or len(appinfo) == 0:
            return False
        params = {"f" : 'json'}
        url = "%soauth2/apps/%s/unregister" % (self._portal.resturl, appinfo["client_id"])
        res =  self._portal.con.post(url, params)
        if res['success']:
            self._hydrated = False
            self._hydrate()
            return True
        return res['success']
    #----------------------------------------------------------------------
    @property
    def app_info(self):
        """
        If the parent item is registered using the register app operation,
        this resource returns information pertaining to the registered app.
        Every registered app gets an App ID and App Secret which in OAuth
        speak are known as client_id and client_secret respectively.

        :returns: dict

        """
        if "Registered App" not in self.typeKeywords:
            return {}
        url = "{base}content/users/{user}/items/{itemid}/registeredAppInfo".format(base=self._portal.resturl,
                                                                                   user=self._user_id,
                                                                                   itemid=self.id)
        params = {'f': 'json'}
        try:
            return self._portal.con.get(url, params)
        except:
            return {}

########################################################################
class ItemDependency(object):
    """
    Manage, monitor, and control Item dependencies.

    Depencies allows users to better understand the inter-dependency between their spatial assets.
    This capability provides the users with the following:

    - Users will be warned during item deletion if the deletion is going to break item/layer references in a web map or web application.
    - Users will be able to explore the dependents and dependencies of a specific item.
    - Portal administrators will be able to efficiently update the URLs of their hosted/federated services in an single edit operation.

    When an item is updated, its dependencies are updated as well and always kept in sync.

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Required Item. Item object to examine dependencies on.
    ===============     ====================================================================

    """
    _url = None
    _gis = None
    _con = None
    _item = None
    _portal = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, item):
        """Constructor"""
        self._item = item
        self._gis = item._gis
        self._con = self._gis._con
        self._portal = self._gis._portal
        self._url = '%scontent/items/%s/dependencies' % (self._portal.resturl,
                                                         item.itemid)
    #----------------------------------------------------------------------
    def __str__(self):
        return "<Dependencies for %s>" % self._item.itemid
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<Dependencies for %s>" % self._item.itemid
    #----------------------------------------------------------------------
    def __len__(self):
        return len(dict(self.properties)['items'])
    #----------------------------------------------------------------------
    def _init(self):
        params = {'f' : 'json',
                  'num' : 100,
                  'start' : 0}
        res = self._con.get(self._url, params)
        items = res['list']
        start = 0
        num = 100
        while res['nextStart'] > -1:

            start += num
            params = {'f' : 'json',
                      'num' : 100,
                      'start' : res['nextStart']}
            res = self._con.get(self._url, params)
            if 'list' in res:
                items += res['list']
        self._properties = PropertyMap({'items' : items})
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the dependencies properties"""
        if self._properties is None:
            self._init()
        return self._properties
    #----------------------------------------------------------------------
    def add(self, depend_type, depend_value):
        """
        Assigns a dependency to the current item

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        depend_type         Required String. This is the type of dependency that is registered
                            for the item. The allowed values are: table, url, or itemid.
        ---------------     --------------------------------------------------------------------
        depend_value        Required string. This is the associated value for the type above.
        ===============     ====================================================================

        :returns: Boolean

        """
        dtlu = {
            'table' : 'table',
            'url' : 'url',
            'itemid' : 'id'
        }
        params = {
            'f' : 'json',
            "type" : dtlu[depend_type],
            "id" : depend_value
        }
        url = "%s/addDependency" % self._url
        res = self._con.post(url, params)
        if 'error' in res:
            return res
        self._properties = None
        return True
    #----------------------------------------------------------------------
    def remove(self, depend_type, depend_value):
        """
        Deletes a dependency to the current item

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        depend_type         Required String. This is the type of dependency that is registered
                            for the item. The allowed values are: table, url, or itemid.
        ---------------     --------------------------------------------------------------------
        depend_value        Required string. This is the associated value for the type above.
        ===============     ====================================================================

        :returns: Boolean

        """
        dtlu = {
            'table' : 'table',
            'url' : 'url',
            'itemid' : 'id',
            'id' : 'id'
        }
        params = {
            'f' : 'json',
            "type" : dtlu[depend_type],
            "id" : depend_value

        }
        url = "%s/removeDependency" % self._url
        res = self._con.post(url, params)
        if 'error' in res:
            return res
        self._properties = None
        return True
    #----------------------------------------------------------------------
    def remove_all(self):
        """
        Revokes all dependencies for the current item

        :returns: boolean

        """
        if self._gis.version <= [8,2]:
            for i in dict(self.properties)['items']:
                if 'url' in i:
                    self.remove(i['dependencyType'], i['id'])
                elif 'id' in i:
                    self.remove(i['dependencyType'], i['id'])
                elif 'table' in i:
                    self.remove(i['dependencyType'], i['id'])
            self._properties = None
        else:
            url = "%s/removeAllDependencies" % self._url
            params = {'f' : 'json'}
            res = self._con.post(url, params)
            if 'error' in res:
                return res
            self._properties = None
        return True
    #----------------------------------------------------------------------
    @property
    def to_dependencies(self):
        """
        Returns a list of items that are dependent on the current Item
        """
        url = "%s/listDependentsTo" % self._url
        params = {'f' : 'json',
                  'num' : 100,
                  'start' : 0}
        res = self._con.get(url, params)

        items = res['list']
        num = 100
        while res['nextStart'] > -1:
            params = {'f' : 'json',
                      'num' : 100,
                      'start' : res['nextStart']}
            res = self._con.get(url, params)
            if 'list' in res:
                items += res['list']
        return items


def rot13(s, b64=False, of=False):
    if s is None:
        return None
    result = ""

    # If b64 is True, then first convert back to a string
    if b64:
        try:
            s = base64.b64decode(s).decode()
        except:
            raise RuntimeError('Reading value from profile is not correctly formatted. ' + \
                               'Update by creating a new connection using the profile option.')

    # Loop over characters.
    for v in s:
        # Convert to number with ord.
        c = ord(v)

        # Shift number back or forward.
        if c >= ord('a') and c <= ord('z'):
            if c > ord('m'):
                c -= 13
            else:
                c += 13
        elif c >= ord('A') and c <= ord('Z'):
            if c > ord('M'):
                c -= 13
            else:
                c += 13

        # Append to result.
        result += chr(c)

    # Return transformation.
    if of:
        return result
    if not b64:
        # if not base64 to start, need to convert to base64 for saving to file
        return (base64.b64encode(result.encode())).decode()
    else:
        return result

class _GISResource(object):
    """ a GIS service
    """
    def __init__(self, url, gis=None):

        from ._impl._con import Connection
        self._hydrated = False
        if str(url).lower().endswith("/"):
            url = url[:-1]
        self.url = url
        self._url = url

        if gis is None:
            gis = GIS(set_active=False)
            self._gis = gis
            self._con = gis._con
        else:
            self._gis = gis
            if isinstance(gis, Connection):
                self._con = gis
            else:
                self._con = gis._con

    @classmethod
    def fromitem(cls, item):
        if not item.type.lower().endswith('service'):
            raise TypeError("item must be a type of service, not " + item.type)
        return cls(item.url, item._gis)

    def _refresh(self):
        params = {"f": "json"}
        if type(self).__name__ == 'ImageryLayer' or type(self).__name__ == '_ImageServerRaster':
            if self._fn is not None:
                params['renderingRule'] = self._fn
            if hasattr(self, "_uri"):
                if isinstance(self._uri, bytes):
                    if 'renderingRule' in params.keys():
                        del params['renderingRule']
                params["Raster"] = self._uri

        if type(self).__name__ == 'VectorTileLayer': # VectorTileLayer is GET only
            dictdata = self._con.get(self.url, params, token=self._lazy_token)
        else:
            try:
                dictdata = self._con.post(self.url, params, token=self._lazy_token)
            except Exception as e:
                if hasattr(e, 'msg') and e.msg == "Method Not Allowed":
                    dictdata = self._con.get(self.url, params, token=self._lazy_token)
                elif str(e).lower().find("token required") > -1:
                    dictdata = self._con.get(self.url, params)
                else:
                    raise e

        self._lazy_properties = PropertyMap(dictdata)

    @property
    def properties(self):
        """The properties of this object"""
        if self._hydrated:
            return self._lazy_properties
        else:
            self._hydrate()
            return self._lazy_properties

    @properties.setter
    def properties(self, value):
        self._lazy_properties = value

    def _hydrate(self):
        """Fetches properties and deduces token while doing so"""
        self._lazy_token = None
        err = None

        with _DisableLogger():
            try:
                # try as a federated server
                if self._con.token is None:
                    self._lazy_token = self._con.generate_portal_server_token(serverUrl=self.url)
                else:
                    from ._impl._con import Connection
                    if isinstance(self._con, Connection):
                        self._lazy_token = self._con.generate_portal_server_token(serverUrl=self._url)
                    else:
                        self._lazy_token = self._con.token

                self._refresh()

            except HTTPError as httperror:  # service maybe down
                _log.error(httperror)
                err = httperror
            except RuntimeError as e:
                try:
                    # try as a public server
                    self._lazy_token = None
                    self._refresh()

                except HTTPError as httperror:
                    _log.error(httperror)
                    err = httperror
                except RuntimeError as e:
                    if 'Token Required' in e.args[0]:
                        # try token in the provided gis
                        self._lazy_token = self._con.token
                        self._refresh()
            except:
                try:
                    # try as a public server
                    self._lazy_token = None
                    self._refresh()

                except HTTPError as httperror:
                    _log.error(httperror)
                    err = httperror
                except RuntimeError as e:
                    if 'Token Required' in e.args[0]:
                        # try token in the provided gis
                        self._lazy_token = self._con.token
                        self._refresh()

        if err is not None:
            raise RuntimeError('HTTPError: this service url encountered an HTTP Error: ' + self.url)

        self._hydrated = True

    @property
    def _token(self):
        if self._hydrated:
            return self._lazy_token
        else:
            self._hydrate()
            return self._lazy_token


    def __str__(self):
        return '<%s url:"%s">' % (type(self).__name__, self.url)

    def __repr__(self):
        return '<%s url:"%s">' % (type(self).__name__, self.url)

    def _invoke(self, method, **kwargs):
        """Invokes the specified method on this service passing in parameters from the kwargs name-value pairs"""
        url = self._url + "/" + method
        params = { "f" : "json"}
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                params[k] = v
                del k,v
        return self._con.post(path=url, postdata=params, token=self._token)

class Layer(_GISResource):
    """
    The layer is a primary concept for working with data in a GIS.

    Users create, import, export, analyze, edit, and visualize layers.

    Layers can be added to and visualized using maps. They act as inputs to and outputs from analysis tools.

    Layers are created by publishing data to a GIS, and are exposed as a broader resource (Item) in the
    GIS. Layer objects can be obtained through the layers attribute on layer Items in the GIS.
    """

    def __init__(self, url, gis=None):
        super(Layer, self).__init__(url, gis)
        self.filter = None
        self._time_filter = None
        """optional attribute query string to select features to process by geoanalytics or spatial analysis tools"""

    @classmethod
    def fromitem(cls, item, index=0):
        """
        Returns the layer at the specified index from a layer item.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required string. An item ID representing a layer.
        ------------------     --------------------------------------------------------------------
        index                  Optional int. The index of the layer amongst the item's layers
        ==================     ====================================================================

        :return:
           The layer at the specified index.
        """
        return item.layers[index]

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : type(self).__name__, 'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': type(self).__name__, 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

    @property
    def _lyr_domains(self):
        """
        returns the domain information for any fields in the layer with domains
        """
        domains = []
        for field in [field for field in self.properties.fields if field['domain'] != None]:
            field_domain = dict(field.domain)
            field_domain['fieldName'] = field.name
            domains.append({field.name:field_domain})
        return domains

from arcgis.gis._impl._profile import ProfileManager
login_profiles = ProfileManager()
