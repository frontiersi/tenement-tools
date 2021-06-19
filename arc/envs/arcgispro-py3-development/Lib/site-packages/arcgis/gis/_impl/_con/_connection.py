"""
Connection Object that uses Python Requests

Requires: requests, requests_toolbelt,
Possible optional might be required: requests_ntlm, requests_kerberos, requests-oauthlib

"""
try:
    import arcpy
    HASARCPY = True
except ImportError:
    HASARCPY = False
except:
    HASARCPY = False

import sys
if sys.platform == 'win32':
    try:
        import certifi_win32
    except ImportError:
        pass

import os
import copy
import json
import uuid
import datetime
import mimetypes
import tempfile
from urllib.request import urlparse, unquote, urljoin
import requests
from requests import Session
from requests_toolbelt.downloadutils import stream
from requests_toolbelt.multipart.encoder import MultipartEncoder
from ._helpers import _filename_from_headers, _filename_from_url
from ._authguess import GuessAuth
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._isd import InsensitiveDict

__version__ = "1.8.5"

_DEFAULT_TOKEN = uuid.uuid4()

class Connection(object):
    """
    Universal Connection Object
    """
    _refresh_token = None
    _token = None
    _token_url = None
    _create_time = None
    _session = None
    _header = None
    _username = None
    _password = None
    _proxy_url = None
    _proxy_port = None
    _proxy_password = None
    _proxy_username = None
    _verify_cert = None
    _baseurl = None
    baseurl = None
    _referer = None
    _expiration = None
    _auth = None
    _product = None
    _custom_auth = None
    #----------------------------------------------------------------------
    def __init__(self,
                 baseurl=None,
                 username=None,
                 password=None,
                 **kwargs):
        """initializer

        Optional Kwargs

        client_id=None
        client_secret=None
        expiration=60
        cert_file
        key_file
        proxy_url
        proxy_port
        verify_cert
        referer
        all_ssl = True
        portal_connection = None
        product = "UNKNOWN", "PORTAL", "SERVER", "AGOL", "GEOEVENT"  # SHOULD BE SET BY GIS object, or Server Object, etc...
        token= None If token, _AUTH = BUILTIN
        token_url
        AUTH keys = HOME, BUILTIN, PRO, ANON, PKI, HANDLER, UNKNOWN (Internal)
        custom_auth = Requests authencation handler
        trust_env = T/F if to ignore netrc files
        """
        from arcgis.gis import GIS
        self._all_ssl = kwargs.pop("all_ssl", True)
        self.trust_env = kwargs.pop("trust_env", None)
        if baseurl:
            while baseurl.endswith("/"):
                baseurl = baseurl[:-1]
        self._custom_auth = kwargs.pop('custom_auth', None)
        self._product = kwargs.pop('product', "UNKNOWN")
        if baseurl is None:
            self._product = "AGOL"
            baseurl = "https://www.arcgis.com"


        self._baseurl = baseurl
        self._username = username
        self._password = password

        self._expiration = kwargs.pop('expiration', 60) or 60
        self._portal_connection = kwargs.pop('portal_connection', None) # For Federated Objects (Portal Connection)
        if isinstance(self._portal_connection, GIS):
            self._portal_connection = self._portal_connection._con

        if (self._referer or self._referer is None) and \
           self._portal_connection and \
           str(self._portal_connection._auth).lower() == "home":
            self._referer = None
        elif baseurl.lower() == 'pro':
            try:
                self._referer = arcpy.GetSigninToken().pop('referer', 'http')
            except:
                self._referer = kwargs.pop('referer', 'http')
        else:
            self._referer = kwargs.pop('referer', 'http')

        self._verify_cert = kwargs.pop("verify_cert", False)#True)
        if self._verify_cert == False:
            import warnings
            from requests.packages.urllib3 import exceptions
            warnings.simplefilter("ignore", exceptions.InsecureRequestWarning)

        self._cert_file = kwargs.pop('cert_file', None)
        self._key_file = kwargs.pop('key_file', None)
        if 'proxy_host' in kwargs:
            self._proxy_url = kwargs.pop('proxy_host', None)
        else:
            self._proxy_url = kwargs.pop('proxy_url', None)
        self._proxy_port = kwargs.pop('proxy_port', None)
        self._proxy_username = kwargs.pop('proxy_username', None)
        self._proxy_password = kwargs.pop('proxy_password', None)
        self._header = {'User-Agent' : 'Geosaurus/%s' % __version__}
        self._token = kwargs.pop('token', None)
        self._client_id = kwargs.pop('client_id', None)
        self._client_secret = kwargs.pop('client_secret', None)
        self._token_url = kwargs.pop('token_url', None)

        if username is None and password is None and self._portal_connection is None:
            self._auth = "ANON"
        elif ((not username is None and not password is None) or self._portal_connection):
            self._auth = "BUILTIN"
        if (username and password) and \
           self._client_id is None and \
           str(baseurl).lower() != "pro":
            self._auth = "BUILTIN"
        elif self._portal_connection:
            self._auth = "BUILTIN"
        elif baseurl.lower() == 'pro':
            self._auth = "PRO"
            self._baseurl = arcpy.GetActivePortalURL()
        elif self._cert_file or\
             (self._cert_file and self._key_file):
            self._auth = "PKI"

        self._create_session()

        #  Product Info
        if self._token:
            if self._expiration is None:
                self._expiration = 60
            self._create_time = datetime.datetime.now() + datetime.timedelta(minutes=self._expiration)
            self._auth = "BUILTIN"
        elif self._client_id:
            self._product = self._check_product()
            if self._product in ['PORTAL', "AGOL"]:
                resp = self.post("/portals/self", {'f' : 'json'}, add_token=False)
                issaml = resp.get("samlEnabled", False)
                isoauth = resp.get("supportsOAuth", False)
            else:
                resp = None
                issaml = False
                isoauth = False
            self._auth = 'OAUTH'
            parsed = urlparse(self._baseurl)
            wa = parsed.path
            if wa.startswith('/'):
                wa = wa[1:].split('/')[0]
            else:
                wa = wa.split('/')[0]
            if len(wa) > 0:
                self._token_url = "https://%s/%s/sharing/rest/oauth2/token" % (parsed.netloc, wa)
            else:
                self._token_url = "https://%s/sharing/rest/oauth2/token" % (parsed.netloc)

        else:
            self._product = self._check_product()
        self._baseurl = self._validate_url(self._baseurl)
        self.baseurl = self._baseurl
    #----------------------------------------------------------------------
    def _validate_url(self, url):
        """ensures the base url has the /sharing/rest"""
        if url.lower().find('arcgis.com') > -1:
            self._product = "AGOL"
        if self._product in ['AGO', 'AGOL', 'PORTAL']:
            if not url[-1] == '/':
                url += '/'
            if url.lower().find("www.arcgis.com") > -1:
                urlscheme = urlparse(url).scheme
                return "{scheme}://www.arcgis.com/sharing/rest/".format(scheme=urlscheme)
            elif url.lower().endswith("sharing/"):
                return url + 'rest/'
            elif url.lower().endswith("sharing/rest/"):
                return url
            else:
                return url + 'sharing/rest/'
        elif self._product in ['SERVER', 'FEDERATED_SERVER', "FEDERATED"]:
            parsed = urlparse(url)
            path = parsed.path
            if str(path).startswith("/"):
                path = path[1:]
            url = "%s://%s/%s/rest/services/" % (parsed.scheme,
                                                 parsed.netloc,
                                                 path.split('/')[0])
        return url
    #----------------------------------------------------------------------
    def _create_session(self):
        if self._proxy_port and self._proxy_url:
            url = "%s:%s" % (self._proxy_url, self._proxy_port)
            if self._proxy_password and self._proxy_username:
                proxies = {'http': 'http://%s:%s@%s' % (self._proxy_username,
                                                        self._proxy_password,
                                                        url),
                           'https': 'https://%s:%s@%s' % (self._proxy_username,
                                                          self._proxy_password,
                                                          url)}
            else:
                proxies = {'http': 'http://%s' % url,
                           'https': 'https://%s' % url}
        else:
            proxies = None

        self._session = Session()
        self._session.verify = self._verify_cert
        self._session.stream = True
        self._session.trust_env = self.trust_env
        self._session.headers.update(self._header)
        self._session.proxies = proxies
        if self._referer is None and\
           (self._portal_connection and \
           str(self._portal_connection._auth).lower() == "home"):
            self._referer = "http"
            self._session.headers.pop("Referer", None)
            self._session.headers['Referer'] = json.dumps("")
        elif (self._portal_connection and str(self._portal_connection._auth).lower() == "home"):
            self._referer = "http"
            self._session.headers.pop("Referer", None)
            self._session.headers['Referer'] = json.dumps("")
        elif self._referer is None:
            self._referer = 'http'
            self._session.headers.update({'Referer': self._referer})
        else:
            self._session.headers.update({'Referer': self._referer})
        if self._custom_auth:
            self._session.auth = self._custom_auth
            self._auth = "CUSTOM"
        elif self._username and self._password:
            self._session.auth = GuessAuth(username=self._username,
                                           password=self._password)
        elif self._auth.lower() == "pro":
            self._session.auth = GuessAuth(None, None)
        else:
            try:
                from requests_negotiate_sspi import HttpNegotiateAuth
                HAS_KERBEROS = True
            except:
                HAS_KERBEROS = False
            if HAS_KERBEROS:
                self._session.auth = HttpNegotiateAuth()
            else:
                try:
                    from requests_kerberos import HTTPKerberosAuth, DISABLED
                    self._session.auth = HTTPKerberosAuth(DISABLED)
                except ImportError:
                    pass
                except Exception as e:
                    raise e
        if self._cert_file and self._key_file:
            self._session.cert = (self._cert_file,
                                  self._key_file)
        elif self._cert_file and self._password:
            from arcgis.gis._impl._con._cert import pfx_to_pem
            self._key_file, self._cert_file = pfx_to_pem(pfx_path=self._cert_file, pfx_password=self._password)
            self._session.cert = (self._cert_file, self._key_file)
        elif self._cert_file:
            self._session.cert = (self._cert_file)
        else:
            self._session.cert = None
    #----------------------------------------------------------------------
    def get(self,
            path,
            params=None,
            **kwargs):
        """

        sends a GET request.

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        path                          required string. URI resource to access.
        ===========================   =====================================================

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        params                        optional dictionary.  This is the data payload to the
                                      web resource
        ---------------------------   -----------------------------------------------------
        out_folder                    optional string. Save path on disk.  The default is
                                      the user's temp directory.
        ---------------------------   -----------------------------------------------------
        file_name                     optional string. Name of the file to save. If not
                                      provided, name will try to be parsed from the response.
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If True, the marker f='json' will
                                      be appended to the parameters.
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean. Default True, if true, the call will
                                      add a token to any token based security.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Boolean.  When False, the JSON values will not be encoded.
        ===========================   =====================================================
        """
        json_encode = kwargs.pop('json_encode', True)
        if self._baseurl.endswith('/') == False:
            self._baseurl += "/"
        url = path
        token = kwargs.pop('token', _DEFAULT_TOKEN)
        token_as_header = kwargs.pop('token_as_header', False)
        token_header = kwargs.pop('token_header', "X-Esri-Authorization")
        if url.find('://') == -1:
            if url.startswith('/') == False and \
               self._baseurl.endswith('/') == False:
                url = "/" + url
            url = self._baseurl + url
        if kwargs.pop("ssl", False) or self._all_ssl:
            url = url.replace("http://", "https://")
        if params is None:
            params = {}
        #if self._auth == "IWA":
        #    self._session = None
        if self._session is None:
            self._create_session()

        try_json = kwargs.pop("try_json", True)
        add_token = kwargs.pop('add_token', True)

        if add_token:
            if token != _DEFAULT_TOKEN:
                if token is not None:
                    params['token'] = token
                else:
                    params.pop('token', None)
                    #pass
            elif token_as_header == False and self.token is not None: #as ?token=
                params['token'] = self.token
            elif token_as_header and self.token is not None:#(token and token != _DEFAULT_TOKEN): # as X-Esri-Auth header with given token
                self._session.headers.update({token_header: "Bearer %s" % token})
            elif token_as_header and token_header and self.token: # as X-Esri-Auth header with generated token
                self._session.headers.update({token_header: "Bearer %s" % self.token})
        if try_json:
            params['f'] = 'json'
        if params == {}:
            params = None

        if 'out_folder' in kwargs:
            out_path = kwargs.pop('out_folder',
                                  tempfile.gettempdir())
        else:
            out_path = kwargs.pop('out_path',
                                  tempfile.gettempdir())
        file_name = kwargs.pop('file_name', None)
        if params and json_encode:
            for k, v in copy.copy(params).items():
                if isinstance(v, (tuple, dict, list, bool)):
                    params[k] = json.dumps(v)
                elif isinstance(v, PropertyMap):
                    params[k] = json.dumps(dict(v))
                elif isinstance(v, InsensitiveDict):
                    params[k] = v.json
        try:
            if self._cert_file:
                cert = (self._cert_file, self._key_file)
            else:
                cert = None
            resp = self._session.get(url=url,
                                     params=params,
                                     cert=cert,
                                     verify=self._verify_cert)

        except requests.exceptions.SSLError as err:
            raise requests.exceptions.SSLError(
                "Please set verify_cert=False due to encountered SSL error: %s" % err)
        except requests.exceptions.InvalidURL as errIU:
            raise requests.exceptions.SSLError(
                "Invalid URL provided: %s" % errIU)
        except requests.exceptions.ConnectionError as errCE:
            raise requests.exceptions.ConnectionError(
                "A connection error has occurred: %s" % errCE)
        except requests.exceptions.InvalidHeader as errIH:
            raise requests.exceptions.InvalidHeader(
                "A invalid header was provided: %s" % errIH)
        except requests.exceptions.HTTPError as errh:
            raise requests.exceptions.HTTPError(
                "Http Error: %s" % errh)
        except requests.exceptions.MissingSchema as errMS:
            raise requests.exceptions.MissingSchema(
                "URL scheme must be provided: %s" % errMS)
        except requests.exceptions.RequestException as errRE:
            raise requests.exceptions.RequestException(
                "A general expection was raised: %s"  % errRE)
        except Exception as e:
            raise Exception('A general error occurred: %s' % e)
        except:
            import traceback
            raise Exception(
                'An unknown error occurred: %s' % traceback.format_exc())

        return self._handle_response(resp, file_name, out_path, try_json, force_bytes=kwargs.pop('force_bytes', False))
    #----------------------------------------------------------------------
    def _handle_response(self, resp, file_name, out_path, try_json, force_bytes=False):
        """
        handles the request responses

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        resp                          required resquests.Rsponse object.
        ---------------------------   -----------------------------------------------------
        file_name                     required string.  Name of the output file if needed.
        ---------------------------   -----------------------------------------------------
        out_path                      required string. Name of the save folder.
        ---------------------------   -----------------------------------------------------
        try_json                      required boolean. Determines if the response should
                                      be returned as a dictionary or the native format
                                      send back by a server
        ---------------------------   -----------------------------------------------------
        force_bytes                   Optional Boolean. If True, the results returns as bytes
                                      instead of a file path.
        ===========================   =====================================================

        returns: string, dictionary, or bytes depending on the response.

        """
        if 'Set-Cookie' in resp.headers and \
           (str(self._auth).lower() in ['anon'] or \
            (str(self._username).find("\\") > -1 and str(self._auth).lower() in ['builtin'])):
            self._auth = 'IWA'

        data = None
        url = resp.url
        if out_path and \
           os.path.isdir(out_path) == False:
            os.makedirs(out_path)
        if out_path is None:
            out_path = tempfile.gettempdir()
        if file_name is None and \
           (resp.headers['Content-Type'].lower().find('json') == -1 and \
           resp.headers['Content-Type'].lower().find('text') == -1):
            file_name = _filename_from_url(url) or _filename_from_headers(
                resp.headers) or None
        if force_bytes:
            try:
                return bytes(resp.content)
            except:
                return resp.content
        if file_name is not None:
            file_name = os.path.join(out_path, file_name)
            if os.path.isfile(file_name):
                os.remove(file_name)
            fp = stream.stream_response_to_file(response=resp,
                                                path=file_name,
                                                chunksize=512*2)
            return fp

        if try_json:
            if 'Content-Length' in resp.headers and int(resp.headers.get('Content-Length')) == 0:
                data = {}
            elif 'Transfer-Encoding' in resp.headers and \
               resp.headers['Transfer-Encoding'].lower() == 'chunked':
                data = None
                for it in resp.iter_lines(chunk_size=None, decode_unicode=True, delimiter=None):
                    if data is None:
                        data = it
                    else:
                        data += it
                data = json.loads(data)
                if 'error' in data:
                    raise Exception(data['error'])
            else:
                data = resp.json()
            #if 'error' in data:
                #raise Exception(data['error'])
            #return data
        #else:
            #return resp.text
            if 'error' in data:
                if 'messages' in data:
                    return data
                errorcode = data['error']['code'] if 'code' in data['error'] else 0
                self._handle_json_error(data['error'], errorcode)
            return data
        else:
            return resp.text
    #----------------------------------------------------------------------
    def _handle_json_error(self, error, errorcode):
        errormessage = error.get('message')
        # handles case where message exists in the dictionary but is None
        if errormessage is None:
            errormessage = "Unknown Error"
        #_log.error(errormessage)
        if 'details' in error and error['details'] is not None:
            if isinstance(error['details'], str):
                errormessage = f"{errormessage} \n {error['details']}"
                #_log.error(error['details'])
            else:
                for errordetail in error['details']:
                    if isinstance(errordetail, str):
                        errormessage = errormessage + "\n" + errordetail
                        #_log.error(errordetail)

        errormessage = errormessage + "\n(Error Code: " + str(errorcode) +")"
        raise Exception(errormessage)
    #----------------------------------------------------------------------
    def post(self,
             path,
             params=None,
             files=None,
             **kwargs):
        """
        sends a POST request.

        ===========================   =====================================================
        **Parameters**                **Description**
        ---------------------------   -----------------------------------------------------
        path                          optional string.  URL or part of the url resource
                                      to call.
        ---------------------------   -----------------------------------------------------
        params                        optional dict.  Contains data to pass to the web
                                      resource.
        ---------------------------   -----------------------------------------------------
        files                         optional list of files

                                      Files can be provided two ways:

                                      The most basic way is:
                                      Way1: {key : r"c:\temp\myfile.foo}
                                      This is just the file path and the key.


                                      The preferred way:

                                      Way 2: {key : (file_name, open(c:\temp\myfile.foo, 'rb'), image\jpeg)}

                                      Way 2 requires providing the filename, IO object as 'rb', and the mimetype.
        ===========================   =====================================================


        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call. If
                                      is_geoevent is True, then the token will be appended
                                      to the header.
        ---------------------------   -----------------------------------------------------
        is_geoevent                   optional boolean. True means the auth token will be placed in the header
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        out_folder                    optional string.  This is the save folder for the data.
        ---------------------------   -----------------------------------------------------
        file_name                     optional string. The save name of the file. This will override the file name if provided in the response.
        ---------------------------   -----------------------------------------------------
        force_bytes                   optional boolean.  Deprecated.
        ---------------------------   -----------------------------------------------------
        add_headers                   optional dict.  If provided, additional headers will be given for a single call.
        ---------------------------   -----------------------------------------------------
        post_json                     optional bool. If True, the data is pushed in the request's json parameter.  This is an edge case for Workflow Manager. The default is `False`.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Bool. If False, the key/value parameters will not be JSON encoded.
        ===========================   =====================================================

        :returns: data returned from the URL call.

        """
        json_encode = kwargs.pop("json_encode", True)
        if self._baseurl.endswith("/") == False:
            self._baseurl += "/"
        url = path
        token = kwargs.pop('token', _DEFAULT_TOKEN)
        post_json = kwargs.pop("post_json", False)
        token_as_header = kwargs.pop('token_as_header', False)
        token_header = kwargs.pop('token_header', "X-Esri-Authorization")
        #if self._auth == "IWA":
        #    self._session = None
        if 'postdata' in kwargs: # handles legacy issues
            params = kwargs.pop('postdata')
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)
        add_token = kwargs.pop('add_token', True)
        if url.find('://') == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False) or self._all_ssl:
            url = url.replace("http://", "https://")
        if add_token:
            if token != _DEFAULT_TOKEN:
                if token is not None:
                    params['token'] = token
                else:
                    params.pop('token', None)
                    #pass
            elif token_as_header == False and self.token is not None: #as ?token=
                params['token'] = self.token
            elif token_as_header and self.token is not None:#(token and token != _DEFAULT_TOKEN): # as X-Esri-Auth header with given token
                self._session.headers.update({token_header: "Bearer %s" % token})
            elif token_as_header and token_header and self.token: # as X-Esri-Auth header with generated token
                self._session.headers.update({token_header: "Bearer %s" % self.token})

        if try_json:
            params['f'] = 'json'
        if files:
            fields = { }
            if isinstance(files, dict):
                for k, v in files.items():
                    if isinstance(v, (list, tuple)):
                        fields[k] = v
                    else:
                        fields[k] = (os.path.basename(v),
                                     open(v, 'rb'),
                                     mimetypes.guess_type(v)[0])
            elif isinstance(files, (list, tuple)):
                for key, filePath, fileName in files:
                    if isinstance(fileName, str):
                        fields[key] = (fileName,
                                       open(filePath, 'rb'),
                                       mimetypes.guess_type(filePath)[0])
                    else:
                        fields[key] = v
            files = fields

        out_path = kwargs.pop('out_path', None) or \
            kwargs.pop('out_folder', None) or \
            tempfile.gettempdir()
        file_name = kwargs.pop('file_name', None)
        try:
            if self._cert_file:
                cert = (self._cert_file, self._key_file)
            else:
                cert = None
            if json_encode:
                for k,v in params.items():
                    if isinstance(v, (dict, list, tuple, bool)):
                        params[k] = json.dumps(v)
                    elif isinstance(v, PropertyMap):
                        params[k] = json.dumps(dict(v))
                    elif isinstance(v, InsensitiveDict):
                        params[k] = v.json
            if post_json:  # edge case workflow
                resp = self._session.post(url=url,
                                          json=params,
                                          cert=cert,
                                          files=files)
            else:
                resp = self._session.post(url=url,
                                          data=params,
                                          cert=cert,
                                          files=files)
        except requests.exceptions.SSLError as err:
            raise requests.exceptions.SSLError(
                "Please set verify_cert=False due to encountered SSL error: %s" % err)
        except requests.exceptions.InvalidURL as errIU:
            raise requests.exceptions.SSLError(
                "Invalid URL provided: %s" % errIU)
        except requests.exceptions.ConnectionError as errCE:
            raise requests.exceptions.ConnectionError(
                "A connection error has occurred: %s" % errCE)
        except requests.exceptions.InvalidHeader as errIH:
            raise requests.exceptions.InvalidHeader(
                "A invalid header was provided: %s" % errIH)
        except requests.exceptions.HTTPError as errh:
            raise requests.exceptions.HTTPError(
                "Http Error: %s" % errh)
        except requests.exceptions.RequestException as errRE:
            raise requests.exceptions.RequestException(
                "A general expection was raised: %s"  % errRE)
        except requests.exceptions.MissingSchema as errMS:
            raise requests.exceptions.MissingSchema(
                "URL scheme must be provided: %s" % errMS)
        except Exception as e:
            raise Exception('A general error occurred: %s' % e)
        except:
            import traceback
            raise Exception(
                'An unknown error occurred: %s' % traceback.format_exc())

        return self._handle_response(resp=resp,
                                     out_path=out_path,
                                     file_name=file_name,
                                     try_json=try_json,
                                     force_bytes=kwargs.pop('force_bytes', False))
    #----------------------------------------------------------------------
    def put(self, url, params=None, files=None, **kwargs):
        """
        sends a PUT request

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        url                           Optional String. The web endpoint.
        ---------------------------   -----------------------------------------------------
        params                        Optional dict. The Key/value pairs to send along with the request.
        ---------------------------   -----------------------------------------------------
        files                         Optional list. Allows users to provide a file or files
                                      to the operation.  The format should be:
                                      files = [[key, filePath, fileName], ...,[keyN, filePathN, fileNameN]]
        ===========================   =====================================================

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call.
        ---------------------------   -----------------------------------------------------
        token_as_header               Optional boolean.  If True, the token will go into
                                      the header as `Authorization` header.  This can be
                                      overwritten using the `token_header` parameter
        ---------------------------   -----------------------------------------------------
        token_header                  Optional String. If provided and token_as_header is
                                      True, authentication token will be placed in this
                                      instead on URL string.
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        post_json                     optional bool. If True, the data is pushed in the request's json parameter.  This is an edge case for Workflow Manager. The default is `False`.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Bool. If False, the key/value parameters will not be JSON encoded.
        ===========================   =====================================================

        :returns: dict or string depending on the response

        """
        token = kwargs.pop('token', _DEFAULT_TOKEN)
        post_json = kwargs.pop("post_json", False)
        json_encode = kwargs.pop("json_encode", True)
        out_path = kwargs.pop('out_path', None)
        file_name = kwargs.pop('file_name', None)
        token_as_header = kwargs.pop('token_as_header', True)
        token_header = kwargs.pop('token_header', "X-Esri-Authorization")
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)
        add_token = kwargs.pop('add_token', True)
        if url.find('://') == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False):
            url = url.replace("http://", "https://")
        if add_token:
            if token != _DEFAULT_TOKEN:
                if token is not None:
                    params['token'] = token
                else:
                    params.pop('token', None)
                    #pass
            elif token_as_header == False and self.token is not None: #as ?token=
                params['token'] = self.token
            elif token_as_header and self.token is not None:#(token and token != _DEFAULT_TOKEN): # as X-Esri-Auth header with given token
                self._session.headers.update({token_header: "Bearer %s" % token})
            elif token_as_header and token_header and self.token: # as X-Esri-Auth header with generated token
                self._session.headers.update({token_header: "Bearer %s" % self.token})

        if try_json:
            params['f'] = 'json'

        if files:
            fields = { }
            if isinstance(files, dict):
                for k, v in files.items():
                    if isinstance(v, (list, tuple)):
                        fields[k] = v
                    else:
                        fields[k] = (os.path.basename(v),
                                     open(v, 'rb'),
                                     mimetypes.guess_type(v)[0])
            elif isinstance(files, (list, tuple)):
                for key, filePath, fileName in files:
                    if isinstance(fileName, str):
                        fields[key] = (fileName,
                                       open(filePath, 'rb'),
                                       mimetypes.guess_type(filePath)[0])
                    else:
                        fields[key] = v

            params.update(fields)
            del files, fields
        if self._cert_file:
            cert = (self._cert_file, self._key_file)
        else:
            cert = None
        if json_encode:
            for k,v in params.items():
                if isinstance(v, (dict, list, tuple, bool)):
                    params[k] = json.dumps(v)
                elif isinstance(v, PropertyMap):
                    params[k] = json.dumps(dict(v))
                elif isinstance(v, InsensitiveDict):
                    params[k] = v.json
        if post_json:  # edge case workflow
            resp = self._session.put(url=url,
                                      json=params,
                                      cert=cert,
                                      files=files)
        else:
            resp = self._session.put(url=url,
                                     data=params,
                                     cert=cert,
                                     files=files)
        #
        return self._handle_response(resp=resp,
                                         out_path=out_path,
                                         file_name=file_name,
                                         try_json=try_json,
                                         force_bytes=kwargs.pop('force_bytes', False))
    #----------------------------------------------------------------------
    def delete(self, url, params=None, **kwargs):
        """
        sends a PUT request

        ===========================   =====================================================
        **Parameters**                **Description**
        ---------------------------   -----------------------------------------------------
        url                           Optional String. The web endpoint.
        ---------------------------   -----------------------------------------------------
        params                        Optional dict. The Key/value pairs to send along with the request.
        ===========================   =====================================================

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call.
        ---------------------------   -----------------------------------------------------
        token_as_header               Optional boolean.  If True, the token will go into
                                      the header as `Authorization` header.  This can be
                                      overwritten using the `token_header` parameter
        ---------------------------   -----------------------------------------------------
        token_header                  Optional String. If provided and token_as_header is
                                      True, authentication token will be placed in this
                                      instead on URL string.
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ===========================   =====================================================

        :returns: dict or string depending on the response
        """
        out_path = kwargs.pop('out_path', None)
        file_name = kwargs.pop('file_name', None)
        token_as_header = kwargs.pop('token_as_header', True)
        token_header = kwargs.pop('token_header', "X-Esri-Authorization")
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)
        add_token = kwargs.pop('add_token', True)
        if url.find('://') == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False):
            url = url.replace("http://", "https://")
        if add_token:
            if token_as_header == False and not 'token' in kwargs: #as ?token=
                params['token'] = self.token
            elif token_as_header == False and 'token' in kwargs: #as ?token= and user provides the token
                params['token'] = kwargs['token']
            elif token_as_header and 'token' in kwargs: # as X-Esri-Auth header with given token
                self._session.headers.update({token_header: "Bearer %s" % kwargs['token']})
            elif token_as_header and token_header and self.token: # as X-Esri-Auth header with generated token
                self._session.headers.update({token_header: "Bearer %s" % self.token})

        if try_json:
            params['f'] = 'json'

        resp = self._session.delete(url=url, data=params)
        return self._handle_response(resp=resp,
                                     out_path=out_path,
                                     file_name=file_name,
                                     try_json=try_json,
                                     force_bytes=kwargs.pop('force_bytes', False))
    #----------------------------------------------------------------------
    def streaming_method(self, url, callback,
                         data=None, json_data=None, verb="GET",
                         **kwargs):
        """
        Performs streaming web requests.

        =======================     ===========================================================
        **Parameters**              **Description**
        -----------------------     -----------------------------------------------------------
        url                         Required String. The web resource location.
        -----------------------     -----------------------------------------------------------
        callback                    Required Method.  The callback function to handle the response from the streaming request.

                                    **Example**
                                    ```
                                    def hook(r, *args, **kwargs):
                                        print('called a hook')
                                        return r
                                    ```
                                    See: https://requests.kennethreitz.org/en/master/user/advanced/#event-hooks


        -----------------------     -----------------------------------------------------------
        data                        Optional Dict. The parameters to pass to the method.
        -----------------------     -----------------------------------------------------------
        json_data                   Optional Dict. The parameters to pass to the method. This applies to POST only
        -----------------------     -----------------------------------------------------------
        verb                        Optional String.  The default is GET.  The allowed values are POST, GET, or PUT.
        -----------------------     -----------------------------------------------------------
        kwargs                      Optional Dict.  See https://requests.readthedocs.io/en/master/user/advanced/#request-and-response-objects
        =======================     ===========================================================

        """
        verbs = ['post', 'put', 'get']
        if verb.lower() in verbs:
            hooks = {'response' : callback}
            fn = getattr(self._session, verb.lower())
            if verb.lower() == "post":
                return fn(url=url,
                          data=data,
                          json_data=json,
                          hooks=hooks,
                          stream=True,
                          **kwargs)
            else:
                return fn(url=url,
                          data=data,
                          hooks=hooks,
                          stream=True,
                          **kwargs)
        else:
            allowed_verb = ",".join(verbs)
            raise ValueError(f"Invalid web method only {allowed_verb} as allowed")
        self._session.post(url=url, data=data, json_data=json, stream=True)
    #----------------------------------------------------------------------
    def login(self, username, password, expiration=None):
        """allows a user to login to a site with different credentials"""
        if expiration is None:
            expiration = 1440
        try:
            if self._username != username and \
               self._password != password:
                c = Connection(baseurl=self._baseurl,
                               username=username,
                               password=password)
                self = c
        except:
            raise Exception("Could not create a new login.")
    #----------------------------------------------------------------------
    def relogin(self, expiration=None):
        """ Re-authenticates with the portal using the same username/password. """
        if expiration is None:
            expiration = self._expiration
        self.logout()
        return self.token
    #----------------------------------------------------------------------
    def logout(self):
        """ Logs out of the portal. """
        self._token = None
        self._create_time = None
    #----------------------------------------------------------------------
    @property
    def is_logged_in(self):
        """ Returns true if logged into the portal. """
        return (self._auth in ['ANON', 'UNKNOWN']) == False
    #----------------------------------------------------------------------
    @property
    def product(self):
        """ Returns true if logged into the portal. """
        return self._product
    #----------------------------------------------------------------------
    @property
    def token(self):
        """Gets a Token"""
        if str(self._auth).lower() in ['builtin', 'oauth']:
            if self._expiration is None or self._expiration <= 5:
                self._expiration = 6
            if self._create_time and \
               (datetime.datetime.now() < self._create_time + datetime.timedelta(minutes=self._expiration)):
                return self._token
            elif str(self._auth).lower() == "oauth":
                self._create_time = datetime.datetime.now()
                self._token = self._oauth_token()
                return self._token
            elif self._create_time is None and \
               self._auth == "BUILTIN" and \
               self._product in ['SERVER', 'FEDERATED_SERVER']:
                self._token = self._server_token()
                return self._token
            elif self._product in ['FEDERATED_SERVER', 'GEOEVENT'] and \
                 self._portal_connection:
                self._token = self._server_token()
                return self._token
            elif self._create_time is None and \
                 self._auth == "OAUTH":
                self._token = self._oauth_token()
                return self._token
            elif self._create_time is None and \
                 self._product in ['PORTAL', "AGOL",
                                   "AGO", "ENTERPRISE"] and \
                 self._auth == "BUILTIN":
                self._token = self._enterprise_token()
                return self._token
            else:
                self._create_time = None
                return self.token
        elif str(self._auth.lower()) in ['home']:
            if self._create_time is None:
                self._create_time = datetime.datetime.now()
            if self._expiration is None:
                self._expiration = 1440
            if datetime.datetime.now() >= self._create_time + datetime.timedelta(minutes=self._expiration):
                raise Exception("Token is Expired. Please relogin to portal")
            return self._token
        elif str(self._auth).lower() == 'pro' or\
             self._baseurl.lower() == 'pro':
            self._create_time = datetime.datetime.now()
            self._token = self._pro_token()
            return self._token
        return None
    #----------------------------------------------------------------------
    @token.setter
    def token(self, value):
        """gets/sets the token"""
        if self._token != value:
            self._token = value
    #----------------------------------------------------------------------
    def _pro_token(self):
        """gets the token for various products"""
        if self._auth.lower() == 'pro' and \
           HASARCPY:
            resp = arcpy.GetSigninToken()
            if resp:
                if 'referer' in resp:
                    self._referer = resp['referer']
                if self._session:
                    self._session.headers['Referer'] = self._referer
                else:
                    self._referer = resp['referer']
                    self._session = self._create_session()
                if 'token' in resp:
                    return resp['token']
            else:
                raise Exception(("Could not login using Pro authencation."
                                 "Please verify in Pro that you are logged in."))
        return
    #----------------------------------------------------------------------
    def _server_token(self):
        """generates a server token"""
        if self._token_url is None and \
           self._portal_connection is None and \
           self._product in ['SERVER', "FEDERATED_SERVER"]:
            parsed = urlparse(self.baseurl)
            b = parsed.netloc
            if len(parsed.path) > 1:
                wa = parsed.path[1:].split('/')[0]
            else:
                wa = 'arcgis'
            ep = "admin/generateToken"
            self._token_url = "https://%s/%s/%s" % (b, wa, ep)
        elif self._token_url is None and \
             self._portal_connection:
            token_url = self._portal_connection._baseurl
            p = urlparse(token_url)
            self._token_url = "%s://%s/%s/sharing/rest/generateToken" % ("https",
                                                                         p.netloc,
                                                                         p.path[1:].split('/')[0],)
        if self._portal_connection:
            #self._token_url = token_url
            if self._portal_connection._auth.lower() == 'home':
                self._referer = ""
            ptoken = self._portal_connection.token
            postdata = {'serverURL':self._baseurl,
                        'token': ptoken,
                        'expiration': str(self._expiration),
                        'f': 'json',
                        'request':'getToken',
                        'referer':self._referer}
        else:
            postdata = { 'username': self._username,
                         'password': self._password,
                         #'client': 'requestip',
                         'referer' : self._referer,
                         'expiration': self._expiration,
                         'f': 'json' }
        res = self.post(path=self._token_url,
                        params=postdata,
                        ssl=True,
                        add_token=False)
        if 'error' in res:
            raise Exception(res['error'])
        self._create_time = datetime.datetime.fromtimestamp(int(res['expires'])/1000) - datetime.timedelta(minutes=self._expiration)
        self._token = res['token']
        return res['token']
    #----------------------------------------------------------------------
    def _enterprise_token(self):
        """generates a portal/agol token"""
        if self._referer is None and self._portal_connection is None:
            self._referer = "http"
        elif self._referer is None and self._portal_connection and \
             self._portal_connection._auth.lower() == "home":
            self._referer = ""
        postdata = { 'username': self._username, 'password': self._password,
                     'client': 'referer', 'referer': self._referer,
                     'expiration': self._expiration, 'f': 'json' }
        if self._token_url is None:
            if self._product in ['AGO', 'AGOL']:
                self._token_url = "https://%s/sharing/rest/generateToken" % urlparse(self._baseurl).netloc
            else:
                parsed = urlparse(self._baseurl)
                path = parsed.path
                if path.startswith('/'):
                    path = path[1:]
                self._token_url = "https://%s/%s/sharing/rest/generateToken" % (parsed.netloc,
                                                                                path.split('/')[0])

        res = self.post(path=self._token_url,
                        params=postdata,
                        add_token=False)
        if 'error' in res:
            raise Exception(res['error'])
        self._create_time = datetime.datetime.fromtimestamp(res['expires']/1000) - datetime.timedelta(minutes=self._expiration)
        return res['token']
    #----------------------------------------------------------------------
    def _oauth_token(self):
        """performs the oauth2 when secret and client exist"""
        from oauthlib.oauth2 import BackendApplicationClient
        from requests_oauthlib import OAuth2Session
        auth_url = '%soauth2/authorize' % self.baseurl
        tu = '%soauth2/token' % self.baseurl
        # handles the refreshing of the token
        if not (self._create_time is None) and \
           (datetime.datetime.now() >= self._create_time + datetime.timedelta(minutes=self._expiration)):
            self._token = None
        elif not (self._create_time is None) and \
             not (self._token is None) and \
             (datetime.datetime.now() < self._create_time + datetime.timedelta(minutes=self._expiration)):
            return self._token
        # Handles token generation
        if self._refresh_token is not None and \
           self._client_id is not None and\
           self._token is None: # Case 1: Refreshing a token
            parameters = {
                'client_id': self._client_id,
                'grant_type': 'refresh_token',
                'refresh_token': self._refresh_token,
                'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
            }
            token_info = self.post('oauth2/token', parameters, ssl=True, add_token=False)
            self._token = token_info['access_token']
            return self._token
        elif self._client_id and self._client_secret: # case 2: has both client and secret keys
            client = BackendApplicationClient(client_id=self._client_id)
            oauth = OAuth2Session(client=client)
            res = oauth.fetch_token(
                #method="GET",
                token_url=tu,
                client_id=self._client_id,
                client_secret=self._client_secret,
                include_client_id=True,
            )
            if 'expires_in' in res:
                self._create_time = datetime.datetime.fromtimestamp(res['expires_at']) - datetime.timedelta(seconds=7200)
                self._expiration = res['expires_in'] / 60
                if 'token' in res:
                    return res['token']
                if 'access_token' in res:
                    return res['access_token']
        elif self._client_id and \
             self._username is None and \
             self._password is None: # case 3: client id only

            auth_url = '%s/oauth2/authorize' % self.baseurl
            tu = '%s/oauth2/token' % self.baseurl
            oauth = OAuth2Session(self._client_id,
                                  redirect_uri="urn:ietf:wg:oauth:2.0:oob")
            authorization_url, state = oauth.authorization_url(auth_url)
            print("Please sign in to your GIS and paste the code that is obtained below.")
            print("If a web browser does not automatically open, please navigate to the URL below yourself instead.")
            print("Opening web browser to navigate to: " + authorization_url)
            import webbrowser, getpass
            webbrowser.open_new(authorization_url)
            authorization_response = getpass.getpass("Enter code obtained on signing in using SAML: ")

            self._create_time = datetime.datetime.now()
            token_info = oauth.fetch_token(tu,
                                           code=authorization_response,
                                           verify=False,
                                           include_client_id=True,
                                           authorization_response='authorization_code')
            self._expiration = token_info['expires_in']/60 - 2
            self._refresh_token = token_info['refresh_token']
            self._token = token_info['access_token']
            return self._token
        elif self._client_id and \
             not (self._username is None and \
                  self._password is None): # case 4: client id and username/password (SAML workflow)
            import re
            import json
            from bs4 import BeautifulSoup

            parameters = {
                'client_id': self._client_id,
                'response_type': 'code',
                'expiration': -1, # we want refresh_token to work for the life of the script
                'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
            }
            content = self.get('oauth2/authorize', parameters, ssl=True, try_json=False, add_token=False)

            pattern = re.compile('var oAuthInfo = ({.*?});', re.DOTALL)
            if len(pattern.findall(content)) == 0:
                pattern = re.compile('var oAuthInfo = ({.*?})', re.DOTALL)
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup.find_all('script'):
                script_code = str(script.string).strip()
                matches = pattern.search(script_code)
                if not matches is None:
                    js_object = matches.groups()[0]
                    try:
                        oauth_info = json.loads(js_object)
                    except:
                        oauth_info = json.loads(js_object + "}")
                    break

            parameters = {
                'user_orgkey': '',
                'username': self._username,
                'password': self._password,
                'oauth_state': oauth_info['oauth_state']
            }
            content = self.post('oauth2/signin', parameters, ssl=True, try_json=False, add_token=False)
            soup = BeautifulSoup(content, 'html.parser')

            if soup.title is not None:
                if 'SUCCESS' in soup.title.string:
                    code = soup.title.string[len('SUCCESS code='):]

            oauth = OAuth2Session(self._client_id,
                                  redirect_uri="urn:ietf:wg:oauth:2.0:oob")
            if code is None:
                raise Exception("Could not generate a token.")
            self._create_time = datetime.datetime.now()
            token_info = oauth.fetch_token(tu, code=code, verify=False, include_client_id=True,
                                           authorization_response='authorization_code')
            self._refresh_token = token_info['refresh_token']
            self._token = token_info['access_token']
            self._expiration = token_info['expires_in']/60 - 2

            return self._token
        return None
    #----------------------------------------------------------------------
    def generate_portal_server_token(self, serverUrl, expiration=1440):
        """generates a server token using Portal token"""
        if self._auth.lower() in ["pki", 'iwa']:
            postdata = {'request': 'getToken',
                        'serverURL':serverUrl,
                        'referer': self._referer,
                        'f': 'json'}
            if expiration:
                postdata['expiration'] = expiration
        else:
            token = self.token
            postdata = {'serverURL':serverUrl,
                    'token': token,
                    'expiration':str(expiration),
                    'f': 'json',
                    'request':'getToken',
                    'referer':self._referer}
        if self._token_url is None:
            if self.baseurl.endswith('/'):
                resp = self.post('generateToken', postdata,
                                 ssl=True, add_token=False)
            else:
                resp = self.post('/generateToken', postdata,
                                 ssl=True, add_token=False)
        else:
            resp = self.post(path=self._token_url, postdata=postdata,
                             ssl=True, add_token=False)
        if isinstance(resp, dict) and resp:
            return resp.get('token')
        else:
            raise Exception(f"Could not generate the token for the service. \n Error Message: \n {resp}")
    #----------------------------------------------------------------------
    def _oauth_authenticate(self):
        """performs oauth check when only client_id is provided"""
        from urllib.parse import urlencode
        client_id = self._client_id
        expiration = self._expiration or 1440
        parameters = {
            'client_id': client_id,
            'response_type': 'code',
            'expiration': -1, # we want refresh_token to work for the life of the script
            'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
        }

        code = None

        if self._username is not None and self._password is not None: # built-in user through OAUTH
            content = self.get(self._token_url,
                               parameters,
                               ssl=True, try_json=False, add_token=False)
            import re
            import json
            from bs4 import BeautifulSoup
            pattern = re.compile('var oAuthInfo = ({.*?});', re.DOTALL)
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup.find_all('script'):
                script_code = str(script.string).strip()
                matches = pattern.search(script_code)
                if not matches is None:
                    js_object = matches.groups()[0]
                    oauth_info = json.loads(js_object)
                    break

            parameters = {
                'user_orgkey': '',
                'username': self._username,
                'password': self._password,
                'oauth_state': oauth_info['oauth_state']
            }
            content = self.post('oauth2/signin', parameters, ssl=True, try_json=False, add_token=False)
            soup = BeautifulSoup(content, 'html.parser')

            if soup.title is not None:
                if 'SUCCESS' in soup.title.string:
                    code = soup.title.string[len('SUCCESS code='):]

        if code is None: # try interactive signin
            url = self._token_url#self._baseurl + 'oauth2/authorize'
            paramstring = urlencode(parameters)
            codeurl = "{}?{}".format(url, paramstring)

            import webbrowser
            import getpass

            print("Please sign in to your GIS and paste the code that is obtained below.")
            print("If a web browser does not automatically open, please navigate to the URL below yourself instead.")
            print("Opening web browser to navigate to: " + codeurl)
            webbrowser.open_new(codeurl)
            code = getpass.getpass("Enter code obtained on signing in using SAML: ")

        if code is not None:
            parameters = {
                'client_id': client_id,
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
            }
            token_info = self.post(self._token_url, parameters, ssl=True, add_token=False)
            # print('******' + str(token_info))

            self._refresh_token = token_info['refresh_token']
            self._token = token_info['access_token']

            return self._token
        else:
            print("Unable to sign in using OAUTH")
            return None

    #----------------------------------------------------------------------
    def _check_product(self):
        """
        determines if the product is portal, arcgis online or arcgis server
        """
        from urllib.error import HTTPError
        baseurl = self._baseurl
        if self._product == "SERVER":
            if self._token_url is None:
                parsed = urlparse(self._baseurl)
                path = parsed.path[1:].split('/')[0]
                self._token_url = "https://%s/%s/admin/generateToken" % (parsed.netloc,
                                                                         path)

            return "SERVER"
        if baseurl is None:
            return "UNKNOWN"
        if baseurl.lower().find("arcgis.com") > -1:
            parsed = urlparse(self._baseurl)
            self._token_url = "https://%s/sharing/rest/generateToken" % parsed.netloc
            return "AGOL"
        elif baseurl.lower().find("/sharing/rest") > -1:
            if baseurl.endswith("/"):
                res = self.get(baseurl + 'info', params={'f' : 'json'}, add_token=False)
            else:
                res = self.get(baseurl + '/info', params={'f' : 'json'}, add_token=False)
            if self._token_url is None and \
               res is not None and \
               isinstance(res, dict) and \
               'authInfo' in res and \
               'tokenServicesUrl' in res['authInfo'] and \
               res['authInfo']['isTokenBasedSecurity']:
                parsed_from_system = urlparse(res['authInfo']['tokenServicesUrl'])
                parsed = urlparse(baseurl)
                if parsed.netloc.lower() != parsed_from_system.netloc.lower() and \
                   parsed.netloc.find(":7443") > -1: # WA not being used for token url
                    self._token_url = os.path.join(
                        parsed_from_system.scheme + "://",
                        parsed.netloc  + "/arcgis/" + "/".join(parsed_from_system.path[1:].split("/")[1:])
                    )
                    url_test = self._session.post(self._token_url, {'f': 'json'}, allow_redirects=False)
                    if url_test.status_code == 301:
                        self._token_url = url_test.headers['location']
                        if self._baseurl != os.path.dirname(url_test.headers['location']):
                            self._baseurl = os.path.dirname(url_test.headers['location'])
                else:
                    self._token_url = res['authInfo']['tokenServicesUrl']
            elif self._token_url is None and \
                 res is not None and \
                 isinstance(res, dict) and \
                 'authInfo' in res and \
                 'tokenServicesUrl' in res['authInfo'] and \
                 res['authInfo']['isTokenBasedSecurity'] == False:
                self._token_url = None
                self._auth = "OTHER"
            return "PORTAL"
        else:
            #Brute Force Method
            parsed = urlparse(baseurl)
            root = fr"{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split(r'/')[0]}"
            #root = baseurl.lower().split("/sharing")[0]
            #root = baseurl.lower().split('/rest')[0]
            parts = ['/info', '/rest/info', '/sharing/rest/info']
            params = {"f" : "json"}
            for pt in parts:
                try:
                    #print(pt)
                    res = self.get(root + pt, params=params, add_token=False)
                    if self._token_url is None and \
                       res is not None and \
                       isinstance(res, dict) and \
                       'authInfo' in res and \
                       'tokenServicesUrl' in res['authInfo'] and \
                       res['authInfo']['isTokenBasedSecurity']:
                        self._token_url = res['authInfo']['tokenServicesUrl']
                    elif self._token_url is None and \
                         res is not None and \
                         isinstance(res, dict) and \
                         'authInfo' in res and \
                         'tokenServicesUrl' in res['authInfo'] and \
                         res['authInfo']['isTokenBasedSecurity'] == False:
                        self._token_url = None
                        self._auth = "OTHER"
                except HTTPError as e:
                    res = ""
                except json.decoder.JSONDecodeError:
                    res = ""
                except Exception as e:
                    res = ""
                if isinstance(res, dict) and \
                   "currentVersion" in res and \
                   self._token_url:
                    t_parsed = urlparse(self._token_url[1:]).path
                    b_parsed = urlparse(self._baseurl[1:]).path
                    if t_parsed.startswith("/"):
                        t_parsed = t_parsed[1:].split("/")[0]
                    else:
                        t_parsed = t_parsed.split("/")[0]
                    if b_parsed.startswith("/"):
                        b_parsed = b_parsed[1:].split("/")[0]
                    else:
                        b_parsed = b_parsed.split("/")[0]
                    if t_parsed.lower() != b_parsed.lower():
                        self._token_url = None
                        if self._portal_connection:
                            return "FEDERATED_SERVER"
                        else:
                            from arcgis.gis import GIS
                            self._portal_connection = GIS(url=res['authInfo']['tokenServicesUrl'].split("/sharing/")[0],
                                                          username=self._username, password=self._password,
                                                          verify_cert=self._verify_cert)._con
                            return "FEDERATED_SERVER"
                    return "SERVER"
                elif isinstance(res, dict) and \
                     "currentVersion" in res and \
                     self._token_url is None:
                    return "SERVER"
                del pt
                del res
        return "PORTAL"

