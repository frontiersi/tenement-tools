# __init__ for osgeo package.

# making the osgeo package version the same as the gdal version:
from sys import version_info, path as syspath
from os.path import dirname, join, exists
from os import getenv, makedirs, environ

for entry in syspath:
    if "\\ArcPy" in entry:
        gdalplugins_path = set([join(dirname(dirname(entry)), r"bin\gdalplugins")]);
        if getenv("GDAL_DRIVER_PATH") is not None:
            gdalplugins_path.add(getenv("GDAL_DRIVER_PATH"))

        environ["GDAL_DRIVER_PATH"] = ';'.join(gdalplugins_path)

        config_file = join(dirname(dirname(entry)), r"bin\gdalrc")
        environ['GDAL_CONFIG_FILE'] = config_file

        break;

if version_info >= (2,6,0):
    def swig_import_helper():
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_gdal', [dirname(__file__)])
        except ImportError:
            import _gdal
            return _gdal
        if fp is not None:
            try:
                _mod = imp.load_module('_gdal', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _gdal = swig_import_helper()
    del swig_import_helper
else:
    import _gdal

__version__ = _gdal.__version__ = _gdal.VersionInfo("RELEASE_NAME")

for entry in syspath:
    if "\\ArcPy" in entry:
        gdaldata_dir = join(dirname(entry), r"pedata\gdaldata")
        _gdal.SetConfigOption('GDAL_DATA', _gdal.GetConfigOption('GDAL_DATA', gdaldata_dir))
        break

import errno
import winreg
fnf_exception = getattr(__builtins__,
                        'FileNotFoundError', WindowsError)
def proreg(reg_path, lookup_key, reverseCheck = False):
    """ Look up a specific Pro registry key, optionally returning
        the related hive instead of the value itself."""
    READ_ACCESS = (winreg.KEY_WOW64_64KEY + winreg.KEY_READ)
    root_keys = (
        ('HKCU', winreg.HKEY_CURRENT_USER),
        ('HKLM', winreg.HKEY_LOCAL_MACHINE)
    )
    ordered_root_keys = root_keys[::-1] if reverseCheck else root_keys
    for (key_name, root_key) in ordered_root_keys:
        try:
            key = winreg.OpenKey(root_key, reg_path, 0, READ_ACCESS)
        except fnf_exception as error:
            key = None
            if error.errno == errno.ENOENT:
                pass
            else:
                raise
        if key:
            try:
                raw_value = winreg.QueryValueEx(key, lookup_key)[0]
                if raw_value:
                        return raw_value
            except fnf_exception as error:
                if error.errno == errno.ENOENT:
                    pass
                else:
                    raise
    return None

proxyDir = proreg('Software\\ESRI\\ArcGISPro\\Raster\\Environment', 'TMPDIR', True)
if proxyDir is None:
    proxyDir = getenv('LOCALAPPDATA') + '\\ESRI\\rasterproxies'

if proxyDir is not None and _gdal.GetConfigOption('GDAL_PAM_PROXY_DIR') is None:
    if not exists(proxyDir):
        makedirs(proxyDir)
    _gdal.SetConfigOption('GDAL_PAM_PROXY_DIR', proxyDir)