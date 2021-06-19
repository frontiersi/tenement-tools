import os
import importlib
from pathlib import Path

def getFileVersion(filename):
    from ctypes import (
        byref, c_char, c_uint, cast, windll,
        pointer, POINTER, sizeof, Structure, WinError
    )
    from ctypes.wintypes import BOOL, DWORD, LPVOID, LPCVOID, LPCWSTR

    class VS_FIXEDFILEINFO(Structure):
        _fields_ = [
            ("dwSignature", DWORD),  # will be 0xFEEF04BD
            ("dwStrucVersion", DWORD),
            ("dwFileVersionMS", DWORD),
            ("dwFileVersionLS", DWORD),
            ("dwProductVersionMS", DWORD),
            ("dwProductVersionLS", DWORD),
            ("dwFileFlagsMask", DWORD),
            ("dwFileFlags", DWORD),
            ("dwFileOS", DWORD),
            ("dwFileType", DWORD),
            ("dwFileSubtype", DWORD),
            ("dwFileDateMS", DWORD),
            ("dwFileDateLS", DWORD),
        ]

    PUINT = POINTER(c_uint)
    LPDWORD = POINTER(DWORD)

    GetFileVersionInfoSizeW = windll.version.GetFileVersionInfoSizeW
    GetFileVersionInfoSizeW.restype = DWORD
    GetFileVersionInfoSizeW.argtypes = [LPCWSTR, LPDWORD]
    GetFileVersionInfoSize = GetFileVersionInfoSizeW  # alias

    GetFileVersionInfoW = windll.version.GetFileVersionInfoW
    GetFileVersionInfoW.restype = BOOL
    GetFileVersionInfoW.argtypes = [LPCWSTR, DWORD, DWORD, LPVOID]
    GetFileVersionInfo = GetFileVersionInfoW  # alias

    VerQueryValueW = windll.version.VerQueryValueW
    VerQueryValueW.restype = BOOL
    VerQueryValueW.argtypes = [LPCVOID, LPCWSTR, POINTER(LPVOID), PUINT]
    VerQueryValue = VerQueryValueW  # alias

    dwLen = GetFileVersionInfoSize(filename, None)
    if not dwLen:
        raise WinError()
    lpData = (c_char * dwLen)()
    if not GetFileVersionInfo(filename, 0, sizeof(lpData), lpData):
        raise WinError()
    uLen = c_uint()
    lpffi = POINTER(VS_FIXEDFILEINFO)()
    lplpBuffer = cast(pointer(lpffi), POINTER(LPVOID))
    if not VerQueryValue(lpData, "\\", lplpBuffer, byref(uLen)):
        raise WinError()
    ffi = lpffi.contents
    return [
        int(ffi.dwFileVersionMS >> 16),
        int(ffi.dwFileVersionMS & 0xffff),
        int(ffi.dwFileVersionLS >> 16),
        int(ffi.dwFileVersionLS & 0xffff),
    ]


msg = """The version of the binaries in the active Python environment do
  not match the version of the binaries of the ArcGIS Pro application.

Active environment version : {}
arcgispro-py3 environment version : {}

To create a new environment that is up to date with arcgispro-py3:
 1. Generate a list of additional packages installed in your current environment,
    with the conda command:
        > conda env export > env.yaml
 2. (Optional) If you have additional dependencies installed through pip,
    find those with pip freeze:
        > pip freeze > requirements.txt
 3. Create a new environment by cloning arcgispro-py3, and activate it:
        > conda create --clone arcgispro-py3 --name my-env --pinned
        > activate my-env
 4. Add back missing conda packages from your current environment:
        > conda env update -n my-env -f env.yaml
 5. (Optional) Add back missing packages from pip:
        > pip install -r requirements.txt
 6. (Optional) Make it the default for the ArcGIS Pro application and the
     "Python Command Prompt":
        > proswap my-env
"""

try:
    # check compatibility of arcgisscripting.pyd in the current env
    #  vs the on in the active ArcGISPro install
    active_pyd_pth = Path(__file__).parent.joinpath("_arcgisscripting.pyd")
    pristine_pyd_pth = Path(importlib.util.find_spec("arcpy").origin).parents[
        3].joinpath('bin', 'Python', 'envs', 'arcgispro-py3', 'Lib',
                    'site-packages', "arcgisscripting", "_arcgisscripting.pyd")

    if active_pyd_pth != pristine_pyd_pth:
        active_pyd_ver = getFileVersion(str(active_pyd_pth))
        pristine_pyd_ver = getFileVersion(str(pristine_pyd_pth))

        active_pyd_ver = f"{active_pyd_ver[0] - 10}.{active_pyd_ver[1]}"
        pristine_pyd_ver = f"{pristine_pyd_ver[0] - 10}.{pristine_pyd_ver[1]}"

        # check major and minor version (NOT patch or build)
        assert active_pyd_ver == pristine_pyd_ver

except AssertionError:
    print(msg.format(active_pyd_ver, pristine_pyd_ver))
    raise ImportError("version mis-match") from None

except:
    # don't raise if problem occurred elsewhere with version check
    pass

# delete objects so they don't pollute the arcgisscripting namespace
del(Path)
del(active_pyd_pth)
del(getFileVersion)
del(importlib)
del(msg)
del(os)
del(pristine_pyd_pth)


from ._arcgisscripting import *
from ._arcgisscripting import _addTimeInterval, _analyzeForSD, _attachLocator, \
    _convertWebMapToMapDocument, _createGISServerConnectionFile, \
    _createGeocodeSDDraft, _createMapSDDraft, _createimageservicesddraft, \
    _getImageEXIFProperties, _getRasterKeyMetadata, _getUTMFromLocation, \
    _hasLocalFunctionRasterImplementation, _hgp, _listDateTimeStringFormats, \
    _listStyleItems, _listTimeZones, _listTimeZonesLabels, _mapping, _reserved, \
    _setRasterKeyMetadata, _sharing, _ss, _wrapLocalFunctionRaster, \
    _wrapToolRaster, _ia
