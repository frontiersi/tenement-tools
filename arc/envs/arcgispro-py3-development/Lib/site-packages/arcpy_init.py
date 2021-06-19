# -*- coding: utf-8 -*-
# Detect ArcPy components installed and add them to the environment path.
__version__ = "2.8"

import os
import re
import subprocess
import sys
import winreg

from pathlib import Path

fnf_exception = getattr(__builtins__, "FileNotFoundError", WindowsError)

__all__ = ["product_install_dir", "set_product_paths", "install"]


def get_product_from_history():
    """Determine product from the conda-meta/history file."""
    history_path  = Path(__file__).parent.parent.parent / "conda-meta" / "history"
    if not history_path.exists():
        return None

    product_exes = {
        'Drone2Map': "Drone2Map.exe",
        'Server':    "ArcSOC.exe",
        'Pro':       "ArcGISPro.exe"
    }
    flags = re.IGNORECASE
    with open(history_path) as f:
        for line in f:
            if re.search(r"^# cmd:.+conda(\.exe)? create --clone", line, flags):
                # check for full clone origin path
                bin_search = re.search(r"(?<= --clone ).+?(?=\\\\Python\\\\envs\\\\arcgispro-py3)", line, flags)
                if bin_search:
                    bin_path = Path(bin_search[0])
                # fall back on conda executable path
                else:
                    bin_search = re.search(r"^# cmd: (.*)(?=\\\\Python\\\\Scripts\\\\conda)", line, flags)
                    if bin_search:
                        bin_path = Path(bin_search[1])
                if not bin_path.exists():
                    return None

                for (product, exe) in product_exes.items():
                    if (bin_path / exe).is_file():
                        return product

    return None

def product_install_dir():
    """Returns the ArcGIS product installation directory as a pathlib.Path."""
    install_dir = None

    root_keys = {
        "HKCU": winreg.HKEY_CURRENT_USER,
        "HKLM": winreg.HKEY_LOCAL_MACHINE,
    }

    reg_paths = {
        'Drone2Map': "SOFTWARE\\Esri\\Drone2Map",
        'Server':    "SOFTWARE\\Esri\\ArcGIS_SxS_Server",
        'Pro':       "SOFTWARE\\Esri\\ArcGISPro",
    }

    product_paths = {}
    for (key_name, root_key) in root_keys.items():
        for (product, reg_path) in reg_paths.items():
            try:
                key = winreg.OpenKey(
                    root_key, reg_path, 0, (winreg.KEY_WOW64_64KEY + winreg.KEY_READ)
                )
            except fnf_exception as error:
                key = None
            if key:
                try:

                    install_dir = Path(winreg.QueryValueEx(key, "InstallDir")[0])
                    if "Server" in reg_path:
                        install_dir = install_dir / r"framework\runtime\ArcGIS"

                    if install_dir.exists():
                        product_paths[product] = install_dir

                except fnf_exception as error:
                    pass

    if len(product_paths) > 1:
        try:
            # check for a install dir from a product
            script_install_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent.absolute()
            for (product, product_path) in product_paths.items():
                if script_install_dir.resolve() == product_path:
                    install_dir = product_path
                    raise
            # fall back on product from whence the env was cloned
            product_cloned = get_product_from_history()
            if product_cloned in product_paths.keys():
                install_dir = product_paths.get(product_cloned, install_dir)
        except:
            pass

    if len(product_paths) == 1:
        install_dir = list(product_paths.values())[0]

    if not install_dir:
        aprx = "ArcGISProjectFile"
        aprx_ftype = None
        try:
            aprx_ftype = subprocess.check_output(
                ["ftype", aprx],
                stderr=subprocess.STDOUT,
                shell=True
            )
        except subprocess.CalledProcessError as e:
            # association not found in hive
            pass

        if aprx_ftype:
            try:
                pro_exe_dir = Path(str(aprx_ftype).split('"')[1])
                install_dir = pro_exe_dir.parent.parent
            except:
                pass

    if not install_dir or not install_dir.exists():
        raise ImportError(
            "Unable to locate a valid Pro or Server installation to use with ArcPy. ArcPy must be installed, and match the version of the package used."
        )

    return install_dir


def set_product_paths():
    """Set paths necessary to detect the product installation from Python."""
    install_dir = product_install_dir()
    bin_dir = install_dir / "bin"
    env_dir = Path(__file__).parent.parent.parent.absolute()

    # use Path objects for case insensitive comparisons
    win_path = [Path(p) for p in os.environ["PATH"].split(";")]
    win_add_paths = {
        bin_dir: True,                          # app bin
        bin_dir / "Python" / "Scripts": False,  # scripts
        env_dir: False,                         # env root
        env_dir / "Scripts": False,             # env scripts
    }

    # check if we have these entries, they are not automatically included in app
    for (p, prepend) in win_add_paths.items():
        if p not in win_path:
            if prepend:
                win_path.insert(0, p)
            else:
                win_path.append(p)

    # recreate as normalized variable
    os.environ["PATH"] = ";".join([str(p) for p in win_path])

    # add entries to sys.path to locate ArcPy and ArcToolbox scripts
    add_sys_paths = {
        bin_dir: False,
        install_dir / "Resources" / "ArcPy": True,
        install_dir / "Resources" / "ArcToolbox" / "Scripts": False,
    }

    # use the dict as a fast ordered set to remove duplicate entries
    sys_path = list({Path(p): True for p in sys.path}.keys())
    for (entry_path, prepend) in add_sys_paths.items():
        if entry_path not in sys_path:
            if prepend:
                sys_path.insert(0, entry_path)
            else:
                sys_path.append(entry_path)

    # convert back to string paths
    sys.path = [str(p) for p in sys_path]

    # modify Intel Fortran Control-C handler if unset
    ctrl_c_handler = "FOR_DISABLE_CONSOLE_CTRL_HANDLER"
    if ctrl_c_handler not in os.environ:
        os.environ[ctrl_c_handler] = "1"


def install():
    """If running on a non-final build, reuse the ArcPy DLLs from the installed build."""
    from lxml import etree
    from lxml.objectify import BoolElement
    from shutil import copyfile
    import pefile

    install_dir = product_install_dir()
    bin_dir = install_dir / "bin"
    source_site_packages = bin_dir / "Python/envs/arcgispro-py3/Lib/site-packages"
    dest_site_packages = Path(__file__).parent
    product = None
    is_dev = False
    build_provided_dlls = (
        "arcgisscripting/_arcgisscripting.pyd",
        "arcpy_wmx/_arcpy_wmx.pyd",
        "bapy/_bapy.pyd",
        "gapy/_gapy.pyd",
    )

    try:
        root = etree.parse(str(bin_dir / "ArcGIS.Installation.xml"))
        installation = root.getroot()
        product, version = installation.values()
        if product == "Server":
            # not supported on Server
            return
        dvs = root.xpath("//Installation/DisplayVersion")
    except:
        raise RuntimeError("Failed to parse ArcGIS version information")

    if len(dvs) == 1:
        product_full, show_build = dvs[0].values()
        try:
            is_dev = BoolElement(show_build).__bool__()
        except:
            print("Unable to determine build type")

    if is_dev:
        # check if we have matching interpreter versions
        arcgisscripting_dll = source_site_packages / build_provided_dlls[0]
        current_py_dll = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        if arcgisscripting_dll.exists():
            pe = pefile.PE(arcgisscripting_dll)
            loaded_dlls = [
                entry.dll.decode("utf-8") for entry in pe.DIRECTORY_ENTRY_IMPORT
            ]
            if not current_py_dll in loaded_dlls:
                raise RuntimeError(
                    f"Python version mismatch detected, aborting ArcPy syncing. This package requires Python {sys.version_info.major}.{sys.version_info.minor}."
                )

        linked = False
        for dll in build_provided_dlls:
            source_path = source_site_packages / dll
            dest_path = dest_site_packages / dll
            if source_path == dest_path:
                continue

            if source_path.exists():
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except PermissionError:
                        pass
                #try:
                #    dest_path.symlink_to(source_path)
                #    linked = True
                #except OSError:
                #    pass

                if not linked:
                    try:
                        copyfile(source_path, dest_path)
                    except OSError:
                        raise RuntimeError("Unable to copy ArcPy development files.")

        if linked:
            print(
                "Symbolically linked the ArcPy DLLs. This will keep your environment synced with the build automatically."
            )
        else:
            print(
                "Copied ArcPy DLLs, these files will need to be manually kept in sync with the build. To automatically link the DLLs, start a Python Command Prompt session as an administrator before creating the environment."
            )
