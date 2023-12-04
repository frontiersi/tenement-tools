#import arcpy
import importlib

def import_checks(modules):
    """checks a list of modules to import, modules are in string format"""
    errs = {}
    for item in modules:
        try:
            importlib.import_module(item)
        except ImportError as e:
            errs[item] = e
    return errs


def local_import_checks(modules):
    pass