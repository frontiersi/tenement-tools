import arcpy
import os, sys

from arc.toolbox.import_checks import import_checks, local_import_checks
import pathlib
location = pathlib.Path(__file__).parent.resolve()
os.environ["PATH"] += os.pathsep + str(location)

from arc.toolbox.COG.COG_Fetch_ODC import COG_Fetch_ODC
from arc.toolbox.COG.COG_Shift import COG_Shift
from arc.toolbox.COG.COG_Explore import COG_Explore
from arc.toolbox.GDVSpectra.GDVSpectra_Likelihood import GDVSpectra_Likelihood
from arc.toolbox.GDVSpectra.GDVSpectra_Threshold import GDVSpectra_Threshold
from arc.toolbox.GDVSpectra.GDVSpectra_Trend import GDVSpectra_Trend
from arc.toolbox.GDVSpectra.GDVSpectra_CVA import GDVSpectra_CVA
from arc.toolbox.Phenolopy.Phenolopy_Metrics import Phenolopy_Metrics
from arc.toolbox.Nicher.Nicher_Masker import Nicher_Masker
from arc.toolbox.Nicher.Nicher_SDM import Nicher_SDM
from arc.toolbox.VegFrax.VegFrax_Fractional_Cover import VegFrax_Fractional_Cover
from arc.toolbox.Ensemble.Ensemble_Masker import Ensemble_Masker
from arc.toolbox.Ensemble.Ensemble_Sigmoider import Ensemble_Sigmoider
from arc.toolbox.Ensemble.Ensemble_Model import Ensemble_Model
from arc.toolbox.NRT.NRT_Build_Graphs import NRT_Build_Graphs
from arc.toolbox.NRT.NRT_Monitor_Areas import NRT_Monitor_Areas
from arc.toolbox.NRT.NRT_Visualise_Areas import NRT_Visualise_Areas
from arc.toolbox.NRT.NRT_Create_Monitoring_Areas import NRT_Create_Monitoring_Areas
from arc.toolbox.NRT.NRT_Delete_Monitoring_Areas import NRT_Delete_Monitoring_Areas
from arc.toolbox.NRT.NRT_Modify_Monitoring_Areas import NRT_Modify_Monitoring_Areas
from arc.toolbox.NRT.NRT_Create_Project import NRT_Create_Project


class Toolbox:

    def __init__(self, modules=None, local_modules=None):
        """
        Define the toolbox.
        """

        # set name of toolbox
        self.label = "Toolbox"
        self.alias = "toolbox"

        if modules is None:
            modules = ['rasterio', 'dask' 'pandas', 'xarray', 'numpy', 'affine', 'pyproj',
                       ]
        errs = import_checks(modules)
        self.add_module_errs(errs)
        if local_modules is None:
            local_modules = []
        errs = local_import_checks(local_modules)
        self.add_module_errs(errs)

        # tools to display in toolbox
        self.tools = [
            COG_Fetch_ODC,
            COG_Shift,
            COG_Explore,
            GDVSpectra_Likelihood,
            GDVSpectra_Threshold,
            GDVSpectra_Trend,
            GDVSpectra_CVA,
            Phenolopy_Metrics,
            Nicher_SDM,
            Nicher_Masker,
            VegFrax_Fractional_Cover,
            Ensemble_Sigmoider,
            Ensemble_Model,
            Ensemble_Masker,
            NRT_Create_Project,
            NRT_Create_Monitoring_Areas,
            NRT_Modify_Monitoring_Areas,
            NRT_Delete_Monitoring_Areas,
            NRT_Monitor_Areas,
            NRT_Visualise_Areas,
            NRT_Build_Graphs
        ]

    @staticmethod
    def add_module_errs(errs):
        """dict of errs added to arcpy error log"""
        if isinstance(errs, dict):
            for m, e in errs.items():
                arcpy.AddError(f'Python library {m} not installed')
                arcpy.AddMessage(str(e))