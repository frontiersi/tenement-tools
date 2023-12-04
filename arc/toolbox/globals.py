
import os



tbx_filename = os.path.realpath(__file__)
tbx_folder = os.path.dirname(tbx_filename)
folder = os.path.dirname(tbx_folder)
# globals (non-dev)
FOLDER_MODULES = os.path.join(folder, 'modules')
FOLDER_SHARED = os.path.join(folder, 'shared')
GRP_LYR_FILE = os.path.join(folder, r'arc\lyr\group_template.lyrx')

# globals (dev)
STAC_ENDPOINT_ODC = 'https://explorer.sandbox.dea.ga.gov.au/stac'
STAC_ENDPOINT_LEG = 'https://explorer.sandbox.dea.ga.gov.au/stac/search'
RESULT_LIMIT = 20
# FOLDER_MODULES = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\modules'
# FOLDER_SHARED = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\shared'
# GRP_LYR_FILE = r'C:\Users\Lewis\Documents\GitHub\tenement-tools\arc\lyr\group_template.lyrx'

# globals (tool parameters)
GDVSPECTRA_THRESHOLD = {}
GDVSPECTRA_TREND = {}
GDVSPECTRA_CVA = {}
PHENOLOPY_METRICS = {}
NICHER_MASKER = {}
VEGFRAX_FRACTIONAL_COVER = {}
ENSEMBLE_SIGMOIDER = {}
ENSEMBLE_MASKER = {}
NRT_CREATE_AREA = {}
NRT_MODIFY_AREA = {}
NRT_DELETE_AREA = {}
NRT_VISUALISE_AREA = {}