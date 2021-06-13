__version__ = '1.8.5'

from . import env
from . import (features, geoanalytics, geocoding, geometry)
from . import (geoprocessing, network, raster, realtime, schematics)
import os
if os.environ.get('DISABLE_ARCGIS_LEARN', None) is '1':
    from . import (mapping, apps)
else:
    from . import (mapping, apps, learn)

from arcgis.gis import GIS
from .features.analysis import *
from .geocoding import geocode

try:
    import pandas as pd
    from .features.geo import GeoAccessor, GeoSeriesAccessor
except ImportError as e:
    pass

__all__ = ['GIS', 'geocode', 'features',
           'geoanalytics', 'geocoding', 'geometry',
           'geoprocessing', 'network', 'raster', 'apps',
           'realtime', 'schematics', 'mapping', 'learn',
           'aggregate_points',
           'calculate_density',
           'connect_origins_to_destinations',
           'create_buffers',
           'create_drive_time_areas',
           'create_route_layers',
           'create_viewshed',
           'create_watersheds',
           'derive_new_locations',
           'dissolve_boundaries',
           'enrich_layer',
           'extract_data',
           'find_existing_locations',
           'find_hot_spots',
           'find_nearest',
           'find_similar_locations',
           'find_centroids',
           'interpolate_points',
           'join_features',
           'merge_layers',
           'overlay_layers',
           'plan_routes',
           'summarize_nearby',
           'summarize_center_and_dispersion',
           'summarize_within',
           'trace_downstream']

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'widgets/js/dist',
        'dest': 'arcgis',
        'require': 'arcgis/extension'
    }]

