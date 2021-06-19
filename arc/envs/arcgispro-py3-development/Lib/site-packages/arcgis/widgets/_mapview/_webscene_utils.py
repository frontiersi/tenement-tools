DEFAULT_WEBSCENE_TEXT_PROPERTY = \
  {'operationalLayers': [],
      'baseMap': {
          'id': 'basemap',
          'title': 'Topographic',
           'baseMapLayers': [
               {'url': 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer',
                'id': 'worldTopoBase',
                'layerType': 'ArcGISTiledMapServiceLayer',
                'title': 'Topo'}],
          'elevationLayers': [
              {'url': 'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer',
               'id': 'globalElevation',
               'listMode': 'hide',
               'layerType': 'ArcGISTiledElevationServiceLayer',
               'title': 'Elevation'}]},
        'ground': {
           'layers': [
               {'url': 'https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer',
                'id': 'globalElevation',
                'listMode': 'hide',
                'layerType': 'ArcGISTiledElevationServiceLayer',
                'title': 'Elevation'}]},
        'viewingMode': 'global',
        'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
        'version': '1.11',
        'authoringApp': 'PortalMycontentCreate',
        'authoringAppVersion': '6.2.0.0'}
