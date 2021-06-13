from __future__ import absolute_import

import collections
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from re import search
from uuid import uuid4
import datetime

import arcgis.features
import arcgis.gis
import arcgis.env
from warnings import warn
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._utils import _date_handler
from arcgis.geometry import SpatialReference, Polygon
from arcgis.gis import Layer, _GISResource, Item
from arcgis.mapping._basemap_definitions import basemap_dict
from arcgis.mapping._scenelyrs import SceneLayer
try:
    from traitlets import HasTraits, observe
    from arcgis.widgets._mapview._traitlets_extension import ObservableDict
except ImportError:
    class HasTraits:
        pass
    class ObservableDict(dict):
        def tag(*args, **kwargs):
            pass
    def observe(_=None, *args, **kwargs):
        return observe


_log = logging.getLogger(__name__)
###########################################################################
@contextmanager
def _tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write((bytes(data, 'UTF-8')))
    temp.close()
    yield temp.name
    os.unlink(temp.name)

###########################################################################
class _ApplicationProperties(object):
    """
    This class is responsible for containing the viewing and editing
    properties of the web map. There are specific objects within this
    object that are applicable only to Collector and Offline Mapping.
    """
    _app_prop = None
    def __init__(self, prop=None):
        template = {
            "viewing": {},
            "offline" : {},
            "editing" : {}
        }
        if prop and isinstance(prop, (dict, PropertyMap)):
            self._app_prop = PropertyMap(dict(prop))
        else:
            self._app_prop = PropertyMap(template)
    @property
    def properties(self):
        """represents the application properties"""
        return self._app_prop
    #----------------------------------------------------------------------
    def __repr__(self):
        return json.dumps(dict(self._app_prop))
    #----------------------------------------------------------------------
    def __str__(self):
        return json.dumps(dict(self._app_prop))
    #----------------------------------------------------------------------
    @property
    def location_tracking(self):
        """gets the location_tracking value"""
        if 'editing' in self._app_prop and \
           'locationTracking' in self._app_prop['editing']:
            return self._app_prop['editing']['locationTracking']
        else:
            self._app_prop['editing']['locationTracking'] = {
                "enabled": False
            }
            return self._app_prop['editing']['locationTracking']
    #----------------------------------------------------------------------


###########################################################################
class WebMap(HasTraits, collections.OrderedDict):
    """
    Represents a web map and provides access to its basemaps and operational layers as well
    as functionality to visualize and interact with them.

    An ArcGIS web map is an interactive display of geographic information that you can use to tell stories and answer
    questions. Maps contain a basemap over which a set of data layers called operational layers are drawn. To learn
    more about web maps, refer: https://doc.arcgis.com/en/arcgis-online/reference/what-is-web-map.htm

    Web maps can be used across ArcGIS apps because they adhere to the same web map specification. This means you can
    author web maps in one ArcGIS app (including the Python API) and view and modify them in another. To learn more
    about the web map specification, refer: https://developers.arcgis.com/web-map-specification/

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    webmapitem             Optional Item object whose Item.type is 'Web Map'. If not specified
                           an empty WebMap object is created with some useful defaults.
    ==================     ====================================================================

    .. code-block:: python

            # USAGE EXAMPLE 1: Creating a WebMap object from an existing web map item

            from arcgis.mapping import WebMap
            from arcgis.gis import GIS

            # connect to your GIS and get the web map item
            gis = GIS(url, username, password)
            wm_item = gis.content.get('1234abcd_web map item id')

            # create a WebMap object from the existing web map item
            wm = WebMap(wm_item)
            type(wm)
            >> arcgis.mapping._types.WebMap

            # explore the layers in this web map using the 'layers' property
            wm.layers
            >> [{}...{}]  # returns a list of dictionaries representing each operational layer

    .. code-block:: python
            # USAGE EXAMPLE 2: Creating a new WebMap object

            from arcgis.mapping import WebMap

            # create a new WebMap object
            wm = WebMap()
            type(wm)
            >> arcgis.mapping._types.WebMap

            # explore the layers in this web map using the 'layers' property
            wm.layers
            >> []  # returns an empty list. You can add layers using the `add_layer()` method
    """

    _webmapdict = ObservableDict({}).tag(sync=True)
    @observe('_webmapdict')
    def _webmapdict_changed(self, change):
        try:
            self._mapview._webmap = {}
            self._mapview._webmap = change['new']
        except Exception:
            pass

    def __init__(self, webmapitem=None):
        """
        Constructs an empty WebMap object. If an web map Item is passed, constructs a WebMap object from item on
        ArcGIS Online or Enterprise.
        """

        #Dashboard items.
        self._id = str(uuid4())
        self.type = "mapWidget"

        self._pop_ups = False
        self._navigation = False
        self._scale_bar = "none"
        self._bookmarks = False
        self._legend = False
        self._layer_visibility = False
        self._basemap_switcher = False
        self._search = False
        self._zoom = False
        self._events = Events._create_events()

        self._height = 1
        self._width = 1
        #Dashboard items end here.

        from arcgis.widgets import MapView

        if webmapitem:
            if webmapitem.type.lower() != 'web map':
                raise TypeError("item type must be web map")
            self.item = webmapitem
            self._gis = webmapitem._gis
            self._con = self._gis._con
            self._webmapdict = self.item.get_data()
            pmap = PropertyMap(self._webmapdict)
            self.definition = pmap
            self._layers = None
            self._tables = None
            self._basemap = self._webmapdict["baseMap"]
            self._extent = self.item.extent
        else:
            #default spatial ref for current web map
            self._default_spatial_reference = {'wkid': 102100,
                                               'latestWkid': 3857}

            #pump in a simple, default webmap dict - no layers yet, just basemap
            self._basemap = {
                'baseMapLayers':[{'id':'defaultBasemap',
                                  'layerType':'ArcGISTiledMapServiceLayer',
                                              'url':'https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer',
                                              'visibility':True,
                                              'opacity':1,
                                              'title':'World Topographic Map'
                                              }],
                'title':'Topographic'
            }
            self._gallery_basemaps = {}
            self._webmapdict = {'operationalLayers': [],
                                'baseMap':self._basemap,
                                'spatialReference':self._default_spatial_reference,
                                'version':'2.10',
            'authoringApp': 'ArcGISPythonAPI',
            'authoringAppVersion': str(arcgis.__version__),
            }
            pmap = PropertyMap(self._webmapdict)
            self.definition = pmap
            self._gis = arcgis.env.active_gis
            if self._gis: #you can also have a case where there is no GIS obj
                self._con = self._gis._con
                if self._gis.properties["defaultBasemap"]:
                    self._basemap = self._gis.properties["defaultBasemap"]
                    self._webmapdict['baseMap'] = self._basemap
            else:
                self._con = None
            self.item = None
            self._layers = []
            self._tables = []
            self._extent = []

        # Set up map widget to use in jupyter env: have changes made to
        # self._webmapdict get passed to the widge to render
        self._mapview = MapView(gis=self._gis, item=self)
        self._mapview.hide_mode_switch = True
        self._mapview._webmap = {}
        self._mapview._webmap = self._webmapdict
        rotation = self._webmapdict.get('initialState', {})\
                   .get('viewpoint', {})\
                   .get('rotation', 0)
        self._mapview.rotation = rotation

    @property
    def events(self):
        """
        :return: list of events attached to the widget.
        """
        return self._events

    def _ipython_display_(self, *args, **kwargs):
       return self._mapview._ipython_display_(*args, **kwargs)

    def __repr__(self):
        """
        Hidden, to enhance how the object is represented when you simply query it in non Jupyter envs.
        :return:
        """
        try:
            return 'WebMap at ' + self.item._portal.url  + "/home/webmap/viewer.html?webmap=" + self.item.itemid
        except Exception:
            return super().__repr__()

    def __str__(self):
        return json.dumps(self, default=_date_handler)

    def add_table(self, table, options=None):
        """
        Adds the given layer to the WebMap.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        table                  Required object. You can add:

                                   - Table objects
        ------------------     --------------------------------------------------------------------
        options                Optional dict. Specify properties such as title, symbol, opacity, visibility, renderer
                               for the table that is added. If not specified, appropriate defaults are applied.
        ==================     ====================================================================

        .. code-block:: python

            wm = WebMap()
            table = Table('https://some-url.com/')
            wm.add_layer(table)

        :return:
            True if table was successfully added. Else, raises appropriate exception.
        """
        if not isinstance(table, arcgis.features.Table):
            raise Exception("Type of object passed in must of type 'Table'")
        self.add_layer(table, options)

    def add_layer(self, layer, options=None):
        """
        Adds the given layer to the WebMap.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        layer                  Required object. You can add:

                                   - Layer objects such as FeatureLayer, MapImageLayer, ImageryLayer etc.
                                   - Item objects and FeatureSet and FeatureCollections
                                   - Table objects
        ------------------     --------------------------------------------------------------------
        options                Optional dict. Specify properties such as title, symbol, opacity, visibility, renderer
                               for the layer that is added. If not specified, appropriate defaults are applied.
        ==================     ====================================================================

        .. code-block:: python

           # USAGE EXAMPLE: Add feature layer and map image layer item objects to the WebMap object.

           crime_fl_item = gis.content.search("2012 crime")[0]
           streets_item = gis.content.search("LA Streets","Map Service")[0]

           wm = WebMap()  # create an empty web map with a default basemap
           wm.add_layer(streets_item)
           >> True

           # Add crime layer, but customize the title, transparency and turn off the default visibility.
           wm.add_layer(fl_item, {'title':'2012 crime in LA city',
                                  'opacity':0.5,
                                  'visibility':False})
            >> True

        :return:
            True if layer was successfully added. Else, raises appropriate exception.
        """
        from arcgis.mapping.ogc._base import BaseOGC
        from arcgis.mapping.ogc import WMSLayer, WMTSLayer
        if options is None:
            options = {}
        if isinstance(layer, arcgis.features.FeatureLayer) and \
           'renderer' not in options and not isinstance(layer, arcgis.features.Table):
            options['renderer'] = json.loads(layer.renderer.json)
        elif hasattr(layer, 'spatial'):
            layer = layer.spatial.to_feature_collection()
        # region extact basic info from options
        title = options['title'] if options and 'title' in options else None
        opacity = options['opacity'] if options and 'opacity' in options else 1
        visibility = options['visibility'] if options and 'visibility' in options else True
        layer_spatial_ref = options['spatialReference'] if options and 'spatialReference' in options else None
        popup = options['popup'] if options and 'popup' in options else None  # from draw method
        item_id = None
        # endregion

        # region extract rendering info from options
        # info for feature layers
        definition_expression = options['definition_expression'] if options and 'definition_expression' in options else None
        renderer = options['renderer'] if options and 'renderer' in options else None
        renderer_field = options['field_name'] if options and 'field_name' in options else None
        self._extent = options['extent'] if options and 'extent' in options else self._extent #from map widget
        fset_symbol = options['symbol'] if options and 'symbol' in options else None  # from draw method

        # info for raster layers
        image_service_parameters = options['imageServiceParameters'] \
            if options and 'imageServiceParameters' in options else None

        # endregion

        # region infer layer type
        layer_type = None
        if isinstance(layer, Layer) or isinstance(layer, arcgis.features.FeatureSet):
            if hasattr(layer, 'properties'):
                if hasattr(layer.properties, 'name'):
                    title = layer.properties.name if title is None else title

                # find layer type
                if (isinstance(layer, arcgis.features.FeatureLayer) or \
                    isinstance(layer, arcgis.features.FeatureCollection) or \
                    isinstance(layer, arcgis.features.FeatureSet)):
                    # Can be either a FeatureLayer or a table: figure it out
                    if isinstance(layer, arcgis.features.Table):
                        layer_type = 'Table'
                    else:
                        layer_type = 'ArcGISFeatureLayer'
                elif isinstance(layer, arcgis.raster.ImageryLayer):
                    layer_type='ArcGISImageServiceLayer'
                    #todo : get renderer info

                elif isinstance(layer, arcgis.mapping.MapImageLayer):
                    layer_type='ArcGISMapServiceLayer'
                elif isinstance(layer, arcgis.mapping.VectorTileLayer):
                    layer_type='VectorTileLayer'
                elif isinstance(layer, arcgis.realtime.StreamLayer):
                    layer_type='ArcGISStreamLayer'

                if hasattr(layer.properties, 'serviceItemId'):
                    item_id = layer.properties.serviceItemId
            elif isinstance(layer, arcgis.features.FeatureSet):
                layer_type = 'ArcGISFeatureLayer'
        elif isinstance(layer, arcgis.gis.Item):
            # set the item's extent
            if not self._extent:
                self._extent = layer.extent
            if layer.type.lower() == "map service":
                layer_type = "ArcGISMapServiceLayer"
                item_id = layer.id
            else:
              if not any(hasattr(layer, attr) for attr in ["layers", "tables"]):
                  raise TypeError("Item object without layers or tables is not supported")
              elif hasattr(layer, 'layers'):
                  if layer.type == 'Feature Collection':
                      options['serviceItemId'] = layer.itemid
                  for lyr in layer.layers:  # recurse - works for all.
                      self.add_layer(lyr, options)
              if hasattr(layer, 'tables'):
                  for tbl in layer.tables:  # recurse - works for all.
                      self.add_table(tbl, options)
              return True  # end add_layer execution after iterating through each layer.
        elif isinstance(layer, arcgis.features.FeatureLayerCollection):
            if not self._extent:
                if hasattr(layer.properties, 'fullExtent'):
                    self._extent = layer.properties.fullExtent
            if not any(hasattr(layer, attr) for attr in ["layers", "tables"]):
                raise TypeError("FeatureLayerCollection object without layers or tables is not supported")
            if hasattr(layer, 'layers'):
                for lyr in layer.layers:  # recurse - works for all.
                    self.add_layer(lyr, options)
            if hasattr(layer, 'tables'):
                for tbl in layer.tables:  # recurse - works for all.
                    self.add_table(tbl, options)
            return True
        elif isinstance(layer, BaseOGC):
            lyr = layer._lyr_json
            title = lyr["title"]
            opacity = lyr["opacity"]
            id = lyr["id"]
            layer_type = layer._type
        else:
            raise TypeError("Input layer should either be a Layer object or an Item object. To know the supported layer types, refer" +
                            'to https://developers.arcgis.com/web-map-specification/objects/operationalLayers/')
        # endregion

        # region create the new layer dict in memory
        #dvitale: add ability to specify layer id
        layer_id = options['layerId'] if options and 'layerId' in options else uuid4().__str__()
        #end specify layer id section
        new_layer = {'title':title,
                     'opacity':opacity,
                     'visibility':visibility,
                     'id':layer_id}

        # if renderer info is available, then write layer definition
        layer_definition = {'definitionExpression':definition_expression}

        if renderer:
            layer_definition['drawingInfo'] = {'renderer':renderer}
        new_layer['layerDefinition'] = layer_definition

        if layer_type:
            new_layer['layerType'] = layer_type

        if item_id:
            new_layer['itemId'] = item_id

        if hasattr(layer, 'url'):
            new_layer['url'] = layer.url
        elif isinstance(layer, arcgis.features.FeatureCollection):  # feature collection item on web GIS
            if 'serviceItemId' in options:
                # if ItemId is found, then type is fc and insert item id. Else, leave the type as ArcGISFeatureLayer
                new_layer['type'] = "Feature Collection"
                new_layer['itemId'] = options['serviceItemId']
            elif hasattr(layer, 'properties'):
                if hasattr(layer.properties, 'layerDefinition'):
                    if hasattr(layer.properties.layerDefinition, 'serviceItemId'):
                        new_layer['type'] = 'Feature Collection'   # if ItemId is found, then type is fc and insert item id
                        new_layer['itemId'] = layer.properties.layerDefinition.serviceItemId
                elif hasattr(layer, "layer"):
                    if hasattr(layer.layer, "layers"):
                        if hasattr(layer.layer.layers[0], "layerDefinition"):
                            if hasattr(layer.layer.layers[0].layerDefinition, 'serviceItemId'):
                                new_layer['type'] = 'Feature Collection'  # if ItemId is found, then type is fc and insert item id
                                new_layer['itemId'] = layer.layer.layers[0].layerDefinition.serviceItemId

        if layer_type == 'ArcGISImageServiceLayer':
            #find if raster functions are available
            if 'options' in layer._lyr_json:
                if isinstance(layer._lyr_json['options'], str): #sometimes the rendering info is a string
                    #load json
                    layer_options = json.loads(layer._lyr_json['options'])
                else:
                    layer_options = layer._lyr_json['options']

                if 'imageServiceParameters' in layer_options:
                    #get renderingRule and mosaicRule
                    new_layer.update(layer_options['imageServiceParameters'])

            #if custom rendering rule is passed, then overwrite this
            if image_service_parameters:
                new_layer['renderingRule'] = image_service_parameters['renderingRule']

        # inmem FeatureCollection
        if isinstance(layer, arcgis.features.FeatureCollection):
            if hasattr(layer, "layer"):
                if hasattr(layer.layer, "layers"):
                    fc_layer_definition = dict(layer.layer.layers[0].layerDefinition)
                    fc_feature_set = dict(layer.layer.layers[0].featureSet)
                else:
                    fc_layer_definition = dict(layer.layer.layerDefinition)
                    fc_feature_set = dict(layer.layer.featureSet)
            else:
                fc_layer_definition = dict(layer.properties.layerDefinition)
                fc_feature_set = dict(layer.properties.featureSet)

            if 'title' not in fc_layer_definition:
                fc_layer_definition['title'] = title

            new_layer['featureCollection'] = {'layers':
                                              [{'featureSet': fc_feature_set,
                                                'layerDefinition': fc_layer_definition}
                                               ]}

        # inmem FeatureSets - typically those which users pass to the `MapView.draw()` method
        if isinstance(layer, arcgis.features.FeatureSet):
            if not layer_spatial_ref:
                if hasattr(layer, 'spatial_reference'):
                    layer_spatial_ref = layer.spatial_reference
                else:
                    layer_spatial_ref = self._default_spatial_reference

            if 'spatialReference' not in layer.features[0].geometry: # webmap seems to need spatialref for each geometry
                for feature in layer:
                    feature.geometry['spatialReference'] = layer_spatial_ref

            fset_dict = layer.to_dict()
            fc_layer_definition = {'geometryType':fset_dict['geometryType'],
                                   'fields':fset_dict['fields'],
                                   'objectIdField':layer.object_id_field_name,
                                   'type':'Feature Layer',
                                   'spatialReference':layer_spatial_ref,
                                   'name':title}

            #region set up default symbols if one is not available.
            if not fset_symbol:
                if fc_layer_definition['geometryType'] == 'esriGeometryPolyline':
                    fset_symbol = {"color":[0,0,0,255],
                                   "width": 1.33,
                                   "type": "esriSLS",
                                   "style": "esriSLSSolid"}
                elif fc_layer_definition['geometryType'] in ['esriGeometryPolygon','esriGeometryEnvelope']:
                    fset_symbol={"color": [0,0,0,64],
                                 "outline": {
                                     "color": [0,0,0,255],
                                     "width": 1.33,
                                     "type": "esriSLS",
                                     "style": "esriSLSSolid"},
                                 "type": "esriSFS",
                                 "style": "esriSFSSolid"}
                elif fc_layer_definition['geometryType'] in ['esriGeometryPoint', 'esriGeometryMultipoint']:
                    fset_symbol={"angle": 0,
                                 "xoffset": 0,
                                 "yoffset": 12,
                                 "type": "esriPMS",
                                 "url": "https://esri.github.io/arcgis-python-api/notebooks/nbimages/pink.png",
                                 "contentType": "image/png",
                                 "width": 24,
                                 "height": 24}
            #endregion
            #insert symbol into the layerDefinition of featureCollection - pro style
            if renderer:
                fc_layer_definition['drawingInfo'] = {'renderer':renderer}
            else: #use simple, default renderer
                fc_layer_definition['drawingInfo'] = {'renderer': {
                    'type':'simple',
                    'symbol':fset_symbol
                }
                                                      }

            new_layer['featureCollection'] = {
                'layers':[
                    {'featureSet':{'geometryType':fset_dict['geometryType'],
                                   'features':fset_dict['features']},
                     'layerDefinition':fc_layer_definition
                     }]
            }
        #endregion

        #region
        if isinstance(layer, BaseOGC):
            new_layer = {**new_layer, **layer._operational_layer_json}
        #endregion

        # region Process popup info
        if layer_type in ['ArcGISFeatureLayer',
                          'ArcGISImageServiceLayer',
                          'Feature Collection',
                          'Table']:  # supports popup
            popup = {'title': title,
                     'fieldInfos': [],
                     'description': None,
                     'showAttachments': True,
                     'mediaInfos': []}

            fields_list = []
            if isinstance(layer, arcgis.features.FeatureLayer) or isinstance(layer, arcgis.raster.ImageryLayer):
                if hasattr(layer.properties, 'fields'):
                    fields_list = layer.properties.fields
            elif isinstance(layer, arcgis.features.FeatureSet):
                if hasattr(layer, 'fields'):
                    fields_list = layer.fields
            elif isinstance(layer, arcgis.features.FeatureCollection):
                if hasattr(layer.properties, 'layerDefinition'):
                    if hasattr(layer.properties.layerDefinition, 'fields'):
                        fields_list = layer.properties.layerDefinition.fields

            for f in fields_list:
                if isinstance(f, dict) or isinstance(f, PropertyMap):
                    field_dict = {'fieldName': f['name'],
                                  'label': f['alias'] if 'alias' in f else f['name'],
                                  'isEditable': f['editable'] if 'editable' in f else True,
                                  'visible': True}
                elif isinstance(f, str):  # some layers are saved with fields that are just a list of strings
                    field_dict = {'fieldName': f,
                                  'label': f,
                                  'isEditable': True,
                                  'visible': True}
                if field_dict:
                    popup['fieldInfos'].append(field_dict)
        else:
            popup = None

        if popup:
            if isinstance(layer, arcgis.features.FeatureLayer) or isinstance(layer, arcgis.raster.ImageryLayer):
                new_layer['popupInfo'] = popup
            elif isinstance(layer, arcgis.features.FeatureSet) or isinstance(layer, arcgis.features.FeatureCollection):
                new_layer['featureCollection']['layers'][0]['popupInfo'] = popup

        # endregion

        # region sort layers into 'operationalLayers' or 'tables'
        if isinstance(layer, arcgis.features.Table):
            if 'tables' not in self._webmapdict.keys():
                # There are no tables yet, create one here
                self._webmapdict['tables'] = [new_layer]
                self.definition = PropertyMap(self._webmapdict)
            else:
                # There are tables, just append to it
                self._webmapdict['tables'].append(new_layer)
                self.definition = PropertyMap(self._webmapdict)
        else:
            if 'operationalLayers' not in self._webmapdict.keys():
                # there no layers yet, create one here
                self._webmapdict['operationalLayers'] = [new_layer]
                self.definition = PropertyMap(self._webmapdict)
            else:
                # there are operational layers, just append to it
                self._webmapdict['operationalLayers'].append(new_layer)
                self.definition = (PropertyMap(self._webmapdict))
        # endregion

        # update layers property
        if not self._layers:
            if 'operationalLayers' in self._webmapdict:
                self._layers = []
                for l in self._webmapdict['operationalLayers']:
                    self._layers.append(PropertyMap(l))
                # reverse the layer list - webmap viewer reverses the list always
                self._layers.reverse()
        else:
            # note - no need to add if self._layers was empty as the hydration step above will account for the new layer
            # need this check to avoid duplicating adding a new table to both layers and tables
            if "layerType" in new_layer and new_layer["layerType"] != "Table":
                self._layers.append(PropertyMap(new_layer))

        # update tables property
        if not self._tables:
            self._tables = []
            if 'tables' in self._webmapdict:
                for t in self._webmapdict['tables']:
                    self._tables.append(PropertyMap(t))
            # reverse the layer list - webmap viewer reverses the list always
            self._tables.reverse()
        else:
            # note - no need to add if self._layers was empty as the hydration step above will account for the new layer
            self._tables.append(PropertyMap(new_layer))

        return True

    def _process_extent(self, extent=None):
        """
        internal method to transform extent to a string of xmin, ymin, xmax, ymax
        If extent is not in wgs84, it projects
        :return:
        """
        if extent is None:
            extent = self._extent
        if isinstance(extent, PropertyMap):
            extent = dict(extent)
        if isinstance(extent, list):
            #passed from Item's extent flatten the extent. Item's extent is always in 4326, no need to project
            extent_list = [element for sublist in extent for element in sublist]

            #convert to string
            return ','.join(str(e) for e in extent_list)
        elif isinstance(extent, dict):
            #passed from MapView.extent
            if 'spatialReference' in extent:
                if 'latestWkid' in extent['spatialReference']:
                    if extent['spatialReference']['latestWkid'] != 4326:
                        #use geometry service to project
                        input_geom = [{'x':extent['xmin'], 'y':extent['ymin']},
                                      {'x':extent['xmax'], 'y':extent['ymax']}]

                        result = arcgis.geometry.project(input_geom,
                                                         in_sr=extent['spatialReference']['latestWkid'],
                                                         out_sr=4326)

                        #process and return the result
                        if self._contains_nans(result):
                            return ""
                        else:
                            e = [result[0]['x'],result[0]['y'],result[1]['x'],result[1]['y']]
                            return ','.join(str(i) for i in e)

            #case when there is no spatialReference. Then simply extract the extent
            if 'xmin' in extent:
                e = extent
                e= [e['xmin'], e['ymin'],e['xmax'],e['ymax']]
                return ','.join(str(i) for i in e)

        #if I don't know how to process the extent.
        return extent

    def _contains_nans(self, result):
        """a bool of if projection output `result` contains any NaNs"""
        for value in result:
            if "nan" in str(value['x']).lower():
                return True
            if "nan" in str(value['y']).lower():
                return True
        return False

    def save(self, item_properties, thumbnail=None, metadata=None, owner=None, folder=None):
        """
        Save the WebMap object into a new web map Item in your GIS.

        .. note::
            If you started out with a fresh WebMap object, use this method to save it as a the web map item in your GIS.

            If you started with a WebMap object from an existing web map item, calling this method will create a new item
            with your changes. If you want to update the existing web map item with your changes, call the `update()`
            method instead.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder into which the web map should be
                            saved.
        ===============     ====================================================================

        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        extent             Optional dict, string, or array. The extent of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
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

        The above are the most common item properties (metadata) that you set. To get a complete list, see
        https://developers.arcgis.com/rest/users-groups-and-items/common-parameters.htm#ESRI_SECTION1_1FFBA7FE775B4BDA8D97524A6B9F7C98

        :return:
            Item object corresponding to the new web map Item created.

        .. code-block:: python

            # USAGE EXAMPLE 1: Save a WebMap object into a new web map item
            from arcgis.gis import GIS
            from arcgis.mapping import WebMap

            # log into your GIS
            gis = GIS(url, username, password)

            # compose web map by adding, removing, editing layers and basemaps
            wm = WebMap()  # new web map
            wm.add_layer(...)  # add some layers

            # save the web map
            webmap_item_properties = {'title':'Ebola incidents and facilities',
                         'snippet':'Map created using Python API showing locations of Ebola treatment centers',
                         'tags':['automation', 'ebola', 'world health', 'python'],
                         'extent': {'xmin': -122.68, 'ymin': 45.53, 'xmax': -122.45, 'ymax': 45.6, 'spatialReference': {'wkid': 4326}}}

            new_wm_item = wm.save(webmap_item_properties, thumbnail='./webmap_thumbnail.png')

            # to visit the web map using a browser
            print(new_wm_item.homepage)
            >> 'https://your portal url.com/webadaptor/item.html?id=1234abcd...'

        """

        item_properties['type'] = 'Web Map'
        item_properties['extent'] = self._process_extent(item_properties.get('extent', None))
        item_properties['text'] = json.dumps(self._webmapdict, default=_date_handler)
        if 'typeKeywords' not in item_properties:
            item_properties['typeKeywords'] = self._eval_map_viewer_keywords()

        if 'title' not in item_properties or 'snippet' not in item_properties or 'tags' not in item_properties:
            raise RuntimeError("title, snippet and tags are required in item_properties dictionary")

        new_item = self._gis.content.add(item_properties, thumbnail=thumbnail, metadata=metadata, owner=owner,
                                         folder=folder)
        if not hasattr(self, 'item'):
            self.item = new_item

        return new_item

    def update(self, item_properties=None, thumbnail=None, metadata=None):
        """
        Updates the web map item in your GIS with the changes you made to the WebMap object. In addition, you can update
        other item properties, thumbnail and metadata.

        .. note::
            If you started with a WebMap object from an existing web map item, calling this method will update the item
            with your changes.

            If you started out with a fresh WebMap object (without a web map item), calling this method will raise a
            RuntimeError exception. If you want to save the WebMap object into a new web map item, call the `save()`
            method instead.

            For item_properties, pass in arguments for the properties you want to be updated.
            All other properties will be untouched.  For example, if you want to update only the
            item's description, then only provide the description argument in item_properties.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Optional dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ===============     ====================================================================

        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
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

        The above are the most common item properties (metadata) that you set. To get a complete list, see
        https://developers.arcgis.com/rest/users-groups-and-items/common-parameters.htm#ESRI_SECTION1_1FFBA7FE775B4BDA8D97524A6B9F7C98

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # USAGE EXAMPLE 1: Update an existing web map

            from arcgis.gis import GIS
            from arcgis.mapping import WebMap

            # log into your GIS
            gis = GIS(url, username, password)

            # edit web map by adding, removing, editing layers and basemaps
            wm = WebMap()  # new web map
            wm.add_layer(...)  # add some layers

            # save the web map
            webmap_item_properties = {'title':'Ebola incidents and facilities',
                         'snippet':'Map created using Python API showing locations of Ebola treatment centers',
                         'tags':['automation', 'ebola', 'world health', 'python']}

            new_wm_item = wm.save(webmap_item_properties, thumbnail='./webmap_thumbnail.png')

            # to visit the web map using a browser
            print(new_wm_item.homepage)
            >> 'https://your portal url.com/webadaptor/item.html?id=1234abcd...'
        """

        if self.item is not None:
            if item_properties is None:
                item_properties = {}
            item_properties['text'] = json.dumps(self._webmapdict, default=_date_handler)
            item_properties['extent'] = self._process_extent()
            if 'typeKeywords' not in item_properties:
                item_properties['typeKeywords'] = self._eval_map_viewer_keywords()
            if 'type' in item_properties:
                item_properties.pop('type')  # type should not be changed.
            return self.item.update(item_properties=item_properties,
                                    thumbnail=thumbnail,
                                    metadata=metadata)
        else:
            raise RuntimeError('Item object missing, you should use `save()` method if you are creating a '
                               'new web map item')

    def _eval_map_viewer_keywords(self):
        # if user passes typeKeywords, adhere to what they have set without overriding anything
        type_keywords = set(self.item.typeKeywords) if self.item else set([])
        if 'OfflineDisabled' not in type_keywords:
            if self.layers and self._is_offline_capable_map():
                type_keywords.add("Offline")
            else:
                type_keywords.discard("Offline")
        if 'CollectorDisabled' not in type_keywords:
            if self.layers and self._is_collector_ready_map():
                type_keywords.add("Collector")
                type_keywords.add("Data Editing")
            else:
                type_keywords.discard("Collector")
                type_keywords.discard("Data Editing")
        return list(type_keywords)

    def _is_collector_ready_map(self):
        # check that one layer is an editable feature service
        for layer in self.layers:
            try:
                layer_object = arcgis.gis.Layer(url=layer.url, gis=self._gis)
                if 'ArcGISFeatureLayer' in layer.layerType:
                    if any(capability in layer_object.properties.capabilities for capability in ['Create', 'Update', 'Delete', 'Editing']):
                        return True
            except Exception:
                # Not every layer in self.layers has a URL (local featurelayer, SEDF, etc.)
                continue
        return False

    def _is_offline_capable_map(self):
        # check that feature services are sync-enabled and tiled layers are exportable
        try:
            for layer in self.layers:
                layer_object = arcgis.gis.Layer(url=layer.url, gis=self._gis)
                if 'ArcGISFeatureLayer' in layer.layerType:
                    if 'Sync' not in layer_object.properties.capabilities:
                        return False
                elif 'VectorTileLayer' in layer.layerType \
                        or 'ArcGISMapServiceLayer' in layer.layerType \
                        or 'ArcGISImageServiceLayer' in layer.layerType:
                    if not self._is_exportable(layer_object):
                        return False
                else:
                    return False
            return True
        except Exception:
            return False

    def _is_exportable(self, layer):
        # check SRs are equivalent and exportTilesAllowed is set to true or AGOl-hosted esri basemaps
        if (layer.properties['spatialReference']['wkid'] == self._webmapdict['spatialReference']['wkid']) \
                and (layer.properties['exportTilesAllowed'] or "services.arcgisonline.com" in layer.url or "server.arcgisonline.com" in layer.url):
            return True
        else:
            return False


    @property
    def tables(self):
        """
        Tables in the web map

        :return: List of Tables as dictionaries

        .. code-block:: python

            wm = WebMap()
            table = Table('https://some-url.com/')
            wm.add_layer(table)
            wm.tables
            >> [{"id": "fZvgsrA68ElmNajAZl3sOMSPG3iTnL",
                 "title": "change_table",
                 "url": "https://some-url.com/",
                 "popupInfo": {
                 ...
        """
        if self._tables is not None:
            return self._tables
        else:
            self._tables = []
            if 'tables' in self._webmapdict.keys():
                for l in self._webmapdict['tables']:
                    self._tables.append(PropertyMap(l))

            #reverse the layer list - webmap viewer reverses the list always
            self._tables.reverse()
        return self._tables

    @property
    def layers(self):
        """
        Operational layers in the web map

        :return: List of Layers as dictionaries

        .. code-block:: python

            # USAGE EXAMPLE 1: Get the list of layers from a web map

            from arcgis.mapping import WebMap
            wm = WebMap(wm_item)

            wm.layers
            >> [{"id": "Landsat8_Views_515",
            "layerType": "ArcGISImageServiceLayer",
            "url": "https://landsat2.arcgis.com/arcgis/rest/services/Landsat8_Views/ImageServer",
            ...},
            {...}]

            len(wm.layers)
            >> 2

        """
        if self._layers is not None:
            return self._layers
        else:
            self._layers = []
            if 'operationalLayers' in self._webmapdict.keys():
                for l in self._webmapdict['operationalLayers']:
                    self._layers.append(PropertyMap(l))

            #reverse the layer list - webmap viewer reverses the list always
            self._layers.reverse()
            return self._layers

    @property
    def basemap(self):
        """
        Base map layers in the web map

        :return: List of layers as dictionaries

        .. code-block:: python

            # Usage example 1: Get the basemap used in the web map

            from arcgis.mapping import WebMap
            wm = WebMap(wm_item)

            wm.basemap
            >> {"baseMapLayers": [
                {"id": "defaultBasemap",
                "layerType": "ArcGISTiledMapServiceLayer",
                "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer",
                "visibility": true,
                "opacity": 1,
                "title": "Topographic"
                }],
                "title": "Topographic"
                }

            # Usage example 2: Set the basemap used in the web map
            from arcgis.mapping import WebMap
            wm = WebMap(wm_item)

            print(wm.basemaps)
            >> ['dark-gray', 'dark-gray-vector', 'gray', 'gray-vector', 'hybrid', 'national-geographic', 'oceans', 'osm', 'satellite', 'streets', 'streets-navigation-vector', 'streets-night-vector', 'streets-relief-vector', 'streets-vector', 'terrain', 'topo', 'topo-vector']
            wm.basemap = 'dark-gray'
            print(wm.gallery_basemaps)
            >> ['custom_dark_gray_canvas', 'imagery', 'imagery_hybrid', 'light_gray_canvas', 'custom_basemap_vector_(proxy)', 'world_imagery_(proxy)', 'world_street_map_(proxy)']
            wm.basemap = 'custom_dark_gray_canvas'

            # Usage example 3: Set the basemap equal to an item
            from arcgis.mapping import WebMap
            wm = WebMap(wm_item)
            # Use basemap from another item as your own
            wm.basemap = wm_item_2
            wm.basemap = tiled_map_service_item
            wm.basemap = image_layer_item
            wm.basemap = wm2.basemap
            wm.basemap = wm2

        """
        if self._basemap:
            return PropertyMap(self._basemap)
        else:
            if "baseMap" in self._webmapdict.keys():
                self._basemap = self._webmapdict['baseMap']
            return PropertyMap(self._basemap)

    def _determine_layer_type(self, item):
        # this function determines the basemap layer type for the Web Map Specification
        if item.type == "Image Service":
            layer_type = "ArcGISImageServiceLayer"
            layer = arcgis.raster.ImageryLayer(item.url, gis=self._gis)
        else:
            layer_type = "ArcGISMapServiceLayer"
            layer = arcgis.mapping.MapImageLayer(item.url, gis=self._gis)
        if "tileInfo" in layer.properties:
            layer_type = "ArcGIS" + layer_type.replace("ArcGIS", "Tiled")
        return layer_type

    @basemap.setter
    def basemap(self, value):
        """What basemap you would like to apply to the map (‘topo’,
                ‘national-geographic’, etc.). See `basemaps` and `gallery_basemaps` for a full list
        """
        from arcgis.widgets import MapView
        if isinstance(value, MapView):
            # get basemap from map widget
            if value.basemap in self.basemaps:
                self._basemap = {'baseMapLayers': basemap_dict[value.basemap],
                                 'title': value.basemap.replace("-", " ").title()}
                self._webmapdict['baseMap'] = self._basemap
        elif value in self.basemaps:
            self._basemap = {'baseMapLayers':basemap_dict[value],
                             'title': value.replace("-"," ").title()}
            self._webmapdict['baseMap'] = self._basemap
        elif value in self.gallery_basemaps:
            self._basemap = self._gallery_basemaps[value]
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, Item) and value.type.title() == "Web Map":
            self._basemap = value.get_data()['baseMap']
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, WebMap):
            self._basemap = value.basemap
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, PropertyMap) and "baseMapLayers" in value:
            # for map1.basemap = map2.basemap
            self._basemap = value
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, Item) and (value.type.title() == "Image Service" or value.type.title() == "Map Service"):
            layer_type = self._determine_layer_type(value)
            self._basemap = {
                'baseMapLayers':[{'id': 'newBasemap',
                                  'layerType': layer_type,
                                  'url': value.url,
                                  'visibility': True,
                                  'opacity': 1,
                                  'title': value.title}],
                'title':value.title
            }
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, Item) and value.type.title() == "Vector Tile Service":
            try:
                style_url = "%s/sharing/rest/content/items/%s/resources/styles/root.json" % (value._gis._portal.url, value.id)
                value._gis._con.get(path=style_url)
            except Exception:
                style_url = value.url + "/resources/styles/root.json"
            self._basemap = {
                'baseMapLayers': [{'id': 'newBasemap',
                                   'layerType': 'VectorTileLayer',
                                   'styleUrl': style_url,
                                   'visibility': True,
                                   'itemId': value.id,
                                   'opacity': 1,
                                   'title': value.title}],
                'title': value.title
            }
            self._webmapdict['baseMap'] = self._basemap
        elif isinstance(value, VectorTileLayer):
            try:
                style_url = value.url+"/resources/styles/root.json"
                value._con.get(path=style_url)
                self._basemap = {
                    'baseMapLayers': [{'id': 'newBasemap',
                                       'layerType': 'VectorTileLayer',
                                       'styleUrl': style_url,
                                       'visibility': True,
                                       'opacity': 1,
                                       'title': value.properties["name"]}],
                    'title': value.properties["name"]
                }
                self._webmapdict['baseMap'] = self._basemap
            except Exception:
                raise RuntimeError("Basemap '{}' isn't valid".format(value))
        else:
            raise RuntimeError("Basemap '{}' isn't valid".format(value))

    @property
    def basemaps(self):
        """
        A list of possible basemaps to set for the map
        """
        basemaps = ['dark-gray',
                    'dark-gray-vector',
                    'gray',
                    'gray-vector',
                    'hybrid',
                    'national-geographic',
                    'oceans',
                    'osm',
                    'satellite',
                    'streets',
                    'streets-navigation-vector',
                    'streets-night-vector',
                    'streets-relief-vector',
                    'streets-vector',
                    'terrain',
                    'topo',
                    'topo-vector']
        return basemaps

    @property
    def gallery_basemaps(self):
        """
        View your portal's custom basemap group
        """
        if self._gis:
            bmquery = self._gis.properties['basemapGalleryGroupQuery']
            basemapsgrp = self._gis.groups.search(bmquery, outside_org=True)
            if len(basemapsgrp) == 1:
                for bm in basemapsgrp[0].content():
                    if bm.type.lower() == 'web map':  # Only use WebMaps
                        item_data = bm.get_data()
                        bm_title = bm.title.lower().replace(" ", "_")
                        self._gallery_basemaps[bm_title] = item_data['baseMap']
                return list(self._gallery_basemaps.keys())
            else:
                return list(self._gallery_basemaps.keys())
        else:
            return []

    def remove_table(self, table):
        """
        Removes the specified table from the web map. You can get the list of tables in map using the 'tables' property
        and pass one of those tables to this method for removal form the map.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        table                  Required object. Pass the table that needs to be removed from the map. You can get the
                               list of tables in the map by calling the `tables` property.
        ==================     ====================================================================
        """
        self._webmapdict['tables'].remove(table)
        self._tables.remove(PropertyMap(table))

    def remove_layer(self, layer):
        """
        Removes the specified layer from the web map. You can get the list of layers in map using the 'layers' property
        and pass one of those layers to this method for removal form the map.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        layer                  Required object. Pass the layer that needs to be removed from the map. You can get the
                               list of layers in the map by calling the `layers` property.
        ==================     ====================================================================
        """

        self._webmapdict['operationalLayers'].remove(layer)
        self._layers.remove(PropertyMap(layer))

    def get_layer(self, item_id=None, title=None, layer_id=None):
        """
        Returns the first layer with a matching itemId, title, or layer_id in the webmap's operational layers.
        Pass one of the three parameters into the method to return the layer.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        item_id                Optional string. Pass the item_id for the operational layer you are trying
                               to reference in the webmap.
        ------------------     --------------------------------------------------------------------
        title                  Optional string. Pass the title for the operational layer you are trying
                               to reference in the webmap.
        ------------------     --------------------------------------------------------------------
        layer_id               Optional string. Pass the id for the operational layer you are trying
                               to reference in the webmap.
        ==================     ====================================================================

        :return: Layer as a dictionary
        """
        if item_id is None and title is None and layer_id is None:
            raise ValueError("Please pass at least one parameter into the function")
        if self.layers:
            for layer in self.layers:
                # item id is optional in the webmap spec, so we need to try/except
                try:
                    if (title == layer["title"]) or (layer_id == layer["id"]) or (item_id == layer["itemId"]):
                        return layer
                except Exception:
                    pass
        return None

    def get_table(self, item_id=None, title=None, layer_id=None):
        """
        Returns the first table with a matching itemId, title, or layer_id in the webmap's tables.
        Pass one of the three parameters into the method to return the table.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        item_id                Optional string. Pass the item_id for the table you are trying
                               to reference in the webmap.
        ------------------     --------------------------------------------------------------------
        title                  Optional string. Pass the title for the table you are trying
                               to reference in the webmap.
        ------------------     --------------------------------------------------------------------
        layer_id               Optional string. Pass the id for the table you are trying
                               to reference in the webmap.
        ==================     ====================================================================

        :return: Table as a dictionary
        """
        if item_id is None and title is None and layer_id is None:
            raise ValueError("Please pass at least one parameter into the function")
        if self.tables:
            for table in self.tables:
                # item id is optional in the webmap spec, so we need to try/except
                try:
                    if (title == table["title"]) or (layer_id == table["id"]) or (item_id == table["itemId"]):
                        return table
                except Exception:
                    pass
        return None

    @property
    def offline_areas(self):
        """
        Resource manager for offline areas cached for the web map
        :return:
        """
        return OfflineMapAreaManager(self.item, self._gis)

    @property
    def pop_ups(self):
        """
        :return: True if popups are enabled for dashboard widget.
        """
        return self._pop_ups

    @pop_ups.setter
    def pop_ups(self, value):
        """
        Set popup True or False for dashboard widget.
        """
        self._pop_ups = value
        if value in [0, '0', False, 'false']:
            self._pop_ups = False

    @property
    def bookmarks(self):
        """
        :return: True if bookmarks are enabled for dashboard widget.
        """
        return self._bookmarks

    @bookmarks.setter
    def bookmarks(self, value):
        """
        Set bookmarks True or False for dashboard widget.
        """
        self._bookmarks = value
        if value in [0, '0', False, 'false']:
            self._bookmarks = False

    @property
    def legend(self):
        """
        :return: True if legend visibility is enabled for dashboard widget.
        """
        return self._legend

    @legend.setter
    def legend(self, value):
        """
        Set legend visibility to True or False.
        """
        self._legend = value
        if value in [0, '0', False, 'false']:
            self._legend = False

    @property
    def layer_visibility(self):
        """
        :return: True if layer visibility is enabled for dashboard widget.
        """
        return self._layer_visibility

    @layer_visibility.setter
    def layer_visibility(self, value):
        """
        Set layer visibility for dashboard widget.
        """
        self._layer_visibility = value
        if value in [0, '0', False, 'false']:
            self._layer_visibility = False

    @property
    def basemap_switcher(self):
        """
        :return: True if Basemap switcher is enabled.
        """
        return self._basemap_switcher

    @basemap_switcher.setter
    def basemap_switcher(self, value):
        """
        Set basemap switcher True or False, for dashboard widget.
        """
        self._basemap_switcher = value
        if value in [0, '0', False, 'false']:
            self._basemap_switcher = False

    @property
    def search(self):
        """
        :return: True if search is enabled for dashboard widget.
        """
        return self._search

    @search.setter
    def search(self, value):
        """
        Set search True or False, for dashboard widget.
        """
        self._search = value
        if value in [0, '0', False, 'false']:
            self._search = False

    @property
    def zoom(self):
        """
        :return: zoom enabled or disabled for dashboard widget.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """
        Enable or disable zoom for dashboard widget.
        """
        self._zoom = value
        if value in [0, '0', False, 'false']:
            self._zoom = False

    @property
    def navigation(self):
        """
        :return: navigation enabled or disabled.
        """
        return self._navigation

    @navigation.setter
    def navigation(self, value):
        """
        Enable or disable navigation for dashboard widget.
        """
        if value in [0, '0', False, 'false']:
            self._navigation = False
        self._navigation = True

    @property
    def scale_bar(self):
        """
        :return: Scale bar from ("none", "ruler", "scale")
        """
        return self._scale_bar

    @scale_bar.setter
    def scale_bar(self, value):
        """
        Set scale bar for dashboard widget.
        Choose from "line" or "ruler" or set "none" to disable.
        """
        self._scale_bar = value
        if value not in ["none", "line", "ruler"]:
            self._scale_bar = "none"

    @property
    def height(self):
        """
        :return: Height of the widget
        """
        return self._height

    @height.setter
    def height(self, value):
        """
        Set height of the widget, between 0 and 1.
        """
        if value > 1:
            self._height = 1
        elif value < 0:
            self._height = 0
        else:
            self._height = value

    @property
    def width(self):
        """
        :return: Width of the widget
        """
        return self._width

    @width.setter
    def width(self, value):
        """
        Set width of the widget, between 0 and 1.
        """
        if value > 1:
            self._width = 1
        elif value < 0:
            self._width = 0
        else:
            self._width = value

    def _convert_to_json(self):
        data = {
            "events":[],
            "type": "mapWidget",
            "flashRepeats": 3,
            "itemId": self.item.id,
            "mapTools": [],
            "showNavigation": self.navigation,
            "showPopup": self.pop_ups,
            "scalebarStyle": self.scale_bar,
            "layers": [{"type": "featureLayerDataSource", "layerId": layer['id']} for layer in self.layers],
            "id": self._id,
            "name": self.item.title,
            "caption": self.item.name,
            "showLastUpdate": True,
            "noDataVerticalAlignment": "middle",
            "showCaptionWhenNoData": False,
            "showDescriptionWhenNoData": False
        }

        if self.bookmarks:
            data['mapTools'].append({"type": "bookmarksTool"})

        if self.legend:
            data['mapTools'].append({"type": "legendTool"})

        if self.layer_visibility:
            data['mapTools'].append({'type': "mapContentsTool"})

        if self.basemap_switcher:
            data['mapTools'].append({"type": "basemapGalleryTool"})

        if self.search:
            data['mapTools'].append({"type": "searchTool"})

        if self.events.enable:
            data["events"].append({"type":self.events.type, "actions":self.events.synced_widgets})

        return data

###########################################################################
class PackagingJob(object):
    """
    Represents a Single Packaging Job.


    ================  ===============================================================
    **Argument**      **Description**
    ----------------  ---------------------------------------------------------------
    future            Required ccurrent.futures.Future.  The async object created by
                      the geoprocessing (GP) task.
    ----------------  ---------------------------------------------------------------
    notify            Optional Boolean.  When set to True, a message will inform the
                      user that the geoprocessing task has completed. The default is
                      False.
    ================  ===============================================================

    """
    _future = None
    _gis = None
    _start_time = None
    _end_time = None

    #----------------------------------------------------------------------
    def __init__(self, future, notify=False):
        """
        initializer
        """
        self._future = future
        self._start_time = datetime.datetime.now()
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)
    #----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        if self._end_time:
            return self._end_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time
    #----------------------------------------------------------------------
    def _set_end_time(self, future):
        """sets the finish time"""
        self._end_time = datetime.datetime.now()
    #----------------------------------------------------------------------
    def _notify(self, future):
        """prints finished method"""
        jobid = str(self).replace("<", "").replace(">", "")
        try:
            res = future.result()
            infomsg = '{jobid} finished successfully.'.format(jobid=jobid)
            _log.info(infomsg)
            print(infomsg)
        except Exception as e:
            msg = str(e)
            msg = '{jobid} failed: {msg}'.format(jobid=jobid, msg=msg)
            _log.info(msg)
            print(msg)
    #----------------------------------------------------------------------
    def __str__(self):
        return "<Packaging Job>"
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<Packaging Job>"
    #----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the GP status

        :returns: String
        """
        return self._future.done()
    #----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :returns: boolean
        """
        if self.done():
            return False
        if self.cancelled():
            return False
        return True
    #----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :returns: boolean
        """
        return self._future.cancelled()
    #----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :returns: boolean
        """
        return self._future.running()
    #----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :returns: boolean
        """
        return self._future.done()
    #----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :returns: object
        """
        if self.cancelled():
            return None
        return self._future.result()
###########################################################################
class OfflineMapAreaManager(object):
    """
    Helper class to manage offline map areas attached to a web map item. Users should not instantiate this class
    directly, instead, should access the methods exposed by accessing the `offline_areas` property on the `WebMap`
    object.
    """
    _pm = None
    _gis = None
    _tbx = None
    _item = None
    _portal = None
    _web_map = None
    #----------------------------------------------------------------------
    def __init__(self, item, gis):
        from arcgis.geoprocessing import import_toolbox
        self._gis = gis
        self._portal = gis._portal
        self._item = item
        self._web_map = WebMap(self._item)
        try:
            from arcgis._impl.tools import _PackagingTools
            self._url = self._gis.properties.helperServices.packaging.url
            self._pm = self._gis._tools.packaging


        except Exception:
            warn("GIS does not support creating packages for offline usage")
    #----------------------------------------------------------------------
    @property
    def offline_properties(self):
        """
        This property allows users to configure the offline properties
        for a webmap.  The `offline_properties` allows for the definition
        of how available offline editing, basemap, and read-only layers
        behave in the web map application.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        values                 Required Dict.  The key/value object that defines the offline
                               application properties.
        ==================     ====================================================================

        The dictionary supports the following keys in the dictionary

        ==================     ====================================================================
        **Key**                **Values**
        ------------------     --------------------------------------------------------------------
        download               When editing layers, the edits are always sent to the server. This
                               string value indicates which data is retrieved. For example, none
                               indicates that only the schema is written since neither the features
                               nor attachments are retrieved. For a full sync without downloading
                               attachments, indicate features. Lastly, the default behavior is to
                               have a full sync using `features_and_attachments` where both
                               features and attachments are retrieved.

                               If property is present, must be one of the following values:

                               Allowed Values: [features, features_and_attachments, None]
        ------------------     --------------------------------------------------------------------
        sync                   `sync` applies to editing layers only.  This string value indicates
                               how the data is synced.

                               Allowed Values:

                               sync_features_and_attachments  - bidirectional sync
                               sync_features_upload_attachments - bidirection sync for feaures but upload only for attachments
                               upload_features_and_attachments - upload only for both features and attachments (initial replica is just a schema)


        ------------------     --------------------------------------------------------------------
        reference_basemap      The filename of a basemap that has been copied to a mobile device.
                               This can be used instead of the default basemap for the map to
                               reduce downloads.
        ------------------     --------------------------------------------------------------------
        get_attachments        Boolean value that indicates whether to include attachments with the
                               read-only data.
        ==================     ====================================================================

        :returns: Dictionary

        """
        dl_lu = {
            "features" : "features",
            "featuresAndAttachments" : "features_and_attachments",
            "features_and_attachments" : "featuresAndAttachments",
            "none" : None,
            "None" : "none",
            None : "none",
            "syncFeaturesAndAttachments" : "sync_features_and_attachments",
            "sync_features_and_attachments" : "syncFeaturesAndAttachments",
            "syncFeaturesUploadAttachments" : "sync_features_upload_attachments",
            "sync_features_upload_attachments" : "syncFeaturesUploadAttachments",
            "uploadFeaturesAndAttachments" : "upload_features_and_attachments",
            "upload_features_and_attachments" : "uploadFeaturesAndAttachments"
        }
        values = {
            "download" : None,
            "sync" : None,
            "reference_basemap" : None,
            "get_attachments" : None
        }
        if "applicationProperties" in self._web_map._webmapdict and \
           'offline' in self._web_map._webmapdict['applicationProperties']:
            v = self._web_map._webmapdict["applicationProperties"]['offline']
            if 'editableLayers' in v:
                if 'download' in v['editableLayers']:
                    values['download'] = dl_lu[v['editableLayers']['download']]
                else:
                    values.pop("download")
                if "sync" in v['editableLayers']:
                    values['sync'] = dl_lu[v['editableLayers']['sync']]
                else:
                    values.pop('sync')
            else:
                values.pop('download')
                values.pop('sync')

            if "offlinebasemap" in v and \
               "referenceBasemapName" in v["offlinebasemap"]:
                values["reference_basemap"] = v["offlinebasemap"]['referenceBasemapName']
            else:
                values.pop("reference_basemap")
            if "readonlyLayers" in v and \
               "downloadAttachments" in v['readonlyLayers']:
                values['get_attachments'] = v['readonlyLayers']['downloadAttachments']
            else:
                values.pop('get_attachments')
            return values
        else:
            self._web_map._webmapdict["applicationProperties"] = {"offline" : {}}
            return self._web_map._webmapdict["applicationProperties"]['offline']
    #----------------------------------------------------------------------
    @offline_properties.setter
    def offline_properties(self, values):
        """
        This property allows users to configure the offline properties
        for a webmap.  The `offline_properties` allows for the definition
        of how available offline editing, basemap, and read-only layers
        behave in the web map application.


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        values                 Required Dict.  The key/value object that defines the offline
                               application properties.
        ==================     ====================================================================

        The dictionary supports the following keys in the dictionary

        ==================     ====================================================================
        **Key**                **Values**
        ------------------     --------------------------------------------------------------------
        download               When editing layers, the edits are always sent to the server. This
                               string value indicates which data is retrieved. For example, none
                               indicates that only the schema is written since neither the features
                               nor attachments are retrieved. For a full sync without downloading
                               attachments, indicate features. Lastly, the default behavior is to
                               have a full sync using `features_and_attachments` where both
                               features and attachments are retrieved.

                               If property is present, must be one of the following values:

                               Allowed Values: [features, features_and_attachments, None]
        ------------------     --------------------------------------------------------------------
        sync                   `sync` applies to editing layers only.  This string value indicates
                               how the data is synced.

                               Allowed Values:

                               sync_features_and_attachments  - bidirectional sync
                               sync_features_upload_attachments - bidirection sync for feaures but upload only for attachments
                               upload_features_and_attachments - upload only for both features and attachments (initial replica is just a schema)


        ------------------     --------------------------------------------------------------------
        reference_basemap      The filename of a basemap that has been copied to a mobile device.
                               This can be used instead of the default basemap for the map to
                               reduce downloads.
        ------------------     --------------------------------------------------------------------
        get_attachments        Boolean value that indicates whether to include attachments with the
                               read-only data.
        ==================     ====================================================================

        :returns: Dictionary

        """
        dl_lu = {
            "features" : "features",
            "featuresAndAttachments" : "features_and_attachments",
            "features_and_attachments" : "featuresAndAttachments",
            "none" : None,
            "None" : "none",
            None : "none",
            "syncFeaturesAndAttachments" : "sync_features_and_attachments",
            "sync_features_and_attachments" : "syncFeaturesAndAttachments",
            "syncFeaturesUploadAttachments" : "sync_features_upload_attachments",
            "sync_features_upload_attachments" : "syncFeaturesUploadAttachments",
            "uploadFeaturesAndAttachments" : "upload_features_and_attachments",
            "upload_features_and_attachments" : "uploadFeaturesAndAttachments"
        }
        keys = {'download' : 'download', 'sync': 'sync',
                'reference_basemap' : "referenceBasemapName",
                'get_attachments' : "downloadAttachments"}
        remove = set()
        if "applicationProperties" in self._web_map._webmapdict:
            v = self._web_map._webmapdict["applicationProperties"]
        else:
            v = self._web_map._webmapdict["applicationProperties"] = {}
        if "offline" not in v:
            v['offline'] = {
                "editableLayers": {
                    "download": dl_lu[values['download']] if 'download' in values else remove.add('download'),
                    "sync": dl_lu[values['sync']] if 'sync' in values else remove.add('sync')
                    },
                "offlinebasemap": {
                    "referenceBasemapName": dl_lu[values['reference_basemap']] if 'reference_basemap' in values else remove.add('reference_basemap')
                    },
                "readonlyLayers": {
                    "downloadAttachments": values['get_attachments'] if 'get_attachments' in values else remove.add('get_attachments')
                }
            }
        else:
            v['offline'] = {
                "editableLayers": {
                    "download": dl_lu[values['download']] if 'download' in values else remove.add('download'),
                    "sync": dl_lu[values['sync']] if 'sync' in values else remove.add('sync')
                    },
                "offlinebasemap": {
                    "referenceBasemapName": dl_lu[values['reference_basemap']] if 'reference_basemap' in values else remove.add('reference_basemap')
                    },
                "readonlyLayers": {
                    "downloadAttachments": values['get_attachments'] if 'get_attachments' in values else remove.add('get_attachments')
                }
            }
        for r in remove:
            if 'sync' in remove and 'download' in remove:
                del v['offline']['editableLayers']
            if 'sync' in remove:
                del v['offline']['editableLayers']['sync']
            if 'download' in remove:
                del v['offline']['editableLayers']['download']
            if 'reference_basemap' in remove:
                del v['offline']['offlinebasemap']
            if 'get_attachments' in remove:
                del v['offline']['readonlyLayers']
            del r
        update_items = {
            'clearEmptyFields' : True,
            'text' : json.dumps(self._web_map._webmapdict)
        }
        if self._item.update(item_properties=update_items):
            self._item._hydrated = False
            self._item._hydrate()
            self._web_map = WebMap(self._item)
        else:
            raise Exception("Could not update the offline properties.")
    #----------------------------------------------------------------------
    def _run_async(self, fn, **inputs):
        """runs the inputs asynchronously"""
        import concurrent.futures
        tp = concurrent.futures.ThreadPoolExecutor(1)
        future = tp.submit(fn=fn, **inputs)
        tp.shutdown(False)
        return future
    #----------------------------------------------------------------------
    def create(self, area, item_properties=None, folder=None, min_scale=None,
               max_scale=None, layers_to_ignore=None, refresh_schedule="Never",
               refresh_rates=None, enable_updates=False, ignore_layers=None,
               tile_services=None, future=False):
        """

        Create offline map area items and packages for ArcGIS Runtime powered applications. This method creates two
        different types of items. It first creates 'Map Area' items for the specified extent or bookmark. Next it
        creates one or more map area packages corresponding to each layer type in the extent.

        .. note::
            - There can be only 1 map area item for an extent or bookmark.
            - You need to be the owner of the web map or an administrator of your GIS.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        area                   Required object.  Bookmark or extent. Specify as either:

                                   + bookmark name
                                       `WebMap.definition.bookmarks` returns list of bookmarks.
                                   + list of coordinate pairs:
                                       [['xmin', 'ymin'], ['xmax', 'ymax']]
                                   + dictionary:
                                         {'xmin': <value>,
                                         'ymin': <value>,
                                         'xmax': <value>,
                                         'ymax': <value>,
                                         'spatialReference' : {'wkid' : <value>}}

                               If spatial reference is not specified, it is assumed 'wkid': 4326.
        ------------------     --------------------------------------------------------------------
        item_properties        Required dictionary. See table below for the keys and values.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. Specify a folder name if you want the offline map
                               area item and the packages to be created inside a folder.
        ------------------     --------------------------------------------------------------------
        min_scale              Optional number. Specify the minimum scale to cache tile and vector
                               tile layers. When zoomed out beyond this scale, cached layers would
                               not display.
        ------------------     --------------------------------------------------------------------
        max_scale              Optional number. Specify the maximum scale to cache tile and vector
                               tile layers. When zoomed in beyond this scale, cached layers would
                               not display.
        ------------------     --------------------------------------------------------------------
        layers_to_ignore       Optional List of layer objects to exclude when creating offline
                               packages. You can get the list of layers in a web map by calling
                               the `layers` property on the `WebMap` object.
        ------------------     --------------------------------------------------------------------
        refresh_schedule       Optional string. Allows for the scheduling of refreshes at given times.


                               The following are valid variables:

                                    + Never - never refreshes the offline package (default)
                                    + Daily - refreshes everyday
                                    + Weekly - refreshes once a week
                                    + Monthly - refreshes once a month

        ------------------     --------------------------------------------------------------------
        refresh_rates          Optional dict. This parameter allows for the customization of the
                               scheduler.  The dictionary accepts the following:

                                {
                                    "hour" : 1
                                    "minute" = 0
                                    "nthday" = 3
                                    "day_of_week" = 0
                                }

                               - hour - a value between 0-23 (integers)
                               - minute a value between 0-60 (integers)
                               - nthday - this is used for monthly only. This say the refresh will occur on the 'x' day of the month.
                               - day_of_week - a value between 0-6 where 0 is Sunday and 6 is Saturday.

                               Example **Daily**:

                                {
                                    "hour": 10,
                                    "minute" : 30
                                }

                               This means every day at 10:30 AM UTC

                               Example **Weekly**:

                                {
                                    "hour" : 23,
                                    "minute" : 59,
                                    "day_of_week" : 4
                                }

                               This means every Wednesday at 11:59 PM UTC
        ------------------     --------------------------------------------------------------------
        enable_updates         Optional Boolean.  Allows for the updating of the layers.
        ------------------     --------------------------------------------------------------------
        ignore_layers          Optional List.  A list of individual layers, specified with their
                               service URLs, in the map to ignore. The task generates packages for
                               all map layers by default.

                               Example:

                                [
                                  "https://services.arcgis.com/ERmEceOGq5cHrItq/arcgis/rest/services/SaveTheBaySync/FeatureServer/1",
                                  "https://services.arcgis.com/ERmEceOGq5cHrItq/arcgis/rest/services/WildfireSync/FeatureServer/0"
                                ]

        ------------------     --------------------------------------------------------------------
        tile_services          Optional List.  An array of JSON objects that contains additional
                               export tiles enabled tile services for which tile packages (.tpk or
                               .vtpk) need to be created. Each tile service is specified with its
                               URL and desired level of details.

                               Example:

                                [
                                  {
                                    "url": "https://tiledbasemaps.arcgis.com/arcgis/rest/services/World_Imagery/MapServer",
                                    "levels": "17,18,19"
                                  }
                                ]

        ==================     ====================================================================

        *Hint: Your min_scale is always bigger in value than your max_scale*

        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of
                           strings. Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters)
                           of the what the item is.
        =================  =====================================================================

        :return:
            Item object for the offline map area item that was created.
            If Future==True, then the result is a PackageJob

        .. code-block:: python

            USAGE EXAMPLE: Creating offline map areas

            wm = WebMap(wm_item)

            # create offline areas ignoring a layer and for certain min, max scales for other layers
            item_prop = {'title': 'Clear lake hyperspectral field campaign',
                        'snippet': 'Offline package for field data collection using spectro-radiometer',
                        'tags': ['python api', 'in-situ data', 'field data collection']}

            aviris_layer = wm.layers[-1]

            north_bed = wm.definition.bookmarks[-1]['name']
            wm.offline_areas.create(area=north_bed, item_properties=item_prop,
                                  folder='clear_lake', min_scale=9000, max_scale=4500,
                                   layers_to_ignore=[aviris_layer])

        .. note::
            This method executes silently. To view informative status messages, set the verbosity environment variable
            as shown below:

            .. code-block:: python

               USAGE EXAMPLE: setting verbosity

               from arcgis import env
               env.verbose = True


        """
        if future:
            inputs = {
                "area":area,
                "item_properties":item_properties,
                "folder":folder,
                "min_scale":min_scale,
                "max_scale":max_scale,
                "layers_to_ignore":layers_to_ignore,
                "refresh_schedule":refresh_schedule,
                "refresh_rates":refresh_rates,
                "enable_updates":enable_updates,
                "ignore_layers":ignore_layers,
                "tile_services":tile_services
            }
            future = self._run_async(self._create, **inputs)
            return PackagingJob(future=future)
        else:
            return self._create(area=area,
                                item_properties=item_properties,
                                folder=folder,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                layers_to_ignore=layers_to_ignore,
                                refresh_schedule=refresh_schedule,
                                refresh_rates=refresh_rates,
                                enable_updates=enable_updates,
                                ignore_layers=ignore_layers,
                                tile_services=tile_services)

    #----------------------------------------------------------------------
    def _create(self, area, item_properties=None, folder=None, min_scale=None,
                max_scale=None, layers_to_ignore=None, refresh_schedule="Never",
                refresh_rates=None, enable_updates=False, ignore_layers=None,
                tile_services=None, future=False):
        """
        Create offline map area items and packages for ArcGIS Runtime powered applications. This method creates two
        different types of items. It first creates 'Map Area' items for the specified extent or bookmark. Next it
        creates one or more map area packages corresponding to each layer type in the extent.

        .. note::
            - Offline map area functionality is only available if your GIS is ArcGIS Online.
            - There can be only 1 map area item for an extent or bookmark.
            - You need to be the owner of the web map or an administrator of your GIS.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        area                   Required object. You can specify the name of a web map bookmark or a
                               desired extent.

                               To get the bookmarks from a web map, query the `definition.bookmarks`
                               property.

                               You can specify the extent as a list or dictionary of 'xmin', 'ymin',
                               'xmax', 'ymax' and spatial reference. If spatial reference is not
                               specified, it is assumed to be 'wkid' : 4326.
        ------------------     --------------------------------------------------------------------
        item_properties        Required dictionary. See table below for the keys and values.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. Specify a folder name if you want the offline map
                               area item and the packages to be created inside a folder.
        ------------------     --------------------------------------------------------------------
        min_scale              Optional number. Specify the minimum scale to cache tile and vector
                               tile layers. When zoomed out beyond this scale, cached layers would
                               not display.
        ------------------     --------------------------------------------------------------------
        max_scale              Optional number. Specify the maximum scale to cache tile and vector
                               tile layers. When zoomed in beyond this scale, cached layers would
                               not display.
        ------------------     --------------------------------------------------------------------
        layers_to_ignore       Optional List of layer objects to exclude when creating offline
                               packages. You can get the list of layers in a web map by calling
                               the `layers` property on the `WebMap` object.
        ------------------     --------------------------------------------------------------------
        refresh_schedule       Optional string. Allows for the scheduling of refreshes at given times.


                               The following are valid variables:

                                    + Never - never refreshes the offline package (default)
                                    + Daily - refreshes everyday
                                    + Weekly - refreshes once a week
                                    + Monthly - refreshes once a month

        ------------------     --------------------------------------------------------------------
        refresh_rates          Optional dict. This parameter allows for the customization of the
                               scheduler.  The dictionary accepts the following:

                                {
                                    "hour" : 1
                                    "minute" = 0
                                    "nthday" = 3
                                    "day_of_week" = 0
                                }

                               - hour - a value between 0-23 (integers)
                               - minute a value between 0-60 (integers)
                               - nthday - this is used for monthly only. This say the refresh will occur on the 'x' day of the month.
                               - day_of_week - a value between 0-6 where 0 is Sunday and 6 is Saturday.

                               Example **Daily**:

                                {
                                    "hour": 10,
                                    "minute" : 30
                                }

                               This means every day at 10:30 AM UTC

                               Example **Weekly**:

                                {
                                    "hour" : 23,
                                    "minute" : 59,
                                    "day_of_week" : 4
                                }

                               This means every Wednesday at 11:59 PM UTC
        ------------------     --------------------------------------------------------------------
        enable_updates         Optional Boolean.
        ------------------     --------------------------------------------------------------------
        ignore_layers
        ------------------     --------------------------------------------------------------------
        tile_services
        ------------------     --------------------------------------------------------------------

        ==================     ====================================================================

        *Hint: Your min_scale is always bigger in value than your max_scale*

        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of
                           strings. Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters)
                           of the what the item is.
        =================  =====================================================================

        :return:
            Item object for the offline map area item that was created.

        .. code-block:: python

            USAGE EXAMPLE: Creating offline map areas

            wm = WebMap(wm_item)

            # create offline areas ignoring a layer and for certain min, max scales for other layers
            item_prop = {'title': 'Clear lake hyperspectral field campaign',
                        'snippet': 'Offline package for field data collection using spectro-radiometer',
                        'tags': ['python api', 'in-situ data', 'field data collection']}

            aviris_layer = wm.layers[-1]

            north_bed = wm.definition.bookmarks[-1]['name']
            wm.offline_areas.create(area=north_bed, item_properties=item_prop,
                                  folder='clear_lake', min_scale=9000, max_scale=4500,
                                   layers_to_ignore=[aviris_layer])

        .. note::
            This method executes silently. To view informative status messages, set the verbosity environment variable
            as shown below:

            .. code-block:: python

               USAGE EXAMPLE: setting verbosity

               from arcgis import env
               env.verbose = True


        """
        _dow_lu = {
            0:"SUN",
            1:"MON",
            2:"TUE",
            3: "WED",
            4: "THU",
            5: "FRI",
            6: "SAT",
            7: "SUN"
        }
        # region find if bookmarks or extent is specified
        _bookmark = None
        _extent = None
        if item_properties is None:
            item_properties = {}
        if isinstance(area, str):  # bookmark specified
            _bookmark = area
            area_type = 'BOOKMARK'
        elif isinstance(area, (list, tuple)):  # extent specified as list
            _extent = {'xmin': area[0][0],
                       'ymin': area[0][1],
                       'xmax': area[1][0],
                       'ymax': area[1][1],
                       'spatialReference': {'wkid': 4326}}

        elif isinstance(area, dict) and 'xmin' in area:  # geocoded extent provided
            _extent = area
            if 'spatialReference' not in _extent:
                _extent['spatialReference'] = {'wkid': 4326}
        # endregion

        # region build input parameters - for CreateMapArea tool
        if folder:
            user_folders = self._gis.users.me.folders
            if user_folders:
                matching_folder_ids = [f['id'] for f in user_folders if f['title'] == folder]
                if matching_folder_ids:
                    folder_id = matching_folder_ids[0]
                else:  # said folder not found in user account
                    folder_id = None
            else:  # ignore the folder, output will be created in same folder as web map
                folder_id = None
        else:
            folder_id = None

        if 'tags' in item_properties:
            if type(item_properties['tags']) is list:
                tags = ",".join(item_properties['tags'])
            else:
                tags = item_properties['tags']
        else:
            tags = None

        if refresh_schedule.lower() in ['daily', 'weekly', 'monthly']:
            refresh_schedule = refresh_schedule.lower()
            if refresh_schedule == 'daily':
                if refresh_rates and isinstance(refresh_rates, dict):
                    hour = 1
                    minute = 0
                    if 'hour' in refresh_rates:
                        hour = refresh_rates['hour']
                    if 'minute' in refresh_rates:
                        minute = refresh_rates['minute']
                    map_area_refresh_params = {
                        "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                        "type":"daily",
                        "nthDay":1,
                        "dayOfWeek":0
                    }
                    refresh_schedule = "0 {m} {hour} * * ?".format(m=minute, hour=hour)
                else:
                    map_area_refresh_params = {
                        "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                        "type":"daily",
                        "nthDay":1,
                        "dayOfWeek":0
                    }
                    refresh_schedule = "0 0 1 * * ?"
            elif refresh_schedule == 'weekly':
                if refresh_rates and \
                   isinstance(refresh_rates, dict):
                    hour = 1
                    minute = 0
                    dayOfWeek = "MON"
                    if 'hour' in refresh_rates:
                        hour = refresh_rates['hour']
                    if 'minute' in refresh_rates:
                        minute = refresh_rates['minute']
                    if 'day_of_week' in refresh_rates:
                        dayOfWeek = refresh_rates['day_of_week']
                    map_area_refresh_params = {
                        "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                        "type":"weekly",
                            "nthDay": 1,
                            "dayOfWeek": dayOfWeek
                    }
                    refresh_schedule = "0 {m} {hour} ? * {dow}".format(m=minute, hour=hour,
                                                                       dow=_dow_lu[dayOfWeek])
                else:
                    map_area_refresh_params = {
                        "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                        "type":"weekly",
                        "nthDay":1,
                        "dayOfWeek":1
                    }
                    refresh_schedule = "0 0 1 ? * MON"
            elif refresh_schedule == 'monthly':
                if refresh_rates and \
                   isinstance(refresh_rates, dict):
                    hour = 1
                    minute = 0
                    nthday = 3
                    dayOfWeek = 0
                    if 'hour' in refresh_rates:
                        hour = refresh_rates['hour']
                    if 'minute' in refresh_rates:
                        minute = refresh_rates['minute']
                    if 'nthday' in refresh_rates['nthday']:
                        nthday = refresh_rates['nthday']
                    if 'day_of_week' in refresh_rates:
                        dayOfWeek = refresh_rates['day_of_week']
                    map_area_refresh_params = {
                        "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                        "type":"monthly",
                        "nthDay": nthday,
                        "dayOfWeek": dayOfWeek
                    }
                    refresh_schedule = "0 {m} {hour} ? * {nthday}#{dow}".format(m=minute, hour=hour,
                                                                                nthday=nthday, dow=dayOfWeek)
                else:
                    map_area_refresh_params = {"startDate":int(datetime.datetime.utcnow().timestamp()) * 1000,
                                               "type":"monthly","nthDay":3,"dayOfWeek":3}
                    refresh_schedule = "0 0 14 ? * 4#3"
        else:
            refresh_schedule = None
            refresh_rates_cron = None
            map_area_refresh_params = {'type' : 'never'}


        output_name = {'title': item_properties['title'] if 'title' in item_properties else None,
                       'snippet': item_properties['snippet'] if 'snippet' in item_properties else None,
                       'description': item_properties['description'] if 'description' in item_properties else None,
                       'tags': tags,
                       'folderId': folder_id,
                       'packageRefreshSchedule' : refresh_schedule}

        # endregion



        # region call CreateMapArea tool
        #from arcgis.geoprocessing._tool import Toolbox
        #pkg_tb = Toolbox(url=self._url, gis=self._gis)
        pkg_tb = self._gis._tools.packaging
        if self._gis.version >= [7,2]:

            if _extent:
                area = _extent
                area_type = "ENVELOPE"
            elif _bookmark:
                area = {'name' : _bookmark}
                area_type = "BOOKMARK"

            if isinstance(area, str):
                area_type = "BOOKMARK"
            elif isinstance(area, Polygon) or \
                 (isinstance(area, dict) and 'rings' in area):
                area_type = "POLYGON"
            elif isinstance(area, arcgis.geometry.Envelope) or \
                 (isinstance(area, dict) and 'xmin' in area):
                area_type = "ENVELOPE"
            elif isinstance(area, (list, tuple)):
                area_type = "ENVELOPE"
            if refresh_schedule is None:
                output_name.pop("packageRefreshSchedule")
            if folder_id is None:
                output_name.pop('folderId')
            oma_result = pkg_tb.create_map_area(map_item_id=self._item.id,
                                                area_type=area_type,
                                                area=area,
                                                output_name=output_name)

        else:
            oma_result = pkg_tb.create_map_area(self._item.id, _bookmark, _extent, output_name=output_name)
        # endregion

        # Call update on Item with Refresh Information
        #import datetime
        item = Item(gis=self._gis, itemid=oma_result)
        update_items = {
            'snippet' : "Map with no advanced offline settings set (default is assumed to be features and attachments)",
            'title' : item_properties['title'] if 'title' in item_properties else None,
            'typeKeywords' : "Map, Map Area",
            'clearEmptyFields' : True,
            'text' : json.dumps({"mapAreas": {
                'mapAreaTileScale' : {
                    'minScale': min_scale,
                    'maxScale' : max_scale},
                "mapAreaRefreshParams": map_area_refresh_params,
                "mapAreasScheduledUpdatesEnabled" : enable_updates
            }})
        }
        item.update(item_properties=update_items)
        if _extent is None and area_type == "BOOKMARK":
            for bm in self._web_map._webmapdict['bookmarks']:
                if isinstance(area, dict):
                    if bm['name'].lower() == area['name'].lower():
                        _extent = bm['extent']
                        break
                else:
                    if bm['name'].lower() == area.lower():
                        _extent = bm['extent']
                        break
        update_items = {
            "properties": {
                "status":"processing",
                "packageRefreshSchedule": refresh_schedule
            }
        }
        update_items['properties'].update(item.properties)
        if _extent and not 'extent' in item.properties:
            update_items['properties']['extent'] = _extent
        if area and not 'area' in item.properties:
            update_items['properties']['area'] = _extent
        item.update(item_properties=update_items)
        # End Item Update Refresh Call

        # region build input parameters - for setupMapArea tool
        # map layers to ignore parameter
        map_layers_to_ignore = []
        if isinstance(layers_to_ignore, list):
            for layer in layers_to_ignore:
                if isinstance(layer, PropertyMap):
                    if hasattr(layer, 'url'):
                        map_layers_to_ignore.append(layer.url)
                elif isinstance(layer, str):
                    map_layers_to_ignore.append(layer)
        elif isinstance(layers_to_ignore, PropertyMap):
            if hasattr(layers_to_ignore, 'url'):
                map_layers_to_ignore.append(layers_to_ignore.url)
        elif isinstance(layers_to_ignore, str):
            map_layers_to_ignore.append(layers_to_ignore)

        # LOD parameter
        lods = []
        if min_scale or max_scale:
            # find tile and vector tile layers in map
            cached_layers = [l for l in self._web_map.layers if l.layerType in ['VectorTileLayer',
                                                                                'ArcGISTiledMapServiceLayer']]

            # find tile and vector tile layers in basemap set of layers
            if hasattr(self._web_map, 'basemap'):
                if hasattr(self._web_map.basemap, 'baseMapLayers'):
                    cached_layers_bm = [l for l in self._web_map.basemap.baseMapLayers if l.layerType in
                                        ['VectorTileLayer', 'ArcGISTiledMapServiceLayer']]

                    # combine both the layer lists together
                    cached_layers.extend(cached_layers_bm)

            for cached_layer in cached_layers:
                if cached_layer.layerType == 'VectorTileLayer':
                    if hasattr(cached_layer, 'url'):
                        layer0_obj = VectorTileLayer(cached_layer.url, self._gis)
                    elif hasattr(cached_layer, 'itemId'):
                        layer0_obj = VectorTileLayer.fromitem(self._gis.content.get(cached_layer.itemId))
                else:
                    layer0_obj = MapImageLayer(cached_layer.url, self._gis)

                # region snap logic
                # Objective is to find the LoD that is close to the min scale specified. When scale falls between two
                # levels in the tiling scheme, we will pick the larger limit for min_scale and smaller limit for
                # max_scale.

                # Start by sorting the tileInfo dictionary. Then use Python's bisect_left to find the conservative tile
                # LOD that is closest to min scale. Do similar for max_scale.

                sorted_lods = sorted(layer0_obj.properties.tileInfo.lods, key=lambda x:x['scale'])
                keys = [l['scale'] for l in sorted_lods]

                from bisect import bisect_left
                min_lod_info = sorted_lods[bisect_left(keys, min_scale)]
                max_lod_info = sorted_lods[bisect_left(keys, max_scale) - 1 if bisect_left(keys, max_scale) > 0 else 0]

                lod_span = [str(i) for i in range(min_lod_info['level'], max_lod_info['level'] + 1)]
                lod_span_str = ",".join(lod_span)
                # endregion

                lods.append({'url': layer0_obj.url,
                             'levels': lod_span_str})
            # endregion
        # endregion
        feature_services = None
        if enable_updates:
            if feature_services is None:
                feature_services = {}
                for l in self._web_map.layers:
                    if os.path.dirname(l['url']) not in feature_services:

                        feature_services[os.path.dirname(l['url'])] = {
                            "url": os.path.dirname(l['url']),
                            "layers": [int(os.path.basename(l['url']))],
                            #"returnAttachments": False,
                            #"attachmentsSyncDirection": "upload",
                            #"syncModel": "perLayer",
                            "createPkgDeltas": {
                                "maxDeltaAge": 5
                            }
                        }
                    else:
                        feature_services[os.path.dirname(l['url'])]['layers'].append(int(os.path.basename(l['url'])))
                feature_services = list(feature_services.values())
        # region call the SetupMapArea tool
        #pkg_tb.setup_map_area(map_area_item_id, map_layers_to_ignore=None, tile_services=None, feature_services=None, gis=None, future=False)
        setup_oma_result = pkg_tb.setup_map_area(map_area_item_id=oma_result,
                                                 map_layers_to_ignore=map_layers_to_ignore,
                                                 tile_services=lods,
                                                 feature_services=feature_services,
                                                 gis=self._gis,
                                                 future=True)
        if future:
            return setup_oma_result
        #setup_oma_result.result()
        _log.info(str(setup_oma_result.result()))
        # endregion
        return Item(gis=self._gis, itemid=oma_result)
    #----------------------------------------------------------------------
    def modify_refresh_schedule(self, item, refresh_schedule=None, refresh_rates=None):
        """
        Modifies an Existing Package's Refresh Schedule for offline packages.

        ============================     ====================================================================
        **Argument**                     **Description**
        ----------------------------     --------------------------------------------------------------------
        item                             Required Item. This is the Offline Package to update the refresh
                                         schedule.
        ----------------------------     --------------------------------------------------------------------
        refresh_schedule                 Optional String.  This is the rate of refreshing.

                                         The following are valid variables:

                                            + Never - never refreshes the offline package (default)
                                            + Daily - refreshes everyday
                                            + Weekly - refreshes once a week
                                            + Monthly - refreshes once a month
        ----------------------------     --------------------------------------------------------------------
        refresh_rates                    Optional dict. This parameter allows for the customization of the
                                         scheduler. Note all time is in UTC.

                                         The dictionary accepts the following:

                                         {
                                            "hour" : 1
                                            "minute" = 0
                                            "nthday" = 3
                                            "day_of_week" = 0
                                         }

                                         - hour - a value between 0-23 (integers)
                                         - minute a value between 0-60 (integers)
                                         - nthday - this is used for monthly only. This say the refresh will occur on the 'x' day of the month.
                                         - day_of_week - a value between 0-6 where 0 is Sunday and 6 is Saturday.

                                         Example **Daily**:

                                         {
                                            "hour": 10,
                                            "minute" : 30
                                         }

                                         This means every day at 10:30 AM UTC

                                         Example **Weekly**:

                                         {
                                            "hour" : 23,
                                            "minute" : 59,
                                            "day_of_week" : 4
                                         }

                                         This means every Wednesday at 11:59 PM UTC

        ============================     ====================================================================

        :returns: boolean


        ## Updates Offline Package Building Everyday at 10:30 AM UTC

        .. code-block:: python

            gis = GIS(profile='owner_profile')
            item = gis.content.get('9b93887c640a4c278765982aa2ec999c')
            oa = wm.offline_areas.modify_refresh_schedule(item.id, 'daily', {'hour' : 10, 'minute' : 30})


        """
        if isinstance(item, str):
            item = self._gis.content.get(item)
        _dow_lu = {
            0:"SUN",
            1:"MON",
            2:"TUE",
            3: "WED",
            4: "THU",
            5: "FRI",
            6: "SAT",
            7: "SUN"
        }
        hour = 1
        minute = 0
        nthday = 3
        dayOfWeek = 0
        if refresh_rates is None:
            refresh_rates = {}
        if refresh_schedule is None or str(refresh_schedule).lower() == "never":
            refresh_schedule = None
            refresh_rates_cron = None
            map_area_refresh_params = {'type' : 'never'}
        elif refresh_schedule.lower() == 'daily':
            if 'hour' in refresh_rates:
                hour = refresh_rates['hour']
            if 'minute' in refresh_rates:
                minute = refresh_rates['minute']
            map_area_refresh_params = {
                "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                "type":"daily",
                "nthDay":1,
                "dayOfWeek":0
            }
            refresh_schedule = "0 {m} {hour} * * ?".format(m=minute, hour=hour)
        elif refresh_schedule.lower() == 'weekly':
            if 'hour' in refresh_rates:
                hour = refresh_rates['hour']
            if 'minute' in refresh_rates:
                minute = refresh_rates['minute']
            if 'day_of_week' in refresh_rates:
                dayOfWeek = refresh_rates['day_of_week']
            map_area_refresh_params = {
                "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                "type":"weekly",
                    "nthDay": 1,
                    "dayOfWeek": dayOfWeek
            }
            refresh_schedule = "0 {m} {hour} ? * {dow}".format(m=minute, hour=hour,
                                                               dow=_dow_lu[dayOfWeek])
        elif refresh_schedule.lower() == 'monthly':
            if 'hour' in refresh_rates:
                hour = refresh_rates['hour']
            if 'minute' in refresh_rates:
                minute = refresh_rates['minute']
            if 'nthday' in refresh_rates['nthday']:
                nthday = refresh_rates['nthday']
            if 'day_of_week' in refresh_rates:
                dayOfWeek = refresh_rates['day_of_week']
            map_area_refresh_params = {
                "startDate": int(datetime.datetime.utcnow().timestamp()) * 1000,
                "type":"monthly",
                "nthDay": nthday,
                "dayOfWeek": dayOfWeek,
            }
            refresh_schedule = "0 {m} {hour} ? * {nthday}#{dow}".format(m=minute, hour=hour,
                                                                        nthday=nthday, dow=dayOfWeek)
        else:
            raise ValueError(("Invalid refresh_schedule, value"
                              " can only be Never, Daily, Weekly or Monthly."))
        text = item.get_data()
        text['mapAreas']['mapAreaRefreshParams'] = map_area_refresh_params
        update_items = {
            'clearEmptyFields' : True,
            'text' : json.dumps(text)
        }
        item.update(item_properties=update_items)
        properties = item.properties
        _extent = item.properties['extent']
        _bookmark = None
        update_items = {
            "properties": {
                "extent": _extent,
                "status":"complete",
                "packageRefreshSchedule": refresh_schedule
            }
        }
        item.update(item_properties=update_items)
        try:
            result = self._pm.create_map_area(map_item_id=item.id, future=False)
            return True
        except:
            return False
    #----------------------------------------------------------------------
    def list(self):
        """
        Returns a list of Map Area items related to the current WebMap object.

        .. note::
            Map Area items and the corresponding offline packages cached for each share a relationship of type
            'Area2Package'. You can use this relationship to get the list of package items cached for a particular Map
            Area item. Refer to the Python snippet below for the steps:

            .. code-block:: python

               USAGE EXAMPLE: Finding packages cached for a Map Area item

               from arcgis.mapping import WebMap
               wm = WebMap(a_web_map_item_object)
               all_map_areas = wm.offline_areas.list()  # get all the offline areas for that web map

               area1 = all_map_areas[0]
               area1_packages = area1.related_items('Area2Package','forward')

               for pkg in area1_packages:
                    print(pkg.homepage)  # get the homepage url for each package item.

        :return:
            List of Map Area items related to the current WebMap object
        """
        return self._item.related_items('Map2Area', 'forward')
    #----------------------------------------------------------------------
    def update(self, offline_map_area_items=None, future=False):
        """
        Refreshes existing map area packages associated with the list of map area items specified.
        This process updates the packages with changes made on the source data since the last time those packages were
        created or refreshed.

        .. note::
            - Offline map area functionality is only available if your GIS is ArcGIS Online.
            - You need to be the owner of the web map or an administrator of your GIS.

        ============================     ====================================================================
        **Argument**                     **Description**
        ----------------------------     --------------------------------------------------------------------
        offline_map_area_items           Optional list. Specify one or more Map Area items for which the packages need
                                         to be refreshed. If not specified, this method updates all the packages
                                         associated with all the map area items of the web map.

                                         To get the list of Map Area items related to the WebMap object, call the
                                         `list()` method.
        ----------------------------     --------------------------------------------------------------------
        future                           Optional Boolean.
        ============================     ====================================================================

        :return:
            Dictionary containing update status.

        .. note::
            This method executes silently. To view informative status messages, set the verbosity environment variable
            as shown below:

            .. code-block:: python

               USAGE EXAMPLE: setting verbosity

               from arcgis import env
               env.verbose = True
        """
        # find if 1 or a list of area items is provided
        if isinstance(offline_map_area_items, Item):
            offline_map_area_items = [offline_map_area_items]
        elif isinstance(offline_map_area_items, str):
            offline_map_area_items = [offline_map_area_items]

        # get packages related to the offline area item
        _related_packages = []
        if not offline_map_area_items:  # none specified
            _related_oma_items = self.list()
            for related_oma in _related_oma_items:  # get all offline packages for this web map
                _related_packages.extend(related_oma.related_items('Area2Package', 'forward'))

        else:
            for offline_map_area_item in offline_map_area_items:
                if isinstance(offline_map_area_item, Item):
                    _related_packages.extend(offline_map_area_item.related_items('Area2Package', 'forward'))
                elif isinstance(offline_map_area_item, str):
                    offline_map_area_item = Item(gis=self._gis, itemid=offline_map_area_item)
                    _related_packages.extend(offline_map_area_item.related_items('Area2Package', 'forward'))

        # update each of the packages
        if _related_packages:
            _update_list = [{'itemId': i.id} for i in _related_packages]

            # update the packages
            #from arcgis.geoprocessing._tool import Toolbox
            #pkg_tb = Toolbox(self._url, gis=self._gis)

            #result = pkg_tb.refresh_map_area_package(json.dumps(_update_list,
            #                                                    default=_date_handler))
            job = self._pm.refresh_map_area_package(packages=json.dumps(_update_list), future=True, gis=self._gis)
            if future:
                return job
            return job.result()
        else:
            return None



###########################################################################
class WebScene(collections.OrderedDict):
    """
    Represents a web scene and provides access to its basemaps and operational layers as well
    as functionality to visualize and interact with them.

    If you would like more robust webscene authoring functionality,
    consider using the :class:`~arcgis.widgets.MapView` class. You need to be using a
    Jupyter environment for the MapView class to function properly, but you can
    make copies of WebScenes, add layers using a simple `add_layer()` call,
    adjust the basemaps, save to new webscenes, and more.

    """

    def __init__(self, websceneitem):
        """
        Constructs a WebScene object given its item from ArcGIS Online or Portal.
        """
        if websceneitem.type.lower() != 'web scene':
            raise TypeError("item type must be web scene")
        self.item = websceneitem
        self._gis = websceneitem._gis
        webscenedict = self.item.get_data()
        collections.OrderedDict.__init__(self, webscenedict)

    def _ipython_display_(self, **kwargs):
        from arcgis.widgets import MapView
        mapwidget = MapView(gis=self._gis, item=self.item)
        mapwidget.mode = "3D"
        mapwidget.hide_mode_switch = True
        return mapwidget._ipython_display_(**kwargs)

    def __repr__(self):
        return 'WebScene at ' + self.item._portal.url  + "/home/webscene/viewer.html?webscene=" + self.item.itemid

    def __str__(self):
        return json.dumps(self,
                          default=_date_handler)

    def update(self):
        # with _tempinput(self.__str__()) as tempfilename:
        self.item.update({'text': self.__str__()})

###########################################################################
class VectorTileLayer(Layer):

    def __init__(self, url, gis=None):
        super(VectorTileLayer, self).__init__(url, gis)

    @classmethod
    def fromitem(cls, item):
        if not item.type == 'Vector Tile Service':
            raise TypeError("item must be a type of Vector Tile Service, not " + item.type)

        return cls(item.url, item._gis)

    @property
    def styles(self):
        url = "{url}/styles".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def tile_fonts(self, fontstack, stack_range):
        """This resource returns glyphs in PBF format. The template url for
        this fonts resource is represented in Vector Tile Style resource."""
        url = "{url}/resources/fonts/{fontstack}/{stack_range}.pbf".format(
            url=self._url,
            fontstack=fontstack,
            stack_range=stack_range)
        params = {}
        return self._con.get(path=url,
                             params=params, force_bytes=True)

    # ----------------------------------------------------------------------
    def vector_tile(self, level, row, column):
        """This resource represents a single vector tile for the map. The
        bytes for the tile at the specified level, row and column are
        returned in PBF format. If a tile is not found, an error is returned."""
        url = "{url}/tile/{level}/{row}/{column}.pbf".format(url=self._url,
                                                             level=level,
                                                             row=row,
                                                             column=column)
        params = {}
        return self._con.get(path=url,
                             params=params, try_json=False, force_bytes=True)

    # ----------------------------------------------------------------------
    def tile_sprite(self, out_format="sprite.json"):
        """
        This resource returns sprite image and metadata
        """
        url = "{url}/resources/sprites/{f}".format(url=self._url,
                                                   f=out_format)
        return self._con.get(path=url,
                             params={})

    # ----------------------------------------------------------------------
    @property
    def info(self):
        """This returns relative paths to a list of resource files"""
        url = "{url}/resources/info".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(path=url,
                             params=params)


###########################################################################
class MapImageLayerManager(_GISResource):
    """ allows administration (if access permits) of ArcGIS Online hosted map image layers.
    A map image layer offers access to map and layer content.
    """

    def __init__(self, url, gis=None, map_img_lyr=None):
        if url.split("/")[-1].isdigit():
            url = url.replace(f"/{url.split('/')[-1]}", "")
        super(MapImageLayerManager, self).__init__(url, gis)
        self._ms = map_img_lyr

    # ----------------------------------------------------------------------
    def refresh(self, service_definition=True):
        """
        The refresh operation refreshes a service, which clears the web
        server cache for the service.
        """
        url = self._url + "/MapServer/refresh"
        params = {
            "f": "json",
            "serviceDefinition": service_definition
        }

        res = self._con.post(self._url, params)

        super(MapImageLayerManager, self)._refresh()

        self._ms._refresh()

        return res

    # ----------------------------------------------------------------------
    def cancel_job(self, job_id):
        """
        The cancel job operation supports cancelling a job while update
        tiles is running from a hosted feature service. The result of this
        operation is a response indicating success or failure with error
        code and description.

        Inputs:
           job_id - job id to cancel
        """
        url = self._url + "/jobs/%s/cancel" % job_id
        params = {
            "f": "json"
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def job_statistics(self, job_id):
        """
        Returns the job statistics for the given jobId

        """
        url = self._url + "/jobs/%s" % job_id
        params = {
            "f": "json"
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def import_tiles(self, item,
                     levels=None, extent=None,
                     merge=False, replace=False):
        """

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        item                Required ItemId or Item. The TPK file's item id.
                            This TPK file contains to-be-extracted bundle files
                            which are then merged into an existing cache service.
        ---------------     ----------------------------------------------------
        levels              Optional String / List of integers, The level of details
                            to update. Example: "1,2,10,20" or [1,2,10,20]
        ---------------     ----------------------------------------------------
        extent              Optional String / Dict. The area to update as Xmin, YMin, XMax, YMax
                            example: "-100,-50,200,500" or
                            {'xmin':100, 'ymin':200, 'xmax':105, 'ymax':205}
        ---------------     ----------------------------------------------------
        merge               Optional Boolean. Default is false and applicable to
                            compact cache storage format. It controls whether
                            the bundle files from the TPK file are merged with
                            the one in the existing cached service. Otherwise,
                            the bundle files are overwritten.
        ---------------     ----------------------------------------------------
        replace             Optional Boolean. Default is false, applicable to
                            compact cache storage format and used when
                            merge=true. It controls whether the new tiles will
                            replace the existing ones when merging bundles.
        ===============     ====================================================

        :returns: Dict

        """
        params = {
            'f' : 'json',
            'sourceItemId' : None,
            'extent' : extent,
            'levels' : levels,
            'mergeBundle' : merge,
            'replaceTiles' : replace
        }
        if isinstance(item, str):
            params['sourceItemId'] = item
        elif isinstance(item, Item):
            params['sourceItemId'] = item.itemid
        else:
            raise ValueError("The `item` must be a string or Item")
        url = self._url + "/importTiles"
        res = self._con.post(url, params)
        return res
    #----------------------------------------------------------------------
    def update_tiles(self, levels=None, extent=None):
        """
        The starts tile generation for ArcGIS Online.  The levels of detail
        and the extent are needed to determine the area where tiles need
        to be rebuilt.

        ..Note: This operation is for ArcGIS Online only.

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        levels              Optional String / List of integers, The level of details
                            to update. Example: "1,2,10,20" or [1,2,10,20]
        ---------------     ----------------------------------------------------
        extent              Optional String / Dict. The area to update as Xmin, YMin, XMax, YMax
                            example: "-100,-50,200,500" or
                            {'xmin':100, 'ymin':200, 'xmax':105, 'ymax':205}
        ===============     ====================================================

        :returns:
           Dictionary. If the product is not ArcGIS Online tile service, the
           result will be None.
        """
        if self._gis._portal.is_arcgisonline:
            url = "%s/updateTiles" % self._url
            params = {
                "f": "json"
            }
            if levels:
                if isinstance(levels, list):
                    levels = ",".join(str(e) for e in levels)
                params['levels'] = levels
            if extent:
                if isinstance(extent, dict):
                    extent2 = "{},{},{},{}".format(extent['xmin'], extent['ymin'],
                                                   extent['xmax'], extent['ymax'])
                    extent = extent2
                params['extent'] = extent
            return self._con.post(url, params)
        return None
    #----------------------------------------------------------------------
    @property
    def rerun_job(self, job_id, code):
        """
        The rerun job operation supports re-running a canceled job from a
        hosted map service. The result of this operation is a response
        indicating success or failure with error code and description.

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        code                required string, parameter used to re-run a given
                            jobs with a specific error
                            code: ALL | ERROR | CANCELED
        ---------------     ----------------------------------------------------
        job_id              required string, job to reprocess
        ===============     ====================================================

        :returns:
           boolean or dictionary
        """
        url = self._url + "/jobs/%s/rerun" % job_id
        params = {
            "f" : "json",
            "rerun": code
        }
        return self._con.post(url, params)
    # ----------------------------------------------------------------------
    def edit_tile_service(self,
                          service_definition=None,
                          min_scale=None,
                          max_scale=None,
                          source_item_id=None,
                          export_tiles_allowed=False,
                          max_export_tile_count=100000):
        """
        This operation updates a Tile Service's properties

        Inputs:
           service_definition - updates a service definition
           min_scale - sets the services minimum scale for caching
           max_scale - sets the service's maximum scale for caching
           source_item_id - The Source Item ID is the GeoWarehouse Item ID of the map service
           export_tiles_allowed - sets the value to let users export tiles
           max_export_tile_count - sets the maximum amount of tiles to be exported
             from a single call.
        """
        params = {
            "f": "json",
        }
        if not service_definition is None:
            params["serviceDefinition"] = service_definition
        if not min_scale is None:
            params['minScale'] = float(min_scale)
        if not max_scale is None:
            params['maxScale'] = float(max_scale)
        if not source_item_id is None:
            params["sourceItemId"] = source_item_id
        if not export_tiles_allowed is None:
            params["exportTilesAllowed"] = export_tiles_allowed
        if not max_export_tile_count is None:
            params["maxExportTileCount"] = int(max_export_tile_count)
        url = self._url + "/edit"
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def delete_tiles(self, levels, extent=None ):
        """
        Deletes tiles for the current cache

        ===============     ====================================================
        **Argument**        **Description**
        ---------------     ----------------------------------------------------
        extent              optional dictionary,  If specified, the tiles within
                            this extent will be deleted or will be deleted based
                            on the service's full extent.
                            Example:
                            6224324.092137296,487347.5253569535,
                            11473407.698535524,4239488.369818687
                            the minx, miny, maxx, maxy values or,
                            {"xmin":6224324.092137296,"ymin":487347.5253569535,
                            "xmax":11473407.698535524,"ymax":4239488.369818687,
                            "spatialReference":{"wkid":102100}} the JSON
                            representation of the Extent object.
        ---------------     ----------------------------------------------------
        levels              required string, The level to delete.
                            Example, 0-5,10,11-20 or 1,2,3 or 0-5
        ===============     ====================================================

        :returns:
           dictionary
        """
        params = {
            "f" : "json",
            "levels" : levels,
        }
        if extent:
            params['extent'] = extent
        url = self._url + "/deleteTiles"
        return self._con.post(url, params)


###########################################################################
class MapImageLayer(Layer):
    """
    MapImageLayer allows you to display and analyze data from sublayers defined in a map service, exporting images
    instead of features. Map service images are dynamically generated on the server based on a request, which includes
    an LOD (level of detail), a bounding box, dpi, spatial reference and other options. The exported image is of the
    entire map extent specified.

    MapImageLayer does not display tiled images. To display tiled map service layers, see TileLayer.
    """

    def __init__(self, url, gis=None):
        """
        .. Creates a map image layer given a URL. The URL will typically look like the following.

            https://<hostname>/arcgis/rest/services/<service-name>/MapServer

        :param url: the layer location
        :param gis: the GIS to which this layer belongs
        """
        super(MapImageLayer, self).__init__(url, gis)

        self._populate_layers()
        self._admin = None
        try:
            from arcgis.gis.server._service._adminfactory import AdminServiceGen
            self.service = AdminServiceGen(service=self, gis=gis)
        except: pass

    @classmethod
    def fromitem(cls, item):
        if not item.type == 'Map Service':
            raise TypeError("item must be a type of Map Service, not " + item.type)
        return cls(item.url, item._gis)

    @property
    def _lyr_dict(self):
        url = self.url

        if "lods" in self.properties:
            lyr_dict =  { 'type' : 'ArcGISTiledMapServiceLayer', 'url' : url }

        else:
            lyr_dict =  { 'type' : type(self).__name__, 'url' : url }

        if self._token is not None :
            lyr_dict['serviceToken'] = self._token or self._con.token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            token = self._token or self._con.token
            url += '?token=' + token

        if "lods" in self.properties:
            lyr_dict =  { 'type' : 'ArcGISTiledMapServiceLayer', 'url' : url }
        else:
            lyr_dict =  { 'type' : type(self).__name__, 'url' : url }

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

    def _populate_layers(self):
        layers = []
        tables = []
        if self.properties.layers:
            for lyr in self.properties.layers:
                if 'subLayerIds' in lyr and lyr.subLayerIds is not None: # Group Layer
                    lyr = Layer(self.url + '/' + str(lyr.id), self._gis)
                else:
                    lyr = arcgis.mapping._msl.MapServiceLayer(self.url + '/' + str(lyr.id), self._gis)
                layers.append(lyr)
        if self.properties.tables:
            for lyr in self.properties.tables:
                lyr = arcgis.mapping._msl.MapServiceLayer(self.url + '/' + str(lyr.id), self._gis, self)
                tables.append(lyr)
        # fsurl = self.url + '/layers'
        # params = { "f" : "json" }
        # allayers = self._con.post(fsurl, params)

        # for layer in allayers['layers']:
        #    layers.append(FeatureLayer(self.url + '/' + str(layer['id']), self._gis))

        # for table in allayers['tables']:
        #    tables.append(FeatureLayer(self.url + '/' + str(table['id']), self._gis))

        self.layers = layers
        self.tables = tables
    def _str_replace(self, mystring, rd):
        """Replaces a value based on a key/value pair where the
        key is the text to replace and the value is the new value.

        The find/replace is case insensitive.

        """
        import re
        patternDict = {}
        myDict = {}
        for key,value in rd.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            patternDict[value] = pattern
        for key in patternDict:
            regex_obj = patternDict[key]
            mystring = regex_obj.sub(key, mystring)
        return mystring

    @property
    def manager(self):
        if self._admin is None:
            """accesses the administration service"""
            if self._gis._portal.is_arcgisonline:
                rd = {'/rest/services/': '/rest/admin/services/'}
            else:
                rd = {"/rest/" : "/admin/",
                      "/MapServer" : ".MapServer"}
            adminURL = self._str_replace(self._url, rd)
            #res = search("/rest/", url).span()
            #addText = "admin/"
            #part1 = url[:res[1]]
            #part2 = url[res[1]:]
            #adminURL = url.replace("/rest/", "/admin/").replace("/MapServer", ".MapServer")#"%s%s%s" % (part1, addText, part2)
            if adminURL.split("/")[-1].isdigit():
                url = adminURL.replace(f'/{adminURL.split("/")[-1]}', "")
            self._admin = MapImageLayerManager(adminURL, self._gis, self)
        return self._admin

    #----------------------------------------------------------------------
    def create_dynamic_layer(self, layer):
        """
        A dynamic layer / table method represents a single layer / table
        of a map service published by ArcGIS Server or of a registered
        workspace. This resource is supported only when the map image layer
        supports dynamic layers, as indicated by supportsDynamicLayers on
        the map image layer properties.

        =================     ====================================================================
        **Argument**          **Description**
        -----------------     --------------------------------------------------------------------
        layer                 required dict.  Dynamic layer/table source definition.
                              Syntax:
                              {
                                "id": <layerOrTableId>,
                                "source": <layer source>, //required
                                "definitionExpression": "<definitionExpression>",
                                "drawingInfo":
                                {
                                  "renderer": <renderer>,
                                  "transparency": <transparency>,
                                  "scaleSymbols": <true,false>,
                                  "showLabels": <true,false>,
                                  "labelingInfo": <labeling info>
                                },
                                "layerTimeOptions": //supported only for time enabled map layers
                                {
                                  "useTime" : <true,false>,
                                  "timeDataCumulative" : <true,false>,
                                  "timeOffset" : <timeOffset>,
                                  "timeOffsetUnits" : "<esriTimeUnitsCenturies,esriTimeUnitsDays,
                                                    esriTimeUnitsDecades,esriTimeUnitsHours,
                                                    esriTimeUnitsMilliseconds,esriTimeUnitsMinutes,
                                                    esriTimeUnitsMonths,esriTimeUnitsSeconds,
                                                    esriTimeUnitsWeeks,esriTimeUnitsYears |
                                                    esriTimeUnitsUnknown>"
                                }
                              }
        =================     ====================================================================

        :returns: arcgis.features.FeatureLayer or None (if not enabled)

        """
        if "supportsDynamicLayers" in self.properties and \
           self.properties["supportsDynamicLayers"]:
            from urllib.parse import urlencode
            url = "%s/dynamicLayer" % self._url
            d = urlencode(layer)
            url += "?layer=%s" % d
            return arcgis.features.FeatureLayer(url=url, gis=self._gis, dynamic_layer=layer)
        return None
    # ----------------------------------------------------------------------
    @property
    def kml(self):
        """returns the KML file for the layer"""
        url = "{url}/kml/mapImage.kmz".format(url=self._url)
        return self._con.get(url, {"f": 'json'},
                             file_name="mapImage.kmz",
                             out_folder=tempfile.gettempdir())

    # ----------------------------------------------------------------------
    @property
    def item_info(self):
        """returns the service's item's infomation"""
        url = "{url}/info/iteminfo".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(url, params)

    #----------------------------------------------------------------------
    @property
    def legend(self):
        """
        The legend resource represents a map service's legend. It returns
        the legend information for all layers in the service. Each layer's
        legend information includes the symbol images and labels for each
        symbol. Each symbol is an image of size 20 x 20 pixels at 96 DPI.
        Additional information for each layer such as the layer ID, name,
        and min and max scales are also included.

        The legend symbols include the base64 encoded imageData as well as
        a url that could be used to retrieve the image from the server.
        """
        url = "%s/legend" % self._url
        return self._con.get(path=url, params={'f': 'json'})

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """returns the service's XML metadata file"""
        url = "{url}/info/metadata".format(url=self._url)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def thumbnail(self, out_path=None):
        """if present, this operation will download the image to local disk"""
        if out_path is None:
            out_path = tempfile.gettempdir()
        url = "{url}/info/thumbnail".format(url=self._url)
        params = {"f": "json"}
        if out_path is None:
            out_path = tempfile.gettempdir()
        return self._con.get(url,
                             params,
                             out_folder=out_path,
                             file_name="thumbnail.png")

    # ----------------------------------------------------------------------
    def identify(self,
                 geometry,
                 map_extent,
                 image_display,
                 geometry_type="Point",
                 sr=None,
                 layer_defs=None,
                 time_value=None,
                 time_options=None,
                 layers="all",
                 tolerance=None,
                 return_geometry=True,
                 max_offset=None,
                 precision=4,
                 dynamic_layers=None,
                 return_z=False,
                 return_m=False,
                 gdb_version=None,
                 return_unformatted=False,
                 return_field_name=False,
                 transformations=None,
                 map_range_values=None,
                 layer_range_values=None,
                 layer_parameters=None,
                 **kwargs):

        """
        The identify operation is performed on a map service resource
        to discover features at a geographic location. The result of this
        operation is an identify results resource. Each identified result
        includes its name, layer ID, layer name, geometry and geometry type,
        and other attributes of that result as name-value pairs.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        geometry               required Geometry or list. The geometry to identify on. The type of
                               the geometry is specified by the geometryType parameter. The
                               structure of the geometries is same as the structure of the JSON
                               geometry objects returned by the API. In addition to the JSON
                               structures, for points and envelopes, you can specify the geometries
                               with a simpler comma-separated syntax.
        ------------------     --------------------------------------------------------------------
        geometry_type          required string.The type of geometry specified by the geometry
                               parameter. The geometry type could be a point, line, polygon, or an
                               envelope.
                               Values: Point,Multipoint,Polyline,Polygon,Envelope
        ------------------     --------------------------------------------------------------------
        sr                     optional dict, string, or SpatialReference. The well-known ID of the
                               spatial reference of the input and output geometries as well as the
                               map_extent. If sr is not specified, the geometry and the map_extent
                               are assumed to be in the spatial reference of the map, and the
                               output geometries are also in the spatial reference of the map.
        ------------------     --------------------------------------------------------------------
        layer_defs             optional dict. Allows you to filter the features of individual
                               layers in the exported map by specifying definition expressions for
                               those layers. Definition expression for a layer that is
                               published with the service will be always honored.
        ------------------     --------------------------------------------------------------------
        time_value             optional list. The time instant or the time extent of the features
                               to be identified.
        ------------------     --------------------------------------------------------------------
        time_options           optional dict. The time options per layer. Users can indicate
                               whether or not the layer should use the time extent specified by the
                               time parameter or not, whether to draw the layer features
                               cumulatively or not and the time offsets for the layer.
        ------------------     --------------------------------------------------------------------
        layers                 optional string. The layers to perform the identify operation on.
                               There are three ways to specify which layers to identify on:
                                - top: Only the top-most layer at the specified location.
                                - visible: All visible layers at the specified location.
                                - all: All layers at the specified location.
        ------------------     --------------------------------------------------------------------
        tolerance              optional integer. The distance in screen pixels from the specified
                               geometry within which the identify should be performed. The value for
                               the tolerance is an integer.
        ------------------     --------------------------------------------------------------------
        map_extent             required string. The extent or bounding box of the map currently
                               being viewed.
        ------------------     --------------------------------------------------------------------
        image_display          optional string. The screen image display parameters (width, height,
                               and DPI) of the map being currently viewed. The mapExtent and the
                               image_display parameters are used by the server to determine the
                               layers visible in the current extent. They are also used to
                               calculate the distance on the map to search based on the tolerance
                               in screen pixels.
                               Syntax: <width>, <height>, <dpi>
        ------------------     --------------------------------------------------------------------
        return_geometry        optional boolean. If true, the resultset will include the geometries
                               associated with each result. The default is true.
        ------------------     --------------------------------------------------------------------
        max_offset             optional integer. This option can be used to specify the maximum
                               allowable offset to be used for generalizing geometries returned by
                               the identify operation.
        ------------------     --------------------------------------------------------------------
        precision              optional integer. This option can be used to specify the number of
                               decimal places in the response geometries returned by the identify
                               operation. This applies to X and Y values only (not m or z-values).
        ------------------     --------------------------------------------------------------------
        dynamic_layers         optional dict. Use dynamicLayers property to reorder layers and
                               change the layer data source. dynamicLayers can also be used to add
                               new layer that was not defined in the map used to create the map
                               service. The new layer should have its source pointing to one of the
                               registered workspaces that was defined at the time the map service
                               was created.
                               The order of dynamicLayers array defines the layer drawing order.
                               The first element of the dynamicLayers is stacked on top of all
                               other layers. When defining a dynamic layer, source is required.
        ------------------     --------------------------------------------------------------------
        return_z               optional boolean. If true, Z values will be included in the results
                               if the features have Z values. Otherwise, Z values are not returned.
                               The default is false.
        ------------------     --------------------------------------------------------------------
        return_m               optional boolean.If true, M values will be included in the results
                               if the features have M values. Otherwise, M values are not returned.
                               The default is false.
        ------------------     --------------------------------------------------------------------
        gdb_version            optional string. Switch map layers to point to an alternate
                               geodatabase version.
        ------------------     --------------------------------------------------------------------
        return_unformatted     optional boolean. If true, the values in the result will not be
                               formatted i.e. numbers will returned as is and dates will be
                               returned as epoch values. The default is False.
        ------------------     --------------------------------------------------------------------
        return_field_name      optional boolean. Default is False. If true, field names will be
                               returned instead of field aliases.
        ------------------     --------------------------------------------------------------------
        transformations        optional list. Use this parameter to apply one or more datum
                               transformations to the map when sr is different than the map
                               service's spatial reference. It is an array of transformation
                               elements.
                               Transformations specified here are used to project features from
                               layers within a map service to sr.
        ------------------     --------------------------------------------------------------------
        map_range_values       optional list. Allows for the filtering features in the exported map
                               from all layer that are within the specified range instant or extent.
        ------------------     --------------------------------------------------------------------
        layer_range_values     optional list. Allows for the filtering of features for each
                               individual layer that are within the specified range instant or
                               extent.
        ------------------     --------------------------------------------------------------------
        layer_parameters       optional list. Allows for the filtering of the features of
                               individual layers in the exported map by specifying value(s) to an
                               array of pre-authored parameterized filters for those layers. When
                               value is not specified for any parameter in a request, the default
                               value, that is assigned during authoring time, gets used instead.
        =================     ====================================================================

        :returns: dictionary
        """

        if geometry_type.find("esriGeometry") == -1:
            geometry_type = "esriGeometry" + geometry_type
        if sr is None:
            sr = kwargs.pop('sr', None)
        if layer_defs is None:
            layer_defs = kwargs.pop('layerDefs', None)
        if time_value is None:
            time_value = kwargs.pop('layerTimeOptions', None)
        if return_geometry is None:
            return_geometry = kwargs.pop('returnGeometry', True)
        if return_m is None:
            return_m = kwargs.pop('returnM', False)
        if return_z is None:
            return_z = kwargs.pop('returnZ', False)
        if max_offset is None:
            max_offset = kwargs.pop('maxAllowableOffset', None)
        if precision is None:
            precision = kwargs.pop('geometryPrecision', None)
        if dynamic_layers is None:
            dynamic_layers = kwargs.pop('dynamicLayers', None)
        if gdb_version is None:
            gdb_version = kwargs.pop('gdbVersion', None)

        params = {'f': 'json',
                  'geometry': geometry,
                  'geometryType': geometry_type,
                  'tolerance': tolerance,
                  'mapExtent': map_extent,
                  'imageDisplay': image_display
                  }
        if sr:
            params['sr'] = sr
        if layer_defs:
            params['layerDefs'] = layer_defs
        if time_value:
            params['time'] = time_value
        if time_options:
            params['layerTimeOptions'] = time_options
        if layers:
            params['layers'] = layers
        if tolerance:
            params['tolerance'] = tolerance
        if return_geometry is not None:
            params['returnGeometry'] = return_geometry
        if max_offset:
            params['maxAllowableOffset'] = max_offset
        if precision:
            params['geometryPrecision'] = precision
        if dynamic_layers:
            params['dynamicLayers'] = dynamic_layers
        if return_m is not None:
            params['returnM'] = return_m
        if return_z is not None:
            params['returnZ'] = return_z
        if gdb_version:
            params['gdbVersion'] = gdb_version
        if return_unformatted is not None:
            params['returnUnformattedValues'] = return_unformatted
        if return_field_name is not None:
            params['returnFieldName'] = return_field_name
        if transformations:
            params['datumTransformations'] = transformations
        if map_range_values:
            params['mapRangeValues'] = map_range_values
        if layer_range_values:
            params['layerRangeValues'] = layer_range_values
        if layer_parameters:
            params['layerParameterValues'] = layer_parameters
        identifyURL = "{url}/identify".format(url=self._url)
        return self._con.post(identifyURL, params)

    # ----------------------------------------------------------------------
    def find(self,
             search_text,
             layers,
             contains=True,
             search_fields=None,
             sr=None,
             layer_defs=None,
             return_geometry=True,
             max_offset=None,
             precision=None,
             dynamic_layers=None,
             return_z=False,
             return_m=False,
             gdb_version=None,
             return_unformatted=False,
             return_field_name=False,
             transformations=None,
             map_range_values=None,
             layer_range_values=None,
             layer_parameters=None,
             **kwargs
             ):
        """
        performs the map service find operation

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        search_text            required string.The search string. This is the text that is searched
                               across the layers and fields the user specifies.
        ------------------     --------------------------------------------------------------------
        layers                 optional string. The layers to perform the identify operation on.
                               There are three ways to specify which layers to identify on:
                                - top: Only the top-most layer at the specified location.
                                - visible: All visible layers at the specified location.
                                - all: All layers at the specified location.
        ------------------     --------------------------------------------------------------------
        contains               optional boolean. If false, the operation searches for an exact
                               match of the search_text string. An exact match is case sensitive.
                               Otherwise, it searches for a value that contains the search_text
                               provided. This search is not case sensitive. The default is true.
        ------------------     --------------------------------------------------------------------
        search_fields          optional string. List of field names to look in.
        ------------------     --------------------------------------------------------------------
        sr                     optional dict, string, or SpatialReference. The well-known ID of the
                               spatial reference of the input and output geometries as well as the
                               map_extent. If sr is not specified, the geometry and the map_extent
                               are assumed to be in the spatial reference of the map, and the
                               output geometries are also in the spatial reference of the map.
        ------------------     --------------------------------------------------------------------
        layer_defs             optional dict. Allows you to filter the features of individual
                               layers in the exported map by specifying definition expressions for
                               those layers. Definition expression for a layer that is
                               published with the service will be always honored.
        ------------------     --------------------------------------------------------------------
        return_geometry        optional boolean. If true, the resultset will include the geometries
                               associated with each result. The default is true.
        ------------------     --------------------------------------------------------------------
        max_offset             optional integer. This option can be used to specify the maximum
                               allowable offset to be used for generalizing geometries returned by
                               the identify operation.
        ------------------     --------------------------------------------------------------------
        precision              optional integer. This option can be used to specify the number of
                               decimal places in the response geometries returned by the identify
                               operation. This applies to X and Y values only (not m or z-values).
        ------------------     --------------------------------------------------------------------
        dynamic_layers         optional dict. Use dynamicLayers property to reorder layers and
                               change the layer data source. dynamicLayers can also be used to add
                               new layer that was not defined in the map used to create the map
                               service. The new layer should have its source pointing to one of the
                               registered workspaces that was defined at the time the map service
                               was created.
                               The order of dynamicLayers array defines the layer drawing order.
                               The first element of the dynamicLayers is stacked on top of all
                               other layers. When defining a dynamic layer, source is required.
        ------------------     --------------------------------------------------------------------
        return_z               optional boolean. If true, Z values will be included in the results
                               if the features have Z values. Otherwise, Z values are not returned.
                               The default is false.
        ------------------     --------------------------------------------------------------------
        return_m               optional boolean.If true, M values will be included in the results
                               if the features have M values. Otherwise, M values are not returned.
                               The default is false.
        ------------------     --------------------------------------------------------------------
        gdb_version            optional string. Switch map layers to point to an alternate
                               geodatabase version.
        ------------------     --------------------------------------------------------------------
        return_unformatted     optional boolean. If true, the values in the result will not be
                               formatted i.e. numbers will returned as is and dates will be
                               returned as epoch values.
        ------------------     --------------------------------------------------------------------
        return_field_name      optional boolean. If true, field names will be returned instead of
                               field aliases.
        ------------------     --------------------------------------------------------------------
        transformations        optional list. Use this parameter to apply one or more datum
                               transformations to the map when sr is different than the map
                               service's spatial reference. It is an array of transformation
                               elements.
        ------------------     --------------------------------------------------------------------
        map_range_values       optional list. Allows you to filter features in the exported map
                               from all layer that are within the specified range instant or
                               extent.
        ------------------     --------------------------------------------------------------------
        layer_range_values     optional dictionary. Allows you to filter features for each
                               individual layer that are within the specified range instant or
                               extent. Note: Check range infos at the layer resources for the
                               available ranges.
        ------------------     --------------------------------------------------------------------
        layer_parameters       optional list. Allows you to filter the features of individual
                               layers in the exported map by specifying value(s) to an array of
                               pre-authored parameterized filters for those layers. When value is
                               not specified for any parameter in a request, the default value,
                               that is assigned during authoring time, gets used instead.
        ==================     ====================================================================

        :returns: dictionary
        """
        url = "{url}/find".format(url=self._url)
        params = {
            "f": "json",
            "searchText": search_text,
            "contains": contains,
        }
        if search_fields:
            params['searchFields'] = search_fields
        if sr:
            params['sr'] = sr
        if layer_defs:
            params['layerDefs'] = layer_defs
        if return_geometry is not None:
            params['returnGeometry'] = return_geometry
        if max_offset:
            params['maxAllowableOffset'] = max_offset
        if precision:
            params['geometryPrecision'] = precision
        if dynamic_layers:
            params['dynamicLayers'] = dynamic_layers
        if return_z is not None:
            params['returnZ'] = return_z
        if return_m is not None:
            params['returnM'] = return_m
        if gdb_version:
            params['gdbVersion'] = gdb_version
        if layers:
            params['layers'] = layers
        if return_unformatted is not None:
            params['returnUnformattedValues'] = return_unformatted
        if return_field_name is not None:
            params['returnFieldName'] = return_field_name
        if transformations:
            params['datumTransformations'] = transformations
        if map_range_values:
            params['mapRangeValues'] = map_range_values
        if layer_range_values:
            params['layerRangeValues'] = layer_range_values
        if layer_parameters:
            params['layerParameterValues'] = layer_parameters
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                params[k] = v
        res = self._con.post(path=url,
                             postdata=params,
                             )
        return res

    # ----------------------------------------------------------------------
    def generate_kml(self, save_location, name, layers, options="composite"):
        """
        The generateKml operation is performed on a map service resource.
        The result of this operation is a KML document wrapped in a KMZ
        file. The document contains a network link to the KML Service
        endpoint with properties and parameters you specify.

        =================     ====================================================================
        **Argument**          **Description**
        -----------------     --------------------------------------------------------------------
        save_location         required string. Save folder.
        -----------------     --------------------------------------------------------------------
        name                  The name of the resulting KML document. This is the name that
                              appears in the Places panel of Google Earth.
        -----------------     --------------------------------------------------------------------
        layers                required string. the layers to perform the generateKML operation on.
                              The layers are specified as a comma-separated list of layer ids.
        -----------------     --------------------------------------------------------------------
        options               required string. The layer drawing options. Based on the option
                              chosen, the layers are drawn as one composite image, as separate
                              images, or as vectors. When the KML capability is enabled, the
                              ArcGIS Server administrator has the option of setting the layer
                              operations allowed. If vectors are not allowed, then the caller will
                              not be able to get vectors. Instead, the caller receives a single
                              composite image.
                              values: composite, separateImage, nonComposite
        =================     ====================================================================

        :returns: string to file path

        """
        kmlURL = self._url + "/generateKml"
        params = {
            "f": "json",
            'docName': name,
            'layers': layers,
            'layerOptions': options
        }
        return self._con.get(kmlURL, params,
                             out_folder=save_location,
                             )
    # ----------------------------------------------------------------------
    def export_map(self,
                   bbox,
                   bbox_sr=None,
                   size="600,550",
                   dpi=200,
                   image_sr=None,
                   image_format="png",
                   layer_defs=None,
                   layers=None,
                   transparent=False,
                   time_value=None,
                   time_options=None,
                   dynamic_layers=None,
                   gdb_version=None,
                   scale=None,
                   rotation=None,
                   transformation=None,
                   map_range_values=None,
                   layer_range_values=None,
                   layer_parameter=None,
                   f="json",
                   save_folder=None,
                   save_file=None,
                   **kwargs):
        """
        The export operation is performed on a map service resource.
        The result of this operation is a map image resource. This
        resource provides information about the exported map image such
        as its URL, its width and height, extent and scale.

        ==================     ====================================================================
        **Argument**          **Description**
        ------------------     --------------------------------------------------------------------
        bbox                   required string. The extent (bounding box) of the exported image.
                               Unless the bbox_sr parameter has been specified, the bbox is assumed
                               to be in the spatial reference of the map.
                               Example: bbox="-104,35.6,-94.32,41"
        ------------------     --------------------------------------------------------------------
        bbox_sr                optional integer, SpatialReference. spatial reference of the bbox.
        ------------------     --------------------------------------------------------------------
        size                   optional string. size - size of image in pixels
        ------------------     --------------------------------------------------------------------
        dpi                    optional integer. dots per inch
        ------------------     --------------------------------------------------------------------
        image_sr               optional integer, SpatialReference. spatial reference of the output
                               image
        ------------------     --------------------------------------------------------------------
        image_format           optional string. The format of the exported image.
                               The default format is .png.
                               Values: png | png8 | png24 | jpg | pdf | bmp | gif
                                       | svg | svgz | emf | ps | png32
        ------------------     --------------------------------------------------------------------
        layer_defs             optional dict. Allows you to filter the features of individual
                               layers in the exported map by specifying definition expressions for
                               those layers. Definition expression for a layer that is
                               published with the service will be always honored.
        ------------------     --------------------------------------------------------------------
        layers                 optional string. Determines which layers appear on the exported map.
                               There are four ways to specify which layers are shown:
                                 show: Only the layers specified in this list will
                                       be exported.
                                 hide: All layers except those specified in this
                                       list will be exported.
                                 include: In addition to the layers exported by
                                          default, the layers specified in this list
                                          will be exported.
                                 exclude: The layers exported by default excluding
                                          those specified in this list will be
                                          exported.
        ------------------     --------------------------------------------------------------------
        transparent            optional boolean. If true, the image will be exported with the
                               background color of the map set as its transparent color. The
                               default is false. Only the .png and .gif formats support
                               transparency.
        ------------------     --------------------------------------------------------------------
        time_value             optional list. The time instant or the time extent of the features
                               to be identified.
        ------------------     --------------------------------------------------------------------
        time_options           optional dict. The time options per layer. Users can indicate
                               whether or not the layer should use the time extent specified by the
                               time parameter or not, whether to draw the layer features
                               cumulatively or not and the time offsets for the layer.
        ------------------     --------------------------------------------------------------------
        dynamic_layers         optional dict. Use dynamicLayers property to reorder layers and
                               change the layer data source. dynamicLayers can also be used to add
                               new layer that was not defined in the map used to create the map
                               service. The new layer should have its source pointing to one of the
                               registered workspaces that was defined at the time the map service
                               was created.
                               The order of dynamicLayers array defines the layer drawing order.
                               The first element of the dynamicLayers is stacked on top of all
                               other layers. When defining a dynamic layer, source is required.
        ------------------     --------------------------------------------------------------------
        gdb_version            optional string. Switch map layers to point to an alternate
                               geodatabase version.
        ------------------     --------------------------------------------------------------------
        scale                  optional float. Use this parameter to export a map image at a
                               specific map scale, with the map centered around the center of the
                               specified bounding box (bbox)
        ------------------     --------------------------------------------------------------------
        rotation               optional float. Use this parameter to export a map image rotated at
                               a specific angle, with the map centered around the center of the
                               specified bounding box (bbox). It could be positive or negative
                               number.
        ------------------     --------------------------------------------------------------------
        transformations        optional list. Use this parameter to apply one or more datum
                               transformations to the map when sr is different than the map
                               service's spatial reference. It is an array of transformation
                               elements.
        ------------------     --------------------------------------------------------------------
        map_range_values       optional list. Allows you to filter features in the exported map
                               from all layer that are within the specified range instant or
                               extent.
        ------------------     --------------------------------------------------------------------
        layer_range_values     optional dictionary. Allows you to filter features for each
                               individual layer that are within the specified range instant or
                               extent. Note: Check range infos at the layer resources for the
                               available ranges.
        ------------------     --------------------------------------------------------------------
        layer_parameter        optional list. Allows you to filter the features of individual
                               layers in the exported map by specifying value(s) to an array of
                               pre-authored parameterized filters for those layers. When value is
                               not specified for any parameter in a request, the default value,
                               that is assigned during authoring time, gets used instead.
        ==================     ====================================================================

        :return: string, image of the map.
        """

        params = {
        }
        params["f"] = f
        params['bbox'] = bbox
        if bbox_sr:
            params['bboxSR'] = bbox_sr
        if dpi is not None:
            params['dpi'] = dpi
        if size is not None:
            params['size'] = size
        if image_sr is not None and \
           isinstance(image_sr, int):
            params['imageSR'] = {'wkid': image_sr}
        if image_format is not None:
            params['format'] = image_format
        if layer_defs is not None:
            params['layerDefs'] = layer_defs
        if layers is not None:
            params['layers'] = layers
        if transparent is not None:
            params['transparent'] = transparent
        if time_value is not None:
            params['time'] = time_value
        if time_options is not None:
            params['layerTimeOptions'] = time_options
        if dynamic_layers is not None:
            params['dynamicLayers'] = dynamic_layers
        if scale is not None:
            params['mapScale'] = scale
        if rotation is not None:
            params['rotation'] = rotation
        if gdb_version is not None:
            params['gdbVersion'] = gdb_version
        if transformation is not None:
            params['datumTransformations'] = transformation
        if map_range_values is not None:
            params['mapRangeValues'] = map_range_values
        if layer_range_values is not None:
            params['layerRangeValues'] = layer_range_values
        if layer_parameter:
            params['layerParameterValues'] = layer_parameter
        url = self._url + "/export"
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                params[k] = v
        #return self._con.get(exportURL, params)

        if f == "json":
            return self._con.post(url, params)
        elif f == "image":
            if save_folder is not None and save_file is not None:
                return self._con.post(url, params,
                                      out_folder=save_folder, try_json=False,
                                      file_name=save_file)
            else:
                return self._con.post(url, params,
                                      try_json=False, force_bytes=True)
        elif f == "kmz":
            return self._con.post(url, params,
                                  out_folder=save_folder,
                                  file_name=save_file)
        else:
            print('Unsupported output format')

    # ----------------------------------------------------------------------
    def estimate_export_tiles_size(self,
                                   export_by,
                                   levels,
                                   tile_package=False,
                                   export_extent="DEFAULTEXTENT",
                                   area_of_interest=None,
                                   asynchronous=True,
                                   **kwargs):
        """
        The estimate_export_tiles_size method is an asynchronous task that
        allows estimation of the size of the tile package or the cache data
        set that you download using the Export Tiles operation. This
        operation can also be used to estimate the tile count in a tile
        package and determine if it will exceced the maxExportTileCount
        limit set by the administrator of the service. The result of this
        operation is Map Service Job. This job response contains reference
        to Map Service Result resource that returns the total size of the
        cache to be exported (in bytes) and the number of tiles that will
        be exported.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        tile_package           optional boolean. Allows estimating the size for either a tile
                               package or a cache raster data set. Specify the value true for tile
                               packages format and false for Cache Raster data set. The default
                               value is False
        ------------------     --------------------------------------------------------------------
        levels                 required string. Specify the tiled service levels for which you want
                               to get the estimates. The values should correspond to Level IDs,
                               cache scales or the Resolution as specified in export_by parameter.
                               The values can be comma separated values or a range.
                               Example 1: 1,2,3,4,5,6,7,8,9
                               Example 2: 1-4,7-9
        ------------------     --------------------------------------------------------------------
        export_by              required string. The criteria that will be used to select the tile
                               service levels to export. The values can be Level IDs, cache scales
                               or the Resolution (in the case of image services).
                               Values: LevelID, Resolution, Scale
        ------------------     --------------------------------------------------------------------
        export_extent          The extent (bounding box) of the tile package or the cache dataset
                               to be exported. If extent does not include a spatial reference, the
                               extent values are assumed to be in the spatial reference of the map.
                               The default value is full extent of the tiled map service.
                               Syntax: <xmin>, <ymin>, <xmax>, <ymax>
                               Example: -104,35.6,-94.32,41
        ------------------     --------------------------------------------------------------------
        area_of_interest       optiona dictionary or Polygon. This allows exporting tiles within
                               the specified polygon areas. This parameter supersedes extent
                               parameter.
                               Example: { "features": [{"geometry":{"rings":[[[-100,35],
                                          [-100,45],[-90,45],[-90,35],[-100,35]]],
                                          "spatialReference":{"wkid":4326}}}]}
        ------------------     --------------------------------------------------------------------
        asynchronous           optional boolean. The estimate function is run asynchronously
                               requiring the tool status to be checked manually to force it to
                               run synchronously the tool will check the status until the
                               estimation completes.  The default is True, which means the status
                               of the job and results need to be checked manually.  If the value
                               is set to False, the function will wait until the task completes.
        ==================     ====================================================================

        :returns: dictionary

        """
        if self.properties['exportTilesAllowed'] == False:
            return
        import time
        url = self._url + "/estimateExportTilesSize"
        params = {
            "f": "json",
            "levels": levels,
            "exportBy": export_by,
            "tilePackage": tile_package,
            "exportExtent": export_extent
        }
        params["levels"] = levels
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                params[k] = v
        if not area_of_interest is None:
            params['areaOfInterest'] = area_of_interest
        if asynchronous == True:
            return self._con.get(url, params)
        else:
            exportJob = self._con.get(url, params)

            job_id = exportJob['jobId']
            path = "%s/jobs/%s" % (url, exportJob['jobId'])

            params = {"f": "json"}
            job_response = self._con.post(path, params)

            if "status" in job_response:
                status = job_response.get("status")
                while not status == "esriJobSucceeded":
                    time.sleep(5)

                    job_response = self._con.post(path, params)
                    status = job_response.get("status")
                    if status in ['esriJobFailed',
                                  'esriJobCancelling',
                                  'esriJobCancelled',
                                  'esriJobTimedOut']:
                        print(str(job_response['messages']))
                        raise Exception('Job Failed with status ' + status)
            else:
                raise Exception("No job results.")

            return job_response['results']

    # ----------------------------------------------------------------------
    def export_tiles(self,
                     levels,
                     export_by="LevelID",
                     tile_package=False,
                     export_extent="DEFAULT",
                     optimize_for_size=True,
                     compression=75,
                     area_of_interest=None,
                     asynchronous=False,
                     **kwargs
                     ):
        """
        The exportTiles operation is performed as an asynchronous task and
        allows client applications to download map tiles from a server for
        offline use. This operation is performed on a Map Service that
        allows clients to export cache tiles. The result of this operation
        is Map Service Job. This job response contains a reference to the
        Map Service Result resource, which returns a URL to the resulting
        tile package (.tpk) or a cache raster dataset.
        exportTiles can be enabled in a service by using ArcGIS for Desktop
        or the ArcGIS Server Administrator Directory. In ArcGIS for Desktop
        make an admin or publisher connection to the server, go to service
        properties, and enable Allow Clients to Export Cache Tiles in the
        advanced caching page of the Service Editor. You can also specify
        the maximum tiles clients will be allowed to download. The default
        maximum allowed tile count is 100,000. To enable this capability
        using the Administrator Directory, edit the service, and set the
        properties exportTilesAllowed=true and maxExportTilesCount=100000.

        At 10.2.2 and later versions, exportTiles is supported as an
        operation of the Map Server. The use of the
        http://Map Service/exportTiles/submitJob operation is deprecated.
        You can provide arguments to the exportTiles operation as defined
        in the following parameters table:


        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        levels                 required string. Specifies the tiled service levels to export. The
                               values should correspond to Level IDs, cache scales. or the
                               resolution as specified in export_by parameter. The values can be
                               comma separated values or a range. Make sure tiles are present at
                               the levels where you attempt to export tiles.
                               Example 1: 1,2,3,4,5,6,7,8,9
                               Example 2: 1-4,7-9
        ------------------     --------------------------------------------------------------------
        export_by              required string. The criteria that will be used to select the tile
                               service levels to export. The values can be Level IDs, cache scales.
                               or the resolution.  The defaut is 'LevelID'.
                               Values: LevelID | Resolution | Scale
        ------------------     --------------------------------------------------------------------
        tile_package           optiona boolean. Allows exporting either a tile package or a cache
                               raster data set. If the value is true, output will be in tile
                               package format, and if the value is false, a cache raster data
                               set is returned. The default value is false.
        ------------------     --------------------------------------------------------------------
        export_extent          optional dictionary or string. The extent (bounding box) of the tile
                               package or the cache dataset to be exported. If extent does not
                               include a spatial reference, the extent values are assumed to be in
                               the spatial reference of the map. The default value is full extent
                               of the tiled map service.
                               Syntax: <xmin>, <ymin>, <xmax>, <ymax>
                               Example 1: -104,35.6,-94.32,41
                               Example 2: {"xmin" : -109.55, "ymin" : 25.76,
                                            "xmax" : -86.39, "ymax" : 49.94,
                                            "spatialReference" : {"wkid" : 4326}}
        ------------------     --------------------------------------------------------------------
        optimize_for_size      optional boolean. Use this parameter to enable compression of JPEG
                               tiles and reduce the size of the downloaded tile package or the
                               cache raster data set. Compressing tiles slightly compromises the
                               quality of tiles but helps reduce the size of the download. Try
                               sample compressions to determine the optimal compression before
                               using this feature.
                               The default value is True.
        ------------------     --------------------------------------------------------------------
        compression=75,        optional integer. When optimize_for_size=true, you can specify a
                               compression factor. The value must be between 0 and 100. The value
                               cannot be greater than the default compression already set on the
                               original tile. For example, if the default value is 75, the value
                               of compressionQuality must be between 0 and 75. A value greater
                               than 75 in this example will attempt to up sample an already
                               compressed tile and will further degrade the quality of tiles.
        ------------------     --------------------------------------------------------------------
        area_of_interest       optional dictionary, Polygon. The area_of_interest polygon allows
                               exporting tiles within the specified polygon areas. This parameter
                               supersedes the exportExtent parameter.
                               Example: { "features": [{"geometry":{"rings":[[[-100,35],
                                                      [-100,45],[-90,45],[-90,35],[-100,35]]],
                                                      "spatialReference":{"wkid":4326}}}]}
        ------------------     --------------------------------------------------------------------
        asynchronous           optional boolean. Default False, this value ensures the returns are
                               returned to the user instead of the user having the check the job
                               status manually.
        ==================     ====================================================================

        :returns: path to download file is asynchronous is False. If True, a dictionary is returned.
        """
        import time
        params = {
            "f": "json",
            "tilePackage": tile_package,
            "exportExtent": export_extent,
            "optimizeTilesForSize": optimize_for_size,
            "compressionQuality": compression ,
            "exportBy": export_by,
            "levels": levels
        }
        if len(kwargs) > 0:
            for k,v in kwargs.items():
                params[k] = v
        url = self._url + "/exportTiles"
        if area_of_interest is not None:
            params["areaOfInterest"] = area_of_interest

        if asynchronous == True:
            return self._con.get(path=url, params=params)
        else:
            exportJob = self._con.get(path=url, params=params)

            job_id = exportJob['jobId']
            path = "%s/jobs/%s" % (url, exportJob['jobId'])

            params = {"f": "json"}
            job_response = self._con.post(path, params)

            if "status" in job_response:
                status = job_response.get("status")
                while not status == 'esriJobSucceeded':
                    time.sleep(5)

                    job_response = self._con.post(path, params)
                    status = job_response.get("status")
                    if status in ['esriJobFailed',
                                  'esriJobCancelling',
                                  'esriJobCancelled',
                                  'esriJobTimedOut']:
                        print(str(job_response['messages']))
                        raise Exception('Job Failed with status ' + status)
            else:
                raise Exception("No job results.")

            allResults = job_response['results']

            for k, v in allResults.items():
                if k == "out_service_url":
                    value = v.value
                    params = {
                        "f": "json"
                    }
                    gpRes = self._con.get(path=value, params=params)
                    if tile_package == True:
                        files = []
                        for f in gpRes['files']:
                            name = f['name']
                            dlURL = f['url']
                            files.append(
                                self._con.get(dlURL, params,
                                              out_folder=tempfile.gettempdir(),
                                              file_name=name))
                        return files
                    else:
                        return gpRes['folders']
                else:
                    return None
###########################################################################

class Events(object):

    @classmethod
    def _create_events(cls, enable=False):
        events = Events()

        events._enable = False
        events._type = "extentChanged"
        events._actions = []

        events.enable = enable

        return events

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = bool(value)

    @property
    def type(self):
        return self._type

    @property
    def synced_widgets(self):
        return self._actions

    def sync_widget(self, widgets):

        if self.enable == False:
            raise Exception("Please enable events")

        else:
            if isinstance(widgets, list):
                for widget in widgets:
                    if widget.type == "mapWidget":
                        action_type = "setExtent"
                        self._actions.append({"type":action_type, "targetId":widget._id})
                    else:
                        action_type = "filter"
                        widget_id = str(widget._id)+'#main'
                        self._actions.append({"type":action_type, "by":"geometry", "targetId":widget_id})
            else:
                if widgets.type == "mapWidget":
                    action_type = "setExtent"
                    self._actions.append({"type":action_type, "targetId":widgets._id})
                else:
                    action_type = "filter"
                    widget_id = str(widgets._id)+'#main'
                    self._actions.append({"type":action_type, "by":"geometry", "targetId":widget_id})





