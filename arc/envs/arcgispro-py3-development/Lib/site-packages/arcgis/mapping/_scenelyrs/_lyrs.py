import json
from arcgis.gis import Layer, _GISResource, Item, GIS
###########################################################################
class Object3DLayer(Layer):
    """
    Represents a Web scene Point Cloud layer. Point Cloud layer are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(Object3DLayer, self).__init__(url, gis)

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : "SceneLayer",
                      'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': "SceneLayer", 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

###########################################################################
class IntegratedMeshLayer(Layer):
    """
    Represents a Web scene Point Cloud layer. Point Cloud layer are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(IntegratedMeshLayer, self).__init__(url, gis)

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : "IntegratedMeshLayer",
                      'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': "IntegratedMeshLayer", 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

###########################################################################
class Point3DLayer(Layer):
    """
    Represents a Web scene Point Cloud layer. Point Cloud layer are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.Point3DLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(Point3DLayer, self).__init__(url, gis)
    #----------------------------------------------------------------------
    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : "SceneLayer",
                      'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': "SceneLayer", 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict

###########################################################################
class PointCloudLayer(Layer):
    """
    Represents a Web scene Point Cloud layer. Point Cloud layer are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(PointCloudLayer, self).__init__(url, gis)

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : 'PointCloudLayer',
                      'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': 'PointCloudLayer', 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
###########################################################################
class BuildingLayer(Layer):
    """
    Represents a Web scene layer. Web scene layers are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.BuildingLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(BuildingLayer, self).__init__(url, gis)

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict =  { 'type' : 'BuildingSceneLayer',
                      'url' : url }
        if self._token is not None:
            lyr_dict['serviceToken'] = self._token

        if self.filter is not None:
            lyr_dict['filter'] = self.filter
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += '?token=' + self._token

        lyr_dict = {'type': 'BuildingSceneLayer', 'url': url}

        if self.filter is not None:
            lyr_dict['options'] = json.dumps({ "definition_expression": self.filter })
        if self._time_filter is not None:
            lyr_dict['time'] = self._time_filter
        return lyr_dict
###########################################################################
class _SceneLayerFactory(type):
    """
    Factory that generates the Scene Layers

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __call__(cls,
                 url,
                 gis=None):
        lyr = Layer(url=url, gis=gis)
        props = lyr.properties
        if 'sublayers' in props:
            return BuildingLayer(url=url, gis=gis)
        elif 'layerType' in props:
            lt = props.layerType
        else:
            lt = props.layers[0].layerType
        if str(lt).lower() == "pointcloud":
            return PointCloudLayer(url=url, gis=gis)
        elif str(lt).lower() == "point":
            return Point3DLayer(url=url, gis=gis)
        elif str(lt).lower() == "3dobject":
            return Object3DLayer(url=url, gis=gis)
        elif str(lt).lower() == "building":
            return BuildingLayer(url=url, gis=gis)
        elif str(lt).lower() == "IntegratedMesh".lower():
            return IntegratedMeshLayer(url=url, gis=gis)
        return lyr
###########################################################################
class SceneLayer(Layer, metaclass=_SceneLayerFactory):
    """
    Represents a Web scene layer. Web scene layers are cached web layers that are optimized for displaying a large
    amount of 2D and 3D features. You can use scene layers to represent 3D points, point clouds, 3D objects and
    integrated mesh layers.

    ==================     ====================================================================
    **Argument**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string, specify the url ending in /SceneServer/
    ------------------     --------------------------------------------------------------------
    gis                    Optional GIS object. If not specified, the active GIS connection is
                           used.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Instantiating a SceneLayer object

        from arcgis.mapping import SceneLayer
        s_layer = SceneLayer(url='https://your_portal.com/arcgis/rest/services/service_name/SceneServer/')

        type(s_layer)
        >> arcgis.mapping._types.PointCloudLayer

        print(s_layer.properties.layers[0].name)
        >> 'your layer name'
    """
    def __init__(self, url, gis=None):
        """
        Constructs a SceneLayer given a web scene layer URL
        """
        super(SceneLayer, self).__init__(url, gis)
