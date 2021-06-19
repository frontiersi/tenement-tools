import uuid
from arcgis._impl.common._mixins import PropertyMap
###########################################################################
class BaseOGC(object):
    """

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The web address of the endpoint.
    ---------------     --------------------------------------------------------------------
    gis                 Optional GIS. The `GIS` connection object
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    id                  Optional String. The unique ID of the layer.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ---------------     --------------------------------------------------------------------
    title               Optional String. The title of the layer used to identify it in places such as the Legend and Layer List widgets.
    ===============     ====================================================================

    """
    _id = None
    _con = None
    _gis = None
    _url = None
    _type = "UnknownLayer"
    _title = None
    _opacity = None
    _copyright = None
    _min_scale = None
    _max_scale = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        self._url = url
        self._gis = gis
        self._min_scale, self._max_scale = kwargs.pop('scale', (0,0))
        self._title = kwargs.pop("title", "Layer")
        self._opacity = kwargs.pop('opacity', 1)
        self._id = kwargs.pop('id', uuid.uuid4().hex)
        self._copyright = kwargs.pop("copyright", "")
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Returns the properties of the Layer.
        
        :returns: PropertyMap
        """
        return PropertyMap(self._lyr_json)
    #----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__} @ {self._url}>"
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def title(self) -> str:
        """
        The title of the layer used to identify it in places such as the Legend and LayerList widgets.

        :returns: String
        """
        return self._title
    #----------------------------------------------------------------------
    @title.setter
    def title(self, value:str):
        """
        The title of the layer used to identify it in places such as the Legend and LayerList widgets.

        :returns: String
        """
        if self._title != value:
            self._title = value
    #----------------------------------------------------------------------
    @property
    def opacity(self) -> float:
        """
        This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.

        :returns: Float
        """
        return self._opacity
    #----------------------------------------------------------------------
    @opacity.setter
    def opacity(self, value:float):
        """
        This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.

        :returns: Float
        """
        if isinstance(value, (float, int)):
            self._opacity = value
    #----------------------------------------------------------------------
    @property
    def scale(self):
        """Gets/Sets the Min/Max Scale for the layer"""
        return self._min_scale, self._max_scale
    #----------------------------------------------------------------------
    @scale.setter
    def scale(self, scale:tuple):
        """Gets/Sets the Min/Max Scale for the layer"""
        if isinstance(scale, (tuple, list)) and len(scale) == 2:
            self._min_scale, self._max_scale = scale
    #----------------------------------------------------------------------
    @property
    def copyright(self):
        """Copyright information for the layer."""
        return self._copyright
    #----------------------------------------------------------------------
    @copyright.setter
    def copyright(self):
        """Copyright information for the layer."""
        return self._copyright
    #----------------------------------------------------------------------
    @property
    def _lyr_json(self) -> dict:
        """Represents the MapView's JSON format"""
        return {
            "id" : uuid.uuid4().hex,
            "title" : self._title or "Layer",
            "url" : self._url,
            "type" : self._type,
            "minScale" : self.scale[0],
            "maxScale" : self.scale[1],
            "opacity" : self.opacity
        }
    @property
    def _operational_layer_json(self) -> dict:
        """Represents the WebMap's JSON format"""
        return self._lyr_json

###########################################################################
class BaseOpenData(BaseOGC):
    """

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The web address of the endpoint.
    ---------------     --------------------------------------------------------------------
    gis                 Optional GIS. The `GIS` connection object
    ---------------     --------------------------------------------------------------------
    copyright           Optional String. Describes limitations and usage of the data.
    ---------------     --------------------------------------------------------------------
    id                  Optional String. The unique ID of the layer.
    ---------------     --------------------------------------------------------------------
    opacity             Optional Float.  This value can range between 1 and 0, where 0 is 100 percent transparent and 1 is completely opaque.
    ---------------     --------------------------------------------------------------------
    scale               Optional Tuple. The min/max scale of the layer where the positions are: (min, max) as float values.
    ---------------     --------------------------------------------------------------------
    sql_expression      Optional String. Optional query string to apply to the layer when displayed on the widget or web map.
    ---------------     --------------------------------------------------------------------
    title               Optional String. The title of the layer used to identify it in places such as the Legend and Layer List widgets.
    ===============     ====================================================================

    """
    _sql = None
    def __init__(self, url, gis=None, **kwargs):
        super(BaseOpenData, self)
        self._url = url
        self._gis = gis
        self._min_scale, self._max_scale = kwargs.pop('scale', (0,0))
        self._title = kwargs.pop("title", "Layer")
        self._opacity = kwargs.pop('opacity', 0)
        self._id = kwargs.pop('id', uuid.uuid4().hex)
        self._copyright = kwargs.pop("copyright", "")
        self._sql = kwargs.pop("sql_expression", None)
    #----------------------------------------------------------------------
    @property
    def sql_expression(self):
        """
        The SQL where clause used to filter features on the client. Only
        the features that satisfy the definition expression are displayed
        in the widget. Setting a definition expression is useful when the
        dataset is large and you don't want to bring all features to the
        client for analysis. The `sql_expressions` may be set when a
        layer is constructed prior to it loading in the view or after it
        has been loaded into the class.

        :return: String
        """
        return self._sql
    #----------------------------------------------------------------------
    @sql_expression.setter
    def sql_expression(self, value):
        """
        The SQL where clause used to filter features on the client. Only
        the features that satisfy the definition expression are displayed
        in the widget. Setting a definition expression is useful when the
        dataset is large and you don't want to bring all features to the
        client for analysis. The `sql_expressions` may be set when a
        layer is constructed prior to it loading in the view or after it
        has been loaded into the class.

        :return: String
        """
        if self._sql != value:
            self._sql = value