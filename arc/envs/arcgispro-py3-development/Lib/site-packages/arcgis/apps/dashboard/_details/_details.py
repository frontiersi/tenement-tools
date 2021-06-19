import uuid
import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import NoDataProperties


class Details(_BaseWidget):
    """
    Creates a dashboard Details widget.

    =========================   ===========================================
    **Argument**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal Item object. Item object can
                                be a Feature Layer or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the widget.
    -------------------------   -------------------------------------------
    layer                       Optional integer. Layer number when item is
                                a mapwidget.
    -------------------------   -------------------------------------------
    title                       Optional string. Title of the widget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description of the widget.
    -------------------------   -------------------------------------------
    max_features_displayed      Optional integer. Maximum number of features
                                to display.
    =========================   ===========================================
    """
    def __init__(self, item, name='Details', layer=0, title="", description="", max_features_displayed=50):
        super().__init__(name, title, description)
        if item.type not in ['Feature Service', 'mapWidget']:
            raise Exception("Please specify an item")

        self.item = item

        self._max_features_displayed = max_features_displayed

        self.type = "detailsWidget"
        self.layer = layer

        self._show_title = True
        self._show_content = True
        self._show_media = True
        self._show_attachment = True

        self._no_data = NoDataProperties._nodata_init()
    
    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        description = widget_json["description"]
        details = Details(item, name, title, description)

        return details

    @property
    def max_features(self):
        """
        :return: Max Features to display for feature data.
        """
        return self._max_features_displayed

    @max_features.setter
    def max_features(self, value):
        """
        Set max features to display for data from features.
        """
        self._max_features_displayed = value

    @property
    def no_data(self):
        """
        :return: NoDataProperties Object
        """
        return self._no_data

    @property
    def show_title(self):
        """
        :return: True if title is enabled else False.
        """
        return self._show_title

    @show_title.setter
    def show_title(self, value):
        """
        Set true to show title in the widget.
        """
        self._show_title = bool(value)

    @property
    def show_content(self):
        """
        :return: True if content is enabled else False.
        """
        return self._show_content

    @show_content.setter
    def show_content(self, value):
        """
        Set true to show content in the widget.
        """
        self._show_content = bool(value)

    @property
    def show_media(self):
        """
        :return: True if media is enabled else False.
        """
        return self._show_media

    @show_media.setter
    def show_media(self, value):
        """
        Set true to show media in the widget.
        """
        self._show_media = bool(value)

    @property
    def show_attachment(self):
        """
        :return: True if attachment is enabled else False.
        """
        return self._show_attachment

    @show_attachment.setter
    def show_attachment(self, value):
        """
        Set true to show attachment in the widget.
        """
        self._show_attachment = bool(value)

    def _convert_to_json(self):
        if self.item.type == 'mapWidget':
            wlayer = self.item.layers[self.layer]
            widget_id = self.item._id
            layer_id = wlayer["id"]
            self._datasource = {"id":str(widget_id)+'#'+str(layer_id)}
        else:
            self._datasource = {
                        "type": "featureServiceDataSource",
                        "itemId": self.item.itemid,
                        "layerId": 0,
                        "table": True
                    }
        json_data = {
            "type": "detailsWidget",
            "showTitle": self._show_title,
            "showContents": self._show_content,
            "showMedia": self._show_media,
            "showAttachments": self._show_attachment,
            "datasets": [],
            "id": self._id,
            "name": self.name,
            "caption": self.title,
            "description": self.description,
            "showLastUpdate": True,
            "noDataVerticalAlignment": self._no_data._alignment,
            "showCaptionWhenNoData": self._no_data._show_title,
            "showDescriptionWhenNoData": self._no_data._show_description
        }

        if self._no_data._text:
            json_data['noDataText'] = self._no_data._text

        if self._background_color:
            json_data['backgroundColor'] = self._background_color

        if self._text_color:
            json_data['textColor'] = self._text_color

        json_data['datasets'] = [
            {
                "type": "serviceDataset",
                "dataSource": self._datasource,
                "outFields": ["*"],
                "groupByFields": [],
                "orderByFields": [],
                "statisticDefinitions": [],
                "maxFeatures": self._max_features_displayed,
                "querySpatialRelationship": "esriSpatialRelIntersects",
                "returnGeometry": False,
                "clientSideStatistics": False,
                "name": "main"
            }
        ]

        return json_data
