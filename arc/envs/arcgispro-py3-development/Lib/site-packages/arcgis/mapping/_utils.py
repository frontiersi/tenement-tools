import logging as _logging
import arcgis


_log = _logging.getLogger(__name__)

_use_async = False


def _get_list_value(index, array):
    """
    helper operation to loop a list of values regardless of the index value

    Example:
    >>> a = [111,222,333]
    >>> list_loop(15, a)
    111
    """
    if len(array) == 0:
        return None
    elif index >= 0 and index < len(array):
        return array[index]
    return array[index % len(array)]



def export_map(web_map_as_json = None,
               format = """PDF""",
               layout_template = """MAP_ONLY""",
               gis=None):
    """


This function takes the state of the web map(for example, included services, layer visibility
settings, client-side graphics, and so forth) and returns either (a) a page layout or
(b) a map without page surrounds of the specified area of interest in raster or vector format.
The input for this function is a piece of text in JavaScript object notation (JSON) format describing the layers,
graphics, and other settings in the web map. The JSON must be structured according to the WebMap specification
in the ArcGIS HelpThis tool is shipped with ArcGIS Server to support web services for printing, including the
preconfigured service named PrintingTools.
Parameters:

   web_map_as_json: Web Map as JSON (str). Required parameter.  A JSON representation of the state of the map to be exported as it appears in the web application. See the WebMap specification in the ArcGIS Help to understand how this text should be formatted. The ArcGIS web APIs (for JavaScript, Flex, Silverlight, etc.) allow developers to easily get this JSON string from the map.

   format: Format (str). Optional parameter.  The format in which the map image for printing will be delivered. The following strings are accepted.For example:PNG8 (default if the parameter is left blank)PDFPNG32JPGGIFEPSSVGSVGZ
      Choice list:['PDF', 'PNG32', 'PNG8', 'JPG', 'GIF', 'EPS', 'SVG', 'SVGZ']

   layout_template: Layout Template (str). Optional parameter.  Either a name of a template from the list or the keyword MAP_ONLY. When MAP_ONLY is chosen or an empty string is passed in, the output map does not contain any page layout surroundings (for example title, legends, scale bar, and so forth)
      Choice list:['A3 Landscape', 'A3 Portrait', 'A4 Landscape', 'A4 Portrait', 'Letter ANSI A Landscape', 'Letter ANSI A Portrait', 'Tabloid ANSI B Landscape', 'Tabloid ANSI B Portrait', 'MAP_ONLY']

    gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.


Returns:
   output_file - Output File as a DataFile

See https://utility.arcgisonline.com/arcgis/rest/directories/arcgisoutput/Utilities/PrintingTools_GPServer/Utilities_PrintingTools/ExportWebMapTask.htm for additional help.
    """

    from arcgis.geoprocessing import DataFile
    from arcgis.geoprocessing._support import _execute_gp_tool

    kwargs = locals()

    param_db = {
        "web_map_as_json": (str, "Web_Map_as_JSON"),
        "format": (str, "Format"),
        "layout_template": (str, "Layout_Template"),
        "output_file": (DataFile, "Output File"),
    }
    return_values = [
        {"name": "output_file", "display_name": "Output File", "type": DataFile},
    ]

    if gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.printTask.url[:-len('/Export%20Web%20Map%20Task')]

    return _execute_gp_tool(gis, "Export Web Map Task", kwargs, param_db, return_values, _use_async, url)

export_map.__annotations__ = {
               'web_map_as_json': str,
               'format': str,
               'layout_template': str
            }

def get_layout_templates(gis=None):
    """


This function returns the content of the GIS's layout templates formatted as dict.

Parameters:

gis: Optional, the GIS on which this tool runs. If not specified, the active GIS is used.


Returns:
   output_json - layout templates as Python dict

See https://utility.arcgisonline.com/arcgis/rest/directories/arcgisoutput/Utilities/PrintingTools_GPServer/Utilities_PrintingTools/GetLayoutTemplatesInfo.htm for additional help.
    """
    from arcgis.geoprocessing import DataFile
    from arcgis.geoprocessing._support import _execute_gp_tool

    kwargs = locals()

    param_db = {
        "output_json": (str, "Output JSON"),
    }
    return_values = [
        {"name": "output_json", "display_name": "Output JSON", "type": str},
    ]

    if gis is None:
        gis = arcgis.env.active_gis

    url = gis.properties.helperServices.printTask.url[:-len('/Export%20Web%20Map%20Task')]

    return _execute_gp_tool(gis, "Get Layout Templates Info Task", kwargs, param_db, return_values, _use_async, url)

get_layout_templates.__annotations__ = {'return': str}