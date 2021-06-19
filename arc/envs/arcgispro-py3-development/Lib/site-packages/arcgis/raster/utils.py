from arcgis.raster import _util

def generate_direct_access_url(expiration=None, *, gis=None):
    '''
    Function to get the direct access url for user's rasterStore on ArcGIS Online.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    expiration                               Optional integer.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional GIS. The GIS on which this function runs. 
                                             If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        String. Direct Access Url
    '''

    return _util._generate_direct_access_url(expiration=expiration, gis=gis)

def upload_imagery_to_agol_userstore(files, direct_access_url=None, *, gis=None):
    '''
    Uploads file/files to the user's rasterStore on ArcGIS Online and returns the list of urls.
    The list of urls can then be used with arcgis.raster.analytics.copy_raster() method to create imagery layer on AGOL.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    files                                    Required. It can be a folder, list of files or single file that needs to be uplaoded
    ------------------------------------     --------------------------------------------------------------------
    direct_access_url                        Optional string. The direct access url generated using generate_direct_access_url function.
                                             If not specified, the function would generate the direct access url internally which is valid for 1440 minutes.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional GIS. The GIS on which this function runs. 
                                             If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        List of file paths.
    '''

    return _util._upload_imagery_agol(files=files, direct_access_url=direct_access_url, gis=gis)
