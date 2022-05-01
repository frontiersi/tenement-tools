#!!!!!!!! I split this into read_shapefile and extract_occurence_data
# todo: remove when sdm code updated
def read_coordinates_shp(shp_path=None, p_a_column=None):
    """
    Read observation records from a projected ESRI Shapefile and extracts the x and y values
    located within. This must be a point geometry-type dataset and it must be projected in
    the GDA94 Albers projection system (EPSG 3577).

    Parameters
    ----------
    shp_path : string
        A single string with full path and filename of shapefile.
    p_a_column : string
        A column of presence/absence values (1, 0). If any value other than
        1, 0 exists, error thrown. Do not enter a name to use every point in
        the dataset as presence (default).

    Returns
    ----------
    df_presence : pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """
    
    # imports check
    try:
        from osgeo import ogr
    except:
        raise ImportError('Could not import osgeo.')
    
    # notify user
    print('Reading species point locations from shapefile.')

    # check if string, if not bail
    if not isinstance(shp_path, str):
        raise ValueError('> Shapefile path must be a string. Please check the file path.')

    # check if shp exists
    if not os.path.exists(shp_path):
        raise OSError('> Unable to read species point locations. Please check the file path.')

    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()

        # get epsg code
        epsg = int(lyr.GetSpatialRef().GetAttrValue('AUTHORITY', 1))

        # get num feats
        num_feats = lyr.GetFeatureCount()

    except Exception:
        raise TypeError('> Could not read species point locations. Is the file corrupt?')

    # check if point/multi point type
    if lyr.GetGeomType() not in [ogr.wkbPoint, ogr.wkbMultiPoint]:
        raise ValueError('> Shapefile is not a point/multi-point type.')

    # check if shapefile is empty
    if num_feats == 0:
        raise ValueError('> Shapefile has no features in it. Please check.')

    # check if shapefile is projected (i.e. in metres)
    if epsg != 3577:
        raise ValueError('> Shapefile is not projected in GDA94 Albers. Please reproject into EPSG: 3577.')
        
    # check if pres/abse column exists if requested

    # loop through feats
    coords = []
    for feat in lyr:
        geom = feat.GetGeometryRef()

        # get x and y of individual point type
        if geom.GetGeometryName() == 'POINT':
            coords.append([geom.GetX(), geom.GetY()])

        # get x and y of each point in multipoint type
        elif geom.GetGeometryName() == 'MULTIPOINT':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)   
                coords.append([sub_geom.GetX(), sub_geom.GetY()])

        # error, a non-point type exists
        else:
            raise TypeError('> Unable to read point location, geometry is invalid.')

    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_presence = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    shp, lyr = None, None

    # notify user and return
    print('> Species point presence observations loaded successfully.')
    return df_presence  


# vegfrax, older
# todo, long in tooth, move these into funcs when needed
def extract_rast_info(rast_path):
    """
    Read a raster (e.g. tif) and extract geo-transformation, coordinate 
    system, projection type, size of dimensions (x and y), nodata value.

    Parameters
    ----------
    rast_path: string
        A single string with full path and filename of a raster.

    Returns
    ----------
    rast_meta_dict : dictionary with keys:
        layer = name of layer
        type = type of data, vector or raster.
        geo_trans = raster geotransformation info.
        epsg = the epsg code of spatial reference system.
        is_projected = is projection system, true or false.
        x_dim = number of raster cells along x axis.
        y_dim = number of raster cells along y axis.
        nodata_val = no data value embedded in raster.
    """
    
    # check if string, if not bail
    if not isinstance(rast_path, str):
        raise ValueError('> Raster path must be a string. Please check the file path.')

    # check if raster exists
    if not os.path.exists(rast_path):
        raise OSError('> Unable to read raster, file not found. Please check the file path.')    

    # init dict
    rast_info_dict = {
        'layer': os.path.basename(rast_path),
        'type': 'raster',
        'geo_tranform': None,
        'x_dim': None,
        'y_dim': None,
        'epsg': None,
        'is_projected': 0,
        'nodata_val': None
    }
        
    try:
        # open raster
        rast = gdal.Open(rast_path, 0)

        # add transform, dims
        rast_info_dict['geo_tranform'] = rast.GetGeoTransform()
        rast_info_dict['x_dim'] = rast.RasterXSize
        rast_info_dict['y_dim'] = rast.RasterYSize
        rast_info_dict['nodata_val'] = rast.GetRasterBand(1).GetNoDataValue()

        # get spatial ref
        srs = rast.GetSpatialRef()

        # get epsg if exists
        if srs and srs.GetAttrValue('AUTHORITY', 1):
            rast_info_dict['epsg'] = srs.GetAttrValue('AUTHORITY', 1)

        # get is projected if exists
        if srs and srs.IsProjected():
            rast_info_dict['is_projected'] = srs.IsProjected()

        # get nodata value if exists
        if srs and srs.IsProjected():
            rast_info_dict['is_projected'] = srs.IsProjected()    

        # drop
        rast = None

    except Exception:
        raise IOError('Unable to read raster: {0}. Please check.'.format(rast_path))

    return rast_info_dict
     
     
def group_dupe_times(ds):
    """
    Read a netcdf file (e.g. nc) and combine duplicate times.

    Parameters
    ----------
    ds: xarray dataset
        A netcdf dataset file.

    Returns
    ----------
    ds : xarray Dataset
    """
    
    # check if time in cube
    if not isinstance(ds, xr.Dataset):
        print('Input is not xarray dataset type.')
        raise
    elif 'time' not in ds:
        print('No time dimension in input dataset.')
        raise
        
    # get attributes from dataset
    ds_attrs = ds.attrs
    ds_band_attrs = ds[list(ds)[0]].attrs
    ds_spatial_ref_attrs = ds['spatial_ref'].attrs
    
    # get datetimes in array, count unique counts
    dts = ds['time'].values
    _, num_dupes = np.unique(dts, return_counts=True)
    
    # flag if duplicate times occur
    if len(np.unique(num_dupes)) > 1:
        print('Duplicate times detected in dataset. Merging duplicates.')
        
        # group by times
        ds = ds.groupby('time').max()
                
        # append attrbutes on to dataset and bands
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in list(ds):
            ds[var].attrs = ds_band_attrs
            
        # rechunk if chunked 
        if bool(ds.chunks) is True:
            ds = ds.chunk({'time': -1})

    return ds