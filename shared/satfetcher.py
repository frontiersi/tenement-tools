"""
Functions to fetch satellite data.
"""

# import required libraries
import os, sys
import pandas as pd
import numpy as np
import xarray as xr

sys.path.append('../../shared')
import satfetcher, tools

# # # # # DEA # # #
# todo - metadata, checks
def load_dea_ard(platform=None, bands=None, x_extent=None, y_extent=None, 
                 time_range=None, min_gooddata=0.90, use_dask=False):
    """
    """
    try:
        # imports
        import datacube

        sys.path.append('../../Scripts')
        from dea_datahandling import load_ard
        from dea_dask import create_local_dask_cluster
        from dea_plotting import display_map, rgb
        
    except:
        raise ImportError('Could not import DEA ODC.')

    # notify user
    print('Loading DEA ODC ARD satellite data.')
    
    # check if platform landsat or sentinel
    
    # check if bands
    
    # check lon extent, lat extent
    
    # check time
        
    # set up allowed bands for landsat
    landsat_dea_bands = [
        'nbart_blue', 
        'nbart_green', 
        'nbart_red', 
        'nbart_nir', 
        'nbart_swir_1', 
        'nbart_swir_2'
    ]
    
    # set up allowed bands for sentinel
    sentinel_dea_bands = [
        'nbart_blue', 
        'nbart_green', 
        'nbart_red', 
        'nbart_red_edge_1', 
        'nbart_red_edge_2', 
        'nbart_nir_1', 
        'nbart_nir_2', 
        'nbart_swir_2', 
        'nbart_swir_3'
    ]
    
    # check bands supported for platform and prepare parameters
    if platform == 'landsat':
        for band in bands:
            if band not in landsat_dea_bands:
                raise ValueError('Band: {0} not supported for landsat.'.format(band))
                
        # set products, resolution
        products = ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3']
        resolution = (-30, 30)
        
    elif platform == 'sentinel':
        for band in bands:
            if band not in sentinel_dea_bands:
                raise ValueError('Band: {0} not supported for landsat.'.format(band))
                
        # set products, resolution
        products = ['s2a_ard_granule', 's2b_ard_granule']
        resolution = (-10, 10)
    
    else:
        raise ValueError('Platform: {0} not supported.'.format(platform))
            
    # build query
    if len(bands) > 0:
        query = {
            'x': x_extent,
            'y': y_extent,
            'time': time_range,
            'products': products,
            'measurements': bands,
            'output_crs': 'EPSG:3577',
            'resolution': resolution,
            'group_by': 'solar_day',
            'min_gooddata': min_gooddata
        }
            
        # if dask, add chunks
        if use_dask:
            query.update({'dask_chunks': {'time': 1}})
  
    else:
        raise ValueError('No DEA bands in query. Please check requested bands.')
            
    # fetch data from dea cube
    if query:
            dc = datacube.Datacube(app='gdvspectra')
            ds = load_ard(dc=dc, **query)
            
            # add nodata value attribute
            ds.attrs.update({'nodatavals': np.nan})
    else:
        raise ValueError('Query could not be created.')
  
    # notify user, return
    print('Satellite imagery fetched successfully.')
    return ds


def conform_dea_ard_band_names(ds, platform=None):
    """
    Takes an xarray dataset containing spectral bands from various different
    satellite digital earth australia (DEA) products and conforms band names.
    Only for satellite data from DEA.
    
    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing spectral 
        bands that will be renamed.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with renamed 
        spectral bands.
    """
    
    # notify user
    print('Conforming DEA ARD satellite band names.')
    
    # if da provided, attempt convert to ds, check for ds after that
    was_da = False
    if isinstance(ds, xr.DataArray):
        try:
            ds = ds.to_dataset(dim='variable')
            was_da = True
        except:
            raise TypeError('Failed to convert xarray DataArray to Dataset. Provide a Dataset.')
    
    elif not isinstance(ds, xr.Dataset):
        raise TypeError('Not an xarray dataset. Please provide Dataset.')
    
    # create band rename mapping dictionary based on platform
    if platform == 'landsat':
        
        # create landsat 5,7,8 rename map
        band_map_dict = {
            'nbart_blue': 'blue',
            'nbart_green': 'green',
            'nbart_red': 'red',
            'nbart_nir': 'nir',
            'nbart_swir_1': 'swir1',
            'nbart_swir_2': 'swir2',
            'nbar_blue': 'blue',
            'nbar_green': 'green',
            'nbar_red': 'red',
            'nbar_nir': 'nir',
            'nbar_swir_1': 'swir1',
            'nbar_swir_2': 'swir2',
        }
        
    elif platform == 'sentinel':
        
        # create sentinel 2 rename map
        band_map_dict = {
            'nbart_blue': 'blue',
            'nbart_green': 'green',
            'nbart_red': 'red',
            'nbart_nir_1': 'nir',
            'nbart_red_edge_1': 'red_edge_1', 
            'nbart_red_edge_2': 'red_edge_2',    
            'nbart_swir_2': 'swir1',
            'nbart_swir_3': 'swir2',
            'nbar_blue': 'blue',
            'nbar_green': 'green',
            'nbar_red': 'red',
            'nbar_nir_1': 'nir',
            'nbar_red_edge_1': 'red_edge_1', 
            'nbar_red_edge_2': 'red_edge_2',    
            'nbar_swir_2': 'swir1',
            'nbar_swir_3': 'swir2',
        }
 
    # rename bands in dataset to use conformed naming conventions
    bands_to_rename = {
        k: v for k, v in band_map_dict.items() if k in list(ds.data_vars)
    }
    
    # apply the rename
    ds = ds.rename(bands_to_rename)
    
    # convert back to datarray
    if was_da:
        ds = ds.to_array()
    
    # notify user, return
    print('Satellite band names conformed successfully.')
    return ds

# # # # # LOCAL DATA # # #
# meta
def load_local_rasters(rast_path_list=None, use_dask=True, conform_nodata_to=-9999):
    """
    Read a list of rasters (e.g. tif) and convert them into an xarray dataset, 
    where each raster layer becomes a new dataset variable.

    Parameters
    ----------
    rast_path_list: list
        A list of strings with full path and filename of a raster.
    use_dask : bool
        Defer loading into memory if dask is set to True.
    conform_nodata_to : numeric or numpy.nan
        A value in which no data values will be changed to. We need
        to conform various different datasets, so keeping this
        the same throughout is vital. Use np.nan for smaller
        datasets, and keep it an int (i.e. -9999) for larger ones.

    Returns
    ----------
    ds : xarray Dataset
    """
    
    # import checks
    try:
        import dask
    except:
        print('Could not import dask.')
        use_dask = False
    
    # notify user
    print('Converting rasters to an xarray dataset.')
    
    # check if raster exists
    if not rast_path_list:
        raise ValueError('No raster paths provided.')
    elif not isinstance(rast_path_list, list):
        rast_path_list = [rast_path_list]
        
    # ensure raster paths in list exist and are strings
    for rast_path in rast_path_list:
        if not isinstance(rast_path, str):
            raise ValueError('Raster path must be a string.')
        elif not os.path.exists(rast_path):
            raise OSError('Unable to read raster, file not found.')

    # loop thru raster paths and convert to data arrays
    da_list = []
    for rast_path in rast_path_list:
        try:
            # get filename
            rast_filename = os.path.basename(rast_path)
            rast_filename = os.path.splitext(rast_filename)[0]
            
            # if dask use it, else rasterio
            if use_dask:
                da = xr.open_rasterio(rast_path, chunks=-1)
            else:
                da = xr.open_rasterio(rast_path)

            # rename band to var, add var name 
            da = da.rename({'band': 'variable'})
            da['variable'] = np.array([rast_filename])
            
            # check if compoite and fail if so
            if da.shape[0] != 1:
                raise ValueError('Raster composite provided, split into seperate tifs.')
            
            # check if no data val attributes exist, replace with nan
            if hasattr(da, 'nodatavals') and da.nodatavals is not None:

                # check if nodata values a iterable, if not force it
                nds = da.nodatavals
                if not isinstance(nds, (list, tuple)):
                    nds = [nds]
                    
                # mask nan for nodata values
                for nd in nds:
                    da = da.where(da != nd, conform_nodata_to)
                    
                # update xr attributes to new nodata val
                if hasattr(da, 'attrs'):
                    da.attrs.update({'nodatavals': conform_nodata_to})
                    
                # convert from float64 to float32 if nan is nodata
                if conform_nodata_to is np.nan:
                    da = da.astype(np.float32)
                    
            else:
                # mask via provided nodata
                print('No NoData values found in raster: {0}.'.format(rast_path))
                da.attrs.update({'nodatavals': 'unknown'})

            # notify and append
            print('Converted raster to xarray data array: {0}'.format(rast_filename))
            da_list.append(da)
            
        except Exception:
            raise IOError('Unable to read raster: {0}.'.format(rast_path))
            
    # check if anything came back, then proceed
    if not da_list:
        raise ValueError('No rasters were converted. Please check validity.')
        
    # check if all arrays have same shape
    for da in da_list:
        if da_list[0].shape != da.shape:
            raise ValueError('Not all rasters are same extent. Please check.')
    
    # check if all arrays have crs epsg: 3577
    for da in da_list:
        if tools.get_xr_crs(da) != 3577:
            raise ValueError('Raster CRS not projected in EPSG:3577.')
                
    # combine all da together and create dataset
    try:
        ds = xr.concat(da_list, dim='variable')
        ds = ds.to_dataset(dim='variable')
        
    except Exception:
        raise ValueError('Could not concat data arrays. Check your rasters.')
              
    # notify user and return
    print('Rasters converted to dataset successfully.\n')
    return ds
    
    
# meta
def load_local_nc(nc_path=None, use_dask=True, conform_nodata_to=-9999):
    """
    Read a netcdf file (e.g. nc) and convert into an xarray dataset.

    Parameters
    ----------
    nc_path: str
        A string with full path and filename of a netcdf.
    use_dask : bool
        Defer loading into memory if dask is set to True.
    conform_nodata_to : numeric or numpy.nan
        A value in which no data values will be changed to. We need
        to conform various different datasets, so keeping this
        the same throughout is vital. Use np.nan for smaller
        datasets, and keep it an int (i.e. -9999) for larger ones.

    Returns
    ----------
    ds : xarray Dataset
    """
    
    # import checks
    try:
        import dask
    except:
        print('Could not import dask.')
        use_dask = False
    
    # notify user
    print('Converting netcdf to an xarray dataset.')
    
    # check if netcdf exists
    if not nc_path:
        raise ValueError('No netcdf path provided.')
        
    # ensure netcdf path in list exist and are strings
    if not isinstance(nc_path, str):
        raise ValueError('Netcdf path must be a string.')
    elif not os.path.exists(nc_path):
        raise OSError('Unable to read netcdf, file not found.')

    # try open netcdf
    try:
        # if dask use it, else rasterio
        if use_dask:
            ds = xr.open_dataset(nc_path, chunks=-1)
        else:
            ds = xr.open_dataset(nc_path)

        # check if no data val attributes exist, replace with nan
        if hasattr(ds, 'nodatavals') and ds.nodatavals is not None:

            # check if nodata values a iterable, if not force it
            nds = ds.nodatavals
            if not isinstance(nds, (list, tuple)):
                nds = [nds]

            # mask nan for nodata values
            for nd in nds:
                ds = ds.where(ds != nd, conform_nodata_to)

            # update xr attributes to new nodata val
            if hasattr(ds, 'attrs'):
                ds.attrs.update({'nodatavals': conform_nodata_to})

            # convert from float64 to float32 if nan is nodata
            if conform_nodata_to is np.nan:
                ds = ds.astype(np.float32)

        else:
            # mask via provided nodata
            print('No NoData values found in raster: {0}.'.format(nc_path))
            da.attrs.update({'nodatavals': 'unknown'})

        # notify and append
        print('Converted netcdf to xarray dataset: {0}'.format(nc_path))

    except Exception:
        raise IOError('Unable to read netcdf: {0}.'.format(nc_path))
                
    # check if dataset has crs epsg: 3577
    if tools.get_xr_crs(ds) != 3577:
        raise ValueError('Netcdf CRS not projected in EPSG:3577.')
                              
    # notify user and return
    print('Netcdf converted to xarray dataset successfully.')
    return ds


# # # # # COG # # #

# meta, checks
def make_vrt_dataset_xml(x_size, y_size, axis_map, srs, trans):
    """
    take paramets for vrt and create a raster xml object
    """
    
    # imports
    from lxml import etree as et

    # set up root vrt dataset elem
    xml_ds = '<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}"></VRTDataset>'
    xml_ds = et.fromstring(xml_ds.format(x_size=x_size, y_size=y_size))

    # set up srs element and add to vrt dataset
    xml_srs = '<SRS dataAxisToSRSAxisMapping="{axis_map}">{srs}</SRS>'
    xml_ds.append(et.fromstring(xml_srs.format(axis_map=axis_map, srs=srs)))

    # set up geo transform element and add to vrt dataset
    xml_trans = '<GeoTransform>{trans}</GeoTransform>'
    xml_ds.append(et.fromstring(xml_trans.format(trans=trans)))
    
    # return xml dataset
    return xml_ds

# meta, checks
def make_vrt_raster_xml(x_size, y_size, dtype, band_num, nodata, dt, rel_to_vrt, url, src_band):
    """
    take paramets for vrt and create a raster xml object
    """

    # imports
    from lxml import etree as et
    
    # set up root vrt raster elem
    xml_rast = '<VRTRasterBand dataType="{dtype}" band="{band_num}"></VRTRasterBand>'
    xml_rast = et.fromstring(xml_rast.format(dtype=dtype, band_num=band_num))
        
    # add a top-level nodata value element and add to vrt raster
    xml_ndv = '<NoDataValue>{nodata}</NoDataValue>'
    xml_rast.append(et.fromstring(xml_ndv.format(nodata=nodata)))    
    
    # set up top-level complexsource element, dont add it to rast yet
    xml_complex = '<ComplexSource></ComplexSource>'
    xml_complex = et.fromstring(xml_complex)
    
    # add a description elem to hold datetime to the complex source
    xml_desc = '<Description>{dt}</Description>'
    xml_complex.append(et.fromstring(xml_desc.format(dt=dt)))

    # add source filename to complex source
    xml_filename = '<SourceFilename relativeToVRT="{rel_to_vrt}">/vsicurl/{url}</SourceFilename>'
    xml_complex.append(et.fromstring(xml_filename.format(rel_to_vrt=rel_to_vrt, url=url)))
    
    # add source band num to complex source
    xml_src_band = '<SourceBand>{src_band}</SourceBand>'
    xml_complex.append(et.fromstring(xml_src_band.format(src_band=src_band)))
    
    # add source properties to complex source. hardcoded block size for now
    xml_src_props = '<SourceProperties RasterXSize="{x_size}" RasterYSize="{y_size}"' + ' ' + \
                    'DataType="{dtype}" BlockXSize="512" BlockYSize="512"></SourceProperties>'
    xml_complex.append(et.fromstring(xml_src_props.format(x_size=x_size, y_size=y_size, dtype=dtype)))
    
    # add a src rect to complex source. hardcoded offset for now
    xml_src_rect = '<SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}"></SrcRect>'
    xml_complex.append(et.fromstring(xml_src_rect.format(x_size=x_size, y_size=y_size)))
    
    # add a dst rect to complex source. hardedcoded offset for now
    xml_dst_rect = '<DstRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}"></DstRect>'
    xml_complex.append(et.fromstring(xml_dst_rect.format(x_size=x_size, y_size=y_size)))
    
    # add a lower-level nodata elem to complex source
    xml_nd = '<NODATA>{nodata}</NODATA>'
    xml_complex.append(et.fromstring(xml_nd.format(nodata=nodata)))
        
    # finally, add filled in complex source element to rast
    xml_rast.append(xml_complex)
    
    # return xml raster
    return xml_rast

# todo checks, meta
def make_vrt_list(feat_list, band=None):
    """
    take a list of stac features and band(s) names and build gdal
    friendly vrt xml objects in list.
    band : list, str
        Can be a list or string of name of band(s) required.
    """
    
    # imports
    from lxml import etree as et
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    
    # check if band provided, if so and is str, make list
    if band is None:
        bands = []
    elif not isinstance(band, list):
        bands = [band]
    else:
        bands = band
                    
    # check features type, length
    if not isinstance(feat_list, list):
        raise TypeError('Features must be a list of xml objects.')
    elif not len(feat_list) > 0:
        raise ValueError('No features provided.')
        
    # set list vrt of each scene
    vrt_list = []

    # iter stac scenes, build a vrt
    for feat in feat_list:

        # get scene identity and properties
        f_id = feat.get('id')
        f_props = feat.get('properties')

        # get scene-level date
        f_dt = f_props.get('datetime')

        # get scene-level x, y parameters
        f_x_size = f_props.get('proj:shape')[1]
        f_y_size = f_props.get('proj:shape')[0]

        # get scene-level epsg src as wkt
        f_srs = CRS.from_epsg(f_props.get('proj:epsg'))
        f_srs = f_srs.wkt

        # get scene-level transform
        aff = Affine(*f_props.get('proj:transform')[0:6])
        f_transform = ', '.join(str(p) for p in Affine.to_gdal(aff))

        # build a top-level vrt dataset xml object
        xml_ds = satfetcher.make_vrt_dataset_xml(x_size=f_x_size,
                                                 y_size=f_y_size,
                                                 axis_map='1,2',  # hardcoded
                                                 srs=f_srs,
                                                 trans=f_transform)
        
        # iterate bands and build raster vrts
        band_idx = 1
        for band in bands:
            if band in feat.get('assets'):

                # get asset
                asset = feat.get('assets').get(band)

                # set dtype to int16... todo bug in rasterio with int8?
                #a_dtype = 'UInt8' if band == 'oa_fmask' else 'Int16'
                a_dtype = 'Int16'

                # get asset raster x, y sizes
                a_x_size = asset.get('proj:shape')[1]
                a_y_size = asset.get('proj:shape')[0]

                # get raster url, replace s3 with https
                a_url = asset.get('href')
                a_url = a_url.replace('s3://dea-public-data', 'https://data.dea.ga.gov.au')
                
                # get nodata value
                a_nodata = 0 if band == 'oa_fmask' else -999

                # build raster xml
                xml_rast = satfetcher.make_vrt_raster_xml(x_size=a_x_size,
                                                          y_size=a_y_size,
                                                          dtype=a_dtype,
                                                          band_num=band_idx,
                                                          nodata=a_nodata,
                                                          dt=f_dt,
                                                          rel_to_vrt=0,  # hardcoded
                                                          url=a_url,
                                                          src_band=1)  # hardcoded

                # append raster xml to vrt dataset xml
                xml_ds.append(xml_rast)

                # increase band index
                band_idx += 1

        # decode to utf-8 string and append to vrt list
        xml_ds = et.tostring(xml_ds).decode('utf-8')
        vrt_list.append(xml_ds)
        
    return vrt_list

# meta, check
def get_dea_landsat_vrt_dict(feat_list):
    """
    this func is designed to take all releveant landsat bands
    on the dea public database for each scene in stac query.
    it results in a list of vrts for each band seperately and maps
    them to a dict where band name is the key, list is the value pair.
    """
        
    # notify
    print('Getting landsat vrts for each relevant bands.')
                        
    # check features type, length
    if not isinstance(feat_list, list):
        raise TypeError('Features must be a list of xml objects.')
    elif not len(feat_list) > 0:
        raise ValueError('No features provided.')
    
    # required dea landsat ard band names
    bands = [
        'nbart_blue', 
        'nbart_green',
        'nbart_red',
        'nbart_nir',
        'nbart_swir_1',
        'nbart_swir_2',
        'oa_fmask'
    ]
    
    # iter each band name and build associated vrt list
    band_vrts_dict = {}
    for band in bands:
        print('Building landsat vrt list for band: {}.'.format(band))
        band_vrts_dict[band] = make_vrt_list(feat_list, band=band)
        
    # notify and return
    print('Got {} landsat vrt band lists successfully.'.format(len(band_vrts_dict)))
    return band_vrts_dict
    

    
# checks, meta - resample, warp tech needed NEEDS WORK!
def build_vrt_file(vrt_list):

    # imports
    import tempfile
    import gdal
        
    # check features type, length
    if not isinstance(vrt_list, list):
        raise TypeError('VRT list must be a list of xml objects.')
    elif not len(vrt_list) > 0:
        raise ValueError('No VRT xml objects provided.')
    
    # build vrt
    with tempfile.NamedTemporaryFile() as tmp:

        # append vrt extension to temp file
        f = tmp.name + '.vrt'

        # create vrt options
        opts = gdal.BuildVRTOptions(separate=True,
                                    bandList=[1])
        #outputBounds=boundingbox,
        #resampleAlg='bilinear',
        #resolution='user',
        #xRes=30.0,
        #yRes=30.0,
        #outputSRS=rasterio.crs.CRS.from_epsg(3577).wkt
        #targetAlignedPixels=True
        
        # warp/translate?
        # todo

        # consutruct vrt in memory, write it with none
        vrt = gdal.BuildVRT(f, vrt_list, options=opts)
        vrt.FlushCache()
        
        # decode ytf-8?
        
        return f

# meta, checks
def combine_vrts_per_band(band_vrt_dict):
    """
    takes a dictionary of band name : vrt list key, value pairs and
    for each band, combines vrts into one vrt using the build vrt file 
    function (just a call to gdal.BuildVRT).
    """
        
    # notify
    print('Combining VRTs into single VRTs per band.')
                        
    # check features type, length
    if not isinstance(band_vrt_dict, dict):
        raise TypeError('Features must be a dict of band : vrt list objects.')
    elif not len(band_vrt_dict) > 0:
        raise ValueError('No band vrts in dictionary.')
    
    # get list of band names in dict
    bands = [band for band in band_vrt_dict]
    
    # iter each band name and build associated vrt dict
    vrt_file_dict = {}
    for band in bands:
        print('Combining VRTs into temp. file for band: {}.'.format(band))
        vrt_list = band_vrt_dict[band]
        vrt_file_dict[band] = satfetcher.build_vrt_file(vrt_list)

    # notify and return
    print('Combined {} band vrt lists successfully.'.format(len(vrt_file_dict)))
    return vrt_file_dict



# meta, checks, rethink it
def parse_vrt_datetimes(vrt_list):
    """
    takes a list of vrt files and extracts datetime from
    the descriptiont tag.
    """
    
    # imports
    from lxml import etree as et
    
    # checks
    
    # set dt map and counter
    dt_map = {}
    i = 1
    
    # iter items and parse description, skip errors
    for item in vrt_list:
        try:
            # parse out description tag as text
            root = et.fromstring(item)
            desc = root.findall('.//Description')[0]
            desc = desc.text

            # add index and datetime to datetime
            dt_map[i] = desc
            i += 1

        except Exception as e:
            print('Warning: {} at index: {}.'.format(e, i))
          
    # return
    return dt_map

# meta, checks, as above
def get_vrt_file_datetimes(vrt_file_dict):
    """
    takes a dictionary of band : vrt files and parses
    datetimes from each vrt file. spits out a dict of
    band name : dicts (band indexes : datetimes)
    """
    
    # imports
    import gdal
    
    # notify
    print('Extracting datetimes for VRTs per band.')
                        
    # check features type, length
    if not isinstance(vrt_file_dict, dict):
        raise TypeError('VRTs must be a dict of band name : vrt files.')
    elif not len(vrt_file_dict) > 0:
        raise ValueError('No vrts in dictionary.')
    
    # get list of band names in dict
    bands = [band for band in vrt_file_dict]
    
    # iter each band name and build associated vrt dict
    dt_dict = {}
    for band in bands:
        print('Extracting datetimes from VRTs for band: {}.'.format(band))
        
        # get vrt file for current band, open with gdal and extract
        vrt_list = vrt_file_dict[band]
        tmp = gdal.Open(vrt_list).GetFileList()
        dt_dict[band] = satfetcher.parse_vrt_datetimes(tmp)

    # notify and return
    print('Extracted {} band vrt datetimes successfully.'.format(len(dt_dict)))
    return dt_dict



# checks, meta
def prepare_full_vrt_dicts(vrt_file_dict, vrt_dt_dict):
    """
    takes vrt file and datetime file dicts and combines
    into one final dict
    """
    
    # imports
    from collections import Counter
    
    # notify
    print('Combining vrt files and datetimes per band.')
    
    # checks
    
    # get list of band names in dict
    file_bands = [band for band in vrt_file_dict]
    dt_bands = [band for band in vrt_dt_dict]
    
    # check if same bands lists identical
    if Counter(file_bands) != Counter(dt_bands):
        raise ValueError('VRT and datetime band names not identical.')
        
    # iter vrt file dict and create as we go
    vrt_dict = {}
    for band in file_bands:
        vrt_dict[band] = {
            'vrt_datetimes': vrt_dt_dict[band],
            'vrt_file': vrt_file_dict[band]}
        
    # notify and return
    print('Combined vrt files and datetimes per band successfully.')
    return vrt_dict



# meta, checks, more features
def build_xr_datasets(vrt_dict):
    """
    """

    # imports
    import xarray as xr
    from dateutil.parser import parse

    # notify
    print('Building an xarray dataset from vrt files and datetimes.')

    # checks

    # get list of band names in dict
    bands = [band for band in vrt_dict]

    # iter bands and append to dataset list
    ds_list = []
    for band in bands:
        
        # notify
        print('Working on dataset for band: {}'.format(band))

        # get date dt index and values
        vrt_dt = vrt_dict[band].get('vrt_datetimes')

        # prepare datetimes
        np_dt = {}
        for k, v in vrt_dt.items():
            v = parse(v, ignoretz=True)
            np_dt[k] = np.datetime64(v)

        # get vrt file
        vrt = vrt_dict[band].get('vrt_file')

        # create chunks
        chunks = {'band': 1, 'x': 'auto', 'y': 'auto'}

        # open into xarray via rasterio
        ds = xr.open_rasterio(vrt, chunks=chunks)

        # rename dim band  to time
        ds = ds.rename({'band': 'time'})

        # convert to dataset and name var to band
        ds = ds.to_dataset(name=band, promote_attrs=True)

        # remap times from indexes to datetimes
        ds['time'] = [np_dt[i] for i in ds['time'].values.tolist()]

        # sort datetime
        ds = ds.sortby('time')

        # append
        ds_list.append(ds)

    # bail if nothing worked
    if not ds_list:
        raise ValueError('No datasets were created.')

    # concatenate datasets into one
    ds = xr.merge(ds_list, combine_attrs='override')
    
    # notify and return
    print('Built an xarray dataset successfully.')
    return ds # change back to ds