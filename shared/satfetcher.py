"""
Functions to fetch satellite data.
"""

# import required libraries
import os, sys
import pandas as pd
import numpy as np
import xarray as xr

sys.path.append('../../shared')
import tools


# todo - metadata, checks
def load_dea_ard(platform=None, bands=None, x_extent=None, y_extent=None, 
                 time_range=None, min_gooddata=0.90, use_dask=False):
    """
    """
    # imports
    import datacube

    sys.path.append('../../Scripts')
    from dea_datahandling import load_ard
    from dea_dask import create_local_dask_cluster
    from dea_plotting import display_map, rgb


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
    

# todo all
def load_local_nc(filepath=None):
    print('todo')
    