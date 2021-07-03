# canopy
'''
This script contains functions for calculating groundwater-dependent 
vegetation (GDV) from landsat or sentinel data. This model has been 
validated for three key Pilbara, Western Australia species Euc. victrix, Euc. 
camaldulenesis and Mel. argentea. It offers a SMCE approach to detecting
this vegetation. GDV is detected using a time series of vegetation indices,
moisture indices and seasonal stability in an AHP process, resulting in
a GDV likelihood (probability) map. Thresholding can be implemented via
standard deviation or groundtruthed point locations. GDV health
trends can be determined using Mann-Kendall trend analysis, Theil-sen slopes, 
or Change Vector Analysis (CVA) functions. Finally, significant breaks 
in vegetation can be detected using change point detection.

See associated Jupyter Notebook gdvspectra.ipynb for a basic tutorial on the
main functions and order of execution.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os, sys
import numpy as np
import xarray as xr

#sys.path.append('../../shared')
#import tools


def binary_mask(ds, remove_lt=None, remove_gt=None, inplace=True):
    """
    Builds a tree presence/absence mask from tree canopy heights.
    Best explained with examples:
        Setting remove_lt to 2 and remove_gt to None will remove 
        canopy height less below 2m tall and keep all else.
        Setting remove_gt to 4 and remove_lt to None will remove 
        canopy height above to 4m tall and keep all else.
        Setting remove_lt to 2 and remove_gt to 4 will keep all 
        canopy heights between 2m and 4m and remove all else.
        
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.
    remove_lt : int
        A value in metres (or raster units). All values less than
        or equal to this will be masked out of dataset. Set to None
        to ignore.
    remove_gt : int
        A value in metres (or raster units). All values greater than
        or equal to this will be masked out of dataset. Set to None
        to ignore.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
        
    """
    
    # notify
    print('Creating binary mask from canopy height model.')
    
    # dataset
    if not isinstance(ds, (xr.DataArray, xr.Dataset)):
        raise TypeError('Must provide a xarray dataset or array type.')
    
    # check if any values given
    if remove_lt is None and remove_gt is None:
        raise ValueError('Must provide at least one remove value.')

    # checks if greater val lower than lower val
    if remove_lt is not None and remove_gt is not None:
        if remove_lt >= remove_gt:
            raise ValueError('Lower cannot be >= than Greater.')
    
    # if inplace, dont copy
    if not inplace:
        ds = ds.copy(deep=True)
    
    if remove_lt is not None and remove_gt is None:
        print('Generating mask of height > {}m.'.format(remove_lt))
        return xr.where(ds <= remove_lt, False, True)
    
    if remove_lt is None and remove_gt is not None:
        print('Generating mask of height < {}m.'.format(remove_gt))
        return xr.where(ds >= remove_gt, False, True)
    
    if remove_lt is not None and remove_gt is not None:
        msg = 'Generating mask of height between {}m and {}m.'
        print(msg.format(remove_lt, remove_gt))
        return xr.where((ds <= remove_lt) | 
                        (ds >= remove_gt), False, True)
    

def inc_sigmoid(ds, a, b, inplace=True):
    """
    Apply a fuzzy membership function to data
    using increasing sigmoidal function. Requires a
    low inflection (a) and high inflection (b) point
    to set the bounds in which to rescale all values to.
    The output dataset will have values rescaled to 0-1.
    
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.
    a : int
        Low inflection point. 
    b : int
        High inflection point.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    
    """
    
    if not inplace:
        ds = ds.copy(deep=True)
            
    # create masks to handle out of bound values 
    mask_a = xr.where((ds > a) & (ds < b), 1.0, 0.0)
    mask_b = xr.where(ds > b, 1.0, 0.0)
    
    # perform inc sigmoidal
    ds = np.cos((1 - ((ds - a) / (b - a))) * (np.pi / 2))**2
    
    # apply masks (0 all < a, add 1 all > b)
    ds = (ds * mask_a) + mask_b
    
    return ds


def dec_sigmoid(ds, c, d, inplace=True):
    """
    Apply a fuzzy membership function to data
    using decreasing sigmoidal function. Requires a
    high inflection (c) and low inflection (d) point
    to set the bounds in which to rescale all values to.
    The output dataset will have values rescaled to 0-1.
    
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.
    c : int
        High inflection point. 
    d : int
        Low inflection point.
    inplace : bool
        Create a copy of the dataset in memory to preserve original
        outside of function. Default is True.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    if not inplace:
        ds = ds.copy(deep=True)
        
    # not sure if need this - 0 value acts odd
    if c == 0:
        print('Zero provided for c, setting to 0.0001 to avoid errors.')
        c = 0.0001
            
    # create masks to handle out of bound values 
    mask_c = xr.where(ds < c, 1.0, 0.0)
    mask_d = xr.where((ds > c) & (ds < d), 1.0, 0.0)
    
    # perform inc sigmoidal
    ds = np.cos(((ds - c) / (d - c)) * (np.pi / 2))**2
    
    # apply masks (0 all < a, add 1 all > b)
    ds = (ds * mask_d) + mask_c
    
    return ds