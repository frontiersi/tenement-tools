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
    

def inc_sigmoid(ds, a=None, b=None, inplace=True):
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
    
    # check inputs
    if a is None or b is None:
        raise ValueError('Must provide values for a and b.')
    
    # create copy
    if not inplace:
        ds = ds.copy(deep=True)
    
    # create masks to handle out of bound values 
    mask_lt_a = xr.where(ds < a, True, False)
    mask_gt_b = xr.where(ds > b, True, False)
    
    # perform inc sigmoidal
    ds = np.cos((1 - ((ds - a) / (b - a))) * (np.pi / 2))**2
    
    # mask out out of bounds values
    ds = ds.where(~mask_lt_a, 0.0)
    ds = ds.where(~mask_gt_b, 1.0)
    
    return ds


def dec_sigmoid(ds, c=None, d=None, inplace=True):
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
    
    # check inputs
    if c is None or d is None:
        raise ValueError('Must provide values for c and d.')
    
    # create copy
    if not inplace:
        ds = ds.copy(deep=True)
                    
    # create masks to handle out of bound values 
    mask_lt_c = xr.where(ds < c, True, False)
    mask_gt_d = xr.where(ds > d, True, False)
    
    # perform dec sigmoidal
    ds = np.cos(((ds - c) / (d - c)) * (np.pi / 2))**2
    
    # mask out out of bounds values
    ds = ds.where(~mask_lt_c, 1.0)
    ds = ds.where(~mask_gt_d, 0.0)

    return ds


def bell_sigmoid(ds, a=None, bc=None, d=None, inplace=True):
    """
    Apply a fuzzy membership function to data
    using bell-shaped sigmoidal function. Requires a
    low left inflection (a), a mid-point (bc), and a low 
    right inflection (d) point to set the bounds in which to 
    rescale all values to. Values at or closer to the bc
    inflection point will be boosted, where as values on 
    right and left sides will be reduced. The output dataset 
    will have values rescaled to 0-1.
    
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.
    a : int
        Lower left slope inflection point. 
    bc : int
        Mid slope inflection point.
    d : int
        Lower right slope inflection point.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # check inputs
    if a is None or bc is None or d is None:
        raise ValueError('Must provide values for a, bc and d.')
    
    # create copy
    if not inplace:
        ds = ds.copy(deep=True)
        
    # create masks to handle out of bound values 
    mask_lt_bc = xr.where((ds >= a) & (ds <= bc), True, False)
    mask_gt_bc = xr.where((ds > bc) & (ds <= d), True, False)

    # perform inc sigmoidal (left side of bell curve)
    left = np.cos((1 - ((ds - a) / (bc - a))) * (np.pi / 2))**2
    left = left.where(mask_lt_bc, 0.0)

    # perform dec sigmoidal (right side of bell curve)
    right = right = np.cos(((ds - bc) / (d - bc)) * (np.pi / 2))**2
    right = right.where(mask_gt_bc, 0.0)

    # sum
    ds = left + right

    return ds