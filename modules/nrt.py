# nrt
'''
Temp.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os, sys
import numpy as np
import xarray as xr


def temp(ds):
    """
    Temp
        
    Parameters
    ----------
    ds: xarray dataset/array
        A dataset with x, y dims.

    Returns
    ----------
    ds : xarray dataset or array.
    """
    
    # notify
    print('Temp.')
    
    # dataset
    #if not isinstance(ds, (xr.DataArray, xr.Dataset)):
        #raise TypeError('Must provide a xarray dataset or array type.')
    
    # check if any values given
    #if remove_lt is None and remove_gt is None:
        #raise ValueError('Must provide at least one remove value.')

    # checks if greater val lower than lower val
    #if remove_lt is not None and remove_gt is not None:
        #if remove_lt >= remove_gt:
            #raise ValueError('Lower cannot be >= than Greater.')
    
    # if inplace, dont copy
    #if not inplace:
        #ds = ds.copy(deep=True)
    
    #if remove_lt is not None and remove_gt is None:
        #print('Generating mask of height > {}m.'.format(remove_lt))
        #return xr.where(ds <= remove_lt, False, True)
    
    #if remove_lt is None and remove_gt is not None:
        #print('Generating mask of height < {}m.'.format(remove_gt))
        #return xr.where(ds >= remove_gt, False, True)
    
    #if remove_lt is not None and remove_gt is not None:
        #msg = 'Generating mask of height between {}m and {}m.'
        #print(msg.format(remove_lt, remove_gt))
        #return xr.where((ds <= remove_lt) | 
                        #(ds >= remove_gt), False, True)