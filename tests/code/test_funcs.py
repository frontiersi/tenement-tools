# imports
import os
import numpy as np
import xarray as xr
import arcpy

# singlur functions
def create_temp_nc(in_nc, out_nc):
    """duplicate a good nc for use in testing"""
    print('Duplicating cube: {}'.format(in_nc))
    
    if os.path.exists(out_nc):
        os.remove(out_nc)
        
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds.to_netcdf(out_nc)
    
        ds.close()
        del ds


def remove_coord(in_nc, coord='x'):
    """remove coord - e.g. x, y, time, spatial_ref"""
    print('Removing dim: {}'.format(coord))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.drop(coord)
        ds.close()
        
        ds.to_netcdf(in_nc)


def remove_all_vars(in_nc):
    """remove all var data """
    print('Removing all var data')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            ds = ds.drop_vars(ds.data_vars)

        ds.close()
        
        ds.to_netcdf(in_nc)


def remove_var(in_nc, var='nbart_red'):
    """remove var - e.g. nbart_red, oa_fmask"""
    print('Removing var: {}'.format(var))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.drop_vars(var)
        ds.close()
        
        ds.to_netcdf(in_nc)


def set_nc_vars_all_nan(in_nc):
    """set all nc var data to nan"""
    print('Setting nc var data to all nan')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = xr.full_like(ds, fill_value=np.nan)
        ds.close()
        
        ds.to_netcdf(in_nc)


def set_nc_vars_random_all_nan(in_nc, num=10):
    """random set some nc var data to all nan"""
    print('Setting {} random times in nc var data to all nan'.format(num))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()

        if isinstance(ds, xr.Dataset):
            ds = ds.astype('float32')
        
            times = np.random.choice(ds['time'].data, size=num)           
            ds = ds.where(~ds['time'].isin(times))
            
            print(times)

        ds.close()

        ds.to_netcdf(in_nc)
        

def strip_nc_attributes(in_nc):
    """strip attributes from nc"""
    print('Stripping attributes from nc dim')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            ds.attrs = {}

            if 'time' in ds:
                ds['time'].attrs = {}
            if 'x' in ds:
                ds['x'].attrs = {}
            if 'y' in ds:
                ds['y'].attrs = {}
            if 'spatial_ref' in ds:
                ds['spatial_ref'].attrs = {}

            for var in ds.data_vars: 
                ds[var].attrs = {}

        ds.close()
        
        ds.to_netcdf(in_nc)


def set_end_times_to_all_nan(in_nc):
    """sets first and last times to all nan"""
    print('Setting end times to all nan')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            ds = ds.astype('float32')
            
            times = ds['time'].isel(time=slice(1, -1))
            ds = ds.where(ds['time'].isin(times))

        ds.close()
        
        ds.to_netcdf(in_nc)
        

def reduce_to_one_scene(in_nc):
    """remove all images except one"""
    print('Remvoing all images in nc except one random selection')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()

        if isinstance(ds, xr.Dataset):       
            time = np.random.choice(ds['time'].data, size=1)           
            ds = ds.sel(time = time)
            
            print(time)

        ds.close()

        ds.to_netcdf(in_nc)
        

def set_all_specific_season_nan(in_nc, months=[]):
    """for every year, set whole season of data to nan"""
    print('Setting all season to nan with months: {}'.format(months))
    
    if not isinstance(months, list):
        months = [months]
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = ds.where(~ds['time.month'].isin(months))
        ds.close()
        
        ds.to_netcdf(in_nc)
        

def set_specific_years_season_nan(in_nc, years=[], months=[]):
    """for specific years, set season of data to nan"""
    print('Setting season in years: {} to nan with months: {}'.format(years, months))
    
    if not isinstance(years, list):
        years = [years]
    
    if not isinstance(months, list):
        months = [months]
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = ds.where((~ds['time.year'].isin(years)) & 
                      (~ds['time.month'].isin(months)))
        ds.close()
        
        ds.to_netcdf(in_nc)
        
 
def remove_all_specific_season_nan(in_nc, months=[]):
    """for every year, drop whole season of data to nan"""
    print('Remove all season to nan with months: {}'.format(months))
    
    if not isinstance(months, list):
        months = [months]
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = ds.where(~ds['time.month'].isin(months), drop=True)
        ds.close()
        
        ds.to_netcdf(in_nc)
        
        
def remove_specific_years_season_nan(in_nc, years=[], months=[]):
    """for specific years, remove season of data to nan"""
    print('Removing season in years: {} to nan with months: {}'.format(years, months))
    
    if not isinstance(years, list):
        years = [years]
    
    if not isinstance(months, list):
        months = [months]
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = ds.where((~ds['time.year'].isin(years)) & 
                      (~ds['time.month'].isin(months)), 
                       drop=True)
        ds.close()
        
        ds.to_netcdf(in_nc)
        
        
# combo functions
