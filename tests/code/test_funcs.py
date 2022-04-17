# imports
import os
import random
import numpy as np
import xarray as xr
import arcpy

# netcdf corruptors
def create_temp_nc(in_nc, out_nc):
    """duplicate a good nc for use in testing"""
    print('Duplicating cube: {}'.format(in_nc))
    
    if os.path.exists(out_nc):
        ds = xr.open_dataset(out_nc)  # if locked, reset lock and close
        ds = ds.load()
        ds.close()
        
        os.remove(out_nc)
        
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds.to_netcdf(out_nc)
    
        ds.close()
        del ds


def nc_default(in_nc):
    """no changes to raw dataset, used for default test"""
    print('No changes, setting up for default dataset.')


def remove_coord(in_nc, coord='x'):
    """remove coord - e.g. x, y, time, spatial_ref"""
    print('Removing coord: {}'.format(coord))
    
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


def limit_years(in_nc, s_year=2000, e_year=2010):
    """limit dataset to a number of years e.g. 2000-2010"""
    print('Limiting years to {} - {}.'.format(s_year, e_year))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.sel(time=((ds['time.year'] >= s_year) & 
                          (ds['time.year'] <= e_year)))
        
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


def set_nc_vars_all_zero(in_nc):
    """set all nc var data to zero"""
    print('Setting nc var data to all zero')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('int16')
        
        ds = xr.full_like(ds, fill_value=0)
        ds.close()
        
        ds.to_netcdf(in_nc)


def set_nc_vars_all_ones(in_nc):
    """set all nc var data to ones"""
    print('Setting nc var data to all ones')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('int16')
        
        ds = xr.full_like(ds, fill_value=1)
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
        
        ds = ds.where(~((ds['time.year'].isin(years)) & 
                        (ds['time.month'].isin(months))))

        ds.close()
        
        ds.to_netcdf(in_nc)
        
 
def remove_all_specific_season_nan(in_nc, months=[]):
    """for every year, drop whole season of data to nan"""
    print('Remove all season with months: {}'.format(months))
    
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
    print('Removing season in years: {} with months: {}'.format(years, months))
    
    if not isinstance(years, list):
        years = [years]
    
    if not isinstance(months, list):
        months = [months]
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.astype('float32')
        
        ds = ds.where(~((ds['time.year'].isin(years)) & 
                        (ds['time.month'].isin(months))), drop=True)

        ds.close()
        
        ds.to_netcdf(in_nc)


def remove_crs_attr(in_nc):
    """strip crs attribute from nc"""
    print('Stripping crs attribute from nc dim')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            if hasattr(ds, 'crs'):
                ds.attrs.pop('crs')

        ds.close()
        
        ds.to_netcdf(in_nc)


def invalidate_crs_attr(in_nc, crs_text='EPSG:4326'):
    """invalidate crs attribute from nc"""
    print('Invalidating crs attribute from nc dim with: {}'.format(crs_text))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            if hasattr(ds, 'crs'):
                ds.attrs['crs'] = crs_text

        ds.close()
        
        ds.to_netcdf(in_nc)
        

def remove_nodatavals_attr(in_nc):
    """strip nodatavals attribute from nc"""
    print('Stripping nodatavals attribute from nc dim')
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        if isinstance(ds, xr.Dataset):
            if hasattr(ds, 'nodatavals'):
                ds.attrs.pop('nodatavals')

        ds.close()
        
        ds.to_netcdf(in_nc)


# shapefile corruptors
def shp_default(shp_path, temp_shp):
    """no changes to raw shapefile, used for default test"""
    print('No changes, setting up for default shapefile.')
    
    arcpy.env.addOutputsToMap = False
    
    arcpy.management.CopyFeatures(in_features=shp_path, 
                                  out_feature_class=temp_shp)


def project_shp_to_wgs84(shp_path, temp_shp):
    """takes shp path and projects to wgs84"""
    print('Projecting shapefile to wgs84')

    arcpy.env.addOutputsToMap = False

    out_srs = arcpy.SpatialReference(4326)
    arcpy.management.Project(in_dataset=shp_path, 
                             out_dataset=temp_shp, 
                             out_coor_system=out_srs)


def subset_shp_to_area(shp_path, temp_shp, area_code='a'):
    """subset shapefile to specific area"""
    print('Subsetting shapefile to records with area: {}'.format(area_code))

    arcpy.env.addOutputsToMap = False

    arcpy.analysis.Select(in_features=shp_path, 
                          out_feature_class=temp_shp, 
                          where_clause="area = '{}'".format(area_code))


def strip_shp_proj_file(shp_path, temp_shp):
    """strips projection file from shapefile"""
    print('Stripping shapefile of projection file')
    
    arcpy.env.addOutputsToMap = False

    arcpy.management.CopyFeatures(in_features=shp_path, 
                                  out_feature_class=temp_shp)

    prj_file = os.path.splitext(temp_shp)[0] + '.prj'
    os.remove(prj_file)


def convert_shp_pa_field_to_text(shp_path, temp_shp, pa_column='p_a'):
    """converts long field to text for pres/abse field"""
    print('Converting pres/abse field {} to string'.format(pa_column))

    arcpy.env.addOutputsToMap = True

    arcpy.management.CopyFeatures(in_features=shp_path, out_feature_class=temp_shp)

    arcpy.management.AddField(in_table=temp_shp, field_name=pa_column + '_txt', field_type='TEXT')
    arcpy.management.CalculateField(in_table=temp_shp, field=pa_column + '_txt', expression="str(!{}!)".format(pa_column))
    arcpy.management.DeleteField(in_table=temp_shp, drop_field=pa_column)

    arcpy.management.AddField(in_table=temp_shp, field_name=pa_column, field_type='TEXT')
    arcpy.management.CalculateField(in_table=temp_shp, field=pa_column, expression="!{}!".format(pa_column + '_txt'))

    arcpy.management.DeleteField(in_table=temp_shp, drop_field=pa_column + '_txt')


def random_set_shp_p_a_value(shp_path, temp_shp, pa_column, num_rand_samples, set_to_value):
    """random set a specified number of pres/abse values to specified value"""
    print('Setting random {} pres/abse values to {}'.format(num_rand_samples, set_to_value))

    arcpy.env.addOutputsToMap = False

    arcpy.management.CopyFeatures(in_features=shp_path, out_feature_class=temp_shp)

    num_rows = int(arcpy.management.GetCount(temp_shp)[0])
    rand_fids = random.sample(range(0, num_rows - 1), num_rand_samples)

    with arcpy.da.UpdateCursor(temp_shp, ['FID', pa_column]) as cursor:
        for row in cursor:
            if row[0] in rand_fids:
                row[1] = set_to_value
                cursor.updateRow(row)


def random_set_shp_p_a_null(shp_path, temp_shp, pa_column, num_rand_samples):
    """random set a specified number of pres/abse values to null"""
    print('Setting random {} pres/abse values to null'.format(num_rand_samples))

    arcpy.env.addOutputsToMap = False
    
    arcpy.management.CopyFeatures(in_features=shp_path, out_feature_class=temp_shp)

    arcpy.management.AddField(in_table=temp_shp, field_name=pa_column + '_tmp', field_type='LONG')
    arcpy.management.CalculateField(in_table=temp_shp, field=pa_column + '_tmp', expression="!{}!".format(pa_column))
    arcpy.management.DeleteField(in_table=temp_shp, drop_field=pa_column)

    arcpy.management.AddField(in_table=temp_shp, field_name=pa_column, field_type='LONG', field_is_nullable='NULLABLE')
    arcpy.management.CalculateField(in_table=temp_shp, field=pa_column, expression="!{}!".format(pa_column + '_tmp'))
    arcpy.management.DeleteField(in_table=temp_shp, drop_field=pa_column + '_tmp')

    num_rows = int(arcpy.management.GetCount(temp_shp)[0])
    rand_fids = random.sample(range(0, num_rows - 1), num_rand_samples)

    with arcpy.da.UpdateCursor(temp_shp, ['FID', pa_column]) as cursor:
        for row in cursor:
            if row[0] in rand_fids:
                row[1] = None
                cursor.updateRow(row)


def reduce_shp_pa_num_points(shp_path, temp_shp, area_code='a', pa_column='p_a', num_points=5):
    """reduce number of pa points to specified num of points for specified area code"""
    print('Reducing num of pres/abse points to: {} each for area: {}'.format(num_points, area_code))

    arcpy.env.addOutputsToMap = False

    arcpy.analysis.Select(in_features=shp_path, 
                          out_feature_class=temp_shp, 
                          where_clause="area = '{}'".format(area_code))
    
    p_fids, a_fids = [], []
    with arcpy.da.SearchCursor(temp_shp, ['FID', pa_column]) as cursor:
        for row in cursor:
            if row[1] == 1:
                p_fids.append(row[0])
            elif row[1] == 0:
                a_fids.append(row[0])
                
    rand_fids = random.sample(p_fids, num_points) + random.sample(a_fids, num_points)
    
    with arcpy.da.UpdateCursor(temp_shp, ['FID']) as cursor:
        for row in cursor:
            if row[0] not in rand_fids:
                cursor.deleteRow()


def remove_all_shp_points(shp_path, temp_shp, pa_column='p_a'):
    """remove all rows in shapefile to create empty shapefile. note, pa_column not used"""
    print('Removing all rows in shapefile to create empty shapefile')

    arcpy.env.addOutputsToMap = True

    arcpy.management.CopyFeatures(in_features=shp_path, out_feature_class=temp_shp)
    
    with arcpy.da.UpdateCursor(temp_shp, ['FID']) as cursor:
        for row in cursor:
            cursor.deleteRow()

 
def set_all_shp_points_to_value(shp_path, temp_shp, pa_column='p_a', new_val=1):
    """set all pres/abse values in shapefile to specified new value"""
    print('Setting all pres/abse values in shapefile to {}'.format(new_val))

    arcpy.env.addOutputsToMap = True

    arcpy.management.CopyFeatures(in_features=shp_path, out_feature_class=temp_shp)
    
    with arcpy.da.UpdateCursor(temp_shp, ['FID', pa_column]) as cursor:
        for row in cursor:
            row[1] = new_val
            cursor.updateRow(row)



# mask nc corruptors
def shift_dataset_via_coords(in_nc, x=300, y=500):
    """shift mask dataset via coordinates"""
    print('Shifting dataset x: {}, y: {}'.format(x, y))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds['x'] = ds['x'] + x
        ds['y'] = ds['y'] + y
        
        ds.close()
        
        ds.to_netcdf(in_nc)


def reduce_resolution(in_nc, factor=2):
    """reduces dataset resolution by a factor"""
    print('Reducing dataset resolution by factor of: {}'.format(factor))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds = ds.coarsen(x=factor, y=factor, boundary='exact').mean()
        
        ds.close()
        
        ds.to_netcdf(in_nc)
        
        
def multiply_values(in_nc, factor=30):
    """multiply dataset by a factor"""
    print('Multiplyinh dataset by factor of: {}'.format(factor))
    
    if os.path.exists(in_nc):
        ds = xr.open_dataset(in_nc)
        ds = ds.load()
        
        ds['like'] = ds['like'] * factor
        
        ds.close()
        
        ds.to_netcdf(in_nc)