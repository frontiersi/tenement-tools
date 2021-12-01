# nrt
'''
Temp.

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os
import sys
import shutil
import datetime
import arcpy


# todo: remove code for setting defaults if not needed
def create_nrt_project(out_folder, out_filename):
    """
    Creates a new empty geodatabase with required features
    for nrt monitoring tools.
    
    Parameters
    ----------
    out_folder: str
        An output path for new project folder.
    out_filename: str
        An output filename for new project.
    """
    
    # notify
    print('Creating new monitoring project database...')
    
    # check inputs are not none and strings
    if out_folder is None or out_filename is None:
        raise ValueError('Blank folder or filename provided.')
    elif not isinstance(out_folder, str) or not isinstance(out_folder, str):
        raise TypeError('Folder or filename not strings.')
    
    # get full path
    out_filepath = os.path.join(out_folder, out_filename + '.gdb')
    
    # check folder exists
    if not os.path.exists(out_folder):
        raise ValueError('Requested folder does not exist.')
        
    # check file does not already exist
    if os.path.exists(out_filepath):
        raise ValueError('Requested file location arleady exists. Choose a different name.')
    
    # build project geodatbase
    out_filepath = arcpy.management.CreateFileGDB(out_folder, out_filename)
    
    
    # notify
    print('Generating database feature class...')
    
    # temporarily disable auto-visual of outputs
    arcpy.env.addOutputsToMap = False
    
    # create feature class and wgs84 spatial ref sys
    srs = arcpy.SpatialReference(4326)
    out_feat = arcpy.management.CreateFeatureclass(out_path=out_filepath, 
                                                   out_name='monitoring_areas', 
                                                   geometry_type='POLYGON',
                                                   spatial_reference=srs)
    
    
    # notify
    print('Generating database domains...')
    
    # create platform domain
    arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                  domain_name='dom_platforms', 
                                  domain_description='Platform name (Landsat or Sentinel)',
                                  field_type='TEXT', 
                                  domain_type='CODED')
    
    # generate coded values to platform domain
    dom_values = {'Landsat': 'Landsat', 'Sentinel': 'Sentinel'}
    for dom_value in dom_values:
        arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                               domain_name='dom_platforms', 
                                               code=dom_value, 
                                               code_description=dom_values.get(dom_value))
        
    # create index domain
    arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                  domain_name='dom_indices', 
                                  domain_description='Vegetation index name',
                                  field_type='TEXT', 
                                  domain_type='CODED')
    
    # generate coded values to index domain
    dom_values = {'NDVI': 'NDVI', 'MAVI': 'MAVI', 'kNDVI': 'kNDVI'}
    for dom_value in dom_values:
        arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                               domain_name='dom_indices', 
                                               code=dom_value, 
                                               code_description=dom_values.get(dom_value))

    # create year domain
    arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                  domain_name='dom_years', 
                                  domain_description='Training years (1980 - 2050)',
                                  field_type='LONG', 
                                  domain_type='RANGE')
    
    # generate range values to year domain
    arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath, 
                                            domain_name='dom_years', 
                                            min_value=1980, 
                                            max_value=2050)
    
    # create boolean domain
    arcpy.management.CreateDomain(in_workspace=out_filepath, 
                                  domain_name='dom_boolean', 
                                  domain_description='Boolean (Yes or No)',
                                  field_type='TEXT', 
                                  domain_type='CODED')
    
    # generate coded values to boolean domain
    dom_values = {'Yes': 'Yes', 'No': 'No'}
    for dom_value in dom_values:
        arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath, 
                                               domain_name='dom_boolean', 
                                               code=dom_value, 
                                               code_description=dom_values.get(dom_value))
        

    # notify
    print('Generating database fields...') 
    
    # add area id field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='area_id', 
                              field_type='TEXT', 
                              field_alias='Area ID',
                              field_length=200,
                              field_is_required='REQUIRED')
            
    # add platforms field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='platform', 
                              field_type='TEXT', 
                              field_alias='Platform',
                              field_length=20,
                              field_is_required='REQUIRED',
                              field_domain='dom_platforms')    
    
    # add s_year field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='s_year', 
                              field_type='LONG', 
                              field_alias='Start Year of Training Period',
                              field_is_required='REQUIRED',
                              field_domain='dom_years')
    
    # add e_year field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='e_year', 
                              field_type='LONG', 
                              field_alias='End Year of Training Period',
                              field_is_required='REQUIRED',
                              field_domain='dom_years')
    
    # add index field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='index', 
                              field_type='TEXT', 
                              field_alias='Vegetation Index',
                              field_length=20,
                              field_is_required='REQUIRED',
                              field_domain='dom_indices')
    
    # add alert field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='alert', 
                              field_type='TEXT', 
                              field_alias='Alert User',
                              field_length=20,
                              field_is_required='REQUIRED',
                              field_domain='dom_boolean')
    
    # add email field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='email', 
                              field_type='TEXT', 
                              field_alias='Email of User',
                              field_is_required='REQUIRED')
    
    # add last_run field to featureclass   
    arcpy.management.AddField(in_table=out_feat, 
                              field_name='last_run', 
                              field_type='DATE', 
                              field_alias='Last Run',
                              field_is_required='NON_REQUIRED')   
    
    # notify todo - delete if we dont want defaults
    print('Generating database defaults...')  
    
    # set default platform
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='platform',
                                          default_value='Landsat')   

    # set default index
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='index',
                                          default_value='MAVI')        

    # set default alert
    arcpy.management.AssignDefaultToField(in_table=out_feat, 
                                          field_name='alert',
                                          default_value='No')    
           
           
    # notify
    print('Creating NetCDF data folder...') 
    
    # create output folder
    out_nc_folder = os.path.join(out_folder, '{}_cubes'.format(out_filename))
    if os.path.exists(out_nc_folder):
        try:
            shutil.rmtree(out_nc_folder)
        except:
            raise ValueError('Could not delete {}'.format(out_nc_folder))

    # create new folder
    os.makedirs(out_nc_folder)
    
    
    # notify
    print('Adding data to current map...') 
    
    # enable auto-visual of outputs
    arcpy.env.addOutputsToMap = True
    
    try:
        # get active map, add feat
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(out_feat)
    
    except:
        arcpy.AddWarning('Could not find active map. Add monitor areas manually.')        
        
    # notify
    print('Created new monitoring project database successfully.')