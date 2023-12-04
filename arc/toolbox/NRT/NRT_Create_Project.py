# safe imports
import os
import arcpy
import dask

from shared import arc

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class NRT_Create_Project(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Create Project'
        self.description = 'Create a new project geodatabase to hold ' \
                           'monitoring areas.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # output gdb folder
        par_out_folder = arcpy.Parameter(
            displayName='Output project folder',
            name='out_folder',
            datatype='DEFolder',
            parameterType='Required',
            direction='Input')

        # combine parameters
        parameters = [par_out_folder]

        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Create Project module.
        """

        # grab parameter values
        out_folder = parameters[0].valueAsText  # output gdb folder path

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Create Project.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=8)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Checking input parameters...')
        arcpy.SetProgressorPosition(1)

        # check inputs are not none and are strings
        if out_folder is None:
            arcpy.AddError('No project folder provided.')
            return
        elif not isinstance(out_folder, str):
            arcpy.AddError('Project folder is not a string.')
            return

        # check if monitoring area gdb already exists
        gdb_path = os.path.join(out_folder, 'monitoring_areas.gdb')
        if os.path.exists(gdb_path):
            arcpy.AddError('Project folder already exists, provide a different folder.')
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Creating new project geodatabase...')
        arcpy.SetProgressorPosition(2)

        try:
            # build project geodatbase
            out_filepath = arcpy.management.CreateFileGDB(out_folder_path=out_folder,
                                                          out_name='monitoring_areas.gdb')
        except Exception as e:
            arcpy.AddError('Could not create file geodatabase.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database feature class...')
        arcpy.SetProgressorPosition(3)

        # temporarily disable auto-visual of outputs
        arcpy.env.addOutputsToMap = False

        try:
            # create feature class and aus albers spatial ref sys
            srs = arcpy.SpatialReference(3577)
            out_feat = arcpy.management.CreateFeatureclass(out_path=out_filepath,
                                                           out_name='monitoring_areas',
                                                           geometry_type='POLYGON',
                                                           spatial_reference=srs)
        except Exception as e:
            arcpy.AddError('Could not create featureclass.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database domains...')
        arcpy.SetProgressorPosition(4)

        try:
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

            # create start year domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_start_years',
                                          domain_description='Training years (1980 - 2050)',
                                          field_type='LONG',
                                          domain_type='RANGE')

            # generate range values to year domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_start_years',
                                                    min_value=1980,
                                                    max_value=2050)

            # create train length domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_train_length',
                                          domain_description='Training length (5 - 99999)',
                                          field_type='LONG',
                                          domain_type='RANGE')

            # generate range values to train length domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_train_length',
                                                    min_value=5,
                                                    max_value=99999)

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

            # create persistence domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_persistence',
                                          domain_description='Vegetation persistence (0.001 - 9.999)',
                                          field_type='FLOAT',
                                          domain_type='RANGE')

            # generate range values to persistence domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_persistence',
                                                    min_value=0.001,
                                                    max_value=9.999)

            # create rule 1 min consequtives domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_rule_1_consequtives',
                                          domain_description='Rule 1 Consequtives (0 - 999)',
                                          field_type='LONG',
                                          domain_type='RANGE')

            # generate range values to consequtives domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_rule_1_consequtives',
                                                    min_value=0,
                                                    max_value=999)

            # create rule 2 min zone domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_rule_2_min_zone',
                                          domain_description='Rule 2 Minimum Zone (1 - 11)',
                                          field_type='LONG',
                                          domain_type='RANGE')

            # generate range values for min zone domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_rule_2_min_zone',
                                                    min_value=1,
                                                    max_value=11)

            # create rule 3 num zones domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_rule_3_num_zones',
                                          domain_description='Rule 3 Number of Zones (1 - 11)',
                                          field_type='LONG',
                                          domain_type='RANGE')

            # generate range values to num zones domain
            arcpy.management.SetValueForRangeDomain(in_workspace=out_filepath,
                                                    domain_name='dom_rule_3_num_zones',
                                                    min_value=1,
                                                    max_value=11)

            # create ruleset domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_ruleset',
                                          domain_description='Various rulesets',
                                          field_type='TEXT',
                                          domain_type='CODED')

            # generate coded values to ruleset domain
            dom_values = {
                '1 only': '1 only',
                '2 only': '2 only',
                '3 only': '3 only',
                '1 and 2': '1 and 2',
                '1 and 3': '1 and 3',
                '2 and 3': '2 and 3',
                '1 or 2': '1 or 2',
                '1 or 3': '1 or 3',
                '2 or 3': '2 or 3',
                '1 and 2 and 3': '1 and 2 and 3',
                '1 or 2 and 3': '1 or 2 and 3',
                '1 and 2 or 3': '1 and 2 or 3',
                '1 or 2 or 3': '1 or 2 or 3'
            }
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath,
                                                       domain_name='dom_ruleset',
                                                       code=dom_value,
                                                       code_description=dom_values.get(dom_value))

                # create alert method domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_alert_method',
                                          domain_description='Alert method',
                                          field_type='TEXT',
                                          domain_type='CODED')

            # generate coded values to alert method domain
            dom_values = {'Static': 'Static', 'Dynamic': 'Dynamic'}
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath,
                                                       domain_name='dom_alert_method',
                                                       code=dom_value,
                                                       code_description=dom_values.get(dom_value))

            # create alert direction domain
            arcpy.management.CreateDomain(in_workspace=out_filepath,
                                          domain_name='dom_alert_direction',
                                          domain_description='Alert directions',
                                          field_type='TEXT',
                                          domain_type='CODED')

            # generate coded values to boolean domain
            dom_values = {
                'Incline only (any)': 'Incline only (any)',
                'Decline only (any)': 'Decline only (any)',
                'Incline only (+ zones only)': 'Incline only (+ zones only)',
                'Decline only (- zones only)': 'Decline only (- zones only)',
                'Incline or Decline (any)': 'Incline or Decline (any)',
                'Incline or Decline (+/- zones only)': 'Incline or Decline (+/- zones only)'
            }
            for dom_value in dom_values:
                arcpy.management.AddCodedValueToDomain(in_workspace=out_filepath,
                                                       domain_name='dom_alert_direction',
                                                       code=dom_value,
                                                       code_description=dom_values.get(dom_value))

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
        except Exception as e:
            arcpy.AddError('Could not create domains.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database fields...')
        arcpy.SetProgressorPosition(5)

        try:
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
                                      field_domain='dom_start_years')

            # add e_year field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='e_year',
                                      field_type='LONG',
                                      field_alias='Training Period Length',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_train_length')

            # add index field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='index',
                                      field_type='TEXT',
                                      field_alias='Vegetation Index',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_indices')

            # add persistence field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='persistence',
                                      field_type='FLOAT',
                                      field_alias='Vegetation Persistence',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_persistence')

            # add rule 1 min consequtives field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='rule_1_min_conseqs',
                                      field_type='LONG',
                                      field_alias='Rule 1 Minimum Consequtives',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_1_consequtives')

            # add include plateaus field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='rule_1_inc_plateaus',
                                      field_type='TEXT',
                                      field_alias='Rule 1 Include Pleateaus',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')

            # add rule 2 min stdv field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='rule_2_min_zone',
                                      field_type='LONG',
                                      field_alias='Rule 2 Minimum Zone',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_2_min_zone')

            # add rule 3 num zones field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='rule_3_num_zones',
                                      field_type='LONG',
                                      field_alias='Rule 3 Number of Zones',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_rule_3_num_zones')

            # add ruleset field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='ruleset',
                                      field_type='TEXT',
                                      field_alias='Ruleset',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_ruleset')

            # add alert field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='alert',
                                      field_type='TEXT',
                                      field_alias='Alert via Email',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')

            # add method field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='method',
                                      field_type='TEXT',
                                      field_alias='Alert via Method',
                                      field_length=20,
                                      field_is_required='REQUIRED',
                                      field_domain='dom_alert_method')

            # add alert direction field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='alert_direction',
                                      field_type='TEXT',
                                      field_alias='Change Direction for Alert',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_alert_direction')

            # add email field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='email',
                                      field_type='TEXT',
                                      field_alias='Email of User',
                                      field_is_required='REQUIRED')

            # add ignore field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='ignore',
                                      field_type='TEXT',
                                      field_alias='Ignore When Run',
                                      field_is_required='REQUIRED',
                                      field_domain='dom_boolean')

            # add color border field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='color_border',
                                      field_type='LONG',
                                      field_alias='Color Code (Border)',
                                      field_is_required='REQUIRED')

            # add color fill field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='color_fill',
                                      field_type='LONG',
                                      field_alias='Color Code (Fill)',
                                      field_is_required='REQUIRED')

            # add global_id field to featureclass
            arcpy.management.AddField(in_table=out_feat,
                                      field_name='global_id',
                                      field_type='TEXT',
                                      field_alias='Global ID',
                                      field_length=200,
                                      field_is_required='REQUIRED')
        except Exception as e:
            arcpy.AddError('Could not create fields.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Generating database defaults...')
        arcpy.SetProgressorPosition(6)

        try:
            # set default platform
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='platform',
                                                  default_value='Landsat')

            # set default index
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='index',
                                                  default_value='MAVI')

            # set default persistence
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='persistence',
                                                  default_value=0.5)

            # set default min conseqs
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='rule_1_min_conseqs',
                                                  default_value=3)

            # set default inc plateaus
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='rule_1_inc_plateaus',
                                                  default_value='No')

            # set default min zone
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='rule_2_min_zone',
                                                  default_value=2)

            # set default num zones
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='rule_3_num_zones',
                                                  default_value=2)

            # set default ruleset
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='ruleset',
                                                  default_value='1 and 2 or 3')

            # set default alert
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='alert',
                                                  default_value='No')

            # set default alert method
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='method',
                                                  default_value='Static')

            # set default alert direction
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='alert_direction',
                                                  default_value='Decline only (any)')

            # set default ignore
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='ignore',
                                                  default_value='No')

            # set default border color
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='color_border',
                                                  default_value=0)

            # set default fill color
            arcpy.management.AssignDefaultToField(in_table=out_feat,
                                                  field_name='color_fill',
                                                  default_value=0)
        except Exception as e:
            arcpy.AddError('Could not assign default values.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(7)

        try:
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(out_feat)
        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
            arcpy.AddMessage(str(e))
            pass

        try:
            # apply symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)
        except Exception as e:
            arcpy.AddWarning('Could not colorise output, aborting colorisation.')
            arcpy.AddMessage(str(e))
            pass

        try:
            # show labels
            for layer in m.listLayers('monitoring_areas'):
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "'Zone: ' +  Text($feature.color_fill)"
                layer.showLabels = True
        except Exception as e:
            arcpy.AddWarning('Could not show labels, aborting label display.')
            arcpy.AddMessage(str(e))
            pass

            # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(8)

        # finish up
        arcpy.AddMessage('Created new NRT Project.')
        return