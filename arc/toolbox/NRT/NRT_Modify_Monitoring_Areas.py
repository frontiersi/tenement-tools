# safe imports
import os
import arcpy

from shared import arc
from modules import nrt

class NRT_Modify_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set input feature class
        self.label = 'NRT Modify Monitoring Areas'
        self.description = 'Modify existing NRT monitoring area features.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # set input feature class
        par_in_feat = arcpy.Parameter(
            displayName='Input monitoring area feature',
            name='in_feat',
            datatype='GPFeatureLayer',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_feat.filter.list = ['Polygon']

        # set monitoring area
        par_in_set_area = arcpy.Parameter(
            displayName='Select the monitoring area to modify',
            name='in_set_area',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_set_area.filter.type = 'ValueList'
        par_in_set_area.enabled = False

        # platform
        par_in_platform = arcpy.Parameter(
            displayName='Satellite platform',
            name='in_platform',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_platform.filter.type = 'ValueList'
        par_in_platform.filter.list = ['Landsat', 'Sentinel']
        par_in_platform.value = 'Landsat'
        par_in_platform.enabled = False

        # start year
        par_in_s_year = arcpy.Parameter(
            displayName='Pre-impact start year',
            name='in_s_year',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_s_year.filter.type = 'Range'
        par_in_s_year.filter.list = [1980, 2050]
        par_in_s_year.enabled = False

        # training length (was end year)
        par_in_e_year = arcpy.Parameter(
            displayName='Minimum number of training dates',
            name='in_e_year',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_in_e_year.filter.type = 'Range'
        par_in_e_year.filter.list = [5, 99999]
        par_in_e_year.enabled = False

        # vegetation index
        par_in_veg_idx = arcpy.Parameter(
            displayName='Vegetation index',
            name='in_veg_idx',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_veg_idx.filter.type = 'ValueList'
        par_in_veg_idx.filter.list = ['NDVI', 'MAVI', 'kNDVI']
        par_in_veg_idx.value = 'MAVI'
        par_in_veg_idx.enabled = False

        # persistence
        par_in_persistence = arcpy.Parameter(
            displayName='Vegetation persistence',
            name='in_persistence',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_persistence.filter.type = 'Range'
        par_in_persistence.filter.list = [0.001, 9.999]
        par_in_persistence.value = 0.5
        par_in_persistence.enabled = False

        # rule 1 min conseqs
        par_in_min_conseqs = arcpy.Parameter(
            displayName='Rule 1: minimum consequtives',
            name='in_min_conseqs',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            category='Rules',
            multiValue=False)
        par_in_min_conseqs.filter.type = 'Range'
        par_in_min_conseqs.filter.list = [0, 99]
        par_in_min_conseqs.value = 3
        par_in_min_conseqs.enabled = False

        # rule 1 include plateaus
        par_in_inc_plateaus = arcpy.Parameter(
            displayName='Rule 1: include plateaus',
            name='in_inc_plateaus',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Rules',
            multiValue=False)
        par_in_inc_plateaus.filter.type = 'ValueList'
        par_in_inc_plateaus.filter.list = ['Yes', 'No']
        par_in_inc_plateaus.value = 'No'
        par_in_inc_plateaus.enabled = False

        # rule 2 minimum zone
        par_in_min_zone = arcpy.Parameter(
            displayName='Rule 2: minimum zone',
            name='in_min_zone',
            datatype='GPLong',
            parameterType='Required',
            category='Rules',
            direction='Input',
            multiValue=False)
        par_in_min_zone.filter.type = 'Range'
        par_in_min_zone.filter.list = [1, 99]
        par_in_min_zone.value = 2
        par_in_min_zone.enabled = False

        # rule 3 number of zones
        par_in_num_zones = arcpy.Parameter(
            displayName='Rule 3: number of zones',
            name='in_num_zones',
            datatype='GPLong',
            parameterType='Required',
            category='Rules',
            direction='Input',
            multiValue=False)
        par_in_num_zones.filter.type = 'Range'
        par_in_num_zones.filter.list = [1, 99]
        par_in_num_zones.value = 2
        par_in_num_zones.enabled = False

        # ruleset
        par_in_ruleset = arcpy.Parameter(
            displayName='Ruleset',
            name='in_ruleset',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Rules',
            multiValue=False)
        par_in_ruleset.filter.type = 'ValueList'
        ruleset = [
            '1 only',
            '2 only',
            '3 only',
            '1 and 2',
            '1 and 3',
            '2 and 3',
            '1 or 2',
            '1 or 3',
            '2 or 3',
            '1 and 2 and 3',
            '1 or 2 and 3',
            '1 and 2 or 3',
            '1 or 2 or 3'
        ]
        par_in_ruleset.filter.list = ruleset
        par_in_ruleset.value = '1 and 2 or 3'
        par_in_ruleset.enabled = False

        # alert user
        par_in_alert_user = arcpy.Parameter(
            displayName='Send alerts',
            name='in_alert_user',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Alerts and Email',
            multiValue=False)
        par_in_alert_user.filter.type = 'ValueList'
        par_in_alert_user.filter.list = ['Yes', 'No']
        par_in_alert_user.value = 'No'
        par_in_alert_user.enabled = False

        # alert method
        par_in_alert_method = arcpy.Parameter(
            displayName='Alert based on method',
            name='in_alert_method',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Alerts and Email',
            multiValue=False)
        par_in_alert_method.filter.type = 'ValueList'
        par_in_alert_method.filter.list = ['Static', 'Dynamic']
        par_in_alert_method.value = 'Static'
        par_in_alert_method.enabled = False

        # alert direction
        par_in_alert_direction = arcpy.Parameter(
            displayName='Direction of change to trigger alert',
            name='in_alert_direction',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Alerts and Email',
            multiValue=False)
        directions = [
            'Incline only (any)',
            'Decline only (any)',
            'Incline only (+ zones only)',
            'Decline only (- zones only)',
            'Incline or Decline (any)',
            'Incline or Decline (+/- zones only)'
        ]
        par_in_alert_direction.filter.type = 'ValueList'
        par_in_alert_direction.filter.list = directions
        par_in_alert_direction.value = 'Decline only (any)'
        par_in_alert_direction.enabled = False

        # email
        par_in_email = arcpy.Parameter(
            displayName='Email to recieve alerts',
            name='in_email',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            category='Alerts and Email',
            multiValue=False)
        par_in_email.enabled = False

        # ignore
        par_in_ignore = arcpy.Parameter(
            displayName='Ignore during monitoring',
            name='in_ignore_user',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            category='Other',
            multiValue=False)
        par_in_ignore.filter.type = 'ValueList'
        par_in_ignore.filter.list = ['Yes', 'No']
        par_in_ignore.value = 'No'
        par_in_ignore.enabled = False

        # combine parameters
        parameters = [
            par_in_feat,
            par_in_set_area,
            par_in_platform,
            par_in_s_year,
            par_in_e_year,
            par_in_veg_idx,
            par_in_persistence,
            par_in_min_conseqs,
            par_in_inc_plateaus,
            par_in_min_zone,
            par_in_num_zones,
            par_in_ruleset,
            par_in_alert_user,
            par_in_alert_method,
            par_in_alert_direction,
            par_in_email,
            par_in_ignore
        ]

        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """
        Enable and disable certain parameters when
        controls are changed on ArcGIS Pro panel.
        """

        # set required fields
        required_fields = [
            'area_id',
            'platform',
            's_year',
            'e_year',
            'index',
            'persistence',
            'rule_1_min_conseqs',
            'rule_1_inc_plateaus',
            'rule_2_min_zone',
            'rule_3_num_zones',
            'ruleset',
            'alert',
            'method',
            'alert_direction',
            'email',
            'ignore',
            'color_border',
            'color_fill',
            'global_id'
        ]

        # globals
        global NRT_MODIFY_AREA

        # unpack global parameter values
        curr_feat = NRT_MODIFY_AREA.get('in_feat')
        curr_area_id = NRT_MODIFY_AREA.get('in_area_id')

        # if not first run or no change to area id, skip
        if curr_area_id is not None and curr_area_id == parameters[1].value:
            return

            # check feature input
        if parameters[0].value is not None:
            try:
                # load column names
                in_feat = parameters[0].valueAsText
                cols = [f.name for f in arcpy.ListFields(in_feat)]
            except:
                cols = []

            # if valid fields, proceed to get all rows
            if all(f in cols for f in required_fields):
                try:
                    with arcpy.da.SearchCursor(in_feat, field_names=required_fields) as cursor:
                        rows = [rec for rec in cursor]
                except:
                    return

                # if first time, get first row values, else user selected
                row = None
                if parameters[1].value is None:
                    row = rows[0]
                else:
                    for row in rows:
                        if row[0] == parameters[1].value:
                            break

                # enable, populate and set parameters
                if row is not None:
                    parameters[1].enabled = True
                    parameters[1].filter.list = [rec[0] for rec in rows]
                    parameters[1].value = row[0]

                    # use loop for rest, they're all the same
                    for i in range(1, 16):
                        parameters[i + 1].enabled = True
                        parameters[i + 1].value = row[i]

                    # update global
                    NRT_MODIFY_AREA = {'in_area_id': parameters[1].value}

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Modify Monitoring Areas tool.
        """
            # grab parameter values
        in_feat = parameters[0]  # input monitoring areas feature
        in_area_id = parameters[1].value  # input monitoring area id
        in_platform = parameters[2].value  # input platform
        in_s_year = parameters[3].value  # input start year
        in_e_year = parameters[4].value  # input end year
        in_veg_idx = parameters[5].value  # input vegetation index
        in_persistence = parameters[6].value  # input persistence
        in_rule_1_min_conseqs = parameters[7].value  # input rule 1 min conseqs
        in_rule_1_inc_plateaus = parameters[8].value  # input rule 1 include plateaus
        in_rule_2_min_zone = parameters[9].value  # input rule 2 min stdvs
        in_rule_3_num_zones = parameters[10].value  # input rule 3 num zones
        in_ruleset = parameters[11].value  # input rulesets
        in_alert_user = parameters[12].value  # input alert user
        in_alert_method = parameters[13].value  # input alert method
        in_alert_direction = parameters[14].value  # input alert direction
        in_email = parameters[15].value  # input email
        in_ignore = parameters[16].value  # input ignore

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Modify Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=7)

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Validating existing feature...')
        arcpy.SetProgressorPosition(1)

        # get full path to existing monitoring areas
        exist_feat_desc = arcpy.Describe(parameters[0].value)
        in_exist_feat = os.path.join(exist_feat_desc.path, exist_feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_exist_feat)
        except Exception as e:
            arcpy.AddError('Existing monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Validating new information...')
        arcpy.SetProgressorPosition(2)

        # set required fields
        data = {
            'area_id': in_area_id,
            'platform': in_platform,
            's_year': in_s_year,
            'e_year': in_e_year,
            'index': in_veg_idx,
            'persistence': in_persistence,
            'rule_1_min_conseqs': in_rule_1_min_conseqs,
            'rule_1_inc_plateaus': in_rule_1_inc_plateaus,
            'rule_2_min_zone': in_rule_2_min_zone,
            'rule_3_num_zones': in_rule_3_num_zones,
            'ruleset': in_ruleset,
            'alert': in_alert_user,
            'method': in_alert_method,
            'alert_direction': in_alert_direction,
            'email': in_email,
            'ignore': in_ignore
        }

        try:
            # validate new area before insertion
            nrt.validate_monitoring_area(tuple(data.values()))
        except Exception as e:
            arcpy.AddError('Modified monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Updating feature...')
        arcpy.SetProgressorPosition(3)

        try:
            # open feature and update it at current id
            with arcpy.da.UpdateCursor(in_exist_feat, field_names=list(data.keys())) as cursor:
                for row in cursor:

                    # update for current id
                    if row[0] == in_area_id:
                        row[1] = data.get('platform')
                        row[2] = data.get('s_year')
                        row[3] = data.get('e_year')
                        row[4] = data.get('index')
                        row[5] = data.get('persistence')
                        row[6] = data.get('rule_1_min_conseqs')
                        row[7] = data.get('rule_1_inc_plateaus')
                        row[8] = data.get('rule_2_min_zone')
                        row[9] = data.get('rule_3_num_zones')
                        row[10] = data.get('ruleset')
                        row[11] = data.get('alert')
                        row[12] = data.get('method')
                        row[13] = data.get('alert_direction')
                        row[14] = data.get('email')
                        row[15] = data.get('ignore')

                        # update row
                        cursor.updateRow(row)
        except Exception as e:
            arcpy.AddError('Could not update existing feature.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Refreshing spatial index...')
        arcpy.SetProgressorPosition(4)

        try:
            # recalculate spatial index
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping..')
            arcpy.AddMessage(str(e))
            pass

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Removing features from map...')
        arcpy.SetProgressorPosition(5)

        try:
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove all layers associated with monitoring areas
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'monitoring_areas':
                    m.removeLayer(layer)
        except Exception as e:
            arcpy.AddWarning('Could not remove new feature from map, skipping.')
            arcpy.AddMessage(str(e))
            pass

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(6)

        try:
            # for current project, open current map, re-add area feature
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap
            m.addDataFromPath(in_exist_feat)

            # update all monitoring area features symbology
            for layer in m.listLayers('monitoring_areas'):
                arc.apply_monitoring_area_symbology(layer)

        except Exception as e:
            arcpy.AddWarning('Could not visualise output, aborting visualisation.')
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
        arcpy.SetProgressorPosition(7)

        # notify user
        arcpy.AddMessage('Modified existing NRT Monitoring Area.')

        return