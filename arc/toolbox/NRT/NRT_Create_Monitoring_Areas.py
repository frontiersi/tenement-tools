import os
import uuid
import arcpy

from shared import arc
from modules import nrt


class NRT_Create_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Create Monitoring Areas'
        self.description = 'Create new monitoring area boundaries and set ' \
                           'monitoring options.'
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

        # set input feature class
        par_in_new_feat = arcpy.Parameter(
            displayName='Draw new monitoring area',
            name='in_new_feat',
            datatype='GPFeatureRecordSetLayer',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_new_feat.filter.list = ['Polygon']
        par_in_new_feat.enabled = False

        # set monitoring area
        par_in_set_area = arcpy.Parameter(
            displayName='Unique identifier',
            name='in_area_id',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
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
        par_in_min_stdv = arcpy.Parameter(
            displayName='Rule 2: minimum zone',
            name='in_min_zone',
            datatype='GPLong',
            parameterType='Required',
            category='Rules',
            direction='Input',
            multiValue=False)
        par_in_min_stdv.filter.type = 'Range'
        par_in_min_stdv.filter.list = [1, 99]
        par_in_min_stdv.value = 2
        par_in_min_stdv.enabled = False

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
            par_in_new_feat,
            par_in_set_area,
            par_in_platform,
            par_in_s_year,
            par_in_e_year,
            par_in_veg_idx,
            par_in_persistence,
            par_in_min_conseqs,
            par_in_inc_plateaus,
            par_in_min_stdv,
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
        global NRT_CREATE_AREA

        # unpack global parameter values
        curr_feat = NRT_CREATE_AREA.get('in_feat')

        # check existing feature input
        if parameters[0].valueAsText != curr_feat:
            try:
                # load column names
                in_feat = parameters[0].valueAsText
                cols = [f.name for f in arcpy.ListFields(in_feat)]
            except:
                cols = []

            # if invalid fields, abort
            if all(f in cols for f in required_fields):
                for i in range(1, 18):
                    parameters[i].enabled = True
            else:
                for i in range(1, 18):
                    parameters[i].enabled = False

        # update global values
        NRT_CREATE_AREA = {'in_feat': parameters[0].valueAsText}

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Create Monitoring Areas tool.
        """



        # grab parameter values
        in_exist_feat = parameters[0]  # input monitoring areas feature
        in_new_feat = parameters[1]  # input new monitoring areas feature
        in_area_id = parameters[2].value  # input monitoring area id
        in_platform = parameters[3].value  # input platform
        in_s_year = parameters[4].value  # input start year
        in_e_year = parameters[5].value  # input end year
        in_veg_idx = parameters[6].value  # input vegetation index
        in_persistence = parameters[7].value  # input persistence
        in_rule_1_min_conseqs = parameters[8].value  # input rule 1 min conseqs
        in_rule_1_inc_plateaus = parameters[9].value  # input rule 1 include plateaus
        in_rule_2_min_zone = parameters[10].value  # input rule 2 min stdvs
        in_rule_3_num_zones = parameters[11].value  # input rule 3 num zones
        in_ruleset = parameters[12].value  # input rulesets
        in_alert_user = parameters[13].value  # input alert user
        in_alert_method = parameters[14].value  # input alert method
        in_alert_direction = parameters[15].value  # input alert direction
        in_email = parameters[16].value  # input email
        in_ignore = parameters[17].value  # input ignore

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Create Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=8)

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
        arcpy.SetProgressorLabel('Validating new feature...')
        arcpy.SetProgressorPosition(2)

        # get path to new layer
        new_feat_desc = arcpy.Describe(in_new_feat)
        in_new_feat = os.path.join(new_feat_desc.path, new_feat_desc.name)

        # check if new feature is valid based on name
        if 'NRT Create Monitoring Areas Draw' not in parameters[1].valueAsText:
            arcpy.AddError('New monitoring feature is incompatible.')
            return

        try:
            # check if only one new feature drawn
            with arcpy.da.SearchCursor(in_new_feat, field_names=['SHAPE@']) as cursor:
                row_count = len([row for row in cursor])
        except Exception as e:
            arcpy.AddError('Could not count new feature records.')
            arcpy.AddMessage(str(e))
            return

        # check if row count is one
        if row_count != 1:
            arcpy.AddError('Only one new monitoring area can be added at a time.')
            return

        try:
            # fetch a list of all existing feature ids
            with arcpy.da.SearchCursor(in_exist_feat, field_names=['area_id']) as cursor:
                existing_area_ids = [row[0] for row in cursor]
        except Exception as e:
            arcpy.AddError('Could not count new feature records.')
            arcpy.AddMessage(str(e))
            return

        # check if existing id already exists
        if in_area_id in existing_area_ids:
            arcpy.AddError('New area identifier {} already used.'.format(in_area_id))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Preparing new feature geometry...')
        arcpy.SetProgressorPosition(3)

        try:
            # temporarily disable auto add to map
            arcpy.env.addOutputsToMap = False

            # ensure new polygon is in albers
            poly_wgs = arcpy.management.Project(in_dataset=in_new_feat,
                                                out_dataset='poly_prj',
                                                out_coor_system=3577)

            # extract binary of current geometry and set as poly
            cursor = arcpy.da.SearchCursor(poly_wgs, ['SHAPE@WKB'])
            poly = cursor.next()[0]

            # re-enable add to map
            arcpy.env.addOutputsToMap = True

        except Exception as e:
            arcpy.AddError('Could not prepare new feature geometry.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Inserting new feature into geodatabase...')
        arcpy.SetProgressorPosition(4)

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
            'ignore': in_ignore,
            'color_border': 0,  # default border color
            'color_fill': 0,  # default fill color
            'global_id': uuid.uuid4().hex,  # generate guid
            'SHAPE@WKB': poly
        }

        try:
            # validate new area before insertion
            nrt.validate_monitoring_area(tuple(data.values())[:-4])
        except Exception as e:
            arcpy.AddError('New monitoring feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        try:
            # insert new area into existing feature
            with arcpy.da.InsertCursor(in_exist_feat, field_names=list(data.keys())) as cursor:
                inserted = cursor.insertRow(list(data.values()))
        except Exception as e:
            arcpy.AddError('Could not insert new area into existing features.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Refreshing spatial index...')
        arcpy.SetProgressorPosition(5)

        try:
            # recalculate spatial index
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping..')
            arcpy.AddMessage(str(e))
            pass

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Removing features on map...')
        arcpy.SetProgressorPosition(6)

        try:
            # for current project, open current map
            aprx = arcpy.mp.ArcGISProject('CURRENT')
            m = aprx.activeMap

            # remove all layers associated with monitoring areas
            for layer in m.listLayers():
                if layer.supports('NAME') and layer.name == 'monitoring_areas':
                    m.removeLayer(layer)
                elif layer.supports('NAME') and 'NRT Create Monitoring Areas Draw' in layer.name:
                    m.removeLayer(layer)
        except Exception as e:
            arcpy.AddWarning('Could not remove new feature from map, skipping.')
            arcpy.AddMessage(str(e))
            pass

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Adding output to map...')
        arcpy.SetProgressorPosition(7)

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
        arcpy.SetProgressorPosition(8)

        # notify user
        arcpy.AddMessage('Created new NRT Monitoring Area.')

        return
