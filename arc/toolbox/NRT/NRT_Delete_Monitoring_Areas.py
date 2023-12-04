# safe imports
import os
import arcpy

from shared import arc
from modules import nrt

class NRT_Delete_Monitoring_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Delete Monitoring Areas'
        self.description = 'Delete existing NRT monitoring area features.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # set input feature
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
            displayName='Select the monitoring area to delete',
            name='in_area_id',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_set_area.enabled = False

        # combine parameters
        parameters = [
            par_in_feat,
            par_in_set_area
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
        global NRT_DELETE_AREA

        # unpack global parameter values
        curr_feat = NRT_DELETE_AREA.get('in_feat')

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
                try:
                    # get all rows as list of tuples with specific field order
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

                # if row exists, enable + populate controls with first row values
                if row is not None:
                    parameters[1].enabled = True
                    parameters[1].filter.list = [rec[0] for rec in rows]
                    parameters[1].value = row[0]

                else:
                    parameters[1].enabled = True
                    parameters[1].filter.list = []
                    parameters[1].value = None

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Delete Monitoring Areas tool.
        """
        # grab parameter values
        in_exist_feat = parameters[0]  # input monitoring areas feature
        in_area_id = parameters[1].value  # input monitoring area id

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning NRT Delete Monitoring Areas.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=6)

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
        arcpy.SetProgressorLabel('Deleting monitoring area feature...')
        arcpy.SetProgressorPosition(2)

        try:
            # delete feature at current id
            with arcpy.da.UpdateCursor(in_exist_feat, field_names=['area_id']) as cursor:
                for row in cursor:
                    if row[0] == in_area_id:
                        cursor.deleteRow()
        except Exception as e:
            arcpy.AddError('Could not delete existing feature.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Refreshing spatial index...')
        arcpy.SetProgressorPosition(3)

        try:
            # recalculate spatial index
            arcpy.AddSpatialIndex_management(in_exist_feat)
        except Exception as e:
            arcpy.AddWarning('Could not refresh spatial index, skipping.')
            arcpy.AddMessage(str(e))
            pass

            # # # # #
        # notify user and increment progress bar
        arcpy.SetProgressorLabel('Removing features from map...')
        arcpy.SetProgressorPosition(4)

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
        arcpy.SetProgressorPosition(5)

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
        arcpy.SetProgressorPosition(6)

        # notify user
        arcpy.AddMessage('Deleted existing NRT Monitoring Area.')

        return