import os
import sys
import time
import arcpy
import dask

from modules import nrt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)


class NRT_Monitor_Areas(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Monitor Areas'
        self.description = 'Perform on-demand or on-going vegetation ' \
                           'change monitoring for pre-created monitoring ' \
                           'areas. Optionally send emails when certain changes ' \
                           'occur.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input monitoring area feature
        par_in_feat = arcpy.Parameter(
            displayName='Input monitoring area feature',
            name='in_feat',
            datatype='GPFeatureLayer',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_feat.filter.list = ['Polygon']

        # continuous monitor
        par_continuous = arcpy.Parameter(
            displayName='Continuously monitor areas',
            name='in_continuous',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_continuous.value = False

        # num days for checks
        par_num_days = arcpy.Parameter(
            displayName='Number of days between cycles',
            name='in_num_days',
            datatype='GPLong',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_num_days.filter.type = 'Range'
        par_num_days.filter.list = [1, 365]
        par_num_days.value = 1

        # send email alerts
        par_send_email = arcpy.Parameter(
            displayName='Send email alerts',
            name='in_send_email',
            datatype='GPBoolean',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_send_email.value = False

        # email host (i.e. from)
        par_email_host = arcpy.Parameter(
            displayName='Host email address',
            name='in_email_host',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_email_host.value = None

        # email smtp server
        par_email_server = arcpy.Parameter(
            displayName='Host server address',
            name='in_email_server',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_email_server.value = None

        # email smtp port
        par_email_port = arcpy.Parameter(
            displayName='Host server port',
            name='in_email_port',
            datatype='GPLong',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_email_port.value = None

        # email smtp username
        par_email_username = arcpy.Parameter(
            displayName='Host username',
            name='in_email_username',
            datatype='GPString',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_email_username.value = None

        # email smtp password
        par_email_password = arcpy.Parameter(
            displayName='Host password',
            name='in_email_password',
            datatype='GPStringHidden',
            parameterType='Optional',
            direction='Input',
            multiValue=False)
        par_email_password.value = None

        # combine parameters
        parameters = [
            par_in_feat,
            par_continuous,
            par_num_days,
            par_send_email,
            par_email_host,
            par_email_server,
            par_email_port,
            par_email_username,
            par_email_password
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

        # enable number of days selector
        if parameters[1].value is True:
            parameters[2].enabled = True
        else:
            parameters[2].enabled = False

        # enable email parameters
        if parameters[3].value is True:
            parameters[4].enabled = True
            parameters[5].enabled = True
            parameters[6].enabled = True
            parameters[7].enabled = True
            parameters[8].enabled = True
        else:
            parameters[4].enabled = False
            parameters[5].enabled = False
            parameters[6].enabled = False
            parameters[7].enabled = False
            parameters[8].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Monitor Areas module.
        """

        # set rasterio env
        rio_env = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'VSI_CACHE': 'TRUE',
            'GDAL_HTTP_MULTIRANGE': 'YES',
            'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES'
        }

        # apply rio env settings
        for k, v in rio_env.items():
            os.environ[k] = v

        # grab parameter values
        in_feat = parameters[0]  # input monitoring area feature
        in_continuous = parameters[1].value  # continuous monitoring
        in_num_days = parameters[2].value  # days between checks
        in_send_email = parameters[3].value  # send email alerts
        in_email_host = parameters[4].value  # host email address
        in_email_server = parameters[5].value  # host email server
        in_email_port = parameters[6].value  # host email port
        in_email_username = parameters[7].value  # host email username
        in_email_password = parameters[8].value  # host email password

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Validating monitoring area geodatabase...')

        # get full path to existing monitoring areas
        feat_desc = arcpy.Describe(parameters[0].value)
        in_feat = os.path.join(feat_desc.path, feat_desc.name)

        try:
            # check if input is valid (error if invalid)
            nrt.validate_monitoring_areas(in_feat)
        except Exception as e:
            arcpy.AddError('Monitoring areas feature is incompatible, see messages for details.')
            arcpy.AddMessage(str(e))
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Obtaining and validating monitoring areas...')

        # set required fields
        fields = [
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
            'global_id',
            'SHAPE@'
        ]

        try:
            # get features
            with arcpy.da.SearchCursor(in_feat, fields) as cursor:
                feats = [row for row in cursor]
        except Exception as e:
            arcpy.AddError('Could not read monitoring areas.')
            arcpy.AddMessage(str(e))
            return

        # check if feature exists
        if len(feats) == 0:
            arcpy.AddError('No monitoring areas exist.')
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.SetProgressor('default', 'Converting features to monitoring areas objects...')

        # prepare path to current geodatabase folder
        in_path = os.path.dirname(in_feat)
        in_path = os.path.splitext(in_path)[0]
        in_path = os.path.dirname(in_path)

        try:
            # iter feature and convert to monitoring area objects
            areas = [nrt.MonitoringArea(feat, path=in_path) for feat in feats]
        except Exception as e:
            arcpy.AddError('Could not convert to monitoring area objects.')
            arcpy.AddMessage(str(e))
            return

        # check if areas exist
        if len(areas) == 0:
            arcpy.AddError('No monitoring area objects exist.')
            return

        # # # # #
        # notify and set on-going progess bar
        arcpy.AddMessage('Beginning monitoring cycle...')
        arcpy.SetProgressor('default', 'Monitoring existing areas...')

        # check continuous and num days are valid
        if in_continuous not in [True, False]:
            arcpy.AddError('Did not provide continuous monitoring option.')
            return
        elif in_num_days < 1:
            arcpy.AddError('Number of days between cycles must be >= 1.')
            return

        # prepare monitor iterator
        iterations = 99999 if in_continuous else 1

        # begin monitoring cycles...
        for _ in range(iterations):

            # iter monitoring area objects...
            for area in areas:

                # # # # #
                # notify user and set up progress bar
                arcpy.AddMessage(u'\u200B')
                arcpy.AddMessage('Starting area {}.'.format(area.area_id))
                arcpy.SetProgressor(type='step',
                                    message='Starting area: {}...'.format(area.area_id),
                                    min_range=0, max_range=30)

                # notify and skip area if user requested
                if area.ignore is True:
                    arcpy.AddMessage('Area set to ignore, skipping.')
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Validating monitoring area...')
                arcpy.SetProgressorLabel('Validating monitoring area...')
                arcpy.SetProgressorPosition(1)

                try:
                    # validate area, skip if error
                    area.validate_area()
                except Exception as e:
                    arcpy.AddWarning('Area is invalid, see messages for details.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Checking for existing area NetCDF...')
                arcpy.SetProgressorLabel('Checking for existing area NetCDF...')
                arcpy.SetProgressorPosition(2)

                try:
                    # get old xr, if none set old to none, if error set old to none
                    area.set_old_xr()

                    # notify if no old xr and proceed
                    if area.ds_old is None:
                        arcpy.AddMessage('No existing NetCDF found.')
                        pass
                    else:
                        arcpy.AddMessage('Found existing NetCDF.')
                        pass
                except:
                    arcpy.AddMessage('No existing area NetCDF, fetching a new NetCDF.')
                    pass

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Validating existing NetCDF...')
                arcpy.SetProgressorLabel('Validating existing NetCDF...')
                arcpy.SetProgressorPosition(3)

                try:
                    # validate old xr, skip if none or attrs differ, skip if error
                    area.validate_old_xr()

                    # notify if no old xr and proceed
                    if area.ds_old is None:
                        arcpy.AddMessage('Existing NetCDF is invalid (or does not exist).')
                        pass
                    else:
                        arcpy.AddMessage('Existing NetCDF is valid.')
                        pass
                except:
                    arcpy.AddWarning('Attributes of existing NetCDF have changed, resetting.')
                    pass

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Obtaining new satellite data, if exists...')
                arcpy.SetProgressorLabel('Obtaining new satellite data, if exists...')
                arcpy.SetProgressorPosition(4)

                try:
                    # get all sat data, set new xr to new dates if old exists else all new, on error set new to none and skip
                    area.set_new_xr()

                    # skip if no new dates, else notify and proceed
                    if area.new_xr_dates_found() is False:
                        arcpy.AddMessage('No new satellite images, skipping area.')
                        continue
                    else:
                        arcpy.AddMessage('Found {} new satellite images.'.format(len(area.ds_new['time'])))
                        pass
                except Exception as e:
                    arcpy.AddWarning('Could not obtain new satellite data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                    # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing satellite images with clouds...')
                arcpy.SetProgressor('default', 'Removing satellite images with clouds...')

                try:
                    # apply fmask on new xr. if error or no dates, skip
                    area.apply_new_xr_fmask()

                    # check if any valid images remain, else notify and skip
                    if area.new_xr_dates_found() is False:
                        arcpy.AddMessage('No cloud-free satellite images remain, skipping area.')
                        continue
                    else:
                        arcpy.AddMessage('Found {} cloud-free satellite images'.format(len(area.ds_new['time'])))
                        pass
                except Exception as e:
                    arcpy.AddWarning('Could not apply fmask, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                    # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating vegetation index for new satellite images...')
                arcpy.SetProgressorLabel('Calculating vegetation index for new satellite images...')
                arcpy.SetProgressorPosition(6)

                try:
                    # calculate vege index for new xr. if error or no dates, skip
                    area.apply_new_xr_index()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate vegetation index, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and on-going progress bar
                arcpy.AddMessage('Downloading new satellite data, please wait...')
                arcpy.SetProgressor('default', 'Downloading new satellite data, please wait...')

                try:
                    # load new xr, skip if error
                    area.load_new_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not download satellite data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing edge pixels via mask...')
                arcpy.SetProgressorLabel('Removing edge pixels via mask...')
                arcpy.SetProgressorPosition(8)

                try:
                    # remove edge pixels. if error or empty, return original xr
                    area.remove_new_xr_edges()
                except Exception as e:
                    arcpy.AddWarning('Could not remove edge pixels, proceeding without mask.')
                    arcpy.AddMessage(str(e))
                    pass

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Reducing new satellite images down to temporal medians...')
                arcpy.SetProgressorLabel('Reducing new satellite images down to temporal medians...')
                arcpy.SetProgressorPosition(9)

                try:
                    # reduce new xr to median value per date. skip if error
                    area.reduce_new_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not reduce satellite images, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Interpolating missing pixels...')
                arcpy.SetProgressorLabel('Interpolating missing pixels...')
                arcpy.SetProgressorPosition(10)

                try:
                    # interp nans. if any remain, drop. if empty, error and skip
                    area.interp_new_xr_nans()
                except Exception as e:
                    arcpy.AddWarning('Could not interpolate missing pixels, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Appending required variables...')
                arcpy.SetProgressorLabel('Appending required variables...')
                arcpy.SetProgressorPosition(11)

                try:
                    # append required vars to new xr. skip if error
                    area.append_new_xr_vars()
                except Exception as e:
                    arcpy.AddWarning('Could not append variables, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Combining old and new satellite images, if required...')
                arcpy.SetProgressorLabel('Combining old and new satellite images, if required...')
                arcpy.SetProgressorPosition(12)

                try:
                    # combine old and new xrs. if no old xr, new xr used. if error, skip
                    area.set_cmb_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not combine old and new images, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Removing vegetation spikes in combined images...')
                arcpy.SetProgressorLabel('Removing vegetation spikes in combined images...')
                arcpy.SetProgressorPosition(13)

                try:
                    # replace cmb spikes, interp removed. set result to var veg_clean. skip if error or empty.
                    area.fix_cmb_xr_spikes()
                except Exception as e:
                    arcpy.AddWarning('Could not remove spikes, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Smoothing vegetation in combined images...')
                arcpy.SetProgressorLabel('Smoothing vegetation spike in combined images...')
                arcpy.SetProgressorPosition(14)

                try:
                    # smooth cmb veg_clean var with savitsky filter. skip if error.
                    area.smooth_cmb_xr_index()
                except Exception as e:
                    arcpy.AddWarning('Could not smooth vegetation, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Building analysis dataset...')
                arcpy.SetProgressorLabel('Building analysis dataset...')
                arcpy.SetProgressorPosition(15)

                try:
                    # build anl xr, remove pre-training years. skip if error.
                    area.set_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not build analysis dataset, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Performing change detection on analysis dataset...')
                arcpy.SetProgressorLabel('Performing change detection on analysis dataset...')
                arcpy.SetProgressorPosition(16)

                try:
                    # detect static, dynamic change via anl xr, skip if error.
                    area.detect_change_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not perform change detection, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Smoothing change detection on analysis dataset...')
                arcpy.SetProgressorLabel('Smoothing change detection on analysis dataset...')
                arcpy.SetProgressorPosition(17)

                try:
                    # smooth static, dynamic change via anl xr, skip if error.
                    area.smooth_anl_xr_change()
                except Exception as e:
                    arcpy.AddWarning('Could not smooth change detection data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Transfering old data to analysis dataset...')
                arcpy.SetProgressorLabel('Transfering old data to analysis dataset...')
                arcpy.SetProgressorPosition(18)

                try:
                    # transfer old xr change values to anl xr, skip if error.
                    area.transfer_old_to_anl_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not transfer data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating zones...')
                arcpy.SetProgressorLabel('Calculating zones...')
                arcpy.SetProgressorPosition(19)

                try:
                    # build zones, skip if error.
                    area.build_zones()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate zones, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating rule one, two and three...')
                arcpy.SetProgressorLabel('Calculating rule one, two and three...')
                arcpy.SetProgressorPosition(20)

                try:
                    # build rule one, two, three. skip if error.
                    area.build_rule_one()
                    area.build_rule_two()
                    area.build_rule_three()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate rules, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Calculating alerts...')
                arcpy.SetProgressorLabel('Calculating alerts...')
                arcpy.SetProgressorPosition(21)

                try:
                    # build alerts, skip if error.
                    area.build_alerts()
                except Exception as e:
                    arcpy.AddWarning('Could not calculate alerts, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Transfering analysis data to combined dataset...')
                arcpy.SetProgressorLabel('Transfering analysis data to combined dataset...')
                arcpy.SetProgressorPosition(22)

                try:
                    # transfer anl xr to cmb xr. skip if error.
                    area.transfer_anl_to_cmb_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not transfer analysis data, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Appending area attributes on to combined dataset...')
                arcpy.SetProgressorLabel('Appending area attributes on to combined dataset...')
                arcpy.SetProgressorPosition(23)

                try:
                    # append area field attrs on to cmb xr. skip if error.
                    area.append_cmb_xr_attrs()
                except Exception as e:
                    arcpy.AddWarning('Could not append area attributes, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                    # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Generating area info, html and graph...')
                arcpy.SetProgressorLabel('Generating area info, html and graph...')
                arcpy.SetProgressorPosition(24)

                try:
                    # generate alert info, html, graph data. skip if error.
                    area.set_alert_data()
                except Exception as e:
                    arcpy.AddWarning('Could not generate alert information, html, graphs.')
                    arcpy.AddMessage(str(e))
                    continue

                    # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Refreshing monitoring area symbology...')
                arcpy.SetProgressorLabel('Refreshing monitoring area symbology...')
                arcpy.SetProgressorPosition(25)

                try:
                    # refresh monitoring area symbology. skip if error.
                    area.refresh_area_symbology()
                except Exception as e:
                    arcpy.AddWarning('Could not refresh symbology, see messages.')
                    arcpy.AddMessage(str(e))
                    pass

                    # # # # #
                # notify user and increment progress bar
                arcpy.AddMessage('Exporting combined NetCDF...')
                arcpy.SetProgressorLabel('Exporting combined data...')
                arcpy.SetProgressorPosition(26)

                try:
                    area.export_cmb_xr()
                except Exception as e:
                    arcpy.AddWarning('Could not export NetCDF, see messages.')
                    arcpy.AddMessage(str(e))
                    continue

                    # # # # #
                # notify user and show on-going progress bar
                arcpy.AddMessage('Cooling off for 10 seconds...')
                arcpy.SetProgressor('default', 'Cooling off for 10 seconds...')

                # pause for 10 seconds
                for sec in range(10):
                    time.sleep(1.0)

            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage('Preparing and sending email alert, if requested...')
            arcpy.SetProgressor('default', 'Preparing and sending email alert, if requested...')

            # if requested...
            if in_send_email is True:
                try:
                    # prepare email alerts for each area and send, skip if error
                    nrt.email_alerts(areas,
                                     host_email=in_email_host,
                                     host_server=in_email_server,
                                     host_port=in_email_port,
                                     host_user=in_email_username,
                                     host_pass=in_email_password)
                except Exception as e:
                    arcpy.AddWarning('Could not send email, see messages for details.')
                    arcpy.AddMessage(str(e))
                    pass

            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage(u'\u200B')
            arcpy.AddMessage('Resetting monitoring areas...')
            arcpy.SetProgressor('default', 'Resetting monitoring areas...')

            try:
                # reset each monitor area's internal data
                for area in areas:
                    area.reset()
            except Exception as e:
                arcpy.AddWarning('Could not reset areas, see messages for details.')
                arcpy.AddMessage(str(e))
                pass

            # # # # #
            # notify and set on-going progess bar
            arcpy.AddMessage('Waiting for next cycle, if requested...')
            arcpy.SetProgressor('default', 'Waiting for next cycle, if requested...')

            # if requested, were done!
            if not in_continuous:
                break

                # prepare new progressor for wait
            arcpy.SetProgressor(type='step',
                                message='Waiting for next monitoring cycle...',
                                min_range=0,
                                max_range=in_num_days * 86400)

            # begin iter until next cycle
            for sec in range(in_num_days * 86400):
                arcpy.SetProgressorPosition(sec)
                time.sleep(1.0)

                # ensure we leave on tool cancel properly
                if arcpy.env.isCancelled:
                    sys.exit(0)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(3)

        # notify user
        arcpy.AddMessage('Monitored areas successfully.')
        return
