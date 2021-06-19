"""

"""
from __future__ import absolute_import
from __future__ import print_function
import csv
from datetime import datetime
from .._common import BaseServer


########################################################################
class LogManager(BaseServer):
    """
    Helper class for the management of logs by administrators. 
    
    Logs are the transaction records written by the various components 
    of ArcGIS Server.  You can query the logs, change various log settings, 
    and check error messages for helping to determine the nature of an issue.

    """
    _url = None
    _con = None
    _json_dict = None
    _json = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis,
                 initialize=False):
        """Constructor

               
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object..
        ==================     ====================================================================
        
        """
        connection = gis
        super(LogManager, self).__init__(gis=gis,
                                  url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def count_error_reports(self, machine="*"):
        """
        This operation counts the number of error reports (crash reports) that have been generated 
        on each machine.
            
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        machine                Optional string. The name of the machine on which to count the 
                               reports. The default will return the count for all machines in a site.
        ==================     ====================================================================

        :return:
           A dictionary with machine name and report count.
        
        """
        params = {
            "f": "json",
            "machine" : machine
        }
        url = self._url + "/countErrorReports"
        return self._con.post(path=url,
                              postdata=params)
    #----------------------------------------------------------------------
    def clean(self):
        """
        Deletes all the log files on all server machines in the site. This is an irreversible
        operation.
        
        This operation forces the server to clean the logs, which has the effect of freeing 
        up disk space. However, it is not required that you invoke this operation because 
        the server periodically purges old logs.
        
        :return:
           A boolean indicating success (True) or failure (False).
           
        """
        params = {
            "f" : "json",
        }
        url = "{}/clean".format(self._url)
        res = self._con.post(path=url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def settings(self):
        """Gets the current log settings. """
        params = {
            "f" : "json"
        }
        url = self._url + "/settings"
        try:
            return self._con.post(path=url,
                                  postdata=params)['settings']
        except:
            return ""
    #----------------------------------------------------------------------
    def edit(self,
             level="WARNING",
             log_dir=None,
             max_age=90,
             max_report_count=10):
        """
        Provides log editing capabilities for the entire site. 
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        level                  Optional string. The log level.  Can be one of (in severity order): 
                               OFF, DEBUG, VERBOSE, FINE, INFO, WARNING, SEVERE. The default is WARNING.
        ------------------     --------------------------------------------------------------------
        log_dir                Optional string. The file path to the root of the log directory.
        ------------------     --------------------------------------------------------------------
        max_age                Optional integer. The number of days that a server should save a 
                               log file. The default is 90.
        ------------------     --------------------------------------------------------------------
        ax_report_count        Optional integer. The maximum number of error report files per 
                               machine. The default is 10.
        ==================     ====================================================================
        

        :return:
           A JSON with the edited settings.
           
        """
        url = self._url + "/settings/edit"
        allowed_levels = ("OFF", "SEVERE", "WARNING", "INFO", "FINE", "VERBOSE", "DEBUG")
        current_settings = self.settings
        current_settings["f"] = "json"

        if level.upper() in allowed_levels:
            current_settings['logLevel'] = level.upper()
        if log_dir is not None:
            current_settings['logDir'] = log_dir
        if max_age is not None and \
           isinstance(max_age, int):
            current_settings['maxLogFileAge'] = max_age
        if max_report_count is not None and \
           isinstance(max_report_count, int) and\
           max_report_count > 0:
            current_settings['maxErrorReportsCount'] = max_report_count
        return self._con.post(path=url,
                              postdata=current_settings)
    #----------------------------------------------------------------------
    def query(self,
              start_time=None,
              end_time=None,
              since_server_start=False,
              level="WARNING",
              services="*",
              machines="*",
              server="*",
              codes=None,
              process_IDs=None,
              export=False,
              export_type="CSV", #CSV or TAB
              out_path=None):
        """
        The query operation on the logs resource provides a way to
        aggregate, filter, and page through logs across the entire site.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        start_time             Optional string. The most recent time to query.  Default is now. 
                               Time can be specified in milliseconds since UNIX epoch, or as an 
                               ArcGIS Server timestamp. For example { "startTime": "2011-08-01T15:17:20,123", ... },
                               { "startTime": 1312237040123, ... }, respectively.
        ------------------     --------------------------------------------------------------------
        end_time               Optional string. The oldest time to include in the result set. You 
                               can use this to limit the query to the last n minutes or hours as 
                               needed. Default is the beginning of all logging.
        ------------------     --------------------------------------------------------------------
        since_server_start     Optional string. Gets only the records written since the server 
                               started (True).  The default is False.
        ------------------     --------------------------------------------------------------------
        level                  Optional string. Gets only the records with a log level at or more 
                               severe than the level declared here. Can be one of (in severity 
                               order): DEBUG, VERBOSE, FINE, INFO, WARNING, SEVERE. The 
                               default is WARNING.
        ------------------     --------------------------------------------------------------------
        services               Optional string. Query records related to a specific service. 
                               The default is all.
        ------------------     --------------------------------------------------------------------
        machines               Optional string. Query records related to a specific machine. 
                               The default is all.
        ------------------     --------------------------------------------------------------------
        server                 Optional string. Query records related to a specific server. 
                               The default is all.
        ------------------     --------------------------------------------------------------------
        codes                  Optional string. Gets only the records with the specified code.  
                               The default is all.  See http://server.arcgis.com/en/server/latest/administer/windows/log-codes-overview.htm
        ------------------     --------------------------------------------------------------------
        process_IDs            Optional string. Query by the machine process ID that logged the event.
        ------------------     --------------------------------------------------------------------
        export                 Optional string. Boolean indicating whether to export the query 
                               results.  The default is False (don't export).
        ------------------     --------------------------------------------------------------------
        export_type            Optional string. The export file type. CSV or TAB are the choices, 
                               CSV is the default.
        ------------------     --------------------------------------------------------------------
        out_path               Optional string. The path to download the log file to.
        ==================     ====================================================================
        
        :return:
           A JSON of the log items that match the query. If export option is set to True, the 
           output log file path is returned.
        
        
        """
     
    
        if codes is None:
            codes = []
        if process_IDs is None:
            process_IDs = []
        allowed_levels = ("SEVERE", "WARNING", "INFO",
                          "FINE", "VERBOSE", "DEBUG")
        qFilter = {
            "services": "*",
            "machines": "*",
            "server" : "*"
        }
        if len(process_IDs) > 0:
            qFilter['processIds'] = process_IDs
        if len(codes) > 0:
            qFilter['codes'] = codes
        params = {
            "f" : "json",
            "sinceServerStart" : since_server_start,
            "pageSize" : 10000
        }
        url = "{url}/query".format(url=self._url)
        if start_time is not None and \
           isinstance(start_time, datetime):
            params['startTime'] = start_time.strftime("%Y-%m-%dT%H:%M:%S,%f")
        if end_time is not None and \
           isinstance(end_time, datetime):
            params['endTime'] = end_time.strftime("%Y-%m-%dT%H:%M:%S,%f")
        if level.upper() in allowed_levels:
            params['level'] = level
        if server != "*":
            qFilter['server'] = server.split(',')
        if services != "*":
            qFilter['services'] = services.split(',')
        if machines != "*":
            qFilter['machines'] = machines.split(",")
        params['filter'] = qFilter
        if export is True and \
           out_path is not None:

            messages = self._con.post(path=url,
                                      postdata=params)
            with open(name=out_path, mode='wb') as f:
                hasKeys = False
                if export_type == "TAB":
                    csvwriter = csv.writer(f, delimiter='\t')
                else:
                    csvwriter = csv.writer(f)
                for message in messages['logMessages']:
                    if hasKeys == False:
                        csvwriter.writerow(message.keys())
                        hasKeys = True
                    csvwriter.writerow(message.values())
                    del message
            del messages
            return out_path
        else:
            return self._con.post(path=url,
                                  postdata=params)
