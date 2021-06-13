import os
import datetime
from concurrent.futures import Future
import logging
import json
_log = logging.getLogger(__name__)
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData

class GPJob(object):
    """
    Represents a Single Geoprocessing Job.  The `GPJob` class allows for the asynchronous operation
    of any geoprocessing task.  To request a GPJob task, the code must be called with `future=True`
    or else the operation will occur synchronously.  This class is not intended for users to call
    directly.


    ================  ===============================================================
    **Argument**      **Description**
    ----------------  ---------------------------------------------------------------
    future            Required ccurrent.futures.Future.  The async object created by
                      the geoprocessing (GP) task.
    ----------------  ---------------------------------------------------------------
    gptool            Required Layer. The Geoprocessing Service
    ----------------  ---------------------------------------------------------------
    jobid             Required String. The unique ID of the GP Job.
    ----------------  ---------------------------------------------------------------
    task_url          Required String. The URL to the GP Task.
    ----------------  ---------------------------------------------------------------
    gis               Required GIS. The GIS connection object
    ----------------  ---------------------------------------------------------------
    notify            Optional Boolean.  When set to True, a message will inform the
                      user that the geoprocessing task has completed. The default is
                      False.
    ================  ===============================================================

    """
    _future = None
    _jobid = None
    _url = None
    _gis = None
    _task_name = None
    _is_fa = False
    _is_ra = False
    _is_ortho = False
    _start_time = None
    _end_time = None
    _item_properties = None
    #----------------------------------------------------------------------
    def __init__(self, future, gptool, jobid, task_url, gis, notify=False):
        """
        initializer
        """
        assert isinstance(future, Future)
        self._future = future
        self._start_time = datetime.datetime.now()
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)
        self._gptool = gptool
        self._jobid = jobid
        self._url = task_url
        self._gis = gis
    #----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        if self._end_time:
            return self._end_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time
    #----------------------------------------------------------------------
    def _set_end_time(self, future):
        """sets the finish time"""
        self._end_time = datetime.datetime.now()
    #----------------------------------------------------------------------
    def _notify(self, future):
        """prints finished method"""
        jobid = str(self).replace("<", "").replace(">", "")
        try:
            res = future.result()
            infomsg = '{jobid} finished successfully.'.format(jobid=jobid)
            _log.info(infomsg)
            print(infomsg)
        except Exception as e:
            msg = str(e)
            msg = '{jobid} failed: {msg}'.format(jobid=jobid, msg=msg)
            _log.info(msg)
            print(msg)
    #----------------------------------------------------------------------
    def __str__(self):
        return "<%s GP Job: %s>" % (self.task, self._jobid)
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<%s GP Job: %s>" % (self.task, self._jobid)
    #----------------------------------------------------------------------
    @property
    def task(self):
        """Returns the task name.
        :returns: string
        """
        if self._task_name is None:
            self._task_name = os.path.basename(self._url)
        return self._task_name
    #----------------------------------------------------------------------
    @property
    def messages(self):
        """
        Returns the service's messages
        
        :returns: List
        """
        url = self._url + "/jobs/%s" % self._jobid
        params = {'f' : 'json',
                  'returnMessages': True}


        res = self._gis._con.post(url, params)
        if 'messages' in res:
            return res['messages']
        return []
    #----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the GP status

        :returns: String
        """
        url = self._url + "/jobs/%s" % self._jobid
        params = {'f' : 'json',
                  'returnMessages': True}


        res = self._gis._con.post(url, params)
        if 'jobStatus' in res:
            return res['jobStatus']
        return res
    #----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :returns: boolean
        """
        if self.done():
            return False
        if self.cancelled():
            return False
        try:
            url = self._url + "/jobs/%s/cancel" % self._jobid
            params = {'f' : 'json'}
            res = self._gis._con.post(url, params)
            if 'jobStatus' in res:
                self._future.set_result({'jobStatus' : 'esriJobCancelled'})
                self._future.cancel()
                return True
            self._future.set_result({'jobStatus' : 'esriJobCancelled'})
            self._future.cancel()
            #self._future.set_result({'jobStatus' : 'esriJobCancelled'})
            return res
        except:
            self._future.cancel()
        return True
    #----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :returns: boolean
        """
        return self._future.cancelled()
    #----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :returns: boolean
        """
        return self._future.running()
    #----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :returns: boolean
        """
        return self._future.done()
    #----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :returns: object
        """
        if self.cancelled():
            return None
        if self._is_fa:
            return self._process_fa(self._future.result())
        elif self._is_ra:
            return self._process_ra(self._future.result())
        elif self._is_ortho:
            return self._process_ortho(self._future.result())
        return self._future.result()

    def _process_ortho(self, result):
        """handles the ortho imagery response"""
        import arcgis

        if hasattr(result, '_fields'):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if isinstance(value, dict) and 'featureSet' in value:
                    r[key] = arcgis.features.FeatureCollection(value)
                elif isinstance(value, dict) and 'url' in value and value['url'].lower().find("imageserver"):
                    return value["url"]
                elif isinstance(value, dict) and 'url' in value and value['url'].lower().find("featureserver"):
                    return arcgis.features.FeatureLayerCollection(url=value['url'], gis=self._gis)
                elif isinstance(value, dict) and 'itemId' in value and len(value['itemId']) > 0:
                    if not value['itemId'] in iids:
                        r[key] = arcgis.gis.Item(self._gis, value['itemId'])
                        iids.append(value['itemId'])
                elif len(str(value)) > 0 and value:
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]

            return r
        else:
            value = result
            if self.task == 'AlterProcessingStates':
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    processing_states = value.replace("'",'"')
                    processing_states=json.loads( processing_states.replace('u"','"'))
                    return processing_states

            if isinstance(value, DataFile):
                return self._gis._con.post(value.to_dict()["url"], {})
            if isinstance(value, (RasterData, LinearUnit)):
                return value
            elif isinstance(value, str) and value.lower().find('imageserver') >-1:
                return value
            elif isinstance(value, (dict, tuple, list)) == False:
                return value
            elif 'itemId' in value and \
               len(value['itemId']) > 0:
                itemid = value['itemId']
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif isinstance(value, dict) and "items" in value:
                itemid = list(value['items'].keys())[0]
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif self.task == 'QueryCameraInfo':
                import pandas as pd
                columns = value['schema']
                data = value['content']
                return pd.DataFrame(data, columns=columns)
            elif isinstance(value, dict) and 'url' in value and value['url'].lower().find("imageserver"):
                return value["url"]
            elif isinstance(value, dict) and 'url' in value and value['url'].lower().find("featureserver"):
                return arcgis.features.FeatureLayerCollection(url=value['url'], gis=self._gis)
            elif isinstance(value, dict) and 'featureSet' in value:
                return arcgis.features.FeatureCollection(value)
            return value
        return result

    def _process_ra(self, result):
        import arcgis
        if isinstance(result, arcgis.features.FeatureLayer):
            if self._item_properties:
                _item_properties = {
                    "properties": {
                        "jobUrl": self._url + '/jobs/' + self._jobid,
                        "jobType": "GPServer",
                        "jobId": self._jobid,
                        "jobStatus": "completed"
                    }
                }
                self._return_item.update(item_properties=_item_properties)
            return self._return_item
        if hasattr(result, '_fields'):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if isinstance(value, dict) and 'featureSet' in value:
                    r[key] = arcgis.features.FeatureCollection(value)
                elif isinstance(value, dict) and 'itemId' in value and len(value['itemId']) > 0:
                    if not value['itemId'] in iids:
                        r[key] = arcgis.gis.Item(self._gis, value['itemId'])
                        if self._item_properties:
                            _item_properties = {
                                "properties": {
                                    "jobUrl": self._url + '/jobs/' + self._jobid,
                                    "jobType": "GPServer",
                                    "jobId": self._jobid,
                                    "jobStatus": "completed"
                                }
                            }
                            r[key].update(item_properties=_item_properties)
                        iids.append(value['itemId'])
                elif len(str(value)) > 0 and value:
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]
            if (self.task == "CalculateDistance" or
                self.task == "DetermineOptimumTravelCostNetwork" or
                self.task == "FlowDirection" or
                self.task == "CalculateTravelCost"):
                m = {}
                if isinstance (r,dict):
                    for key, value in r.items():
                        m[key[0:key.rindex('_')+1]+'service'] = r[key]
                    r=m

            if(self.task == 'InterpolatePoints'):
                if "process_info" in  r.keys():
                    process_info = r['process_info']
                    html_final="<b>The following table contains cross validation statistics:</b><br></br><table style='width: 250px;margin-left: 2.5em;'><tbody>"
                    for row in process_info:
                        temp_dict=json.loads(row)
                        if isinstance(temp_dict["message"],list):
                            html_final+="<tr><td>"+temp_dict["message"][0]+"</td><td style='float:right'>"+temp_dict["params"][temp_dict["message"][1].split("${")[1].split("}")[0]]+"</td></tr>"

                    html_final+="</tbody></table><br></br>"
                    from IPython.display import HTML
                    process_info_html = HTML(html_final)
                    r['process_info'] = process_info_html
                    r['output_raster'].update(item_properties={"description":html_final})

            return_value_names = []
            for key, value in r.items():
                return_value_names.append(key)
            num_returns = len(r)
            if num_returns == 1:
                return r[return_value_names[0]]

            else:
                ret_names = []
                for return_value in return_value_names:
                    ret_names.append(return_value)
                import collections
                NamedTuple = collections.namedtuple('FunctionOutput', ret_names)
                function_output = NamedTuple(**r)
                return function_output

        elif isinstance(result, arcgis.raster.ImageryLayer):
            return result
        else:
            value = result
            if isinstance(value, dict) and 'itemId' in value and \
               len(value['itemId']) > 0:
                itemid = value['itemId']
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + '/jobs/' + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed"
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif isinstance(value, dict) and 'url' in value:
                return value["url"]
            elif isinstance(value, dict) and 'contentList' in value:
                if value is "":
                    return None
                elif isinstance(value["contentList"], str):
                    return json.loads(value["contentList"])
                return value['contentList']
            elif isinstance(value, dict) and 'modelInfo' in value:
                try:
                    dict_output =  json.loads(value["modelInfo"])
                    return dict_output
                except:
                    return value
            elif isinstance(value, dict) and 'result' in value:
                return value["result"]
            elif isinstance(value, dict) and "items" in value:
                itemid = list(value['items'].keys())[0]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + '/jobs/' + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed"
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif isinstance(value, dict) and 'featureSet' in value:
                return arcgis.features.FeatureCollection(value)
            elif isinstance(value, list) and value is not None:
                output_model_list = []
                from arcgis.learn import Model
                for element in value:
                    if isinstance(element,dict):
                        if "id" in element.keys():
                            item = arcgis.gis.Item(gis=self._gis, itemid=element["id"])
                            output_model_list.append(Model(item))
                return output_model_list
            elif isinstance(value, dict) and "id" in value:
                itemid = value["id"]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + '/jobs/' + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed"
                        }
                    }
                    item.update(item_properties=_item_properties)
        return result


    def _process_fa(self, result):
        import arcgis
        HAS_ITEM = False
        if hasattr(result, '_fields'):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if self.task in ['AggregatePoints', 'ConnectOriginsToDestinations', 
                                 'SummarizeNearby', "InterpolatePoints"] and \
                   isinstance(value, dict) and 'featureSet' in value:
                    r[key] = arcgis.features.FeatureCollection(value)
                elif isinstance(value, dict) and 'featureSet' in value:
                    HAS_ITEM = True
                    r[key] = arcgis.features.FeatureCollection(value)
                elif isinstance(value, dict) and 'itemId' in value and len(value['itemId']) > 0:
                    if not value['itemId'] in iids:
                        HAS_ITEM = True
                        r[key] = arcgis.gis.Item(self._gis, value['itemId'])
                        iids.append(value['itemId'])
                elif len(str(value)) > 0 and value:
                    r[key] = value
                elif HAS_ITEM == False and \
                     (self.task in ['AggregatePoints', 'CreateWatersheds', 'PlanRoutes',
                                   'ConnectOriginsToDestinations',
                                   'SummarizeNearby', "InterpolatePoints"] or \
                      self.task == 'ConnectOriginsToDestinations'):
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]
            return r
        else:
            value = result
            if 'itemId' in value and \
               len(value['itemId']) > 0:
                itemid = value['itemId']
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif self.task.lower() == 'createroutelayers':
                return [arcgis.gis.Item(gis=self._gis, itemid=itemid) for itemid in result['items']]
            elif isinstance(value, dict) and "items" in value and len(set(value['items'].keys())) == 1:
                itemid = list(value['items'].keys())[0]
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif isinstance(value, dict) and "items" in value and len(set(value['items'].keys())) > 1:
                return [arcgis.gis.Item(gis=self._gis, itemid=itemid) for itemid in result['items']]
            elif isinstance(value, dict) and 'featureSet' in value:
                return arcgis.features.FeatureCollection(value)
            return value
        return result

