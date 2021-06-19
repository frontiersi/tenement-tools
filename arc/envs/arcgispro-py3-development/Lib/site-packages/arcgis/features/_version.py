import os
import time
import uuid
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features import FeatureLayerCollection, FeatureLayer

class VersionManager(object):
    """
    VersionManager allows users to manage the branch versioning for FeatureLayerCollection
    services. The Version Management Service is responsible for exposing the management
    capabilities necessary to support feature services that work with branch versioned
    datasets.

    See the following for more information: https://developers.arcgis.com/rest/services-reference/version-management-service.htm

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required String.  The URI to the web resource.
    ---------------     --------------------------------------------------------------------
    gis                 Required GIS. The enterprise connection to the Portal site.
    ---------------     --------------------------------------------------------------------
    flc                 Optional FeatureLayerCollection. This is the parent container that
                        the branch versioning is enabled on.
    ===============     ====================================================================

    """
    _con = None
    _flc = None
    _gis = None
    _json = None
    _versions = None
    _properties = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis, flc=None):
        """init"""
        if isinstance(gis, GIS):
            self._gis = gis
        else:
            raise ValueError('gis must be of type GIS')
        self._con = self._gis._portal.con
        self._url = url
        if isinstance(flc, FeatureLayer):
            self._flc = flc.container
        elif flc is None or \
           isinstance(flc, FeatureLayerCollection) == False:
            furl = os.path.dirname(url) + "/FeatureServer"
            self._flc = FeatureLayerCollection(url=furl,
                                               gis=self._gis)
        else:
            self._flc = flc
    #----------------------------------------------------------------------
    def __str__(self):
        return "<VersionManager @ {url}>".format(url=self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the service properties"""
        if self._properties is None:
            res = self._con.get(self._url, {'f':'json'})
            self._properties = PropertyMap(res)
        return self._properties
    #----------------------------------------------------------------------
    def create(self, name,
               permission='public',
               description=""):
        """
        Create the named version off of DEFAULT. The version is associated
        with the specified feature service. During creation, the description
        and access (default is public) may be optionally set.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required String. The name of the version
        ---------------     --------------------------------------------------------------------
        permission          Optional String. The access permissions of the new version. The
                            default access permission is public.

                            Values: private, public, protected, or hidden
        ---------------     --------------------------------------------------------------------
        description         Optional String. The description of the new version
        ===============     ====================================================================


        :return: Boolean

        """
        params = {
            'f' : 'json',
            'versionName' : name,
            'description' : description,
            'accessPermission' : permission
        }
        url = self._url + "/create"
        res = self._con.post(url, params)
        self._versions = None
        if 'success' in res:
            return res['success']
        else:
            return res
    #----------------------------------------------------------------------
    def purge(self, version, owner=None):
        """
        Removes a lock from a version


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        version             Required String. The name of the version that is locked.
        ---------------     --------------------------------------------------------------------
        owner               Required String. The owner of the lock. (Deprecated)
        ===============     ====================================================================


        :return: Boolean

        """
        if isinstance(version, Version):
            version = version.properties.versionName
        url = "%s/purgeLock" % self._url
        params = {
            'f': 'json',
            "versionName": version
        }
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return False
    #----------------------------------------------------------------------
    @property
    def locks(self):
        """
        For the specified feature service, return the versions which are locked.
        
        :return: List of locked versions
        
        """
        try:    
            return [v for v in self.all if v.properties.isLocked]
        except:
            return []
        return []
    #----------------------------------------------------------------------
    @property
    def all(self):
        """returns all visible versions on a service"""
        if self._versions is None or \
           len(self._versions) == 0:
            url = "%s/versions" % self._url
            params = {'f':'json'}
            res = self._con.get(url, params)
            self._versions = []
            if 'versions' in res:
                for v in res['versions']:
                    guid = v['versionGuid'][1:-1]
                    vurl = "%s/versions/%s" % (self._url, guid)
                    self._versions.append(Version(url=vurl,
                                                  flc=self._flc,
                                                  gis=self._gis))
            return self._versions
        return self._versions
    #----------------------------------------------------------------------
    def search(self, owner=None, show_hidden=False):
        """
        For the specified feature service, return the info of all versions
        that the client has access to. If the client is the service owner
        (the user that published the service), all versions are accessible
        and will be returned.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        owner               Optional String. A filter the versions by the owner.
        ---------------     --------------------------------------------------------------------
        show_hidden         Optional Boolean. If False (default) hidden versions will not be
                            returned.
        ===============     ====================================================================

        :return: dict

        """
        url = "%s/versionInfos" % self._url
        params = {
            'ownerFilter' : owner,
            'includeHidden' : show_hidden,
            'f' : 'json'
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def get(self, version, mode=None):
        """
        Finds and Locations a Version by it's name

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        version             Required String. This is the name of the version to locate.
        ---------------     --------------------------------------------------------------------
        mode                Optional String. This allows users to get a version in a specific
                            state of edit or read.  If None is provided (default) the version
                            is created without entering a mode.

                            Values:

                                - edit - starts editting mode
                                - read - starts reading mode
                                - None - no mode is started.  This is default.
        ===============     ====================================================================


        """
        for v in self.all:
            if version.lower() == v.properties['versionName'].lower():
                if mode:
                    v.mode = mode
                return v
        return
########################################################################
class Version(object):
    """
    A `Version` represents a single branch in the version tree.

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required String.  The URI to the web resource.
    ---------------     --------------------------------------------------------------------
    gis                 Required GIS. The enterprise connection to the Portal site.
    ---------------     --------------------------------------------------------------------
    flc                 Optional FeatureLayerCollection. This is the parent container that
                        the branch versioning is enabled on.
    ---------------     --------------------------------------------------------------------
    session_guid        Optional String. If a GUID is known for specific version, a user
                        can specify it.
    ---------------     --------------------------------------------------------------------
    mode                Optional String. If a user wants to start either editing or reading
                        on creation of the Version, it can be specified here.  This is useful
                        when a user is using the `Version` with a `with` statement.

                        Allowed Values:

                            + edit - starts an edit session
                            + read - starts a read session

    ===============     ====================================================================

    """
    _flc = None
    _gis = None
    _url = None
    _guid = None
    _mode = None
    _save = None
    _properties = None
    _validation = None
    #----------------------------------------------------------------------
    def __init__(self,
                 url,
                 flc,
                 gis=None,
                 session_guid=None,
                 mode=None):
        """Constructor"""
        if mode:
            self.mode = mode
        self._url = url
        self._save = False
        if gis is None:
            from arcgis import env
            self._gis = env.active_gis
        self._gis = gis
        self._con = self._gis._portal.con
        if session_guid is None:
            self._guid = "{%s-%s-%s-%s-%s}" % (uuid.uuid4().hex[:8],
                                               uuid.uuid4().hex[:4],
                                               uuid.uuid4().hex[:4],
                                               uuid.uuid4().hex[:4],
                                               uuid.uuid4().hex[:12])
        else:
            self._guid = session_guid
        self._flc = flc
    #----------------------------------------------------------------------
    @property
    def validation(self):
        """
        Provides access to a validation manager.

        :returns: ValidationManager
        """
        if self._validation is None:
            from arcgis.mapping import MapImageLayer
            ms = MapImageLayer(url=os.path.dirname(self._flc.url) + "/MapServer", gis=self._gis)
            if 'validationserver' in ms.properties.supportedExtensions.lower():    
                from arcgis.features._validation import ValidationManager
                url = os.path.dirname(self._flc.url) + "/ValidationServer"
                self._validation = ValidationManager(url=url, version=self, gis=self._gis)
        return self._validation
        
    #----------------------------------------------------------------------
    @property
    def parcel_fabric(self):
        """
        Provides access to a parcel fabric manager

        :returns: ParcelFabricManager
        """
        if "controllerDatasetLayers" in self._flc.properties and \
           "parcelLayerId" in self._flc.properties.controllerDatasetLayers:
            from arcgis.features._parcel import ParcelFabricManager
            url = os.path.dirname(self._flc.url) + "/ParcelFabricServer"
            return ParcelFabricManager(url=url, gis=self._gis,
                                       version=self, flc=self._flc)
        return
    #----------------------------------------------------------------------
    @property
    def utility(self):
        """provides access to the utility service manager"""

        if "controllerDatasetLayers" in self._flc.properties and \
           "utilityNetworkLayerId" in self._flc.properties.controllerDatasetLayers:
            from arcgis.features._utility import UtilityNetworkManager
            url = "%s/UtilityNetworkServer" % os.path.dirname(self._flc.url)
            return UtilityNetworkManager(url=url,
                                         version=self)
        return None
    #----------------------------------------------------------------------
    def __str__(self):
        return "<Version {name} @ {guid}>".format(name=self.properties.versionName,
                                                  guid=self.properties.versionGuid)
    #----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()
    #----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the service properties"""
        if self._properties is None:
            res = self._con.get(self._url, {'f':'json'})
            self._properties = PropertyMap(res)
        return self._properties
    #----------------------------------------------------------------------
    @property
    def layers(self):
        """returns the layers in the FeatureLayerCollection"""
        return self._flc.layers
    #----------------------------------------------------------------------
    @property
    def mode(self):
        """
        The `mode` allows versoin editors to start and stop edit, read, or
        view mode.

        Allowed Values:

            + edit - calls the `start_editing` method and creates a lock
            + read - calls the `start_reading` method and creates a lock
            + None - terminates all sessions and lets a user view the version information (default)


        """
        if self.properties.isBeingEdited and \
           self.properties.isBeingRead:
            self._mode = 'edit'
            return 'edit'
        elif self.properties.isBeingEdited == False and \
             self.properties.isBeingRead:
            self._mode = 'read'
            return 'read'
        else:
            self._mode = None
            return None
        return self._mode
    #----------------------------------------------------------------------
    @mode.setter
    def mode(self, value):
        """
        The `mode` allows versoin editors to start and stop edit, read, or
        view mode.

        Allowed Values:

            + edit - calls the `start_editing` method and creates a lock
            + read - calls the `start_reading` method and creates a lock
            + None - terminates all sessions and lets a user view the version information (default)


        """
        # edit means reading is started and edit is started
        # read means reading is start edit is stopped
        # None means reading is stopped and edit is stopped.
        value = str(value).lower()
        if value != str(self.mode).lower():
            if value == 'edit':
                if self.properties.isBeingRead == False:
                    self._mode = None
                    self.start_reading()
                    self._properties = None
                if self.start_editing():
                    self._mode = 'edit'
            elif value == 'read':
                if self.properties.isBeingEdited:
                    self.stop_editing(save=self.save_edits)
                    self._properties = None
                if self.start_reading():
                    self._mode = value
            elif value in [None, 'none']:
                if self.properties.isBeingEdited:
                    self.stop_editing(save=self.save_edits)
                    self._properties = None
                if self.properties.isBeingRead:
                    self._properties = None
                    self.stop_reading()
                    self._properties = None
                    self.stop_reading()
                    self._properties = None
                self._mode = None
            self._properties = None
    #----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the current version

        :return: Boolean

        """
        url = "%s/delete" % os.path.dirname(os.path.dirname(self._url))
        params = {
            'f' : 'json',
            'versionName' : self.properties.versionName,
            'sessionID' : self._guid
        }
        try:
            res = self._con.post(url, params)
        except:
            params.pop("sessionID")
            res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    @property
    def save_edits(self):
        """
        Get/Set the Property to Save the Changes.

        When set to true, any edits performed on the version will be saved.
        """
        return self._save
    #----------------------------------------------------------------------
    @save_edits.setter
    def save_edits(self, value):
        """
        Get/Set the Property to Save the Changes.

        When set to true, any edits performed on the version will be saved.
        """
        if value != self._save:
            self._save = value
    #----------------------------------------------------------------------
    def start_editing(self):
        """
        Starts an edit session for the current user.

        :returns: boolean
        """
        if self.properties.isBeingEdited == False:
            if self.properties.isBeingRead == False:
                self.start_reading()
                self._properties = None
            params = {
            'f' : 'json',
            'sessionID' : self._guid
            }
            url = "%s/startEditing" % self._url
            res = self._con.post(url, params)
            if res['success']:
                self._mode = 'edit'
                self._properties = None
            self._properties = None
            return res['success']
        elif self.properties.isBeingEdited:
            return True
        return False
    #----------------------------------------------------------------------
    def stop_editing(self, save=None):
        """
        Starts an edit session for the current user.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        save                Optional Boolean. States if the values should be saved. If the value
                            is set, the `save_edits` property will be overrided.
        ===============     ====================================================================


        :returns: boolean

        """
        self._properties = None
        if self.properties.isBeingEdited:
            self._mode = None
            if save is None:
                save = self.save_edits
            params = {
            'f' : 'json',
            'sessionID' : self._guid,
            'saveEdits': save
            }
            url = "%s/stopEditing" % self._url
            res = self._con.post(url, params)
            if res['success']:
                self._mode = 'read'
            self._properties = None
            return res['success']
        elif self.properties.isBeingEdited == False:
            return True
        return False
    #----------------------------------------------------------------------
    def start_reading(self):
        """
        Start reading represents a long-term service session. When `start_reading`
        is enabled, it will prevent other users from editing or reconciling the
        version.

        :returns: Boolean

        """
        self._properties = None
        if self.properties.isBeingRead:
            return True
        elif self.properties.isBeingRead == False:
            params = {
            'f' : 'json',
            'sessionID' : self._guid
            }
            url = "%s/startReading" % self._url
            res = self._con.post(url, params)
            if res['success']:
                self._mode = 'read'
                self._properties = None
            return res['success']
        return False
    #----------------------------------------------------------------------
    def stop_reading(self):
        """
        Stops and releases a reading session.

        :returns: Boolean

        """
        self._properties = None
        if self.properties.isBeingRead:

            params = {
            'f' : 'json',
            'sessionID' : self._guid
            }
            url = "%s/stopReading" % self._url
            res = self._con.post(url, params)
            if res['success']:
                self._mode = None
            self._properties = None
            return res['success']
        elif self.properties.isBeingRead == False:
            return True
        return False
    #----------------------------------------------------------------------
    def delete_forward_edits(self, moment):
        """
        If the input moment does not match a specific moment (a moment
        corresponding to an edit operation), the call will return an error.
        The client app must correctly manage the edit session's edit
        operations moments (for example, the undo/redo stack) and not
        blindly pass in a timestamp that could mistakenly delete all the
        forward moments. Thus, the input moment must be equal to a moment
        in which an edit operation for the version was applied. The call
        will also fail if the session does not have a write lock on the
        version.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        moment              Required String. Moment representing the new tail of the version;
                            all forward moments will be trimmed.
        ===============     ====================================================================

        :return: Boolean

        """
        url = "%s/deleteForwardEdits" % self._url
        params = {
            'f' : "json",
            'sessionID' : self._guid,
            'moment' : moment
        }
        res = self._con.post(url, params)
        if 'success' in res:
            return res['success']
        return res
    #----------------------------------------------------------------------
    def reconcile(self,
                  end_with_conflict=False,
                  with_post=False):
        """
        Reconcile a version against the DEFAULT version. The reconcile
        operation requires that you are the only user currently editing the
        version and the only user able to edit the version throughout the
        reconcile process until you save or post. The reconcile operation
        requires that you have full permissions to all the feature classes
        that have been modified in the version being edited. The reconcile
        operation detects differences between the branch version and the
        default version and flags these differences as conflicts. If
        conflicts exist, they should be resolved.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        end_with_conflict      Optional Boolean. Specifies if the reconcile should abort when
                               conflicts are found. The default is False
        ------------------     --------------------------------------------------------------------
        with_post              Optional Boolean. If True the with_post causes a post of the current
                               version following the reconcile.
        ==================     ====================================================================

        """
        if self._mode == 'edit':
            params = {
               'f' : 'json',
               "sessionID" : self._guid,
               'abortIfConflicts' : end_with_conflict,
               'withPost' : with_post
            }
            url = "%s/reconcile" % self._url
            res = self._con.post(url, params)
            return res['success']
        return False
    #----------------------------------------------------------------------
    def restore(self, rows):
        """
        The `restore` method allows users to restore rows from a common 
        ancestor version.  This method is intended to be used when a 
        `DeleteUpdate` conflicts are identified during the last reconcile.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rows                   Required List.  An array of the rows to be restored
        
                               **Syntax**
                               
                               [ 

                                    { 
                                 
                                       "layerId": <layerId>,                                 
                                       "objectIds":[<objectId>] 
                                 
                                    } 
                               ]
                               
                               
                               
        ==================     ====================================================================
        
        :returns: Boolean, String where the Boolean is the Success and the String is the Moment
        
        """
        url = "%s/restoreRows" % self._url
        params = {
            'f' : "json",
            'sessionID' : self._guid,
            'rows' : rows
        }        
        
        res = self._con.post(url, params)
        
        if 'success' in res:
            return res['success'], res.get("moment", "")
        return res        
        
    #----------------------------------------------------------------------
    def alter(self,
              owner=None,
              version=None,
              description=None,
              permission=None):
        """
        The ```alter``` operation changes the geodatabase version's name,
        description, and access permissions.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        owner               Optional String. The new name of the owner.
        ---------------     --------------------------------------------------------------------
        version             Optional String. The new name of the version.
        ---------------     --------------------------------------------------------------------
        permission          Optional String. The new access level of the version.

                            Values: private, public, protected, or hidden
        ---------------     --------------------------------------------------------------------
        description         Optional String. The description of the new version
        ===============     ====================================================================


        :return: Boolean

        """
        url = "%s/alter" % self._url
        params = {
            'f' : 'json'
        }
        if owner or\
           version or\
           description or\
           permission:
            if owner:
                params['ownerName'] = owner
            if version:
                params['versionName'] = version
            if description:
                params['description'] = description
            if permission:
                params['accessPermission'] = permission
            res  = self._con.post(url, params)
            self._properties = None
            return res['success']
        return False
    #----------------------------------------------------------------------
    def differences(self, result_type="objectIds", moment=None):
        """
        The ```differences``` operation allows you to view differences between
        the current version and the default version. The two versions can
        be compared to check for the following conditions.

        - Inserts - features that are present in the current version but not the default version
        - Updates - features that have different attributes or geometry in the current version than the default version
        - Deletions - features that are present in the default version but not in the current version

        Both differences and conflicts will be returned. It is the clients
        responsibility to determine which are differences, and which are conflicts.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        result_type         Required String.  Determines the type of results to return. 
                            The default result type is `objectIds`.

                            Values : `objectIds` or `features`
        ---------------     --------------------------------------------------------------------
        moment              Required String. Moment used to compare current version with default.
        ===============     ====================================================================


        :returns: dict

        """
        url = "%s/differences" % self._url
        params = {
            "f" : "json",
            "sessionID" : self._guid,
            "resultType" : result_type
        }
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def conflicts(self):
        """
        The ```conflicts``` operation allows you to view the conflicts by layer
        and type (update-update, update-delete, delete-update) that were
        identified during the last Reconcile operation. The features that
        are in conflicts will also be returned as they existed in the branch,
        ancestor, and default versions.
        """
        params = {
            'f' : 'json',
            'sessionID' : self._guid
        }
        url = "%s/conflicts" % self._url
        return self._con.post(url, params)
    #----------------------------------------------------------------------
    def inspect(self, conflicts, inspect_all=False, set_inspected=False):
        """
        The ```inspect``` operation allows the client to annotate conflicts
        from the conflict set that was obtained during the last reconcile
        operation. Users can mark the conflicts as being inspected;
        additionally, a description or note can be associated with the
        conflict.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        conflicts           Required List.  The conflicts that are being inspected (removed)
                            from the conflict set.

                            Parameter Format:

                            [
                                {
                                  "layerId" : <layerId>,
                                  "features" : [
                                    {
                                      "objectId" : <objectId>,
                                      "note" : string
                                    }
                                  ]
                                }
                            ]

                            The objectId key is required. The note parameter is optional.

        ---------------     --------------------------------------------------------------------
        inspect_all         Optional Boolean. This parameter, if true, will mark all conflicts
                            as being inspected.
        ---------------     --------------------------------------------------------------------
        set_inspected       Optional Boolean. If True, the examined values will be set to
                            inspected. If ```inspect_all``` is True, this parameter is ignored.
        ===============     ====================================================================


        :return: Boolean


        """
        url = "%s/inspectConflicts" % self._url
        params = {
            'f' : 'json',
            'sessionId' : self._guid,
            'inspectAll' : inspect_all,
            'conflicts' : conflicts,
            'setInspected' : set_inspected
        }
        res = self._con.post(url, params)
        return res['success']
    #----------------------------------------------------------------------
    def post(self):
        """
        The Post operation allows the client to post the changes in their
        branch to the default version. The client can only post changes if
        the branch version has not been modified since the last reconcile.
        If the default version has been modified in the interim, the client
        will have to reconcile again before posting.

        :return: Boolean

        """
        if self._mode == 'edit':
            url = "%s/post" % self._url
            params = {
                "f" : "json",
                "sessionID" : self._guid
            }
            res = self._con.post(url, params)
            return res['success']
        return False
    #----------------------------------------------------------------------
    def __enter__(self):
        if self._mode == 'edit':
            self.start_editing()
        elif self.mode == 'read':
            self.start_reading()
        return self
    #----------------------------------------------------------------------
    def __exit__(self, type, value, traceback):
        self._properties = None
        if self.properties.isLocked:
            if self.properties.isBeingEdited:
                if self._mode != "edit":
                    self._mode = "edit"
                self.stop_editing(self.save_edits)
            if self.properties.isBeingRead:
                if self._mode != "read":
                    self._mode = 'read'
                self.stop_reading()
        if self._mode == 'edit':
            self.stop_editing(save=self.save_edits)
        elif self._mode == 'read':
            self.stop_reading()
    #----------------------------------------------------------------------
    def edit(self,
             layer,
             adds=None,
             updates=None,
             deletes=None,
             use_global_ids=False,
             rollback_on_failure=True):
        """
        The `edit` operation allows users to apply changes to the current version. The edit
        session must be in the mode of `edit` or an exception will be raised.

        =====================   ===========================================
        **Inputs**              **Description**
        ---------------------   -------------------------------------------
        layer                   Required FeatureLayer. The layer to perform
                                the edit on.
        ---------------------   -------------------------------------------
        adds                    Optional FeatureSet/List. The array of
                                features to be added.
        ---------------------   -------------------------------------------
        updates                 Optional FeatureSet/List. The array of
                                features to be updateded.
        ---------------------   -------------------------------------------
        deletes                 Optional FeatureSet/List. string of OIDs to
                                remove from service
        ---------------------   -------------------------------------------
        use_global_ids          Optional boolean. Instead of referencing
                                the default Object ID field, the service
                                will look at a GUID field to track changes.
                                This means the GUIDs will be passed instead
                                of OIDs for delete, update or add features.
        ---------------------   -------------------------------------------
        rollback_on_failure     Optional boolean. Optional parameter to
                                specify if the edits should be applied only
                                if all submitted edits succeed. If false, the
                                server will apply the edits that succeed
                                even if some of the submitted edits fail.
                                If true, the server will apply the edits
                                only if all edits succeed. The default
                                value is true.
        =====================   ===========================================

        :returns: dictionary

        """
        if self._mode == 'edit':
            return layer.edit_features(adds=adds,
                                       updates=updates,
                                       deletes=deletes,
                                       gdb_version=self.properties.versionName,
                                       use_global_ids=use_global_ids,
                                       rollback_on_failure=rollback_on_failure)
        else:
            raise Exception("Version must be in `edit` mode inorder to apply edits to this version.")
        return None
