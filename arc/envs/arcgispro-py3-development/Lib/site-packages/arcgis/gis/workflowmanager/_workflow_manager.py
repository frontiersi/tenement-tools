import ast
import datetime
import json
import sys
import urllib.parse
import arcgis.gis
import re

def _camelCase_to_underscore(name):
    """PEP8ify name"""
    name = name.replace(" ", "_")
    if '_' in name:
        return name.lower()
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _underscore_to_camelcase(name):
    def camelcase(): 
        yield str.lower
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else '_' for x in name.split("_"))

class WorkflowManagerAdmin:
    """
    Represents a a series of CRUD functions for Workflow Manager Items

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    gis                 Optional GIS. The connection to the Enterprise.
    ===============     ====================================================================
    """

    def __init__(self, gis):
        self._gis = gis
        if self._gis.users.me is None:
            raise ValueError("An authenticated `GIS` is required.")
        if not any(prov.itemid == '50a5f00bcc574358b15eab0e2bdadf39' for prov in self._gis.users.me.provisions):
            raise ValueError("No Workflow Manager license is available for the current user")
        self._url = self._wmx_server_url[0]
        if self._url is None:
            raise ValueError("No WorkflowManager Registered with your Organization")
        if not any(prov.itemid == '50a5f00bcc574358b15eab0e2bdadf39' for prov in self._gis.users.me.provisions):
            raise ValueError("No Workflow Manager license is available for the current user")

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl
        res = self._gis._con.get(f"{baseurl}/portals/self/servers", {'f': 'json'})
        for s in res['servers']:
            server_functions = [x.strip() for x in s.get("serverFunction", "").lower().split(",")]
            if 'workflowmanager' in server_functions:
                self._url = s.get("url", None)
                self._private_url = s.get("adminUrl", None)
                if self._url is None:
                    raise RuntimeError("Cannot find a WorkflowManager Server")
                self._url += f"/workflow"
                self._private_url += f"/workflow"
                return self._url, self._private_url
        return None

    def create_item(self, name) -> tuple:
        """
        Creates a `Workflow Manager` schema that stores all the configuration 
        information and location data in the data store on Portal. This can
        be run by any user assigned to the administrator role in Portal.
        
        For users that do not belong to the administrator role, the 
        following privileges are required to run Create Workflow Item:
        
        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        name                Required String. The name of the new schema.
        ==================  =========================================================
        
        :returns: string (item_id)
        """
        url = '{base}/admin/createWorkflowItem?token={token}&name={name}'.format(base=self._url,
                                                                                 token=self._gis._con.token, name=name)
        params = {
            'name': name
        }
        return_obj = json.loads(
            self._gis._con.post(url, params=params, try_json=False, add_token=False, json_encode=False,
                                post_json=True))['itemId']
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return res['success']
        return return_obj


class JobManager:
    """
    Represents a helper class for workflow manager jobs

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    item                The Workflow Manager Item
    ===============     ====================================================================

    """

    def __init__(self, item):
        """initializer"""
        if item is None:
            raise ValueError("Item cannot be None")
        self._item = item
        self._gis = item._gis
        if self._gis.users.me is None:
            raise ValueError("An authenticated `GIS` is required.")

        self._url = self._wmx_server_url[0]
        if self._url is None:
            raise ValueError("No WorkflowManager Registered with your Organization")
        if not any(prov.itemid == '50a5f00bcc574358b15eab0e2bdadf39' for prov in self._gis.users.me.provisions):
            raise ValueError("No Workflow Manager license is available for the current user")
        
    def _handle_error(self, info):
        """Basic error handler - separated into a function to allow for expansion in future releases"""
        error_class = info[0]
        error_text = info[1]
        raise Exception(error_text)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl
        res = self._gis._con.get(f"{baseurl}/portals/self/servers", {'f': 'json'})
        for s in res['servers']:
            server_functions = [x.strip() for x in s.get("serverFunction", "").lower().split(",")]
            if 'workflowmanager' in server_functions:
                self._url = s.get("url", None)
                self._private_url = s.get("adminUrl", None)
                if self._url is None:
                    raise RuntimeError("Cannot find a WorkflowManager Server")
                self._url += f"/workflow/{self._item.id}"
                self._private_url += f"/workflow/{self._item.id}"
                return self._url, self._private_url
        return None

    def close(self, job_ids):
        """
        Closes a single or multiple jobs with specific JobIDs

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             A list of job ID strings
        ===============     ====================================================================

        :return: success object
        """
        try:
            url = '{base}/jobs/manage?token={token}'.format(base=self._url, token=self._gis._con.token)
            return Job.manage_jobs(self._gis, url, job_ids, 'Close')
        except:
            self._handle_error(sys.exc_info())

    def create(self, template, count=1, name=None, start=None, end=None, priority=None, description=None,
                   owner=None, group=None, assigned=None, complete=None, notes=None, parent=None):
        """
        Adds a job to the Workflow Manager instance given a user-defined template

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        template            Workflow Manager Job Template ID (required)
        ---------------     --------------------------------------------------------------------
        count               Number of jobs to create
        ---------------     --------------------------------------------------------------------
        name                Job Name
        ---------------     --------------------------------------------------------------------
        start               Job Start Date
        ---------------     --------------------------------------------------------------------
        end                 Job End Date
        ---------------     --------------------------------------------------------------------
        priority            Job Priority Level
        ---------------     --------------------------------------------------------------------
        description         Job Description
        ---------------     --------------------------------------------------------------------
        owner               Job Owner
        ---------------     --------------------------------------------------------------------
        group               Job Group
        ---------------     --------------------------------------------------------------------
        assigned            Initial Job Assignee
        ---------------     --------------------------------------------------------------------
        complete            Integer Percentage Complete
        ---------------     --------------------------------------------------------------------
        notes               Job Notes
        ---------------     --------------------------------------------------------------------
        parent              Parent Job
        ===============     ====================================================================

        :return: Workflow Manager Job Object
        """
        job_object = {
            "numberOfJobs": count,
            "jobName": name,
            "startDate": start,
            "dueDate": end,
            "priority": priority,
            "description": description,
            "ownedBy": owner,
            "assignedType": group,
            "assignedTo": assigned,
            "percentComplete": complete,
            "notes": notes,
            "parentJob": parent
        }
        filtered_object = {}
        for key in job_object:
            if job_object[key] is not None:
                filtered_object[key] = job_object[key]
        url = '{base}/jobTemplates/{template}/job?token={token}'.format(base=self._url, template=template,
                                                                        token=self._gis._con.token)
        post_job = Job(filtered_object, self._gis, url)
        return_obj = post_job.post()
        return return_obj['jobIds']

    def delete_attachment(self, job_id, attachment_id):
        """
        Deletes a job attachment given a job ID and attachment ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Job ID
        ---------------     --------------------------------------------------------------------
        attachment_id       Attachment ID
        ===============     ====================================================================

        :return: status code
        """
        try:
            res = Job.delete_attachment(self._gis,
                                         '{base}/jobs/{jobId}/attachments/{attachmentId}?token={token}'.format(
                                             base=self._url, jobId=job_id, attachmentId=attachment_id,
                                             item=self._item.id, token=self._gis._con.token))
            return res
        except:
            self._handle_error(sys.exc_info())

    def diagram(self, id):
        """
        Returns the job diagram for the user-defined job

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Job ID
        ===============     ====================================================================

        :return: Workflow Manager Job Diagram Object
        """
        try:
            return JobDiagram.get(self._gis, '{base}/jobs/{job}/diagram'.format(base=self._url, job=id),
                                  {"token": self._gis._con.token})
        except:
            self._handle_error(sys.exc_info())

    def get(self, id):
        """
        Returns an active job with the given ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Job ID
        ===============     ====================================================================

        :return: Workflow Manager Job Object
        """    
        try:
            url = f'{self._url}/jobs/{id}'
            job_dict = ast.literal_eval(str(self._gis._con.get(url, {"token": self._gis._con.token})).encode('cp850', 'replace').decode('utf-8'))
            return Job(job_dict, self._gis, self._url)
        except:
            self._handle_error(sys.exc_info())

    def search(self, query=None, search_string=None, fields=None, display_names=[], sort_by=[], num=10,
                    start_num=0):
        """
        Runs a search against the jobs stored inside the Workflow Manager instance

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        query               SQL query to search against (e.g. "priority='High'")
        ---------------     --------------------------------------------------------------------
        search_str          Search string to search against (e.g. "High")
        ---------------     --------------------------------------------------------------------
        fields              Field list to return
        ---------------     --------------------------------------------------------------------
        display_names       Display names for the return fields
        ---------------     --------------------------------------------------------------------
        sort_by             Field to sort by (e.g. {'field': 'priority', 'sortOrder': 'Asc'})
        ---------------     --------------------------------------------------------------------
        num                 Number of return results
        ---------------     --------------------------------------------------------------------
        start_num           Index of first return value
        ===============     ====================================================================

        :return: List of search results
        """
        try:
            search_object = {
                'q': query,
                'search': search_string,
                'num': num,
                'displayNames': display_names,
                'start': start_num,
                'sortFields': sort_by,
                'fields': fields
            }
            url = '{base}/jobs/search?token={token}'.format(base=self._url, token=self._gis._con.token)
            return Job.search(self._gis, url, search_object)
        except:
            self._handle_error(sys.exc_info())

    def update(self, job_id, update_object):
        """
        Updates a job object by ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              ID for the job to update
        ---------------     --------------------------------------------------------------------
        update_object       an object containing the fields and new values to add to the job
        ===============     ====================================================================

        :return: success object
        """
        try:
            current_job = self.get(job_id).__dict__
            for k in update_object.keys():
                current_job[k] = update_object[k]
            url = '{base}/jobs/{jobId}/update?token={token}'.format(base=self._url, jobId=job_id,
                                                                    token=self._gis._con.token)
            new_job = Job(current_job, self._gis, url)
            return new_job.post()
        except:
            self._handle_error(sys.exc_info())

    def upgrade(self, job_ids):
        """
        Upgrades a single or multiple jobs with specific JobIDs

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             A list of job ID strings
        ===============     ====================================================================

        :return: success object
        """
        try:
            url = '{base}/jobs/manage?token={token}'.format(base=self._url, token=self._gis._con.token)
            return Job.manage_jobs(self._gis, url, job_ids, 'Upgrade')
        except:
            self._handle_error(sys.exc_info())


class WorkflowManager:
    """
    Represents a connection to a Workflow Manager instance or item.

    Users create, update, delete workflow diagrams, job templates and jobs
    or the various other properties with a workflow item.

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    item                The Workflow Manager Item
    ===============     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE: Creating a WorkflowManager object from a workflow item

        from arcgis.workflow import WorkflowManager
        from arcgis.gis import GIS

        # connect to your GIS and get the web map item
        gis = GIS(url, username, password)
        wf_item = gis.content.get('1234abcd_workflow item id')

        # create a WorkflowManager object from the workflow item
        wm = WorkflowManager(wf_item)
        type(wm)
        >> arcgis.workflowmanager.WorkflowManager

        # explore the users in this workflow using the 'users' property
        wm.users
        >> [{}...{}]  # returns a list of dictionaries representing each user
    """

    def __init__(self, item):
        """initializer"""
        if item is None:
            raise ValueError("Item cannot be None")
        self._item = item
        self._gis = item._gis
        if self._gis.users.me is None:
            raise ValueError("An authenticated `GIS` is required.")

        self.job_manager = JobManager(item)

        self._url = self._wmx_server_url[0]
        if self._url is None:
            raise ValueError("No WorkflowManager Registered with your Organization")
        if not any(prov.itemid == '50a5f00bcc574358b15eab0e2bdadf39' for prov in self._gis.users.me.provisions):
            raise ValueError("No Workflow Manager license is available for the current user")

    def _handle_error(self, info):
        """Basic error handler - separated into a function to allow for expansion in future releases"""
        error_class = info[0]
        error_text = info[1]
        raise Exception(error_text)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl
        res = self._gis._con.get(f"{baseurl}/portals/self/servers", {'f': 'json'})
        for s in res['servers']:
            server_functions = [x.strip() for x in s.get("serverFunction", "").lower().split(",")]
            if 'workflowmanager' in server_functions:
                self._url = s.get("url", None)
                self._private_url = s.get("adminUrl", None)
                if self._url is None:
                    raise RuntimeError("Cannot find a WorkflowManager Server")
                self._url += f"/workflow/{self._item.id}"
                self._private_url += f"/workflow/{self._item.id}"
                return self._url, self._private_url
        return None

    @property
    def jobs(self):
        """
        The job manager for a workflow item. See :class:`~arcgis.workflowmanager.JobManager`.
        """

        return self.job_manager

    def evaluate_arcade(self, expression, context = None, context_type = "BaseContext", mode = "Standard"):
        """
        Evaluates an arcade expression
        
        ======================  ===============================================================
        **Argument**            **Description**
        ----------------------  ---------------------------------------------------------------
        expression              Required String.
        ----------------------  ---------------------------------------------------------------
        context                 Optional String.
        ----------------------  ---------------------------------------------------------------
        context_type            Optional String.
        ----------------------  ---------------------------------------------------------------
        mode                    Optional String. 
        ======================  ===============================================================
        
        :returns: String
        """
        url = f"{self._url}/evaluateArcade?token={self._gis._con.token}"
        params = {
            "expression": expression,
            "contextType": context_type,
            "context": context,
            "parseMode": mode
        }
        res = self._gis._con.post(url, params=params, json_encode=False, post_json=True)
        return res.get("result", None)

    @property
    def wm_roles(self):
        """
        Returns an array of user roles available in the local Workflow Manager instance

        :return: array
        """
        try:
            role_array = ast.literal_eval(str(self._gis._con.get('{base}/community/roles'.format(base=self._url),
                                                           params={"token": self._gis._con.token})['roles']).encode(
                'cp850', 'replace').decode('utf-8'))
            return_array = [WMRole(r) for r in role_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def users(self):
        """
        Returns an array of all user profiles stored in Workflow Manager

        :return: array
        """
        try:
            user_array = ast.literal_eval(str(self._gis._con.get('{base}/community/users'.format(base=self._url),
                                                           params={"token": self._gis._con.token})['users']).encode(
                'cp850', 'replace').decode('utf-8'))
            return_array = [self.user(u['username']) for u in user_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def settings(self):
        """
        Returns an array of all settings for the Workflow Manager instance

        :return: array
        """
        try:
            return ast.literal_eval(str(
                self._gis._con.get('{base}/settings'.format(base=self._url), params={"token": self._gis._con.token})[
                    'settings']).encode('cp850', 'replace').decode('utf-8'))
        except:
            self._handle_error(sys.exc_info())

    @property
    def assignable_users(self):
        """
        Get all assignable users for a user in the workflow system

        :return: A list of the assignable user objects
        """
        try:
            user_array = ast.literal_eval(str(self._gis._con.get('{base}/community/usersAssign'.format(base=self._url),
                                                           params={"token": self._gis._con.token})['users']).encode(
                'cp850', 'replace').decode('utf-8'))
            return_array = [self.user(u['username']) for u in user_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def groups(self):
        """
        Returns an array of all user groups stored in Workflow Manager

        :return: array
        """
        try:
            group_array = ast.literal_eval(str(self._gis._con.get('{base}/community/groups'.format(base=self._url),
                                                           params={"token": self._gis._con.token})['groups']).encode(
                'cp850', 'replace').decode('utf-8'))
            return_array = [self.group(g['id']) for g in group_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def assignable_groups(self):
        """
        Get portal groups associated with Workflow Manager roles, to which the current user
        can assign work based on their Workflow Manager assignment privileges.

        :return: A list of the assignable group objects
        """
        try:
            group_array = ast.literal_eval(str(self._gis._con.get('{base}/community/groupsAssign'.format(base=self._url),
                                                           params={"token": self._gis._con.token})['groups']).encode(
                'cp850', 'replace').decode('utf-8'))
            return_array = [self.group(g['id']) for g in group_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def searches(self):
        """
        Returns an array of all saved searches.

        :return: List
        """
        try:
            return ast.literal_eval(str(
                self._gis._con.get('{base}/searches'.format(base=self._url), params={"token": self._gis._con.token})[
                    'searches']).encode('cp850', 'replace').decode('utf-8'))
        except:
            self._handle_error(sys.exc_info())

    @property
    def job_templates(self):
        """
        Gets all the job templates in a workflow item.

        :return: List of all current job templates in the Workflow Manager (required information for create_job call)
        """
        try:
            template_array = ast.literal_eval(str(self._gis._con.get('{base}/jobTemplates'.format(base=self._url),
                                                            params={"token": self._gis._con.token})[
                                            'jobTemplates']).encode('cp850', 'replace').decode('utf-8'))
            return_array = [JobTemplate(t, self._gis, self._url) for t in template_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def diagrams(self):
        """
        Gets the workflow diagrams within the workflow item.

        :return: List of all current diagrams in the Workflow Manager
        """
        try:
            diagram_array = ast.literal_eval(str(
                self._gis._con.get('{base}/diagrams'.format(base=self._url), params={"token": self._gis._con.token})[
                    'diagrams']).encode('cp850', 'replace').decode('utf-8'))
            return_array = [JobDiagram(d, self._gis, self._url) for d in diagram_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    def update_settings(self, props):
        """
        Returns an active job with the given ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        props               a list of Props objects to update 
                            (Prop object example: {'propName': 'string', 'value': 'string'})
        ===============     ====================================================================

        :return: success object
        """
        url = '{base}/settings?token={token}'.format(base=self._url, token=self._gis._con.token)
        params = {
            'settings': props
        }
        return_obj = json.loads(
            self._gis._con.post(url, params, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return return_obj

    def wm_role(self, name):
        """
        Returns an active role with the given name

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Role Name
        ===============     ====================================================================

        :return: Workflow Manager Role Object
        """
        try:
            return WMRole.get(self._gis,
                            '{base}/community/roles/{role}'.format(base=self._url, role=urllib.parse.quote(name),
                                                                   item=self._item.id), {"token": self._gis._con.token})
        except:
            self._handle_error(sys.exc_info())

    def job_template(self, id):
        """
        Returns a job template with the given ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Job Template ID
        ===============     ====================================================================

        :return: Workflow Manager Job Template Object
        """
        try:
            print(id)
            return JobTemplate.get(self._gis, '{base}/jobTemplates/{jobTemplate}'.format(base=self._url, jobTemplate=id),
                                   {"token": self._gis._con.token})
        except:
            self._handle_error(sys.exc_info())

    def delete_job_template(self, id):
        """
        Deletes a job template with the given ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Job Template ID
        ===============     ====================================================================

        :return: status code
        """
        try:
            res = JobTemplate.delete(self._gis,
                                      '{base}/jobTemplates/{jobTemplate}?token={token}'.format(base=self._url,
                                                                                               jobTemplate=id,
                                                                                               item=self._item.id,
                                                                                               token=self._gis._con.token))
            return res
        except:
            self._handle_error(sys.exc_info())

    def user(self, username):
        """
        Returns a user profile with the given username

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        username            Workflow Manager Username
        ===============     ====================================================================

        :return: Workflow Manager User Object
        """
        try:
            return arcgis.gis.User(self._gis, username)
        except:
            self._handle_error(sys.exc_info())

    def group(self, group_id):
        """
        Returns group information with the given group ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        group_id            Workflow Manager Group ID
        ===============     ====================================================================

        :return: Workflow Manager Group Object
        """
        try:
            wmx_group = Group.get(self._gis, '{base}/community/groups/{groupid}'.format(base=self._url, groupid=group_id,
                                                                                   item=self._item.id),
                             {"token": self._gis._con.token})
            arcgis_group = arcgis.gis.Group(self._gis, group_id)
            arcgis_group.roles = wmx_group.roles
            return arcgis_group
        except:
            self._handle_error(sys.exc_info())

    def diagram(self, id):
        """
        Returns the diagram with the given ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Diagram ID
        ===============     ====================================================================

        :return: Workflow Manager Job Diagram Object
        """
        try:
            return JobDiagram.get(self._gis, '{base}/diagrams/{diagram}'.format(base=self._url, diagram=id),
                                  {"token": self._gis._con.token})
        except:
            self._handle_error(sys.exc_info())

    def create_wm_role(self, name, description='', privileges=[]):
        """
        Adds a role to the Workflow Manager instance given a user-defined name

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Role Name (required)
        ---------------     --------------------------------------------------------------------
        description         Role Description
        ---------------     --------------------------------------------------------------------
        privileges          List of privileges associated with the role
        ===============     ====================================================================

        :return: Workflow Manager Role Object
        """
        try:
            url = '{base}/community/roles/{name}?token={token}'.format(base=self._url, name=name,
                                                                       token=self._gis._con.token)
            post_role = WMRole({
                "roleName": name,
                "description": description,
                "privileges": privileges
            })
            return post_role.post(self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    def create_job_template(self, name, priority, id="", category="", job_duration=0,
                            assigned_to="", default_due_date=None, default_start_date=None,
                            start_date_type="CreationDate", diagram_id="", diagram_name="",
                            assigned_type="Unassigned", description="", default_description="",
                            state="Draft", last_updated_by="", last_updated_date=None,
                            extended_property_table_definitions=[]):
        """
        Adds a job template to the Workflow Manager instance given a user-defined name and default priority level

        ====================================     ====================================================================
        **Argument**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        name                                     Job Template Name (required)
        ------------------------------------     --------------------------------------------------------------------
        priority                                 Default Job Template Priority Level (required)
        ------------------------------------     --------------------------------------------------------------------
        id                                       Job Template ID
        ------------------------------------     --------------------------------------------------------------------
        category                                 Job Template Category
        ------------------------------------     --------------------------------------------------------------------
        job_duration                             Default Job Template Duration
        ------------------------------------     --------------------------------------------------------------------
        assigned_to                              Job Owner
        ------------------------------------     --------------------------------------------------------------------
        default_due_date                         Due Date for Job Template
        ------------------------------------     --------------------------------------------------------------------
        default_start_date                       Start Date for Job Template
        ------------------------------------     --------------------------------------------------------------------
        start_date_type                          Type of Start Date (e.g. creationDate)
        ------------------------------------     --------------------------------------------------------------------
        diagram_id                               Job Template Diagram ID
        ------------------------------------     --------------------------------------------------------------------
        diagram_name                             Job Template Diagram Name
        ------------------------------------     --------------------------------------------------------------------
        assigned_type                            Type of Job Template Assignment
        ------------------------------------     --------------------------------------------------------------------
        description                              Job Template Description
        ------------------------------------     --------------------------------------------------------------------
        default_description                      Default Job Template Description
        ------------------------------------     --------------------------------------------------------------------
        state                                    Default Job Template State
        ------------------------------------     --------------------------------------------------------------------
        last_updated_by                          User Who Last Updated Job Template
        ------------------------------------     --------------------------------------------------------------------
        last_updated_date                        Date of Last Job Template Update
        ------------------------------------     --------------------------------------------------------------------
        extended_property_table_definitions      List of Extended Properties for Job Template
        ====================================     ====================================================================

        :return: Workflow Manager Job Template ID
        """
        try:
            if default_due_date is None:
                default_due_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            if default_start_date is None:
                default_start_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            if last_updated_date is None:
                last_updated_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            url = '{base}/jobTemplates?token={token}'.format(base=self._url, token=self._gis._con.token)
            post_job_template = JobTemplate({
                "jobTemplateId": id,
                "jobTemplateName": name,
                "category": category,
                "defaultJobDuration": job_duration,
                "defaultAssignedTo": assigned_to,
                "defaultDueDate": default_due_date,
                "defaultStartDate": default_start_date,
                "jobStartDateType": start_date_type,
                "diagramId": diagram_id,
                "diagramName": diagram_name,
                "defaultPriorityName": priority,
                "defaultAssignedType": assigned_type,
                "description": description,
                "defaultDescription": default_description,
                "state": state,
                "extendedPropertyTableDefinitions": extended_property_table_definitions,
                "lastUpdatedBy": last_updated_by,
                "lastUpdatedDate": last_updated_date
            })
            return post_job_template.post(self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    def update_job_template(self, template):
        """
        Updates a job template object by ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        body                Job Template body - existing Job Template object that inherits required/optional
                            fields.
        ===============     ====================================================================

        :return: success object
        """
        try:
            url = '{base}/jobTemplates/{jobTemplate}?token={token}'.format(base=self._url,
                                                                           jobTemplate=template['job_template_id'],
                                                                           item=self._item.id,
                                                                           token=self._gis._con.token)
            template_object = JobTemplate(template)
            res = template_object.put(self._gis, url)
            return res
        except:
            self._handle_error(sys.exc_info())

    def create_diagram(self, name, steps, display_grid, description="", active=False, annotations=[], data_sources=[]):
        """
        Adds a diagram to the Workflow Manager instance given a user-defined name and array of steps

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Diagram Name (required)
        ---------------     --------------------------------------------------------------------
        steps               List of Step objects associated with the Diagram (required)
        ---------------     --------------------------------------------------------------------
        display_grid        Boolean indicating whether the grid will be displayed in the Diagram (required)
        ---------------     --------------------------------------------------------------------
        description         Diagram description
        ---------------     --------------------------------------------------------------------
        active              Boolean indicating whether the Diagram is active
        ---------------     --------------------------------------------------------------------
        annotations         List of Annotation objects associated with the Diagram
        ---------------     --------------------------------------------------------------------
        data_sources        List of Data Source objects associated with the Diagram
        ===============     ====================================================================

        :return: Workflow Manager Diagram ID
        """
        try:
            url = '{base}/diagrams?token={token}'.format(base=self._url, token=self._gis._con.token)
            post_diagram = JobDiagram({
                "diagramId": "",
                "diagramName": name,
                "description": description,
                "active": active,
                "initialStepId": "",
                "initialStepName": "",
                "steps": steps,
                "dataSources": data_sources,
                "annotations": annotations,
                "displayGrid": display_grid
            })
            return post_diagram.post(self._gis, url)['diagram_id']
        except:
            self._handle_error(sys.exc_info())

    def update_diagram(self, body, delete_draft=True):
        """
        Updates a diagram object by ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        body                Diagram body - existing Diagram object that inherits required/optional
                            fields.
        ---------------     --------------------------------------------------------------------
        delete_draft        Boolean - option to delete the Diagram draft (optional)
        ===============     ====================================================================

        :return: success object
        """
        try:
            url = '{base}/diagrams/{diagramid}?token={token}'.format(base=self._url, diagramid=body['diagram_id'],
                                                                     token=self._gis._con.token)
            post_diagram = JobDiagram({
                "diagramId": body['diagram_id'],
                "diagramName": body['diagram_name'],
                "description": (body['description'] if 'description' in body else ''),
                "active": (body['active'] if 'active' in body else False),
                "initialStepId": (body['initial_step_id'] if 'initial_step_id' in body else ''),
                "initialStepName": (body['initial_step_name'] if 'initial_step_name' in body else ''),
                "steps": body['steps'],
                "dataSources": (body['data_sources'] if 'data_sources' in body else []),
                "annotations": (body['annotations'] if 'annotations' in body else ''),
                "displayGrid": body['display_grid']
            })
            res = post_diagram.update(self._gis, url, delete_draft)

            return res
        except:
            self._handle_error(sys.exc_info())

    def delete_diagram(self, id):
        """
        Deletes a diagram object by ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Diagram id (required)
        ===============     ====================================================================

        :return: Workflow Manager Diagram ID
        """
        try:
            url = '{base}/diagrams/{diagramid}?token={token}'.format(base=self._url, diagramid=id,
                                                                     token=self._gis._con.token)
            return JobDiagram.delete(self._gis, url)
        except:
            self._handle_error(sys.exc_info())


class Job(object):
    """
    Helper class for managing Workflow Manager jobs in a workflow item. This class is not created
    by users directly. An instance of this class, can be created by calling `jobs.get(id)` with the appropriate ID
    for the job. `jobs` is available as property on `arcgis.WorkflowManager`.

    """
    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        self.job_status = self.notes = self.attachments = self.diagram_id = self.end_date = self.due_date = self.description = self.started_date = self.current_steps = self.job_template_name = self.job_template_id = self.extended_properties = self.diagram_name = self.parent_job = self.job_name = self.diagram_version = self.active_versions = self.percent_complete = self.priority = self.job_id = self.created_date = self.created_by = self.closed = self.owned_by = self.start_date = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url

    def post(self):
        post_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None and not k.startswith('_')}
        return_obj = json.loads(
            self._gis._con.post(self._url, post_dict, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return return_obj

    def search(gis, url, search_object):
        return_obj = json.loads(
            gis._con.post(url, search_object, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def get_attachment(self, attachment_id):
        """
        Returns a job attachment given an attachment ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment_id       Attachment ID
        ===============     ====================================================================

        :return: Job Attachment
        """

        url = '{base}/jobs/{jobId}/attachments/{attachmentId}'.format(base=self._url, jobId=self.job_id,
                                                            attachmentId=attachment_id)
        return_obj = self._gis._con.get(url, {"token": self._gis._con.token}, try_json=False)
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return return_obj

    def add_attachment(self, attachment):
        """
        Adds an attachment to the job

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment          Filepath to attachment
        ===============     ====================================================================

        :return: Job Attachment
        """
        url = '{base}/jobs/{jobId}/attachments'.format(base=self._url, jobId=self.job_id)
        return_obj = json.loads(
            self._gis._con.post(url, files={'attachment': attachment}, add_token=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        return {
            'id': return_obj['url'].split('/')[-1],
            'alias': return_obj['alias']
        }

    def update_attachment(self, attachment_id, alias):
        """
        Updates an attachment alias given a Job ID and attachment ID

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment_id       Attachment ID
        ---------------     --------------------------------------------------------------------
        alias               Alias
        ===============     ====================================================================

        :return: success
        """
        url = '{base}/jobs/{jobId}/attachments/{attachmentid}'.format(base=self._url, jobId=self.job_id,
                                                                          attachmentid=attachment_id)
        post_object = {
            'alias': alias
        }
        return_obj = json.loads(self._gis._con.post(url, params=post_object, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def delete_attachment(gis, url):
        return_obj = json.loads(gis._con.delete(url, add_token=False, try_json=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def update_step(self, step_id, assigned_type, assigned_to):
        """
        Update the assignment of the current step in a job based on the current user's Workflow Manager assignment privileges

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        step_id             Active Step ID
        ---------------     --------------------------------------------------------------------
        assigned_type       Type of assignment designated (NOTE: Must be User, Group or Unassigned)
        ---------------     --------------------------------------------------------------------
        assigned_to         User to which the active step is assigned
        ===============     ====================================================================

        :return: success object
        """

        if step_id is None:
            step_id = self.currentSteps[0]['step_id']
        url = '{base}/jobs/{jobId}/{stepId}?token={token}'.format(base=self._url, jobId=self.job_id, stepId=step_id,
                                                                      token=self._gis._con.token)
        post_object = {
            'assignedType': assigned_type,
            'assignedTo': assigned_to
        }
        return_obj = json.loads(
            self._gis._con.post(url, params=post_object, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def set_current_step(self, step_id):
        """
        Sets a single step to be the active step on the job. The ability to set a step as current is controlled by the **workflowSetStepCurrent** privilege.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        step_id             Active Step ID
        ===============     ====================================================================

        :return: success object
        """

        url = '{base}/jobs/{jobId}/action?token={token}'.format(base=self._url, jobId=self.job_id,
                                                                    token=self._gis._con.token)
        post_object = {
            'type': 'SetCurrentStep',
            'stepIds': [
                step_id
            ]
        }
        return_obj = json.loads(
            self._gis._con.post(url, params=post_object, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            self._gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    @property
    def history(self):
        """
        Gets the history of a job given job ID

        :return: success object
        """

        url = '{base}/jobs/{jobId}/history?token={token}'.format(base=self._url, jobId=self.job_id,
                                                                    token=self._gis._con.token)
        return_obj = ast.literal_eval(str(self._gis._con.get(url)).encode('cp850', 'replace').decode('utf-8'))
        if 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    @property
    def location(self):
        """
        Returns the job location for the user-defined job

        :return: Workflow Manager Job Location Object
        """

        return JobLocation.get(self._gis, '{base}/jobs/{job}/location'.format(base=self._url, job=self.job_id),
                                {"token": self._gis._con.token})

    def manage_jobs(gis, url, ids, action):
        post_object = {
            'jobIds': ids,
            'type': action
        }
        return_obj = json.loads(
            gis._con.post(url, params=post_object, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj


class WMRole(object):
    """
    Represents a Workflow Manager Role object with accompanying GET, POST, and DELETE methods
    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """
    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data):
        self.privileges = self.roleName = self.description = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(gis, url, params):
        role_dict = ast.literal_eval(str(gis._con.get(url, params)).encode('cp850', 'replace').decode('utf-8'))
        return WMRole(role_dict)

    def post(self, gis, url):
        post_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None}
        return_obj = json.loads(
            gis._con.post(url, post_dict, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return return_obj


class JobTemplate(object):
    """
    Represents a Workflow Manager Job Template object with accompanying GET, POST, and DELETE methods
    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """
    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url
    
    def __getattr__(self, item):
        possible_fields = ['default_assigned_to', 'last_updated_by', 'diagram_id', 'extended_property_table_definitions', 'description', 'job_template_name', 'job_template_id', 'default_start_date', 'default_priority_name', 'last_updated_date', 'job_start_date_type', 'diagram_name', 'default_job_duration', 'default_due_date', 'state', 'category', 'default_assigned_type', 'default_description']
        gis = object.__getattribute__(self, '_gis')
        url = object.__getattribute__(self, '_url')
        id = object.__getattribute__(self, 'job_template_id')
        full_object = ast.literal_eval(str(gis._con.get(url, {"token": gis._con.token})).encode('cp850', 'replace').decode('utf-8'))
        try:
            setattr(self, _camelCase_to_underscore(item), full_object[item])
            return full_object[item]
        except KeyError:
            if item in possible_fields:
                setattr(self, _camelCase_to_underscore(item), None)
                return None
            else:
                raise KeyError(f'The attribute "{item}" is invalid for Job Templates')

    def get(gis, url, params):
        job_template_dict = ast.literal_eval(str(gis._con.get(url, params)).encode('cp850', 'replace').decode('utf-8'))
        return JobTemplate(job_template_dict, gis, url)

    def put(self, gis, url):
        put_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None}
        return_obj = json.loads(gis._con.put(url, put_dict, add_token=False, post_json=True, try_json=False,
                                  json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def post(self, gis, url):
        post_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None}
        return_obj = json.loads(
            gis._con.post(url, post_dict, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return return_obj['jobTemplateId']

    def delete(gis, url):
        return_obj = json.loads(gis._con.delete(url, add_token=False, try_json=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj


class Group(object):
    """
    Represents a Workflow Manager Group object with accompanying GET, POST, and DELETE methods
    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """
    _camelCase_to_underscore = _camelCase_to_underscore

    def __init__(self, init_data):
        self.roles = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(gis, url, params):
        group_dict = ast.literal_eval(str(gis._con.get(url, params)).encode('cp850', 'replace').decode('utf-8'))
        return Group(group_dict)


class JobDiagram(object):
    """
    Helper class for managing Workflow Manager jobs diagrams in a workflow item. This class is not created
    by users directly. An instance of this class, can be created by calling `diagram.get(id)` with the appropriate ID
    for the diagram. `diagrams` is available as property on `arcgis.WorkflowManager`.

    """
    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url
    
    def __getattr__(self, item):
        possible_fields = ['display_grid', 'diagram_version', 'diagram_name', 'diagram_id', 'description', 'annotations', 'initial_step_id', 'data_sources', 'steps', 'initial_step_name']
        gis = object.__getattribute__(self, '_gis')
        url = object.__getattribute__(self, '_url')
        id = object.__getattribute__(self, 'diagram_id')
        full_object = ast.literal_eval(str(gis._con.get(url, {"token": gis._con.token})).encode('cp850', 'replace').decode('utf-8'))
        try:
            setattr(self, _camelCase_to_underscore(item), full_object[item])
            return full_object[item]
        except KeyError:
            if item in possible_fields:
                setattr(self, _camelCase_to_underscore(item), None)
                return None
            else:
                raise KeyError(f'The attribute "{item}" is invalid for Diagrams')

    def get(gis, url, params):
        job_diagram_dict = ast.literal_eval(str(gis._con.get(url, params)).encode('cp850', 'replace').decode('utf-8'))
        return JobDiagram(job_diagram_dict, gis, url)

    def post(self, gis, url):
        post_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None}
        return_obj = json.loads(
            gis._con.post(url, post_dict, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def update(self, gis, url, delete_draft):
        clean_dict = {_underscore_to_camelcase(k): v for k, v in self.__dict__.items() if v is not None}
        post_object = {
            "deleteDraft": delete_draft,
            "diagram": clean_dict
        }
        return_obj = json.loads(
            gis._con.post(url, post_object, add_token=False, post_json=True, try_json=False, json_encode=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj

    def delete(gis, url):
        return_obj = json.loads(gis._con.delete(url, add_token=False, try_json=False))
        if 'error' in return_obj:
            gis._con._handle_json_error(return_obj['error'], 0)
        elif 'success' in return_obj:
            return return_obj['success']
        return_obj = { _camelCase_to_underscore(k): v for k, v in return_obj.items() if v is not None and not k.startswith('_')}
        return return_obj


class JobLocation(object):
    """
    Represents a Workflow Manager Job Location object with accompanying GET, POST, and DELETE methods
    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """
    _camelCase_to_underscore = _camelCase_to_underscore
    def __init__(self, init_data):
        self.geometry = self.geometryType = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(gis, url, params):
        job_location_dict = ast.literal_eval(str(gis._con.get(url, params)).encode('cp850', 'replace').decode('utf-8'))
        return JobLocation(job_location_dict)
