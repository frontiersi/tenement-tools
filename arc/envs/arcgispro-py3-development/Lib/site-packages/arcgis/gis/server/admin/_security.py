"""
The security resource is a container for all resources and
operations that deal with security for your site. Under this
resource, you will find resources that represent the users and
roles in your current security configuration.
Since the content sent to and from this resource (and operations
within it) could contain confidential data like passwords, it is
recommended that this resource be accessed over HTTPS protocol.
"""
from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
########################################################################
class Security(BaseServer):
    """ 
    This security resource is a container for all resources and
    operations that deal with the security of your site. Under this
    resource, you will find resources that represent the users and
    roles in your current security configuration.
    
    Since the content sent to and from this resource (and operations
    within it) could contain confidential data like passwords, it is
    recommended that this resource be accessed over HTTPS protocol.
    """
    _url = None
    _con = None
    _resources = None
    _json_dict = None
    _json = None
    _um = None
    _rm = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis,
                 initialize=False):
        """Constructor
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at  
                               creation (True). Default is False.
        ==================     ====================================================================
        

        """
        self._url = url
        self._con = gis
        if initialize:
            self._init(gis)
    #----------------------------------------------------------------------
    @property
    def users(self):
        """
        Gets an object to control/manage users.
        """
        if self._um is None:
            self._um = UserManager(url=self._url,
                                   gis=self._con,
                                   initialize=False)
        return self._um
    #----------------------------------------------------------------------
    @property
    def roles(self):
        """
        Gets an object to manage a site's roles.
        """
        if self._rm is None:
            self._rm = RoleManager(url=self._url, gis=self._con)
        return self._rm
    #----------------------------------------------------------------------
    def disable_primary_site_administrator(self):
        """
        Use this operation to disable login privileges for the
        primary site administrator account. This operation can only be
        invoked by an administrator in the system. To re-enable this
        account, use the Enable Primary Site Administrator operation.
        

        .. note::
            - Once disabled, you cannot use the primary site administrator 
            account to log into Manager. Therefore, you should disable 
            this account only if you have other administrators in the system.
            
            - If you are currently logged into the Administrator Directory 
            using the primary site administrator account, you will need to 
            log back in with another administrative account.

        
        :return:
            A boolean indicating success (True) or failure (False).
           
        """
        dURL = self._url + "/psa/disable"
        params = {
            "f" : "json"
        }
        res = self._con.post(path=dURL, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def primary_site_administrator_status(self):
        """
        Gets the disabled status of the primary site administrator account.
        """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/psa"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    def update_primary_site_administrator(self, username, password):
        """
        Updates account properties of the primary site administrator
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. You can optionally provide a new name for the 
                               primary site administrator account.
        ------------------     --------------------------------------------------------------------
        password               Required string. The password for the new primary site 
                               administrator account.
        ==================     ====================================================================
        

        :return:
              A JSON message as dictionary
        """
        params = {
            "f" : "json",
        }
        if username is not None:
            params['username'] = username
        if password is not None:
            params['password'] = password
        u_url = self._url + "/psa/update"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res

########################################################################
class UserManager(BaseServer):
    """
    This resource represents all users available in the user store that can
    administer ArcGIS Server and access the GIS services hosted on the
    server. As the user space could be potentially large, and there is not a listing
    of users, but you can use Get Users or Search operations to access
    their account information.
    
    ArcGIS Server is capable of connecting to your enterprise identity
    stores such as Active Directory or other directory services exposed
    through the LDAP protocol. Such identity stores are treated as read
    only, and ArcGIS Server does not attempt to update them. As a result,
    operations that need to update the identity store (such as adding
    users, removing users, updating users, assigning roles and removing
    assigned roles) are not supported when identity stores are read only.
    On the other hand, you could configure your ArcGIS Server to use the
    default identity store (shipped with the server) which is treated as a
    read-write store.
    
    The total number of users are returned in the response.

    .. note::
        Typically, this resource must be accessed over an HTTPS connection.
    
    """
    _url = None
    _con = None
    _resources = None
    _json_dict = None
    _json = None
    _rm = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis,
                 initialize=False):
        """
        Constructor
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at  
                               creation (True). Default is False.
        ==================     ====================================================================


        """
        self._url = url
        self._con = gis
        if initialize:
            self._init(gis)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    @property
    def me(self):
        """
        Gets the user object as the current logged in user. If the username
        cannot be found, for example, the site administrator account, then
        just the username is returned.
        """
        res = self.search(username=self._con._username,
                          max_results=1)
        if len(res) == 0:
            return self._con._username
        return res[0]
    #----------------------------------------------------------------------
    def create(self,
                 username,
                 password,
                 fullname=None,
                 description=None,
                 email=None):
        """
        Adds a user account to the user store.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the new user. The name must be unique 
                               in the user store.
        ------------------     --------------------------------------------------------------------
        password               Optional string. The password for this user.
        ------------------     --------------------------------------------------------------------
        fullname               Optional string. A full name for this user.
        ------------------     --------------------------------------------------------------------
        description            Optional string. Provide comments or description for this user.
        ------------------     --------------------------------------------------------------------
        email                  Optional string. An email for this user account.
        ==================     ====================================================================
        
        :return:
            A JSON indicating success.
            
        """
        params = {
            "f" : "json",
            "username" : username,
            "password" : password,
        }
        if fullname is not None:
            params['fullname'] = fullname
        if description is not None:
            params['description'] = description
        if email is not None:
            params['email'] = email
        a_url = self._url + "/users/add"
        res = self._con.post(path=a_url, postdata=params)
        if 'status' in res:
            if res['status'] == 'success':
                return self.get(username=username)
            else:
                return res['status']
        return res
    #----------------------------------------------------------------------
    def _get_user_privileges(self, username):
        """
        Returns the privilege associated with a user.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The user name of the user.
        ==================     ====================================================================
        
        :return:
            A JSON message as dictionary indicating the privilege level.
        
        """
        params = {
            "f" : "json",
            "username" : username
        }
        url = self._url + "/users/getPrivilege"
        return self._con.post(path=url, postdata=params)
    #----------------------------------------------------------------------
    def _get_user_roles(self, username, user_filter=None, max_count=None):
        """
        This operation returns a list of role names that have been
        assigned to a particular user account.
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to get roles for. 
        ------------------     --------------------------------------------------------------------
        user_filter            Optional string. The filter to be applied to the resultant role 
                               set. The default is None, nothing will be filtered.
        ------------------     --------------------------------------------------------------------
        max_count              Optional integer. The maximum number of results to return for this 
                               query. The default is None, all results will be returned.
        ==================     ====================================================================
        

        :return:
           A JSON dictionary containing the list of roles found associated with the user and a 
           boolean indicating if there are more roles (if True, adjust the two arguments 
           accordingly to view the rest).
        """
        u_url = self._url + "/roles/getRolesForUser"
        params = {
            "f" : "json",
            "username" : username
        }
        if user_filter:
            params['filter'] = user_filter

        if max_count:
            params['maxCount'] = max_count
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _list_users(self, start_index=0, page_size=10):
        """
        This operation gives you a pageable view of users in the user
        store. It is intended for iterating over all available user
        accounts. To search for specific user accounts instead, use the
        Search Users operation.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        start_index            Optional integer. The starting index (zero-based) from the users 
                               list to be returned in the result page. The default is 0.
        ------------------     --------------------------------------------------------------------
        page_size              Optional integer. The maximum number of users to return in the 
                               result page. The default size is 10.
        ==================     ====================================================================
        

        :return:
            A JSON dictionary containing the list of users found and a boolean indicating if there 
            are more users (if True, adjust the two arguments accordingly to view the rest).
            
        """
        u_url = self._url + "/users/getUsers"
        params = {
            "f" : "json",
            "startIndex" : start_index,
            "pageSize" : page_size
        }
        return self._con.post(path=u_url,
                              postdata=params)
    #----------------------------------------------------------------------
    def _remove_roles_from_user(self, username, roles):
        """
        This operation removes roles that have been previously assigned
        to a user account. This operation is supported only when the
        user and role store supports reads and writes.
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to remove roles from. 
        ------------------     --------------------------------------------------------------------
        roles                  Required string. A comma-seperated list of the role names to remove 
                               from the user.
        ==================     ====================================================================
        
        :return:
            A JSON Messages indicating success.
            
        """
        u_url = self._url + "/users/removeRoles"
        params = {
            "f" : "json",
            "username" : username,
            "roles" : roles
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _delete_user(self, username):
        """
        Removes an existing user account from the user store. This operation is 
        available only when the user store is a read-write store, such as the default 
        ArcGIS Server store. 
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to remove. 
        ==================     ====================================================================
        

        :return:
            A JSON Messages indicating success.
            
        """
        params = {
            "f" : 'json',
            "username" : username
        }
        u_url = self._url + "/users/remove"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _remove_users_from_role(self, rolename, users):
        """
           Removes a role assignment from multiple users.
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role to remove from multiple users. 
        ------------------     --------------------------------------------------------------------
        users                  Required string. A comma-seperated list usernames to remove the 
                               role from.
        ==================     ====================================================================
        

        :return:
            A JSON Messages indicating success.
            
        """
        params = {
            "f" : 'json',
            "rolename" : rolename,
            "users" : users
        }
        u_url = self._url + "/roles/removeUsersFromRole"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def search(self, username, max_results=25):
        """
        You can use this operation to search a specific user or a group of
        users from the user store. The size of the search result can be
        controlled with the max_results parameter.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The user or users to find. 
        ------------------     --------------------------------------------------------------------
        max_results            Optional integer. The maximum number of users to return for this 
                               query. The default is 25.
        ==================     ====================================================================
        

        :return:
            A list of users found.
            
        """
        users = []
        res = self._find_users(criteria=username, max_count=max_results)
        if "users" in res:
            for user in res["users"]:
                users.append(User(self, user))
                del user
        return users
    #----------------------------------------------------------------------
    def get(self, username):
        """
        Finds a specific user.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The user to find. 
        ==================     ====================================================================
        

        :return:
            The user object.
        """
        res = self.search(username=username, max_results=1)
        if len(res) == 0:
            return None
        return res[0]
    @property
    def roles(self):
        """Helper object to manage custom roles for users"""
        if self._rm is None:
            self._rm = RoleManager(self._url,
                                   gis=self._con)
        return self._rm
    #----------------------------------------------------------------------
    def _find_users(self, criteria=None, max_count=10):
        """
        You can use this operation to search a specific user or a group
        of users from the user store. The size of the search result can
        be controlled with the max_count parameter.
           
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        criteria               Optional string. The filter to be applied to search for the users. 
                               The default is None, nothing will be filtered.
        ------------------     --------------------------------------------------------------------
        max_count              Optional integer. The maximum number of results to return for this 
                               query. The default is 10.
        ==================     ====================================================================
        

        :return:
            A JSON dictionary containing the list of users found and a boolean indicating if there 
            are more users (if True, adjust the two arguments accordingly to view the rest).
            
        """
        params = {
            "f" : "json",
            "filter" : criteria,
            "maxCount" : max_count
        }
        u_url = self._url + "/users/search"
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _update_user(self, username, password=None,
                    fullname=None, description=None,
                    email=None):
        """
        Updates a user account in the user store.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to update.
        ------------------     --------------------------------------------------------------------
        password               Optional string. The password for this user.
        ------------------     --------------------------------------------------------------------
        fullname               Optional string. A full name for this user.
        ------------------     --------------------------------------------------------------------
        description            Optional string. Provide comments or description for this user.
        ------------------     --------------------------------------------------------------------
        email                  Optional string. An email for this user account.
        ==================     ====================================================================
        
        :return:
            A JSON indicating success.
            
        """
        user = {"username" : username}
        params = {
            "f" : "json",
            "user" : {}
        }
        if password is not None:
            user['password'] = password
        if fullname is not None:
            user['fullname'] = fullname
        if description is not None:
            user['description'] = description
        if email is not None:
            user['email'] = email
        params['user'] = user
        u_url = self._url + "/users/update"
        return self._con.post(path=u_url, postdata=params)
########################################################################
class User(dict):
    """
    A resource representing a user in the user store that can administer ArcGIS Server.
    """
    _security = None
    _user_dict = None
    #----------------------------------------------------------------------
    def __init__(self, usermanager, user_dict):
        """Constructor"""
        dict.__init__(self)
        if usermanager is None or \
           user_dict is None:
            raise ValueError("Values of UserManager and user_dict" + \
                             " must be provided")
        self._security = usermanager
        self._user_dict = user_dict
        self.__dict__.update(self._user_dict)
    #----------------------------------------------------------------------
    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
    #----------------------------------------------------------------------
    def __getitem__(self, k): # support user attributes as dictionary keys on this object, eg. user['role']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            return self.__dict__[k]
    #----------------------------------------------------------------------
    def __str__(self):
        state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        return '\n'.join(state)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s username:%s>' % (type(self).__name__, self.username)
    #----------------------------------------------------------------------
    def _repr_html_(self):
        fullName = 'Not Provided'
        email = 'Not Provided'
        description = 'Not Provided'
        role = 'Not Provided'
        try:
            fullName = self.fullname
        except:
            fullName = 'Not Provided'
        try:
            description = self.description
        except:
            description = 'Not Provided'
        try:
            email = self.email
        except:
            email = 'Not Provided'
        import datetime
        return """<div class="9item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_right" style="float: none; width: auto; overflow: hidden;">
                    <br/><b>Username</b>: """ + str(self.username) + """
                        <br/><b>Full Name</b>: """ + str(fullName) + """
                        <br/><b>Description</b>: """ + str(description)  + """
                        <br/><b>Email</b>: """ + str(email)  + """
                        <br/><b>Disabled</b>: """ + str(self.disabled)  + """
                        <br/><b>Current As:</b>: """ + str(datetime.datetime.now().strftime("%B %d, %Y")) + """

                    </div>
                </div>
                """
    #----------------------------------------------------------------------
    def update(self, password=None, full_name=None,
               description=None, email=None):
        """
        Updates this user account in the user store.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        password               Optional string. The password for this user.
        ------------------     --------------------------------------------------------------------
        fullname               Optional string. A full name for this user.
        ------------------     --------------------------------------------------------------------
        description            Optional string. Provide comments or description for this user.
        ------------------     --------------------------------------------------------------------
        email                  Optional string. An email for this user account.
        ==================     ====================================================================
        

        :return:
            A JSON indicating success.
            
        """

        res = self._security._update_user(self.username, password,
                                          full_name, description,
                                          email)
        if res['status'] == 'success':
            user = self._security._find_users(criteria=self.username, max_count=1)['users'][0]
            self._user_dict = user
            self.__dict__.update(user)
            return True
        return res
    #----------------------------------------------------------------------
    def add_role(self, role_name):
        """
        Use this operation to assign roles to a user account when
        working with a user and role store that supports reads and writes.
        By assigning a role to a user, the user account automatically
        inherits all the role's permissions.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        role_name              Required string. A role name to assign to this user.
        ==================     ====================================================================

        :return:
            A JSON indicating success.

        """

        return self._security.roles._assign_roles(username=self.username,
                                                  roles=role_name)
    #----------------------------------------------------------------------
    def delete(self):
        """
        Deletes this user account.
        
        :return:
            A JSON indicating success.
        """
        username = self.username
        return self._security._delete_user(username=username)
########################################################################
class RoleManager(BaseServer):
    """
    This resource represents all roles available in the role store. The
    ArcGIS Server security model supports a role-based access control in
    which each role can be assigned certain permissions (privileges) to
    access one or more resources. Users are assigned to these roles. The
    server then authorizes each requesting user based on all the roles
    assigned to the user.

    ArcGIS Server is capable of connecting to your enterprise identity
    stores such as Active Directory or other directory services exposed via
    the LDAP protocol. Such identity stores are treated as read-only stores
    and ArcGIS Server does not attempt to update them. As a result,
    operations that need to update the role store (such as adding roles,
    removing roles, updating roles) are not supported when the role store
    is read-only. On the other hand, you can configure your ArcGIS Server
    to use the default role store shipped with the server, which is
    treated as a read-write store.
    """
    _url = None
    _con = None
    _resources = None
    _json_dict = None
    _json = None
    #----------------------------------------------------------------------
    def __init__(self, url, gis,
                 initialize=False):
        """
        Constructor
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at  
                               creation (True). Default is False.
        ==================     ====================================================================
        

        """
        self._url = url
        self._con = gis
        if initialize:
            self._init(gis)
    #----------------------------------------------------------------------
    def __str__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s at %s>' % (type(self).__name__, self._url)
    #----------------------------------------------------------------------
    def create(self, name, description=""):
        """ 
        Adds a role to the role store. This operation is available only
        when the role store is a read-write store such as the default
        ArcGIS Server store.
        
        If the name of the role exists in the role store, an error will
        be returned.
        
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the new role. The name must be unique 
                               in the role store.
        ------------------     --------------------------------------------------------------------
        description            Optional string. Provide comments or description for the role.
        ==================     ====================================================================
        

        :return:
            A JSON message as dictionary
        """
        params = {
            "f" : "json",
            "rolename" : name,
            "description" : description
        }
        a_url = self._url + "/roles/add"
        res = self._con.post(path=a_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _add_users_to_role(self, rolename, users):
        """
        Assigns a role to multiple users. 
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role. 
        ------------------     --------------------------------------------------------------------
        users                  Required string. A comma-separated list of user names. Each user 
                               name must exist in the user store.
        ==================     ====================================================================
    
    
        :retrun:
            A JSON message indicating success (True).
    
        """
        params = {
            "f" : "json",
            "rolename" : rolename,
            "users" : users
        }
        rURL = self._url + "/roles/addUsersToRole"
        res = self._con.post(path=rURL, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _assign_privilege(self, rolename, privilege="ACCESS"):
        """
        Assigns a privilege to the desired role.
    
        Administrative access to ArcGIS Server is modeled as three broad
        tiers of privileges:
    
          - ADMINISTER - A role that possesses this privilege has unrestricted administrative access to ArcGIS Server.
          - PUBLISH - A role with PUBLISH privilege can only publish GIS services to ArcGIS Server.
          - ACCESS - No administrative access. A role with this privilege can only be granted permission to access one or more GIS services.
    
        By assigning these privileges to one or more roles in the role
        store, ArcGIS Server's security model supports role-based access
        control to its administrative functionality.
    
        These privilege assignments are stored independent of ArcGIS
        Server's role store. As a result, you don't need to update your
        enterprise identity stores (like Active Directory).
    
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role. 
        ------------------     --------------------------------------------------------------------
        privilege              Optional string. The capability to assign to the role. Default 
                               value is ACCESS.  Choices are ADMINISTER, PUBLISH, ACCESS.
        ==================     ====================================================================
    
        :returns:
            A JSON message indicating success (True).
    
    
        """
        a_url = self._url + "/roles/assignPrivilege"
        params = {
            "f" : "json",
            "rolename" : rolename,
            "privilege" : privilege
        }
        res = self._con.post(path=a_url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _assign_roles(self, username, roles):
        """
        Assigns one or more roles to a user account. The role store must aloow read 
        and write edits.
    
        By assigning a role to a user, the user account automatically inherits all the 
        permissions that have been granted to the role.
    
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user. 
        ------------------     --------------------------------------------------------------------
        roles                  Required string. A comma-separated list of role names. Each role 
                               name must exist in the role store.
        ==================     ====================================================================
    
    
        :retrun:
            A JSON message indicating success (True).
    
        """
        params = {
            "f" : "json",
            "username" : username,
            "roles" : roles
        }
        u_url = self._url + "/users/assignRoles"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _get_privilege_for_role(self, rolename):
        """
        Returns the privilege associated with a role. 
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role. 
        ==================     ====================================================================
    
    
        :retrun:
            A JSON message stating the privilege (one of ADMINISTER, PUBLISH, ACCESS).
    
        """
        params = {
            "f" : "json",
            "rolename" : rolename
        }
        pURL = self._url + "/roles/getPrivilege"
        return self._con.post(path=pURL,
                              postdata=params)
    #----------------------------------------------------------------------
    def _get_user_privileges(self, username):
        """
        Returns the privilege associated with a user. This operation tests all the roles to which the 
        user belongs and returns the highest privilege.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user. 
        ==================     ====================================================================
    
    
        :retrun:
            A JSON dictionary stating the privilege (one of ADMINISTER, PUBLISH, ACCESS).
    
        """
        params = {
            "f" : "json",
            "username" : username
        }
        url = self._url + "/users/getPrivilege"
        return self._con.post(path=url, postdata=params)
    #----------------------------------------------------------------------
    def all(self, start_index=0, page_size=10):
        """ 
        This operation gives you a pageable view of roles in the role
        store. It is intended for iterating through all available role
        accounts. To search for specific role accounts instead, use the
        Search Roles operation.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        start_index            Optional integer. The starting index (zero-based) from the roles 
                               list to be returned in the result page. The default is 0.
        ------------------     --------------------------------------------------------------------
        page_size              Optional integer. The maximum number of roles to return in the 
                               result page. The default size is 10.
        ==================     ====================================================================
    
    
        :return:
            A JSON dictionary containing the list of roles found and a boolean on if there are more roles.
    
        """
        u_url = self._url + "/roles/getRoles"
        params = {
            "f" : "json",
            "startIndex" : start_index,
            "pageSize" : page_size
        }
        roles = []
        res = self._con.post(path=u_url, postdata=params)
        if 'roles' in res:
            for role in res['roles']:
                roles.append(Role(rolemanager=self, roledict=role))
        return roles
    #----------------------------------------------------------------------
    def _get_roles_by_privilege(self, privilege):
        """
        Returns all roles that are at the declared privilege level.
    
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        privilege              Required string. The name of the privilege. Choices are ADMINISTER, PUBLISH, ACCESS.
        ==================     ====================================================================
    
        :return:
            A JSON dictionary with a list of roles that match the declared privilege level.
    
        """
        u_url = self._url + "/roles/getRolesByPrivilege"
        params = {
            "f" : "json",
            "privilege" : privilege
        }
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _get_user_roles(self, username, user_filter=None, max_count=None):
        """
        Supplies a list of roles that have been assigned to a particular user account.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to get roles for. 
        ------------------     --------------------------------------------------------------------
        user_filter            Optional string. The filter to be applied to the resultant role 
                               set. The default is None, nothing will be filtered.
        ------------------     --------------------------------------------------------------------
        max_count              Optional integer. The maximum number of results to return for this 
                               query. The default is None, all results will be returned.
        ==================     ====================================================================
    
    
        :return:
            A JSON dictionary with a list of roles found associated with the declared user.
    
        """
        u_url = self._url + "/roles/getRolesForUser"
        params = {
            "f" : "json",
            "username" : username
        }
        if user_filter:
            params['filter'] = user_filter

        if max_count:
            params['maxCount'] = max_count
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _get_users_within_role(self, rolename, user_filter=None, max_count=20):
        """
        You can use this operation to conveniently see all the user
        accounts to whom this role has been assigned.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role. 
        ------------------     --------------------------------------------------------------------
        user_filter            Optional string. The filter to be applied to the resultant user 
                               set. The default is None, nothing will be filtered.
        ------------------     --------------------------------------------------------------------
        max_count              Optional integer. The maximum number of results to return for this 
                               query. The default is None, all results will be returned.
        ==================     ====================================================================
    
    
        :return:
            A JSON dictionary with a list of users found associated with the declared role.
    
        """
        u_url = self._url + "/roles/getUsersWithinRole"
        params = {
            "f" : "json",
            "rolename" : rolename,
            "maxCount" : max_count
        }
        if user_filter and \
           isinstance(user_filter, str):
            params['filter'] = user_filter
        return self._con.post(path=u_url, postdata=params)
    #----------------------------------------------------------------------
    def _delete_role(self, rolename):
        """
        Removes an existing role from the role store. This operation is
        available only when the role store is a read-write store such as
        the default ArcGIS Server store.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role to remove. 
        ==================     ====================================================================
    
    
        :return:
            A JSON message.
    
        """
        params = {
            "f" : "json",
            "rolename" : rolename
        }
        u_url = self._url + "/roles/remove"
        res = self._con.post(path=u_url,
                             postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _remove_roles_from_user(self, username, roles):
        """
        This operation removes roles that have been previously assigned
        to a user account. This operation is supported only when the
        user and role store supports reads and writes.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The name of the user to remove roles from. 
        ------------------     --------------------------------------------------------------------
        roles                  Required string. A comma-seperated list of the role names to remove 
                               from the given user.
        ==================     ====================================================================
    
        :return:
            A JSON message as a dictionary.
    
        """
        u_url = self._url + "/users/removeRoles"
        params = {
            "f" : "json",
            "username" : username,
            "roles" : roles
        }
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    def _remove_users_from_role(self, rolename, users):
        """
        Removes a role assignment from multiple users.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role to remove from the given users. 
        ------------------     --------------------------------------------------------------------
        users                  Required string. A comma-seperated list of the user names to remove 
                               from the given role.  These users must exist.
        ==================     ====================================================================
    
        :return:
            A JSON message as a dictionary.
    
        """
        params = {
            "f" : 'json',
            "rolename" : rolename,
            "users" : users
        }
        u_url = self._url + "/roles/removeUsersFromRole"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
    #----------------------------------------------------------------------
    @property
    def count(self):
        """
        Gets the number of roles for ArcGIS Server.
        """
        params = {
            "f" : "json"
        }
        u_url = self._url + "/roles"
        return self._con.get(path=u_url, params=params)
    #----------------------------------------------------------------------
    def get_role(self, role_id=None, max_count=10):
        """
        Use this operation to search a specific role or a group
        of roles from the role store.
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        role_id                Optional string. A filter string to search for the roles
        ------------------     --------------------------------------------------------------------
        max_count              Optional integer. The maximum number of the result items that can 
                               be retuned.  The default is 10.
        ==================     ====================================================================
    
    
        :return:
            A JSON message as dictionary.
    
        """
        params = {
            "f" : "json",
            "filter" : role_id,
            "maxCount" : max_count
        }
        roles = []
        u_url = self._url + "/roles/search"
        res = self._con.post(path=u_url, postdata=params)
        if 'roles' in res:
            for r in res['roles']:
                roles.append(Role(rolemanager=self, roledict=r))
            return roles
        return roles
    #----------------------------------------------------------------------
    def _update_role(self, rolename, description):
        """
        Updates a role description in the role store
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        rolename               Required string. The name of the role. The name must be unique in 
                               the role store. to remove from the given users. 
        ------------------     --------------------------------------------------------------------
        description            Optional string. Provide comments or description for the role.
        ==================     ====================================================================
    
        :return:
            A JSON message indicating success (True).
        """
        params = {
            "f" : "json",
            "rolename" : rolename
        }
        if description is not None:
            params['description'] = description
        u_url = self._url + "/roles/update"
        res = self._con.post(path=u_url, postdata=params)
        if 'status' in res:
            return res['status'] == 'success'
        return res
########################################################################
class Role(dict):
    """
    Represents a single role on server.

    """
    _roledict = None
    _security = None
    #----------------------------------------------------------------------
    def __init__(self, rolemanager, roledict):
        """Constructor"""
        dict.__init__(self)
        if rolemanager is None or \
           roledict is None:
            raise ValueError("Values of RoleManager and roledict" + \
                             " must be provided")
        self._security = rolemanager
        self._roledict = roledict
        self.__dict__.update(self._roledict)
    #----------------------------------------------------------------------
    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
    #----------------------------------------------------------------------
    def __getitem__(self, k): # support user attributes as dictionary keys on this object, eg. user['role']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            return self.__dict__[k]
    #----------------------------------------------------------------------
    def __str__(self):
        state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        return '\n'.join(state)
    #----------------------------------------------------------------------
    def __repr__(self):
        return '<%s rolename:%s>' % (type(self).__name__, self.rolename)
    #----------------------------------------------------------------------
    def update(self, description=None):
        """
        Updates this role in the role store with new information. This
        operation is available only when the role store is a read-write
        store such as the default ArcGIS Server store.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        description            Optional string. An optional field to add comments or a description for the role.
        ==================     ====================================================================
        

        :return:
            A status dictionary.
        """
        res =  self._security._update_role(rolename=self.rolename,
                                          description=description)
        if res:
            b = self._security.get_role(self.rolename, max_count=1)[0]
            self.__dict__.update(b._roledict)
        return res
    #----------------------------------------------------------------------
    def delete(self):
        """
        Deletes this current role.
        
        :return:
            A boolean indicating success (True) or failure (False).
            
        """
        res = self._security._delete_role(rolename=self.rolename)
        self = None
        return True
    #----------------------------------------------------------------------
    def set_privileges(self, privilage):
        """
        Assigns a privilege to this role.
    
        Administrative access to ArcGIS Server is modeled as three broad
        tiers of privileges:
    
          - ADMINISTER - A role that possesses this privilege has unrestricted administrative access to ArcGIS Server.
          - PUBLISH - A role with PUBLISH privilege can only publish GIS services to ArcGIS Server.
          - ACCESS - No administrative access. A role with this privilege can only be granted permission to access one or more GIS services.
    
        By assigning these privileges to one or more roles in the role
        store, ArcGIS Server's security model supports role-based access
        control to its administrative functionality.
    
        These privilege assignments are stored independent of ArcGIS
        Server's role store. As a result, you don't need to update your
        enterprise identity stores (like Active Directory).
    
    
        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        privilage              Required string. The capability to assign to the role. Choices are 
                               ADMINISTER, PUBLISH, ACCESS
        ==================     ====================================================================
    
        :return:
            A boolean indicating success (True) or failure (False).
    
        """
        allowed = ['administer', 'publish', 'access']
        if privilage.lower() in allowed:
            privilage = privilage.upper()
        else:
            raise ValueError("Invalid privilage.")
        return self._security._assign_privilege(rolename=self.rolename,
                                               privilege=privilage)
    #----------------------------------------------------------------------
    def grant(self, username):
        """
        Adds a user to this role.

        =========  =================================================
        Parmeters  **Description**
        ---------  -------------------------------------------------
        username   Required string. The account name to add to the role.
        =========  =================================================

        :returns:
            A boolean indicating success (True) or failure (False).
            
        """
        return self._security._add_users_to_role(rolename=self.rolename,
                                                 users=username)

