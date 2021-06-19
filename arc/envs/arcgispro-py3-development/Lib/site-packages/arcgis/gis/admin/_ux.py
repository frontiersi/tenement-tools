import os
import json
from ._resources import PortalResourceManager

###########################################################################
class UX(object):
    """Helper class for modifying the portal home page. This class is not created by users directly. An instance of
    the class, called 'ux', is available as a property of the GIS object. Users call methods on this 'ux' object
    to set banner, background, logo, name etc."""
    #----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
    #----------------------------------------------------------------------
    @property
    def summary(self):
        """
        Allows the get/setting of a brief summary to describe your organization on the sign in page
        associated with its custom apps and sites. This summary has a maximum of 310 characters.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        text              Required string. The brief description of the organization.
        ================  ===============================================================

        :return: string
        """
        portal_resources = PortalResourceManager(self._gis)
        res = json.loads(open(portal_resources.get('localizedOrgProperties'), 'r').read())
        return res['default']['description']
    #----------------------------------------------------------------------

    @summary.setter
    def summary(self, text):
        """
        Allows the get/setting of a brief summary to describe your organization on the sign in page
        associated with its custom apps and sites. This summary has a maximum of 310 characters.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        text              Required string. The brief description of the organization.
        ================  ===============================================================

        :return: string
        """
        if text == "":
            text = None
        params = {"key": "localizedOrgProperties",
                  "text": {"default":{"name": self.name,
                                      "description":text}},
                  "f": "json"
        }
        portal_resources = PortalResourceManager(self._gis)
        portal_resources.add(key='localizedOrgProperties', text=json.dumps(params['text']))
    #----------------------------------------------------------------------
    def set_banner(self, banner_file=None, is_built_in=False, custom_html = None):
        """
        Configure your home page by setting the organization's banner. You can choose one of the 5 built-in banners or
        upload your own. For best results the dimensions of the banner image should be 960 x 180 pixels. You can also
        specify a custom html for how the banner space should appear. For more information, refer to
        http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        banner_file       optional string. If uploading a custom banner, then path to the
                           banner file. If using a built-in banner, valid values are
                           banner-1, banner-2, banner-3, banner-4, banner-5. If None, existing
                           banner is remove.
        ----------------  ---------------------------------------------------------------
        is_built_in       optional bool, default=False. Specify True if using a built-in
                            banner file.
        ----------------  ---------------------------------------------------------------
        custom_html       optional string. Specify exactly how the banner should appear in
                            html. For help on this, refer to
                            http://server.arcgis.com/en/portal/latest/administer/windows/supported-html.htm
        ================  ===============================================================

        :return: True | False
        """
        #region check if banner has to be removed
        if not banner_file and not custom_html:
            #remove code
            portal_resources = PortalResourceManager(self._gis)
            #find existing banner resource file
            resource_list = portal_resources.list()
            e_banner = [banner for banner in resource_list if banner['key'].startswith('banner')]

            #loop through and remove existing banner resource file
            for banner in e_banner:
                try:
                    portal_resources.delete(banner['key'])
                except:
                    continue

            #reset the home page - recurse
            return self.set_banner('banner-2',True)
        #endregion

        #region: Set banner using banner file - built-in or new image
        if banner_file:
            rotator_panel = []
            if not is_built_in: #adding a new image file
                # find image extension
                from pathlib import Path
                fpath = Path(banner_file)
                f_splits = fpath.name.split('.')
                if len(f_splits) > 1 and f_splits[1] == 'png':
                    key_val = 'banner.png'
                elif len(f_splits) > 1 and f_splits[1] == 'jpg':
                    key_val = 'banner.jpg'
                else:
                    raise RuntimeError('Invalid image extension')

                portal_resources = PortalResourceManager(self._gis)
                add_result = portal_resources.add(key_val, banner_file)

                if add_result and custom_html:
                    rotator_panel = [{"id": "banner-custom",
                                      "innerHTML": custom_html}]

                elif add_result and not custom_html:
                    # set rotator_panel_text
                    rotator_panel = [{"id": "banner-custom",
                                      "innerHTML": "<img src='{}/portals/self/resources/{}?token=SECURITY_TOKEN' "
                                                   "style='-webkit-border-radius:0 0 10px 10px; -moz-border-radius:0 0 10px 10px;"
                                                   " -o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px; margin-top:0; "
                                                   "width:960px;'/>".format(
                                          self._portal.con.baseurl, key_val)}]
            else:  # using built-in image
                if not custom_html:  # if no custom html is specified for built-in image
                    rotator_panel = [{"id": banner_file,
                                      "innerHTML": "<img src='images/{}.jpg' "
                                                   "style='-webkit-border-radius:0 0 10px 10px; -moz-border-radius:0 0 10px 10px; "
                                                   "-o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px; margin-top:0; "
                                                   "width:960px; height:180px;'/><div style='position:absolute; bottom:80px; "
                                                   "left:80px; max-height:65px; width:660px; margin:0;'>"
                                                   "<img src='{}/portals/self/resources/thumbnail.png?token=SECURITY_TOKEN' "
                                                   "class='esriFloatLeading esriTrailingMargin025' style='margin-bottom:0; "
                                                   "max-height:100px;'/><span style='position:absolute; bottom:0; margin-bottom:0; "
                                                   "line-height:normal; font-family:HelveticaNeue,Verdana; font-weight:600; "
                                                   "font-size:32px; color:#369;'>{}</span></div>".format(banner_file,
                                                                                                         self._portal.con.baseurl,
                                                                                                         self._gis.properties.name)}]
                else:  # using custom html for built-in image
                    rotator_panel = [{"id": banner_file,
                                      "innerHTML": custom_html}]
        #endregion

        #region: Set banner just using a html text
        elif custom_html:
            rotator_panel = [{"id": "banner-html",
                              "innerHTML": custom_html}]
        #endregion

        # Update the portal self with these banner values
        update_result = self._gis.update_properties({"rotatorPanels": rotator_panel})
        return update_result
    #----------------------------------------------------------------------
    def set_logo(self, logo_file=None):
        """
        Configure your home page by setting the organization's logo image. For best results the logo file should be
        65 x 65 pixels in dimension.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-general.htm

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        logo_file         optional string. Specify path to image file. If None, existing thumbnail is removed.
        ================  ===============================================================

        :return: True | False
        """

        # Add resource file
        portal_resources = PortalResourceManager(self._gis)
        key_val = ""
        # find image extension
        if logo_file is not None and \
           os.path.isfile(logo_file):
            from pathlib import Path
            fpath = Path(logo_file)
            f_splits = fpath.name.split('.')
            if len(f_splits) > 1 and f_splits[1] == 'png':
                key_val = 'thumbnail.png'
            elif len(f_splits) > 1 and f_splits[1] == 'jpg':
                key_val = 'thumbnail.jpg'
            elif len(f_splits) > 1 and f_splits[1] == 'gif':
                key_val = 'thumbnail.gif'

            add_result = portal_resources.add(key_val, logo_file)
        elif logo_file is None:
            if 'thumbnail' in dict(self._gis.properties):
                resource = self._gis.properties['thumbnail']
                if resource and len(resource) > 0:
                    portal_resources.delete(resource)
                key_val = ''
        else:
            for ext in ['.png', '.jpg', '.gif']:
                try:
                    portal_resources.delete('thumbnail' + ext)
                except:
                    continue
            key_val = None

        # Update the portal self with these banner values
        if logo_file is not None:
            update_result = self._gis.update_properties({"thumbnail": key_val})
        else:
            rp = self._gis.properties['rotatorPanels']
            for idx, r in enumerate(rp):
                if r['id'].lower() == 'banner-2':
                    r['innerHTML'] = "<img src='images/banner-2.jpg' style='-webkit-border-radius:0 0 10px 10px;" + \
                        " -moz-border-radius:0 0 10px 10px; -o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px;" +\
                        " margin-top:0; width:960px; height:180px;'/><div style='position:absolute; bottom:80px; left:80px;"+\
                        " max-height:65px; width:660px; margin:0;'><span style='position:absolute; bottom:0; " \
                        "margin-bottom:0; line-height:normal; "+\
                        "font-family:HelveticaNeue,Verdana; font-weight:600; font-size:32px; " \
                        "color:#369;'>{}</span></div>".format(self._gis.properties.name)

            update_result = self._gis.update_properties({"clearEmptyFields": True,
                                                         "thumbnail": "",
                                                         "rotatorPanels": rp})

        return update_result
    #----------------------------------------------------------------------
    def get_logo(self, download_path):
        """
        Get your organization's logo/thumbnail. You can use the `set_logo()` method to set an image as your logo.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the logo file.
        ================  ===============================================================

         :return: Path to downloaded logo file.
        """
        portal_resources = PortalResourceManager(self._gis)
        props = self._gis.properties
        if 'thumbnail' in props:
            resource = props['thumbnail']
            if resource is not None and \
               len(str(resource)) > 0:
                output = portal_resources.get(resource_name=resource,
                                              download_path=download_path)
                return output
        return None
    #----------------------------------------------------------------------
    @property
    def name(self):
        """
        Returns the site's name.  The name can get defined using the 'set_name()'.

         :return: string of the name of the site
        """
        return self._gis.properties['name']
    #----------------------------------------------------------------------
    @name.setter
    def name(self, name):
        """
        Allows for the setting of a site's name.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        name              required string. Name of the site.
        ================  ===============================================================


         :return: boolean
        """
        import json
        if self._gis.properties.name != name:
            rps = [dict(r) for r in self._gis.properties.rotatorPanels]
            for r in  rps:
                r['innerHTML'] = r['innerHTML'].replace(self._gis.properties.name, name)

            res = self._gis.update_properties({"name": name,
                                               "rotatorPanels" : json.dumps(rps)})
            params = {
                "key": "localizedOrgProperties",
                "text": json.dumps({"default":{"name":name,"description":None}}),
                "f" : "json",
                "token" : self._gis._con.token
            }
            url = f"{self._gis._portal.resturl}portals/self/addResource"
            res = self._gis._con.post(url, params)
            return res
    #----------------------------------------------------------------------
    @property
    def description(self):
        """
        Returns the site's description.

         :return: dictionary
        """
        return self._gis.properties['description']
    #----------------------------------------------------------------------
    @property
    def description_visibility(self):
        """
        Returns the site's description visibility

         :return: boolean
        """
        return self._gis.properties['showHomePageDescription']
    #----------------------------------------------------------------------
    @description_visibility.setter
    def description_visibility(self, visiblity):
        """
        Allows for the setting of a site's description.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        visiblity         optional boolean. If True, the desciptive text will show on the
                          home page. If False, the descriptive text will not be displayed
        ================  ===============================================================

         :return: boolean
        """
        return self._gis.update_properties({'showHomePageDescription' : visiblity})
    #----------------------------------------------------------------------
    @description.setter
    def description(self, description=None):
        """
        Allows for the setting of a site's description.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        description       optional string. descriptive text of the site. If None, the
                          value is reset to default.
        ================  ===============================================================

         :return: boolean
        """
        if description is None:
            description = "<br/>"
        return self._gis.update_properties({'description': description})
    #----------------------------------------------------------------------
    @property
    def featured_content(self):
        """
        Returns the featured content group information.  The information
        can then be set using the 'set_featured_content()'.

          :return: dictionary

        :Usage Example:

        >>> data = ux.get_featured_content()
        >>> ux.set_featured_content(data)
        True

        """
        return {'group' : self._gis.properties['homePageFeaturedContent'],
                'count' : self._gis.properties['homePageFeaturedContentCount']}
    #----------------------------------------------------------------------
    @featured_content.setter
    def featured_content(self, content):
        """
        Sets the featured content group for the homepage.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        content           optional dictionary, defines the group and count of the feature
                          content area on an organizational site.  A value of None will
                          reset the value back to the install defaults.
                          Example:
                          {'group': <group id>, 'count' : 12}
        ================  ===============================================================

         :return: boolean
        """
        from .. import Group
        if content is None:
            content = {'homePageFeaturedContent': "",
                        'homePageFeaturedContentCount': 12,
                       'clearEmptyFields':True}
        elif 'group' in content and \
           isinstance(content['group'], Group):
            content['homePageFeaturedContent'] = content['group'].groupid
        elif isinstance(content, dict) and \
             'group' in content and \
             isinstance(content['group'], str):
            c = {}
            c['homePageFeaturedContent'] = content['group']
            if 'count' in content:
                c['homePageFeaturedContentCount'] = content['count']
            else:
                c['homePageFeaturedContentCount'] = 12
            content = c
        elif isinstance(content, str):
            c = {}
            c['homePageFeaturedContent'] = content
            c['homePageFeaturedContentCount'] = 12
            content = c
        self._gis.update_properties(content)
    #----------------------------------------------------------------------
    def set_background(self, background_file=None, is_built_in=True):
        """
        Configure your home page by setting the organization's background image. You can choose no image, a built-in image
        or upload your own. If you upload your own image, the image is positioned at the top and center of the page.
        The image repeats horizontally if it is smaller than the browser or device window. For best results, if you want
        a single, nonrepeating background image, the image should be 1,920 pixels wide (or smaller if your users are on
        smaller screens). The website does not resize the image. You can upload a file up to 1 MB in size.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        background_file   optional string. If using a custom background, specify path to image file.
                            To remove an existing background, specify None for this argument and
                            False for is_built_in argument.
        ----------------  ---------------------------------------------------------------
        is_built_in       optional bool, default=True. The built-in background is set by default.
                            If uploading a custom image, this parameter is ignored.
        ================  ===============================================================

        :return: True | False
        """

        # Add resource if using a custom background file.
        background_update_val = None
        if background_file:
            # find image extension
            from pathlib import Path
            fpath = Path(background_file)
            f_splits = fpath.name.split('.')
            if len(f_splits) > 1 and f_splits[1] == 'png':
                key_val = 'background.png'
            elif len(f_splits) > 1 and f_splits[1] == 'jpg':
                key_val = 'background.jpg'
            else:
                raise RuntimeError('Invalid image extension')

            portal_resources = PortalResourceManager(self._gis)
            add_result = portal_resources.add(key_val, background_file)
            if not add_result:
                raise RuntimeError("Error adding background image as a resource file")
            background_update_val = key_val

        elif is_built_in:  # using built-in
            background_update_val = 'images/arcgis_background.jpg'
        else:
            background_update_val = "none"

        # Update the portal self with these banner values
        update_result = self._gis.update_properties({"backgroundImage": background_update_val})
        return update_result
    #----------------------------------------------------------------------
    def get_banner(self, download_path):
        """
        Get your organization's home page banner image. You can use the `set_banner()` method to set an image or custom HTML
        code as your banner.
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the banner file.
        ================  ===============================================================

         :return: Path to downloaded banner file.
        """
        #create a portal resource manager obj
        portal_resources = PortalResourceManager(self._gis)

        #find existing banner resource file
        resource_list = portal_resources.list()
        e_banner = [banner for banner in resource_list if banner['key'].startswith('banner')]

        #loop through and remove existing banner resource file
        for banner in e_banner:

            try:
                download_path = portal_resources.get(banner['key'], download_path)

            except:
                continue
        return download_path
    #----------------------------------------------------------------------
    def get_background(self, download_path):
        """
        Get your organization's home page background image. You can use the `set_background()` method to set an image
        as the home page background image.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the background file.
        ================  ===============================================================

        :return: Path to downloaded background file.
        """

        #create a portal resource manager obj
        portal_resources = PortalResourceManager(self._gis)

        #find existing banner resource file
        resource_list = portal_resources.list()
        e_background = [banner for banner in resource_list if banner['key'].startswith('background')]

        #loop through and remove existing banner resource file
        for background in e_background:

            try:
                download_path = portal_resources.get(background['key'], download_path)

            except:
                continue
        return download_path
    #----------------------------------------------------------------------
    @property
    def enable_comments(self):
        """
        Turn on item comments
        """
        return self._gis.properties['commentsEnabled']
    #----------------------------------------------------------------------
    @enable_comments.setter
    def enable_comments(self, enable=False):
        """
        Sets the comments property on the items
        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            optional boolean. If True, the comments for the site are turned
                          on.  False will disable comments (default)
        ================  ===============================================================

         :return: boolean

        """
        return self._gis.update_properties({'commentsEnabled' : enable})

    #----------------------------------------------------------------------
    @property
    def default_extent(self):
        """
        returns the site's default extent

          :return: dictionary
        """
        return self._gis.properties['defaultExtent']
    #----------------------------------------------------------------------
    @default_extent.setter
    def default_extent(self, extent):
        """
        defines the site's default extent

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        extent            optional dictionary. The default extent defines where a webmap
                          will open.
                          If a value of None is given, the default extent will be provided.
                          Example Extent (default):
                          {"type":"extent","xmin":-17999999.999994524,"ymin":-11999999.999991827,
                          "xmax":17999999.999994524,"ymax":15999999.999982955,
                          "spatialReference":{"wkid":102100}}
        ================  ===============================================================

          :return: boolean
        """
        if extent is None:
            extent = {"type":"extent","xmin":-17999999.999994524,"ymin":-11999999.999991827,
                      "xmax":17999999.999994524,"ymax":15999999.999982955,
                      "spatialReference":{"wkid":102100}}
        return self._gis.update_properties({'defaultExtent' : extent})
    #----------------------------------------------------------------------
    @property
    def default_basemap(self):
        """
        returns the site's default extent

          :return: dictionary
        """
        return self._gis.properties['defaultBasemap']
    #----------------------------------------------------------------------
    @default_basemap.setter
    def default_basemap(self, basemap):
        """
        The Default Basemap opens when users click New Map. Set the group
        in the Basemap Gallery above and choose the map to open. It will
        open at the default extent you set.

        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           optional string. The new default basemap to set. If None, the
                          default value will be set.
        ================  ===============================================================
        """
        if basemap is None:
            basemap = ""
        return self._gis.update_properties({'defaultBasemap' : basemap})
    #----------------------------------------------------------------------
    @property
    def vector_basemap(self):
        """
        gets/sets the default vector basemap
        """
        return self._gis.properties['defaultVectorBasemap']
    #----------------------------------------------------------------------
    @vector_basemap.setter
    def vector_basemap(self, basemap):
        """
        gets/sets the default vector basemap


        ================  ===============================================================
        **Argument**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           required dictionary. The new default vector basemap to set for
                          a given site.
        ================  ===============================================================
        """
        return self._gis.update_properties({'defaultVectorBasemap' : basemap})