﻿using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Catalog;
using ArcGIS.Desktop.Core;
using ArcGIS.Desktop.Core.Geoprocessing;
using ArcGIS.Desktop.Editing;
using ArcGIS.Desktop.Extensions;
using ArcGIS.Desktop.Framework;
using ArcGIS.Desktop.Framework.Contracts;
using ArcGIS.Desktop.Framework.Dialogs;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Layouts;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace TenementToolsApp
{
    internal class cogGallery : Gallery
    {
        private bool _isInitialized;

        protected override void OnDropDownOpened()
        {
            Initialize();
        }

        private void Initialize()
        {
            // check if already initialised
            if (_isInitialized)
                return;

            // add gallery item for fetch data
            Add(new GalleryItem(text: "Fetch Satellite Data",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/COG_Fetch_32.png",
                                tooltip: "Fetch satellite data from Digital Earth Australia (DEA).",
                                group: "Fetching Tools"));

            // add gallery item for visualise data
            Add(new GalleryItem(text: "Explore Satellite Data",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/COG_Explore_32.png",
                                tooltip: "Visualise and graph satellite data.",
                                group: "Visualisation Tools"));

            // initialise
            _isInitialized = true;
        }

        protected override void OnClick(GalleryItem item)
        {
            // ensure users can re-click on already selected items
            base.AlwaysFireOnClick = true;

            // obtain clicked gallery item
            base.OnClick(item);

            // get name of clicked gallery item
            var gallery_item = base.SelectedItem.ToString();

            // open tool pane
            if (gallery_item == "Fetch Satellite Data")
            {
                try
                {
                    string toolname = "COG_Fetch_ODC";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find COG Fetch (ODC) tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Explore Satellite Data")
            {                
                try
                {
                    string toolname = "COG_Explore";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find COG Explore tool. Did you add the toolbox?");
                };
            };
        }
    }
}
