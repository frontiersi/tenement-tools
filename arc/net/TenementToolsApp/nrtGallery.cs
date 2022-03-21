using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using ArcGIS.Core.CIM;
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

namespace TenementToolsApp
{
    internal class nrtGallery : Gallery
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

            // add gallery item for nrt create projects tools
            Add(new GalleryItem(text: "Create New Project",
                                //icon: this.LargeImage != null ? ((ImageSource)this.LargeImage).Clone() : null,
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Create_Project_32.png",
                                tooltip: "Create a new monitoring project.",
                                group: "Manage monitoring projects"));

            // add gallery item for nrt create areas tools
            Add(new GalleryItem(text: "Create New Monitoring Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Create_Area_32.png",
                                tooltip: "Create new monitoring areas.",
                                group: "Manage monitoring areas"));

            // add gallery item for nrt modify areas tools
            Add(new GalleryItem(text: "Modify Existing Monitoring Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Modify_Area_32.png",
                                tooltip: "Modify existing monitoring areas.",
                                group: "Manage monitoring areas"));

            // add gallery item for nrt delete areas tools
            Add(new GalleryItem(text: "Delete Existing Monitoring Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Delete_Area_32.png",
                                tooltip: "Delete existing monitoring areas.",
                                group: "Manage monitoring areas"));

            // add gallery item for nrt monitor
            Add(new GalleryItem(text: "Monitor Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Monitor_32.png",
                                tooltip: "Start monitoring areas for a specific project.",
                                group: "Perform monitoring"));

            //// initialise
            _isInitialized = true;

        }

        protected override void OnClick(GalleryItem item)
        {
            // ensure users can e-click on already selected items
            base.AlwaysFireOnClick = true;

            // obtain clicked gallery item
            base.OnClick(item);

            // get name of clicked gallery item
            var gallery_item = base.SelectedItem.ToString();

            // open tool pane
            if (gallery_item == "Create New Project")
            {
                // set toolname and create empty input array
                string toolname = "NRT_Create_Project";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                Debug.WriteLine("Could not find NRT Create Project tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Create New Monitoring Areas")
            {
                // set toolname and create empty input array
                string toolname = "NRT_Create_Monitoring_Areas";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Create Monitoring Area tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Modify Existing Monitoring Areas")
            {
                // set toolname and create empty input array
                string toolname = "NRT_Modify_Monitoring_Areas";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Modify Monitoring Area tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Delete Existing Monitoring Areas")
            {
                // set toolname and create empty input array
                string toolname = "NRT_Delete_Monitoring_Areas";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Delete Monitoring Area tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Monitor Areas")
            {
                // set toolname and create empty input array
                //string toolname = "Nicher_Masker";
                //var inputs = Geoprocessing.MakeValueArray();
                //inputs = null;

                // open toolpane
                //try
                //{
                //Geoprocessing.OpenToolDialog(toolname, inputs);
                //}
                //catch (Exception)
                //{
                //Debug.WriteLine("Could not find Nicher Masker tool. Did you add the Tenement Tools toolbox?");
                //};

                // temp: warn not yet implemented
                ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("The NRT Monitor tool has not yet been implemented.");
            };
        }
    }
}
