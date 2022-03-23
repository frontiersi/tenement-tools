using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
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

            // add gallery item for nrt graph
            Add(new GalleryItem(text: "Graph Monitoring Area",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Monitor_32.png",
                                tooltip: "Graph the vegetation and change history for a selected monitoring area.",
                                group: "Graph monitoring area"));

            //// initialise
            _isInitialized = true;
        }

        protected override async void OnClick(GalleryItem item)  // added async for awaiting on executetoolasync, remove if breaks things
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
                string toolname = "Nicher_Masker";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find Nicher Masker tool. Did you add the Tenement Tools toolbox?");
                };

                // temp: warn not yet implemented
                //ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("The NRT Graph tool has not yet been implemented.");
            }
            else if (gallery_item == "Graph Monitoring Area")
            {
                // set toolname and create empty input array
                string toolname = "NRT_Fetch_Dates";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                //try
                //{
                var result = await Geoprocessing.ExecuteToolAsync(toolname, inputs);  // set this await if want to get values before moving on and set async above
                //Geoprocessing.ShowMessageBox(result.Messages, "GP Messages", result.IsFailed ? GPMessageBoxStyle.Error : GPMessageBoxStyle.Default);

                int index = 0;
                foreach (var msg in result.Messages)
                {
                    if (msg.Type == 0)
                    {
                        string dts = msg.Text;
                        List<string> dates = dts.Split(',').ToList();
                        //string[] dates;


                        // x is the element and index the current index
                        System.Diagnostics.Debug.WriteLine("Val:{0} Index:{1}", msg.Text, index);
                        index++;
                    }
                }

                //}
                //catch (Exception)
                //{
                //Debug.WriteLine("Could not find NRT Create Monitoring Area tool. Did you add the Tenement Tools toolbox?");
                //};

                // temp: warn not yet implemented
                //ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Oh.");


                // get selected feature and flash (temp)
                QueuedTask.Run(() =>
                {
                    // get active map 
                    var mapView = MapView.Active;
                    if (mapView == null)
                    {
                        return;
                    }

                    // get selected feature and filter to selected feature row
                    var selectedFeatures = mapView.Map.GetSelection()
                    .Where(kvp => kvp.Key is BasicFeatureLayer)
                    .ToDictionary(kvp => (BasicFeatureLayer)kvp.Key, kvp => kvp.Value);

                    // flash the collection of features.
                    mapView.FlashFeature(selectedFeatures);
                });

                // check if num selected == 1, else return

                // get area_id from current selection

                // call geoprocessor to get datetimes and values from associated cube

                // prepare html via stringwritter for javascripts like so: https://github.com/Esri/arcgis-pro-sdk/wiki/ProGuide-Custom-Pop-ups

                // create html via text (or via template.html) and use google api to graph

                // display html in popupcontent

                // html time!
                // https://github.com/Esri/arcgis-pro-sdk/wiki/ProGuide-Custom-Pop-ups
                var htmlPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "template.html");

                // custom popup
                var popups = new List<PopupContent>();
                popups.Add(new PopupContent("<b>This text is bold.</b>", "Yeah yeah!"));
                //popups.Add(new PopupContent(new Uri(@"C:\Users\Lewis\Desktop\operations-tab2-weather-operations.png"), "URI")); // works!!!

                // define popup style
                var popupDef = new PopupDefinition()
                {
                    Size = new System.Windows.Size(800, 500)
                };

                // show popup
                MapView.Active.ShowCustomPopup(popups, null, false, popupDef);
            }
        }
    }
}
