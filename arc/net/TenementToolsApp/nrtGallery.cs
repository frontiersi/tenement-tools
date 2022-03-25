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

                string outPath = "";

                // get selected feature and flash (temp)  // remvoe await if async above
                await QueuedTask.Run(() =>
                {
                    // get active map 
                    var mapView = MapView.Active;
                    if (mapView == null)
                    {
                        return;
                    }

                    // get the currently selected features in the map
                    var selectedFeatures = mapView.Map.GetSelection();

                    // if only one feature selected...
                    if (selectedFeatures.Count() == 1)
                    {
                        // get the first layer and its corresponding selected feature OIDs
                        var firstFeature = selectedFeatures.First();

                        if (firstFeature.Value.Count() == 1)
                        {
                            // if layer is called monitoring areas
                            if (firstFeature.Key.Name == "monitoring_areas")
                            {

                                // get path to selected feature feature
                                var featurePath = firstFeature.Key.GetPath().AbsolutePath;
                                var gdbPath = Path.GetDirectoryName(featurePath);
                                var folderPath = System.IO.Path.ChangeExtension(gdbPath, null);
                                folderPath = folderPath + "_" + "cubes";

                                // get selected monitoring area code 
                                var inspector = new ArcGIS.Desktop.Editing.Attributes.Inspector();
                                inspector.Load(firstFeature.Key, firstFeature.Value);
                                var area_id = inspector["area_id"];
                                string cubeFilePath = "cube" + "_" + area_id + "_" + "change.nc";

                                // combine expected cube name with current folder
                                string cubePath = Path.Combine(folderPath, cubeFilePath);

                                // check if file exists
                                if (File.Exists(cubePath)) {
                                    outPath = cubePath;
                                }

                            }
                        }  
                        // todo tell user only one mont area selected at a time
                    }
                    // todo tell user that features from only one layer selected at a time allowed
                });



                // now, send to geoprocessor
                if (outPath != "")
                {
                    // set toolname and create input array 
                    string toolname = "NRT_Fetch_Dates";
                    var inputs = Geoprocessing.MakeValueArray(outPath);

                    // run geoprocessor
                    var result = await Geoprocessing.ExecuteToolAsync(toolname, inputs);
                    
                    // get output message (i.e. html) from result
                    foreach (var msg in result.Messages)
                    {
                        if (msg.Type == 0)
                        {
                            string html = msg.Text;

                            // if not empty... todo!

                            // custom popup
                            var popups = new List<PopupContent>();
                            
                            // add veg signal to popup
                            popups.Add(new PopupContent(html, "Vegetation (Average)"));

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
        }
    }
}
