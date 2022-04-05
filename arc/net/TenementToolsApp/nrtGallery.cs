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
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Graph_32.png",
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
                string toolname = "NRT_Monitor_Areas";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Monitor Areas tool. Did you add the Tenement Tools toolbox?");
                    //ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Could not find NRT Monitor Areas tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Graph Monitoring Area")
            {
                // set up parameter and cube out path
                string outPath = "";
                string outParams = "";

                // get selected feature (remvoe await if async above)
                await QueuedTask.Run(() =>
                {
                    // get active map if exists, else abort
                    var mapView = MapView.Active;
                    if (mapView == null)
                    {
                        return;
                    }

                    // get the currently selected features on map, abort if multi-select or no select
                    var selectedFeatures = mapView.Map.GetSelection();
                    if (selectedFeatures.Count() == 0)
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Please select a monitoring area.");
                        return;
                    }
                    else if (selectedFeatures.Count() > 1)
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Please select only one area at a time.");
                        return;
                    }

                    // get first feature in selection, abort if multi-select
                    var firstFeature = selectedFeatures.First();
                    if (firstFeature.Value.Count() != 1)
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Please select only one area at a time.");
                        return;
                    }

                    // check if feature name is adequate, abort if not
                    if (firstFeature.Key.Name != "monitoring_areas")
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Your selected area must be called monitoring_areas.");
                        return;
                    }

                    // construct expected path to current layers cube data folder
                    var featurePath = firstFeature.Key.GetPath().AbsolutePath;
                    var gdbPath = Path.GetDirectoryName(featurePath);
                    var folderPath = System.IO.Path.ChangeExtension(gdbPath, null);
                    folderPath = System.IO.Directory.GetParent(folderPath).ToString();

                    //folderPath = folderPath + "_" + "cubes";

                    // get attributes from current feature selection
                    var ins = new ArcGIS.Desktop.Editing.Attributes.Inspector();
                    ins.Load(firstFeature.Key, firstFeature.Value);

                    // construct expected cube filepath and filename
                    string cubeFilePath = ins["global_id"] + ".nc";
                    string cubePath = Path.Combine(folderPath, cubeFilePath);

                    // check if file exists, abort if missing
                    if (!File.Exists(cubePath))
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("No cube detected for this area. Run the monitoring area tool first.");
                        return;
                    }

                    // return cube path
                    outPath = cubePath;

                    // build feature parameter string
                    outParams = ins["area_id"].ToString() + ";" +
                                ins["platform"].ToString() + ";" +
                                ins["s_year"].ToString() + ";" +
                                ins["e_year"].ToString() + ";" +
                                ins["index"].ToString() + ";" +
                                ins["persistence"].ToString() + ";" +
                                ins["rule_1_min_conseqs"].ToString() + ";" +
                                ins["rule_1_inc_plateaus"].ToString() + ";" +
                                ins["rule_2_min_stdv"].ToString() + ";" +
                                ins["rule_2_bidirectional"].ToString() + ";" +
                                ins["rule_3_num_zones"].ToString() + ";" +
                                ins["ruleset"].ToString() + ";" +
                                ins["alert"].ToString() + ";" +
                                ins["alert_direction"].ToString() + ";" +
                                ins["email"].ToString() + ";" +
                                ins["ignore"].ToString() + ";";

                    return;
                });

                // check if path exists, if not abort
                if (outPath == "")
                {
                    return;
                }

                // set up toolbox geoprocessor inputs
                string toolname = "NRT_Build_Graphs";
                var inputs = Geoprocessing.MakeValueArray(outPath, outParams);

                // run geoprocessor
                var result = await Geoprocessing.ExecuteToolAsync(toolname, inputs);

                //create popup list
                var popups = new List<PopupContent>();

                // iter through result messages, append to list
                int i = 0;
                foreach (var msg in result.Messages)
                {
                    if (msg.Type == 0)
                    {
                        if (i == 0)
                        {
                            popups.Add(new PopupContent(msg.Text, "Overview"));
                        }
                        else if (i == 1) {
                            popups.Add(new PopupContent(msg.Text, "Raw Vegetation"));
                        }
                        else if (i == 2)
                        {
                            popups.Add(new PopupContent(msg.Text, "Change Deviation"));
                        }
                        else if (i == 3)
                        {
                            popups.Add(new PopupContent(msg.Text, "Alert History"));
                        }
                        else if (i == 4)
                        {
                            popups.Add(new PopupContent(msg.Text, "Zone Legend"));
                        }

                        // increment counter 
                        i++;
                    }
                }

                // show me!
                MapView.Active.ShowCustomPopup(popups, null, false);
            }
        }
    }
}
