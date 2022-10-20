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
            Add(new GalleryItem(text: "Create Project",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Create_Project_32.png",
                                tooltip: "Create a new monitoring area project.",
                                group: "Management Tools"));

            // add gallery item for nrt create areas tools
            Add(new GalleryItem(text: "Create Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Create_Area_32.png",
                                tooltip: "Create new monitoring areas.",
                                group: "Management Tools"));

            // add gallery item for nrt modify areas tools
            Add(new GalleryItem(text: "Modify Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Modify_Area_32.png",
                                tooltip: "Modify existing monitoring areas.",
                                group: "Management Tools"));

            // add gallery item for nrt delete areas tools
            Add(new GalleryItem(text: "Delete Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Delete_Area_32.png",
                                tooltip: "Delete existing monitoring areas.",
                                group: "Management Tools"));

            // add gallery item for nrt monitor
            Add(new GalleryItem(text: "Monitor Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Monitor_32.png",
                                tooltip: "Perform near-real time monitoring process.",
                                group: "Monitoring Tools"));

            // add gallery item for nrt monitor
            Add(new GalleryItem(text: "Visualise Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Visualise_Area_32.png",
                                tooltip: "Visualse vegetation and change density for an area.",
                                group: "Exploration Tools"));

            // add gallery item for nrt graph
            Add(new GalleryItem(text: "Graph Areas",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/NRT_Graph_32.png",
                                tooltip: "Graph and explore change for a selected area.",
                                group: "Exploration Tools"));

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

            if (gallery_item == "Create Project")
            {
                try
                {
                    string toolname = "NRT_Create_Project";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Create Project tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Create Areas")
            {
                try
                {
                    string toolname = "NRT_Create_Monitoring_Areas";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Create Monitoring Area tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Modify Areas")
            {
                try
                {
                    string toolname = "NRT_Modify_Monitoring_Areas";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Modify Monitoring Area tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Delete Areas")
            {
                try
                {
                    string toolname = "NRT_Delete_Monitoring_Areas";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Delete Monitoring Area tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Monitor Areas")
            {
                try
                {
                    string toolname = "NRT_Monitor_Areas";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Monitor Areas tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Visualise Areas")
            {
                try
                {
                    string toolname = "NRT_Visualise_Areas";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find NRT Visualise Areas tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Graph Areas")
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
                    if (selectedFeatures.Count == 0)
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Please select a monitoring area.");
                        return;
                    }
                    else if (selectedFeatures.Count > 1)
                    {
                        ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Please select only one area at a time.");
                        return;
                    }

                    // get first feature in selection, abort if multi-select
                    var firstFeature = selectedFeatures.ToDictionary().First();
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
                    var featurePath = firstFeature.Key.GetPath().OriginalString;
                    var gdbPath = Path.GetDirectoryName(featurePath);
                    var folderPath = System.IO.Path.ChangeExtension(gdbPath, null);
                    folderPath = System.IO.Directory.GetParent(folderPath).ToString();

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
                                ins["rule_2_min_zone"].ToString() + ";" +
                                ins["rule_3_num_zones"].ToString() + ";" +
                                ins["ruleset"].ToString() + ";" +
                                ins["alert"].ToString() + ";" +
                                ins["method"].ToString() + ";" +
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
                            popups.Add(new PopupContent(msg.Text, "Area Overview"));
                        }
                        else if (i == 1) {
                            popups.Add(new PopupContent(msg.Text, "Vegetation History (Full)"));
                        }
                        else if (i == 2)
                        {
                            popups.Add(new PopupContent(msg.Text, "Vegetation History (Analysis Only)"));
                        }
                        else if (i == 3)
                        {
                            popups.Add(new PopupContent(msg.Text, "Change & Alert History"));
                        }
                        else if (i == 4)
                        {
                            popups.Add(new PopupContent(msg.Text, "Zone & Alert History"));
                        }
                        else if (i == 5)
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
