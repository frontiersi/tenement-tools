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
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

namespace TenementToolsApp
{
    internal class gdvspectraGallery : Gallery
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

            // add gallery item for gdv likelihood solo function
            Add(new GalleryItem(text: "GDV Likelihood",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Wizard_32.png",
                                tooltip: "Calculate GDV likelihood from satellite data.", 
                                group: "Detection Tools"));

            // add gallery item for likelihood threshold solo function
            Add(new GalleryItem(text: "Threshold GDV Likelihood", 
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Threshold_32.png",
                                tooltip: "Threshold GDV likelihood into high probability areas.", 
                                group: "Detection Tools"));

            // add gallery item for trend solo function
            Add(new GalleryItem(text: "Trend Analysis",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Trend_32.png",
                                tooltip: "Determine change over time using trend analyses.",
                                group: "Trend Tools"));

            // add gallery item for trend solo function
            Add(new GalleryItem(text: "Change Vector Analysis (CVA)",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_CVA_32.png",
                                tooltip: "Determine change over time compared to a baseline.",
                                group: "Trend Tools"));

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

            // open gdvspectra likelihood tool pane
            if (gallery_item == "GDV Likelihood") {
                try 
                {
                    string toolname = "GDVSpectra_Likelihood";
                    Geoprocessing.OpenToolDialog(toolname, null);
                } 
                catch (Exception) { 
                    Debug.WriteLine("Could not find GDVSpectra Likelihood tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Threshold GDV Likelihood")
            {
                try
                {
                    string toolname = "GDVSpectra_Threshold";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra Threshold tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Trend Analysis")
            {
                try
                {
                    string toolname = "GDVSpectra_Trend";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra Trend Anaysis tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Change Vector Analysis (CVA)")
            {
                try
                {
                    string toolname = "GDVSpectra_CVA";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra CVA tool. Did you add the toolbox?");
                };
            };
        }
    }
}
