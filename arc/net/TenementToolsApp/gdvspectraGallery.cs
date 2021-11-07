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

            // add gallery item for wizard
            //Add(new GalleryItem(text: "Run All",
                                //icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Wizard_32.png",
                                //tooltip: "Generate GDV likelihood, threshold and trends.", 
                                //group: "Wizard"));

            // add gallery item for gdv likelihood solo function
            Add(new GalleryItem(text: "GDV Likelihood",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Wizard_32.png",
                                tooltip: "Generate GDV Likelihood from satellite time series.", 
                                group: "Run individual functions"));

            // add gallery item for likelihood threshold solo function
            Add(new GalleryItem(text: "Threshold Likelihood", 
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Threshold_32.png",
                                tooltip: "Threshold GDV Likelihood via occurrence points or standard deviation.", 
                                group: "Run individual functions"));

            // add gallery item for trend solo function
            Add(new GalleryItem(text: "Trend Analysis",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Trend_32.png",
                                tooltip: "Detect vegetation change using Mann-Kendall or Theil-Sen analysis.",
                                group: "Run individual functions"));

            // add gallery item for trend solo function
            Add(new GalleryItem(text: "Change Vector Analysis",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_CVA_32.png",
                                tooltip: "Peform Change Vector Analysis (CVA).",
                                group: "Run individual functions"));

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

                // set toolname and create empty input array
                string toolname = "GDVSpectra_Likelihood";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try { 
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                } 
                catch (Exception) { 
                    Debug.WriteLine("Could not find GDVSpectra Likelihood tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Threshold Likelihood")
            {
                // set toolname and create empty input array
                string toolname = "GDVSpectra_Threshold";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra Threshold tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Trend Analysis")
            {
                // set toolname and create empty input array
                string toolname = "GDVSpectra_Trend";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra Trend Anaysis tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Change Vector Analysis")
            {
                // set toolname and create empty input array
                string toolname = "GDVSpectra_CVA";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find GDVSpectra CVA tool. Did you add the Tenement Tools toolbox?");
                };
            };
        }
    }
}
