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
            Add(new GalleryItem(text: "Run All",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Wizard_32.png",
                                tooltip: "Generate GDV likelihood, threshold and trends.", 
                                group: "Wizard"));

            // add gallery item for gdv likelihood solo function
            Add(new GalleryItem(text: "GDV Likelihood",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Wizard_32.png",
                                tooltip: "Generate GDV likelihood from satellite time series.", 
                                group: "Run individual functions"));

            // add gallery item for likelihood threshold solo function
            Add(new GalleryItem(text: "Threshold Likelihood", 
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Threshold_32.png",
                                tooltip: "Threshold GDV likelihood using occurrence points or standard deviation.", 
                                group: "Run individual functions"));

            // add gallery item for trend solo function
            Add(new GalleryItem(text: "Trend Analysis",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/GDVSpectra_Trend_32.png",
                                tooltip: "Detect vegetation change trends using Mann-Kendall or Theil-Sen.",
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
            //TODO - insert your code to manipulate the clicked gallery item here
            //System.Diagnostics.Debug.WriteLine("Remove this line after adding your custom behavior.");
            
            // obtain clicked gallery item
            base.OnClick(item);

            // get name of clicked gallery item
            var gallery_item = base.SelectedItem.ToString();

            // temp: tell dev what was called
            //ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show(gallery_item);

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
                catch (Exception e) { 
                    Debug.WriteLine("Could not find GDVSpectra tool.");
                };
            }

            // open gdvspectra threshold tool pane
            else if (gallery_item == "GDV Threshold")
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
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find GDVSpectra tool.");
                };
            }

            // open gdvspectra threshold tool pane
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
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find GDVSpectra tool.");
                };
            };
        }
    }
}
