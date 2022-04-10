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
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

namespace TenementToolsApp
{
    internal class ensembleGallery : Gallery
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
            Add(new GalleryItem(text: "Prepare Evidence Layers",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_Sigmoid_32.png",
                                tooltip: "Rescale raw layers using fuzzy membership functions.",
                                group: "Run individual functions"));

            // add gallery item for wizard
            Add(new GalleryItem(text: "Perform Ensemble Modelling",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_32.png",
                                tooltip: "Combine fuzzy evidence layers via Dempster-Shafer modelling.",
                                group: "Run individual functions"));

            // add gallery item for wizard
            Add(new GalleryItem(text: "Mask Ensemble Output",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_Masker_32.png",
                                tooltip: "Mask ensemble output via a mask layer (e.g., canopy height layer).",
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

            // open tool pane
            if (gallery_item == "Prepare Evidence Layers")
            {
                // set toolname and create empty input array
                string toolname = "Ensemble_Sigmoider";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Sigmoider tool. Did you add the Tenement Tools toolbox?");
                };
            }

            // open tool pane
            else if (gallery_item == "Perform Ensemble Modelling")
            {
                // set toolname and create empty input array
                string toolname = "Ensemble_Model";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Model tool. Did you add the Tenement Tools toolbox?");
                };
            }
            // open tool pane
            if (gallery_item == "Mask Ensemble Output")
            {
                // set toolname and create empty input array
                string toolname = "Ensemble_Masker";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Masker tool. Did you add the Tenement Tools toolbox?");
                };
            };
        }
    }
}
