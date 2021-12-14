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
            Add(new GalleryItem(text: "Ensemble Model",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_32.png",
                                tooltip: "Combine outputs into a Dempster-Shafer model.",
                                group: "Run individual functions"));

            // add gallery item for wizard
            Add(new GalleryItem(text: "Ensemble Masker",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/CanoPy_Wizard_32.png",
                                tooltip: "Mask out Ensemble areas using an elevation-based mask (e.g., Canopy Height).",
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
            if (gallery_item == "Ensemble Model")
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
            if (gallery_item == "Ensemble Masker")
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
