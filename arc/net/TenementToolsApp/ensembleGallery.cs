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
                                tooltip: "Prepare evidence layers using fuzzy functions.",
                                group: "Pre-processing Tools"));

            // add gallery item for wizard
            Add(new GalleryItem(text: "Ensemble Modelling",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_32.png",
                                tooltip: "Combine fuzzy evidence layers via ensemble modelling.",
                                group: "Ensemble Modelling Tools"));

            // add gallery item for wizard
            Add(new GalleryItem(text: "Mask Ensemble Output",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Ensemble_Masker_32.png",
                                tooltip: "Mask out erroneous pixels from Ensemble output.",
                                group: "Post-processing Tools"));

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

            if (gallery_item == "Prepare Evidence Layers")
            {
                try
                {
                    string toolname = "Ensemble_Sigmoider";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Sigmoider tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Ensemble Modelling")
            {
                try
                {
                    string toolname = "Ensemble_Model";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Model tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Mask Ensemble Output")
            {
                try
                {
                    string toolname = "Ensemble_Masker";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find Ensemble Masker tool. Did you add the toolbox?");
                };
            };
        }
    }
}
