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
using System.Windows.Media.Imaging;

namespace TenementToolsApp
{
    internal class cogGallery : Gallery
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

            // add gallery item for fetch data
            Add(new GalleryItem(text: "Fetch Data",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/COG_Fetch_32.png",
                                tooltip: "Fetch satellite data from DEA public database.",
                                group: "Run individual functions"));

            // add gallery item for sync data
            Add(new GalleryItem(text: "Sync Data",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/COG_Sync_32.png",
                                tooltip: "Sync satellite data with DEA public database.",
                                group: "Run individual functions"));

            // add gallery item for visualise data
            Add(new GalleryItem(text: "Explore Data",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/COG_Explore_32.png",
                                tooltip: "Visualise, graph and explore satellite data.",
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
            if (gallery_item == "Fetch Data")
            {
                // set toolname and create empty input array
                string toolname = "COG_Fetch";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find COG Fetch tool.");
                };
            }
            else if (gallery_item == "Sync Data")
            {
                // temp: warn not yet implemented
                ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("Sync Data not yet implemented.");
            }
            else if (gallery_item == "Explore Data")
            {
                string toolname = "COG_Explore";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception e)
                {
                    Debug.WriteLine("Could not find COG Explore tool.");
                };

            };
        }
    }
}
