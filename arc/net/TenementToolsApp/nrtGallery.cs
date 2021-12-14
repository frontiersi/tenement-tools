using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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

            // add gallery item for nrt area selector
            Add(new GalleryItem(text: "NRT Manage Areas",
                                icon: this.LargeImage != null ? ((ImageSource)this.LargeImage).Clone() : null,
                                tooltip: "Create new monitoring areas or manage existing ones.",
                                group: "Run individual functions"));

            // add gallery item for nrt monitor
            Add(new GalleryItem(text: "NRT Monitor",
                                icon: this.LargeImage != null ? ((ImageSource)this.LargeImage).Clone() : null,
                                tooltip: "Monitor vegetation within NRT Areas.",
                                group: "Run individual functions"));

            // initialise
            _isInitialized = true;

        }

        protected override void OnClick(GalleryItem item)
        {
            // ensure users can e-click on already selected items
            base.AlwaysFireOnClick = true;

            // obtain clicked gallery item
            base.OnClick(item);

            // get name of clicked gallery item
            var gallery_item = base.SelectedItem.ToString();

            // open tool pane
            if (gallery_item == "NRT Manage Areas")
            {
                // set toolname and create empty input array
                //string toolname = "Nicher_SDM";
                //var inputs = Geoprocessing.MakeValueArray();
                //inputs = null;

                // open toolpane
                //try
                //{
                //Geoprocessing.OpenToolDialog(toolname, inputs);
                //}
                //catch (Exception)
                //{
                //Debug.WriteLine("Could not find Nicher SDM tool. Did you add the Tenement Tools toolbox?");
                //};

                // temp: warn not yet implemented
                ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("The NRT Manage Areas tool has not yet been implemented.");
            }
            else if (gallery_item == "NRT Monitor")
            {
                // set toolname and create empty input array
                //string toolname = "Nicher_Masker";
                //var inputs = Geoprocessing.MakeValueArray();
                //inputs = null;

                // open toolpane
                //try
                //{
                //Geoprocessing.OpenToolDialog(toolname, inputs);
                //}
                //catch (Exception)
                //{
                //Debug.WriteLine("Could not find Nicher Masker tool. Did you add the Tenement Tools toolbox?");
                //};

                // temp: warn not yet implemented
                ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show("The NRT Monitor tool has not yet been implemented.");
            };
        }
    }
}
