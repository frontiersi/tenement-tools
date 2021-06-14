using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Catalog;
using ArcGIS.Desktop.Core;
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
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

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

            // add gallery item for wizard
            Add(new GalleryItem(text: "Run All",
                                icon: this.LargeImage != null ? ((ImageSource)this.LargeImage).Clone() : null,
                                tooltip: "Fetch satellite data from DEA Public Data.",
                                group: "Wizard"));

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
            ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show(gallery_item);
        }
    }
}
