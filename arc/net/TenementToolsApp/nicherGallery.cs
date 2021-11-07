using System;
using System.Diagnostics;
using ArcGIS.Desktop.Core.Geoprocessing;
using ArcGIS.Desktop.Framework.Contracts;

namespace TenementToolsApp
{
    internal class nicherGallery : Gallery
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
            //icon: this.LargeImage != null ? ((ImageSource)this.LargeImage).Clone() : null,
            //tooltip: "Generate species distribution (niche) model.",
            //group: "Wizard"));

            // add gallery item for nicher data
            Add(new GalleryItem(text: "Nicher SDM",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Nicher_SDM_32.png",
                                tooltip: "Generate a Species Distribution Model via species locations and topgraphic variables.",
                                group: "Run individual functions"));

            // add gallery item for nicher data
            Add(new GalleryItem(text: "Nicher Masker",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/CanoPy_Wizard_32.png",
                                tooltip: "Mask out SDM areas using an elevation-based mask (e.g., Canopy Height).",
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

            // open gdvspectra likelihood tool pane
            if (gallery_item == "Nicher SDM")
            {

                // set toolname and create empty input array
                string toolname = "Nicher_SDM";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find Nicher SDM tool. Did you add the Tenement Tools toolbox?");
                };
            }
            else if (gallery_item == "Nicher Masker")
            {

                // set toolname and create empty input array
                string toolname = "Nicher_Masker";
                var inputs = Geoprocessing.MakeValueArray();
                inputs = null;

                // open toolpane
                try
                {
                    Geoprocessing.OpenToolDialog(toolname, inputs);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find Nicher Masker tool. Did you add the Tenement Tools toolbox?");
                };
            };
        }
    }
}
