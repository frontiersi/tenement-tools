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

            // add gallery item for nicher data
            Add(new GalleryItem(text: "Nicher SDM",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Nicher_SDM_32.png",
                                tooltip: "Generate ecological niche of a target species.",
                                group: "Niche Modelling Tools"));

            // add gallery item for nicher data
            Add(new GalleryItem(text: "Mask Nicher Output",
                                icon: "pack://application:,,,/TenementToolsApp;component/Images/Nicher_Masker_32.png",
                                tooltip: "Mask out erroneous pixels from Nicher output.",
                                group: "Post-processing Tools"));

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
                try
                {
                    string toolname = "Nicher_SDM";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find Nicher SDM tool. Did you add the toolbox?");
                };
            }
            else if (gallery_item == "Mask Nicher Output")
            {
                try
                {
                    string toolname = "Nicher_Masker";
                    Geoprocessing.OpenToolDialog(toolname, null);
                }
                catch (Exception)
                {
                    Debug.WriteLine("Could not find Nicher Masker tool. Did you add the toolbox?");
                };
            };
        }
    }
}
