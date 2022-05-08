using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
    internal class vegfraxButton : Button
    {
        protected override void OnClick()
        {
            try
            {
                string toolname = "VegFrax_Fractional_Cover";
                Geoprocessing.OpenToolDialog(toolname, null);
            }
            catch (Exception)
            {
                Debug.WriteLine("Could not find VegFrax Fractional Cover tool. Did you add the toolbox?");
            };
        }
    }
}
