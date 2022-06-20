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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TenementToolsApp
{
    internal class aboutButton : Button
    {
        protected override void OnClick()
        {
            try
            {
                // set message for message box
                string msg = "This work represents years of intelligent conversations over " +
                             "coffee between Lewis Trotter and Todd Robinson.\n\n" + 
                             "Please cite this tool as: \n" +
                             "Trotter, L. and Robinson, T. (2022) - ENVestigator software. Curtin University.";

                // show message box on click
                ArcGIS.Desktop.Framework.Dialogs.MessageBox.Show(msg);
            }
            catch (Exception e)
            {
                Debug.WriteLine("Could not find About tool. Did you add the toolbox?");
            };
        }
    }
}
