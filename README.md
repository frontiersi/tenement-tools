# Tenement Tools
The Tenement Tools ArcGIS Pro plug-in is now version 1.0 and has been released! Please follow the below instructions to install the plug-in.

<br/>

## Requirements
* ArcGIS Pro version 2.8, 2.9, 3.0, or 3.1
* At least 8gb of ram (>=16gb recommended)
* Fast and stable internet connection with unlimited download capacity (imagery downloads are large!)
* Access to the following websites (IT may need to exclude from firewall):
  * https://conda.anaconda.org
  * https://repo.anaconda.com
  * https://conda-forge.org/
  * https://data.dea.ga.gov.au
  * https://explorer.sandbox.dea.ga.gov.au/stac
  * https://explorer.sandbox.dea.ga.gov.au/stac/search

<br/>

## Setup
Please choose the correct installation process for your version of ArcGIS Pro, the version number can be found in the 'Help -> About' menu within ArcGIS Pro.
- [Installation for ArcGIS Pro 2.8 and 2.9](./docs/install_arcgispro_29.md)
- [Installation for ArcGIS Pro 3.0](./docs/install_arcgispro_30.md)
- [Installation for ArcGIS Pro 3.1](./docs/install_arcgispro_31.md)

## Updating to newer versions of ArcGIS Pro
New versions of ArcGIS Pro may include different versions of the dependencies required by Tenement Tools. Incompatibilities have been observed when minor ArcGIS Pro version updates are made (e.g.; from ArcGIS Pro 3.0.0 to 3.1.0), patch updates (e.g.; from 3.1.0 to 3.1.1) have not introduced incompatibilities. This update process can be skipped for patch updates.

To ensure the installation of Tenement Tools is compatible with the newer version of ArcGIS Pro it is recommended that Tenement Tools be removed and reinstalled using a compatible version of Tenement Tools according to the following process before updating ArcGIS Pro;

To remove the Tenement Tools conda environment first identify the existing conda environment. Open an ArcGIS Python Command Prompt and run the following command.

    conda env list

From the list output by this command identify the old conda environment (e.g.; `arcgispro31-py3-dev-tool`) and run the following command being sure to replace the name with that identified by the prior command.

    conda remove --name arcgispro31-py3-dev-tool --all

Delete the folder that the Tenement Tools zip file was extracted to. After this is complete perform the ArcGIS Pro update and follow the [installation process for the appropriate version of ArcGIS Pro](#setup).

## Updating to newer versions of Tenement Tools
Generally newer versions of Tenement Tools do not introduce new dependencies, this means the existing Tenement Tools conda environment can be used across updates. A new release will state if this is not the case in the release notes.

The steps to update to a newer version of Tenement Tools are as follows;
1. Remove the Tenement Tools Python plugin
    - In Catalog Pane > expand the 'Toolboxes' tree item > right-click on 'tenement-tools-toolbox.pyt' > click 'Remove from Project'
    - Save project
1. Close ArcGIS Pro if open
2. Find and delete the location to which the previous Tenement Tools release package (.zip) was extracted to
3. Download and extract the latest version of Tenement Tools ensuring the release package matches that of the ArcGIS Pro version being used
4. Restart ArcGIS Pro
5. Within ArcGIS Pro update the path of the TenementToolsApp Add-In to refer to the new location the Tenement Tools release package was extracted to.
    - Details of this can be found in the installation process
6. Add the new 'tenement-tools-toolbox.pyt' plugin to the ArcGIS Pro project.
    - Details of this can be found in the installation process

<br/>

## Known bugs
### Running a tool results in 'Tool has failed to open' error
Check that you have added the associated tenement tools toolbox into your current project (see step 7 above).

### UTF-8 encoding error when using COG Fetch or NRT Monitor Areas tools
This is a very rare and seemingly random error that has been difficult to locate. Simply re-run the tool and it should work.

### The COG Fetch or NRT Monitor Areas tool 'hang' forever when querying STAC metadata
This is either a firewall issue or DEA is undergoing maintenance. Please see the above requirements section and discuss opening these websites with IT. 

### The COG Fetch or NRT Monitor Areas tool 'hang' forever when removing clouds from satellite data
This is a very rare and seemingly random error that has been difficult to locate. Restarting ArcGIS Pro fixes it.

### The COG Fetch or NRT Monitor Areas tool takes forever to download satellite imagery
This can be caused by many issues, including your internet connection speed or stability or the DEA database connection status (i.e., internal issues, maintenance). Please check your internet connection and try again. Also, try restarting ArcGIS Pro and downloading again.
