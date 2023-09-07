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

## Updating to new version
Version 1.0 has now been released! Any prior installations versions must be discarded and all steps above must be done again.

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
