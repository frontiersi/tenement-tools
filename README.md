# Tenement Tools
The Tenement Tools ArcGIS Pro plug-in is currently under active development. Bugs should be expected. 
If you would like to test the pre-release alpha version, please follow the below instructions.

## Requirements
* ArcGIS Pro version >= 2.8
* At least 8gb of ram (>=16gb recommended)
* Fast and stable internet connection
* Access to the following websites (IT may need to exclude from firewall):
  * https://conda.anaconda.org
  * https://repo.anaconda.com
  * https://conda-forge.org/
  * https://data.dea.ga.gov.au

## Setup
### 1. Download Tenement Tools plug-in and tutorial material
Download the <i>tenement-tools.zip</i>, <i>tutorial.zip</i> and <i>tutorial.pdf</i> files here: 
- https://github.com/frontiersi/tenement-tools/releases/tag/v0.9.1-prerelease-alpha.

### 2. Extract tenement-tools folder
Extract the tenement-tools folder from the <i>tenement-tools.zip</i> and store it somewhere permanent.
- Recommended location: <code>C:/Users/%USERNAME%/Documents/ArcGIS/tenement-tools</code>

### 3. Clone default ArcGIS Pro python environment
Tenement Tools requires a custom Python virtual environment. Luckily, ArcGIS Pro makes this easy:
1. Open Windows Start Menu > ArcGIS folder > Python Command Prompt
2. Enter and run <code>conda create --clone arcgispro-py3 --name arcgispro-py3-dev-tool</code>
3. When finished, enter and run: <code>activate arcgispro-py3-dev-tool</code>

### 4. Install addtional Python libraries
Now, enter and run each of the below in order (enter 'y' if asked at end of each):
- <code>conda install dask=2.30.0</code>
- <code>conda install -c conda-forge rasterio=1.1.5</code>
- <code>pip install pyproj==3.2.1</code>
- <code>conda install -c conda-forge scikit-learn=1.0</code>

Close the Python Command Prompt when finished.

### 5. Add the Tenement Tools plug-in to ArcGIS Pro
Now, add the actual plug-in to ArcGIS Pro:
- Run ArcGIS Pro > Add-In Manager > Options tab > Add Folder > Select extracted tenement-tools folder (from step 2)

### 6. Initialise the new python environment in ArcGIS Pro
Change ArcGIS Pro software to use the custom Python environment (from step 4):
- Run ArcGIS Pro > Settings > Python > Manage Environments > Select <i>arcgispro-py3-dev-tool</i> > OK

Restart ArcGIS Pro.

### 7. Create new ArcGIS Pro project and add Tenement Tools Toolbox
Create a new ArcGIS Pro project for the tutorial:
- New > Map 

Finally, add the Tenement Tools toolbox to the project:
- open Catalog Pane > right-click Toolboxes icon > Add Toolbox > go to extracted folder from step 2 > toolbox folder > add tenement-tools-toolbox.pyt

### 8. Walkthrough the tutorial pdf
A good place to start is with the Tutorial.pdf document downloaded in step 1.


## Install Issues
There are a known issues when cloning the arcgispro-py3 environment in step 3. You maye see the following when activating:
- The system cannot find the path specified
- set "GDAL_DRIVER_PATH='
- rem proj-data is installed because its license was copied over.

The plug-in should still work, regardless. We are currently investigating this issue.
