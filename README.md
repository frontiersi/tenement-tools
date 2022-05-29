# Tenement Tools
The Tenement Tools ArcGIS Pro plug-in is currently under active development and bugs should be expected. 
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
  * https://explorer.sandbox.dea.ga.gov.au/stac

<br />

## Setup (first time)
The Tenement Tools setup process can take up to 30 minutes. Please follow the below steps to install.

### 1. Download Tenement Tools plug-in and tutorial material
Download the <i>tenement-tools.zip</i>, <i>tutorial.zip</i> and <i>tutorial.pdf</i> files from the latest release located here: 
- https://github.com/frontiersi/tenement-tools/releases.

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
- <code>conda install -c conda-forge rasterio==1.1.5</code>
- <code>conda install -c conda-forge pyproj==2.6.1.post1</code>
- <code>conda install -c conda-forge scikit-learn==1.0</code>
- <code>conda install -c conda-forge odc-stac==0.2.2</code>
- <code>pip install pystac-client==0.3.0</code>

Close the Python Command Prompt when finished.

### 5. Add the Tenement Tools plug-in to ArcGIS Pro
Now, add the actual plug-in to ArcGIS Pro:
- Run ArcGIS Pro > Add-In Manager > Options tab > Add Folder > Select extracted tenement-tools folder (from step 2)

Note: you may need to click "Load all Add-Ins without restrictions" option.

### 6. Initialise the new python environment in ArcGIS Pro
Change ArcGIS Pro software to use the custom Python environment (from step 4):
- Run ArcGIS Pro > Settings > Python > Manage Environments > Select <i>arcgispro-py3-dev-tool</i> > OK

Restart ArcGIS Pro.

### 7. Create new ArcGIS Pro project and add Tenement Tools Toolbox
Create a new ArcGIS Pro project for the tutorial:
- New > Map 

Finally, add the Tenement Tools toolbox to the project:
- In Catalog Pane > right-click Toolboxes icon > Add Toolbox > go to extracted folder (step 2) > toolbox folder > add tenement-tools-toolbox.pyt

### 8. Walkthrough the tutorial pdf
A good place to start is with the Tutorial.pdf document downloaded in step 1.

## Updating to new version
Very 1.0 has now been released! Any prior installations using any development versions must be discarded and all steps must be done again.

## Known bugs
### UTF-8 encoding error when using COG Fetch or NRT Monitor Areas
This is a very rare and seemingly random error that can occur. Re-running the tool will fix it.

### Running a tool results in 'Tool has failed to open' error
Check that you have added the associated tenement tools toolbox into your current project (see step 7 above).

### The COG Fetch tools 'hang' forever when querying STAC metadata
This is a firewall issue. Please see the above requirements section and discuss opening these websites with IT. 

