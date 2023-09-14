## Setup for ArcGIS Pro v2.8 and v2.9
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
2. Enter and run <code>conda create --clone arcgispro-py3 --name arcgispro29-py3-dev-tool</code>
3. When finished, enter and run: <code>activate arcgispro29-py3-dev-tool</code>

### 4. Install addtional Python libraries
Additional Python libraries are defined within a conda environment file. This file can be downloaded
from the following link.
[https://raw.githubusercontent.com/frontiersi/tenement-tools/arcgispro29/arc/envs/arcgispro.yml](https://raw.githubusercontent.com/frontiersi/tenement-tools/arcgispro29/arc/envs/arcgispro.yml)

To download this file, first open the link, then right-click and save the file to a suitable location.

Now, enter and run the following command. This assumes the environment file downloaded above was saved
to the active directory. Note: this command may take some time to complete.
- <code>conda env update --file arcgispro.yml</code>

Close the Python Command Prompt when finished.

### 5. Add the Tenement Tools plug-in to ArcGIS Pro
Now, add the actual plug-in to ArcGIS Pro:
- Run ArcGIS Pro > Add-In Manager > Options tab > Add Folder > Select extracted tenement-tools folder (from step 2)

Note: you may need to click "Load all Add-Ins without restrictions" option.

### 6. Initialise the new python environment in ArcGIS Pro
Change ArcGIS Pro software to use the custom Python environment (from step 4):
- Run ArcGIS Pro > Settings > Python > Manage Environments > Select <i>arcgispro29-py3-dev-tool</i> > OK

Restart ArcGIS Pro.

### 7. Create new ArcGIS Pro project and add Tenement Tools Toolbox
Create a new ArcGIS Pro project for the tutorial:
- New > Map 

Finally, add the Tenement Tools toolbox to the project:
- In Catalog Pane > right-click Toolboxes icon > Add Toolbox > go to extracted folder (step 2) > toolbox folder > add tenement-tools-toolbox.pyt

### 8. Walkthrough the tutorial pdf
A good place to start is with the Tutorial.pdf document downloaded in step 1.
