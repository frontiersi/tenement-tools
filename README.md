# Tenement Tools
Tenement Tools placeholder!

## Requirements
* ArcGIS Pro version 2.8 or above
* At least 8gb of ram (16gb recommended)
* Access to ArcGIS Python Command Prompt (if Pro is installed, you should have)
* Fast and stable internet connection
* Access to the following websites (IT will need open these if blocked by firewall):
  * https://conda.anaconda.org
  * https://repo.anaconda.com
  * https://conda-forge.org/
  * https://data.dea.ga.gov.au

## Installing (alpha development build only)
Tenement Tools is currently under active development and is not suitable for use in production. If you're interested in testing the current development build, please follow the steps below. 
<br/>
Please note: the UI is incomplete, there are known bugs, and we hope to make the below process easier.

#### 1. Download ArcGIS Pro 2.8 (skip to step 2 if already installed)
First, download and install ArcGIS Pro 2.8. This can be obtained from your organisation's ESRI account. Full instructions are provided here: https://pro.arcgis.com/en/pro-app/latest/get-started/download-arcgis-pro.htm. Please note: this build has only been tested for ArcGIS Pro v2.8 - other versions may not be compatible.

#### 2. Download and install the latest version of tenement-tools
To obtain the latest version of tenement tools, click the releases link on the right panel of this page, click the top version number (e.g., v0.9.0), then download tenement-tools.zip. Also download the Tutorial.zip file if you are following the official tutorial. 
<br/>
Once downloaded, unzip the tenement-tools folder, and copy and paste it into the default ArcGIS Pro AddIns folder: C:\Users\<YOUR USERNAME>\Documents\ArcGIS\AddIns\ArcGISPro. <br/>
You should now have a path like: ...\AddIns\ArcGISPro\tenement-tools\...

#### 3. Install the custom ArcGIS Pro Python Environment
Tenement Tools relies on several Python 3 libraries that are not enabled in ArcGIS Pro 2.8 by default. In order to enable them, a custom Python 3 virtual environment must be created. Follow these steps:
1. Open the Windows Start menu, find the ArcGIS start menu folder, and within it click Python Command Prompt.
2. Make sure arcgispro-py3 environment exists by entering <code>conda env list</code>. You should see it listed in the result.
3. Create the custom environment by entering: <code>conda env create -f "C:\Users\<YOUR USERNAME>\Documents\ArcGIS\AddIns\ArcGISPro\tenement-tools\arc\envs\arcgispro-py3-dev-tool.yml"</code>.
4. When complete, activate the custom environment with: <code>activate arcgispro-py3-dev-tool-extra</code>.
5. Close the Python Command Prompt.
  
#### 4. Initialise the custom ArcGIS Pro Python environment
Start ArcGIS Pro 2.8. When loaded, click Settings and go to Python. Click the Manage Environments button at the top of the page, and change the environment to arcgispro-py3-dev-tool-extra. You have now initialised the custom environment. Restart ArcGIS Pro once you have clicked OK.

#### 5. Add the Tenement Tools Python Toolbox to ArcGIS
Finally, add the ArcGIS Pro Python Toolbox to ArcGIS so the UI has access to the underlying methods. Create a new project (or open an existing one). Open the Catalog Panel (if you can't see it, click the View tab at the top of the ribbon, then click Catalog Pane). Right-click the Toolboxes icon and click Add Toolbox. Navigate to the toolbox folder from the unzipped tenement-tools folder here: <code>tenement-tools\toolbox</code>, click the toolbox called tenement-tools-toolbox.pyt and click Ok.

Congratulations, you can now try Tenement Tools. Click the Add-In tab at the top of the application to access the Tenement Tools UI.
