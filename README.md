# Tenement Tools
Tenement Tools placeholder!

## Requirements
* Local install of ArcGIS Pro version 2.8 or above
* At least 8gb of ram (16gb recommended)
* Access to ArcGIS Python Command Prompt (comes with Pro)
* Fast and stable internet connection
* Access to the following websites (IT will need open these if blocked by firewall):
  * https://conda.anaconda.org
  * https://repo.anaconda.com
  * https://conda-forge.org/
  * https://data.dea.ga.gov.au

## Installing (alpha build)
The Tenement Tools ArcGIS Pro plug-in is currently under active development and should only be used for testing. If you are interested in testing this pre-release alpha, follow the installation instructions below.

#### 1. Download and install the latest version of Tenement Tools
Download the <code>tenement-tools.zip</code>, <code>tutorial.zip</code> and <code>tutorial.pdf</code> files here: https://github.com/frontiersi/tenement-tools/releases/tag/v0.9.1-prerelease-alpha.

Once downloaded, extract the tenement-tools folder from tenement-tools.zip file and copy and paste it into the default ArcGIS Pro AddIns\ArcGISPro folder typically located in your documents folder. 

For me, that is <code>C:\Users\YOUR USER NAME\Documents\ArcGIS\AddIns\ArcGISPro\[copy folder here]</code>.
 
The path should now look something like: ...\Documents\ArcGIS\AddIns\ArcGISPro\tenement-tools\...

#### 2. Install a custom ArcGIS Pro Python Environment
Tenement Tools uses a custom ArcGIS Pro Python environment to run some of its modules. 

Follow the below steps:
1. From Windows Start Menu, find the ArcGIS folder and open Python Command Prompt. You won't need admin rights.
2. Ensure you can see default env (arcgispro-py3) by running <code>conda env list</code>.
3. Create custom env by using a provided .yml file in the tenement-tools/arc/envs folder: <br/>
   <code>conda env create -f "C:\Users\YOUR USER NAME\Documents\ArcGIS\AddIns\ArcGISPro\tenement-tools\arc\envs\arcgispro-py3-dev-tool.yml"</code>. </br>
4. Activate the custom environment via <code>activate arcgispro-py3-dev-tool-extra</code>.
5. Close the Python Command Prompt.
  
#### 3. Initialise the custom ArcGIS Pro Python environment
Start ArcGIS Pro 2.8. When loaded, go to Settings > Python tab > Manage Environments button and change the current env to arcgispro-py3-dev-tool-extra. Restart ArcGIS Pro.

#### 4. Add the Tenement Tools Python Toolbox to ArcGIS
Finally, we need to add the ArcGIS Python Toolbox. Create a new project (or open an existing one) and open the Catalog Panel (click the View tab > Catalog Pane if missing).

Right-click the Toolboxes icon in Catalog Pane and click Add Toolbox. Navigate to the toolbox folder in the un-zipped tenement-tools folder from step 1 and add the toolbox called tenement-tools-toolbox.pyt. Click Ok. 

You will need to add this toolbox for every new ArcGIS Pro project you create.


