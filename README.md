# Tenement Tools
Tenement Tools placeholder!

## Installing (development build only)
Tenement Tools is currently under active development and is not suitable for use in production. If you're interested in testing the current development build, please follow the steps below. Please note: the UI is incomplete and it is likely that you will encounter bugs.
#### 1. Download ArcGIS Pro
First, download and install ArcGIS Pro 2.8. This can be obtained from your organisation's ESRI account. Full instructions are provided here: https://pro.arcgis.com/en/pro-app/latest/get-started/download-arcgis-pro.htm. Please note: development build has currently only been tested for ArcGIS Pro v2.8 - other versions may not be compatible.

#### 2. Clone the default ArcGIS Pro Python 3 virtual environment
Tenement Tools relies on several Python 3 libraries that are not enabled in ArcGIS Pro 2.8 by default. In order to enable them, the default Python 3 virtual environment must be cloned and these extra libraries installed into the clone. ESRI has made this process relatively painless. Follow the following steps below:
1. Make sure ArcGIS Pro is not running
2. Open the Windows Start menu and find the ArcGIS folder
3. Run the Python Command Prompt
4. Create the clone by entering: <code>conda create --clone arcgispro-py3 --name arcgispro-py3-dev-tool</code>
5. Activate the cloned environment: <code>activate arcgispro-py3-dev-tool</code>
6. Install dask library: <code>conda install dask=2.30.0</code>
7. Install rasterio library: <code>conda install -c esri rasterio</code>
8. Install pyproj library: <code>conda install pyproj</code>
ArcGIS Pro will now be using the cloned Python virtual environment to run all its geoprocessors, including Tenement Tools. If you no longer need to use the cloned environment, simply use the ArcGIS Pro Package Manager within the application to turn the default environment back on: https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/work-with-python-environments.htm.


