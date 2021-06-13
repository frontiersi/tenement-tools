set DISABLE_ARCGIS_LEARN=1
set err_msg=jupyter nbextension command failed: map widgets in the jupyter notebook not uninstalled, uninstallation continuing...

"%PREFIX%\Scripts\jupyter-nbextension.exe" disable --py --sys-prefix arcgis  >> "%PREFIX%/.messages.txt" 2>&1 
if %ERRORLEVEL% NEQ 0 (
    echo %err_msg%
)

"%PREFIX%\Scripts\jupyter-nbextension.exe" uninstall --py --sys-prefix arcgis  >> "%PREFIX%/.messages.txt" 2>&1 
if %ERRORLEVEL% NEQ 0 (
    echo %err_msg%
)
set DISABLE_ARCGIS_LEARN=''
