set DISABLE_ARCGIS_LEARN=1
"%PREFIX%\python.exe" -m arcgis.install --remove  >> "%PREFIX%/.messages.txt" 2>&1

set err_msg=jupyter nbextension command failed: map widgets in the jupyter notebook may not work, installation continuing...

"%PREFIX%\Scripts\jupyter-nbextension.exe" install --py --sys-prefix arcgis  >> "%PREFIX%/.messages.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %err_msg%
) 

"%PREFIX%\Scripts\jupyter-nbextension.exe" enable  --py --sys-prefix arcgis  >> "%PREFIX%/.messages.txt" 2>&1 
if %ERRORLEVEL% NEQ 0 (
    echo %err_msg%
)
set DISABLE_ARCGIS_LEARN=''
