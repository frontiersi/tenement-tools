"%prefix%\python.exe" -c "import arcpy_init; arcpy_init.install()" >> "%prefix%\.messages.txt" 2>&1

if %errorlevel% neq 0 (
    echo Unable to link ArcPy DLLs to environment.
)
