@REM Store existing GeoTIFF env vars and set to this conda env
@REM so other GeoTIFF installs don't pollute the environment

@if defined GeoTIFF_DATA (
    set "_CONDA_SET_GEOTIFF_CSV=%GEOTIFF_CSV%"
)
@set "GEOTIFF_CSV=%CONDA_PREFIX%\Library\share\epsg_csv"

