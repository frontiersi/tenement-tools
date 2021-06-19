@REM Restore previous GeoTIFF env vars if they were set

@set "GEOTIFF_CSV="
@if defined _CONDA_SET_GEOTIFF_CSV (
  set "GEOTIFF_CSV=%_CONDA_SET_GEOTIFF_CSV%"
  set "_CONDA_SET_GEOTIFF_CSV="
)

