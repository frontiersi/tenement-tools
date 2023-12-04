import xarray as xr

def check_xarray(ds, levels):
    messages = []
    if 0 in levels:
        if not isinstance(ds, xr.Dataset):
            messages.append('Input NetCDF must be a xr dataset.')
        if len(ds) == 0:
            messages.append('Input NetCDF has no data/variables/bands.')
        if 'x' not in ds.dims or 'y' not in ds.dims:
            messages.append('Input NetCDF must have x, y dimensions.')
        if 'x' not in ds.coords or 'y' not in ds.coords:
            messages.append('Input NetCDF must have x, y coords.')
        if 'spatial_ref' not in ds.coords:
            messages.append('Input NetCDF must have a spatial_ref coord.')
        if len(ds['x']) == 0 or len(ds['y']) == 0:
            messages.append('Input NetCDF must have all at least one x, y index.')
        if ds.attrs == {}:
            messages.append('NetCDF must have attributes.')
        if not hasattr(ds, 'crs'):
            messages.append('NetCDF CRS attribute not found. CRS required.')
        if ds.crs != 'EPSG:3577':
            messages.append('NetCDF CRS is not in GDA94 Albers (EPSG:3577).')
        if not hasattr(ds, 'nodatavals'):
            messages.append('NetCDF nodatavals attribute not found.')

    if 1 in levels:
        if ds[list(ds)[0]].isnull().all() or (ds[list(ds)[0]] == 0).all():
            messages.append('NetCDF has empty variables. Please download again.')

    if 2 in levels:
        pass

    return messages