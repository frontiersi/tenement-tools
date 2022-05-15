# deprecated
def generate_random_samples(ds_raw, ds_class, num_samples=1000, snap=True, res_factor=3):
    """
    Generates random point locations within dataset mask. These points are used to
    train the random forest classifier. A pandas dataframe of x and y coordinates 
    extracted from the dataset are returned. This is a custom func specifically for 
    the gdv fractional cover module.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    num_samples: int
        A int indicating how many points to generated.
    snap : bool
        If true, pixel centroids will replace our random sample point x and y values
        to reduce chance of assigning a random sample point right on the border of 
        multiple pixels. Default is True.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_samples: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """
    
    # notify user
    print('Generating {0} randomised sample points.'.format(num_samples))

    # check if raw dataset is xarray dataeset type
    if not isinstance(ds_raw, xr.Dataset):
        raise ValueError('> Raw dataset is not an xarray dataset.')
        
    # check if class dataset is xarray dataeset type
    if not isinstance(ds_class, xr.Dataset):
        raise ValueError('> Classified dataset is not an xarray dataset.')
    
    # check if number of absence points is an int
    if not isinstance(num_samples, int):
        raise ValueError('> Num of points value is not an integer. Please check the entered value.')
        
    # check if number of absence points is an int
    if not isinstance(snap, bool):
        raise ValueError('> Snap must be a boolean (True or False.')
        
    # check if number of absence points is an int
    if not isinstance(res_factor, int):
        raise ValueError('> Resolution factor must be an integer.')
        
    # get cell resolution for both datasets    
    res_raw = tools.get_xr_resolution(ds_raw)
    res_class = tools.get_xr_resolution(ds_class)
    
    # check if class res greater than raw - 
    if res_class >= res_raw:
        raise ValueError('> Classified raster is lower resolution than raw raster(s) - must be other way around.')
                
    # create a dummy grid based on a single slice of raw data
    if 'variable' in ds_raw.to_array().dims:
        raw_dummy = xr.ones_like(ds_raw.to_array().isel(variable=0))
        raw_dummy = raw_dummy.drop('variable', errors='ignore')
    else:
        raise AttributeError('> Raw dataset does not contain a variable dim.')
        
    # get extent of class raster
    class_extent = tools.get_xr_extent(ds=ds_class)
        
    # 'clip' raw dataset to class image bounds incase it is small subsection
    raw_dummy = raw_dummy.sel(x=slice(class_extent.get('l'), class_extent.get('r')), 
                              y=slice(class_extent.get('t'), class_extent.get('b')))   

    # get 10% of x, y size, use to trim 10% off extent bounds, subset dummy with it
    x_slice, y_slice = round(raw_dummy['x'].size * 0.1), round(raw_dummy['y'].size * 0.1)
    raw_dummy = raw_dummy.isel(x=slice(x_slice, -x_slice), y=slice(y_slice, -y_slice))
    
    # check if raw dummy has pixels still
    if raw_dummy.size <= 0:
        raise ValueError('> No pixels exist when raw and classified rasters clipped. Do they overlap?')
            
    # ensure enough pixels in final dummy to handle num of random samples
    num_cells = raw_dummy['x'].size * raw_dummy['y'].size
    if num_samples > num_cells:
        print('Too many random samples requested - reducing to: {0}.'.format(num_cells))
        num_samples = num_cells

    # notify
    print('> Randomising points within mask area.')

    # get bounds of final dummy
    dummy_extent = tools.get_xr_extent(ds=ds_class)
        
    # create random points and fill a list with x and y
    counter = 0
    coords = []
    for i in range(num_samples):
        while counter < num_samples:

            # get random x and y coord
            rand_x = random.uniform(dummy_extent.get('l'), dummy_extent.get('r'))
            rand_y = random.uniform(dummy_extent.get('b'), dummy_extent.get('t'))

            # create point and add x and y to it
            pnt = ogr.Geometry(ogr.wkbPoint)
            pnt.AddPoint(rand_x, rand_y)
            
            try:
                # get pixel
                pixel = raw_dummy.sel(x=rand_x, y=rand_y, method='nearest', tolerance=res_raw * res_factor)

                # add if pixel is valid value
                if int(pixel) == 1:
                    
                    # get x and y based on snap (where point was generated or nearest pixel centroid)
                    if snap:
                        coords.append([pixel['x'].item(), pixel['y'].item()])
                    else:
                        coords.append([pnt.GetX(), pnt.GetY()])

                    # inc counter
                    counter += 1
           
            except:
                continue
                
    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_samples = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    raw_dummy, buff_geom = None, None

    # notify and return
    print('> Generated random sample points successfully.')
    return df_samples


# deprectaed
def generate_strat_random_samples(ds_raw, ds_class, req_class=None, num_samples=500, 
                                  snap=True, res_factor=3):
    """
    Generates stratified random point locations within dataset classes. These points 
    are used to train the random forest classifier. A pandas dataframe of x and y coordinates 
    extracted from the dataset are returned. This is a custom func specifically for 
    the gdv fractional cover module.

    Parameters
    ----------
    ds_raw : xarray dataset
        A dataset holding the low resolution, raw raster bands.
    ds_class : xarray dataset
        A dataset holding the high resolution, classified raster.
    req_class : int, list
        A integer or list of class numbers in which to generate
        random stratified samples within.
    num_samples: int
        A int indicating how many points to generated.
    snap : bool
        If true, pixel centroids will replace our random sample point x and y values
        to reduce chance of assigning a random sample point right on the border of 
        multiple pixels. Default is True.
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_samples: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """

    # notify user
    print('Generating {} stratified randomised sample points.'.format(num_samples))

    # check if raw dataset is xarray dataeset type
    if not isinstance(ds_raw, xr.Dataset):
        raise ValueError('Raw dataset is not an xarray dataset.')
        
    # check if class dataset is xarray dataeset type
    if not isinstance(ds_class, xr.Dataset):
        raise ValueError('Classified dataset is not an xarray dataset.')
    
    # check if number of absence points is an int
    if not isinstance(num_samples, int):
        raise ValueError('Number of points value is not an integer.')
    elif not isinstance(snap, bool):
        raise ValueError('Snap must be a boolean (True or False).')
    elif not isinstance(res_factor, int):
        raise ValueError('Resolution factor must be an integer.')
        
    # check if req classes is list or int, convert to list
    req_class = req_class if req_class is not None else []
    req_classes = req_class if isinstance(req_class, list) else [req_class]
        
    # get cell resolution for both datasets    
    res_raw = tools.get_xr_resolution(ds_raw)
    res_class = tools.get_xr_resolution(ds_class)
    
    # check if class res greater than raw
    if res_class >= res_raw:
        raise ValueError('Classified raster must be higher resolution than raw raster(s).')
                
    # create a dummy grid based on a single slice of raw data
    if 'variable' in ds_raw.to_array().dims:
        raw_dummy = xr.ones_like(ds_raw.to_array().isel(variable=0))
        raw_dummy = raw_dummy.drop('variable', errors='ignore')
    else:
        raise AttributeError('> Raw dataset does not contain a variable dim.')
                       
    # get extent of class raster
    class_extent = tools.get_xr_extent(ds=ds_class)
        
    # 'clip' raw dataset to class image bounds incase it is small subsection
    raw_dummy = raw_dummy.sel(x=slice(class_extent.get('l'), class_extent.get('r')), 
                              y=slice(class_extent.get('t'), class_extent.get('b')))   

    # get 5% of x, y size, use to trim 5% off extent bounds, subset dummy with it
    x_slice, y_slice = round(raw_dummy['x'].size * 0.05), round(raw_dummy['y'].size * 0.05)
    raw_dummy = raw_dummy.isel(x=slice(x_slice, -x_slice), y=slice(y_slice, -y_slice))
    
    # check if raw dummy has pixels still
    if raw_dummy.size <= 0:
        raise ValueError('No pixels exist when raw and classified rasters clipped. Do they overlap?')
            
    # ensure enough pixels in final dummy to handle num of random samples
    num_cells = raw_dummy['x'].size * raw_dummy['y'].size
    if num_samples > num_cells:
        print('Too many random samples requested - reducing to: {}.'.format(num_cells))
        num_samples = num_cells
        
    # get bounds of final dummy
    dummy_extent = tools.get_xr_extent(ds=raw_dummy)  

    # notify
    print('Generating stratified random points.')
    
    coords = []
    for req_class in req_classes:
        
        # notify
        print('Preparing samples for class: {}'.format(req_class))

        # convert to pandas
        df_mask = xr.where(ds_class == req_class, True, False).to_dataframe()
        df_mask = df_mask.loc[df_mask['classes'] == True].sample(5000)
        df_mask = df_mask.drop('classes', axis='columns')

        # create random points and fill a list with x and y
        counter = 0
        while counter < num_samples:

            # get a single, random row
            xy = df_mask.sample(1)
            xy = xy.index.to_list()[0]
            x, y = xy[0], xy[1]

            try:
                # get pixel value from low res ds
                pixel = raw_dummy.sel(x=x, y=y, 
                                      method='nearest', 
                                      tolerance=res_raw * res_factor)
                
                # removed check for valid (i.e., val = 1)

                if snap:
                    coords.append([float(pixel['x']), 
                                   float(pixel['y'])])
                else:
                    coords.append([float(x), 
                                   float(y)])
                
                # up the counter
                counter += 1

            except:
                continue
      
    # check if list is populated
    if not coords:
        raise ValueError('No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_samples = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    raw_dummy = None

    # notify and return
    print('Generated stratified random sample points successfully.')
    return df_samples
