# nicher
'''
This script contains functions for calculating species distribution models
(SDMs), also known as ecological niche models (ENMs) - hence, Nicher. This
script is intended to accept a digital elevation model (DEM) geotiff and any
pre-generated derivatives (e.g. slope, aspect, topographic wetness index, etc.) 
that are generated from it. The methodology is based on the approach used by
MaxEnt (https://biodiversityinformatics.amnh.org/open_source/maxent/), but replaces
the Logistic Regression technique with ExtraTrees and RandomForest from Sklearn. 
Various functions in Nicher are based on the excellent RSGISLib SDM library 
(https://www.rsgislib.org/rsgislib_sdm.html), especially the creation of response
curves and lek matrices. If you cite this library, please also cite RSGISLib.

See associated Jupyter Notebook Nicher.ipynb for a basic tutorial on the
main functions and order of execution.

Links:
MaxEnt: https://biodiversityinformatics.amnh.org/open_source/maxent
RSGISLib: https://www.rsgislib.org/rsgislib_sdm

Contacts: 
Lewis Trotter: lewis.trotter@postgrad.curtin.edu.au
'''

# import required libraries
import os
import sys
import random
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier as et
from sklearn.ensemble import RandomForestClassifier as rt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pointbiserialr

sys.path.append('../../shared')
import tools

# increase number of pandas rows displayed
pd.set_option('display.max_rows', 1000)

def get_files_from_path(folder_path, file_type='.tif'):
    """
    Quick helper to read folder and obtain list of files with 
    a specific extension.
    
    Parameters
    ----------
    folder_path: str
        A single string with full path to folder of files.
    file_type : str
        A single string of requested file type.

    Returns
    ----------
    file_list : list
        List of file paths and names.
    """
    
    # check if folder path given and exists
    if folder_path is None:
        raise ValueError('Must provide a folder path.')
    elif not os.path.isdir(folder_path):
        raise TypeError('Folder path does not exist or incorrect.')
        
    # get files from path
    file_list = []
    for fn in os.listdir(path=folder_path):
        if fn.endswith(file_type):
            print('Found: {}'.format(fn))
            file_list.append(os.path.join(folder_path, fn))
            
    # return
    return file_list


def generate_absences(ds, num_abse=1000, occur_shp_path=None, buff_m=100, res_factor=3):
    """
    Generates pseudo-absence (random or pseudo-random) locations within dataset mask. 
    Pseudo-absence points are key in sdm work. A dataset, value for number
    of pseudo-absence points, and NoData value to generate is required. Optionally, if 
    user provides an occurrence shapefile path and buffer length (in metres), proximity 
    buffers can be also created around species occurrences - another often used function 
    in sdm work.

    Parameters
    ----------
    ds : xarray dataset
        A dataset holding at least one variable.
    num_abse: int
        A int indicating how many pseudo-absence points to generated.
    occur_shp_path : string (optional)
        A single string with full path and filename of shapefile of occurrence records.
    buff_m : int (optional)
        A int indicating radius of buffers (proximity zones) around occurrence points (in metres).
    res_factor : int
        A threshold multiplier used during pixel + point intersection. For example
        if point within 3 pixels distance, get nearest (res_factor = 3). Default 3.

    Returns
    ----------
    df_absence: pandas dataframe
        A pandas dataframe containing two columns (x and y) with coordinates.
    """
    
    # imports
    # imports check
    try:
        from osgeo import ogr
    except:
        raise ImportError('Could not import osgeo.')
    
    # notify user
    print('Generating {0} randomised psuedo-absence locations.'.format(num_abse))

    # do various checks
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise ValueError('Dataset is not an xarray dataset or dataarray.')
    elif not num_abse > 0:
        raise ValueError('Num of absence points must be > 0.')  
    elif not hasattr(ds, 'nodatavals'):
        raise ValueError('No NoData attribute in dataset.')
        
    # make mask array, flag any pixel where nan exists
    da_mask = xr.where(ds != ds.nodatavals, 1, 0)
    da_mask = da_mask.to_array(dim='mask').min('mask')

    # check if any mask pixels (i.e. 1s) were returned
    if not bool(da_mask.any()):
        raise ValueError('No non-data pixels found - check your rasters.')

    # erase proximities from mask if user provides values for it
    buff_geom = None
    if occur_shp_path and buff_m:
        print('Generating buffer areas from occurrence points.')

        # read proximity buffer geoms
        buff_geom = generate_proximity_areas(occur_shp_path, buff_m)

        # check if proximity buffer geom exists
        if not buff_geom or not buff_geom.GetGeometryCount() > 0:
            raise ValueError('No proximity buffers generated in dissolved mask.')

    # notify
    print('Randomising absence points within mask area.')
    
    # get cell resolution from dataset
    res = tools.get_xr_resolution(ds)

    # get bounds of mask  # todo update this to gdv_tools.get_dataset_extent
    bb = tools.get_xr_extent(ds)
    x_min, x_max = bb['l'], bb['r']
    y_min, y_max = bb['b'], bb['t']

    # create random points and fill a list with x and y
    counter = 0
    coords = []
    #for i in range(num_abse): #Appears redundant
    while counter < num_abse:

        # get random x and y coord
        rand_x = random.uniform(x_min, x_max)
        rand_y = random.uniform(y_min, y_max)

        # create point and add x and y to it
        pnt = ogr.Geometry(ogr.wkbPoint)
        pnt.AddPoint(rand_x, rand_y)
        
        try:
            pixel = int(da_mask.sel(x=rand_x, y=rand_y, 
                                    method='nearest', 
                                    tolerance=res * res_factor))
            
            # if within pixel and buffer areas, else just pixel
            if pixel == 1 and buff_geom:
                if not pnt.Within(buff_geom):
                    coords.append([pnt.GetX(), pnt.GetY()])
                    counter += 1
                    
            elif pixel == 1:
                coords.append([pnt.GetX(), pnt.GetY()])
                counter += 1
                
        except:
            continue

    # check if list is populated
    if not coords:
        raise ValueError('> No coordinates in coordinate list.')

    # convert coord array into dataframe
    df_absence = pd.DataFrame(coords, columns=['x', 'y'])

    # drop variables
    da_mask, buff_geom = None, None

    # notify and return
    print('Generated pseudo-absence points successfully.')
    return df_absence


def generate_proximity_areas(shp_path, buff_m=100):
    """
    Read species point locations from a projected ESRI Shapefile and buffer them to a user 
    defined number of metres. This must be a point geometry-type shapefile and it must be 
    projected. The resulting buffer areas can be used to eliminate pseudo-absences.
    
    Parameters
    ----------
    shp_path : string
        A single string with full path and filename of shapefile.
    buff_m : int
        A numeric value in which points a buffered by. Must be metres. Default 100m.

    Returns
    ----------
    buff_geom : ogr vector geometry
        An ogr vector of type multipolygon geometry with a single feature (dissolved).
    """
    
    # imports check
    try:
        from osgeo import ogr
    except:
        raise ImportError('Could not import osgeo.')

    # notify user
    print('Generating proximity buffers around species point locations.')

    # check inputs
    if not isinstance(shp_path, str):
        raise ValueError('Shapefile path must be a string.')
    elif not os.path.exists(shp_path):
        raise OSError('Unable to read species point locations, file not found.')
    elif not buff_m > 0:
        raise ValueError('Buffer length must be > 0.')    

    try:
        # read shapefile as layer
        shp = ogr.Open(shp_path, 0)
        lyr = shp.GetLayer()

        # try and get epsg code via authority
        epsg = lyr.GetSpatialRef().GetAttrValue('AUTHORITY', 1)
        
        # arcgis can sometimes miss authority, try backup method
        if epsg is None:
            tag = lyr.GetSpatialRef().GetAttrValue('PROJCS', 0)
            epsg = '3577' if 'albers' in tag.lower() else None
        
        # final check
        if epsg is not None:
            epsg = int(epsg)
        else:
            raise AttriuteError('Could not get EPSG from shapefile. Please check.')
        
        # get num feats
        num_feats = lyr.GetFeatureCount()

    except:
        raise TypeError('Could not read species point locations.')
        
    # check shapefile parameters
    if epsg != 3577:
        raise ValueError('Shapefile is not projected in GDA94 Albers. Please reproject into EPSG: 3577.')
    elif lyr.GetGeomType() not in [ogr.wkbPoint, ogr.wkbMultiPoint]:
        raise ValueError('Shapefile is not a point/multi-point type.')
    elif num_feats == 0:
        raise ValueError('Shapefile has no features in it. Please check.')

    # loop feats
    buff_geom = ogr.Geometry(ogr.wkbMultiPolygon)
    for feat in lyr:
        geom = feat.GetGeometryRef()

        # add geom if individual polygon type
        if geom.GetGeometryName() == 'POINT':
            geom = geom.Buffer(buff_m)
            buff_geom.AddGeometry(geom)

        # add geom if multi-polygon type
        elif geom.GetGeometryName() == 'MULTIPOINT':
            for i in range(geom.GetGeometryCount()):
                sub_geom = geom.GetGeometryRef(i)
                sub_geom = sub_geom.Buffer(buff_m)
                buff_geom.AddGeometry(sub_geom)

        # error, a non-polygon type exists
        else:
            raise TypeError('Unable to read point, geometry is invalid.')

    # union all features together (dissolve)
    buff_geom = buff_geom.UnionCascaded()

    # check if buffer geom is populated
    if not buff_geom or not buff_geom.GetGeometryCount() > 0:
        raise ValueError('No features exist in proximity buffer. Check point shapefile.')

    # drop variables
    shp, lyr = None, None

    # notify user and return
    print('Proximity buffers loaded and dissolved successfully.')
    return buff_geom


def equalise_abse_records(df_presence, df_absence):
    """
    Read presence and absence dataframes and reduce number of absence records to match
    the number of presence records.
    
    Parameters
    ----------
    df_presence : pandas dataframe
        A dataframe holding records with presence values.
    df_absence : pandas dataframe
        A dataframe holding records with absence values.

    Returns
    ----------
    df_absence : pandas dataframe
        Equalised pandas dataframe with absence records.
    """
    
    # notify
    print('Equalising absence record number.')
    
    # check if presence is dataframe
    if not isinstance(df_presence, pd.DataFrame):
        raise TypeError('Presence records is not a pandas dataframe.')

    # check if absence is dataframe
    if not isinstance(df_absence, pd.DataFrame):
        raise TypeError('Absence records is not a pandas dataframe.')

    # count number of records in presence and absence
    num_pres = df_presence.shape[0]
    num_abse = df_absence.shape[0]

    #Set default return to df_absence
    result = df_absence

    # select same number of presence in absence random sample.
    if num_abse <= num_pres:
        print('Number of absence records already <= number of presence. No need to equalise.')

    elif num_abse > num_pres:
        print('Reduced number of absence records from {0} to {1}'.format(num_abse, num_pres))
        df_absence = df_absence.sample(n=num_pres)
        result = df_absence

    else:
        raise ValueError('Could not equalise absence records.')
        return result
    
    #Return the result
    return result

        
def combine_pres_abse_records(df_presence, df_absence):
    """
    Combine pandas dataframes for presence and absence records. Adds new column of 1s and 0s 
    for pres/abse records.

    Parameters
    ----------
    df_presence: pandas dataframes
        A dataframe type containing values extracted from env variables for presence locations.
    df_absence: pandas dataframes
        A dataframe type containing values extracted from env variables for absence locations.

    Returns
    ----------
    df_pres_abse : pandas dataframes
        A dataframe containing both presence and absence records.
    """

    # notify user
    print('Combining presence and pseudo-absence point locations.')

    # check if presence is numpy rec array
    if not isinstance(df_presence, pd.DataFrame):
        raise TypeError('> Presence data not a pandas dataframe type.')
    elif not isinstance(df_absence, pd.DataFrame):
        raise TypeError('> Absence data not a pandas dataframe type.')

    try:
        # add new pres_abse column to both dataframes
        df_presence['pres_abse'] = 1
        df_absence['pres_abse'] = 0
        
        # combine dataframes
        df_pres_abse = df_presence.append(df_absence, 
                                          ignore_index=True)

    except:
        raise ValueError('Could not append presence/absence to dataframe.')

    # check if something came back
    if df_pres_abse.shape[0] == 0:
        raise ValueError('No presence/absence data was returned.')

    # notify user and return
    print('Combined total of {0} records.'.format(df_pres_abse.shape[0]))
    return df_pres_abse
   

def generate_correlation_matrix(df_records, rast_cate_list=None, show_fig=False, show_text=False):
    """
    Calculate a pearson correlation matrix and present correlation pairs. Optional 
    correlation matrix will be visualised if show_fig set to True. Categorical variables
    are excluded.
    
    How to interprete pearson correlation values:
        < 0.6 = No collinearity.
          0.6 - 0.8 = Moderate collinearity.
        > 0.8 = Strong collinearity. One of these predictors should be removed.

    Parameters
    ----------
    df_records : pandas dataframe
        A dataframe with rows (observations) and columns (continuous and categorical vars).
    rast_cate_list : list or string 
        A list or string of raster paths of continous and categorical layers.
    show_fig : bool
        If true, a correlation matrix will be plotted.
    show_text : bool
        If true, a correlation pair print out will be shown.
    """

    # do various checks
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Presence-absence data is not a dataframe type.')
    elif df_records.shape[0] == 0 or df_records.shape[1] == 0:
        raise ValueError('Presence-absence dataframe has rows and/or columns.')
    elif 'pres_abse' not in df_records:
         raise ValueError('No pres_abse column exists in dataframe.')
            
    # check raster categorical list
    if rast_cate_list is None:
        rast_cate_list = []
    elif not isinstance(rast_cate_list, (list, str)):
        raise TypeError('Raster categorical filepath list must be a list or single string.')

    # select presence records only
    df_presence = df_records.loc[df_records['pres_abse'] == 1]

    # check if any rows came back, if so, drop pres_abse column
    if df_presence.shape[0] != 0:
        df_presence = df_presence.drop(columns='pres_abse')
    else:
        raise ValueError('No presence records exist in pandas dataframe.')

    # iterate categorical names, drop if any exist, ignore if error
    if rast_cate_list is not None:
        for rast_path in rast_cate_list:
            cate_name = os.path.basename(rast_path)
            cate_name = os.path.splitext(cate_name)[0]
            df_presence = df_presence.drop(columns=cate_name, errors='ignore')

    try:
        # calculate correlation matrix and then correlation pairs
        cm = df_presence.corr()
        cp = cm.unstack()     

        # check if correlation matrix exist, plot if so
        if len(cm) > 0 and show_fig:
            print('Presenting correlation matrix.')

            # create figure, axis, matrix, colorbar
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm, interpolation='nearest', cmap='RdYlBu')
            fig.colorbar(cax)

            # create labels
            ax.set_xticks(np.arange(len(df_presence.columns)))
            ax.set_yticks(np.arange(len(df_presence.columns)))
            ax.set_xticklabels([''] + df_presence.columns.to_list(), rotation=90)
            ax.set_yticklabels([''] + df_presence.columns.to_list(), rotation=0)

            # show, add padding
            plt.show()
            print('\n')

        # check if correlation pairs exist, print if so
        if show_text:
            if len(cp) > 0:
                print('Presenting correlation pairs.')
                pd.set_option('display.max_rows', None)
                print(cp)

    except:
        raise ValueError('Could not generate and show correlation matrix results.')


def generate_vif_scores(df_records, rast_cate_list=None):
    """
    Calculate variance inflaction factor scores and presents them. Categorical variables
    are excluded.

    How to interperate vif scores:
        1 = No multicollinearity.
        1 - 5 = Moderate multicollinearity.
        > 5 = High multicollinearity.
        > 10 = This predictor should be removed from the model.

    Parameters
    ----------
    df_records : pandas dataframe
        A dataframe with rows (observations) and columns (continuous and categorical vars).
    rast_cate_list : list or string 
        A list or string of raster paths of continous and categorical layers.
    """
    
    # check dataframe
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Presence-absence data is not a dataframe type.')
    elif df_records.shape[0] == 0 or df_records.shape[1] == 0:
        raise ValueError('Presence-absence dataframe has rows and/or columns.')
    elif 'pres_abse' not in df_records:
         raise ValueError('> No pres_abse column exists in dataframe.')
            
    # check raster categorical list
    if rast_cate_list is None:
        rast_cate_list = []
    if not isinstance(rast_cate_list, (list, str)):
        raise TypeError('Raster categorical filepath list must be a list or single string.')

    # select presence records only
    df_presence = df_records.loc[df_records['pres_abse'] == 1]

    # check if any rows came back, if so, drop pres_abse column
    if df_presence.shape[0] != 0:
        df_presence = df_presence.drop(columns='pres_abse')
    else:
        raise ValueError('No presence records exist in pandas dataframe.')

    # iterate categorical names, drop if any exist, ignore if error
    for rast_path in rast_cate_list:
        cate_name = os.path.basename(rast_path)
        cate_name = os.path.splitext(cate_name)[0]
        df_presence = df_presence.drop(columns=cate_name, errors='ignore')

    try:
        # create empty dataframe
        df_vif_scores = pd.DataFrame(columns=['Variable', 'VIF Score'])

        # init a lin reg model
        linear_model = LinearRegression()

        # loop
        for col in df_presence.columns:

            # get response and predictors
            y = np.array(df_presence[col])
            x = np.array(df_presence.drop(columns=col, errors='ignore'))

            # fit lin reg model and predict
            linear_model.fit(x, y)
            preds = linear_model.predict(x)

            # calculate to coeff of determination
            ss_tot = sum((y - np.mean(y))**2)
            ss_res = sum((y - preds)**2)
            r2 = 1 - (ss_res / ss_tot)

            # set r2 == 1 to 0.9999 in order to prevent % by 0 error
            if r2 == 1:
                r2 = 0.9999

            # calculate vif 
            vif = 1 / (1 - r2)
            vif = vif.round(3)

            # add to dataframe
            vif_dict = {'Variable': str(col), 'VIF Score': vif}
            df_vif_scores = df_vif_scores.append(vif_dict, ignore_index=True) 

    except:
        raise ValueError('Could not generate and show vif results.')

    # check if vif results exist, print if so
    if len(df_vif_scores) > 0:
        print('- ' * 30)
        print(df_vif_scores.sort_values(by='VIF Score', ascending=True))
        print('- ' * 30)
        print('')


def create_estimator(estimator_type='rf', n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                     min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, 
                     verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
    """
    Create one of several types of tree-type statistical estimators. This essentially sets the parameters
    that are used by ExtraTrees or RandomForest regression to train and predict.
    
    Parameters
    ----------
    estimator_type : str {'rf', 'et'}
        Type of estimator to use. Random Forest is 'rf', ExtraTrees is 'et'. Random Forest typically 
        produces 'stricter' results (i.e. fewer mid-probability pixels)  than ExtraTrees.
    n_estimators : int
        The number of trees in the forest.
    criterion : str {'gini' or 'entropy'}
        Function to measure quality of a split. Use 'gini' for Gini impunity and 'entropy'
        for information gain. Default 'gini'.
    max_depth : int
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are 
        pure or until all leaves contain less than min_samples_split samples. Default is None.
    min_samples_split : int or float
        The minimum number of samples required to split an internal node. If int used, min_samples_split
        is the minimum number. See sklearn docs for different between int and float. Default is 2.
    min_samples_leaf : int or float
        The minimum number of samples required to be at a leaf node. A split point at any depth will 
        only be considered if it leaves at least min_samples_leaf training samples in each of the left 
        and right branches. This may have the effect of smoothing the model, especially in regression.
        See sklearn docs for different between int and float. Default is 1.
    min_weight_fraction_leaf : float
        The minimum weighted fraction of the sum total of weights (of all the input samples) required 
        to be at a leaf node. Samples have equal weight when sample_weight is not provided. Default 
        is 0.0.
    max_features : str {'auto', 'sqrt', 'log2'} or None,   
        The number of features to consider when looking for the best split. See sklearn docs for
        details on methods behind each. Default is 'auto'.
    max_leaf_nodes : int
        Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative 
        reduction in impurity. If None then unlimited number of leaf nodes. Default is None.
    min_impurity_decrease : float
        A node will be split if this split induces a decrease of the impurity greater than 
        or equal to this value. See sklearn docs for equations. Default is 0.0.
    bootstrap : bool
        Whether bootstrap samples are used when building trees. If False, the whole dataset is used 
        to build each tree. Default is False.
    oob_score : bool
        Whether to use out-of-bag samples to estimate the generalization accuracy. Default is False.
    n_jobs : int
        The number of jobs to run in parallel. Value of None is 1 core, -1 represents all. Default is
        None.
    random_state : int
        Controls 3 sources of randomness. See sklearn docs for each. Default is None.
    verbose : int
        Controls the verbosity when fitting and predicting. Default is 0.
    warm_start : bool
        When set to True, reuse the solution of the previous call to fit and add more estimators to 
        the ensemble, otherwise, just fit a whole new forest. Default is false.
    class_weight : str {'balanced', 'balanced_subsample'}
        Weights associated with classes in the form {class_label: weight}. If not given, all classes 
        are supposed to have weight one. For multi-output problems, a list of dicts can be provided in 
        the same order as the columns of y. See sklearn docs for more information. Default is None.
    ccp_alpha : float
        Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost 
        complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. Must
        be a positive value. Default is 0.0.
    max_samples : int or float
        If bootstrap is True, the number of samples to draw from X to train each base estimator. See
        sklearn docs for more information. Default is None.

    Returns
    ----------
    estimator : sklearn estimator object
        An estimator object.
    """
    
    # notify
    print('Creating species distribution model estimator.')
    
    # check parameters
    if estimator_type not in ['rf', 'et']:
        raise ValueError('Estimator type must be of type rf or et.')
    elif not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError('Num of estimators must be > 0.')
    elif criterion not in ['gini', 'entropy']:
        raise ValueError('Criterion must be of type gini or entropy.')
    elif (isinstance(max_depth, int) and max_depth <= 0) or isinstance(max_depth, (str, float)):
        raise ValueError('Max depth must be empty or > 0.')
    elif not isinstance(min_samples_split, (int, float)) or min_samples_split <= 0:
        raise ValueError('Min sample split must be numeric and > 0.')
    elif not isinstance(min_samples_leaf, (int, float)) or min_samples_leaf <= 0:
        raise ValueError('Min samples at leaf node must be numeric and > 0.')
    elif not isinstance(min_weight_fraction_leaf, float) or min_weight_fraction_leaf < 0.0:
        raise ValueError('Min weight fraction at leaf node must be numeric and >= 0.0.')
    elif max_features not in ['auto', 'sqrt', 'log2', None]:
        raise ValueError('Max features must be an empty or either: auto, sqrt, log2.')
    elif (isinstance(max_leaf_nodes, int) and max_leaf_nodes <= 0) or isinstance(max_leaf_nodes, (str, float)):
        raise ValueError('> Max leaf nodes must be empty or > 0.')
    elif not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0.0:
        raise ValueError('> Min impurity decrease must be a float and >= 0.0.')    
    elif not isinstance(bootstrap, bool):
        raise ValueError('> Boostrap must be boolean.')
    elif not isinstance(oob_score, bool):
        raise ValueError('> OOB score must be boolean.')    
    elif (isinstance(n_jobs, int) and n_jobs < -1) or isinstance(n_jobs, (str, float)):
        raise ValueError('> Num jobs must be > -1.')
    elif (isinstance(random_state, int) and random_state <= 0) or isinstance(random_state, (str, float)):
        raise ValueError('> Random state must be empty or > 0.')    
    elif not isinstance(verbose, int) or verbose < 0:
        raise ValueError('> Verbose >= 0.')
    elif not isinstance(warm_start, bool):
        raise ValueError('> Warm start must be boolean.')
    elif class_weight not in ['balanced', 'balanced_subsample', None]:
        raise ValueError('> Class weight must be an empty or either: balanced or balanced_subsample.')
    elif not isinstance(ccp_alpha, float) or ccp_alpha < 0.0:
        raise ValueError('> CCP Alpha must be a float and >= 0.0.')
    elif (isinstance(max_samples, (int, float)) and max_samples <= 0) or isinstance(max_samples, (str)):
        raise ValueError('> Max samples state must be empty or > 0.')
               
    # set estimator type
    try:
        estimator = None
        if estimator_type == 'rf':
            print('Setting up estimator for Random Forest.')
            estimator = rt(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, 
                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, 
                           verbose=verbose, warm_start=warm_start, class_weight=class_weight, 
                           ccp_alpha=ccp_alpha, max_samples=max_samples)
        elif estimator_type == 'et': 
            print('Setting up estimator for Extra Trees.')
            estimator = et(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, 
                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, 
                           verbose=verbose, warm_start=warm_start, class_weight=class_weight, 
                           ccp_alpha=ccp_alpha, max_samples=max_samples)
        else:
            raise ValueError('Selected estimator type does not exist. Please use Random Forest or Extra Trees.')
            
    except:
        raise ValueError('Could not create estimator. Is sklearn version correct?')
        
    # notify user and return
    print('Estimator created successfully.')
    return estimator


def get_dims_order_and_length(ds):
    """
    Read dataset and get order and length of x and y dimensions. 

    Parameters
    ----------
    ds: xarray dataset
        A dataset with x and y dimensions.

    Returns
    ----------
    dims_order : list
        A basic list with dataset's x and y ordering.
    dims_length : list
        A basic list with dims lengths in same order of dims_order
    """
    
    # check if dataset type provided
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Provided dataset is not an xarray dataset type. Please check input.')
        
    # get data array dimensions
    da_dims = ds.to_array().dims
    
    # get order and length based on index location
    if da_dims.index('y') < da_dims.index('x'):
        dims_order = ['y', 'x']
        dims_length = [len(ds['y']), len(ds['x'])]
        
    elif da_dims.index('y') > da_dims.index('x'):
        dims_order = ['x', 'y']
        dims_length = [len(ds['x']), len(ds['y'])]
        
    else:
        raise ValueError('Dimensions x and y ordering could not be determined.')
        
    # return
    return dims_order, dims_length


def generate_sdm(ds, df_records, estimator, rast_cont_list, rast_cate_list, 
                 replicates=5, test_ratio=0.1, equalise_test_set=False, 
                 shuffle_split=True, calc_accuracy_stats=False, plot_stats=True):
    """
    Generate a species distribution model (SDM) for given estimator and provided species occurrence
    points. Numerous parameters can be set to get more out of your data - see parameters below. Two or more
    iterations are required, and outputs are combined based on mean values. General approach based on
    based on RSGISLib SDM library (https://www.rsgislib.org/rsgislib_sdm.html).
    
    Parameters
    ----------
    ds : xarray dataset
        Xarray dataset holding categorical and/or continuous variables.
    df_records : pandas dataframe
        Pandas dataframe with presence/absence records.
    estimator : estimator obj
        A pre-defined estimator from sklearn. Can be either RandomForest or ExtraTrees. 
    rast_cont_list : list
        List of raster paths for continuous variables.
    rast_catest : list
        List of raster paths for categorical variables.
    replicates : int
        Number of times to perform an SDM. All results are combined at the end. Default is 5.
    test_ratio : float
        The ratio of which to split the presence/absence data in the df_records dataframe. Default is
        0.1 (i.e. 10% will be designated for testing).
    equalise_test_set : bool
        Reduce number of absence records to meet same number of presence. Imbalance can
        occur between presence and absence if large number of absence are generated prior.
    shuffle_split : bool
        When data is split into testing and training, shuffle the records randomly to reduce ordering 
        bias. Default is True.
    calc_accuracy_stats : bool
        Numerous accuracy metrics can be generated for each SDM and presented if set to True. These
        metrics include response curves, general classification and probability accuracy, and 
        others.
    plot_stats : bool
        Whether to show plots or only text based metrics. Default is True.

    Returns
    ----------
    df_result : xarray dataset
        A dataset with four variables - SDM mean, median, standard dev., variance.
    """
    
    # notify user
    print('Beginning species distribution modelling (SDM) process.')

    # check datasets
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Provided dataset is not an xarray dataset type. Please check input.')
    elif not isinstance(df_records, pd.DataFrame):
        raise TypeError('Presence/absence data array is not a pandas data frame.')
    elif estimator == None:
        raise ValueError('No estimator provided. Please create an estimator first.')
        
    # check rast cont list
    if rast_cont_list is None:
        rast_cont_list = []
    elif not isinstance(rast_cont_list, list):
        raise TypeError('Raster continuous paths is not a list.')
    
    # check rast cate list
    if rast_cate_list is None:
        rast_cate_list = []
    elif not isinstance(rast_cate_list, list):
        raise TypeError('Raster categorical paths is not a list.')        

    # check if valid estimator
    allowed_estimators = ['RandomForestClassifier()', 'ExtraTreeClassifier()', 'DecisionTreeClassifier()']
    if str(estimator.base_estimator) not in allowed_estimators:
        raise TypeError('Estimator must be a RandomForestClassifier, ExtraTreeClassifier or DecisionTreeClassifier.')

    # check parameters
    if not isinstance(test_ratio, float):
        raise TypeError('Test ratio must be a float.')
    elif test_ratio <= 0 or test_ratio >= 1:
        raise ValueError('Test ratio must be between 0 and 1.')
    elif not isinstance(replicates, int):
        raise TypeError('Replicates must be an integer.')
    elif not replicates > 1:
        raise ValueError('Replicates must be greater than 1.')
    elif not isinstance(rast_cate_list, list):
        raise TypeError('Raster categorical list is not a list.')
    elif not isinstance(shuffle_split, bool):
        raise TypeError('Shuffle split is not a boolean, must be True or False.')
    elif not isinstance(calc_accuracy_stats, bool):
        raise TypeError('Calculate accuracy statistics is not boolean, must be True or False.')
    elif not isinstance(equalise_test_set, bool):
        raise TypeError('Equalise test set value must be boolean (true or false).')          
        
    # get dim order and sizes, create empty ds, stack original ds
    dims_order, dims_length = get_dims_order_and_length(ds)
    ds_result = xr.zeros_like(ds).drop(list(ds.data_vars))
    ds = ds.stack(z=(dims_order))

    # generate matrices for predicting and plotting response curves
    if calc_accuracy_stats:

        # set up empty result dict list
        result_dict_list = []

        # generate lek matrices for response curves
        cont_matrices, cate_matrices = create_lek_matrices(df_records=df_records, 
                                                           rast_cate_list=rast_cate_list,
                                                           nodata_value=ds.nodatavals)

    # perform sdm for n replications
    for i in range(replicates):

        # notify
        print('Generating SDM replicate: {0} of {1}.'.format(i + 1, replicates))

        # split data in to train/test and x and y (predictors and response) sets 
        df_train, df_test = split_train_test(df_records=df_records, 
                                             test_ratio=test_ratio, 
                                             shuffle=shuffle_split, 
                                             equalise_test_set=equalise_test_set)
        
        # further split train and test data into x and y (predictors and response) sets         
        np_train_y, np_train_x = split_y_x(df_train)
        np_test_y, np_test_x = split_y_x(df_test)

        # fit model with training preds and response values
        estimator.fit(np_train_x, np_train_y)

        # calc probability of pres-abse and subset only positive probabilities
        y_prob = estimator.predict_proba(np_test_x)
        y_prob = y_prob[:, -1]

        # predict pres-abse probability, transpose, predict, drop
        absence_prob, presence_prob = estimator.predict_proba(ds.to_array().transpose()).T
        del absence_prob

        # reshape pres probabilities back to original
        presence_prob = presence_prob.reshape(dims_length)

        # add prediction output to result dataset, drop
        var_name = 'pred_{0}'.format(i + 1)
        ds_result[var_name] = xr.DataArray(presence_prob, dims=dims_order).astype('float32')
        del presence_prob

        # append result dicts to list
        if calc_accuracy_stats:

            # create empties
            result_dict = {}

            # store basic stats
            result_dict['y_true'] = np_test_y 
            result_dict['y_pred'] = estimator.predict(np_test_x)
            result_dict['y_prob'] = y_prob

            # generate and store importance scores and training accuracy
            result_dict['imp_scores'] = estimator.feature_importances_                           
            result_dict['train_acc'] = accuracy_score(estimator.predict(np_train_x), np_train_y)

            # generate and store roc curve info
            np_fpr, np_tpr, np_thresholds = generate_roc_curve(np_test_y, y_prob)
            result_dict['fpr'] = np_fpr
            result_dict['tpr'] = np_tpr

            # store predictions for response curves - continuous
            if cont_matrices:
                y_probs = []
                for m in cont_matrices:
                    y_probs.append(estimator.predict_proba(m)[:, -1])
                result_dict['cont_probs'] = y_probs

            # store predictions for response curves - categorical
            if cate_matrices:
                y_probs = []
                for m in cate_matrices:
                    y_probs.append(estimator.predict_proba(m)[:, -1])
                result_dict['cate_probs'] = y_probs

            # append result dict to dict list
            result_dict_list.append(result_dict)

    # notify
    print('SDM processing completed. Getting outputs in order.')

    # unstack dataset
    ds = ds.unstack()

    # prepare statistics if user requests
    if calc_accuracy_stats:
        try:
            # notify
            print('Preparing accuracy information.')

            # get all variable names
            all_var_names = list(ds.data_vars)

            # get continuous variable names
            cont_var_names = []
            for r in rast_cont_list:
                var = os.path.basename(r)
                var = os.path.splitext(var)[0]
                cont_var_names.append(var)

            # get categorical variable names
            cate_var_names = []
            for r in rast_cate_list:
                var = os.path.basename(r)
                var = os.path.splitext(var)[0]
                cate_var_names.append(var)

            # show mean viarbale importance (mvi) scores
            np_vis = np.array([d['imp_scores'] for d in result_dict_list])
            plot_mvi_scores(var_names=all_var_names, np_vis=np_vis)
            del np_vis

            # plot roc curve
            if plot_stats:
                np_fpr = np.array([d['fpr'] for d in result_dict_list])
                np_tpr = np.array([d['tpr'] for d in result_dict_list])
                plot_roc_curve(np_fpr, np_tpr)
                del np_fpr, np_tpr

            # show training out of the bag accuracy
            np_train_acc = np.array([d['train_acc'] for d in result_dict_list])
            plot_training_oob_accuracy(np_train_acc)
            del np_train_acc

            # extract all y true, prob and pred values into numpies
            np_y_true = np.array([d['y_true'] for d in result_dict_list])
            np_y_pred = np.array([d['y_pred'] for d in result_dict_list])
            np_y_prob = np.array([d['y_prob'] for d in result_dict_list])

            # concat all y true, prob and pred numpies
            np_y_true = np.concatenate(np_y_true).astype('uint8')
            np_y_pred = np.concatenate(np_y_pred).astype('uint8')
            np_y_prob = np.concatenate(np_y_prob).astype('float64')

            # calc accuracy metrics (prob and binary classification results) of presence
            calc_accuracy_metrics(np_y_true, np_y_prob, np_y_pred)
            del np_y_true, np_y_pred, np_y_prob
        
        except:
            print('Could not generate accuracy measurements. Aborting.')

        # plot continuous responses
        if plot_stats:
            cont_responses = np.array([d.get('cont_probs') for d in result_dict_list])
            if len(cont_responses) > 0 and len(cont_matrices) > 0:
                if len(cont_responses) == replicates:
                    try:
                        # get continuous variable names and pred values as numpies
                        x_names = np.array(cont_var_names)
                        x_values = np.array([cont_matrices[i][:, i] for i in range(len(cont_matrices))])

                        # get mean of each continuous var response
                        y_means = np.mean(cont_responses, axis=0)

                        # plot continuous responses
                        plot_continuous_response_curves(x_names, x_values, y_means, ncols=3)
                        del x_names, x_values, y_means

                    except:
                        print('> Could not plot continuous response curves. Aborting response curves.')
                else:
                    print('> Continous responses does not match number of replicates.')
            else:
                print('> Not enough information to plot continuous response curves.')

        # plot categorical responses
        if plot_stats:
            cate_responses = np.array([d.get('cate_probs') for d in result_dict_list])
            if len(cate_responses) > 0 and len(cate_matrices) > 0:
                if len(cate_responses) == replicates:
                    try:
                        # get categorical variable names
                        x_names = np.array(cate_var_names)

                        # get indexes for continuous and categoricals within full dataframe
                        cate_mask = np.array([True if v in cate_var_names else False for v in df_records.columns.tolist()])
                        cont_idx = np.where(cate_mask == False)[0]
                        cate_idx = np.where(cate_mask == True)[0]

                        # get unique class labels
                        x_values = np.array([cate_matrices[i][:, cate_idx[i]] for i in range(len(cate_matrices))])

                        # put code below 
                        plot_categorical_response_bars(x_names, x_values, cate_responses, ncols=3)
                        del x_names, cate_mask, cont_idx, cate_idx, x_values

                    except Exception as e:
                        print(e)
                        print('> Could not plot categorical response curves. Error occured. Moving on.')
                else:
                    print('> Continous responses does not match number of replicates.')
            else:
                print('> Not enough information to plot categorical response curves.')

    # create mask, replace with nan. replace 0 with very low 0 so not masked out   
    da_mask = xr.where(ds != ds.nodatavals, 1, 0)
    da_mask = da_mask.to_array(dim='mask').min('mask')
    ds_result = ds_result.where(da_mask, np.nan)
    ds_result = ds_result.where(ds_result != 0, 0.00001)  
    del da_mask

    # get mean, stdv, coeffs of variances into a final result dataset
    ds_result = xr.merge([
        ds_result.to_array(dim='mean').mean('mean').to_dataset(name='sdm_mean'),
        ds_result.to_array(dim='medn').median('medn').to_dataset(name='sdm_medn'),
        ds_result.to_array(dim='stdv').std('stdv').to_dataset(name='sdm_stdv'),
        ds_result.to_array(dim='cvar').var('cvar').to_dataset(name='sdm_cvar')
    ])

    # notify and return
    print('SDM process completed successfully.')
    return ds_result


def split_train_test(df_records, test_ratio=0.1, shuffle=True, equalise_test_set=False):
    """
    Split dataframe into a training and testing set by a user-specified ratio.
    The default is 10% of records set aside for testing.

    Parameters
    ----------
    df_records : pandas dataframe
        A pandas dataframe type containing values extracted from env variables for 
        presence and absence locations.
    test_ratio : float
        The ratio of samples that will be assigned as the test set e.g. 0.1 = 10% of 
        samples. Default is 0.1 (10%.)
    shuffle : bool
        Shuffle the dataset randomly to reduce bias.
    presence_only : bool
        Limit test set to presence records only to presence like MaxEnt.
    equalise_test_set : bool
        Reduce number of absence records to meet same number of presence. Imbalance can
        occur between presence and absence if large number of absence are generated prior.
   
    Returns
    ----------
    df_train : pandas dataframe
        A pandas dataframe containing training set.
    df_test : pandas dataframe
        A pandas dataframe containing testing set.
    """

    # check if we have a numpy record array
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Presence/absence dataframe is not a pandas dataframe type.')
    elif not isinstance(test_ratio, float):
        raise TypeError('Test ratio must be a float.')
    elif test_ratio <= 0 or test_ratio >= 1:
        raise ValueError('Test ratio must be between 0 and 1.')

    
    # prepare stratify option - keeps testing 1s and 0s in proportion to full dataset
    df_stratify = df_records['pres_abse']
        
    # split in to train and test, with ratio, and shuffle
    df_train, df_test = train_test_split(df_records, 
                                         test_size=test_ratio, 
                                         shuffle=shuffle, 
                                         stratify=df_stratify)
    
    # equalise records if requested (make num absence = num presence)
    if equalise_test_set:
        # slice test set into presence and absence sets
        df_test_p = df_test[df_test['pres_abse'] == 1]
        df_test_a = df_test[df_test['pres_abse'] == 0]

        # reduce num absence recocrds if lower than presence
        if (df_test_p.shape[0] < df_test_a.shape[0]):    
            
            # sample randomly to match presence num
            df_test_a_sample = df_test_a.sample(n=df_test_p.shape[0])
            
            # get absence records that were not sampled
            df_merged = df_test_a.merge(df_test_a_sample, how='left', indicator=True)
            df_merged = df_merged.query('_merge == "left_only"').drop(['_merge'], axis=1)
            
            # append omitted absence records back on to train set
            df_train.append(df_merged, ignore_index=True)
                        
            # merge test presence and absence subset records together
            df_test = df_test_p.append(df_test_a, ignore_index=True)
            
    return df_train, df_test


def split_y_x(df_records, y_column='pres_abse'):
    """
    Read dataset and split into response (pres_abse values) and predictor (env raster values) 
    variables. Required for ExtraTrees/RandomForest regression.

    Parameters
    ----------
    df_records: pandas dataframe
        A single dataframe with variable names and values extracted from raster.
    y_column: string
        The name of the column that holds presence and absence binary values (1s and 0s). 
        Default is pres_abse.

    Returns
    ----------
    np_y :numpy array
        A numpy array with binary 1s and 0s of species presence.
    np_x : numpy array
        A numpy array with env variable values.
    """
    
    # check for records array
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Presence/absence dataframe is not a pandas dataframe.')
    elif not isinstance(y_column, str):
        raise TypeError('Response (y) column name must be a string.')
    elif y_column not in df_records:
        raise ValueError('There is no column called {0} in dataframe.'.format(y_column))

    try:
        # get response (y) and predictors (x) as numpy arrays
        np_y = np.array(df_records[y_column].astype('int8'))
        np_x = np.array(df_records.drop(columns=y_column, errors='ignore'))

    except:
        raise ValueError('Could not seperate response from predictors.')

    return np_y, np_x


def create_lek_matrices(df_records, rast_cate_list, remove_col='pres_abse', nodata_value=-999):
    """
    Builds lek matrices required to generate response curves. Based on Lek et al. 1995, 1996. 
    Original code was implemented in paper Roberts et al. (2020). Sensitivity Analysis of the 
    DART Model for Forest Mensuration with Airborne Laser Scanning. Remote Sensing, 12(2), 247.
    
    Parameters
    ----------
    df_records : pandas dataframe
        Dataframe of presence and absence records.
    rast_cast_list : list
        List of categorical raster layers used in model.
    remove_col : str
        Name of the presence/absence column. Default is 'pres_abse'.
    nodata_value : int
        NoData value within dataset. Default is -9999.
        
    Returns
    ----------
    cate_matrices : numpy array
        A array of categorical matrices.
    cont_matrices : numpy array
        A array of continuous matrices.
        
    """
        
    # check if df records is pandas dataframe
    if not isinstance(df_records, pd.DataFrame):
        raise TypeError('Records are not in pandas dataframe type.')
    elif not isinstance(rast_cate_list, list):
        raise TypeError('Raster categorical list is not a list.')
    elif not isinstance(remove_col, (str)):
        raise TypeError('Remove column value is not a string.')
    elif not isinstance(nodata_value, (int, float)):
        raise TypeError('NoData value is not a number.')

    # drop presence/absence column, get col names, and num cols
    df_vars = df_records.drop(columns=remove_col, errors='ignore')
    col_names = np.array(df_vars.columns.tolist())
    num_cols = col_names.size

    # get categorical names if exist
    cate_col_names = []
    for r in rast_cate_list:
        col_name = os.path.splitext(os.path.basename(r))[0]
        cate_col_names.append(col_name)

    # create index arrays for categorical and continous
    cate_mask = np.array([True if var in cate_col_names else False for var in col_names])
    cont_idx = np.where(cate_mask == False)[0]
    cate_idx = np.where(cate_mask == True)[0]

    # get continus values as a numpy with no nodata
    x = df_vars[col_names[cont_idx]].to_numpy()
    x = np.ma.masked_equal(x, nodata_value)

    # calculate values for generating response matrix for continuous variables
    cont_values = []
    for percentile in np.arange(1, 100, 1, dtype='uint8'):
        cont_values.append(np.percentile(x, percentile, axis=0))

    # convert vontinuous values to array and transpose
    cont_values = np.array(cont_values).T

    # calculate mean of each predictor
    nominal_values = np.array([np.mean(x, axis=0)])

    # get the unique and modal values for categorical vars
    try:
        if cate_idx.size > 0:
            cate_values, modal_classes = [], []

            # loop each categorical var and get unique and modal (max) values 
            for i in cate_idx:
                x = df_vars[col_names[i]].to_numpy()
                x = x[x != nodata_value] 
                clf, counts = np.unique(x, return_counts=True)
                cate_values.append(clf)
                modal_classes.append(clf[np.argmax(counts)])

            # convert to numpy
            cate_values = np.array(cate_values)
            modal_classes = np.array(modal_classes)

        # create Lek matrices for each categorical variable
        cate_matrices = []
        if cate_idx.size > 0:

            # loop each categorical array
            for idx, cate_predictor in enumerate(cate_values):
                matrix = []

                # loop each unique value in predictor
                for unique_clf in cate_predictor:
                    for constants in nominal_values:
                        x = np.copy(constants)

                        if idx == 0 and cate_idx.size == 1:
                            # insert unique class
                            x = np.insert(x, cate_idx[idx], unique_clf)

                        elif idx == 0 and cate_idx.size != 1:
                            # insert unique class
                            curr_idx = cate_idx[idx]
                            x = np.insert(x, obj=curr_idx, values=unique_clf)

                            # insert modal values for subsequent categorical variables (in order)
                            remaining_idxs, remaining_classes = cate_idx[idx + 1:], modal_classes[idx + 1:]
                            for r_indx, r_class in zip(remaining_idxs, remaining_classes):
                                x = np.insert(x, obj=r_indx, values=r_class)

                        else:
                            # insert modal values for prior categorical variables (in order)
                            prior_idx, prior_classes = cate_idx[:idx], modal_classes[:idx]
                            for p_indx, p_class in zip(prior_idx, prior_classes):
                                x = np.insert(x, p_indx, p_class)

                            # insert unique class:
                            x = np.insert(x, cate_idx[idx], unique_clf)

                        # add to matrix if equal to predictor, pad with mean of array if size not right
                        if x.size == num_cols:
                            matrix.append(x)

                        elif x.size != num_cols:
                            diff = abs(x.size - num_cols)
                            for v in range(0, diff, 1):
                                x = np.append(x, np.array([np.mean(x)]))
                            matrix.append(x)

                        else:
                            raise ValueError('Unable to create Lek matrix for generating response curves.')

                # transpose, clean up and append to list
                matrix = np.array(matrix).T.round(decimals=10)
                cate_matrices.append(matrix.T)

        # create Lek matrices for each continuous variable
        cont_matrices = []
        for idx, scale in enumerate(cont_values):
            matrix = []

            for element in scale:
                for constants in nominal_values:
                    x = constants.copy()

                    # insert element from the predictor variable to be varied
                    x[cont_idx[idx]] = element

                    if cate_idx.size != 0:

                        # insert modal classes of categorical variables
                        for i, clf in enumerate(modal_classes):
                            x = np.insert(x, cate_idx[i], clf)

                    if x.size == num_cols:
                        matrix.append(x)

            # transpose, clean up and append to list
            matrix = np.array(matrix).T.round(decimals=10)
            cont_matrices.append(matrix.T)
    
    except:
        raise ValueError('Unable to generate Continuous or Categorical matrices')

    # return
    return cont_matrices, cate_matrices

        
def plot_mvi_scores(var_names, np_vis):
    """
    Plot the mean variable importance (MVI) scores.

    Parameters
    ----------
    var_names : list
        A list of env variables in dataset. 
    np_vis : numpy array
        A array of importance scores from modelling.
    """

    # checks
    if not isinstance(var_names, list):
        raise TypeError('The var names variable is not a list.')
    elif not var_names:
        raise TypeError('The var names variable is empty.')
    elif not isinstance(np_vis, np.ndarray):
        raise TypeError('The vi scores is not a numpy array.')
    elif np_vis.shape[0] <= 1:
        raise TypeError('The vi scores variable must have more than 1 model iteration.')    

    # get mean vi scores
    np_vis_mean = np.mean(np_vis, axis=0)

    # create df, fill with mean vi scores, sort 
    df_vis_mean = pd.DataFrame(columns=['Variable', 'Mean Importance Score'])
    for var, val in zip(var_names, np_vis_mean):
        vis_dict = {'Variable': var, 'Mean Importance Score': val}
        df_vis_mean = df_vis_mean.append(vis_dict, ignore_index=True)

    # check if vis results exist, print if so
    if len(df_vis_mean) > 0:
        print('- ' * 30)
        print(df_vis_mean.sort_values(by='Mean Importance Score', ascending=False))
        print('- ' * 30)
        print('')
    else:
        print('- - - - ERROR: No results to show - - - -')
        print('')
        

def plot_training_oob_accuracy(np_train_accuracy):
    """
    Basic function to display mean training accuracy (OOB).

    Parameters
    ----------
    np_training_accuracy : numpy array
        A array of training accuracy values from modelling.
    """

    # check train accuracy array
    if not isinstance(np_train_accuracy, np.ndarray):
        raise TypeError('> The training accuracy is not a numpy array.')
    elif np_train_accuracy.shape[0] <= 1:
        raise TypeError('> The training accuracy variable must have more than 1 model iteration.')    

    # get mean train accuracy
    train_accuracy_mean = float(np.mean(np_train_accuracy).round(decimals=3))

    # check if train accuracy mean results exist, print if so
    if train_accuracy_mean:
        print('- ' * 30)
        print('Training Out-Of-Bag (OOB) Accuracy:\t\t{0}'.format(train_accuracy_mean))
        print('- ' * 30)
        print('')
    else:
        print('- - - - ERROR: No results to show - - - -')
        print('')
        
    
def generate_roc_curve(y_true, y_prob, num_retain=101):
    """
    Generate reciever operator characteristic (roc) curve via known presence/absence values 
    (y_true) and target scores (probability estimates from a trained model). Uses
    sklearn roc curve library.

    Parameters
    ----------
    y_true : numpy array
        A 1d numpy array with known 1s and 0s (typically from test set).
    y_prob : numpy array
        A 1d numpy array with predicted probabilities via trained model for 
        corresponding test set.
    num_retain : int
        The number of measurements to retain. Default is 101.
        
    Returns
    ----------
    np_fpr_out : numpy array
        Array of false positive rate values.
    np_tpr_out : numpy array
        Array of true positive rate values.
    np_thresholds : numpy array
        Decreasing thresholds on the decision function used to compute 
        fpr and tpr.
    """
    
    # check y_true type
    if not isinstance(y_true, np.ndarray):
        raise TypeError('The y true values are not of type numpy array.')
    elif not isinstance(y_prob, np.ndarray):
        raise TypeError('The y score values are not of type numpy array.')
    elif y_true.ndim  != 1 and y_prob.ndim  != 1:
        raise ValueError('The y true and/or y score arrays are not 1-dimensional.')
               
    try:
        # generate roc curve
        np_fpr, np_tpr, np_thresholds = roc_curve(y_true, y_prob)
        
    except:
        raise ValueError('Could not calculate roc curve.')
        
    # check if anything came out
    if len(np_fpr) == 0 or len(np_tpr) == 0 or len(np_thresholds) == 0:
        raise ValueError('Nothing came out of roc curve.')
        
    # check if lengths are equal
    if np_fpr.shape[0] != np_tpr.shape[0]:
        raise ValueError('The fpr and tpr arrays are not equal in size.')
        
    # create new range of values for x
    np_fpr_out = np.linspace(0, 1.0, num_retain)

    # linearly interp new tpr values at each value in fpr_new
    np_tpr_out = np.interp(np_fpr_out, np_fpr, np_tpr)
    
    # return
    return np_fpr_out, np_tpr_out, np_thresholds


def plot_roc_curve(np_fpr, np_tpr):
    """
    Plot the mean reciever operator characteristic (roc) curve.

    Parameters
    ----------
    np_fpr : numpy array of arrays 
        A array of arrays holding fpr values of multiple model runs.
    np_tpr : numpy array of arrays 
        A array of arrays holding tpr values of multiple model runs.
    """
    
    # checks
    if not isinstance(np_fpr, np.ndarray):
        raise TypeError('> The fpr variable is not a numpy array.')
    elif not isinstance(np_tpr, np.ndarray):
        raise TypeError('> The tpr variable is not a numpy array.')
    elif np_fpr.shape[0] <= 1:
        raise TypeError('> The fpr variable must have more than 1 model iteration.')    
    elif np_tpr.shape[0] <= 1:
        raise TypeError('> The tpr variable must have more than 1 model iteration.')
        
    # set up broad figure and plot
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(4.25, 3.75))
    plt.tight_layout(w_pad=0, h_pad=0)
    
    # set up tick sizes
    ax.get_yaxis().set_tick_params(which='major', direction='out')
    ax.get_xaxis().set_tick_params(which='major', direction='out')
    ax.get_xaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
    ax.get_yaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    # set up border lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    
    # get mean fpr and tpr arrays
    np_fpr_mean = np.mean(np_fpr, axis=0)
    np_tpr_mean = np.mean(np_tpr, axis=0)
    
    # add mean roc line to plot
    ax.plot(np_fpr_mean, np_tpr_mean, c='blue', linestyle='-', lw=1.5, label='Mean ROC')
    
    # plot 1 standard deviation
    np_tpr_std = np.std(np_tpr, axis=0)
    np_tpr_lower = np_tpr_mean - np_tpr_std
    np_tpr_upper = np_tpr_mean + np_tpr_std
    
    # add fill for standard deviation area
    ax.fill_between(np_fpr_mean, np_tpr_lower, np_tpr_upper, alpha=0.5, edgecolor='silver',
                    lw=0.0, facecolor='silver', label='$\pm$ 1 St.Dev')
    
    # clean up axis sizes, add 0.5 line, label axes
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.plot(ax.get_xlim(), ax.get_ylim(),  c='k', ls=':', lw=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    
    # add legend
    legend = ax.legend(loc='lower right', fancybox=True, frameon=True, ncol=1, handlelength=0.75, fontsize=9)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor('k')
    
    # show
    print('- ' * 30)
    plt.show()
    print('- ' * 30)
    print('')
    

def calc_accuracy_metrics(np_y_true, np_y_prob, np_y_pred):
    """
    Calculates and displays accuracy metrics for both probability and
    binary classification results.
    
    Parameters
    ----------
    np_y_true : numpy array
        A array holding response labels of multiple model runs.
    np_y_prob : numpy array
        A array holding probabilities of multiple model runs. 
    np_y_pred : numpy array
        A array holding predictions of multiple model runs.  
    """
    
    # checks
    if not isinstance(np_y_true, np.ndarray):
        raise ValueError('> The response labels are not a numpy array.')
    elif not isinstance(np_y_prob, np.ndarray):
        raise ValueError('> The response probabilities are not a numpy array.')
    elif not isinstance(np_y_pred, np.ndarray):
        raise ValueError('> The response predictions are not a numpy array.')
    elif np_y_true.ndim != 1:
        raise ValueError('> The response labels are not 1-dimensional.')
    elif np_y_prob.ndim != 1:
        raise ValueError('> The response probabilities are not 1-dimensional.')
    elif np_y_pred.ndim != 1:
        raise ValueError('> The response predictions are not 1-dimensional.')
    elif np_y_true.size != np_y_pred.size and np_y_prob != np_y_pred:
        raise ValueError('> The arrays do not have the same sizes.')
    elif np.min(np_y_true) != 0 or np.max(np_y_true) != 1:
        raise ValueError('> The response labels are not binary 0 and 1.')
    elif np.min(np_y_pred) != 0 or np.max(np_y_pred) != 1:
        raise ValueError('> The response predictions are not binary 0 and 1.')

    # remove any nan or infinite values from y true labels
    bad_idx = np.isfinite(np_y_true)
    np_y_true = np_y_true[bad_idx]
    np_y_prob = np_y_prob[bad_idx]
    np_y_pred = np_y_pred[bad_idx]

    # remove any nan or infinite values from y prob labels
    bad_idx = np.isfinite(np_y_prob)
    np_y_true = np_y_true[bad_idx]
    np_y_prob = np_y_prob[bad_idx]
    np_y_pred = np_y_pred[bad_idx]

    # remove any nan or infinite values from y pred labels
    bad_idx = np.isfinite(np_y_pred)
    np_y_true = np_y_true[bad_idx]
    np_y_prob = np_y_prob[bad_idx]
    np_y_pred = np_y_pred[bad_idx]

    # calculates area under roc curve (auc)
    roc_auc = roc_auc_score(np_y_true, np_y_prob)

    # calculates area under the precision-recall curve
    avg_precision = average_precision_score(np_y_true, np_y_prob)

    # calculates brier loss score 
    brier_score = brier_score_loss(np_y_true, np_y_prob)

    # calculates log loss score
    log_loss_score = log_loss(np_y_true, np_y_prob)

    # calculates point biserial correlation coefficient (r value) between known and pred.
    r_val = pointbiserialr(np_y_true, np_y_prob)[0]

    try:
        # get presence and absence
        num_pres = np.where(np_y_true == 1)[0].size
        num_abse = np.where(np_y_true == 0)[0].size

        # calculate true-pres, true-absence, false-pres, false-abse
        tp = np.where((np_y_true == 1) & (np_y_pred == 1))[0].size
        ta = np.where((np_y_true == 0) & (np_y_pred == 0))[0].size
        fp = np.where((np_y_true == 1) & (np_y_pred == 0))[0].size
        fa = np.where((np_y_true == 0) & (np_y_pred == 1))[0].size

        # proportion of presence records
        prevalence = (tp / fa) / np_y_true.size

        # proportion of absence records
        odp = 1 - prevalence

        # correct classification / misclassification rate
        ccr = (tp + ta) / np_y_true.size
        mcr = (fp + fa) / np_y_true.size

        # sensitivity (or recall)
        sensitivity = tp / num_pres

        # presence / absence predictive power
        p_pp = tp / (tp + fp)
        a_pp = ta / (ta + fa)

        # specificity
        specificity = ta / num_abse

        # accuracy
        accuracy = (tp + ta) / (num_pres + num_abse)
        balanced_accuracy = ((tp / num_pres) + (ta / num_abse)) / 2

        # precision
        precision = tp / (tp + fp)

        # f1 score
        f1s = f1_score(np_y_true, np_y_pred)

        # matthews correlation coefficient
        mcc = matthews_corrcoef(np_y_true, np_y_pred)

        # normalised mutual info score
        nmi_score = normalized_mutual_info_score(np_y_true, np_y_pred)

        # kappa
        kappa = cohen_kappa_score(np_y_true, np_y_pred)

    except Exception as e:
        raise ValueError('> Could not calculate classification accuracy metrics.')

    # show auc result
    print('- - Probabilities Accuracy Metrics  ' + '- ' * 12)
    print('Area Under the ROC Curve (AUC).')
    print('Terrible = 0.5 | Moderate = 0.75 | Perfect = 1.0')
    print('AUC: {0}'.format(roc_auc.round(3)) + '\n') 

    # show precision recall result
    print('Area Under the Precision Recall Curve (PR Score).')
    print('Terrible = 0.0 | Perfect = 1.0.')
    print('PR Score: {0}'.format(avg_precision.round(3)) + '\n')

    # show brier score result
    print('Brier Loss Score.')
    print('Terrible: 1.0 | Perfect: 0.0 (lower score is better).')
    print('Brier Loss Score: {0}'.format(brier_score.round(3)) + '\n')

    # show log-loss result
    print('Log-Loss Score (stricter than Brier Loss Score).')
    print('Perfect: 0.0 (lower score is better).')
    print('Log-Loss Score: {0}'.format(log_loss_score.round(3)) + '\n')

    # show point biserial correlation coefficient (r value)
    print('Point-biserial Correlation Coefficient (R-value).')
    print('Perfect + relationship = 1 | Perfect - relationship = -1 | No relationship = 0. ')
    print('R-value: {0}'.format(r_val.round(3)))
    print('- ' * 30)
    print('')

    print('- - Classification Accuracy Metrics ' + '- ' * 12)
    print('Proportion of presence records: {0}'.format(round(prevalence, 3)))
    print('Proportion of absence records: {0}'.format(round(odp, 3)))
    print('Correct classification rate: {0}'.format(round(ccr, 3)))
    print('Misclassification rate: {0}'.format(round(mcr, 3)))
    print('Sensitivity: {0}'.format(round(sensitivity, 3)))
    print('Presence predictive power: {0}'.format(round(p_pp, 3)))
    print('Absence predictive power: {0}'.format(round(a_pp, 3)))
    print('Specificity: {0}'.format(round(specificity, 3)))
    print('Accuracy: {0}'.format(round(accuracy, 3)))
    print('Accuracy (balanced): {0}'.format(round(balanced_accuracy, 3)))
    print('Precision: {0}'.format(round(precision, 3)))
    print('F1 Score: {0}'.format(round(f1s, 3)))
    print('Matthew Correlation Coefficient: {0}'.format(round(mcc, 3)))
    print('Normalised Mutual Info Score: {0}'.format(round(nmi_score, 3)))
    print('Kappa: {0}'.format(round(kappa, 3)))
    print('- ' * 30)
    print('')
    

def plot_continuous_response_curves(x_names, x_values, y_means, ncols=3):
    """
    Read response and predictor arrays from lek matrices and plot response curves for continuous
    variables.

    Parameters
    ----------
    x_names: numpy array
        A numpy array of continuous predictor variable names.
    x_values: numpy array
        A numpy array of continuous predictor variable values.
    y_means: numpy array
        A numpy array of continuous response value means.
    n_cols : int
        Number of columns on plot.
    """

    # check predictor names type and dimensionality
    if not isinstance(x_names, np.ndarray) and x_names.ndim != 1:
        raise TypeError('> Predictor variable names not in numpy array and/or not 1-dimensional.')
    elif not isinstance(x_values, np.ndarray) and not x_values.ndim >= 1:
        raise TypeError('> Predictor variable values not in numpy array and/or not 1 or 2-dimensional.')
    elif not isinstance(y_means, np.ndarray) and not y_means.ndim >= 1:
        raise TypeError('> Response means not in numpy array and/or not 1 or 2-dimensional.')   
    elif x_values.shape[0] != len(x_names) or y_means.shape[0] != len(x_names):
        raise TypeError('> Unequal number of predictors was provided.')
    elif x_values.shape[1] != y_means.shape[1]:
        raise TypeError('> Unequal number of elements provided.')
    elif not isinstance(ncols, int) or ncols < 1:
        raise ValueError('> Number of columns must be an integer greater than 0.')

    # get number of continuous variables
    num_vars = len(x_names)

    # prepare number of rows
    nrows = math.ceil(num_vars / ncols)

    try:
        # create height
        height = 3 * nrows

        # create top level figure and subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10, height))
        plt.tight_layout(w_pad=2, h_pad=2)

        # add title
        fig.suptitle('Response Curves (Continuous Variables)', y=1.025, fontsize='x-large', fontweight='bold')

        i = 0
        for r in range(nrows):
            for c in range(ncols):

                # config axis depending on num rows
                if nrows == 1:
                    axis = ax[c]
                else:
                    axis = ax[r, c]

                # plot subplot if data exists
                if i < len(x_names):

                    # set axis ticks, formatting
                    axis.get_yaxis().set_tick_params(which='major', direction='out')
                    axis.get_xaxis().set_tick_params(which='major', direction='out')
                    axis.get_xaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
                    axis.get_yaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
                    axis.get_xaxis().tick_bottom()
                    axis.get_yaxis().tick_left()
                    axis.spines['top'].set_visible(False)
                    axis.spines['right'].set_visible(False)
                    axis.xaxis.set_tick_params(width=0.5)
                    axis.yaxis.set_tick_params(width=0.5)

                    # set labels and y limits 0-1
                    axis.set_ylim((0, 1))
                    axis.set_xlabel(x_names[i], fontsize=9)
                    axis.set_ylabel('Probability of Presence', fontsize=9)

                    # plot the subplot
                    axis.plot(x_values[i], y_means[i], c='seagreen')

                else:
                    # create empty subplot
                    axis.axis('off')

                # increase counter    
                i += 1

    except:
        raise ValueError('Could not plot continuous variables. Skipping plot.')

        
def plot_categorical_response_bars(x_names, x_values, y_responses, ncols=3):
    """
    Read response and predictor arrays from lek matrices and plot response curves for categorical
    variables.

    Parameters
    ----------
    x_names: numpy array
        A numpy array of continuous predictor variable names.
    x_values : numpy array
        A numpy array of unique predictor values (i.e. unique categorical classes).
    y_responses: numpy array
        A numpy array of categorical probability responses.
    n_cols : int
        Number of columns on plot.
    """

    # checks
    if not isinstance(x_names, np.ndarray) and x_names.ndim != 1:
        raise TypeError('> Predictor variable names not in numpy array and/or not 1-dimensional.')
    elif not isinstance(x_values, np.ndarray) and not x_values.ndim >= 1:
        raise TypeError('> Class labels not in numpy array and/or not 1 or 2-dimensional.')
    elif not isinstance(y_responses, np.ndarray) and not y_responses.ndim >= 1:
        raise TypeError('> Response means not in numpy array and/or not 1 or 2-dimensional.')
    elif not isinstance(ncols, int) or ncols < 1:
        raise ValueError('> Number of columns must be an integer greater than 0.')

    # get number of continuous variables
    num_vars = len(x_names)

    # prepare number of rows
    nrows = math.ceil(num_vars / ncols)

    try: 
        # create height
        height = 3 * nrows

        # create top level figure and subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10, height))
        plt.tight_layout(w_pad=2, h_pad=2)

        # add title
        fig.suptitle('Response Curves (Categorical Variables)', y=1.025, fontsize='x-large', fontweight='bold')

        i = 0
        for r in range(nrows):
            for c in range(ncols):

                # config axis depending on num rows
                if nrows == 1:
                    axis = ax[c]
                else:
                    axis = ax[r, c]

                if i < len(x_names):
                    response = []
                    for y_response in y_responses:
                        response.append(y_response[i])

                    # get mean of each continuous var
                    y_means = np.mean(response, axis=0)

                    # set axis ticks, formatting
                    axis.get_yaxis().set_tick_params(which='major', direction='out')
                    axis.get_xaxis().set_tick_params(which='major', direction='out')
                    axis.get_xaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
                    axis.get_yaxis().set_tick_params(which='minor', direction='out', length=0, width=0)
                    axis.get_xaxis().tick_bottom()
                    axis.get_yaxis().tick_left()
                    axis.spines['top'].set_visible(False)
                    axis.spines['right'].set_visible(False)
                    axis.xaxis.set_tick_params(width=0.5)
                    axis.yaxis.set_tick_params(width=0.5)

                    # set bar width and create
                    bar_width = 0.5
                    x_loc = np.arange(x_values[i].size) - (bar_width * 0.125)
                    axis.bar(x_loc, y_means, color='seagreen', width=bar_width, lw=0.25, edgecolor='k',
                             tick_label=x_values[i], error_kw=dict(lw=0.35, capsize=1.25, capthick=0.25))

                    # set labels and y limits 0-1, class labels
                    axis.set_ylim((0, 1))
                    axis.set_xlabel(x_names[i], fontsize=9)
                    axis.set_ylabel('Probability of Presence', fontsize=9)
                    axis.set_xticklabels(x_values[i], rotation=35, horizontalalignment='right', verticalalignment='top', fontsize=7.5)

                else:
                    # create empty subplot
                    axis.axis('off')

                # increase counter    
                i += 1

    except:
        raise ValueError('Could not plot continuous variables. Skipping plot.')
