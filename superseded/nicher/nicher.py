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