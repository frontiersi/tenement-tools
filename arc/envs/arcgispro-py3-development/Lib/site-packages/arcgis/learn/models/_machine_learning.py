import os
import random
import json
import pickle
import warnings
import math
import tempfile
from pathlib import Path
from zipfile import ZipFile

import arcgis
from arcgis.features import FeatureLayer

from .._utils.tabular_data import TabularDataObject

HAS_SK_LEARN = True
try:
    import sklearn
    from sklearn import *
except:
    HAS_SK_LEARN = False

HAS_FAST_PROGRESS = True
try:
    from fastprogress.fastprogress import progress_bar
except:
    HAS_FAST_PROGRESS = False

_PROTOCOL_LEVEL = 2


def _get_model_type(model_type):
    if not model_type:
        raise Exception("Invalid model type.")

    if not isinstance(model_type, str):
        return model_type

    if not model_type.startswith('sklearn.'):
        raise Exception("Invalid model_type.")

    model_type = model_type.replace('sklearn.', '')

    module = model_type.split('.')[0]

    if len(model_type.split('.')) > 1:
        model = model_type.split('.')[1]
    else:
        raise Exception("Invalid model_type.")

    if not hasattr(sklearn, module) or \
            not hasattr(getattr(sklearn, module), model):
        raise Exception("Invalid model_type.")

    model = getattr(getattr(sklearn, module), model)

    return model


def raise_data_exception():
    raise Exception("Cannot call this function without data.")


class MLModel(object):
    """
    Creates a machine learning model based on its implementation from scikit-learn.
    For supervised learning:
    Refer https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    For unsupervised learning:
    1. Clustering Models
    2. Gaussian Mixture Models
    3. Novelty and outlier detection
    Refer https://scikit-learn.org/stable/unsupervised_learning.html

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required TabularDataObject. Returned data object from
                            `prepare_tabulardata` function.
    ---------------------   -------------------------------------------
    model_type              Required string path to the module.
                            For example for SVM:
                                sklearn.svm.SVR or sklearn.svm.SVC
                            For tree:
                                sklearn.tree.DecisionTreeRegressor or sklearn.tree.DecisionTreeClassifier
    ---------------------   -------------------------------------------
    **kwargs                model_type specific arguments.
                            Refer Parameters section
                            https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    =====================   ===========================================

    :returns: `MLModel` Object
    """

    def __init__(self, data, model_type, **kwargs):
        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        self._data = data
        self._training_data, self._training_labels, self._validation_data, self._validation_labels = self._data._ml_data
        if kwargs.get('pretrained_model'):
            self._model = kwargs.get('pretrained_model')
        else:
            model = _get_model_type(model_type)

            if model == sklearn.cluster._kmeans.KMeans:
                if not kwargs.get('n_clusters'):
                    kwargs['n_clusters'] = self._get_number_of_clusters(**kwargs)
            elif model == sklearn.mixture._gaussian_mixture.GaussianMixture:
                if not kwargs.get('n_components'):
                    kwargs['n_components'] = self._get_number_of_components(**kwargs)

            self._model = model(**kwargs)

    def fit(self):
        if (not self._data._is_unsupervised and (self._training_data is None or self._training_labels is None))\
                or (self._data._is_unsupervised and self._training_data is None):
            raise_data_exception()

        if self._data._is_unsupervised:
            self._model.fit(self._training_data)
        else:
            self._model.fit(self._training_data, self._training_labels)

    def show_results(self, rows=5):
        """
        Shows sample results for the model.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional number of rows. By default, 5 rows
                                are displayed.
        =====================   ===========================================
        :returns dataframe
        """
        if (not self._data._is_unsupervised and (self._validation_data is None or self._validation_labels is None)) \
                or (self._data._is_unsupervised and self._validation_data is None):
            raise_data_exception()

        min_size = len(self._validation_data)

        if rows < min_size:
            min_size = rows

        # sample_batch = random.sample(self._data._validation_indexes, min_size)
        sample_batch = random.sample(range(len(self._validation_data)), min_size)
        validation_data_batch = self._validation_data.take(sample_batch, axis=0)
        sample_indexes = [self._data._validation_indexes[i] for i in sample_batch]

        output_labels = self._predict(validation_data_batch)
        df = self._data._dataframe.loc[sample_indexes]#.loc[sample_batch]#.reset_index(drop=True).loc[sample_batch].reset_index(drop=True)

        if self._data._dependent_variable:
            df[self._data._dependent_variable + '_results'] = output_labels
        else:
            df['prediction_results'] = output_labels

        return df.sort_index()

    def score(self):
        """
        :returns output from scikit-learn's model.score(), R2 score in case of regression and Accuracy in case of classification.
        For KMeans returns Opposite of the value of X on the K-means objective.
        """
        if (not self._data._is_unsupervised and (self._validation_data is None or self._validation_labels is None)) \
                or (self._data._is_unsupervised and self._validation_data is None):
            raise_data_exception()

        if self._data._is_unsupervised:
            if hasattr(self._model, 'score'):
                return self._model.score(self._training_data)

            raise Exception("Score function not applicable for unsupervised data")

        return self._model.score(self._validation_data, self._validation_labels)

    def decision_function(self):
        """
        :returns output from scikit-learn's model.decision_function()
        """
        if self._training_data is None:
            raise_data_exception()

        if not hasattr(self._model, 'decision_function'):
            raise Exception("Function not implemented for this model.")

        return self._model.decision_function(self._training_data)

    def mahalanobis(self):
        """
        :returns output from scikit-learn's model.mahalanobis()
        """
        if self._training_data is None:
            raise_data_exception()

        if not hasattr(self._model, 'mahalanobis'):
            raise Exception("Function not implemented for this model.")

        return self._model.mahalanobis(self._training_data)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        :returns output from scikit-learn's model.kneighbors()
        """
        if not hasattr(self._model, 'kneighbors'):
            raise Exception("Function not implemented for this model.")

        kwargs = {}
        if X:
            kwargs['X'] = X
        elif self._training_data is None:
            raise Exception("No data found")
        else:
            kwargs['X'] = self._training_data

        if n_neighbors:
            kwargs['n_neigbors'] = n_neighbors

        kwargs['return_distance'] = return_distance

        return self._model.kneighbors(**kwargs)

    def predict_proba(self):
        """
        :returns output from scikit-learn's model.predict_proba()
        """

        if not hasattr(self._model, 'predict_proba'):
            raise Exception("Function not implemented for this model.")

        if self._training_data is None:
            raise Exception("No data found.")

        return self._model.predict_proba(self._training_data)

    @property
    def feature_importances_(self):
        """
        :returns output from scikit-learn's model.feature_importances_
        """
        if not hasattr(self._model, 'feature_importances_'):
            raise Exception("Property not implemented for this model.")

        return self._model.feature_importances_

    def save(self, name_or_path):
        """
        Saves the model, creates an Esri Model Definition. Uses pickle to save the model.
        Using protocol level 2. Protocol level is backward compatible.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Folder path to save the model.
        =====================   ===========================================
        :returns dataframe
        """

        if '\\' in name_or_path or '/' in name_or_path:
            path = name_or_path
        else:
            path = os.path.join(self._data.path, 'models', name_or_path)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))

        if not os.path.exists(os.path.dirname(path)):
            raise Exception("Path doesn't exist")

        if not os.path.exists(path):
            os.mkdir(path)

        base_file_name = os.path.basename(path)

        model_file = os.path.join(path, base_file_name + '.pkl')

        with open(model_file, 'wb') as f:
            f.write(pickle.dumps(self._model, protocol=_PROTOCOL_LEVEL))

        MLModel._save_encoders(self._data._encoder_mapping, path, base_file_name)

        if self._data._procs:
            MLModel._save_transforms(self._data._procs, path, base_file_name)

        self._write_emd(path, base_file_name)

        return Path(path)

    def _write_emd(self, path, base_file_name):
        emd_file = os.path.join(path, base_file_name + '.emd')
        emd_params = {}
        emd_params['version'] = str(sklearn.__version__)
        if not self._data._is_unsupervised:
            if self._data._is_empty:
                emd_params['score'] = self._data._emd['score']
            else:
                emd_params['score'] = self.score()
        emd_params['_is_classification'] = "classification" if self._data._is_classification else "regression"
        emd_params['ModelName'] = type(self._model).__name__
        emd_params['ModelFile'] = base_file_name + '.pkl'
        emd_params['ModelParameters'] = self._model.get_params()
        emd_params['categorical_variables'] = self._data._categorical_variables

        if self._data._dependent_variable:
            emd_params['dependent_variable'] = self._data._dependent_variable

        emd_params['continuous_variables'] = self._data._continuous_variables

        with open(emd_file, 'w') as f:
            f.write(json.dumps(emd_params, indent=4))

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a `MLModel` Object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Esri Model Definition
                                file.
        ---------------------   -------------------------------------------
        data                    Required TabularDataObject or None. Returned data
                                object from `prepare_tabulardata` function or None for
                                inferencing.
        =====================   ===========================================

        :returns: `MLModel` Object
        """
        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        emd_path = str(emd_path)

        if emd_path.endswith('.dlpk'):
            with ZipFile(emd_path, 'r') as zip_obj:
                temp_dir = tempfile.TemporaryDirectory().name
                zip_obj.extractall(temp_dir)
                MLModel.from_model(temp_dir, data)

        if not emd_path.endswith('.emd'):
            emd_path = os.path.join(emd_path, (str(os.path.basename(emd_path))+'.emd'))

        if not os.path.exists(emd_path):
            raise Exception("Invalid data path.")

        with open(emd_path, 'r') as f:
            emd = json.loads(f.read())

        categorical_variables = emd['categorical_variables']
        dependent_variable = emd.get('dependent_variable', None)
        continuous_variables = emd['continuous_variables']
        model_parameters = emd['ModelParameters']

        if emd['version'] != str(sklearn.__version__):
            warnings.warn(f"Sklearn version has changed. Model Trained using version {emd['version']}")

        _is_classification = True
        if emd['_is_classification'] != "classification":
            _is_classification = False

        encoder_mapping = None
        if categorical_variables:
            encoder_path = os.path.join(os.path.dirname(emd_path), os.path.basename(emd_path).split('.')[0] + '_encoders.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder_mapping = pickle.loads(f.read())

        column_transformer = None
        transforms_path = os.path.join(os.path.dirname(emd_path), os.path.basename(emd_path).split('.')[0] + '_transforms.pkl')
        if os.path.exists(transforms_path):
            with open(transforms_path, 'rb') as f:
                column_transformer = pickle.loads(f.read())

        if data is None:
            data = TabularDataObject._empty(categorical_variables, continuous_variables, dependent_variable, encoder_mapping, column_transformer)
            data._is_classification = _is_classification

        data._emd = emd

        model_file = os.path.join(os.path.dirname(emd_path), emd['ModelFile'])
        with open(model_file, 'rb') as f:
            model = pickle.loads(f.read())

        return cls(data, emd['ModelName'], pretrained_model=model, **model_parameters)

    def _predict(self, data):
        return self._model.predict(data)

    @staticmethod
    def _save_encoders(encoder_mapping, path, base_file_name):
        if not encoder_mapping:
            return

        encoder_file = os.path.join(path, base_file_name + '_encoders.pkl')
        with open(encoder_file, 'wb') as f:
            f.write(pickle.dumps(encoder_mapping, protocol=_PROTOCOL_LEVEL))

    @staticmethod
    def _save_transforms(column_transformer, path, base_file_name):
        if not column_transformer:
            return

        transforms_file = os.path.join(path, base_file_name + '_transforms.pkl')
        with open(transforms_file, 'wb') as f:
            f.write(pickle.dumps(column_transformer, protocol=_PROTOCOL_LEVEL))

    @property
    def _is_kmeans(self):
        return self._model.__class__ == sklearn.cluster._kmeans.KMeans

    def _get_number_of_clusters(self, **kwargs):
        print("Finding optimum number of clusters")

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        threshold = 5
        count = 0

        max_score = -1
        max_cluster = 2
        scores = []
        for n_clusters in range_n_clusters:
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, **kwargs)
            cluster_labels = clusterer.fit_predict(self._training_data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self._training_data, cluster_labels)
            scores.append(silhouette_avg)

            if silhouette_avg > max_score:
                max_score = silhouette_avg
                max_cluster = n_clusters
                count = 0
            else:
                count = count + 1

            if count == threshold:
                break

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            range_n_clusters[0:len(scores)],
            scores
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Silhouette scores")
        # ax.set_xscale('log')
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(
            max_cluster,
            max_score,
            markersize=10,
            marker='o',
            color='red'
        )

        plt.show()

        print(f"Selecting n_clusters={max_cluster}")

        return max_cluster

    def load(self, name_or_path):
        """
        Loads a compatible saved model for inferencing or fine tuning from the disk.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Name or Path to
                                Esri Model Definition(EMD) file.
        =====================   ===========================================
        """
        if not ('\\' in str(name_or_path) or '/' in str(name_or_path)):
            name_or_path = self._data.path / 'models' / name_or_path
        model = MLModel.from_model(name_or_path, self._data)
        self._model = model._model

    def _get_number_of_components(self, **kwargs):
        print("Finding optimum number of components")

        from sklearn.mixture import GaussianMixture

        range_n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        max_score = -1
        max_component = 2

        threshold = 5
        count = 0
        scores = []
        for component in range_n_components:
            gm_model = GaussianMixture(n_components=component, **kwargs)

            gm_model.fit(self._training_data)
            score = gm_model.bic(self._training_data)

            scores.append(score)

            if score > max_score:
                max_component = component
                max_score = score
                count = 0
            else:
                count = count + 1

            if count == threshold:
                break

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            range_n_components[0:len(scores)],
            scores
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("BIC score")
        # ax.set_xscale('log')
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(
            max_component,
            max_score,
            markersize=10,
            marker='o',
            color='red'
        )

        plt.show()

        print(f"Selecting n_components={max_component}")

        return max_component

    def predict(
            self,
            input_features=None,
            explanatory_rasters=None,
            datefield=None,
            distance_features=None,
            output_layer_name="Prediction Layer",
            gis=None,
            prediction_type='features',
            output_raster_path=None,
            match_field_names=None):
        """

        Predict on data from feature layer, dataframe and or raster data.

        =================================   =========================================================================
        **Argument**                        **Description**
        ---------------------------------   -------------------------------------------------------------------------
        input_features                      Optional Feature Layer or spatial dataframe. Required if prediction_type='features'.
                                            Contains features with location and
                                            some or all fields required to infer the dependent variable value.
        ---------------------------------   -------------------------------------------------------------------------
        explanatory_rasters                 Optional list. Required if prediction_type='raster'.
                                            Contains a list of raster objects containing
                                            some or all fields required to infer the dependent variable value.
        ---------------------------------   -------------------------------------------------------------------------
        datefield                           Optional string. Field name from feature layer
                                            that contains the date, time for the input features.
                                            Same as `prepare_tabulardata()`.
        ---------------------------------   -------------------------------------------------------------------------
        distance_features                   Optional List of Feature Layer objects.
                                            These layers are used for calculation of field "NEAR_DIST_1",
                                            "NEAR_DIST_2" etc in the output dataframe.
                                            These fields contain the nearest feature distance
                                            from the input_features.
                                            Same as `prepare_tabulardata()`.
        ---------------------------------   -------------------------------------------------------------------------
        output_layer_name                   Optional string. Used for publishing the output layer.
        ---------------------------------   -------------------------------------------------------------------------
        gis                                 Optional GIS Object. Used for publishing the item.
                                            If not specified then active gis user is taken.
        ---------------------------------   -------------------------------------------------------------------------
        prediction_type                     Optional String.
                                            Set 'features' or 'dataframe' to make output feature layer predictions.
                                            With this feature_layer argument is required.

                                            Set 'raster', to make prediction raster.
                                            With this rasters must be specified.
        ---------------------------------   -------------------------------------------------------------------------
        output_raster_path                  Optional path.
                                            Required when prediction_type='raster', saves
                                            the output raster to this path.
        ---------------------------------   -------------------------------------------------------------------------
        match_field_names                   Optional dictionary.
                                            Specify mapping of field names from prediction set
                                            to training set.
                                            For example:
                                                {
                                                    "Field_Name_1": "Field_1",
                                                    "Field_Name_2": "Field_2"
                                                }
        =================================   =========================================================================

        :returns Feature Layer if prediction_type='features', dataframe for prediction_type='dataframe' else creates an output raster.

        """

        rasters = explanatory_rasters if explanatory_rasters else []
        if prediction_type in ['features', 'dataframe']:

            if input_features is None:
                raise Exception("Feature Layer required for predict_features=True")

            gis = gis if gis else arcgis.env.active_gis
            return self._predict_features(input_features, rasters, datefield, distance_features, output_layer_name, gis, match_field_names, prediction_type)
        else:
            if not rasters:
                raise Exception("Rasters required for predict_features=False")

            if not output_raster_path:
                raise Exception("Please specify output_raster_folder_path to save the output.")

            return self._predict_rasters(output_raster_path, rasters, match_field_names)

    def _predict_features(
            self,
            input_features,
            rasters=None,
            datefield=None,
            distance_feature_layers=None,
            output_name="Prediction Layer",
            gis=None,
            match_field_names=None,
            prediction_type="features"
    ):
        if isinstance(input_features, FeatureLayer):
            dataframe = input_features.query().sdf
        else:
            dataframe = input_features.copy()

        fields_needed = self._data._categorical_variables + self._data._continuous_variables
        distance_feature_layers = distance_feature_layers if distance_feature_layers else []
        continuous_variables = self._data._continuous_variables

        columns = dataframe.columns
        feature_layer_columns = []
        for column in columns:
            column_name = column
            categorical = False

            if column_name in fields_needed:
                if column_name not in continuous_variables:
                    categorical = True
            elif match_field_names and match_field_names.get(column_name):
                if match_field_names.get(column_name) not in continuous_variables:
                    categorical = True
            else:
                continue

            feature_layer_columns.append((column_name, categorical))

        raster_columns = []
        if rasters:
            for raster in rasters:
                column_name = raster.name
                categorical = False
                if column_name in fields_needed:
                    if column_name not in continuous_variables:
                        categorical = True
                elif match_field_names and match_field_names.get(column_name):
                    column_name = match_field_names.get(column_name)
                    if column_name not in continuous_variables:
                        categorical = True
                else:
                    continue

                raster_columns.append((raster, categorical))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            processed_dataframe, fields_mapping = TabularDataObject._prepare_dataframe_from_features(
                input_features,
                self._data._dependent_variable,
                feature_layer_columns,
                raster_columns,
                datefield,
                distance_feature_layers
            )

        if match_field_names:
            processed_dataframe.rename(columns=match_field_names, inplace=True)

        for field in fields_needed:
            if field not in processed_dataframe.columns:
                raise Exception(f"Field missing {field}")

        for column in processed_dataframe.columns:
            if column not in fields_needed:
                processed_dataframe = processed_dataframe.drop(column, axis=1)

        processed_numpy = self._data._process_data(processed_dataframe.reindex(sorted(processed_dataframe.columns), axis=1), fit=False)
        predictions = self._predict(processed_numpy)
        dataframe["prediction_results"] = predictions

        if prediction_type == "dataframe":
            return dataframe

        if 'SHAPE' in list(dataframe.columns):
            return dataframe.spatial.to_featurelayer(output_name, gis)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                table_file = os.path.join(tmpdir, output_name + '.xlsx')
                dataframe.to_excel(table_file, index=False, header=True)
                online_table = gis.content.add({'type': 'Microsoft Excel', 'overwrite': True}, table_file)
                return online_table.publish(overwrite=True)

    def _predict_rasters(self, output_folder_path, rasters, match_field_names=None):

        if not os.path.exists(os.path.dirname(output_folder_path)):
            raise Exception("Output directory doesn't exist")

        if os.path.exists(output_folder_path):
            raise Exception("Output Folder already exists")

        try:
            import arcpy
        except:
            raise Exception("This function requires arcpy.")

        try:
            import numpy as np
        except:
            raise Exception("This function requires numpy.")

        try:
            import pandas as pd
        except:
            raise Exception("This function requires pandas.")

        if not HAS_FAST_PROGRESS:
            raise Exception("This function requires fastprogress.")

        fields_needed = self._data._categorical_variables + self._data._continuous_variables

        try:
            arcpy.env.outputCoordinateSystem = rasters[0].extent['spatialReference']['wkt']
        except:
            arcpy.env.outputCoordinateSystem = rasters[0].extent['spatialReference']['wkid']

        xmin = rasters[0].extent['xmin']
        xmax = rasters[0].extent['xmax']
        ymin = rasters[0].extent['ymin']
        ymax = rasters[0].extent['ymax']
        min_cell_size_x = rasters[0].mean_cell_width
        min_cell_size_y = rasters[0].mean_cell_height

        default_sr = rasters[0].extent['spatialReference']

        for raster in rasters:
            point_upper = arcgis.geometry.Point(
                {'x': raster.extent['xmin'], 'y': raster.extent['ymax'], 'sr': raster.extent['spatialReference']})
            point_lower = arcgis.geometry.Point(
                {'x': raster.extent['xmax'], 'y': raster.extent['ymin'], 'sr': raster.extent['spatialReference']})
            cell_size = arcgis.geometry.Point(
                {'x': raster.mean_cell_width, 'y': raster.mean_cell_height, 'sr': raster.extent['spatialReference']})

            points = arcgis.geometry.project([point_upper, point_lower, cell_size], raster.extent['spatialReference'],
                                             default_sr)
            point_upper = points[0]
            point_lower = points[1]
            cell_size = points[2]

            if xmin > point_upper.x:
                xmin = point_upper.x
            if ymax < point_upper.y:
                ymax = point_upper.y
            if xmax < point_lower.x:
                xmax = point_lower.x
            if ymin > point_lower.y:
                ymin = point_lower.y

            if min_cell_size_x > cell_size.x:
                min_cell_size_x = cell_size.x

            if min_cell_size_y > cell_size.y:
                min_cell_size_y = cell_size.y

        max_raster_columns = int(abs(math.ceil((xmax-xmin)/min_cell_size_x)))
        max_raster_rows = int(abs(math.ceil((ymax-ymin)/min_cell_size_y)))

        point_upper = arcgis.geometry.Point({'x': xmin, 'y': ymax, 'sr': default_sr})
        cell_size = arcgis.geometry.Point({'x': min_cell_size_x, 'y': min_cell_size_y, 'sr': default_sr})

        raster_data = {}
        for raster in rasters:
            field_name = raster.name
            point_upper_translated = arcgis.geometry.project([point_upper], default_sr, raster.extent['spatialReference'])[0]
            cell_size_translated = arcgis.geometry.project([cell_size], default_sr, raster.extent['spatialReference'])[0]
            if field_name in fields_needed:
                raster_read = raster.read(origin_coordinate=(point_upper_translated.x, point_upper_translated.y), ncols=max_raster_columns, nrows=max_raster_rows, cell_size=(cell_size_translated.x, cell_size_translated.y))
                for row in range(max_raster_rows):
                    for column in range(max_raster_columns):
                        values = raster_read[row][column]
                        index = 0
                        for value in values:
                            key = field_name
                            if index != 0:
                                key = key + f'_{index}'
                            if not raster_data.get(key):
                                raster_data[key] = []
                            index = index + 1
                            raster_data[key].append(value)
            elif match_field_names and match_field_names.get(raster.name):
                field_name = match_field_names.get(raster.name)
                raster_read = raster.read(origin_coordinate=(point_upper_translated.x, point_upper_translated.y), ncols=max_raster_columns, nrows=max_raster_rows, cell_size=(cell_size_translated.x, cell_size_translated.y))
                for row in range(max_raster_rows):
                    for column in range(max_raster_columns):
                        values = raster_read[row][column]
                        index = 0
                        for value in values:
                            key = field_name
                            if index != 0:
                                key = key + f'_{index}'
                            if not raster_data.get(key):
                                raster_data[key] = []
                            index = index + 1
                            raster_data[key].append(value)
            else:
                continue

        for field in fields_needed:
            if field not in list(raster_data.keys()) and match_field_names and match_field_names.get(field, None) is None:
                raise Exception(f"Field missing {field}")

        processed_data = []

        length_values = len(raster_data[list(raster_data.keys())[0]])
        for i in range(length_values):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_data.append(processed_row)

        processed_df = pd.DataFrame(data=np.array(processed_data), columns=sorted(raster_data))

        processed_numpy = self._data._process_data(processed_df, fit=False)

        predictions = self._predict(processed_numpy)

        predictions = np.array(predictions.reshape([max_raster_rows, max_raster_columns]), dtype='float64')

        processed_raster = arcpy.NumPyArrayToRaster(predictions, arcpy.Point(xmin, ymin), x_cell_size=min_cell_size_x, y_cell_size=min_cell_size_y)
        processed_raster.save(output_folder_path)

        return True
