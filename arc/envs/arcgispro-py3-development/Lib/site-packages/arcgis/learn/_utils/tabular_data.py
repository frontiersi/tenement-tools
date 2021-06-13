import random
import tempfile
import warnings
import sys
import math
import os
from pathlib import Path

import arcgis
from arcgis.features import FeatureLayer

HAS_FASTAI = True
try:
    from fastai.tabular import TabularList
    from fastai.tabular import TabularDataBunch
    from fastai.tabular.transform import FillMissing, Categorify, Normalize
    from fastai.tabular import cont_cat_split, add_datepart
    from fastai.data_block import ItemLists, CategoryList, FloatList
    from .._utils.TSData import TimeSeriesList, To3dTensor
    from fastai.data_block import DatasetType
    import torch
except Exception as e:
    HAS_FASTAI = False

HAS_NUMPY = True
try:
    import numpy as np
except:
    HAS_NUMPY = False

HAS_SK_LEARN = True
try:
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler
except:
    HAS_SK_LEARN = False


class DummyTransform(object):
    def __int__(self):
        pass

    def fit_transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class TabularDataObject(object):
    _categorical_variables = []
    _continuous_variables = []
    dependent_variables = []

    @classmethod
    def prepare_data_for_layer_learner(
        cls,
        input_features,
        dependent_variable,
        feature_variables=None,
        raster_variables=None,
        date_field=None,
        distance_feature_layers=None,
        procs=None,
        val_split_pct=0.1,
        seed=42,
        batch_size=64,
        index_field=None,
        column_transforms_mapping=None
    ):

        if not HAS_FASTAI:
            return

        feature_variables = feature_variables if feature_variables else []
        raster_variables = raster_variables if raster_variables else []

        tabular_data = cls()
        tabular_data._dataframe, tabular_data._field_mapping = TabularDataObject._prepare_dataframe_from_features(
            input_features,
            dependent_variable,
            feature_variables,
            raster_variables,
            date_field,
            distance_feature_layers,
            index_field
        )

        if input_features is None:
            tabular_data._is_raster_only = True
        else:
            tabular_data._is_raster_only = False

        tabular_data._dataframe = tabular_data._dataframe.reindex(sorted(tabular_data._dataframe.columns), axis=1)

        tabular_data._categorical_variables = tabular_data._field_mapping['categorical_variables']
        tabular_data._continuous_variables = tabular_data._field_mapping['continuous_variables']
        tabular_data._dependent_variable = tabular_data._field_mapping['dependent_variable']
        tabular_data._index_data = tabular_data._field_mapping['index_data']
        tabular_data._index_field = index_field

        tabular_data._procs = procs
        tabular_data._column_transforms_mapping = column_transforms_mapping
        tabular_data._val_split_pct = val_split_pct
        tabular_data._bs = batch_size
        tabular_data._seed = seed

        validation_indexes = []
        if tabular_data._dependent_variable:
            random.seed(seed)
            validation_indexes = random.sample(range(len(tabular_data._dataframe)), round(val_split_pct * len(tabular_data._dataframe)))
            tabular_data._validation_indexes = validation_indexes

        tabular_data._training_indexes = list(set([i for i in range(len(tabular_data._dataframe))]) - set(validation_indexes))
        if not tabular_data._dependent_variable:
            tabular_data._validation_indexes = list(set([i for i in range(len(tabular_data._dataframe))]))

        tabular_data._is_empty = False

        if tabular_data._dependent_variable:
            tabular_data._is_classification = tabular_data._is_classification()
        else:
            tabular_data._is_classification = True

        tabular_data.path = Path(os.getcwd())
        return tabular_data

    @staticmethod
    def _min_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            if len(values) == 1:
                return_values.append(values[0])
                continue

            min_value = values[0]
            for value in values:
                if value < min_value:
                    min_value = value
            return_values.append(min_value)

        return return_values

    @staticmethod
    def _max_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(max(values))

        return return_values

    @staticmethod
    def _mean_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(sum(values) / len(values))

        return return_values

    @staticmethod
    def _majority_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(max(values, key=values.count))

        return return_values

    @staticmethod
    def _minority_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(min(values, key=values.count))

        return return_values

    @staticmethod
    def _sum_of(values_list):
        return_values = []
        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(sum(values))

        return return_values

    @staticmethod
    def _std_dev_of(values_list):
        import statistics
        return_values = []

        for values in values_list:
            if len(values) == 0:
                return_values.append(0)
                continue

            return_values.append(statistics.stdev(values))

        return return_values

    @staticmethod
    def _variety(values_list):
        return_values = []
        for values in values_list:
            return_values.append(len(list(set(values))))

        return return_values

    @staticmethod
    def _get_calc(raster_type, calc_type):
        calc_type = calc_type.lower()

        cont_mapping = {
            'min': TabularDataObject._min_of,
            'max': TabularDataObject._max_of,
            'mean': TabularDataObject._mean_of,
            'majority': TabularDataObject._majority_of,
            'minority': TabularDataObject._minority_of,
            'std_dev': TabularDataObject._std_dev_of,
            'sum': TabularDataObject._sum_of,
            'variety': TabularDataObject._variety
        }

        cat_mapping = {
            'majority': TabularDataObject._majority_of,
            'minority': TabularDataObject._minority_of,
            'variety': TabularDataObject._variety
        }

        if raster_type:
            return cat_mapping.get(calc_type, TabularDataObject._majority_of)
        else:
            return cont_mapping.get(calc_type, TabularDataObject._mean_of)

    def _prepare_validation_databunch(self, dataframe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kwargs_variables = {'num_workers': 0} if sys.platform == 'win32' else {}
            # kwargs_variables['tfm_y'] = True
            fm = FillMissing(self._categorical_variables, self._continuous_variables)
            fm.add_col = False
            fm(dataframe)
            databunch_half = TabularList.from_df(
                dataframe,
                path=tempfile.NamedTemporaryFile().name,
                cat_names=self._categorical_variables,
                cont_names=self._continuous_variables,
                procs=[Categorify, Normalize]
            ).split_by_idx([i for i in range(int(len(dataframe)/2))]).label_empty().databunch(**kwargs_variables)

            databunch_second_half = TabularList.from_df(
                dataframe,
                path=tempfile.NamedTemporaryFile().name,
                cat_names=self._categorical_variables,
                cont_names=self._continuous_variables,
                procs=[Categorify, Normalize]
            ).split_by_idx([i for i in range(int(len(dataframe)/2), len(dataframe))]).label_empty().databunch(**kwargs_variables)

        return databunch_half, databunch_second_half

    @property
    def _is_unsupervised(self):
        if not self._dependent_variable:
            return True

        return False

    def _is_classification(self):
        if self._is_empty:
            return True

        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        df = self._dataframe
        labels = df[self._dependent_variable]

        if labels.isna().sum().sum() != 0:
            raise Exception("You have some missing values in dependent variable column.")

        unique_labels = labels.unique()

        labels = np.array(labels)

        from numbers import Integral
        if isinstance(labels[0], (float, np.float32)) or len(unique_labels) > 20:
            return False

        if isinstance(int(labels[0]), (str, Integral)):
            return True

    def _is_categorical(self, labels):
        unique_labels = labels.unique()

        labels = np.array(labels)

        from numbers import Integral
        if isinstance(labels[0], (float, np.float32)) or len(unique_labels) > 20:
            return False

        if isinstance(int(labels[0]), (str, Integral)):
            return True

    @property
    def _databunch(self):
        if self._is_empty:
            return None

        if self._procs is not None and not isinstance(self._procs, list):
            self._procs = []

        return TabularDataObject._prepare_databunch(
            self._dataframe,
            self._field_mapping,
            self._procs,
            self._validation_indexes,
            self._bs
        )

    @property
    def _ml_data(self):
        if self._is_empty:
            return None, None, None, None

        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        dataframe = self._dataframe

        labels = None

        if self._dependent_variable:
            labels = np.array(dataframe[self._dependent_variable])
            dataframe = dataframe.drop(self._dependent_variable, axis=1)

        if not self._procs:
            numerical_transformer = make_pipeline(
                SimpleImputer(strategy='median'),
                Normalizer())

            categorical_transformer = make_pipeline(
                SimpleImputer(strategy='constant')
            )

            self._procs = make_column_transformer(
                (numerical_transformer, self._continuous_variables),
                (categorical_transformer, self._categorical_variables))

        _procs = self._procs

        self._encoder_mapping = None
        if self._categorical_variables:
            mapping = {}
            for variable in self._categorical_variables:
                labelEncoder = LabelEncoder()
                dataframe[variable] = np.array(labelEncoder.fit_transform(dataframe[variable]), dtype='int64')
                mapping[variable] = labelEncoder
            self._encoder_mapping = mapping

        processed_data = _procs.fit_transform(dataframe)

        training_data = processed_data.take(self._training_indexes, axis=0)
        training_labels = None
        if self._dependent_variable:
            training_labels = labels.take(self._training_indexes)

        validation_data = processed_data.take(self._validation_indexes, axis=0)
        validation_labels = None
        if self._dependent_variable:
            validation_labels = labels.take(self._validation_indexes)

        return training_data, training_labels, validation_data, validation_labels

    def _time_series_bunch(self, seq_len, normalize=True, bunch=True):
        if self._index_data is not None:
            bunched = []
            for i in range(len(self._index_data) - seq_len - 1):
                bunched.append(list(self._index_data[i:i + seq_len]))

            self._index_seq = np.array(bunched)

        if self._is_raster_only:
            return self._raster_timeseries_bunch(normalize, bunch)

        if len(list(self._dataframe.columns.values)) == 1:
            return self._univariate_bunch(seq_len, normalize, bunch)
        else:
            return self._multivariate_bunch(seq_len, normalize, bunch)

    def _raster_timeseries_bunch(self, normalize=True, bunched=True):
        kwargs_variables = {'num_workers': 0} if sys.platform == 'win32' else {}

        kwargs_variables['bs'] = self._bs

        if hasattr(arcgis, "env") and getattr(arcgis.env, "_processorType", "") == "CPU":
            kwargs_variables["device"] = torch.device('cpu')

        self._encoder_mapping = None
        mapping = {}
        df = self._dataframe.copy()  # .drop(self._dependent_variable, axis=1)

        for col in list(df.columns.values):
            if self._is_categorical(df[col]):
                labelEncoder = LabelEncoder()
                df[col] = np.array(labelEncoder.fit_transform(df[col]), dtype='int64')
                mapping[col] = labelEncoder

        self._encoder_mapping = mapping

        if normalize:
            if len(self._column_transforms_mapping) == 0:
                for col in list(df.columns):
                    self._column_transforms_mapping[col] = [MinMaxScaler()]
            else:
                for col in list(df.columns):
                    if len(self._column_transforms_mapping.get(col, [])) == 0:
                        self._column_transforms_mapping[col] = [DummyTransform()]

            processed_dataframe = df.copy()
            for col in list(df.columns):
                transformed_data = df[col]
                for transform in self._column_transforms_mapping.get(col, []):
                    transformed_data = transform.fit_transform(
                        np.array(transformed_data, dtype=df[col].dtype).reshape(-1, 1))
                    transformed_data = transformed_data.squeeze(1)
                processed_dataframe[col] = np.array(transformed_data, dtype=df[col].dtype)
        else:
            processed_dataframe = df.copy()

        big_bunch = []

        proc_df = processed_dataframe.copy()
        proc_df = proc_df.drop(self._dependent_variable, axis=1)

        for i in range(len(processed_dataframe)):
            big_bunch.append([proc_df.iloc[i].values])

        big_bunch = np.array(big_bunch)

        random.seed(self._seed)
        validation_indexes = random.sample(range(big_bunch.shape[0]),
                                           round(self._val_split_pct * big_bunch.shape[0]))
        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(big_bunch.shape[0])]) - set(validation_indexes))

        X_train = big_bunch.take(self._training_indexes_ts, axis=0)
        X_valid = big_bunch.take(self._validation_indexes_ts, axis=0)

        y_train = np.array(processed_dataframe[self._dependent_variable].take(self._training_indexes_ts))
        y_valid = np.array(processed_dataframe[self._dependent_variable].take(self._validation_indexes_ts))

        if bunched is False:
            return X_train, X_valid, y_train, y_valid

        data = (ItemLists('.', TimeSeriesList(X_train), TimeSeriesList(X_valid))
                .label_from_lists(y_train, y_valid, label_cls=FloatList)
                .databunch(**kwargs_variables))

        return data

    def _multivariate_bunch(self, seq_len, normalize=True, bunched=True):
        kwargs_variables = {'num_workers': 0} if sys.platform == 'win32' else {}

        kwargs_variables['bs'] = self._bs

        if hasattr(arcgis, "env") and getattr(arcgis.env, "_processorType", "") == "CPU":
            kwargs_variables["device"] = torch.device('cpu')

        self._encoder_mapping = None
        mapping = {}
        df = self._dataframe.copy()#.drop(self._dependent_variable, axis=1)

        for col in list(df.columns.values):
            if self._is_categorical(df[col]):
                labelEncoder = LabelEncoder()
                df[col] = np.array(labelEncoder.fit_transform(df[col]), dtype='int64')
                mapping[col] = labelEncoder

        self._encoder_mapping = mapping

        if normalize:
            if len(self._column_transforms_mapping) == 0:
                for col in list(df.columns):
                    self._column_transforms_mapping[col] = [MinMaxScaler()]
            else:
                for col in list(df.columns):
                    if len(self._column_transforms_mapping.get(col, [])) == 0:
                        self._column_transforms_mapping[col] = [DummyTransform()]

            processed_dataframe = df.copy()
            for col in list(df.columns):
                transformed_data = df[col]
                for transform in self._column_transforms_mapping.get(col, []):
                    transformed_data = transform.fit_transform(np.array(transformed_data, dtype=df[col].dtype).reshape(-1, 1))
                    transformed_data = transformed_data.squeeze(1)
                processed_dataframe[col] = np.array(transformed_data, dtype=df[col].dtype)
        else:
            processed_dataframe = df.copy()

        big_bunch = []

        for i in range(len(processed_dataframe) - seq_len-1):
            bunch = []
            for col in list(processed_dataframe.columns.values):
                bunch.append(list(processed_dataframe[col][i:i + seq_len]))

            big_bunch.append(bunch)

        big_bunch = np.array(big_bunch)

        random.seed(self._seed)
        validation_indexes = random.sample(range(big_bunch.shape[0]),
                                           round(self._val_split_pct * big_bunch.shape[0]))
        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(big_bunch.shape[0])]) - set(validation_indexes))

        X_train = big_bunch.take(self._training_indexes_ts, axis=0)
        X_valid = big_bunch.take(self._validation_indexes_ts, axis=0)

        y_train = np.array(processed_dataframe[self._dependent_variable].take(self._training_indexes_ts))
        y_valid = np.array(processed_dataframe[self._dependent_variable].take(self._validation_indexes_ts))

        if bunched is False:
            return X_train, X_valid, y_train, y_valid

        data = (ItemLists('.', TimeSeriesList(X_train), TimeSeriesList(X_valid))
                .label_from_lists(y_train, y_valid, label_cls=FloatList)
                .databunch(**kwargs_variables))

        return data

    def _univariate_bunch(self, seq_len, normalize=True, bunch=True):
        kwargs_variables = {'num_workers': 0} if sys.platform == 'win32' else {}

        kwargs_variables['bs'] = self._bs

        if hasattr(arcgis, "env") and getattr(arcgis.env, "_processorType", "") == "CPU":
            kwargs_variables["device"] = torch.device('cpu')

        df_columns = {}
        for i in range(seq_len):
            df_columns[f'att{i + 1}'] = []

        df_columns['target'] = []

        if self._is_classification:
            self._encoder_mapping = None
            mapping = {}
            labelEncoder = LabelEncoder()
            self._dataframe[self._dependent_variable] = np.array(labelEncoder.fit_transform(self._dataframe[self._dependent_variable]), dtype='int64')
            mapping[self._dependent_variable] = labelEncoder
            self._encoder_mapping = mapping

        if normalize:
            if not self._column_transforms_mapping.get(self._dependent_variable):
                self._column_transforms_mapping[self._dependent_variable] = [MinMaxScaler()]

            processed_dataframe = self._dataframe.copy()
            transformed_data = processed_dataframe[self._dependent_variable]
            for transform in self._column_transforms_mapping[self._dependent_variable]:
                transformed_data = transform.fit_transform(np.array(transformed_data, dtype=processed_dataframe[self._dependent_variable].dtype).reshape(-1, 1))
                transformed_data = transformed_data.squeeze(1)

            processed_dataframe[self._dependent_variable] = np.array(transformed_data, dtype=self._dataframe[self._dependent_variable].dtype)
        else:
            processed_dataframe = self._dataframe.copy()

        for i in range(len(processed_dataframe[self._dependent_variable]) - seq_len):
            for j in range(seq_len):
                if len(processed_dataframe[self._dependent_variable]) > i + seq_len - 1:
                    df_columns[f'att{j + 1}'].append(processed_dataframe[self._dependent_variable][i + j])
                else:
                    continue

            df_columns['target'].append(processed_dataframe[self._dependent_variable][i + seq_len])

        import pandas as pd
        df = pd.DataFrame(df_columns)

        columns = list(df.columns.values)
        columns.remove('target')

        random.seed(self._seed)
        validation_indexes = random.sample(range(len(df)),
                                           round(self._val_split_pct * len(df)))
        self._validation_indexes_ts = validation_indexes

        self._training_indexes_ts = list(
            set([i for i in range(len(df))]) - set(validation_indexes))

        y_train = np.array(df['target'].take(self._training_indexes_ts))
        y_valid = np.array(df['target'].take(self._validation_indexes_ts))

        X_train = To3dTensor(df.iloc[:, :-1].take(self._training_indexes_ts).values.astype(np.float32))
        X_valid = To3dTensor(df.iloc[:, :-1].take(self._validation_indexes_ts).values.astype(np.float32))

        if bunch is False:
            return X_train, X_valid, y_train, y_valid

        if self._is_classification:
            label_cls = CategoryList
        else:
            label_cls = FloatList

        data = (ItemLists('.', TimeSeriesList(X_train), TimeSeriesList(X_valid))
              .label_from_lists(y_train, y_valid, label_cls=label_cls)
              .databunch(**kwargs_variables)
              )

        return data

    def _process_data(self, dataframe, fit=True):
        if not HAS_NUMPY:
            raise Exception("This module requires numpy.")

        if not HAS_SK_LEARN:
            raise Exception("This module requires scikit-learn.")

        if not self._procs:
            numerical_transformer = make_pipeline(
                SimpleImputer(strategy='median'),
                Normalizer())

            categorical_transformer = make_pipeline(
                SimpleImputer(strategy='constant')
            )

            self._procs = make_column_transformer(
                (numerical_transformer, self._continuous_variables),
                (categorical_transformer, self._categorical_variables))

        _procs = self._procs

        if self._encoder_mapping:
            for variable, encoder in self._encoder_mapping.items():
                dataframe[variable] = np.array(encoder.fit_transform(dataframe[variable]), dtype='int64')

        if fit:
            processed_data = _procs.fit_transform(dataframe)
        else:
            processed_data = _procs.transform(dataframe)

        return processed_data

    def show_batch(self, rows=5, graph=False, seq_len=None):
        """
        Shows a chunk of data prepared without applying transforms.
        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional integer. Number of rows of dataframe
                                or graph to plot. This parameter is not used
                                when plotting complete data.
        ---------------------   -------------------------------------------
        graph                   Optional boolean. Used for visualizing
                                time series data. The index_field passed
                                in prepare_tabulardata is used on the x-axis.
                                Use this option to plot complete data.
        ---------------------   -------------------------------------------
        seq_len                 Optional integer. Used for visualizing data
                                in the form of graph of seq_len duration.
        =====================   ===========================================
        """

        if seq_len is not None or graph is True:
            self._show_graph(seq_len=seq_len, rows=rows)
            return

        if not rows or rows <= 0:
            rows = self._bs
        elif rows > len(self._training_indexes):
            rows = len(self._training_indexes)

        random_batch = random.sample(self._training_indexes, rows)
        return self._dataframe.loc[random_batch].sort_index()

    def _show_graph(self, seq_len=None, rows=5):
        """
        Shows a batch of prepared data in the form of graphs
        """

        if self._is_unsupervised:
            raise Exception("Show Graphs is used for Time Series Network")

        import matplotlib.pyplot as plt

        if seq_len is not None:
            X_train, X_valid, y_train, y_valid = self._time_series_bunch(seq_len, False, False)

            n_items = rows ** 2
            if n_items > len(X_train):
                n_items = len(X_train)

            sample = random.sample(range(len(X_train)), n_items)
            X_train_sample = np.array(X_train).take(sample, axis=0)
            y_train_sample = np.array(y_train).take(sample, axis=0)

            batched_index = []
            if self._index_data is not None:
                indexes = 0
                while indexes < len(self._index_data):
                    batched_index.append(self._index_data[indexes: indexes+seq_len])
                    indexes = indexes + seq_len
            else:
                j = 0
                while j < n_items:
                    batched_index.append([i for i in range(seq_len)])
                    j = j + 1

            rows = int(math.sqrt(n_items))

            fig, axs = plt.subplots(rows, rows, figsize=(10, 10))

            for i in range(rows):
                for j in range(rows):
                    for predictor in X_train_sample[i+j]:
                        axs[i, j].plot(batched_index[i+j], predictor)
                    axs[i, j].set_title(y_train_sample[i+j])
                    axs[i, j].tick_params(axis="x", labelrotation=60)

            plt.tight_layout()
            plt.show()
        else:
            # plotting the points
            # y = self._dataframe[self._dependent_variable]
            x_field = 'Time'
            if self._index_data is not None:
                x = self._index_data
                x_field = self._index_field
            else:
                x = [i for i in range(len(self._dataframe[self._dependent_variable]))]

            fig, axs = plt.subplots(len(list(self._dataframe.columns)), 1, figsize=(25, 5*len(list(self._dataframe.columns))))
            counter = 0
            for col in list(self._dataframe.columns):
                if isinstance(axs, np.ndarray):
                    axs[counter].plot(x, self._dataframe[col], label=col)
                    axs[counter].set_title(col)
                    axs[counter].tick_params(axis="x", labelrotation=60)
                else:
                    axs.plot(x, self._dataframe[col], label=col)
                    axs.set_title(col)
                    axs.tick_params(axis="x", labelrotation=60)
                counter = counter + 1

            plt.show()

    @staticmethod
    def _prepare_dataframe_from_features(
            input_features,
            dependent_variable,
            feature_variables=None,
            raster_variables=None,
            date_field=None,
            distance_feature_layers=None,
            index_field=None
    ):
        feature_variables = feature_variables if feature_variables else []
        raster_variables = raster_variables if raster_variables else []
        distance_feature_layers = distance_feature_layers if distance_feature_layers else []

        continuous_variables = []
        categorical_variables = []
        for field in feature_variables:
            if isinstance(field, tuple):
                if field[1]:
                    categorical_variables.append(field[0])
                else:
                    continuous_variables.append(field[0])
            else:
                continuous_variables.append(field)

        rasters = []
        for raster in raster_variables:
            if isinstance(raster, tuple):
                rasters.append(raster[0])
                if raster[1]:
                    band_count = raster[0].band_count
                    for index in range(band_count):
                        if index == 0:
                            categorical_variables.append(raster[0].name)
                        else:
                            categorical_variables.append(raster[0].name + f'_{index}')
                else:
                    band_count = raster[0].band_count
                    for index in range(band_count):
                        if index == 0:
                            continuous_variables.append(raster[0].name)
                        else:
                            continuous_variables.append(raster[0].name + f'_{index}')
            else:
                rasters.append(raster)
                band_count = raster.band_count
                for index in range(band_count):
                    if index == 0:
                        continuous_variables.append(raster.name)
                    else:
                        continuous_variables.append(raster.name + f'_{index}')

        dataframe, index_data = TabularDataObject._process_layer(
            input_features,
            date_field,
            distance_feature_layers,
            raster_variables,
            index_field
        )

        dataframe_columns = dataframe.columns
        if distance_feature_layers:
            count = 1
            while f'NEAR_DIST_{count}' in dataframe_columns:
                continuous_variables.append(f'NEAR_DIST_{count}')
                count = count + 1

        fields_to_keep = continuous_variables + categorical_variables
        if dependent_variable:
            fields_to_keep = fields_to_keep + [dependent_variable]

        for column in dataframe_columns:
            if column not in fields_to_keep:
                dataframe = dataframe.drop(column, axis=1)
            elif dependent_variable and column == dependent_variable:
                continue
            elif column in categorical_variables and dataframe[column].dtype == float:
                warnings.warn(f"Changing column {column} to continuous")
                categorical_variables.remove(column)
                continuous_variables.append(column)
            elif column in categorical_variables and dataframe[column].unique().shape[0] > 20:
                warnings.warn(f"Column {column} has more than 20 unique value. Sure this is categorical?")

        if date_field:
            date_fields = [
                ('Year', True), ('Month', True), ('Week', True),
                ('Day', True), ('Dayofweek', True), ('Dayofyear', False),
                ('Is_month_end', True), ('Is_month_start', True),
                ('Is_quarter_end', True), ('Is_quarter_start', True),
                ('Is_year_end', True), ('Is_year_start', True),
                ('Hour', True), ('Minute', True), ('Second', True), ('Elapsed', False)]

            for field in date_fields:
                if field[0] in dataframe_columns:
                    if field[1]:
                        categorical_variables.append(field[0])
                    else:
                        continuous_variables.append(field[1])

        return dataframe, {'dependent_variable': dependent_variable,
                           'categorical_variables': categorical_variables if categorical_variables else [],
                           'continuous_variables': continuous_variables if continuous_variables else [],
                           'index_data': index_data}

    @staticmethod
    def _process_layer(input_features, date_field, distance_layers, rasters, index_field):
        index_data = None
        if input_features is not None:
            if isinstance(input_features, FeatureLayer):
                input_layer = input_features
                sdf = input_features.query().sdf
            else:
                sdf = input_features.copy()
                input_layer = None
                try:
                    input_layer = sdf.spatial.to_feature_collection()
                except:
                    warnings.warn("Dataframe is not spatial, Rasters and distance layers will not work")

            if input_layer is not None and distance_layers:
                # Use proximity tool
                print("Calculating Distances.")
                count = 1
                for distance_layer in distance_layers:
                    output = arcgis.features.use_proximity.find_nearest(input_layer, distance_layer, max_count=1)
                    connecting_df = output['connecting_lines_layer'].query().sdf
                    near_dist = []

                    for i in range(len(connecting_df)):
                        near_dist.append(connecting_df.iloc[i]['Total_Miles'])

                    sdf[f'NEAR_DIST_{count}'] = near_dist
                    count = count + 1

            # Process Raster Data to get information.
            rasters_data = {}

            if input_layer is not None:
                original_points = []
                for i in range(len(sdf)):
                    original_points.append(sdf.iloc[i]["SHAPE"])

                input_layer_spatial_reference = sdf.spatial._sr
                for raster in rasters:
                    raster_type = 0

                    raster_calc = TabularDataObject._mean_of

                    if isinstance(raster, tuple):
                        if isinstance(raster[1], bool):
                            if raster[1] is True:
                                raster_type = 1
                                raster_calc = TabularDataObject._majority_of
                            if len(raster) > 2:
                                raster_calc = TabularDataObject._get_calc(raster_type, raster[2])
                        else:
                            raster_calc = TabularDataObject._get_calc(raster_type, raster[1])

                        raster = raster[0]

                    for i in range(raster.band_count):
                        if i == 0:
                            rasters_data[raster.name] = []
                        else:
                            rasters_data[raster.name + f'_{i}'] = []

                    shape_objects_transformed = arcgis.geometry.project(original_points, input_layer_spatial_reference,
                                                                        raster.extent['spatialReference'])
                    for shape in shape_objects_transformed:
                        shape['spatialReference'] = raster.extent['spatialReference']
                        if isinstance(shape, arcgis.geometry._types.Point):
                            raster_value = raster.read(origin_coordinate=(shape['x'], shape['y']), ncols=1, nrows=1)
                            value = raster_value[0][0]
                        elif isinstance(shape, arcgis.geometry._types.Polygon):
                            xmin, ymin, xmax, ymax = shape.extent
                            start_x, start_y = xmin + (raster.mean_cell_width / 2), ymin + (raster.mean_cell_height / 2)
                            values = []
                            while start_y < ymax:
                                while start_x < xmax:
                                    if shape.contains(arcgis.geometry._types.Point(
                                            {'x': start_x, 'y': start_y, 'sr': raster.extent['spatialReference']})):
                                        raster_read = raster.read(origin_coordinate=(start_x - raster.mean_cell_width, start_y),
                                                    ncols=1, nrows=1)[0][0]
                                        if len(values) == 0:
                                            for band in raster_read:
                                                values.append([band])
                                        else:
                                            index = 0
                                            for band_value in raster_read:
                                                values[index].append(band_value)
                                                index = index + 1

                                    start_x = start_x + raster.mean_cell_width
                                start_y = start_y + raster.mean_cell_height
                                start_x = xmin + (raster.mean_cell_width / 2)

                            if len(values) == 0:
                                raster_read = raster.read(origin_coordinate=(shape.true_centroid['x'] - raster.mean_cell_width, shape.true_centroid['y']), ncols=1,
                                                nrows=1)[0][0]
                                for band_value in raster_read:
                                    values.append([band_value])

                            value = raster_calc(values)
                        else:
                            raise Exception("Input features can be point or polygon only.")

                        for i in range(len(value)):
                            if i == 0:
                                rasters_data[raster.name].append(value[i])
                            else:
                                rasters_data[raster.name + f'_{i}'].append(value[i])

            # Append Raster data to sdf
            for key, value in rasters_data.items():
                sdf[key] = value
        else:
            try:
                import arcpy
            except:
                raise Exception("This function requires arcpy.")

            try:
                import pandas as pd
            except:
                raise Exception("This function requires pandas.")

            raster = rasters[0]
            if isinstance(raster, tuple):
                raster = raster[0]

            try:
                arcpy.env.outputCoordinateSystem = raster.extent['spatialReference']['wkt']
            except:
                arcpy.env.outputCoordinateSystem = raster.extent['spatialReference']['wkid']

            xmin = raster.extent['xmin']
            xmax = raster.extent['xmax']
            ymin = raster.extent['ymin']
            ymax = raster.extent['ymax']
            min_cell_size_x = raster.mean_cell_width
            min_cell_size_y = raster.mean_cell_height

            default_sr = raster.extent['spatialReference']

            for raster in rasters:
                if isinstance(raster, tuple):
                    raster = raster[0]

                point_upper = arcgis.geometry.Point(
                    {'x': raster.extent['xmin'], 'y': raster.extent['ymax'], 'sr': raster.extent['spatialReference']})
                point_lower = arcgis.geometry.Point(
                    {'x': raster.extent['xmax'], 'y': raster.extent['ymin'], 'sr': raster.extent['spatialReference']})
                cell_size = arcgis.geometry.Point(
                    {'x': raster.mean_cell_width, 'y': raster.mean_cell_height,
                     'sr': raster.extent['spatialReference']})

                points = arcgis.geometry.project([point_upper, point_lower, cell_size],
                                                 raster.extent['spatialReference'],
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

            max_raster_columns = int(abs(math.ceil((xmax - xmin) / min_cell_size_x)))
            max_raster_rows = int(abs(math.ceil((ymax - ymin) / min_cell_size_y)))

            point_upper = arcgis.geometry.Point({'x': xmin, 'y': ymax, 'sr': default_sr})
            cell_size = arcgis.geometry.Point({'x': min_cell_size_x, 'y': min_cell_size_y, 'sr': default_sr})

            raster_data = {}
            for raster in rasters:
                if isinstance(raster, tuple):
                    raster = raster[0]
                field_name = raster.name
                point_upper_translated = \
                arcgis.geometry.project([point_upper], default_sr, raster.extent['spatialReference'])[0]
                cell_size_translated = \
                arcgis.geometry.project([cell_size], default_sr, raster.extent['spatialReference'])[0]
                raster_read = raster.read(
                    origin_coordinate=(point_upper_translated.x, point_upper_translated.y),
                    ncols=max_raster_columns, nrows=max_raster_rows,
                    cell_size=(cell_size_translated.x, cell_size_translated.y))

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

            sdf = pd.DataFrame.from_dict(raster_data)

        if date_field:
            try:
                add_datepart(sdf, date_field)
            except:
                pass

        if index_field in list(sdf.columns.values):
            index_data = sdf[index_field].values

        return sdf, index_data

    @staticmethod
    def _prepare_databunch(
        dataframe,
        fields_mapping,
        procs=None,
        validation_indexes=[],
        batch_size=64
    ):

        if procs is None:
            procs = [Categorify, Normalize]
            fm = FillMissing(fields_mapping['categorical_variables'], fields_mapping['continuous_variables'])
            fm.add_col = False
            fm(dataframe)

        temp_file = tempfile.NamedTemporaryFile().name

        kwargs_variables = {'num_workers': 0} if sys.platform == 'win32' else {}

        kwargs_variables['cat_names'] = fields_mapping['categorical_variables']
        kwargs_variables['cont_names'] = fields_mapping['continuous_variables']
        kwargs_variables['bs'] = batch_size

        if hasattr(arcgis, "env") and getattr(arcgis.env, "_processorType", "") == "CPU":
            kwargs_variables["device"] = torch.device('cpu')

        data_bunch = TabularDataBunch.from_df(
            temp_file,
            dataframe,
            fields_mapping['dependent_variable'],
            procs=procs,
            valid_idx=validation_indexes,
            **kwargs_variables
        )

        return data_bunch

    @classmethod
    def _empty(cls, categorical_variables, continuous_variables, dependent_variable, encoder_mapping, procs=None):
        class_object = cls()
        class_object._dependent_variable = dependent_variable
        class_object._continuous_variables = continuous_variables
        class_object._categorical_variables = categorical_variables
        class_object._encoder_mapping = encoder_mapping
        class_object._is_empty = True
        class_object._procs = procs
        class_object.path = Path(os.path.abspath('.'))

        return class_object
