import os
import tempfile
import traceback
import json
import warnings
import pickle
import math
from pathlib import Path

HAS_FASTAI = True
import_exception=None

import arcgis
from arcgis.features import FeatureLayer

try:
    from ._arcgis_model import ArcGISModel, _raise_fastai_import_error
    from fastai.basic_train import Learner, load_learner
    from fastprogress.fastprogress import progress_bar
    from .._utils.tabular_data import TabularDataObject
    from fastai.torch_core import split_model_idx
    import torch
    from fastai.metrics import r2_score
    from ._tsmodel_archs._InceptionTime import _TSInceptionTime
    from ._tsmodel_archs._Resnet import _TSResNet
    from ._tsmodel_archs._ResCNN import _TSResCNN
    from ._tsmodel_archs._FCN import _TSFCN
    from ._tsmodel_archs._LSTM import _TSLSTM
    from .._utils.TSData import To3dTensor, ToTensor
    _model_arch = {
        'inceptiontime': _TSInceptionTime,
        'resnet': _TSResNet,
        'rescnn': _TSResCNN,
        'fcn': _TSFCN,
        'lstm': _TSLSTM
    }
except Exception as e:
    import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    HAS_FASTAI = False
    _model_arch = {}

HAS_NUMPY = True
try:
    import numpy as np
except:
    HAS_NUMPY = False

_PROTOCOL_LEVEL = 2

HAS_PANDAS = True
try:
    import pandas as pd
except:
    HAS_PANDAS = False


def _get_model_from_path(pretrained_path):
    learn = load_learner(
        os.path.dirname(pretrained_path),
        os.path.basename(pretrained_path).split('.')[0] + "_exported.pth",
        no_check=True
    )

    return learn


class TimeSeriesModel(ArcGISModel):
    """
    Creates a TimeSeriesModel Object.
    Based on the Fast.ai's https://github.com/timeseriesAI/timeseriesAI

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    data                    Required TabularDataObject. Returned data object from
                            `prepare_tabulardata` function.
    ---------------------   -------------------------------------------
    seq_len                 Required Integer. Sequence Length for the series.
                            In case of raster only, seq_len = number of rasters,
                            any other passed value will be ignored.
    ---------------------   -------------------------------------------
    model_arch              Optional string. Model Architecture.
                            Allowed "InceptionTime", "ResCNN",
                            "Resnet", "FCN"
    ---------------------   -------------------------------------------
    **kwargs                Optional kwargs.
    =====================   ===========================================

    :returns: `TimeSeriesModel` Object
    """

    def __init__(self, data, seq_len, model_arch='InceptionTime', **kwargs):

        data_bunch = None

        if not data._is_empty:
            data_bunch = data._time_series_bunch(seq_len)

        super().__init__(data, None)

        if not data_bunch:
            self.learn = _get_model_from_path(kwargs.get('pretrained_path'))
        elif kwargs.get('pretrained_path'):
            self.learn = _get_model_from_path(kwargs.get('pretrained_path'))
            self.learn.data = data_bunch
        else:
            if not _model_arch.get(model_arch.lower()):
                raise Exception("Invalid model architecture")

            model_arch_ob = _model_arch.get(model_arch.lower())
            if model_arch.lower() == 'lstm':
                model = model_arch_ob(data_bunch.features, data_bunch.c, self._device, **kwargs).to(self._device)
            else:
                if model_arch.lower() in ['resnet', 'fcn']:
                    kwargs['device'] = self._device
                model = model_arch_ob(data_bunch.features, data_bunch.c, **kwargs).to(self._device)
                if model_arch.lower() in ['resnet', 'fcn']:
                    del kwargs['device']
            self.learn = Learner(data_bunch, model, path=data.path)
            self.learn.data = data_bunch

        self.learn.layer_groups = split_model_idx(self.learn.model, [1])
        self._model_arch = model_arch.lower()
        if kwargs.get('pretrained_path'):
            del kwargs['pretrained_path']
        self._kwargs = kwargs
        self._seq_len = seq_len

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s>' % (type(self).__name__)

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a TimeSeriesModel Object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from `prepare_tabulardata` function or None for
                                inferencing.
        =====================   ===========================================

        :returns: `TimeSeriesModel` Object
        """
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        emd_path = Path(emd_path)
        with open(emd_path) as f:
            emd = json.load(f)

        dependent_variable = emd['dependent_variable']
        categorical_variables = emd['categorical_variables']
        continuous_variables = emd['continuous_variables']

        _is_classification = False
        if emd['_is_classification'] == "classification":
            _is_classification = True

        model_params = emd['model_params']
        model_arch = emd['model_arch']
        seq_len = emd['seq_len']
        index_field = emd.get('index_field', None)
        # encoder_path = os.path.join(os.path.dirname(emd_path),
        #                             os.path.basename(emd_path).split('.')[0] + '_encoders.pkl')

        scaler_path = os.path.join(os.path.dirname(emd_path),
                                    os.path.basename(emd_path).split('.')[0] + '_scaler.pkl')

        encoder_mapping = None
        scaler = None

        # if os.path.exists(encoder_path):
        #     with open(encoder_path, 'rb') as f:
        #         encoder_mapping = pickle.loads(f.read())

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.loads(f.read())

        if data is None:
            data = TabularDataObject._empty(categorical_variables, continuous_variables, dependent_variable, encoder_mapping)
            data._is_classification = _is_classification
            data._column_transforms_mapping = scaler

            if index_field is not None:
                data._index_field = index_field

            class_object = cls(data, seq_len, model_arch=model_arch, pretrained_path=emd_path, **model_params)
            class_object._data.emd = emd
            class_object._data.emd_path = emd_path
            return class_object

        return cls(data, seq_len, model_arch=model_arch, pretrained_path=emd_path, **model_params)

    def save(self, name_or_path, framework='PyTorch', publish=False, gis=None, save_optimizer=False, **kwargs):
        """
        Saves the model weights, creates an Esri Model Definition and Deep
        Learning Package zip for deployment to Image Server or ArcGIS Pro.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Folder path to save the model.
        ---------------------   -------------------------------------------
        framework               Optional string. Defines the framework of the
                                model. (Only supported by ``SingleShotDetector``, currently.)
                                If framework used is ``TF-ONNX``, ``batch_size`` can be
                                passed as an optional keyword argument.

                                Framework choice: 'PyTorch' and 'TF-ONNX'
        ---------------------   -------------------------------------------
        publish                 Optional boolean. Publishes the DLPK as an item.
        ---------------------   -------------------------------------------
        gis                     Optional GIS Object. Used for publishing the item.
                                If not specified then active gis user is taken.
        ---------------------   -------------------------------------------
        save_optimizer          Optional boolean. Used for saving the model-optimizer
                                state along with the model. Default is set to False
        ---------------------   -------------------------------------------
        kwargs                  Optional Parameters:
                                Boolean `overwrite` if True, it will overwrite
                                the item on ArcGIS Online/Enterprise, default False.
        =====================   ===========================================
        """

        if '\\' in name_or_path or '/' in name_or_path:
            path = os.path.abspath(name_or_path)
        else:
            path = os.path.join(self._data.path, 'models', name_or_path)
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))

        if not os.path.exists(path):
            os.mkdir(path)

        base_file_name = os.path.basename(path)

        TimeSeriesModel._save_encoders(self._data._column_transforms_mapping, path, base_file_name)

        self.learn.export(os.path.join(path, os.path.basename(path) + '_exported.pth'))
        from IPython.utils import io
        with io.capture_output() as captured:
            super().save(path, framework, publish, gis, save_optimizer=save_optimizer, **kwargs)

        return Path(path)

    @staticmethod
    def _save_encoders(scaler, path, base_file_name):
        # if not encoder_mapping:
        #     return

        # encoder_file = os.path.join(path, base_file_name + '_encoders.pkl')
        # with open(encoder_file, 'wb') as f:
        #     f.write(pickle.dumps(encoder_mapping, protocol=_PROTOCOL_LEVEL))

        scaler_file = os.path.join(path, base_file_name + '_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            f.write(pickle.dumps(scaler, protocol=_PROTOCOL_LEVEL))

    @property
    def _model_metrics(self):
        # from IPython.utils import io
        # with io.capture_output() as captured:
        #     score = self.score()

        return {'score': self.score()}

    def _get_emd_params(self, save_inference_file):
        _emd_template = {}
        _emd_template["ModelType"] = "TimeSeriesModel"
        _emd_template["model_arch"] = self._model_arch
        _emd_template['model_params'] = self._kwargs
        _emd_template['seq_len'] = self._seq_len
        _emd_template["dependent_variable"] = self._data._dependent_variable
        _emd_template["categorical_variables"] = self._data._categorical_variables
        _emd_template["continuous_variables"] = self._data._continuous_variables

        if self._data._index_field:
            _emd_template['index_field'] = self._data._index_field

        _emd_template['_is_classification'] = "classification" if self._data._is_classification else "regression"

        return _emd_template

    def _predict(self, orig_sequence):
        sequence = np.array(orig_sequence, dtype='float32')
        if sequence.shape[0] == 1:
            seq_arr = To3dTensor(orig_sequence).to(self._device)
        else:
            seq_arr = ToTensor(np.expand_dims(np.array(sequence, dtype='float32'), axis=0)).to(self._device)
        model = self.learn.model
        model.eval()

        with torch.no_grad():
            output = model(seq_arr).item()

        return output

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
            match_field_names=None,
            number_of_predictions=None
    ):
        """

        Predict on data from feature layer and or raster data.

        =================================   =========================================================================
        **Argument**                        **Description**
        ---------------------------------   -------------------------------------------------------------------------
        input_features                      Optional Feature Layer or spatially enabled dataframe.
                                            Contains features with location of the input data.
                                            Required if prediction_type is 'features' or 'dataframe'
        ---------------------------------   -------------------------------------------------------------------------
        explanatory_rasters                 Optional list of Raster Objects.
                                            Required if prediction_type is 'rasters'
        ---------------------------------   -------------------------------------------------------------------------
        datefield                           Optional field_name.
                                            This field contains the date in the input_features.
                                            The field type can be a string or date time field.
                                            If specified, the field will be split into
                                            Year, month, week, day, dayofweek, dayofyear,
                                            is_month_end, is_month_start, is_quarter_end,
                                            is_quarter_start, is_year_end, is_year_start,
                                            hour, minute, second, elapsed and these will be added
                                            to the prepared data as columns.
                                            All fields other than elapsed and dayofyear are treated
                                            as categorical.
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
                                            Set 'features' or 'dataframe' to make output predictions.
        ---------------------------------   -------------------------------------------------------------------------
        output_raster_path                  Optional path. Required when prediction_type='raster', saves
                                            the output raster to this path.
        ---------------------------------   -------------------------------------------------------------------------
        match_field_names                   Optional string.
                                            Specify mapping of the original training set with prediction set.
        ---------------------------------   -------------------------------------------------------------------------
        number_of_predictions               Optional int for univariate time series.
                                            Specify the number of predictions to make, adds new rows to the dataframe.
                                            For multivariate or if None, it expects the dataframe to have empty rows.
                                            For prediction_type='raster', a new raster is created.
        =================================   =========================================================================

        :returns Feature Layer/dataframe if prediction_type='features'/'dataframe', else returns True and saves output
        raster at the specified path.
        """

        rasters = explanatory_rasters if explanatory_rasters else []
        if prediction_type in ['features', 'dataframe']:

            if input_features is None:
                raise Exception("Feature Layer required for predict_features=True")

            gis = gis if gis else arcgis.env.active_gis
            return self._predict_features(input_features, rasters, datefield, distance_features, output_layer_name, gis,
                                          match_field_names, number_of_predictions, prediction_type)
        else:
            if not rasters:
                raise Exception("Rasters required for predict_features=False")

            if not output_raster_path:
                raise Exception("Please specify output_raster_folder_path to save the output.")

            return self._predict_rasters(output_raster_path, rasters, match_field_names)

    def _predict_rasters(self, output_raster_path, rasters, match_field_names=None):
        if not os.path.exists(os.path.dirname(output_raster_path)):
            raise Exception("Output directory doesn't exist")

        if os.path.exists(output_raster_path):
            raise Exception("Output Folder already exists")

        try:
            import arcpy
        except:
            raise Exception("This function requires arcpy.")

        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        try:
            import pandas as pd
        except:
            raise Exception("This function requires pandas.")

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

        max_raster_columns = int(abs(math.ceil((xmax - xmin) / min_cell_size_x)))
        max_raster_rows = int(abs(math.ceil((ymax - ymin) / min_cell_size_y)))

        point_upper = arcgis.geometry.Point({'x': xmin, 'y': ymax, 'sr': default_sr})
        cell_size = arcgis.geometry.Point({'x': min_cell_size_x, 'y': min_cell_size_y, 'sr': default_sr})

        raster_data = {}
        for raster in rasters:
            field_name = raster.name
            point_upper_translated = \
            arcgis.geometry.project([point_upper], default_sr, raster.extent['spatialReference'])[0]
            cell_size_translated = arcgis.geometry.project([cell_size], default_sr, raster.extent['spatialReference'])[
                0]
            if field_name in fields_needed:
                raster_read = raster.read(
                    origin_coordinate=(point_upper_translated.x, point_upper_translated.y), ncols=max_raster_columns,
                    nrows=max_raster_rows, cell_size=(cell_size_translated.x, cell_size_translated.y))
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
                raster_read = raster.read(
                    origin_coordinate=(point_upper_translated.x, point_upper_translated.y), ncols=max_raster_columns,
                    nrows=max_raster_rows, cell_size=(cell_size_translated.x, cell_size_translated.y))
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

        length_values = len(raster_data[list(raster_data.keys())[0]])
        processed_output = []
        for i in progress_bar(range(length_values)):
            processed_row = []
            for raster_name in sorted(raster_data.keys()):
                processed_row.append(raster_data[raster_name][i])
            processed_output.append(self._predict([processed_row]))

        processed_numpy = np.array(processed_output, dtype='float64')
        processed_numpy = processed_numpy.reshape([max_raster_rows, max_raster_columns])
        processed_raster = arcpy.NumPyArrayToRaster(processed_numpy, arcpy.Point(xmin, ymin),
                                                    x_cell_size=min_cell_size_x, y_cell_size=min_cell_size_y)
        processed_raster.save(output_raster_path)

        return True

    def _predict_features(self, input_features, rasters=None, datefield=None, distance_features=None, output_layer_name='Prediction Layer', gis=None, match_field_names=None, number_of_predictions=None, prediction_type='features'):
        if not HAS_PANDAS:
            raise Exception("This function requires pandas library")

        if isinstance(input_features, FeatureLayer):
            orig_dataframe = input_features.query().sdf
        else:
            orig_dataframe = input_features.copy()

        if match_field_names is None:
            match_field_names = {}

        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        if number_of_predictions is not None and number_of_predictions > 0:
            delta = None
            index_field_name = None
            end_value = None
            if self._data._index_field is not None:
                index_field_name = match_field_names.get(self._data._index_field, self._data._index_field)
                if index_field_name in list(orig_dataframe.columns) and is_datetime(orig_dataframe[index_field_name]):
                    delta = orig_dataframe[index_field_name].iloc[1] - orig_dataframe[index_field_name].iloc[0]
                    end_value = None
                    if delta is not None:
                        end_value = orig_dataframe[index_field_name].iloc[len(orig_dataframe)-1]
            for i in range(number_of_predictions):
                orig_dataframe = orig_dataframe.append(pd.Series(), ignore_index=True)
                if delta is not None:
                    orig_dataframe.loc[len(orig_dataframe)-1, index_field_name] = end_value + delta.to_timedelta64()
                    end_value = end_value + delta.to_timedelta64()

        if match_field_names and match_field_names.get(self._data._dependent_variable):
            prediction_sequence_orig = orig_dataframe[match_field_names.get(self._data._dependent_variable)]
        else:
            prediction_sequence_orig = orig_dataframe[self._data._dependent_variable]

        dataframe = orig_dataframe.copy()

        fields_needed = self._data._categorical_variables + self._data._continuous_variables + [self._data._dependent_variable]
        distance_feature_layers = distance_features if distance_features else []

        continuous_variables = self._data._continuous_variables

        feature_layer_columns = []
        for column in dataframe.columns:
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
                orig_dataframe,
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

        processed_dataframe = processed_dataframe.reindex(sorted(processed_dataframe.columns), axis=1)

        index = self._seq_len
        processed_dataframe[self._data._dependent_variable] = processed_dataframe[self._data._dependent_variable].replace(r'^\s*$', np.nan, regex=True)

        processed_dataframe_transform = processed_dataframe.copy()

        for col in list(processed_dataframe.columns):
            transformed_data = processed_dataframe[col]
            for transform in self._data._column_transforms_mapping.get(col, []):
                transformed_data = transform.fit_transform(
                    np.array(transformed_data, dtype=processed_dataframe[col].dtype).reshape(-1, 1))
                transformed_data = transformed_data.squeeze(1)
            processed_dataframe_transform[col] = np.array(transformed_data, dtype=processed_dataframe[col].dtype)

        big_bunch = []
        prediction_sequence_list = None
        processed_dataframe_transform = processed_dataframe_transform.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for col in range(len(processed_dataframe.columns.values)):
                if list(processed_dataframe.columns.values)[col] == self._data._dependent_variable:
                    # big_bunch.append(prediction_sequence_list[index - self._seq_len:(index - self._seq_len + self._seq_len)])
                    prediction_sequence_list = processed_dataframe_transform[:, col]
                    big_bunch.append(prediction_sequence_list[0:self._seq_len])
                else:
                    big_bunch.append(processed_dataframe_transform[:, col][0:self._seq_len])

        if len(prediction_sequence_list) < self._seq_len:
            raise Exception("Basic Sequence not found!")

        while index < len(prediction_sequence_list):
            if prediction_sequence_list[index] in ["", None, "null", "None"] or np.isnan(prediction_sequence_list[index]):
                value = self._predict(np.array(big_bunch))
                prediction_sequence_list[index] = value

            index = index + 1
            big_bunch = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for col in range(len(processed_dataframe.columns.values)):
                    if list(processed_dataframe.columns.values)[col] == self._data._dependent_variable:
                        big_bunch.append(prediction_sequence_list[index - self._seq_len:(index - self._seq_len+self._seq_len)])
                    else:
                        big_bunch.append(processed_dataframe_transform[:, col][index - self._seq_len:(index - self._seq_len+self._seq_len)])

        transformed_results = prediction_sequence_list
        if self._data._column_transforms_mapping.get(self._data._dependent_variable):
            for transform in self._data._column_transforms_mapping.get(self._data._dependent_variable):
                transformed_results = transform.inverse_transform(np.array(transformed_results).reshape(-1, 1))
                transformed_results = transformed_results.squeeze(1)

        orig_dataframe[self._data._dependent_variable + "_results"] = transformed_results
        if prediction_type == "dataframe":
            return orig_dataframe

        if 'SHAPE' in list(orig_dataframe.columns):
            orig_dataframe.spatial.to_featurelayer(output_layer_name, gis)
        else:
            import tempfile
            try:
                import openpyxl
            except:
                warnings.warn("This environment does not have openpyxl installed. Please install this dependency using the command: pip install openpyxl")
            with tempfile.TemporaryDirectory() as tmpdir:
                table_file = os.path.join(tmpdir, output_layer_name + '.xlsx')
                orig_dataframe.to_excel(table_file, index=False, header=True)
                online_table = gis.content.add({'type': 'Microsoft Excel', 'overwrite': True}, table_file)
                return online_table.publish(overwrite=True)

    def score(self):
        """
        :returns R2 score for regression model and Accuracy for classification model.
        """

        self._check_requisites()
        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        model = self.learn.model

        model.eval()

        dl = self.learn.data.valid_dl

        targets = []
        predictions = []
        for i in range(len(dl.x.items)):
            prediction = self._predict(dl.x.items[i])
            target = dl.y.items[i]
            targets.append(target)
            predictions.append(prediction)

        transformed_results = targets
        if self._data._column_transforms_mapping.get(self._data._dependent_variable):
            for transform in self._data._column_transforms_mapping.get(self._data._dependent_variable):
                transformed_results = transform.inverse_transform(np.array(transformed_results).reshape(-1, 1))
                transformed_results = transformed_results.squeeze(1)
        targets = transformed_results

        transformed_results = predictions
        if self._data._column_transforms_mapping.get(self._data._dependent_variable):
            for transform in self._data._column_transforms_mapping.get(self._data._dependent_variable):
                transformed_results = transform.inverse_transform(np.array(transformed_results).reshape(-1, 1))
                transformed_results = transformed_results.squeeze(1)
        predictions = transformed_results

        if self._data._is_classification:
            return (np.array(predictions)==np.array(targets)).mean()
        else:
            targets = torch.tensor(np.array(targets, dtype='float64')).to(self._device)
            predictions = torch.tensor(np.array(predictions, dtype='float64')).to(self._device)
            return float(r2_score(predictions, targets))

    def show_results(self, rows=5):
        """
        Prints the graph with predictions.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        rows                    Optional Integer.
                                Number of rows to print.
        =====================   ===========================================
        """
        self._check_requisites()

        if not HAS_NUMPY:
            raise Exception("This function requires numpy.")

        model = self.learn.model

        model.eval()

        dl = self.learn.data.valid_dl

        targets = []
        predictions = []
        sequence = []
        for i in range(len(dl.x.items)):
            prediction = self._predict(dl.x.items[i])
            target = dl.y.items[i]
            targets.append(target)
            predictions.append(prediction)
            sequence.append(dl.x.items[i])

        targets = np.array(targets, dtype='float64')
        predictions = np.array(predictions, dtype='float64')

        transformed_results = targets
        if self._data._column_transforms_mapping.get(self._data._dependent_variable):
            for transform in self._data._column_transforms_mapping.get(self._data._dependent_variable):
                transformed_results = transform.inverse_transform(np.array(transformed_results).reshape(-1, 1))
                transformed_results = transformed_results.squeeze(1)

        targets_inversed = transformed_results

        transformed_results = predictions
        if self._data._column_transforms_mapping.get(self._data._dependent_variable):
            for transform in self._data._column_transforms_mapping.get(self._data._dependent_variable):
                transformed_results = transform.inverse_transform(np.array(transformed_results).reshape(-1, 1))
                transformed_results = transformed_results.squeeze(1)

        predictions_inversed = transformed_results

        column_transforms_mapping = self._data._column_transforms_mapping.copy()
        # del column_transforms_mapping[self._data._dependent_variable]
        sequence_inversed = []
        keys = list(column_transforms_mapping.keys())
        for seq in sequence:
            seq_inverse = []
            index = 0
            for col_data in seq:
                transformed_data = col_data
                if len(keys) > index:
                    for transform in column_transforms_mapping.get(keys[index]):
                        transformed_data = transform.inverse_transform(np.array(transformed_data).reshape(-1, 1))
                        transformed_data = transformed_data.squeeze(1)

                seq_inverse.append(transformed_data)
                index = index + 1

            sequence_inversed.append(seq_inverse)

        if self._data._index_seq is not None:
            validation_index_seq = self._data._index_seq.take(self._data._validation_indexes_ts, axis=0)
        else:
            validation_index_seq = None

        import matplotlib.pyplot as plt
        n_items = rows
        if n_items > len(targets_inversed):
            n_items = len(targets_inversed)

        rows = int(n_items)

        fig, axs = plt.subplots(rows, 2, figsize=(10, 10))
        fig.suptitle('Ground truth vs Predictions\n\n', fontsize=16)

        for i in range(rows):
            for seq_plot in sequence_inversed[i]:
                if self._data._index_seq is not None:
                    axs[i, 0].plot(validation_index_seq[i], seq_plot)
                    axs[i, 1].plot(validation_index_seq[i], seq_plot)
                else:
                    axs[i, 0].plot(seq_plot)
                    axs[i, 1].plot(seq_plot)

                axs[i, 0].tick_params(axis="x", labelrotation=60)
                axs[i, 1].tick_params(axis="x", labelrotation=60)

            axs[i, 0].set_title(targets_inversed[i])
            axs[i, 1].set_title(predictions_inversed[i])

        plt.tight_layout()
        plt.show()