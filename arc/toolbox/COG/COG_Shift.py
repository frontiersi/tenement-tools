import arcpy
import warnings
import xarray as xr
import dask
from shared import tools
from COG_shift_base import CogShiftBase

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=dask.array.core.PerformanceWarning)

class COGShift:
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'COG Shift'
        self.description = 'Shifts the output of the COG Fetch tool by ' \
                           'a specified number of metres in space to correct ' \
                           'for offset issues.'
        self.canRunInBackground = False

    @staticmethod
    def getParameterInfo():
        """
        Set up UI parameters / controls.
        """

        # input netcdf data file
        par_in_nc_file = arcpy.Parameter(
            displayName='Input satellite NetCDF file',
            name='in_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Input')
        par_in_nc_file.filter.list = ['nc']

        # output netcdf data file
        par_out_nc_file = arcpy.Parameter(
            displayName='Output shifted NetCDF file',
            name='out_nc_file',
            datatype='DEFile',
            parameterType='Required',
            direction='Output')
        par_out_nc_file.filter.list = ['nc']

        # shift x
        par_shift_x = arcpy.Parameter(
            displayName='Shift X (metres)',
            name='in_shift_x',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_shift_x.value = 0.0

        # shift y
        par_shift_y = arcpy.Parameter(
            displayName='Shift Y (metres)',
            name='in_shift_y',
            datatype='GPDouble',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_shift_y.value = 0.0

        # combine parameters
        parameters = [
            par_in_nc_file,
            par_out_nc_file,
            par_shift_x,
            par_shift_y
        ]

        return parameters

    @staticmethod
    def isLicensed():
        """Set whether tool is licensed to execute."""
        return True

    def execute(self, parameters, messages):
        """
        Executes the COG Shift module.
        """
        in_nc = parameters[0].valueAsText  # raw input satellite netcdf
        out_nc = parameters[1].valueAsText  # shifted output satellite netcdf
        in_shift_x = parameters[2].value  # shift x
        in_shift_y = parameters[3].value  # shift y

        # # # # #
        # notify user and set up progress bar
        arcpy.AddMessage('Beginning COG Shift.')
        arcpy.SetProgressor(type='step',
                            message='Preparing parameters...',
                            min_range=0, max_range=4)

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Loading and checking netcdf...')
        arcpy.SetProgressorPosition(1)

        dataset = CogShiftBase(in_nc, in_shift_x, in_shift_y)
        for message in dataset.messages:
            arcpy.AddError(message)

        if len(dataset.messages) != 0:
            return

        dataset.full_load()

        # # # # #
        # notify and increment progress bar
        arcpy.SetProgressorLabel('Shifting NetCDF...')
        arcpy.SetProgressorPosition(2)

        dataset.apply_shift()
        # # # # #
        # notify and increment progess bar
        arcpy.SetProgressorLabel('Exporting NetCDF file...')
        arcpy.SetProgressorPosition(3)

        dataset.export_as_nc(out_nc)

        # # # # #
        # clean up variables
        arcpy.SetProgressorLabel('Finalising process...')
        arcpy.SetProgressorPosition(4)

        # close main dataset and del datasets
        dataset.ds.close()
        dataset.ds_loaded.close()
        del dataset

        # notify user
        arcpy.AddMessage('Generated COG Shift successfully.')

        return