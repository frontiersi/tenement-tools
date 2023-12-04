import xarray as xr
from shared import tools
from arc.toolbox.common_functions.common_functions import check_xarray
class CogShiftBase:
    def __init__(self, in_path=None, shift_x=None, shift_y=None):
        # lazy load
        self.ds = xr.open_dataset(in_path)
        self.ds_loaded = None
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.messages = []
        self.check_netcdf_dataset()
        
    def check_netcdf_dataset(self):
        self.messages.append(check_xarray(self.ds, levels={0}))
    def full_load(self):
        self.ds_loaded = self.ds.load()

    def check_shift_vals(self, shift_x, shift_y):
        if shift_x is None:
            shift_x = self.shift_x
        if shift_y is not None:
            shift_y = self.shift_y
        return shift_x, shift_y

    def apply_shift(self, shift_x=None, shift_y=None):
        if self.ds_loaded is not None:
            # use set values if not given
            shift_x, shift_y = self.check_shift_vals(shift_x, shift_y)

            if isinstance(shift_x, (float, int)) is False:
                shift_x = 0
            if isinstance(shift_y, (float, int)) is False:
                shift_y = 0
            if shift_x == 0 and shift_y == 0:
                print('No shift applied')
            else:
                self.ds_loaded['x'] += float(shift_x)
                self.ds_loaded['y'] += float(shift_y)

    def revert_shift(self, shift_x=None, shift_y=None):
        if self.ds_loaded is not None:
            shift_x, shift_y = self.check_shift_vals(shift_x, shift_y)
            self.ds_loaded['x'] -= float(shift_x)
            self.ds_loaded['y'] -= float(shift_y)

    def export_as_nc(self, out_name):
        if self.ds_loaded is not None:
            tools.export_xr_as_nc(ds=self.ds_loaded, filename=out_name)
        
            
