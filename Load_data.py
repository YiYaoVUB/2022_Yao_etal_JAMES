import netCDF4 as nc
import numpy

class Data_from_nc:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_variable(self, var):        #load the variable in the data
        file_obj = nc.Dataset(self.data_dir)
        data = file_obj.variables[var]
        var_data = numpy.array(data)
        return var_data