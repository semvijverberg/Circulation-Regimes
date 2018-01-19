

import numpy as np
from netCDF4 import Dataset
import os
import xarray as xr



global base_path
base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
filename = os.path.join(base_path, "60.128_1979-1980.nc")
Dataset(filename, mode='r')

ncdf = xr.open_dataset(filename)
ncdf.data_vars
marray = ncdf.to_dataframe()
marray.mean(1)

def calc_anomaly(filename):
    # load in file
    Dataset(filename, mode='r')
    # calculate climatology
