

import numpy as np
from netCDF4 import Dataset
import os
import xarray as xr



global base_path
base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
filename = os.path.join(base_path, "60.128_1979-1980.nc")
Dataset(filename, mode='r')

ncdf = xr.open_dataset(filename, decode_cf=True, decode_coords=True)
ncdf.data_vars
dataframe = ncdf.to_dataframe()
marray    = ncdf.to_array(filename)
marray = marray.rename({filename:'pv'})
marray
print marray.dims
print marray.shape
normalarray = np.squeeze(marray) # this short of procedure builds a non-xarray array!
print marray.shape

# --------------------------------------- #
# INDEXING
# --------------------------------------- #
# by integer label, like numpy
# retrieve values on lat/lon grid on timestep 0 and 1 like numpy
marray[0,0:2,:,:].values # give lon,lat at timestep 0
marray[0,0].shape
marray[0]
marray.dims

# index by coordinate label (in xarray jargon: coordinate == positional label)
# retrieve values on lat/lon grid on timestep 0 and 1 with coordinates
# in other words, retrieving array element values based on values (== label) along dimension
marray[0].loc['1979-06-01':'1979-07-01'].values
marray['time'].loc['1979-06-01':'1980-06-01'].values


marray['time'] # gives values of dimension
marray.coords['time'] # gives values of dimension
marray['pv'].values #<- this is not a dimension, so output = 0

# or pick out values based on integer:
marray.isel(time=2)
# use slicing : cut data in half up to that point ( pick 0 up to 2 and slice array )
marray.isel(time=slice(2)).values
marray.sel(time=slice('1979-06-01','1980-06-01'))
# look up values along latitude
marray.sel(latitude=[50,52.5])
# you can also look up nearest:
marray.sel(latitude=[51], method='nearest')
# and even set up tolerance limits:
marray.sel(latitude=[51], method='nearest', tolerance=1)



# retrieve axis:
marray.coords['latitude']
marray['latitude'] # gives time axis
marray['time']





marray.loc()



def calc_anomaly(filename):
    # load in file
    Dataset(filename, mode='r')
    # calculate climatology
