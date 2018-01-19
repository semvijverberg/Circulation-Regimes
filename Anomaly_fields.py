

import numpy as np
from netCDF4 import Dataset
import os
from cdo import *



global base_path
base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
filename = os.path.join(base_path, "60.128_1979-1980.nc")




Dataset(filename, mode='r')

def calc_anomaly(filename):
    # load in file
    Dataset(filename, mode='r')
    # calculate climatology
