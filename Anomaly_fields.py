

import numpy as np
from netCDF4 import Dataset
import os


global base_path
base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
filename = os.path.join(base_path, "60.128_1979-1980.nc")

def calc_anomaly(filename):
    # load in file

    # calculate climatology
