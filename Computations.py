import numpy as np
from netCDF4 import Dataset
import os
import xarray as xr
import argparse
import sys
import os

execfile("retrieve_ERA_i_field.py 0")

os.system("retrieve_ERA_i_field.py")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_name', type=str, default="pv",
                        help="What variable?")

    args = parser.parse_args()
    sys.stdout.write(calc_anomaly(args)[1])

def calc_anomaly(args, filename, decode_cf=True, decode_coords=True):
    # load in file
    ncdf = xr.open_dataset(filename, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(filename)
    marray = marray.rename({filename: args.var_name})
    # what is temporal freq of dataset
    # calculate climatology


calc_anomaly(args='pv', filename=filename)
if __name__ == '__main__':
    main()
