# import numpy as np
# from netCDF4 import Dataset
# import os
# import xarray as xr
# import argparse
# import sys
# import os
# import IPython
# %run retrieve_ERA_i_field.py


def calc_anomaly(cls, decode_cf=True, decode_coords=True):
    import xarray as xr
    import os
    # load in file
    file_path = os.path.join(cls.base_path, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))

    print("dimensions {}".format(cls.name, marray.shape))
    print(marray.dims)
    clim = marray.mean(dim='time')
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time")
    return clim, anom, upperquan