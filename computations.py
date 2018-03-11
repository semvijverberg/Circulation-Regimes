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
    import numpy as np
    import os
    # load in file
    file_path = os.path.join(cls.base_path, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))

    print("calc_anomaly called for {}".format(cls.name, marray.shape))
    steps_per_year = len(marray.sel(time=str(cls.startyear))['time'])
    month_index = marray.sel(time=str(cls.startyear))['time.month'].values
    months_string = {1:'jan', 2:'feb', 3:'mar', 4:'april', 5:'may', 6:'june', 7:'juli', 8:'aug', 9:'sep', 10:'okt', 11:'nov', 12:'dec'}
    months=[]
    for keys in month_index:
        months.append(months_string[keys])

    clim = xr.DataArray(np.zeros([steps_per_year, len(marray['latitude']), len(marray['longitude'])]), dims=('time', 'latitude', 'longitude'), coords=[months, marray['latitude'].values,marray['longitude'].values])

    for in_year in range(0,steps_per_year):
        indices = np.arange(in_year, in_year + (cls.endyear - cls.startyear) * steps_per_year, step=steps_per_year)
        print(in_year)
    clim = marray.mean(dim='time')
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time")
    return clim, anom, upperquan

def group_time(array, timestep):

    groups = array.groupby('time.' + str(timestep))

    return