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
    months_group = np.tile(months, len(marray['time.year'])/steps_per_year)
    labels = xr.DataArray(months_group, [marray.coords['time']], name='labels')

    clim = np.squeeze(marray.groupby(labels).mean('time'))
    clim = clim.rename({'labels': 'time'})
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time")
    return clim, anom, upperquan

def group_time(array, timestep):

    groups = array.groupby('time.' + str(timestep))

    return


# clim = xr.DataArray(np.zeros([steps_per_year, len(marray['latitude']), len(marray['longitude'])]), dims=('time', 'latitude', 'longitude'), coords=[months, marray['latitude'].values,marray['longitude'].values])