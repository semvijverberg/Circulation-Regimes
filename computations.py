# import numpy as np
# from netCDF4 import Dataset
# import os
# import xarray as xr
# import argparse
# import sys
# import os
# import IPython
# %run retrieve_ERA_i_field.py




def import_array(cls, decode_cf=True, decode_coords=True):
    import xarray as xr
    import numpy as np
    import os
    # load in file
    file_path = os.path.join(cls.base_path, 'input', cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))
    marray.attrs['units'] = cls.units
    for dims in marray.dims:
        if dims == 'lon':
            marray = marray.rename(({'lon': 'longitude'}))
        if dims == 'lat':
            marray = marray.rename(({'lat': 'latitude'}))
        else:
            pass
    marray.attrs['units'] = cls.units
    marray.attrs['dataset'] = cls.dataset
    print("import array {}".format(cls.name))
    if 'units' in marray.time.attrs:
        if marray.time.attrs['units'] == 'months since 1861-01-01':
            print('original timeformat: months since 1861-01-01')
            cls = month_since_to_datetime(marray, cls)
        if marray.time.attrs['units'] == 'hours since 1900-01-01 00:00:0.0':
            marray['time'] = cls.dates_np
            print("Taken numpy dates from what variable function, check time units")
    return marray, cls

def month_since_to_datetime(marray, cls):
    import datetime as datetime
    datelist = [datetime.date(year=cls.startyear, month=1, day=1)]
    year = cls.startyear
    for steps in marray['time'].values[1:]+1:
        step = int(steps % 12)
        datelist.append(datetime.date(year=year, month=step, day=1))
        # print("year is {}, steps is {}".format(year, steps % 12))
        if steps % 12 == 10.:
            year = year + 1
    cls.datelist = datelist
    return cls


def calc_anomaly(marray, cls, q = 0.95):
    import xarray as xr
    import numpy as np
    print("calc_anomaly called for {}".format(cls.name, marray.shape))
    steps_per_year = len(marray.sel(time=str(cls.startyear))['time'])
    month_index = marray.sel(time=str(cls.startyear))['time.month'].values
    months_string = {1:'jan', 2:'feb', 3:'mar', 4:'april', 5:'may', 6:'june', 7:'juli', 8:'aug', 9:'sep', 10:'okt', 11:'nov', 12:'dec'}
    months=[]
    for keys in month_index:
        months.append(months_string[keys])
    months_group = np.tile(months, len(marray['time.year'])/steps_per_year)
    labels = xr.DataArray(months_group, [marray.coords['time']], name='labels')

    clim = marray.groupby(labels).mean('time', keep_attrs=True).rename({'labels': 'time'})
    substract = lambda x, y: (x - y)
    anom = xr.apply_ufunc(substract, marray, np.tile(clim,(1,(cls.endyear+1-cls.startyear),1,1)), keep_attrs=True)
    std = anom.groupby(labels).reduce(np.percentile, dim='time', keep_attrs=True, q=q).rename({'labels': 'time'})

    return clim, anom, std



# clim = xr.DataArray(np.zeros([steps_per_year, len(marray['latitude']), len(marray['longitude'])]), dims=('time', 'latitude', 'longitude'), coords=[months, marray['latitude'].values,marray['longitude'].values])