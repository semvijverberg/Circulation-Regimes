from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import os

class Variable:
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of timesteps   :   mnth
    Daily Streams:
    Operational (for surface)   :   oper

    """
    # below is a class variable
    ecmwf_website = 'http://apps.ecmwf.int/codes/grib/param-db'
    base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
    def __init__(self, name, var_cf_code, levtype, lvllist, startyear, endyear, startmonth, endmonth, grid, stream):
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = name
        self.var_cf_code = var_cf_code
        self.lvllist = lvllist
        self.levtype = levtype
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.grid = grid
        self.stream = stream

        start = datetime(self.startyear, self.startmonth, 1)
        end = datetime(self.endyear, self.endmonth, 1)
        datelist = [start.strftime('%Y-%m-%d')]
        while start <= end:
            if start.month < end.month:
                start += timedelta(days=31)
                datelist.append(datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
            else:
                start = datetime(start.year+1, self.startmonth-1, 1)
                datelist.append(datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
        self.datelist = datelist

        filename = '{}_{}-{}_{}_{}'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth).replace(' ', '_')
        filename = filename.replace('/', 'x')
        self.filename = filename +'.nc'
        print self.filename

def retrieve_ERA_i_field(cls):
    import os
    print 'you are retrieving the following dataset: \n'
    print cls.__dict__
    file_path = os.path.join(cls.base_path, cls.filename)
    print file_path
    datestring = "/".join(cls.datelist)
    # !/usr/bin/python
    from ecmwfapi import ECMWFDataServer
    import numpy as np
    from datetime import datetime, timedelta
    import os
    server = ECMWFDataServer()

    if cls.stream == "mnth" or cls.stream == "oper":
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
    elif cls.stream == "moda":
        time = "00:00:00"
    else:
        print "stream is not available"

    if os.path.isfile(path=file_path) == True:
        pass
    else:
        server.retrieve({
            "dataset"   :   "interim",
            "class"     :   "ei",
            "expver"    :   "1",
            "date"      :   datestring,
            "grid"      :   cls.grid,
            "levtype"   :   cls.levtype,
            # "levelist"  :   cls.lvllist,
            "param"     :   cls.var_cf_code,
            "stream"    :   cls.stream,
            # "time"      :   time,
            "type"      :   "an",
            "format"    :   "netcdf",
            "target"    :   file_path,
            })
    return file_path, " You have downloaded variable {} \n stream is set to {} \n all dates: {} \n".format \
        (cls.var_cf_code, cls.stream, datelist)



# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2,5/2,5', stream='moda')


# retrieve_ERA_i_field(temperature)

def calc_anomaly(cls, decode_cf=True, decode_coords=True):

    # load in file
    # file_path = os.path.join(cls.base_path, cls.filename)
    file_path = os.path.join(cls.base_path, "pressure-moda-295-test-netcdf.nc")
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))

    print "dimensions {}".format(cls.name, marray.shape)
    print marray.dims
    clim = marray.mean(dim='time')
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time", keep_attrs=True)
    return clim, anom, upperquan



clim, anom, upperquan = calc_anomaly(temperature)

# def plot_2D(cls):
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
clim, anom, upperquan = calc_anomaly(temperature)
ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
plt.contourf(np.squeeze(clim))
plt.contourf(np.mean(np.squeeze(anom),axis=0))
ax.set_global() ; ax.coastlines










