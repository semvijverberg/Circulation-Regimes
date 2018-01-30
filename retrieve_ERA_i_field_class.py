from datetime import datetime, timedelta
import xarray as xr
import os

class Variable:
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of timesteps   :   mnth
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

        filename = '{}_{}-{}_{}_{}_{}'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, self.grid).replace(' ', '_')
        filename = filename.replace('/', 'x')
        self.filename = filename
        print self.filename

def retrieve_ERA_i_field(cls):
    import os
    print 'you are retrieving the following dataset: \n'
    cls.__dict__
    filename = os.path.join(cls.base_path, cls.filename + '.nc')
    datestring = "/".join(cls.datelist)
    # !/usr/bin/python
    from ecmwfapi import ECMWFDataServer
    import numpy as np
    from datetime import datetime, timedelta
    import os
    server = ECMWFDataServer()

    if cls.stream == "mnth":
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
    elif cls.stream == "moda":
        time = "00:00:00"
    else:
        print "stream is not available"

    if os.path.isfile(path=filename) == True:
        pass
    else:
        server.retrieve({
            "dataset"   :   "interim",
            "class"     :   "ei",
            "date"      :   datestring,
            "grid"      :   cls.grid,
            "levtype"   :   cls.levtype,
            # if cls.levtype != 'sfc':
            "levelist"  :   cls.lvllist,
            "param"     :   cls.var_cf_code,
            "stream"    :   cls.stream,
            "time"      :   time,
            "type"      :   "an",
            "format"    :   "netcdf",
            "target"    :   filename,
        })
    return filename, " You have downloaded variable {} \n stream is set to {} \n all dates: {} \n".format \
        (cls.var_cf_code, cls.stream, datelist)

def calc_anomaly(cls, filename, decode_cf=True, decode_coords=True):
    # load in file
    ncdf = xr.open_dataset(filename, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(filename)
    marray = marray.rename({filename: cls.name})
    # what is temporal freq of dataset
    # calculate climatology

# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1980, startmonth=6, endmonth=8, grid='2,5/2,5', stream='moda')

retrieve_ERA_i_field(temperature)

