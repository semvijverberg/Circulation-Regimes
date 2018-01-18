#!/usr/bin/python
from ecmwfapi import ECMWFDataServer
import numpy as np
from datetime import datetime, timedelta
import os
server = ECMWFDataServer()

base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
startyear = 1979 ; endyear = 1981
startmonth = 6 ; endmonth = 8
start = datetime(startyear, startmonth, 1)
end = datetime(endyear, endmonth, 1)
datelist = [start.strftime('%Y-%m-%d')]
while start <= end:
    if start.month < end.month:
        start += timedelta(days=31)
        print start.month
        datelist.append( datetime(start.year, start.month, 1).strftime('%Y-%m-%d') )
    else:
        start = datetime(start.year + 1, startmonth, 1)
        datelist.append(datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
datestring = "/".join(datelist)
grid = "2.5/2.5"
# monthly means of individual analysis steps, i.e. 00:00, 06:00, 12:00 etc,
# download synoptic monthly means by setting stream to "mnth"
# normal monthly mean, download monthly mean of daily means by setting stream to "moda"

server.retrieve({
    "dataset"   :   "interim",
    "class"     :   "ei",
    "date"      :   datestring,
    "expver"    :   "1",
    "grid"      :   grid,
    "levelist"  :   "285",
    "levtype"   :   "pt", # potential temperature (Isentrope)
    "param"     :   "54.128/60.128/138.128", # Potential vorticity; Pressure; Relative Vorticity
    "stream"    :   "mnth",
    "time"      :   "00:00:00/06:00:00/12:00:00/18:00:00",
    "type"      :   "an",
    "format"	:   "netcdf",
    "target"    :   os.path.join(base_path, "PV-pressure-RelVort_{}-{}.nc".format(startyear, endyear)),
})
