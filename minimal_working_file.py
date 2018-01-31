import os
from what_variable import Variable

# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
cls = temperature
file_path = os.path.join(cls.base_path, cls.filename)
datestring = "/".join(cls.datelist)
datestring = datestring.replace('-','')
if cls.stream == "mnth" or cls.stream == "oper":
    time = "00:00:00/06:00:00/12:00:00/18:00:00"
elif cls.stream == "moda":
    time = "00:00:00"
else:
    print "stream is not available"


# #!/usr/bin/env python
# from ecmwfapi import ECMWFDataServer
# server = ECMWFDataServer()
# server.retrieve({
#     "class": "ei",
#     "dataset": "interim",
#     "date": "19790101/19790201/19790301/19790401/19790501/19790601/19790701/19790801/19790901/19791001/19791101/19791201",
#     "expver": "1",
#     "grid": "2.5/2.5",
#     "levtype": "sfc",
#     "param": "167.128",
#     "step": "0",
#     "stream": "mnth",
#     "time": "00:00:00/06:00:00/12:00:00/18:00:00",
#     "type": "an",
#     "target": file_path,
# })



#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "dataset": "interim",
    "class": "ei",
    "expver": "1",
    "date": datestring,
    "grid": cls.grid,
    "levtype": cls.levtype,
    # "levelist"  :   cls.lvllist,
    "param": cls.var_cf_code,
    "stream": cls.stream,
    "step" : "0",
     "time"      :   time,
    "type": "an",
    "format": "netcdf",
    "target": file_path,
})