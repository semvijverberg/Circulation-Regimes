#!/usr/bin/python
from ecmwfapi import ECMWFDataServer
import numpy as np

base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"

server = ECMWFDataServer()

server.retrieve({
    "dataset": "interim",
    "class": "ei",
    "date": "20170801",
    "expver": "1",
    "levelist": "285",
    "levtype": "pt",  # potential temperature (Isentrope)
    "param": "54.128/60.128/138.128",  # Potential vorticity; Pressure; Relative Vorticity
    "stream": "mnth",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    "format": "netcdf",
    "target": "{0}PV-pressure-RelVort.nc".format(base_path),
})
