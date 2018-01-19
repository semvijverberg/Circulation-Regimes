import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_cf_code', type=str, default=54.128 / 60.128 / 138.128,
                        help="What variable do you want to retrieve, (in ECMWF cf parameter code)")
    parser.add_argument('--startyear', type=int, default=1979,
                        help='start year of time series')
    parser.add_argument('--endyear', type=int, default=1980,
                        help='end year of time series')
    parser.add_argument('--startmonth', type=int, default=6,
                        help='start month of time series')
    parser.add_argument('--endmonth', type=int, default=8,
                        help='end month of time series')
    parser.add_argument('--grid', type=str, default=2.5/2.5,
                        help='grid resolution in format: 2.5/2.5 ')
    args = parser.parse_args()
    sys.stdout.write(retrieve_ERA_i_field(args))


def retrieve_ERA_i_field(args):
    # !/usr/bin/python
    from ecmwfapi import ECMWFDataServer
    import numpy as np
    from datetime import datetime, timedelta
    import os
    server = ECMWFDataServer()

    base_path = "/Users/semvijverberg/surfdrive/Output_ERA/"
    start = datetime(args.startyear, args.startmonth, 1)
    end = datetime(args.endyear, args.endmonth, 1)
    datelist = [start.strftime('%Y-%m-%d')]
    while start <= end:
        if start.month < end.month:
            start += timedelta(days=31)
            print start.month
            datelist.append(datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
        else:
            start = datetime(start.year + 1, args.startmonth, 1)
            datelist.append(datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
    datestring = "/".join(datelist)
    # monthly means of individual analysis steps, i.e. 00:00, 06:00, 12:00 etc,
    # download synoptic monthly means by setting stream to "mnth"
    # normal monthly mean, download monthly mean of daily means by setting stream to "moda"
       if stream == "mnth":
           time = "00:00:00/06:00:00/12:00:00/18:00:00"
       else:
           time = "00:00:00"

    server.retrieve({
        "dataset": "interim",
        "class": "ei",
        "date": datestring,
        "expver": "1",
        "grid": args.grid,
        "levelist": "285",
        "levtype": "pt",  # potential temperature (Isentrope)
        "param": args.var_cf_code,  # Potential vorticity; Pressure; Relative Vorticity
        "stream": "mnth",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "format": "netcdf",
        "target": os.path.join(base_path, "PV-pressure-RelVort_{}-{}.nc".format(args.startyear, args.endyear)),
    })
    return "You have downloaded"


if __name__ == '__main__':
    main()
