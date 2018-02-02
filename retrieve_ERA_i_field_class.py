def retrieve_ERA_i_field(cls):
    import os
    # print 'you are retrieving the following dataset: \n'
    # print cls.__dict__
    file_path = os.path.join(cls.base_path, cls.filename)

    datestring = "/".join(cls.datelist)
    # !/usr/bin/python
    from ecmwfapi import ECMWFDataServer
    import os
    server = ECMWFDataServer()

    if cls.stream == "mnth" or cls.stream == "oper":
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
    elif cls.stream == "moda":
        time = "00:00:00"
    else:
        print "stream is not available"

    if os.path.isfile(path=file_path) == True:
        print "You have already download the variable {} from {} to {} on grid {} ".format(cls.name, cls.startyear, cls.endyear, cls.grid)
        print "\n to path: \n \n {}".format(file_path)
        pass
    else:
        print " You WILL download variable {} \n stream is set to {} \n all dates: {} \n".format \
            (cls.name, cls.stream, datestring)
        print "\n to path: \n \n {}".format(file_path)
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

    return