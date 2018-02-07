


class Variable:
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper

    """
    from datetime import datetime, timedelta
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


        start = Variable.datetime(self.startyear, self.startmonth, 1)
        end = Variable.datetime(self.endyear, self.endmonth, 1)
        datelist = [start.strftime('%Y-%m-%d')]
        while start <= end:
            if start.month < end.month:
                start += Variable.timedelta(days=31)
                datelist.append(Variable.datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
            else:
                start = Variable.datetime(start.year+1, self.startmonth, 1)
                datelist.append(Variable.datetime(start.year, start.month, 1).strftime('%Y-%m-%d'))
        self.datelist = datelist

        filename = '{}_{}-{}_{}_{}_{}'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, self.stream).replace(' ', '_')
        filename = filename.replace('/', 'x')
        self.filename = filename +'.nc'
        print("Variable function selected {} \n".format(self.filename))

