#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:29:47 2018

@author: semvijverberg
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import generate_varimax
from eofs.xarray import Eof
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy 

def read_T95(T95name, ex):
    filepath = os.path.join(ex['path_pp'], T95name)
    data = pd.read_csv(filepath)
    datelist = []
    values = []
    for r in data.values:
        year = int(r[0][:4])
        month = int(r[0][5:7])
        day = int(r[0][7:11])
        string = '{}-{}-{}'.format(year, month, day)
        values.append(float(r[0][10:]))
        datelist.append( pd.Timestamp(string) )
    
    dates = pd.to_datetime(datelist)
    RVts = xr.DataArray(values, coords=[dates], dims=['time'])
    return RVts, dates

def Ev_timeseries(xarray, threshold):   
    Ev_ts = xarray.where( xarray.values > threshold) 
    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    return Ev_dates

def timeseries_tofit_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    seldays_pp = pd.DatetimeIndex(start=one_yr[0], end=one_yr[-1], 
                                freq=(datetime[1] - datetime[0]))
    end_day = one_yr.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min() ) / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) \
                + np.timedelta64(1, 'D') 
    
    def make_datestr_2(datetime, start_yr):
        breakyr = datetime.year.max()
        datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
        nyears = (datetime.year[-1] - datetime.year[0])+1
        startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
        endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
        firstyear = startday[:4]
        datesdt = start_yr
        def plusyearnoleap(curr_yr, startday, endday, incr):
            startday = startday.replace(firstyear, str(curr_yr+incr))
            endday = endday.replace(firstyear, str(curr_yr+incr))
            next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                            freq=(datetime[1] - datetime[0]))
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
            return next_yr
        
        for yr in range(0,nyears-1):
            curr_yr = yr+datetime.year[0]
            next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
            datesdt = np.append(datesdt, next_yr)
#            print(len(next_yr))
#            nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
#            datesstr = datesstr + nextstr
#            print(nextstr[0])
            
            upd_start_yr = plusyearnoleap(next_yr.year[0], startday, endday, 1)

            if next_yr.year[0] == breakyr:
                break
        datesdt = pd.to_datetime(datesdt)
        return datesdt, upd_start_yr
    
    start_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(datetime[1] - datetime[0]))
    # exluding leap year from cdo select string
    noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
    start_yr = start_yr[noleapdays].dropna(how='all')
    datesdt, next_yr = make_datestr_2(datetime, start_yr)
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
    enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])
    print('adjusted time series to fit bins: \nFrom {} to {}'.format(
                startdatestr, enddatestr))
    adj_array = xarray.sel(time=datesdt)
    return adj_array, datesdt
    

def time_mean_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    if one_yr.size % ex['tfreq'] != 0:
        possible = []
        for i in np.arange(1,20):
            if 214%i == 0:
                possible.append(i)
        print('Error: stepsize {} does not fit in one year\n '
                         ' supply an integer that fits {}'.format(
                             ex['tfreq'], one_yr.size))   
        print('\n Stepsize that do fit are {}'.format(possible))
        print('\n Will shorten the \'subyear\', so that it the temporal'
              'frequency fits in one year')
        xarray, datetime = timeseries_tofit_bins(xarray, ex)
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
          
    else:
        pass
    fit_steps_yr = (one_yr.size)  / ex['tfreq']
    bins = list(np.repeat(np.arange(0, fit_steps_yr), ex['tfreq']))
    n_years = (datetime.year[-1] - datetime.year[0]) + 1
    for y in np.arange(1, n_years):
        x = np.repeat(np.arange(0, fit_steps_yr), ex['tfreq'])
        x = x + fit_steps_yr * y
        [bins.append(i) for i in x]
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])
    
    half_step = ex['tfreq']/2.
    newidx = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
    newdate = label_dates[newidx]
    

    group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
    group_bins['bins'] = newdate.values
    dates = pd.to_datetime(newdate.values)
    return group_bins.rename({'bins' : 'time'}), dates

def expand_times_for_lags(datetime, ex):
    expanded_time = []
    for yr in set(datetime.year):
        one_yr = datetime.where(datetime.year == yr).dropna(how='any')
        start_mcK = one_yr[0]
        #start day shifted half a time step
        half_step = ex['tfreq']/2.
#        origshift = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
        start_mcK = start_mcK - np.timedelta64(int(half_step+0.49), 'D')
        last_day = '{}{}'.format(yr, ex['senddate'][4:])
        end_mcK   = pd.to_datetime(last_day)
#        adj_year = pd.DatetimeIndex(start=start_mcK, end=end_mcK, 
#                                    freq=(datetime[1] - datetime[0]), 
#                                    closed = None).values
        steps = len(one_yr)
        shift_start = start_mcK - (steps) * np.timedelta64(ex['tfreq'], 'D')
        adj_year = pd.DatetimeIndex(start=shift_start, end=end_mcK, 
                                    freq=pd.Timedelta( '1 days'), 
                                    closed = None).values
        [expanded_time.append(date) for date in adj_year]
    
    return pd.to_datetime(expanded_time)

def make_datestr(dates, ex):
    start_yr = pd.DatetimeIndex(start=ex['sstartdate'], end=ex['senddate'], 
                                freq=(dates[1] - dates[0]))
    breakyr = dates.year.max()
    datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
    nyears = (dates.year[-1] - dates.year[0])+1
    startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
    endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
    firstyear = startday[:4]
    def plusyearnoleap(curr_yr, startday, endday, incr):
        startday = startday.replace(firstyear, str(curr_yr+incr))
        endday = endday.replace(firstyear, str(curr_yr+incr))
        next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                        freq=(dates[1] - dates[0]))
        # excluding leap year again
        noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
        next_yr = next_yr[noleapdays].dropna(how='all')
        return next_yr
    

    for yr in range(0,nyears-1):
        curr_yr = yr+dates.year[0]
        next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr

        if next_yr.year[0] == breakyr:
            break
    datesmcK = pd.to_datetime(datesstr)
    return datesmcK

def import_array(filename, ex):
    file_path = os.path.join(ex['path_pp'], filename)        
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: var})))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    marray['time'] = dates
    return marray

def save_figure(data, path):
    import os
    import matplotlib.pyplot as plt
#    if 'path' in locals():
#        pass
#    else:
#        path = '/Users/semvijverberg/Downloads'
    if path == 'default':
        path = '/Users/semvijverberg/Downloads'
    else:
        path = path
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if type(data.name) is not type(None):
        name = data.name.replace(' ', '_')
    if 'name' in locals():
        print('input name is: {}'.format(name))
        name = name + '.jpeg'
        pass
    else:
        name = 'fig_' + today + '.jpeg'
    print(('{} to path {}'.format(name, path)))
    plt.savefig(os.path.join(path,name), format='jpeg', dpi=300, bbox_inches='tight')
    
def area_weighted(xarray):
    # Area weighted     
    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(np.sqrt(coslat)[..., np.newaxis],(1,xarray.longitude.size))
    xarray.values = xarray.values * area_weights 
    return xarray

def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
#    original
    plt.figure()
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        if data.longitude.where(data.longitude==0).dropna(dim='longitude', how='all') == 0.:
            print('hoi')   
            data = convert_longitude(data)
    else:
        pass
    if data.ndim != 2:
        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim))
        data = data[0]
    else:
        pass
    if 'mask' in list(data.coords.keys()):
        cen_lon = data.where(data.mask==True, drop=True).longitude.mean()
        data = data.where(data.mask==True, drop=True)
    else:
        cen_lon = data.longitude.mean().values
    proj = ccrs.PlateCarree(central_longitude=cen_lon)
    ax = plt.axes(projection=proj)
    ax.coastlines()
    vmin = np.round(float(data.min())-0.01,decimals=2) 
    vmax = np.round(float(data.max())+0.01,decimals=2) 
    vmin = -max(abs(vmin),vmax) ; vmax = max(abs(vmin),vmax)
    # ax.set_global()
    if 'mask' in list(data.coords.keys()):
        plot = data.copy().where(data.mask==True).plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax)
    else:
        plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax)
    if saving == True:
        save_figure(data, path=path)
    plt.show()
    
def convert_longitude(data):
    import numpy as np
    import xarray as xr
    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
    # roll all values to the right for len(lon_above amount of steps)
    data = data.roll(longitude=len(lon_above))
    # adapt longitude values above 180 to negative values
    substract = lambda x, y: (x - y)
    lon_above = xr.apply_ufunc(substract, lon_above, 360)
    if lon_normal[0] == 0.:
        convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
    else:
        convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
    data['longitude'] = convert_lon
    return data

def to_datesmcK(datesmcK, to_hour, from_hour):
    dt_hours = to_hour + from_hour
    matchdaysmcK = datesmcK + pd.Timedelta(int(dt_hours), unit='h')
    return xr.DataArray(matchdaysmcK, dims=['time'])

def find_region(data, region='Mckinnonplot'):
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'PEPrectangle':
        west_lon = -215; east_lon = -125; south_lat = 19; north_lat = 50
    elif region ==  'Pacific':
        west_lon = -215; east_lon = -120; south_lat = 19; north_lat = 60
    elif region ==  'Whole':
        west_lon = -360; east_lon = -1; south_lat = -80; north_lat = 80
    elif region ==  'Southern':
        west_lon = -20; east_lon = -2; south_lat = -80; north_lat = -60

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
#    if data.longitude.values[-1] > 180:
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        all_values = data.sel(latitude=slice(north_lat, south_lat), 
                              longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords


def finalfigure(xrdata, file_name, kwrgs):
    #%%
#    clevels = '' ; vmin=-0.4 ; vmax=0.4
    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': kwrgs['map_proj']},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()


    if kwrgs['clevels'] == 'default':
        vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin=kwrgs['vmin']
        vmax=kwrgs['vmax']
        clevels = np.linspace(vmin,vmax,kwrgs['steps'])
    cmap = kwrgs['cmap']
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
#        print(n_ax)
        plotdata = xrdata[n_ax]
        im = plotdata.plot.contourf(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': kwrgs['map_proj']},
                               levels=clevels, add_colorbar=False)
        ax.coastlines(color='grey', alpha=0.3)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
#        lons = [-5.8, -5.8, -5.5, -5.5]
#        lats = [50.27, 50.48, 50.48, 50.27]
        lons_sq = [-215, -215, -125, -125]
        lats_sq = [50, 19, 19, 50]
        ring = LinearRing(list(zip(lons_sq , lats_sq )))
        ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='green')
        if kwrgs['map_proj'].proj4_params['proj'] in ['merc', 'Plat']:
            print(True)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.xlabels_top = False;
    #        gl.xformatter = LONGITUDE_FORMATTER
            gl.ylabels_right = False;
            gl.xlabels_bottom = False
    #        gl.yformatter = LATITUDE_FORMATTER
        else:
            pass
        
    g.fig.text(0.5, 0.95, kwrgs['title'], fontsize=15, horizontalalignment='center')
    cbar_ax = g.fig.add_axes([0.25, (figheight/25)/n_plots, 
                                  0.5, (figheight/25)/n_plots])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                 label=xrdata.attrs['units'], extend='neither')
    g.fig.savefig(file_name ,dpi=250)
    #%%
    return
        
def EOF(data, scaling, neofs=10, center=False, weights=None):
    import numpy as np
    from eofs.xarray import Eof
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data.coords['latitude'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(np.squeeze(data), center=False, weights=wgts)
    loadings = solver.eofs(neofs=neofs, eofscaling=scaling)
    loadings.attrs['units'] = 'mode'
    
    return loadings.rename({'mode':'loads'})

def project_eof_field(xarray, loadings, n_eofs_used):
    
    n_eofs  = loadings[:,0,0].size
    n_space = loadings[0].size
    n_time  = xarray.time.size
    
    matrix = np.reshape(xarray.values, (n_time, xarray[0].size))
    matrix = np.nan_to_num(matrix)

    # convert eof output to matrix
    Ceof = np.reshape( loadings.values, (n_eofs, n_space) ) 
    C = np.nan_to_num(Ceof)

#    # Calculate std over entire eof time series
#    PCs_all = solver.pcs(pcscaling=scaling, npcs=n_eofs_used)
#    PCstd_all = [float(np.std(PCs_all[:,i])) for i in range(n_eofs_used)]
    
    PCi = np.zeros( (n_time, n_eofs_used) )
    PCstd = np.zeros( (n_eofs_used) )
#    PCi_unitvar = np.zeros( (n_time, n_eofs_used) )
    PCi_mean = np.zeros( (n_eofs_used) )
    for i in range(n_eofs_used):
    
        PCi[:,i] = np.dot( matrix, C[i,:] )
        
        PCstd[i] = np.std(PCi[:,i])
        PCi_mean[i] = np.mean(PCi[:,i])
    
    return PCi, PCstd, PCi_mean



def extract_pattern(Composite, totaltimeserie, n_eofs_used, loadings, weights):
    # Get oefs from Composite
    
    
    Composite = Composite * weights
    totaltimeserie = totaltimeserie * weights

    
    # Project hot day fields upon EOFS 
    PCi_comp, PCstd_comp, PCi_mean_comp = project_eof_field(
                                    Composite, loadings, n_eofs_used)

    # Project total timeseries on main EOFs
    PCi_all, PCstd_all, PCi_mean_nor = project_eof_field(
                                    totaltimeserie, loadings, n_eofs_used)

    # Determine deviating PCs of compososite with respect to totaltimeserie 
    # using absolute value of normalized PCs
    #### Depricated #####
    #    ratio_std = PCstd_nor / PCstd_hot
    #    ratio_mean = (PCi_mean_nor / PCi_mean_hot) * np.median(PCi_mean_nor)**-1
    #    plt.figure()
    #    plt.title('Ratio variability in \'normal\' time series\ndivided by variability between hot days')
    #    plt.plot(ratio_std)
    #    plt.figure()
    #    plt.title('Ratio of mean PC values in \'normal\' time series\ndivided by hot days')
    #    plt.ylim( (np.min(ratio_mean)-1, np.max(ratio_mean)+1  ) )
    #    plt.plot(ratio_mean)
    #### End ######
    
    PCstd_all = [float(np.std(PCi_all[:,i])) for i in range(n_eofs_used)]
    # Normalize PC values w.r.t. variability in total time serie
    PCi_mean_comp = PCi_mean_comp / PCstd_all
#    plt.figure()
#    plt.ylim( (np.min(PCi_mean_comp)-1, np.max(PCi_mean_comp)+1  ) )
#    plt.plot(PCi_mean_comp)
#    plt.plot(PCi_mean_comp)
      
#    for i in range(2):
#        plt.figure()
#        plt.title('PC {}\nnormalized by std (std from whole summer PC '
#                  'timeseries)'.format(i))
#        plt.axhline(0, color='black')
#        plt.plot(PCi_comp[:,i]/PCstd_all[i])
#        plt.axhline(PCi_mean_comp[i])
        
#    plt.figure()
#    plt.plot(solver.eigenvalues()[:n_eofs_used])
#    print('Mean value PC time series: {}'.format(PCi_mean[0]))
#    print('std value PC time series: {}'.format(PCstd[0]))
        
    
    def Relevant_PCs(PCi_mean_comp, n_eofs_used):
        
        absolute_values = xr.DataArray(data=PCi_mean_comp, coords=[range(n_eofs_used)], 
                              dims=['loads'], name='xarray')
        anomalous = absolute_values.where(abs(absolute_values.values) > 
                          (absolute_values.mean(dim='loads') + absolute_values.std()).values)
        PC_imp_abs = anomalous.dropna(how='all', dim='loads')
    
        absolute_values = xr.DataArray(data=PCi_mean_comp, coords=[range(n_eofs_used)], 
                                  dims=['loads'], name='xarray')
        return PC_imp_abs, absolute_values
    
#    PC_imp_var, absolute_values = Relevant_PCs(ratio_std, n_eofs_used)
#    PC_imp_rel, absolute_values = Relevant_PCs(ratio_mean, n_eofs_used)
    PC_imp_abs, absolute_values = Relevant_PCs(PCi_mean_comp, n_eofs_used)
    
    plt.figure()
    plt.title('Pcs deviation (in std) from total variability')
    for n_eof in range(n_eofs_used):
        plt.axhline(0, color='black')
        plt.axhline(absolute_values.mean(dim='loads') + absolute_values.std().values, color='red')
        plt.axhline(-1*(absolute_values.mean(dim='loads') + absolute_values.std().values), color='blue')  
        plt.scatter(n_eof, absolute_values[n_eof])
    
    important_modes = list(PC_imp_abs.loads.values)
    array = np.zeros( (len(PC_imp_abs),Composite.latitude.size, Composite.longitude.size) )
    important_eofs = xr.DataArray(data=array, coords=[important_modes, Composite.latitude, Composite.longitude], 
                          dims=['loads','latitude','longitude'], name='loadings')
    
    
    for eof in important_modes:
        idx = important_modes.index(eof)
        single_eof = loadings.sel(loads=eof) * np.sign(absolute_values.sel(loads=eof))
        important_eofs[idx] = single_eof
        
    # calculate weighted mean of relevant PCs (weighted on deviation from total
    # variability)
    weights = abs(PC_imp_abs) / np.mean(abs(PC_imp_abs))
    important_eofs_w = xr.DataArray(data=array, coords=[important_modes, Composite.latitude, Composite.longitude], 
                          dims=['loads','latitude','longitude'], name='loadings')
    for eof in important_modes:
        idx = important_modes.index(eof)
        single_eof = loadings.sel(loads=eof) * np.sign(absolute_values.sel(loads=eof))
        important_eofs_w[idx] = single_eof * weights.sel(loads=eof)
    
    wmean_eofs = important_eofs_w.mean(dim='loads', keep_attrs = True) 
    
    
    return important_eofs, wmean_eofs, PC_imp_abs
    

def varimax_PCA_sem(xarray, max_comps):
    
    xarray = area_weighted(xarray)
    geo_object = xarray
    lats = geo_object.latitude.values
    lons = geo_object.longitude.values
    geo_data = xarray.values
    flattened = np.reshape(np.array(geo_data), 
                           (geo_data.shape[0], np.prod(geo_data.shape[1:])))
    nonmask_flat = np.where(np.array(flattened)[0] != 0.)[0]
    nonmasked = flattened[:,nonmask_flat]
    
    
    # convert to nonmasked array
    data = nonmasked
    truncate_by = 'max_comps'
    max_comps=max_comps
    fraction_explained_variance=0.9
    verbosity=0
    
    var_standard = generate_varimax.get_varimax_loadings_standard(data=data,
                        truncate_by = truncate_by, 
                        max_comps=max_comps,
                        fraction_explained_variance=fraction_explained_variance,
                        verbosity=verbosity,
                        )
    # Plug in nonmasked values into the original lonlat with mask
    npcopy = np.array(flattened[:max_comps])
    npcopy[:,nonmask_flat] = np.swapaxes(var_standard['weights'],1,0)
    nplonlat = np.reshape(npcopy, (npcopy.shape[0], lats.size, lons.size)) 
                                                   
    
    # convert to xarray
    xrarray_patterns = xr.DataArray(nplonlat, coords=[np.arange(0,max_comps), lats, 
                                 lons], 
                                dims=['loads', 'latitude','longitude'], 
                                name='rot_pca_standard')
#    PCs = xr.DataArray(np.swapaxes(var_standard['comps_ts'],1,0), 
#                       coords = [np.arange(0,max_comps)], 
#                       dims = ['modes'], 
#                       name = 'PCs')
    
    return xrarray_patterns

def Welchs_t_test(sample, full, alpha):
    np.warnings.filterwarnings('ignore')
    mask = (sample[0] == 0.).values
#    mask = np.reshape(mask, (mask.size))
    n_space = full.latitude.size*full.longitude.size
    npfull = np.reshape(full.values, (full.time.size, n_space))
    npsample = np.reshape(sample.values, (sample.time.size, n_space))
    
#    npsample = npsample[np.broadcast_to(mask==False, npsample.shape)] 
#    npsample = np.reshape(npsample, (sample.time.size, 
#                                     int(npsample.size/sample.time.size) ))
#    npfull   = npfull[np.broadcast_to(mask==False, npfull.shape)] 
#    npfull = np.reshape(npfull, (full.time.size, 
#                                     int(npfull.size/full.time.size) ))
       
    T, pval = scipy.stats.ttest_ind(npsample, npfull, axis=0, 
                                equal_var=False, nan_policy='omit')
    pval = np.reshape(pval, (full.latitude.size, full.longitude.size))
    T = np.reshape(T, (full.latitude.size, full.longitude.size))
    mask_sig = (pval > alpha) 
    mask_sig[mask] = True
    return T, pval, mask_sig

def merge_neighbors(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets

def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid):
    '''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
    '''
#    print('extracting features ...\n')

	
	# initialize arrays:
	# A final return array 
    A = np.ma.copy(Corr_Coeff)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

    indices_not_masked = np.where(A.mask==False)[0].tolist()

    lo = lon_grid.shape[0]
    la = lat_grid.shape[0]
	
	# create list of potential neighbors:
    N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
	#=====================
    for i in indices_not_masked:
        n = []	

        col_i= i%lo
        row_i = i//lo

		# knoten links oben
        if i==0:	
            n= n+[lo-1, i+1, lo ]

		# knoten rechts oben	
        elif i== lo-1:
            n= n+[i-1, 0, i+lo]

		# knoten links unten
        elif i==(la-1)*lo:
            n= n+ [i+lo-1, i+1, i-lo]

		# knoten rechts unten
        elif i == la*lo-1:
            n= n+ [i-1, i-lo+1, i-lo]

		# erste zeile
        elif i<lo:
            n= n+[i-1, i+1, i+lo]
	
		# letzte zeile:
        elif i>la*lo-1:
            n= n+[i-1, i+1, i-lo]
	
		# erste spalte
        elif col_i==0:
            n= n+[i+lo-1, i+1, i-lo, i+lo]
	
		# letzt spalte
        elif col_i ==lo-1:
            n= n+[i-1, i-lo+1, i-lo, i+lo]
	
		# nichts davon
        else:
            n = n+[i-1, i+1, i-lo, i+lo]
	
	#=====================
	# Criteria 2: must be all at least once be significanlty correlated 
	#=====================	
        m =[]
        for j in n:
            if j in indices_not_masked:
                m = m+[j]
		
		# now m contains the potential neighbors of gridpoint i

	
	#=====================	
	# Criteria 3: sign must be the same for each step 
	#=====================				
        l=[]
	
        cc_i = A.data[i]
        cc_i_sign = np.sign(cc_i)
		
	
        for k in m:
            cc_k = A.data[k]
            cc_k_sign = np.sign(cc_k)
		

            if cc_i_sign *cc_k_sign == 1:
                l = l +[k]

            else:
                l = l
			
            if len(l)==0:
                l =[]
                A.mask[i]=True	
			
            else: l = l +[i]	
		
		
            N_pot[i]=N_pot[i]+ l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
    Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
    B = np.abs(A)
	
	# 3) calculate the area size of each region	
	
    Area =  [[] for i in range(len(Regions))]
	
    for i in range(len(Regions)):
        indices = np.array(list(Regions[i]))
        indices_lat_position = indices//lo
        lat_nodes = lat_grid[indices_lat_position[:]]
        cos_nodes = np.cos(np.deg2rad(lat_nodes))		
		
        area_i = [np.sum(cos_nodes)]
        Area[i]= Area[i]+area_i
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
    # keep only regions which are larger then the mean size of the regions
    
    n_nodes = int(np.mean([len(r) for r in Regions]))
    
    R=[]
    Ar=[]
    for i in range(len(Regions)):
        if len(Regions[i])>=n_nodes:
            R.append(Regions[i])
            Ar.append(Area[i])
	
    Regions = R
    Area = Ar	
	
	
	
	# 4) calcualte region value:
	
    C = np.zeros(len(Regions))
	
    Area = np.array(Area)
    for i in range(len(Regions)):
        C[i]=Area[i]*np.mean(B[list(Regions[i])])


	
	
	# mask out those nodes which didnot fullfill the neighborhood criterias
    A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
    sorted_region_strength = np.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
	
    for i in range(len(Regions)):
        j = list(sorted_region_strength)[i]
        A[list(Regions[j])]=i+1
		
    return np.array(A, dtype=int)

def extract_commun(composite, actbox, event_binary, n_std, n_strongest):
        x=0
    #    T, pval, mask_sig = func_mcK.Welchs_t_test(sample, full, alpha=0.01)
    #    threshold = np.reshape( mask_sig, (mask_sig.size) )
    #    mask_threshold = threshold 
    #    plt.figure()
    #    plt.imshow(mask_sig)
        mean = composite.mean(dim='time')
        nparray = np.reshape(np.nan_to_num(mean.values), mean.size)
        
        threshold = n_std * np.std(nparray)
        mask_threshold = abs(nparray) < ( threshold )
        
        Corr_Coeff = np.ma.MaskedArray(nparray, mask=mask_threshold)
        lat_grid = mean.latitude.values
        lon_grid = mean.longitude.values
#        if Corr_Coeff.ndim == 1:
#            lag_steps = 1
#            n_rows = 1
#        else:
#            lag_steps = Corr_Coeff.shape[1]
#            n_rows = Corr_Coeff.shape[1]
        
        	
        n_lats = lat_grid.shape[0]
        n_lons = lon_grid.shape[0]
        lons_gph, lats_gph = np.meshgrid(lon_grid, lat_grid)
        
        cos_box = np.cos(np.deg2rad(lats_gph))
        cos_box_array = np.repeat(cos_box[None,:], actbox.shape[0], 0)
        cos_box_array = np.reshape(cos_box_array, (cos_box_array.shape[0], -1))
    
        
        Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid)
        
        if Regions_lag_i.max()> 0:
            n_regions_lag_i = int(Regions_lag_i.max())
			
            A_r = np.reshape(Regions_lag_i, (n_lats, n_lons))
            A_r + x
               
        x = A_r.max() 

        
        if n_regions_lag_i < n_strongest:
            n_strongest = n_regions_lag_i
        # this array will be the time series for each region
        ts_regions_lag_i = np.zeros((actbox.shape[0], n_strongest))
				
        for j in range(n_strongest):
            B = np.zeros(Regions_lag_i.shape)
            B[Regions_lag_i == j+1] = 1	
            ts_regions_lag_i[:,j] = np.mean(actbox[:, B == 1] * cos_box_array[:, B == 1], axis =1)
        
        
        # creating arrays of output
        npmap = np.ma.reshape(Regions_lag_i, (len(lat_grid), len(lon_grid)))
        mask_strongest = (npmap!=0.) & (npmap <= n_strongest)
        npmap[mask_strongest==False] = 0
        xrnpmap = mean.copy()
        xrnpmap.values = npmap
        
        mask = (('latitude', 'longitude'), mask_strongest)
        mean.coords['mask'] = mask
        xrnpmap.coords['mask'] = mask
        xrnpmap = xrnpmap.where(xrnpmap.mask==True)
        # normal mean of extracted regions
        norm_mean = mean.where(mean.mask==True)
        
        coeff_features = train_weights_LogReg(ts_regions_lag_i, event_binary)
        features = np.arange(xrnpmap.min(), xrnpmap.max() + 1 ) 
        weights = npmap.copy()
        for f in features:
            mask_single_feature = (npmap==f)
            weight = int(round(coeff_features[int(f-1)], 2) * 100)
            np.place(arr=weights, mask=mask_single_feature, vals=weight)
#            weights = weights/weights.max()
        
        weighted_mean = norm_mean * abs(weights)
    
        return weighted_mean, xrnpmap, ts_regions_lag_i

def extract_precursor(Prec_train, RV_train, ex, hotdaythreshold, lags, n_std, n_strongest):
    array = np.zeros( (len(lags),Prec_train.latitude.size, Prec_train.longitude.size) )
    commun_comp = xr.DataArray(data=array, coords=[lags, Prec_train.latitude, Prec_train.longitude], 
                          dims=['lag','latitude','longitude'], name='communities_composite', 
                          attrs={'units':'Kelvin'})
    array = np.zeros( (len(lags),Prec_train.latitude.size, Prec_train.longitude.size) )
    commun_num = xr.DataArray(data=array, coords=[lags, Prec_train.latitude, Prec_train.longitude], 
                          dims=['lag','latitude','longitude'], name='communities_numbered', 
                          attrs={'units':'regions'})

    
    
    Actors_ts_GPH = [[] for i in lags] #!
    
    x = 0
    for lag in lags:
#            i = lags.index(lag)
        idx = lags.index(lag)
        event_train = Ev_timeseries(RV_train, hotdaythreshold).time
        event_train = to_datesmcK(event_train, event_train.dt.hour[0], 
                                           Prec_train.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train = to_datesmcK(RV_train.time, RV_train.time.dt.hour[0], 
                                           Prec_train.time[0].dt.hour)

        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
    
        event_idx = [list(dates_train.values).index(E) for E in event_train.values]
        event_binary = np.zeros(dates_train.size)    
        event_binary[event_idx] = 1
        
        full = Prec_train.sel(time=dates_train_min_lag)
        composite = Prec_train.sel(time=events_min_lag)
        var = ex['name']
        actbox = np.reshape(full.values, (full.time.size, 
                                          full.latitude.size*full.longitude.size))
        

        
        commun_mean, commun_numbered, ts_regions_lag_i = extract_commun(
                        composite, actbox, event_binary, n_std, n_strongest)  
        
        commun_comp[idx] = commun_mean
        commun_num[idx]  = commun_numbered
    
    return commun_comp, commun_num

def train_weights_LogReg(ts_regions_lag_i, binary_events):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X = np.swapaxes(ts_regions_lag_i, 1,0)
    X = ts_regions_lag_i
    y = binary_events
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.33)
    
#    Log_out = LogisticRegression(random_state=0, penalty = 'l2', solver='saga',
#                       tol = 1E-9, multi_class='ovr').fit(X_train, y_train)
#    print(Log_out.score(X_train, y_train))
#    print(Log_out.score(X_test, y_test))
    
    from sklearn.linear_model import LogisticRegressionCV
    Log_out = LogisticRegressionCV(random_state=0, penalty = 'l2', solver='saga',
                       tol = 1E-9, multi_class='ovr', max_iter=4000).fit(
                               X_train, y_train)
#    print(Log_out.score(X_train, y_train))
#    print(Log_out.score(X_test, y_test))
    
    
    coeff_features = Log_out.coef_
    # predictions score of test 
    # score untrained:
#    score_on_trained = Log_out.score(X_train, y_train)
#    score_untrained = Log_out.score(X_test/Log_out.coef_, y_test)
#    score = Log_out.score(X_test, y_test)
#    print('\nPredictions score using \n'
#          '\ttestdata fit {}\n'.format(score),
#          '\ttestdata normal {}\n'.format(score_untrained),
#          '\ttraindata fit {}\n'.format(score_on_trained))
  
    return np.squeeze(coeff_features)

def cross_correlation_patterns(full_timeserie, pattern):
#    full_timeserie = precursor
    
    n_time = full_timeserie.time.size
    n_space = pattern.size
    
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))
    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
    covself   = np.zeros( (n_time) )
    corrself  = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))
#        sqrt( Var(X) ) = sigma_x = std(X)
#        spatcov[t] = np.cov(M)[0,1] / (np.std(full_ts[t]) * np.std(pattern))        
        covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
        corrself[t] = covself[t] / (np.std(full_ts[t]) * np.std(pattern))
    dates_test = full_timeserie.time
    covself = xr.DataArray(covself, coords=[dates_test.values], dims=['time'])
    return covself

def plot_events_validation(pred, obs, pthreshold, othreshold, test_year):
    #%%
#    pred = crosscorr_Sem
#    obs = RV_ts_test
#    pthreshold = Prec_threshold
#    othreshold = hotdaythreshold
#    test_year = 1983
    
    
    predyear = pred.where(pred.time.dt.year == test_year).dropna(dim='time', how='any')
    predyear['time'] = obs.time
    obsyear  = obs.where(obs.time.dt.year == test_year).dropna(dim='time', how='any')
    
    eventdays = obsyear.where( obsyear.values > othreshold) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    
    preddays = predyear.where(predyear.values > pthreshold)
    preddays = preddays.dropna(how='all', dim='time').time
      
    TP = [day for day in preddays.time.values if day in list(eventdays.values)]
    
    predyearscaled = (predyear - pred.mean()) * obsyear.std()/predyear.std() 
    plt.figure()
    plt.plot(pd.to_datetime(obsyear.time.values),obsyear)
    plt.plot(pd.to_datetime(obsyear.time.values),predyearscaled)

    plt.axhline(y=othreshold)
    for days in eventdays.time.values:
        plt.axvline(x=pd.to_datetime(days), color='blue', alpha=0.5)
    for days in preddays.time.values:
        plt.axvline(x=pd.to_datetime(days), color='orange', alpha=0.5)
    for days in pd.to_datetime(TP):
        plt.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
