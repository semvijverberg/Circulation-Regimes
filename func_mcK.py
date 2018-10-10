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
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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

def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    plt.figure()
    data = np.squeeze(data)
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
        plot = data.where(data.mask==True).plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
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
    return matchdaysmcK

def find_region(data, region='Mckinnonplot'):
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50

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
        clevels = np.linspace(vmin,vmax,17)
    cmap = kwrgs['cmap']
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
        print(n_ax)
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
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False;
#        gl.xformatter = LONGITUDE_FORMATTER
        gl.ylabels_right = False;
        gl.xlabels_bottom = False
#        gl.yformatter = LATITUDE_FORMATTER
        
    g.fig.text(0.5, 0.9, kwrgs['title'], fontsize=15, horizontalalignment='center')
    cbar_ax = g.fig.add_axes([0.25, (figheight/25)/n_plots, 
                                  0.5, (figheight/25)/n_plots])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                 label=xrdata.attrs['units'], extend='neither')
    g.fig.savefig(file_name ,dpi=250)
    #%%
    return
        
def EOF(data, neofs=1, center=False, weights=None):
    import numpy as np
    from eofs.xarray import Eof
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data.coords['latitude'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(np.squeeze(data), center=False, weights=wgts)
    eof_output = solver.eofsAsCovariance(neofs=neofs)
    eof_output.attrs['units'] = 'mode'
    return eof_output, solver
  
#    
#    for row in xrdata.names_row.values:
#        rowidx = list(xrdata.names_row.values).index(row)
#        plotrow = xrdata.sel(names_row=row)
#        for col in xrdata.names_col.values:
#            colidx = list(xrdata.names_col.values).index(col)
#        
#            plotdata = plotrow.sel(names_col=names_col[colidx])
#            if np.sum(plotdata) == 0.:
#                g.axes[rowidx,colidx].text(0.5, 0.5, 'No regions significant',
#                              horizontalalignment='center', fontsize='x-large',
#                              verticalalignment='center', transform=g.axes[rowidx,colidx].transAxes)
#            elif np.sum(plotdata) > 0.:
#                im = plotdata.plot.contourf(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
#                                                cmap=cmap, levels=clevels,
#                                                subplot_kws={'projection':map_proj},
#                                                add_colorbar=False)
#                plotdata = plotrow.sel(names_col=names_col[1])
#                if np.sum(plotdata) != 0.:
#                    contourmask = np.array(np.nan_to_num(plotdata.where(plotdata > 0.)), dtype=int)
#                    plotdata.data = contourmask
#                    plotdata.plot.contour(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
#                                                        colors=['black'], levels=levels,
#                                                        subplot_kws={'projection':map_proj},
#                                                        add_colorbar=False)
#        
#        g.axes[rowidx,0].text(-figwidth/100, 0.5, row,
#                  horizontalalignment='center', fontsize='x-large',
#                  verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
#    for ax in g.axes.flat:
#        ax.coastlines(color='grey', alpha=0.3)
#        ax.set_title('')
#    g.axes[0,1].set_title(names_col[1] + '\nat alpha={} with '
#                  'pc_alpha(s)={}'.format(ex['alpha_level_tig']  , ex['pcA_sets'][ex['pcA_set']]), fontsize='x-large')
#    g.axes[0,0].set_title(names_col[0] + '\nat Corr p-value={}'.format(ex['alpha']),
#                  fontsize='x-large')
##        g.axes[rowidx,0].text(0.5, figwidth/100, 'Black contours are not significant after MCI',
##                      horizontalalignment='center', fontsize='x-large',
##                      verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
#    if ex['plotin1fig'] == False:
#        cbar_ax = g.fig.add_axes([0.25, (figheight/25)/len(g.row_names), 
#                                  0.5, (figheight/150)/len(g.row_names)])
#        plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
##        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#    plt.subplots_adjust(wspace=0.1, hspace=-0.3)
#    g.fig.savefig(os.path.join(ex['fig_path'], file_name + ex['file_type2']),dpi=250)
#    if ex['showplot'] == False:
#        plt.close()
#    return