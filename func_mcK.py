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

def time_mean_bins(xarray, tfreq_new):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
#    tfreq_orig = one_yr[1] - one_yr[0]
    tfreq_new  = 10
    if one_yr.size % tfreq_new != 0:
        print('stepsize {} does not fit in one year'.format(one_yr.size))
    else:
        pass
    fit_steps_yr = int((one_yr.size)  / tfreq_new)
    bins = list(np.repeat(np.arange(0, fit_steps_yr), tfreq_new))
    n_years = datetime.year[-1] - datetime.year[0]
    for y in np.arange(1, n_years+1):
        x = np.repeat(np.arange(0, fit_steps_yr), tfreq_new)
        x = x + fit_steps_yr * y
        [bins.append(i) for i in x]
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])
    
    half_step = tfreq_new/2.
    newidx = np.arange(half_step, datetime.size, tfreq_new, dtype=int)
    newdate = label_dates[newidx]
    

    group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
    group_bins['bins'] = newdate.values
    return group_bins.rename({'bins' : 'time'})

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
    return matchdaysmcK

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
        clevels = np.linspace(vmin,vmax,17)
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
#    from eofs.xarray import Eof
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data.coords['latitude'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(np.squeeze(data), center=False, weights=wgts)
    loadings = solver.eofs(neofs=neofs, eofscaling=scaling)
    loadings.attrs['units'] = 'mode'
    
    
    return loadings

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
    max_comps=60
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

