#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:47:28 2018

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import func_mcK
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

import matplotlib.pyplot as plt

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
map_proj = ccrs.PlateCarree(central_longitude=240)  

ex = dict(
     {'grid_res'     :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'sstartdate'   :       '1982-06-24',
     'senddate'     :       '1982-08-22',
     'fig_path'     :       "/Users/semvijverberg/surfdrive/McKinRepl/T95_ERA-I"}
     )

if os.path.isdir(ex['fig_path']) == False: os.makedirs(ex['fig_path'])
# Load in mckinnon Time series
T95name = 'PEP-T95TimeSeries.txt'
mcKtsfull, datesmcK = func_mcK.read_T95(T95name, ex)
datesmcK = func_mcK.make_datestr(datesmcK, ex)
# full Time series of T95 (Juni, Juli, August summer)
mcKts = mcKtsfull.sel(time=datesmcK)
# binary time serie when T95 exceeds 1 std
hotdaythreshold = mcKts.mean(dim='time').values + mcKts.std().values
hotts = mcKts.where( mcKts.values > hotdaythreshold) 
hotdates = hotts.dropna(how='all', dim='time').time
# plotting same figure as in paper
plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
                                freq=(datesmcK[1] - datesmcK[0])))
plotpaperhotdays = plotpaper.where( plotpaper.values > hotdaythreshold) 
plotpaperhotdays = plotpaperhotdays.dropna(how='all', dim='time').time
plt.figure()
plotpaper.plot()
plt.axhline(y=hotdaythreshold)
for days in plotpaperhotdays.time.values:
    plt.axvline(x=days)



# Load in external ncdf
sst_ERAname = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
t2m_ERAname = 't2mmax_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
# full globe - full time series
varfullgl = func_mcK.import_array(sst_ERAname, ex)
# Converting Mckinnon timestemp to match xarray timestemp
matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)
# full globe - only (mcKinnon) summer days (Juni, Juli, August)
varsumgl = varfullgl.sel(time=matchdaysmcK)
# region mckinnon - Mckinnon summer days (Juni, Juli, August)
varsumreg = func_mcK.find_region(varsumgl, region='Mckinnonplot')[0]
# region mckinnon - full time series
varfullreg = func_mcK.find_region(varfullgl, region='Mckinnonplot')[0]
## region mckinnon - only (mcKinnon) summer 
func_mcK.xarray_plot(varsumreg.mean(dim='time')) 

for i in range(3):
    func_mcK.xarray_plot(varsumreg.isel(time=i)) 


std = varsumgl.std(dim='time')
# not merging hot days which happen consequtively

matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfullgl.time[0].dt.hour)
# Select concurrent hot days
varhotdays = varsumreg.sel(time=matchhotdates)

matchpaperplothotdates = func_mcK.to_datesmcK(plotpaperhotdays.time, hotdates[0].dt.hour, varhotdays.time[0].dt.hour)
plotting = varhotdays.sel(time=matchpaperplothotdates[:5])
for day in matchpaperplothotdates[:5]:
    func_mcK.xarray_plot(varfullreg.sel(time=day)) 

#%% Mean over 230 hot days
lags = [0, 50]

array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    
    varhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = varhotdays
    
xrdata.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.4, 'vmax' : 0.4, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(xrdata, file_name, kwrgs) 
#%% EOFS on composite hot days
lag = 0
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
varhotdays = varsumreg.sel(time=dates_min_lag)
eof_output, solver = func_mcK.EOF(varhotdays, neofs=6)
title = 'EOFs on composite (hot days, n={})'.format(matchhotdates.time.size) 
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             title + ' - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)
explained_var = solver.eigenvalues()[0]/ np.sum(solver.eigenvalues()).values
print('first eof explains {}% of variance'.format(float(explained_var)))

#%% EOFS on composite all summer days

eof_output, solver = func_mcK.EOF(varsumreg, neofs=6)
title = 'EOFs on full timeseries '
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             title + ' - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)
explained_var = solver.eigenvalues()[0]/ np.sum(solver.eigenvalues()).values
print('first eof explains {}% of variance'.format(float(explained_var)))

#%% Project composite on main EOFs:
matrix = np.reshape(varhotdays.values, (varhotdays.time.size, varhotdays[0].size))
eofs = solver.eofs().values
n_eofs  = eofs[:,0,0].size
n_space = eofs[0].size
n_time  = varhotdays.time.size

Ceof = np.reshape( eofs, (n_eofs, n_space) ) 
C = Ceof


n_eofs_used = 100
PCs_all = solver.pcs(pcscaling=0, npcs=n_eofs_used)
PCstd_all = [float(np.std(PCs_all[:,i])) for i in range(n_eofs_used)]

PCi = np.zeros( (n_time, n_eofs_used) )
PCstd_hot = np.zeros( (n_eofs_used) )
PCi_unitvar_hot = np.zeros( (n_time, n_eofs_used) )
PCi_mean_hot = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):

    PCi[:,i] = np.dot( matrix, C[i,:] )
    PCi_unitvar_hot[:,i] = (PCi[:,i]) / PCstd_all[i]
    
    PCstd_hot[i] = np.std(PCi_unitvar_hot[:,i])
    PCi_mean_hot[i] = np.mean(PCi_unitvar_hot[:,i])
    
plt.title('Pcs of normal projection of matrix upon eofs')
plt.plot(PCi[:,0])
plt.plot(PCi[:,-1])
plt.figure()
plt.title('PCs which are normalize by std over whole timeseries')
plt.plot(PCi_unitvar_hot[:,0])
plt.plot(PCi_unitvar_hot[:,-1])
plt.figure()
plt.plot(solver.eigenvalues()[:n_eofs])
print('Mean value PC time series hot: {}'.format(PCi_mean_hot[0]))
print('std value PC time series hot: {}'.format(PCstd_hot[0]))
#%% Project 'not extreme patterns' on main EOFs
varnothot = varsumreg.drop(labels=dates_min_lag.values, dim='time')
matrix = np.reshape(varnothot.values, (varnothot.time.size, varnothot[0].size))
eofs = solver.eofs().values
n_eofs  = eofs[:,0,0].size
n_space = eofs[0].size
n_time  = varnothot.time.size

Ceof = np.reshape( eofs, (n_eofs, n_space) ) 
C = Ceof

PCi = np.zeros( (n_time, n_eofs_used) )
PCstd_nor = np.zeros( (n_eofs_used) )
PCi_unitvar_nor = np.zeros( (n_time, n_eofs_used) )
PCi_mean_nor = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):

    PCi[:,i] = np.dot( matrix, C[i,:] )
    PCi_unitvar_nor[:,i] = (PCi[:,i]) / PCstd_all[i]
    
    PCstd_nor[i] = np.std(PCi_unitvar_nor[:,i])
    PCi_mean_nor[i] = np.mean(PCi_unitvar_nor[:,i])
    
plt.title('Pcs of normal projection of matrix upon eofs')
plt.plot(PCi[:,0])
plt.plot(PCi[:,-1])
plt.figure()
plt.title('PCs which are normalize by std over whole timeseries')
plt.plot(PCi_unitvar_nor[:,0])
plt.plot(PCi_unitvar_nor[:,-1])
print('Mean value PC time series normal: {}'.format(PCi_mean_nor[0]))
print('std value PC time series normal: {}'.format(PCstd_nor[0]))
#%% calculate ratio:
ratio_std = np.zeros( (n_eofs_used) )
ratio_mean = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):
    ratio_std[i] = PCstd_nor[i] / PCstd_hot[i]
    ratio_mean[i] = abs(PCi_mean_hot[i]) / abs(PCi_mean_nor[i])
plt.figure()
plt.plot(ratio_std)
plt.figure()
plt.ylim( (np.min(ratio_mean), np.max(ratio_mean)  ) )
plt.plot(ratio_mean)
#%%

lags = [0, 5, 10, 15, 20, 30, 40, 50]
#lags = [0, 5]

array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    
    varhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = varhotdays
    

xrdata.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.4, 'vmax' : 0.4, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(xrdata, file_name, kwrgs) 

#%% Composite hot days absolute value divided by std of hot days (Signal to Noise)
xrdatastd = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    std_lag =  varfullreg.sel(time=dates_min_lag).std(dim='time')
    varhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = abs(varhotdays/std_lag)

title = 'S/N within composite \nabsolute values / std within composite \nT95 McKinnon data - ERA-I SST'
xrdatastd.attrs['units'] = 'std within composite'
kwrgs = dict( {'vmin' : 0, 'vmax' : 1.2, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.Reds, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'std within composite lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdatastd, file_name, kwrgs)

#%% Composite hot days in standard deviation
lags = [0, 5, 10, 15, 20, 30, 40, 50]
array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')


for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    varhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')

    xrdata[idx] = varhotdays/std

xrdata.attrs['units'] = 'Kelvin (normalized by std)'
file_name = '~/Downloads/finalfiguremckinnon'
title = 'normalized by total std \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'normalized by std values lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdata, file_name, kwrgs) 
#%%

lags = [0, 5, 10, 15, 20, 30, 40, 50]
array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
         
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    varhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')

    xrdata[idx] = varhotdays/std


xrdata.attrs['units'] = 'Kelvin (normalized by std)'
file_name = '~/Downloads/finalfiguremckinnon'
title = 'normalized by total std \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'normalized by std values lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdata, file_name, kwrgs) 
#%%
ex['percentile'] = 80
for lag in lags:
    nparray = np.reshape(xrdata.values, (xrdata.shape))
    nparray = nparray[np.isnan(nparray) == False]
    percentile = np.percentile(np.abs(nparray), ex['percentile'])
    aboveperc = xrdata.where( (xrdata.sel(lag=lag) > percentile) | (xrdata.sel(lag=lag) < -percentile))
    

title = 'exceeding {}th percentile of mean composite \nT95 McKinnon data - ERA-I SST'.format(ex['percentile'])
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'Retrieve dominant pattern mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(aboveperc, file_name, kwrgs) 

#%%
dates = matchhotdates[np.arange(0, matchhotdates.size, int(matchhotdates.size/10))]
lag = 50
array = np.zeros( (len(dates),varsumreg.latitude.size, varsumreg.longitude.size) )
randhot = xr.DataArray(data=array, coords=[dates, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='Scanning SST hotdates')
for date in dates:
    idx = list(dates).index(date)
    singledate_min_lag = date - pd.Timedelta(int(lag), unit='d')
    singledate = varfullreg.sel(time=singledate_min_lag)/std
#    nparray = np.reshape(singledate.values, (singledate.shape))
#    nparray = nparray[np.isnan(nparray) == False]
#    percentile = np.percentile(np.abs(nparray), ex['percentile'])
    randhot[idx] = singledate.where( (singledate > percentile) | (singledate < -percentile))
    randhot[idx] = singledate
    
randhot.attrs['units'] = 'upper 20 percentile anomalies'
title = 'exceeding {}th percentile of mean composite \nT95 McKinnon data - ERA-I SST'.format(ex['percentile'])
kwrgs = dict( {'vmin' : -3, 'vmax' : 3, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 3} )
file_name = os.path.join(ex['fig_path'], 
             'random maps SST prior to heat event with lag {}.png'.format(lag))
func_mcK.finalfigure(randhot, file_name, kwrgs) 

