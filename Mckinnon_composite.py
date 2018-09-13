#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:47:28 2018

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes3')
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
# Load in external ncdf
sst_ERAname = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
varfull = func_mcK.import_array(sst_ERAname, ex)
matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfull.time[0].dt.hour)
mcKts = mcKtsfull.sel(time=datesmcK)
var = varfull.sel(time=matchdaysmcK)

hotts = mcKts.where( mcKts.std().values>mcKts.values) 
plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
                                freq=(datesmcK[1] - datesmcK[0])))
plotpaper.plot()
std = var.std(dim='time')
# not merging hot days which happen consequtively
hotdates = hotts.dropna(how='all', dim='time').time
matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfull.time[0].dt.hour)
#%%
map_proj = ccrs.PlateCarree(central_longitude=240)  
lags = [0, 5, 10, 15, 20, 30, 40, 50]
#lags = [0, 5]
varfull = func_mcK.find_region(varfull, region='Mckinnonplot')[0]
array = np.zeros( (len(lags),varfull.latitude.size, varfull.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    
    varhotdays = varfull.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = varhotdays
    

xrdata.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.4, 'vmax' : 0.4, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r} )
func_mcK.finalfigure(xrdata, file_name, kwrgs) 

#%%
xrdatastd = xr.DataArray(data=array, coords=[lags, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    std_lag =  varfull.sel(time=dates_min_lag).std(dim='time')
    varhotdays = varfull.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = abs(varhotdays/std_lag)

title = 'S/N within composite \nabsolute values / std within composite \nT95 McKinnon data - ERA-I SST'
xrdatastd.attrs['units'] = 'std within composite'
kwrgs = dict( {'vmin' : 0, 'vmax' : 1.2, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.Reds} )
file_name = os.path.join(ex['fig_path'], 
             'std within composite lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdatastd, file_name, kwrgs)

#%%
lags = [0, 5, 10, 15, 20, 30, 40, 50]
varfull = func_mcK.find_region(varfull, region='Mckinnonplot')[0]
array = np.zeros( (len(lags),varfull.latitude.size, varfull.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
         
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    varhotdays = varfull.sel(time=dates_min_lag).mean(dim='time')

    xrdata[idx] = varhotdays/std

xrdata.attrs['units'] = 'Kelvin (normalized by std)'
file_name = '~/Downloads/finalfiguremckinnon'
title = 'normalized by total std \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r} )
file_name = os.path.join(ex['fig_path'], 
             'normalized by std values lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdata, file_name, kwrgs) 
#%%

lags = [0, 5, 10, 15, 20, 30, 40, 50]
varfull = func_mcK.find_region(varfull, region='Mckinnonplot')[0]
array = np.zeros( (len(lags),varfull.latitude.size, varfull.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
         
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    varhotdays = varfull.sel(time=dates_min_lag).mean(dim='time')

    xrdata[idx] = varhotdays/std




xrdata.attrs['units'] = 'Kelvin (normalized by std)'
file_name = '~/Downloads/finalfiguremckinnon'
title = 'normalized by total std \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r} )
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
    

title = 'exceeding {} \nT95 McKinnon data - ERA-I SST'.format(ex['percentile'])
kwrgs = dict( {'vmin' : -0.7, 'vmax' : 0.7, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r} )
file_name = os.path.join(ex['fig_path'], 
             'Retrieve dominant pattern mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(aboveperc, file_name, kwrgs) 
#%%
dates = matchhotdates[np.arange(0, matchhotdates.size, int(matchhotdates.size/20))]
lag = 50
array = np.zeros( (len(dates),varfull.latitude.size, varfull.longitude.size) )
randhot = xr.DataArray(data=array, coords=[dates, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='Scanning SST hotdates')
for date in dates:
    idx = list(dates).index(date)
    singledate_min_lag = date - pd.Timedelta(int(lag), unit='d')
    singledate = varfull.sel(time=singledate_min_lag)/std
#    nparray = np.reshape(singledate.values, (singledate.shape))
#    nparray = nparray[np.isnan(nparray) == False]
#    percentile = np.percentile(np.abs(nparray), ex['percentile'])
    randhot[idx] = singledate.where( (singledate > percentile) | (singledate < -percentile))
    
randhot.attrs['units'] = 'upper 20 percentile anomalies'
title = 'exceeding {}th percentile \nT95 McKinnon data - ERA-I SST'.format(ex['percentile'])
kwrgs = dict( {'vmin' : -3, 'vmax' : 3, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r} )
file_name = os.path.join(ex['fig_path'], 
             'random maps SST prior to heat event with lag {}.png'.format(lag))
func_mcK.finalfigure(randhot, file_name, kwrgs) 


