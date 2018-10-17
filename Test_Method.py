#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:57:07 2018

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import func_mcK
import clustering_temporal
from ROC_score import ROC_score
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
     'map_proj'     :       map_proj,
     'fig_path'     :       "/Users/semvijverberg/surfdrive/McKinRepl/T95_ERA-I"}
     )

if os.path.isdir(ex['fig_path']) == False: os.makedirs(ex['fig_path'])

#'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', Whole



       


region = 'PEPrectangle'

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
## plotting same figure as in paper
#plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
#                                freq=(datesmcK[1] - datesmcK[0])))
#plotpaperhotdays = plotpaper.where( plotpaper.values > hotdaythreshold) 
#plotpaperhotdays = plotpaperhotdays.dropna(how='all', dim='time').time
#plt.figure()
#plotpaper.plot()
#plt.axhline(y=hotdaythreshold)
#for days in plotpaperhotdays.time.values:
#    plt.axvline(x=days)



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
varsumreg = func_mcK.find_region(varsumgl, region=region)[0]
# region mckinnon - full time series
varfullreg = func_mcK.find_region(varfullgl, region=region)[0]
## region mckinnon - only (mcKinnon) summer 
func_mcK.xarray_plot(varsumreg.mean(dim='time')) 

#for i in range(3):
#    func_mcK.xarray_plot(varsumreg.isel(time=i)) 


std = varsumgl.std(dim='time')
# not merging hot days which happen consequtively

matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfullgl.time[0].dt.hour)
# Select concurrent hot days
varhotdays = varsumreg.sel(time=matchhotdates)

#matchpaperplothotdates = func_mcK.to_datesmcK(plotpaperhotdays.time, hotdates[0].dt.hour, varhotdays.time[0].dt.hour)
#plotting = varhotdays.sel(time=matchpaperplothotdates[:5])
#for day in matchpaperplothotdates[:5]:
#    func_mcK.xarray_plot(varfullreg.sel(time=day)) 

#%% Mean over 230 hot days
#lags = [20, 30]
lags = [0, 5, 10, 25, 40, 50]

array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
mcK_mean = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    hotdates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    
    summerdays_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    std_lag =  varfullreg.sel(time=summerdays_min_lag).std(dim='time')
    
    varhotdays = varfullreg.sel(time=hotdates_min_lag).mean(dim='time')
    
    signaltonoise = abs(varhotdays/std_lag)
    weights = signaltonoise / np.mean(signaltonoise)
    
    mcK_mean[idx] = varhotdays 
    
mcK_mean.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -3*mcK_mean.std().values, 'vmax' : 3*mcK_mean.std().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(mcK_mean, file_name, kwrgs) 

#%%

array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
mcK_mean_w = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    hotdates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    
    summerdays_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    std_lag =  varfullreg.sel(time=summerdays_min_lag).std(dim='time')
    
    varhotdays = varfullreg.sel(time=hotdates_min_lag).mean(dim='time')
    
    signaltonoise = abs(varhotdays/std_lag)
    weights = signaltonoise / np.mean(signaltonoise)
    
    mcK_mean_w[idx] = varhotdays * weights
    
mcK_mean_w.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'weighted mean composite lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'weighted mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -3*mcK_mean_w.std().values, 'vmax' : 3*mcK_mean_w.std().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(mcK_mean_w, file_name, kwrgs) 



#%% Weighted eofs
#lags = [0, 5, 10, 25, 40, 50]
n_eofs_used = 30
scaling = 0


array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
w_eofs = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='Sem_Composite_diff_lags')
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    # apply weights:
    w_varfullreg = varfullreg
    
    Composite = w_varfullreg.sel(time=dates_min_lag)
    summerdays_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    totaltimeserie = w_varfullreg.sel(time=summerdays_min_lag)
    
    
    important_eofs, wmean_eofs, PC_imp_abs = func_mcK.extract_pattern(Composite, 
                                          totaltimeserie, scaling, n_eofs_used, weights)
                                              
    w_eofs[idx] = wmean_eofs
    
w_eofs.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'Sem weighted anomalous eofs lag{}-{}.png'.format(lags[0], lags[-1]))
title = 'Sem weighted anomalous eofs \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -3*w_eofs.std().values, 'vmax' : 3*w_eofs.std().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(w_eofs, file_name, kwrgs) 

#%%

## =============================================================================
## clustering Composite 
## =============================================================================
#
#n_clusters = 10
#tfreq = 'daily'
#methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
#linkage = ['complete', 'average']
#ex['distmetric'] = 'jaccard'
#ex['clusmethod'] = methods[0] ; ex['linkage'] = linkage
#
#output, group_clusters = clustering_temporal(Composite, ex, n_clusters, tfreq, region)



#%%
def cross_correlation_patterns(full_timeserie, pattern):
    pattern = mcK_mean.sel(lag=lag)
    full_timeserie = precursor
    
    n_time = full_timeserie.time.size
    n_space = pattern.size
    
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))
    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
    for t in range(n_time):
        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,0]
    crosscorr = crosscorr #/ np.std(crosscorr)
    plt.plot(crosscorr)
    plt.figure()
    plt.plot(spatcov)
    return spatcov


#%%
#lags = [0, 5, 10, 25, 40, 50]

for lag in lags:
    idx = lags.index(lag)

    # select antecedant SST pattern to summer days:
    dates_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    precursor = varfullreg.sel(time=dates_min_lag)
    crosscorr_mcK = cross_correlation_patterns(precursor, mcK_mean.sel(lag=lag))
    crosscorr_mcK_w = cross_correlation_patterns(precursor, mcK_mean_w.sel(lag=lag))
    crosscorr_we = cross_correlation_patterns(precursor, w_eofs.sel(lag=lag))
    
    
    ROC_mcK = ROC_score(predictions=crosscorr_mcK, observed=mcKts, threshold_event=hotdaythreshold)
    ROC_mcK_w = ROC_score(predictions=crosscorr_mcK_w, observed=mcKts, threshold_event=hotdaythreshold)
    ROC_weof = ROC_score(predictions=crosscorr_we, observed=mcKts, threshold_event=hotdaythreshold)
    print('\n*** ROC score lag {} ***\n\nMck {:.2f} \t Mck_w {:.2f} \t eof_w {:.2f}'.format(lag, ROC_mcK,
                                          ROC_mcK_w, ROC_weof))
    








#%% Depricated:

## EOFS on composite hot days
#lag = 0
#neofs = 4
#scaling = 0
#
#
#def plotting_eofs(xarray, lag, scaling, neofs, title, kwrgs=kwrgs):
#    eof_output, solver = func_mcK.EOF(xarray, neofs=6)
#    eof_output = solver.eofs(neofs=neofs, eofscaling=scaling)
#    explained_var = [float(solver.eigenvalues()[x]/np.sum(solver.eigenvalues()).values) for x in range(neofs)] 
#    mode_expl = ['{}, {:02.1f}% expl'.format(x+1, explained_var[x]*100) for x in range(neofs) ]
#    eof_output.attrs['units'] = 'mode'
#    eof_output['mode'] = mode_expl
#    kwrgs = dict( {'vmin' : -eof_output.max(), 'vmax' : eof_output.max(), 'title' : title, 'clevels' : 'notdefault',
#                   'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )  
#    file_name = os.path.join(ex['fig_path'], 
#                 title + ' - lag {}.png'.format(lag))
#    plotting = eof_output
#    func_mcK.finalfigure(plotting, file_name, kwrgs)
#
#
#title = 'EOFs on composite (hot days, n={})'.format(matchhotdates.time.size) 
#dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
#varhotdays = varfullreg.sel(time=dates_min_lag)
#plotting_eofs(Composite, lag, scaling, neofs, title, kwrgs=kwrgs)
##%% EOFS on full summer
##title = 'EOFs on full summer'
##plotting_eofs(varsumreg, lag, scaling, neofs, title, kwrgs=kwrgs)
#
##%% Project composite on main EOFs:
#lag = 0
#n_eofs_used = 20
#dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
#Composite = varfullreg.sel(time=dates_min_lag)
#summerdays_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
#totaltimeserie = varfullreg.sel(time=summerdays_min_lag)
#
#
#important_eofs, wmean_eofs, PC_imp_abs = func_mcK.extract_pattern(Composite, totaltimeserie, scaling, n_eofs_used)
#
##xrdata = xr.DataArray(data=wmeanmodes.values, coords=[['mean'], varsumreg.latitude, varsumreg.longitude], 
##                      dims=['eof','latitude','longitude'], name='weigted_mean_eofs')
#func_mcK.xarray_plot(wmean_eofs)
