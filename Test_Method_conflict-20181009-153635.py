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
# Load in external ncdf
sst_ERAname = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
t2m_ERAname = 't2mmax_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
# full globe - full time series
varfullgl = func_mcK.import_array(t2m_ERAname, ex)
matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)
mcKts = mcKtsfull.sel(time=datesmcK)
# only hot days
var = varfullgl.sel(time=matchdaysmcK)
# region mckinnon - full time series
varfull = func_mcK.find_region(varfullgl, region='Mckinnonplot')[0]

hotts = mcKts.where( mcKts.std().values>mcKts.values) 
plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
                                freq=(datesmcK[1] - datesmcK[0])))
plotpaper.plot()
std = var.std(dim='time')
# not merging hot days which happen consequtively
hotdates = hotts.dropna(how='all', dim='time').time
matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfullgl.time[0].dt.hour)


#%% EOFS on composite hot days
lag = 50
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
varhotdays = varfull.sel(time=dates_min_lag)
eof_output, solver = func_mcK.EOF(varfull, neofs=6)
title = 'EOFs on composite (hot days, n={})'.format(matchhotdates.time.size) 
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             title + ' - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)

#%% Project composite on main EOFs:
matrix = np.reshape(varhotdays.values, (varhotdays.time.size, varhotdays[0].size))
eofs = solver.eofs().values
n_eofs  = eofs[:,0,0].size
n_space = eofs[0].size
n_time  = varhotdays.time.size

Ceof = np.reshape( eofs, (n_eofs, n_space) ) 
C = Ceof


n_eofs_used = 250
PCs_all = solver.pcs(pcscaling=0, npcs=n_eofs_used)
PCstd_all = [float(np.std(PCs_all[:,i])) for i in range(n_eofs_used)]

PCi = np.zeros( (n_time, n_eofs_used) )
PCstd_hot = np.zeros( (n_eofs_used) )
PCi_unitvar_hot = np.zeros( (n_time, n_eofs_used) )
PCi_mean = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):
    PCstd_hot[i] = np.std(PCi_unitvar_hot[:,i])
    PCi[:,i] = np.dot( matrix, C[i,:] )
    PCi_unitvar_hot[:,i] = (PCi[:,i]) / PCstd_all[i]
    
    PCstd_hot[i] = np.std(PCi_unitvar_hot[:,i])
    PCi_mean[i] = np.mean(PCi_unitvar_hot[:,i])
    
plt.plot(PCi[:,0])
plt.plot(PCi[:,-1])
plt.figure()
plt.plot(solver.eigenvalues()[:n_eofs])
print('Mean value PC time series hot: {}'.format(PCi_mean[0]))
print('std value PC time series hot: {}'.format(PCstd_hot[0]))
#%% Project 'not extreme patterns' on main EOFs
varnothot = varfull.drop(labels=dates_min_lag.values, dim='time')
matrix = np.reshape(varnothot.values, (varnothot.time.size, varnothot[0].size))
eofs = solver.eofs().values
n_eofs  = eofs[:,0,0].size
n_space = eofs[0].size
n_time  = varnothot.time.size

Ceof = np.reshape( eofs, (n_eofs, n_space) ) 
C = Ceof


PCi = np.zeros( (n_time, n_eofs_used))
PCstd_nor = np.zeros( (n_eofs_used) )
PCmean_nor = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):
    PCi[:,i] = np.dot( matrix, C[i,:] )
    PCstd_nor[i] = np.std(PCi[:,i])
    PCmean_nor[i] = np.mean(PCi[:,i])
plt.plot(PCi[:,0])
plt.plot(PCi[:,-1])
print('Mean value PC time series normal: {}'.format(PCmean_nor[0]))
#%% calculate ratio:
ratio_std = np.zeros( (n_eofs_used) )
ratio_mean = np.zeros( (n_eofs_used) )
for i in range(n_eofs_used):
    ratio_std[i] = PCstd_nor[i] / PCstd_hot[i]
    ratio_mean[i] = abs(PCmean_hot[i]) / abs(PCmean_nor[i])
plt.figure()
plt.plot(ratio_std)
plt.figure()
plt.ylim( (np.min(ratio_mean), np.max(ratio_mean)  ) )
plt.plot(ratio_mean)