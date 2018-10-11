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

region = 'Pacific'

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
varsumreg = func_mcK.find_region(varsumgl, region=region)[0]
# region mckinnon - full time series
varfullreg = func_mcK.find_region(varfullgl, region=region)[0]
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
kwrgs = dict( {'vmin' : -xrdata.max().values, 'vmax' : xrdata.max().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(xrdata, file_name, kwrgs) 

dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
meanhotdays = varfullreg.sel(time=dates_min_lag).mean(dim='time')

#%% EOFS on composite hot days
lag = 0
neofs = 4
scaling = 0


def plotting_eofs(xarray, lag, scaling, neofs, title, kwrgs=kwrgs):
    eof_output, solver = func_mcK.EOF(xarray, neofs=6)
    eof_output = solver.eofs(neofs=neofs, eofscaling=scaling)
    explained_var = [float(solver.eigenvalues()[x]/np.sum(solver.eigenvalues()).values) for x in range(neofs)] 
    mode_expl = ['{}, {:02.1f}% expl'.format(x+1, explained_var[x]*100) for x in range(neofs) ]
    eof_output.attrs['units'] = 'mode'
    eof_output['mode'] = mode_expl
    kwrgs = dict( {'vmin' : -eof_output.max(), 'vmax' : eof_output.max(), 'title' : title, 'clevels' : 'notdefault',
                   'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )  
    file_name = os.path.join(ex['fig_path'], 
                 title + ' - lag {}.png'.format(lag))
    plotting = eof_output
    func_mcK.finalfigure(plotting, file_name, kwrgs)


title = 'EOFs on composite (hot days, n={})'.format(matchhotdates.time.size) 
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
varhotdays = varfullreg.sel(time=dates_min_lag)
plotting_eofs(varhotdays, lag, scaling, neofs, title, kwrgs=kwrgs)
#%% EOFS on full summer
title = 'EOFs on full summer'
plotting_eofs(varsumreg, lag, scaling, neofs, title, kwrgs=kwrgs)



#%% Project composite on main EOFs:
lag = 0
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
varhotdays = varfullreg.sel(time=dates_min_lag)

# Get oefs from hot days
eof_output, solver = func_mcK.EOF(varhotdays)
# Get oefs from summer days
#eof_output, solver = func_mcK.EOF(varsumreg)


n_eofs_used = 20

def projectfield(xarray, solver, scaling, n_eofs_used):
    eofs = solver.eofs(neofs=n_eofs_used, eofscaling=scaling).values
    n_eofs  = eofs[:,0,0].size
    n_space = eofs[0].size
    n_time  = xarray.time.size
    
    matrix = np.reshape(xarray.values, (n_time, varhotdays[0].size))
    matrix = np.nan_to_num(matrix)

    # convert eof output to matrix
    Ceof = np.reshape( eofs, (n_eofs, n_space) ) 
    C = np.nan_to_num(Ceof)

    # Calculate std over entire eof time series
    PCs_all = solver.pcs(pcscaling=scaling, npcs=n_eofs_used)
    PCstd_all = [float(np.std(PCs_all[:,i])) for i in range(n_eofs_used)]
    
    PCi = np.zeros( (n_time, n_eofs_used) )
    PCstd = np.zeros( (n_eofs_used) )
    PCi_unitvar = np.zeros( (n_time, n_eofs_used) )
    PCi_mean = np.zeros( (n_eofs_used) )
    for i in range(n_eofs_used):
    
        PCi[:,i] = np.dot( matrix, C[i,:])
        PCi_unitvar[:,i] = (PCi[:,i]) / PCstd_all[i]
        
        PCstd[i] = np.std(PCi_unitvar[:,i])
        PCi_mean[i] = np.mean(PCi[:,i])
    

    plt.title('Pcs\nnormal projection of matrix upon eofs')
    for i in range(2):
        plt.axhline(0, color='black')
        plt.plot(PCi[:,i])
        plt.axhline(PCi_mean[i])
    plt.figure()
    plt.title('PCs\nnormalized by std (std from whole summer PC timeseries)')
    for i in range(2):
        plt.axhline(0, color='black')
        plt.plot(PCi_unitvar[:,i])
        plt.axhline(PCi_mean[i]/PCstd_all[i])
    plt.figure()
    plt.plot(solver.eigenvalues()[:n_eofs])
#    print('Mean value PC time series: {}'.format(PCi_mean[0]))
#    print('std value PC time series: {}'.format(PCstd[0]))
    return PCi, PCi_unitvar, PCstd, PCi_mean

# Project hot day fields upon EOFS 
PCi, PCi_unitvar, PCstd_hot, PCi_mean_hot = projectfield(varhotdays, solver, scaling, n_eofs_used)

#%% Project 'not extreme patterns' on main EOFs
#varnothot = varfullreg.drop(labels=dates_min_lag.values, dim='time')


PCi, PCi_unitvar, PCstd_nor, PCi_mean_nor = projectfield(varfullreg, solver, scaling, n_eofs_used)

#%% calculate ratio:
PCs_all = solver.pcs(pcscaling=scaling, npcs=n_eofs_used)
PCstd_all = [float(np.std(PCs_all[:,i])) for i in range(n_eofs_used)]
ratio_std = PCstd_nor / PCstd_hot
ratio_mean = (PCi_mean_nor / PCi_mean_hot) * np.median(PCi_mean_nor)**-1
plt.figure()
plt.title('Ratio variability in \'normal\' time series\ndivided by variability between hot days')
plt.plot(ratio_std)
plt.figure()
plt.title('Ratio of mean PC values in \'normal\' time series\ndivided by hot days')
plt.ylim( (np.min(ratio_mean)-1, np.max(ratio_mean)+1  ) )
plt.plot(ratio_mean)

plt.figure()
plt.ylim( (np.min(PCi_mean_hot/PCstd_all)-1, np.max(PCi_mean_hot/PCstd_all)+1  ) )
plt.plot(PCi_mean_hot/PCstd_all)
plt.plot(PCi_mean_nor/PCstd_all)

#%%
def get_relevant_modes(data, n_eofs_used):
    xarray = xr.DataArray(data=data, coords=[range(n_eofs_used)], 
                          dims=['eofs'], name='xarray')
    anomalous = xarray.where(abs(xarray.values) > (xarray.mean(dim='eofs') + xarray.std()).values)
    return anomalous.dropna(how='all', dim='eofs')

PC_abs = get_relevant_modes(PCi_mean_hot/PCstd_all, n_eofs_used)
PC_var = get_relevant_modes(ratio_std, n_eofs_used)
PC_rel = get_relevant_modes(ratio_mean, n_eofs_used)


#%%
xarray_important_eofs = PC_abs
important_modes = list(xarray_important_eofs.eofs.values)
absolute_values = xr.DataArray(data=PCi_mean_hot, coords=[range(n_eofs_used)], 
                          dims=['eofs'], name='xarray')
array = np.zeros( (len(important_modes),varsumreg.latitude.size, varsumreg.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[important_modes, varsumreg.latitude, varsumreg.longitude], 
                      dims=['eof','latitude','longitude'], name='eofs')
eofs = solver.eofs(neofs=n_eofs_used, eofscaling=scaling)


for eof in important_modes:
    idx = important_modes.index(eof)
    single_eof = eofs.sel(mode=eof) * np.sign(absolute_values.sel(eofs=eof))
    xrdata[idx] = single_eof
    
xrdata.attrs['units'] = 'eof'
file_name = os.path.join(ex['fig_path'], 
             'important eofs')
title = 'important eofs'
kwrgs = dict( {'vmin' : -single_eof.max(), 'vmax' : single_eof.max(), 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(xrdata, file_name, kwrgs) 



weights = abs(xarray_important_eofs) / np.mean(abs(xarray_important_eofs))
for eof in important_modes:
    idx = important_modes.index(eof)
    single_eof = eofs.sel(mode=eof) * np.sign(absolute_values.sel(eofs=eof))
    xrdata[idx] = single_eof * weights.sel(eofs=eof)

wmeanmodes = xrdata.mean(dim='eof', keep_attrs = True) 
#xrdata = xr.DataArray(data=wmeanmodes.values, coords=[['mean'], varsumreg.latitude, varsumreg.longitude], 
#                      dims=['eof','latitude','longitude'], name='weigted_mean_eofs')
func_mcK.xarray_plot(wmeanmodes)

