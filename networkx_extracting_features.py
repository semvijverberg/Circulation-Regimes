#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:40:40 2018

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
map_proj = ccrs.Miller(central_longitude=240)  


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

#'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Southern'



region = 'Whole'
print(region)

# Load in mckinnon Time series
T95name = 'PEP-T95TimeSeries.txt'
mcKtsfull, datesmcK = func_mcK.read_T95(T95name, ex)
datesmcK_daily = func_mcK.make_datestr(datesmcK, ex)

# Selected Time series of T95 ex['sstartdate'] until ex['senddate']
mcKts = mcKtsfull.sel(time=datesmcK_daily)


# Load in external ncdf
ex['name'] = 'sst'
#filename = '{}_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
filename = '{}_1979-2017_2jan_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
# full globe - full time series
varfullgl = func_mcK.import_array(filename, ex)
## region mckinnon - full time series
varfullreg = func_mcK.find_region(varfullgl, region=region)[0]
## Converting Mckinnon timestemp to match xarray timestemp
matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)
ex['tfreq'] = 1 
## full globe - only (mcKinnon) summer days (Juni, Juli, August)
#varsumgl = varfullgl.sel(time=matchdaysmcK)
## region mckinnon - Mckinnon summer days (Juni, Juli, August)
#varsumreg = func_mcK.find_region(varsumgl, region=region)[0]
## region mckinnon - full time series
#varfullreg = func_mcK.find_region(varfullgl, region=region)[0]


## region mckinnon - only (mcKinnon) summer 
#func_mcK.xarray_plot(varsumreg.mean(dim='time')) 

#for i in range(3):
#    func_mcK.xarray_plot(varsumreg.isel(time=i)) 

#%%
# take means over bins over tfreq days
ex['tfreq'] = 1 
mcKts, datesmcK = func_mcK.time_mean_bins(mcKts, ex)

def oneyr(datetime):
    return datetime.where(datetime.year==datetime.year[300]).dropna()

datetime = datesmcK

def expand_times_for_lags(datesmcK, ex):
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




expanded_time = expand_times_for_lags(datesmcK, ex)
# Converting Mckinnon timestemp to match xarray timestemp
expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
# region mckinnon - full time series
varfselreg = varfullgl.sel(time=expandeddaysmcK)

varfullreg = func_mcK.find_region(varfselreg, region=region)[0]
varfullreg, datesvar = func_mcK.time_mean_bins(varfullreg, ex)

matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)
varsumreg = varfullreg.sel(time=matchdaysmcK)


# binary time serie when T95 exceeds 1 std
hotdaythreshold = mcKts.mean(dim='time').values + mcKts.std().values
hotts = mcKts.where( mcKts.values > hotdaythreshold) 
hotdates = hotts.dropna(how='all', dim='time').time
hotindex = np.where( np.isnan(hotts) == False )[0]




## plotting same figure as in paper

year2012 = mcKts.where(mcKts.time.dt.year == 2012).dropna(dim='time', how='any')
plotpaper = mcKts.sel(time=pd.DatetimeIndex(start=year2012.time.values[0], 
                                            end=year2012.time.values[-1], 
                                            freq=(datesmcK[1] - datesmcK[0])))
#plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
#                                freq=(datesmcK[1] - datesmcK[0])))

plotpaperhotdays = plotpaper.where( plotpaper.values > hotdaythreshold) 
plotpaperhotdays = plotpaperhotdays.dropna(how='all', dim='time').time
plt.figure()
plotpaper.plot()
plt.axhline(y=hotdaythreshold)
for days in plotpaperhotdays.time.values:
    plt.axvline(x=days)






std = varfullreg.std(dim='time')
# not merging hot days which happen consequtively

matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfullreg.time[0].dt.hour)
# Select concurrent hot days
varhotdays = varfullreg.sel(time=matchhotdates)

#matchpaperplothotdates = func_mcK.to_datesmcK(plotpaperhotdays.time, hotdates[0].dt.hour, varhotdays.time[0].dt.hour)
#test = varhotdays.sel(time=matchpaperplothotdates[:5])
#for day in matchpaperplothotdates[:5]:
#    func_mcK.xarray_plot(varfullreg.sel(time=day)) 

#%% Mean over 230 hot days
#lags = [20, 30, 40, 50]
#lags = [0, 5]
lags = [0, 50]

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
folder = os.path.join(ex['fig_path'], 'mcKinnon_mean')
if os.path.isdir(folder) != True : os.makedirs(folder)
fname = '{} - mean composite tf{} lags {} {}.png'.format(ex['name'], ex['tfreq'],
         lags, region)
file_name = os.path.join(folder, fname)

title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -3*mcK_mean.std().values, 'vmax' : 3*mcK_mean.std().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(mcK_mean, file_name, kwrgs) 
#%%

import networkx as nx
n_std = 1.5


xarray = mcK_mean.sel(lag=0)
nparray = np.reshape(xarray.values, xarray.size)
mask_threshold = abs(nparray) < ( n_std * np.std(nparray) )



Corr_Coeff = np.ma.MaskedArray(nparray, mask=mask_threshold)
lat_grid = xarray.latitude.values
lon_grid = xarray.longitude.values

npmap = np.ma.reshape(Corr_Coeff, (len(lat_grid), len(lon_grid)))
plt.imshow(npmap)


def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid):
	'''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
	'''
	print('extracting causal precursor regions ...\n')

	
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
	return N_pot
#	Regions = merge_neighbors(N_pot)
define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid)
