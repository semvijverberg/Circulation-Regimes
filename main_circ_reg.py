#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:51:50 2018

@author: semvijverberg
"""
import os
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
#import what_input
#import retrieve_ERA_i
import functions
import numpy as np
import plotting
import what_variable_pp
Variable = what_variable_pp.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region

exp_name = 'exp1'
path = '/Users/semvijverberg/surfdrive/Data_ERAint/input_pp_exp1'
exp = np.load(os.path.join(path, exp_name+'_dic.npy')).item()
RV = exp[exp['vars'][0][0]]
# Select reponse variable period
exp['RV_period'] = np.where((RV.dates_np.month.values == 6).all() or RV.dates_np.month == 7)[0]
#one_year = RV.dates_np.where(RV.dates_np.year==RV.startyear).dropna()
#exp['RV_period'] = one_year.where(one_year > one_year[-8]).fillna(value='NaT', downcast=None)
print exp['RV_period']
# =============================================================================
# =============================================================================
# # solve RV_period
# =============================================================================
# =============================================================================

#%% assign instance
#(self, name, dataset, startyear, endyear, startmonth, endmonth, grid, tfreq, units)
#temperature = Variable(name='2_metre_temperature', dataset='ERA-i', startyear=1979, endyear=2017, 
#                       startmonth=3, endmonth=9, tfreq=tfreq, grid=2.5, exp=exp['exp'])
#%%
# =============================================================================
# clustering predictant / response variable
# =============================================================================

marray, temperature = functions.import_array(RV, path='pp')
#clim, anom, std = functions.calc_anomaly(marray, temperature)
marray
#%%
# =============================================================================
# clustering tests
# =============================================================================

cls = RV
RV_period = exp['RV_period']
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
linkage = ['ward','complete', 'average']
method = methods[1] ; linkage = linkage[2]; n_clusters = 4; month=6; region='U.S.'
data = marray
#%% clustering temporal
output = functions.clustering_temporal(data, method, linkage, n_clusters, RV, region='U.S.', RV_period=exp['RV_period'])

#%% clustering spatial
#output = functions.clustering_spatial(data, method, n_clusters, temperature)
clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True).to_dataset(name='clusters')
#clusters.to_netcdf(path=os.path.join(temperature.path_pp, 'output_clusters.nc'))
to_dict = clusters.to_dict()
np.save(os.path.join(temperature.path_pp,'clusters_dic.npy'), to_dict)
#%%
import pickle
file_path = temperature.path_pp + '/' + 'clusters' + '.npy'
#pickle.dump(file_path, 'wb')
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
save_obj(to_dict, os.path.join(temperature.path_pp,'clusters'))

test = pickle.load(open(os.path.join(temperature.path_pp,'clusters.pkl'), 'rb'))
#%%
anom_region = plotting.find_region(anom.mean(dim='time', keep_attrs=True))
create_masks = np.ma.masked_where(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std(), anom_region)
create_masks = np.ma.make_mask(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std())

#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)










#%%
# =============================================================================
# Depricated tests with EOF
# =============================================================================

# input data EOF
region_values, region_coords = find_region(anom, region='EU')
from eofs.examples import example_data_path
# Read geopotential height data using the xarray module. The file contains
# December-February averages of geopotential height at 500 hPa for the
# European/Atlantic domain (80W-40E, 20-90N).
filename = example_data_path('hgt_djf.nc')
z_djf = xr.open_dataset(filename)['z']
# Compute anomalies by removing the time-mean.
z_djf = z_djf - z_djf.mean(dim='time')



eof_output = functions.EOF(data, neofs=4)
eof_output = functions.EOF(region_values, neofs=4)
PlateCarree_timesteps(eof_output.rename( {'mode':'time'}), temperature, cbar_mode='individual')
plotting.xarray_plot(eof_output)

exit()











