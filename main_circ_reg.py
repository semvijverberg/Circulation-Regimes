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

# load post processed data
exp_name = 't2m_sst_m5-8_dt14'
path = '/Users/semvijverberg/surfdrive/Data_ERAint/t2m_sst_m5-8_dt14/input_pp'
exp = np.load(os.path.join(path, exp_name+'_pp_dic.npy')).item()
RV = exp['t2m']

# =============================================================================
# Select Response Variable period (which period you want to predict)
# =============================================================================
# Select reponse variable period
RV = exp[exp['vars'][0][1]]
marray, RV = functions.import_array(RV, path='pp')
one_year = RV.dates_np.where(RV.dates_np.year == RV.startyear+1).dropna()
months = [7,8]
RV_period = []
for mon in months:
    RV_period.insert(-1, np.where(RV.dates_np.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
exp['RV_period'] = RV_period
#%%
# =============================================================================
# clustering predictant / response variable
# =============================================================================
RV_period = exp['RV_period']
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
linkage = ['ward','complete', 'average']
exp['clusmethod'] = methods[1] ; exp['linkage'] = linkage[0] ; n_clusters = 4 ; region='U.S.'

marray, temperature = functions.import_array(RV, path='pp')
#%% clustering temporal
output = functions.clustering_temporal(marray, exp['clusmethod'], exp['linkage'], n_clusters, RV, region='U.S.', RV_period=exp['RV_period'])
#%% clustering spatial
output = functions.clustering_spatial(marray, exp['clusmethod'], n_clusters, temperature)
#%% Saving output in dictionary
clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True).to_dataset(name='clusters')
to_dict = clusters.to_dict()
np.save(os.path.join(temperature.path_pp,'clusters_dic.npy'), to_dict)



#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)



#%% Depricated
#clusters.to_netcdf(path=os.path.join(temperature.path_pp, 'output_clusters.nc'))


import pickle
file_path = temperature.path_pp + '/' + 'clusters' + '.npy'
#pickle.dump(file_path, 'wb')
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
save_obj(to_dict, os.path.join(temperature.path_pp,'clusters'))

test = pickle.load(open(os.path.join(temperature.path_pp,'clusters.pkl'), 'rb'))
#%%
#anom_region = plotting.find_region(anom.mean(dim='time', keep_attrs=True))
#create_masks = np.ma.masked_where(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std(), anom_region)
#create_masks = np.ma.make_mask(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std())












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











