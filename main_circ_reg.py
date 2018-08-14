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
import regionmask
Variable = what_variable_pp.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region

# load post processed data
path = '/Users/semvijverberg/surfdrive/Data_ERAint/t2mmax_t2m_sst_m6-8_dt14/13Jul-24Aug_ward'
ex = np.load(os.path.join(path, 'input_dic_part_1.npy')).item()
RV = ex['t2mmax']

# =============================================================================
# clustering predictant / response variable
# =============================================================================
RV_period = ex['RV_period']
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
linkage = ['ward','complete', 'average']
ex['clusmethod'] = methods[1] ; ex['linkage'] = linkage ; n_clusters = 5 ; region='U.S.'

marray, temperature = functions.import_array(RV, path='pp')

def Mckin_timeseries(marray, RV):
#    marray.values = np.ma.MaskedArray(marray, mask=np.tile(lsm_mask, (marray.time.size,1,1)))
#    region_values, bounds = find_region(marray, region='U.S.')
    perc = marray.reduce(np.percentile, dim='time', keep_attrs=True, q=95)
    rep_perc = np.tile(perc, (marray.time.size,1,1))
    indic = np.squeeze(marray.where(np.squeeze(marray.values) > rep_perc))
    indic.values = np.nan_to_num(indic)
    indic.values[indic.values > 0 ] = 1
    return indic
McKts = Mckin_timeseries(marray, RV)
#%% clustering spatial
data = McKts
cls = RV
output = functions.clustering_spatial(McKts, ex, n_clusters, region, RV)
#%% Saving output in dictionary
clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True).to_dataset(name='clusters')
to_dict = clusters.to_dict()
RV_name_range = '{}{}-{}{}_'.format(ex['RV_oneyr'].min().day, ex['RV_oneyr'].min().month_name()[:3], 
                 ex['RV_oneyr'].max().day, ex['RV_oneyr'].max().month_name()[:3] )
ex['path_exp_mask_region'] = os.path.join(RV.base_path, ex['exp_pp'], RV_name_range + ex['linkage'][:4])
if os.path.isdir(ex['path_exp_mask_region'] ):
    pass
else:
    os.makedirs(ex['path_exp_mask_region'] )
np.save(os.path.join(ex['path_exp_mask_region'] , 'clusters_dic.npy'), to_dict)
np.save(os.path.join(ex['path_exp_mask_region'], 'input_tig_dic.npy'), exp)



# Depricated

#%% clustering temporal
#output = functions.clustering_temporal(marray, ex['clusmethod'], ex['linkage'], n_clusters, RV, region='U.S.', RV_period=ex['RV_period'])

##%% save to github
#import os
#import subprocess
#runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
#subprocess.call(runfile)
#
#
#
##%% Depricated
## check if days of year are alligned to be able to calculate a meaningful cdo ydaymean
#RV = ex['sst']
#marray, RV = functions.import_array(RV, path='pp')
#time = marray.time.values
#for yr in range(ex['startyear'],ex['endyear']+1):
#    
#    one_year = RV.dates_np.where(RV.dates_np.year == yr).dropna()
#    print yr, len(one_year)
#    print one_year.dayofyear
##%% Check if multi year mean is indeed 0
##print ex['RV_period']
#one_date = marray.sel(time=RV.dates_np[RV_period[::4]])
#one_date.mean(dim='time')[0].plot.contourf()
#
##clusters.to_netcdf(path=os.path.join(temperature.path_pp, 'output_clusters.nc'))
#
#
#import pickle
##pickle.dump(file_path, 'wb')
#def save_obj(obj, name ):
#    with open(name + '.pkl', 'wb') as f:
#        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(to_dict, os.path.join(temperature.path_pp,'clusters'))
#
#test = pickle.load(open(os.path.join(temperature.path_pp,'clusters.pkl'), 'rb'))
##%%
##anom_region = plotting.find_region(anom.mean(dim='time', keep_attrs=True))
##create_masks = np.ma.masked_where(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std(), anom_region)
##create_masks = np.ma.make_mask(clusters.sel(cluster=0)>1*clusters.sel(cluster=0).std())
#
#
#
#
#




#
#
#
##%%
## =============================================================================
## Depricated tests with EOF
## =============================================================================
#
## input data EOF
#region_values, region_coords = find_region(anom, region='EU')
#from eofs.examples import example_data_path
## Read geopotential height data using the xarray module. The file contains
## December-February averages of geopotential height at 500 hPa for the
## European/Atlantic domain (80W-40E, 20-90N).
#filename = example_data_path('hgt_djf.nc')
#z_djf = xr.open_dataset(filename)['z']
## Compute anomalies by removing the time-mean.
#z_djf = z_djf - z_djf.mean(dim='time')
#
#
#
#eof_output = functions.EOF(data, neofs=4)
#eof_output = functions.EOF(region_values, neofs=4)
#PlateCarree_timesteps(eof_output.rename( {'mode':'time'}), temperature, cbar_mode='individual')
#plotting.xarray_plot(eof_output)
#
#exit()
#
#
#
#
#
#
#
#
#
#
#
