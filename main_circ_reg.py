# -*- coding: utf-8 -*-
#!/usr/bin/env python2
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
#import regionmask
import pickle
Variable = what_variable_pp.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region

# load post processed data
path = '/Users/semvijverberg/surfdrive/Data_ERAint/t2mmax_t2m_sst_m6-8_dt14/15jun-24aug_2.5natearth_with_US_mask'
ex = np.load(os.path.join(path, 'input_dic_part_1.npy')).item()
RV = ex['t2mmax']


# =============================================================================
# Get binary timeseries of when gridcells exceed 95th percentile
# and convert to station versus time format (dendogram)
# =============================================================================
marray, temperature = functions.import_array(RV, path='pp')

# add mask to marray:
path_masks = os.path.join('/Users/semvijverberg/surfdrive/Scripts/rasterio', ex['maskname']+'.npy') 
US_mask = np.load(path_masks).item()['US_mask']
nor_lon = US_mask.longitude
US_mask = US_mask.roll(longitude=2)
US_mask['longitude'] = nor_lon
xarray_plot(US_mask)
marray.coords['mask'] = (('latitude','longitude'), US_mask.mask)
RV_period = marray.isel(time=ex['RV_period'])

McKts = functions.Mckin_timeseries(marray, RV)


# =============================================================================
# clustering predictant / response variable
# =============================================================================

methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
linkage = ['complete', 'average']
ex['distmetric'] = 'jaccard'
ex['clusmethod'] = methods[1] ; ex['linkage'] = linkage[0] ; region='U.S.'
n_clusters = [2, 3, 4, 5, 6]

for n in n_clusters:
    output = functions.clustering_spatial(McKts, ex, n, region, RV)

#%% Saving output in dictionary
data = McKts
n_clusters = 5
# Create mask of cluster
output = functions.clustering_spatial(McKts, ex, n_clusters, region, RV)
selclus = 3
arr = np.nan_to_num(output.where(output == selclus))
tmax = marray
tmax.coords['mask'] = (('latitude','longitude'), np.array(arr,dtype=bool))
tmaxfullts = tmax.where(tmax.mask ==True).mean(dim=['latitude','longitude']).squeeze()

RV_name_range = '{}{}-{}{}_'.format(ex['RV_oneyr'].min().day, ex['RV_oneyr'].min().month_name()[:3], 
                 ex['RV_oneyr'].max().day, ex['RV_oneyr'].max().month_name()[:3] )
ex['path_exp_periodmask'] = os.path.join(RV.base_path, ex['exp_pp'], 
                              RV_name_range + ex['linkage'][:4] + ex['clusmethod'][:4] + 
                              ex['distmetric'][:4])
if os.path.isdir(ex['path_exp_periodmask']) != True : os.makedirs(ex['path_exp_periodmask'])
filename = ex['path_exp_periodmask'].split('/')[-1]
to_dict = dict( {'RVfullts' : tmaxfullts} )
np.save(os.path.join(ex['path_pp'] , filename+'.npy'), to_dict)
np.save(os.path.join(ex['path_exp_periodmask'], 'input_tig_dic.npy'), ex)


#pickle.dump(file_path, 'wb')
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
save_obj(to_dict, os.path.join(ex['path_pp'],filename))


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
