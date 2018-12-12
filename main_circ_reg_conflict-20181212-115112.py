# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Tue Jul 10 11:51:50 2018

@author: semvijverberg
"""
import os, sys
# also importing pre procces function from RGCPD
os.sys.path.append('/Users/semvijverberg/surfdrive/Scripts/RGCPD/RGCPD')
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import functions
import numpy as np
import plotting
import what_variable_pp
import xarray as xr
import pandas as pd
from netCDF4 import num2date
import functions_pp
import pickle
Variable = what_variable_pp.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw)
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)

# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information for the experiment
# The dic is saved after the post-processes step, so you can continue the experiment
# from this point onward with different configurations. It also stored as a log
# in the final output.
#
ex = dict(
     {'vars'        :       [['t2mmax']],
      'dataset'     :       'ERA-i',
     'grid_res'     :       2.5,
     'startyear'    :       1979, # download startyear
     'endyear'      :       2017, # download endyear
     'startmonth'   :       1,
     'endmonth'     :       1,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'tfreq'        :       14,
     'RV_months'    :       [5,6,7,8],
     'fig_path'     :       "/Users/semvijverberg/surfdrive/MckinRepl"}
     )

ex['sstartdate'] = '{}-01-1'.format(ex['startyear'])
ex['senddate']   = '{}-08-31'.format(ex['startyear'])

ex['RVnc_name'] = [ex['vars'][0][0], '{}_1979-2017_1_12_daily_2.5deg.nc'.format(ex['vars'][0][0])]

ex['precursor_ncdf'] = [['z', 'z_1979-2017_1_12_daily_2.5deg.nc']]
                   


RV = functions_pp.Var_import_RV_netcdf(ex)
ex[ex['vars'][0][0]] = RV
functions_pp.perform_post_processing(ex)


#%%
RV_period = []
for mon in ex['RV_months']:
    # append the indices of each year corresponding to your RV period
    RV_period.insert(-1, np.where(RV.dates.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
ex['RV_period'] = RV_period
RV.datesRV = RV.dates[RV_period]

# import array
marray, temperature = functions.import_array(RV, path='pp')
print('length of total time series: {}'.format(marray.time.size))


# =============================================================================
# Get binary timeseries of when gridcells exceed 95th percentile
# and convert to station versus time format (dendogram)
# =============================================================================

    
#print('tfreq of ex dic: {} days'.format(ex['tfreq']))
#%%
# add mask to marray:
path_masks = os.path.join('/Users/semvijverberg/surfdrive/Scripts/rasterio', 
                          '{}natearth_with_US_mask.npy'.format(ex['grid_res'])) 
US_mask = np.load(path_masks, encoding='latin1').item()['US_mask']
nor_lon = US_mask.longitude
US_mask = US_mask.roll(longitude=2)
US_mask['longitude'] = nor_lon
#nor_lat = US_mask.latitude
#US_mask = US_mask.roll(latitude=1)
#US_mask['latitude'] = nor_lat
plotting.xarray_mask_plot(US_mask)
#%%
# select RV period for spatial clustering
RV.arrRVperiod = marray.isel(time=ex['RV_period'])
print('length of RV period {}'.format(RV.arrRVperiod.time.size))
RV.binarytocluster = functions.Mckin_timeseries(RV.arrRVperiod, RV)
RV.binarytocluster.coords['mask'] = (('latitude','longitude'), US_mask.mask)
#tmaxRVperiod = marray.isel(time=ex['RV_period'])




# =============================================================================
# clustering predictant / response variable
# =============================================================================
ex['name'] = RV.name
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
linkage = ['complete', 'average']
ex['distmetric'] = 'jaccard'
ex['clusmethod'] = methods[1] ; ex['linkage'] = linkage ; region='U.S.'

#%%
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
for n in n_clusters:
    output = functions.clustering_spatial(RV.binarytocluster, ex, n, region)
#%% Adding mask of the cluster found by spatial clustering to another netcdf
    
# settings for tfreq = 14
ex['linkage'] = linkage[1]           
data = RV.binarytocluster
n_clusters =8

# Create name for spatial clustering
name_spcl = ex['linkage'][:4] + ex['clusmethod'][:4] + ex['distmetric'][:4] + \
            '_' + str(ex['tfreq']) + '_{}'.format(n_clusters)

# Create mask of cluster
output = functions.clustering_spatial(RV.binarytocluster, ex, n_clusters, region)
selclus = 1
mask1 = np.array(np.nan_to_num(output.where(output == selclus)), dtype=bool)
selclus = 1
mask2 = np.array(np.nan_to_num(output.where(output == selclus)), dtype=bool)
mask1[mask2] = True
mask = mask1
print('adding mask to 3d array, with time matching the RV period')
RV.arrRVperiod.coords['mask'] = (('latitude','longitude'), mask)

def add_mask_to_ncdf(file_name, mask, ex):
    # =============================================================================
    # # Load geopotential Height
    # =============================================================================
    file_path = os.path.join(ex['path_pp'], file_name)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    numtime = ncdf.variables['time'].values
    timeattr = ncdf.variables['time'].attrs
    dates = pd.to_datetime(num2date(numtime[:], units=timeattr['units'], calendar=timeattr['calendar']))
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
    var = var.replace(' ', '')
    RV_array = np.squeeze(ncdf.to_array(file_path)).rename(({file_path: var}))
    RV_array.name = var
    RV_array['time'] = dates
    RV_array.coords['mask'] = (('latitude','longitude'), mask)
    lats = RV_array.latitude.values
    cos_box = np.cos(np.deg2rad(lats))
    cos_box_array = np.tile(cos_box, (RV_array.longitude.size,1) )
    weights_box = np.swapaxes(cos_box_array, 1,0)
    weights_box = weights_box / np.mean(weights_box)
    RVarray_w = weights_box[None,:,:] * RV_array
    RVfullts = (RVarray_w).where(RV_array.mask).mean(
                                    dim=['latitude','longitude']).squeeze()
    RVoneyr = RVfullts.where(dates.year == ex['startyear']).dropna(dim='time', how='all')
    # matching RV period
    def finddateidx(date, oneyr):
        McKts_monthday = (int(date.dt.month), int(date.dt.day.values))
        #RV_monthday = (int(date.dt.month), int(date.dt.day.values))
        RV_startdate = oneyr.where((oneyr.dt.month == McKts_monthday[0]) & 
                    (oneyr.dt.day == McKts_monthday[1])).dropna(dim = 'time', how='all')
        idx = list(RVoneyr.time.values).index(RV_startdate.values[0])
        return idx
    
    start_idx = finddateidx(RV.arrRVperiod.time.min(), RVoneyr.time)
    end_idx = finddateidx(RV.arrRVperiod.time.max(), RVoneyr.time)
    steps_oneyr = RVoneyr.size
    years = np.arange((dates.year.max() - dates.year.min())+1)
    new_RV_period = [np.arange(start_idx, end_idx+1) + i*steps_oneyr for i in years]
    new_RV_period = np.array(new_RV_period).flatten()
    RVts = RVfullts.isel(time=new_RV_period)
    return RV_array, RVfullts, RVts, new_RV_period

# make tmax RV:
#RV_array, RVfullts, RVts, new_RV_period = add_mask_to_ncdf(RV.filename_pp, mask, ex)



#%%
#ex['RVnc_name'] = [ex['vars'][0][0], '{}_1979-2017_1_12_daily_2.5deg.nc'.format(ex['vars'][0][0])]

ex['tfreq'] = 1
var_class = functions_pp.Var_import_precursor_netcdf(ex, 0)
ex[var_class.name] = var_class
functions_pp.perform_post_processing(ex)

CT_array, CTfullts, CTts, new_RV_period = add_mask_to_ncdf(var_class.filename_pp, mask, ex)

var = var_class.name ; tfreq = ex['tfreq']
CT_array.attrs['units'] = '...'
CT_data = CT_array.isel(time=new_RV_period)
cls = ex[ ex['vars'][0][1] ]
# settings clustering
n_clusters = 4
region = 'U.S.wide'
ex['clusmethod'] = methods[1]
linkage = ['complete', 'average']
ex['linkage'] = linkage[0]
CT_data_norm = CT_data / CT_data.std()
CT_data_norm.name = 'allts_' + var
CT_data_norm.attrs['units'] = 'std'
output = functions.clustering_temporal(CT_data_norm, ex, n_clusters, RV, tfreq, region=region)

#t_meanplot = plotting.find_region(tmaxRVperiod.drop('mask'), region=region)[0].mean(dim='time')
#plotting.xarray_plot(t_meanplot )
                                       
#%% Temporal clustering of n shottest days

n_hot = int(0.15 * RV.arrRVperiod.time.size) # 15 % hottest
n_tempclusters = 4
CT_data.name = '{}hottest_{}_CT_{}_tf{}'.format(n_hot, RV.name, var_class.name, ex['tfreq'])
t_spatmean = RV.arrRVperiod.where(RV.arrRVperiod.mask).mean(dim=('latitude', 'longitude'))
t_std = t_spatmean.std()
# get most anomalous RV.arrRVperiod days
idx = t_spatmean.argsort().squeeze().values
t_sorted = t_spatmean.isel(time=idx).squeeze()
t_corr_dates = t_sorted[-n_hot:].time
# converting to same hours
dt_hours = t_corr_dates[0].dt.hour - data.time[0].dt.hour
CT_corr_dates = t_corr_dates - pd.Timedelta(int(dt_hours), unit='h')
CT_hottest = CT_array.sel(time=CT_corr_dates.values)
CT_std = CT_array.std()
CT_hot_norm = CT_hottest / CT_std
CT_hot_norm.attrs['units'] = 'std'
output = functions.clustering_temporal(CT_hot_norm, ex, n_tempclusters, RV, tfreq, region=region)

folder = os.path.join(ex['fig_path'],'Clustering_temporal',
                              '{}_{}_{}deg_tfreq{}'.format(RV.name, var_class.name,
                               ex['grid_res'],ex['tfreq']))
if os.path.isdir(folder) == False : os.makedirs(folder)
t_hottest = RV.arrRVperiod.sel(time=t_corr_dates.values)
t_meanplot = plotting.find_region(t_hottest.drop('mask'), region=region)[0].mean(dim='time')
t_meanplot_norm = t_meanplot / t_std
t_meanplot_norm.name = '{}hottest_{}_{}'.format(n_hot, RV.name, region)
plotting.xarray_plot(t_meanplot_norm, path=folder, saving=True)
CT_meanplot = plotting.find_region(CT_hottest.drop('mask'), region=region)[0].mean(dim='time')
CT_meanplot_norm = CT_meanplot / CT_std
CT_meanplot_norm.name = '{}hottest_{}_{}'.format(n_hot, var, region)
plotting.xarray_plot(CT_meanplot_norm, path=folder, saving=True)
                                       
#%%

output_dic_folder = os.path.join(ex['fig_path'], 'RVts2.5')
filename = RV.name +'_'+ str(ex['startyear']) +'-'+ str(ex['endyear']) \
               +'_'+ name_spcl + '_to_{}'.format(var_class.name)
               
                
months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
dates_name = '{}{}-{}{}_'.format(var_class.dates[0].day, months[var_class.dates.month[0]], 
                 var_class.dates[-1].day, months[var_class.dates.month[-1]] )


if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
to_dict = dict( {'RVfullts' : CTfullts,
                'clusterout': output,
                'selclus'   : selclus,
                'RV_array'  : CT_array   } )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)
#np.save(os.path.join(ex['path_exp_periodmask'], 'input_tig_dic.npy'), ex)


#pickle.dump(file_path, 'wb')
#def save_obj(obj, name ):
#    with open(name + '.pkl', 'wb') as f:
#        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(to_dict, os.path.join(ex['path_pp'],'RVts',filename))


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
