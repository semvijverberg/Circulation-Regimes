# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Tue Jul 10 11:51:50 2018

@author: semvijverberg
"""
import os
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
import functions
import numpy as np
import plotting
import what_variable_pp
import xarray as xr
import pandas as pd
from netCDF4 import num2date
import pickle
Variable = what_variable_pp.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region

# load post processed data
#path = '/Users/semvijverberg/surfdrive/Data_ERAint/t2m_u_m4-8_dt10/1jun-24aug_2.5natearth_with_US_mask/'
#ex = np.load(os.path.join(path, 'input_dic_part_1.npy')).item()
new = '/Users/semvijverberg/surfdrive/Data_ERAint/t2mmax_sst_m3-08_dt14/1jun-24aug_averAggljacc_tf14_n8_lag1-1/input_dic_part_1.npy'
ex = np.load(new, encoding='latin1').item()
#ex = np.load(filename_exp_design1, encoding='latin1').item()
RV_name = 't2mmax'
RV = ex[RV_name]
print('tfreq of ex dic: {} days'.format(ex['tfreq']))

# =============================================================================
# Get binary timeseries of when gridcells exceed 95th percentile
# and convert to station versus time format (dendogram)
# =============================================================================
marray, temperature = functions.import_array(RV, path='pp')

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
marray.coords['mask'] = (('latitude','longitude'), US_mask.mask)
tmaxRVperiod = marray.isel(time=ex['RV_period'])

McKts = functions.Mckin_timeseries(tmaxRVperiod, RV)


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
    output = functions.clustering_spatial(McKts, ex, n, region)
#%% Saving output in dictionary
# settings for tfreq = 14
ex['linkage'] = linkage[1]
data = McKts
n_clusters =8
# Create mask of cluster
output = functions.clustering_spatial(McKts, ex, n_clusters, region, RV)
selclus = 3
mask1 = np.array(np.nan_to_num(output.where(output == selclus)), dtype=bool)
selclus = 3
mask2 = np.array(np.nan_to_num(output.where(output == selclus)), dtype=bool)
mask1[mask2] = True
mask = mask1
tmaxRVperiod.coords['mask'] = (('latitude','longitude'), mask)

def add_mask_to_ncdf(file_name, mask):
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
    RVfullts = RV_array.where(RV_array.mask ==True).mean(dim=['latitude','longitude']).squeeze()
    RVoneyr = RVfullts.where(dates.year == ex['startyear']).dropna(dim='time', how='all')
    # matching RV period
    def finddateidx(date, oneyr):
        McKts_monthday = (int(date.dt.month), int(date.dt.day.values))
        #RV_monthday = (int(date.dt.month), int(date.dt.day.values))
        RV_startdate = oneyr.where((oneyr.dt.month == McKts_monthday[0]) & 
                    (oneyr.dt.day == McKts_monthday[1])).dropna(dim = 'time', how='all')
        idx = list(RVoneyr.time.values).index(RV_startdate.values[0])
        return idx
    
    start_idx = finddateidx(McKts.time.min(), RVoneyr.time)
    end_idx = finddateidx(McKts.time.max(), RVoneyr.time)
    steps_oneyr = RVoneyr.size
    years = np.arange((dates.year.max() - dates.year.min())+1)
    new_RV_period = [np.arange(start_idx, end_idx+1) + i*steps_oneyr for i in years]
    new_RV_period = np.array(new_RV_period).flatten()
    RVts = RVfullts.isel(time=new_RV_period)
    return RV_array, RVfullts, RVts, new_RV_period

# make tmax RV:
RV_array, RVfullts, RVts, new_RV_period = add_mask_to_ncdf(RV.filename_pp, mask)

#file_name = 'z_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
#RV_array, RVfullts, RVts, new_RV_period = add_mask_to_ncdf(file_name, mask)

#%%
var = RV_array.name ; tfreq = file_name[file_name.index('days')-1] 
RV_array.attrs['units'] = 'Geopotential Height [m]'
z_data = RV_array.isel(time=new_RV_period)
cls = RV
# settings clustering
n_clusters = 4
region = 'U.S.wide'
ex['clusmethod'] = methods[1]
linkage = ['complete', 'average']
ex['linkage'] = linkage[0]
z_data_norm = z_data / z_data.std()
z_data_norm.name = 'allts_' + var
z_data_norm.attrs['units'] = 'std'
output = functions.clustering_temporal(z_data_norm, ex, n_clusters, RV, tfreq, region=region)
t_meanplot = plotting.find_region(tmaxRVperiod.drop('mask'), region=region)[0].mean(dim='time')
plotting.xarray_plot(t_meanplot )
                                       
#%%
n_hot = 30

z_data.name = '{}hottest_tf{}{}_tf{}{}'.format(n_hot, ex['tfreq'], RV.name, tfreq, var)
t_spatmean = tmaxRVperiod.where(tmaxRVperiod.mask==True).mean(dim=('latitude', 'longitude'))
t_std = t_spatmean.std()
idx = t_spatmean.argsort().squeeze().values
t_sorted = t_spatmean.isel(time=idx).squeeze()
t_corr_dates = t_sorted[-n_hot:].time
# converting to same hours
dt_hours = t_corr_dates[0].dt.hour - data.time[0].dt.hour
z_corr_dates = t_corr_dates - pd.Timedelta(int(dt_hours), unit='h')
z_hottest = z_data.sel(time=z_corr_dates.values)
z_std = z_data.std()
z_hot_norm = z_hottest / z_std
z_hot_norm.attrs['units'] = 'std'
output = functions.clustering_temporal(z_hot_norm, ex, n_clusters, RV, tfreq, region=region)

folder = os.path.join(cls.base_path,'Clustering_temporal/',
                              '{}deg_tfreq{}'.format(ex['grid_res'],tfreq))
t_hottest = tmaxRVperiod.sel(time=t_corr_dates.values)
t_meanplot = plotting.find_region(t_hottest.drop('mask'), region=region)[0].mean(dim='time')
t_meanplot_norm = t_meanplot / t_std
t_meanplot_norm.name = '{}hottest_{}_{}'.format(n_hot, RV.name, region)
plotting.xarray_plot(t_meanplot_norm, path=folder, saving=True)
z_meanplot = plotting.find_region(z_hottest.drop('mask'), region=region)[0].mean(dim='time')
z_meanplot_norm = z_meanplot / z_std
z_meanplot_norm.name = '{}hottest_{}_{}'.format(n_hot, var, region)
plotting.xarray_plot(z_meanplot_norm, path=folder, saving=True)
                                       
#%%
months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
RV_name_range = '{}{}-{}{}_'.format(RV.datesRV[0].day, months[RV.datesRV.month[0]], 
                 RV.datesRV[-1].day, months[RV.datesRV.month[-1]] )

ex['path_exp_periodmask'] = os.path.join(RV.base_path, ex['exp_pp'], 
                              RV_name_range + ex['linkage'][:4] + ex['clusmethod'][:4] + 
                              ex['distmetric'][:4])
if os.path.isdir(ex['path_exp_periodmask']) != True : os.makedirs(ex['path_exp_periodmask'])
filename = str( RV_name +'_'+ str(ex['startyear']) +'-'+ str(ex['endyear']) +'_'+
               ex['path_exp_periodmask'].split('/')[-1] 
                + '_tf{}_n{}'.format(ex['tfreq'], n_clusters) )
to_dict = dict( {'RVfullts' : RVfullts,
                'clusterout': output,
                'selclus'   : selclus,
                'RV_array'  : RV_array   } )
np.save(os.path.join(ex['path_pp'],'RVts2.5',filename+'.npy'), to_dict)
np.save(os.path.join(ex['path_exp_periodmask'], 'input_tig_dic.npy'), ex)


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
