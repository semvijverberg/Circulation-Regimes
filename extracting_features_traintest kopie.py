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
xrplot = func_mcK.xarray_plot
import matplotlib.pyplot as plt
import scipy

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
map_proj = ccrs.Miller(central_longitude=240)  


ex = dict(
     {'grid_res'     :       2.5,
     'startyear'    :       1982,
     'endyear'      :       2015,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'sstartdate'   :       '1982-06-01',
     'senddate'     :       '1982-08-31',
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

# filter out outliers of sst
if ex['name']=='sst':
    varfullgl.where(varfullgl.values < 3.5*varfullgl.std().values)



# take means over bins over tfreq days
ex['tfreq'] = 1
mcKts, datesmcK = func_mcK.time_mean_bins(mcKts, ex)

def oneyr(datetime):
    return datetime.where(datetime.year==datetime.year[300]).dropna()

expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
# Converting Mckinnon timestemp to match xarray timestemp
expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
# region mckinnon - expanded time series
Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=region)[0]
Prec_reg, datesvar = func_mcK.time_mean_bins(Prec_reg, ex)

matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, Prec_reg.time[0].dt.hour)


# binary time serie when T95 exceeds 1 std
hotdaythreshold = mcKts.mean(dim='time').values + mcKts.std().values
Ev_dates = func_mcK.Ev_timeseries(mcKts, hotdaythreshold)

def plot_oneyr_events(xarray, threshold, test_year):
    testyear = xarray.where(xarray.time.dt.year == test_year).dropna(dim='time', how='any')
    freq = pd.Timedelta(testyear.time.values[1] - testyear.time.values[0])
    plotpaper = xarray.sel(time=pd.DatetimeIndex(start=testyear.time.values[0], 
                                                end=testyear.time.values[-1], 
                                                freq=freq ))
    #plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
    #                                freq=(datesmcK[1] - datesmcK[0])))
    eventdays = plotpaper.where( plotpaper.values > threshold) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    plt.figure()
    plotpaper.plot()
    plt.axhline(y=threshold)
    for days in eventdays.time.values:
        plt.axvline(x=days)
## plotting same figure as in paper
plot_oneyr_events(mcKts, hotdaythreshold, 2012)

# not merging hot days which happen consequtively
matchhotdates = func_mcK.to_datesmcK(Ev_dates, Ev_dates[0].dt.hour, varfullgl.time[0].dt.hour)



def plotting_wrapper(plotarr, foldername, kwrgs=None):
    file_name = os.path.join(ex['fig_path'], foldername,
                 '{} - {} tf{} lags {}.png'.format(
                 ex['name'], plotarr.name, ex['tfreq'], lags))
    if os.path.isdir(os.path.join(ex['fig_path'], foldername)) != True : 
        os.makedirs(os.path.join(ex['fig_path'], foldername))
    title = ('{} extracted features {} \n'
             'T95 McKinnon data - ERA-I SST region {}'.format(
            n_strongest, plotarr.name, region))
    if kwrgs == None:
        kwrgs = dict( {'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'title' : title, 'clevels' : 'notdefault', 'steps':17,
                       'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
    else:
        kwrgs = kwrgs
        kwrgs['title'] = title
    func_mcK.finalfigure(plotarr, file_name, kwrgs)
    
#%% Divide into train and validation step
n_runs = 1
leave_n_years_out = 5
lags = [30,40]  
min_detection = 5
min_events    = 1
mcK_ROCS = []
Sem_ROCS = []
score_per_year = []

all_years = np.arange(ex['startyear'], ex['endyear']+1)
initial_years = all_years.copy()
for n in range(n_runs):
    no_events_occuring = True
    while no_events_occuring == True:
        no_events_occuring = False
        # Divide into random sampled 25 year for train & rest for test
#        n_years_sampled = int((ex['endyear'] - ex['startyear']+1)*0.66)
        
        # leave years out to be tested
        no_dublicates = False
        while no_dublicates == False:
            rand_test_years = np.random.choice(initial_years, leave_n_years_out, replace=False)
            # test duplicates
            no_dublicates = (len(set(rand_test_years)) == leave_n_years_out)
        # Update random years to be selected as test years:
#        initial_years = [yr for yr in initial_years if yr not in random_test_years]
        rand_train_years = [yr for yr in all_years if yr not in rand_test_years]
    
        RV_dates = list(matchdaysmcK.time.dt.year.values)
        full_years  = list(Prec_reg.time.dt.year.values)
        RV_years  = list(mcKts.time.dt.year.values)
        
        RV_dates_train_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_train_years]
        var_train_idx = [i for i in range(len(full_years)) if full_years[i] in rand_train_years]
        RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_train_years]
        
        RV_dates_test_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_test_years]
        var_test_idx = [i for i in range(len(full_years)) if full_years[i] in rand_test_years]
        RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_test_years]
        
        
        dates_train = matchdaysmcK.isel(time=RV_dates_train_idx)
        Prec_train = Prec_reg.isel(time=var_train_idx)
        RV_train = mcKts.isel(time=RV_train_idx)
        
    #    if len(RV_dates_test_idx) 
        dates_test = matchdaysmcK.isel(time=RV_dates_test_idx)
        Prec_test = Prec_reg.isel(time=var_test_idx)
        RV_test = mcKts.isel(time=RV_test_idx)
        
        event_train = func_mcK.Ev_timeseries(RV_train, hotdaythreshold).time
        event_test = func_mcK.Ev_timeseries(RV_test, hotdaythreshold).time
        
        test_year = [yr for yr in list(set(RV_years)) if yr in rand_test_years]
        
        print('test year is {}, with {} events'.format(test_year, len(event_test)))
        no_events_occuring = len(event_test) < min_events
    



    # =============================================================================
    #  Mean over 230 hot days
    # =============================================================================
    Prec_train_mcK = func_mcK.find_region(Prec_train, region='PEPrectangle')[0]
    
    time = Prec_train_mcK.time
    lats = Prec_train_mcK.latitude
    lons = Prec_train_mcK.longitude
    pthresholds = np.linspace(1, 9, 9, dtype=int)
    
    array = np.zeros( (len(lags), len(lats), len(lons)) )
    pattern = xr.DataArray(data=array, coords=[lags, lats, lons], 
                          dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags',
                          attrs={'units':'Kelvin'})
    array = np.zeros( (len(lags), len(dates_train)) )
    pattern_ts = xr.DataArray(data=array, coords=[lags, dates_train], 
                          dims=['lag','time'], name='McK_mean_ts_diff_lags',
                          attrs={'units':'Kelvin'})
    
    array = np.zeros( (len(lags), len(pthresholds)) )
    pattern_p = xr.DataArray(data=array, coords=[lags, pthresholds], 
                          dims=['lag','percentile'], name='McK_mean_p_diff_lags')
    for lag in lags:
        idx = lags.index(lag)
        event_train = func_mcK.Ev_timeseries(RV_train, hotdaythreshold).time
        event_train = func_mcK.to_datesmcK(event_train, event_train.dt.hour[0], Prec_train_mcK.time[0].dt.hour)
        events_train_atlag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train_atlag = dates_train - pd.Timedelta(int(lag), unit='d')
        

        pattern_atlag = Prec_train_mcK.sel(time=events_train_atlag).mean(dim='time')
        pattern[idx] = pattern_atlag 
        ts_3d = Prec_train_mcK.sel(time=dates_train_atlag)
        
        
        crosscorr = func_mcK.cross_correlation_patterns(ts_3d, pattern_atlag)
        crosscorr['time'] = pattern_ts.time
        pattern_ts[idx] = crosscorr
        # Percentile values based on training dataset
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(crosscorr.values, p*10))
        pattern_p[idx] = p_pred
    ds_mcK = xr.Dataset( {'pattern' : pattern, 'ts' : crosscorr, 'perc' : pattern_p} )
    
  
        
    
        
#    mcK_mean.attrs['units'] = 'Kelvin (absolute values)'
#    folder = os.path.join(ex['fig_path'], 'mcKinnon_mean')
#    if os.path.isdir(folder) != True : os.makedirs(folder)
#    fname = '{} - mean composite tf{} lags {} {}.png'.format(ex['name'], ex['tfreq'],
#             lags, region)
#    file_name = os.path.join(folder, fname)
#    title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
#    kwrgs = dict( {'vmin' : -3*mcK_mean.std().values, 'vmax' : 3*mcK_mean.std().values, 
#                   'steps' : 17, 'title' : title, 'clevels' : 'notdefault',
#                   'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
#    func_mcK.finalfigure(mcK_mean, file_name, kwrgs) 
    
    
    
    
# =============================================================================
# Extracting feature to build spatial map
# =============================================================================
    
    n_strongest = 15
    n_std = 1.5      
    ds_Sem = func_mcK.extract_precursor(Prec_train, RV_train, ex,
                                            hotdaythreshold, lags, n_std, n_strongest)

    foldername = 'communities_Marlene'
    kwrgs = dict( {'vmin' : 0, 'vmax' : n_strongest, 
                       'clevels' : 'notdefault', 'steps':n_strongest+1,
                       'map_proj' : map_proj, 'cmap' : plt.cm.Dark2, 'column' : 2} )
#    plotting_wrapper(commun_num, foldername, kwrgs=kwrgs)
#    
#    plotting_wrapper(commun_comp, foldername)    
    
        
  

    # =============================================================================
    # calc ROC scores
    # =============================================================================
    for lag in lags:
        idx = lags.index(lag)
    
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(Prec_test, region='PEPrectangle')[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)

        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg = Prec_test.sel(time=dates_min_lag)        

        crosscorr_mcK = func_mcK.cross_correlation_patterns(var_test_mcK, 
                                                            ds_mcK['pattern'].sel(lag=lag))
        crosscorr_Sem = func_mcK.cross_correlation_patterns(var_test_reg, 
                                                            ds_Sem['pattern'].sel(lag=lag))
        
#        plt.plot(crosscorr_mcK)
#        plt.plot(crosscorr_Sem)
#        
        
        
        # check detection of precursor:
        Prec_threshold_mcK = ds_mcK['perc'].sel(percentile=60 /10).values[0]
        Prec_threshold_Sem = ds_Sem['perc'].sel(percentile=60 /10).values[0]
        
        # =============================================================================
        # Determine events in time series
        # =============================================================================
        # check if there are any detections
        Prec_det_mcK = (func_mcK.Ev_timeseries(crosscorr_mcK, Prec_threshold_mcK).size > min_detection)
        Prec_det_Sem = (func_mcK.Ev_timeseries(crosscorr_Sem, Prec_threshold_Sem).size > min_detection)
        
        # select test event predictand series
        RV_ts_test = RV_test
        # plot the detections
        func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, RV_ts_test, Prec_threshold_Sem, 
                                        Prec_threshold_mcK, hotdaythreshold, test_year[0])


        if Prec_det_mcK == True:
            n_boot = 0
            ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
                                  hotdaythreshold, lag, n_boot, ds_mcK['perc'])
            ROC_std = 2 * np.std([ROC_boot_mcK])
            mcK_ROCS.append(ROC_mcK)
        else:
            print('Not enough predictions detected, neglecting this predictions')
            ROC_mcK = ROC_std = 0.0


        
        if Prec_det_Sem == True:
            ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
                                  hotdaythreshold, lag, n_boot, ds_Sem['perc'])
            ROC_std = 2 * np.std([ROC_boot_Sem])
            Sem_ROCS.append(ROC_Sem)
#                Sem_ROCS.append(commun_comp.sel(lag=lag))
        else:
            print('Not enough predictions detected, neglecting this predictions')
            ROC_Sem = ROC_std = 0.0
                                  
            
        print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
            '\t ±{:.2f} 2*std random events\n\n'.format(region, 
              lag, ROC_mcK, ROC_Sem, ROC_std))
        
        score_per_year.append([test_year, len(event_test), ROC_mcK, ROC_Sem])

#                mcK_ROCS.append(mcK_mean.sel(lag=lag))
        
    print('Mean score of mcK {:.2f} ± {:.2f} 2*std'.format(np.mean(mcK_ROCS),np.std(mcK_ROCS)))
    print('Mean score of Sem {:.2f} ± {:.2f} 2*std\n\n'.format(np.mean(Sem_ROCS),np.std(Sem_ROCS)))
            

#%%        
for lag in lags:
    idx = lags.index(lag)

    # select antecedant SST pattern to summer days:
    dates_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    var_full_mcK = func_mcK.find_region(Prec_reg, region='PEPrectangle')[0]
    full_timeserie_regmck = var_full_mcK.sel(time=dates_min_lag)
    full_timeserie = Prec_reg.sel(time=dates_min_lag)
    
    # select test event predictand series
    RV_ts_test = mcKts
    crosscorr_mcK = func_mcK.cross_correlation_patterns(full_timeserie_regmck, 
                                                ds_mcK['pattern'].sel(lag=lag))
    crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, 
                                                ds_Sem['pattern'].sel(lag=lag))
    n_boot = 5
    ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
                                  hotdaythreshold, lag, n_boot, ds_mcK['perc'])
    ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
                                  hotdaythreshold, lag, n_boot, ds_Sem['perc'])
    
#    ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
#                                  hotdaythreshold, lag, n_boot, 'default')
#    ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
#                                  hotdaythreshold, lag, n_boot, 'default')
    
    ROC_std = 2 * np.std([ROC_boot_mcK, ROC_boot_Sem])
    print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
        '\t ±{:.2f} 2*std random events'.format(region, 
          lag, ROC_mcK, ROC_Sem, ROC_std))
#test_year = list(np.arange(2000, 2005))
#func_mcK.plot_events_validation(crosscorr_Sem, RV_ts_test, Prec_threshold, 
#                                hotdaythreshold, test_year)
      
foldername = 'communities_Marlene'

kwrgs = dict( {'vmin' : 0, 'vmax' : n_strongest, 
                   'clevels' : 'notdefault', 'steps':n_strongest+1,
                   'map_proj' : map_proj, 'cmap' : plt.cm.Dark2, 'column' : 2} )
#plotting_wrapper(commun_num, foldername, kwrgs=kwrgs)

plotting_wrapper(ds_Sem['pattern'], foldername)   

