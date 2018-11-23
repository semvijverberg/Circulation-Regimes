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
lags = [30, 40, 50, 60]
min_detection = 5
min_events    = 1
mcK_ROCS = []
Sem_ROCS = []
for n in range(30):
    no_events_occuring = True
    while no_events_occuring == True:
        no_events_occuring = False
        n_years_sampled = int((ex['endyear'] - ex['startyear']+1)*0.66)
        n_years_sampled = int((ex['endyear'] - ex['startyear']+1)) -1
        random_years = np.arange(ex['startyear'], ex['endyear']+1)
        random_years = np.random.choice(random_years, n_years_sampled, replace=False)
    
        RV_dates = list(matchdaysmcK.time.dt.year.values)
        full_years  = list(Prec_reg.time.dt.year.values)
        RV_years  = list(mcKts.time.dt.year.values)
        
        RV_dates_train_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in random_years]
        var_train_idx = [i for i in range(len(full_years)) if full_years[i] in random_years]
        RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in random_years]
        
        RV_dates_test_idx = [i for i in range(len(RV_dates)) if RV_dates[i] not in random_years]
        var_test_idx = [i for i in range(len(full_years)) if full_years[i] not in random_years]
        RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] not in random_years]
        
        
        dates_train = matchdaysmcK.isel(time=RV_dates_train_idx)
        Prec_train = Prec_reg.isel(time=var_train_idx)
        RV_train = mcKts.isel(time=RV_train_idx)
        
    #    if len(RV_dates_test_idx) 
        dates_test = matchdaysmcK.isel(time=RV_dates_test_idx)
        Prec_test = Prec_reg.isel(time=var_test_idx)
        RV_test = mcKts.isel(time=RV_test_idx)
        
        event_train = func_mcK.Ev_timeseries(RV_train, hotdaythreshold).time
        event_test = func_mcK.Ev_timeseries(RV_test, hotdaythreshold).time
        
        test_year = [yr for yr in list(set(RV_years)) if yr not in random_years][0]
        
        print('test year is {}, with {} events'.format(test_year, len(event_test)))
        no_events_occuring = len(event_test) < min_events
    
        


    # Mean over 230 hot days
    
    var_train_mcK = func_mcK.find_region(Prec_train, region='PEPrectangle')[0]
    array = np.zeros( (len(lags),var_train_mcK.latitude.size, var_train_mcK.longitude.size) )
    mcK_mean = xr.DataArray(data=array, coords=[lags, var_train_mcK.latitude, var_train_mcK.longitude], 
                          dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
    for lag in lags:
        idx = lags.index(lag)
        event_train = func_mcK.Ev_timeseries(RV_train, hotdaythreshold).time
        event_train = func_mcK.to_datesmcK(event_train, event_train.dt.hour[0], var_train_mcK.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
        
        varhotdays = var_train_mcK.sel(time=events_min_lag).mean(dim='time')
        mcK_mean[idx] = varhotdays 
        
        std_lag =  var_train_mcK.sel(time=dates_train_min_lag).std(dim='time')    
        signaltonoise = abs(varhotdays/std_lag)
        weights = signaltonoise / np.mean(signaltonoise)
        
    
        
    mcK_mean.attrs['units'] = 'Kelvin (absolute values)'
    folder = os.path.join(ex['fig_path'], 'mcKinnon_mean')
    if os.path.isdir(folder) != True : os.makedirs(folder)
    fname = '{} - mean composite tf{} lags {} {}.png'.format(ex['name'], ex['tfreq'],
             lags, region)
    file_name = os.path.join(folder, fname)
    
    #title = 'mean composite - absolute values \nT95 McKinnon data - ERA-I SST'
    #kwrgs = dict( {'vmin' : -3*mcK_mean.std().values, 'vmax' : 3*mcK_mean.std().values, 
    #               'steps' : 17, 'title' : title, 'clevels' : 'notdefault',
    #               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
    #func_mcK.finalfigure(mcK_mean, file_name, kwrgs) 
    
    
    # Extracting feature to build spatial map
    
        
        
    n_strongest = 20
    n_std = 1.5      
    commun_comp, commun_num = func_mcK.extract_precursor(Prec_train, RV_train, ex,
                                            hotdaythreshold, lags, n_std, n_strongest)
#    foldername = 'communities_Marlene'
#    
#    kwrgs = dict( {'vmin' : 0, 'vmax' : n_strongest, 
#                       'clevels' : 'notdefault', 'steps':n_strongest+1,
#                       'map_proj' : map_proj, 'cmap' : plt.cm.Dark2, 'column' : 2} )
#    plotting_wrapper(commun_num, foldername, kwrgs=kwrgs)
#    
#    plotting_wrapper(commun_comp, foldername)    
    
        

    

    
    for lag in lags:
        idx = lags.index(lag)
    
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(Prec_test, region='PEPrectangle')[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
        full_timeserie = Prec_test.sel(time=dates_min_lag)
        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        
        # select test event predictand series
        RV_ts_test = RV_test
        crosscorr_mcK = func_mcK.cross_correlation_patterns(var_test_mcK, mcK_mean.sel(lag=lag))
        crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, commun_comp.sel(lag=lag))
        
        # check detection of precursor:
        Prec_threshold = (crosscorr_mcK.mean() + np.std(crosscorr_mcK)).values
        Prec_det_mcK = (func_mcK.Ev_timeseries(crosscorr_mcK, Prec_threshold).size > min_detection)
        
        
        Prec_threshold = (crosscorr_Sem.mean() + np.std(crosscorr_Sem)).values
        Prec_det_Sem = (func_mcK.Ev_timeseries(crosscorr_Sem, Prec_threshold).size > min_detection)
        
        func_mcK.plot_events_validation(crosscorr_Sem, RV_test, Prec_threshold, 
                                        hotdaythreshold, test_year)

        
        if Prec_det_mcK == True:
            n_boot = 0
            ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
                                  hotdaythreshold, lag, n_boot)
            ROC_std = 2 * np.std([ROC_boot_mcK])
            mcK_ROCS.append(ROC_mcK)
        else:
            print('Not enough predictions detected, neglecting this predictions')
            ROC_mcK = 0.0


        
        if Prec_det_Sem == True:
            ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
                                  hotdaythreshold, lag, n_boot)
            ROC_std = 2 * np.std([ROC_boot_Sem])
            Sem_ROCS.append(ROC_Sem)
#                Sem_ROCS.append(commun_comp.sel(lag=lag))
        else:
            print('Not enough predictions detected, neglecting this predictions')
            ROC_Sem = 0.0
                                  
            
        print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
            '\t ±{:.2f} 2*std random events\n\n'.format(region, 
              lag, ROC_mcK, ROC_Sem, ROC_std))
        

#                mcK_ROCS.append(mcK_mean.sel(lag=lag))
        
    print('Mean score of mcK {:.2f} ± {:.2f} 2*std'.format(np.mean(mcK_ROCS),np.std(mcK_ROCS)))
    print('Mean score of Sem {:.2f} ± {:.2f} 2*std\n\n'.format(np.mean(Sem_ROCS),np.std(Sem_ROCS)))
            

#%%        
#for lag in lags:
#    idx = lags.index(lag)
#
#    # select antecedant SST pattern to summer days:
#    dates_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
#    var_test_mcK = func_mcK.find_region(Prec_reg, region='PEPrectangle')[0]
#    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
#    full_timeserie = Prec_reg.sel(time=dates_min_lag)
#    
#    # select test event predictand series
#    RV_ts_test = mcKts
#    crosscorr_mcK = func_mcK.cross_correlation_patterns(full_timeserie_regmck, 
#                                                        mcK_mean.sel(lag=lag))
#    crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, 
#                                                        commun_comp.sel(lag=lag))
#    
#    ROC_mcK, ROC_boot_mcK = ROC_score(predictions=crosscorr_mcK, 
#                                      observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
#    ROC_Sem, ROC_boot_Sem = ROC_score(predictions=crosscorr_Sem, 
#                                      observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
#    ROC_std = 2 * np.std([ROC_boot_mcK, ROC_boot_Sem])
#    print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
#        '\t ±{:.2f} 2*std random events'.format(region, 
#          lag, ROC_mcK, ROC_Sem, ROC_std))
#          

