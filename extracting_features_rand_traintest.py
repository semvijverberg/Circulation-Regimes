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
     'figpathbase'     :       "/Users/semvijverberg/surfdrive/McKinRepl/T95_sst_NOAA"}
     )

if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])

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
ex['name'] = 'sst_NOAA'
#filename = '{}_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
filename = '{}_1982-2017_2jan_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
# full globe - full time series
varfullgl = func_mcK.import_array(filename, ex)

# filter out outliers of sst
if ex['name']=='sst':
    varfullgl.where(varfullgl.values < 3.5*varfullgl.std().values)



# take means over bins over tfreq days
ex['tfreq'] = 1
mcKts, datesmcK = func_mcK.time_mean_bins(mcKts, ex)
RV_ts = mcKts

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
ex['hotdaythreshold'] = hotdaythreshold
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



def plotting_wrapper(plotarr, filename, kwrgs=None):
#    if os.path.isdir(os.path.join(ex['fig_path'], foldername)) != True : 
#        os.makedirs(os.path.join(ex['fig_path'], foldername))
    file_name = os.path.join(ex['fig_path'], filename)
    if kwrgs == None:
        kwrgs = dict( {'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'title' : title, 'clevels' : 'notdefault', 'steps':17,
                       'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
    else:
        kwrgs = kwrgs
        kwrgs['title'] = plotarr.attrs['title']
    func_mcK.finalfigure(plotarr, file_name, kwrgs)
    
#%% Divide into train and validation step
n_runs = 20
leave_n_years_out = 8
lags = [30,40]  
min_detection = 5
min_events    = 1
mcK_ROCS = []
Sem_ROCS = []
score_per_year = []

leave_n_out = True


for n in range(n_runs):
    
    if leave_n_out == True:
        train, test, test_years = func_mcK.rand_traintest(RV_ts, Prec_reg, 
                                          leave_n_years_out, min_events, ex)
        
        foldername = 'leave_{}_out_{}_{}_tf{}_{}'.format(leave_n_years_out,
                            ex['startyear'], ex['endyear'], ex['tfreq'],
                            lags)
        ex['exp_folder'] = os.path.join(ex['figpathbase'],foldername)
    elif leave_n_out == False:
        train = dict( { 'Prec'  : Prec_reg,
                        'RV'    : RV_ts,
                        'events': func_mcK.Ev_timeseries(RV_ts, hotdaythreshold)})
        test = train.copy()

        foldername = 'hindcast_{}_{}_tf{}_{}'.format(ex['startyear'],
                             ex['endyear'], ex['tfreq'], lags)
        ex['exp_folder'] = os.path.join(ex['figpathbase'],foldername)
        test_years = 'all_years'
    


# =============================================================================
#  Mean over 230 hot days
# =============================================================================

    ds_mcK = func_mcK.mcKmean(train, hotdaythreshold, ex, lags)  
    
    
    
# =============================================================================
# Extracting feature to build spatial map
# =============================================================================
    
    n_strongest = 15
    n_std = 1.5      
    ds_Sem = func_mcK.extract_precursor(train, ex, hotdaythreshold,
                                            lags, n_std, n_strongest)    
        
   

# =============================================================================
# calc ROC scores
# =============================================================================
    for lag in lags:
        idx = lags.index(lag)
        dates_test = func_mcK.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(test['Prec'], region='PEPrectangle')[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)

        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg = test['Prec'].sel(time=dates_min_lag)        

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
        RV_ts_test = test['RV']
        # plot the detections
        func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, RV_ts_test, Prec_threshold_Sem, 
                                        Prec_threshold_mcK, hotdaythreshold, test_years[0])


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
        
        score_per_year.append([test_years, len(test['events']), ROC_mcK, ROC_Sem])
    
    
# ============================= ================================================
#   # Saving figures in exp_folder
# =============================================================================
    # ROC scores of this run
    ROCsmcK = [round(mcK_ROCS[-(i+1)],2) for i in range(len(lags))][::-1]
    ROCsSem = [round(Sem_ROCS[-(i+1)],2) for i in range(len(lags))][::-1]
    exp_folder = '{}_{:.2f}_{:.2f}'.format(test_years, np.mean(ROCsmcK),
                 np.mean(ROCsSem))
    ex['fig_path'] = os.path.join(ex['exp_folder'], exp_folder)
    if os.path.isdir(ex['fig_path']) == False: os.makedirs(ex['fig_path'])
    # Plotting mcK
    ds_mcK['pattern'].attrs['units'] = 'Kelvin (absolute values)'
    title = 'PEP\ntest years : {}\n{}'.format(test_years, ROCsmcK)
    ds_mcK['pattern'].attrs['title'] = title
    plotting_wrapper(ds_mcK['pattern'], 'PEP') 
    
    # Plotting CPD
    ds_Sem['pattern'].attrs['units'] = 'Weighted Kelvin (absolute values)'
    file_name = 'CPD'
    title = 'CPD\ntest years : {}\n{}'.format(test_years, ROCsSem)
    ds_Sem['pattern'].attrs['title'] = title
    plotting_wrapper(ds_Sem['pattern'], file_name) 
    
    

#                mcK_ROCS.append(ds_mcK['pattern'].sel(lag=lag))
        
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

  

