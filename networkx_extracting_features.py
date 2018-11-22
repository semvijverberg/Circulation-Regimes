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

# filter out outliers of sst
if ex['name']=='sst':
    varfullgl.where(varfullgl.values < 3.5*varfullgl.std().values)


#%%
# take means over bins over tfreq days
ex['tfreq'] = 2
mcKts, datesmcK = func_mcK.time_mean_bins(mcKts, ex)

def oneyr(datetime):
    return datetime.where(datetime.year==datetime.year[300]).dropna()

datetime = datesmcK
expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
# Converting Mckinnon timestemp to match xarray timestemp
expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
# region mckinnon - expanded time series
Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=region)[0]
Prec_reg, datesvar = func_mcK.time_mean_bins(Prec_reg, ex)


matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)

varsumreg = Prec_reg.sel(time=matchdaysmcK)


# binary time serie when T95 exceeds 1 std
def Ev_timeseries(xarray, threshold):   
    Ev_ts = xarray.where( xarray.values > hotdaythreshold) 
    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    return Ev_dates

hotdaythreshold = mcKts.mean(dim='time').values + mcKts.std().values
Ev_dates = Ev_timeseries(mcKts, hotdaythreshold)


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

# not merging hot days which happen consequtively
matchhotdates = func_mcK.to_datesmcK(Ev_dates, Ev_dates[0].dt.hour, varfullgl.time[0].dt.hour)


#%% Divide into train and validation step
mcK_ROCS = []
Sem_ROCS = []
for n in range(10):
    n_years_sampled = int((ex['endyear'] - ex['startyear'])*0.66)
    random_years = np.arange(ex['startyear'], ex['endyear'])
    random_years = np.random.choice(random_years, n_years_sampled, replace=False)
    #random_years = np.random.randint(ex['startyear'],ex['endyear'],size=n_years_sampled)
    #random_years = np.repeat(random_years, oneyr(datesvar).size)
    RV_dates = list(matchdaysmcK.time.dt.year.values)
    years_full  = list(Prec_reg.time.dt.year.values)
    years_RV  = list(mcKts.time.dt.year.values)
    
    RV_dates_train_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in random_years]
    var_train_idx = [i for i in range(len(years_full)) if years_full[i] in random_years]
    RV_train_idx = [i for i in range(len(years_RV)) if years_RV[i] in random_years]
    
    RV_dates_test_idx = [i for i in range(len(RV_dates)) if RV_dates[i] not in random_years]
    var_test_idx = [i for i in range(len(years_full)) if years_full[i] not in random_years]
    RV_test_idx = [i for i in range(len(years_RV)) if years_RV[i] not in random_years]
    
    
    dates_train = matchdaysmcK.isel(time=RV_dates_train_idx)
    Prec_train = Prec_reg.isel(time=var_train_idx)
    RV_train = mcKts.isel(time=RV_train_idx)
    
    dates_test = matchdaysmcK.isel(time=RV_dates_test_idx)
    Prec_test = Prec_reg.isel(time=var_test_idx)
    RV_test = mcKts.isel(time=RV_test_idx)
    
    #%% Mean over 230 hot days
    lags = [30, 40, 50, 60]
    #lags = [0, 5]
    #lags = [10, 50]
    var_train_mcK = func_mcK.find_region(Prec_train, region='PEPrectangle')[0]
    array = np.zeros( (len(lags),var_train_mcK.latitude.size, var_train_mcK.longitude.size) )
    mcK_mean = xr.DataArray(data=array, coords=[lags, var_train_mcK.latitude, var_train_mcK.longitude], 
                          dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')
    for lag in lags:
        idx = lags.index(lag)
        event_train = Ev_timeseries(RV_train, hotdaythreshold).time
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
    #%%
    
    import numpy
    sys.path.append('./../RGCPD/RGCPD/')
    import functions_RGCPD as rgcpd
    
    
    array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
    commun_comp = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                          dims=['lag','latitude','longitude'], name='communities_composite', 
                          attrs={'units':'Kelvin'})
    array = np.zeros( (len(lags),varsumreg.latitude.size, varsumreg.longitude.size) )
    commun_num = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                          dims=['lag','latitude','longitude'], name='communities_numbered', 
                          attrs={'units':'regions'})
    n_strongest = 20
    n_std = 1.5
    
    
    Actors_ts_GPH = [[] for i in lags] #!
    
    x = 0
    for lag in lags:
        i = lags.index(lag)
        idx = lags.index(lag)
        event_train = Ev_timeseries(RV_train, hotdaythreshold).time
        event_train = func_mcK.to_datesmcK(event_train, event_train.dt.hour[0], var_train_mcK.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
    
        event_idx = [list(dates_train.values).index(E) for E in event_train.values]
        event_binary = np.zeros(dates_train.size)    
        event_binary[event_idx] = 1
        
        full = Prec_train.sel(time=dates_train_min_lag)
        sample = Prec_train.sel(time=events_min_lag)
        var = ex['name']
        lat_grid = full.latitude.values
        lon_grid = full.longitude.values
        actbox = np.reshape(full.values, (full.time.size, lat_grid.shape[0]*lon_grid.shape[0]))
        xarray = sample
        
        def extract_commun(xarray, event_binary, n_std, n_strongest):
            x=0
        #    T, pval, mask_sig = func_mcK.Welchs_t_test(sample, full, alpha=0.01)
        #    threshold = np.reshape( mask_sig, (mask_sig.size) )
        #    mask_threshold = threshold 
        #    plt.figure()
        #    plt.imshow(mask_sig)
            mean = xarray.mean(dim='time')
            nparray = np.reshape(np.nan_to_num(mean.values), mean.size)
            
            threshold = n_std * np.std(nparray)
            mask_threshold = abs(nparray) < ( threshold )
            
            Corr_Coeff = np.ma.MaskedArray(nparray, mask=mask_threshold)
            lat_grid = mean.latitude.values
            lon_grid = mean.longitude.values
    #        if Corr_Coeff.ndim == 1:
    #            lag_steps = 1
    #            n_rows = 1
    #        else:
    #            lag_steps = Corr_Coeff.shape[1]
    #            n_rows = Corr_Coeff.shape[1]
            
            	
            la_gph = lat_grid.shape[0]
            lo_gph = lon_grid.shape[0]
            lons_gph, lats_gph = numpy.meshgrid(lon_grid, lat_grid)
            
            cos_box_gph = numpy.cos(numpy.deg2rad(lats_gph))
            cos_box_gph_array = np.repeat(cos_box_gph[None,:], actbox.shape[0], 0)
            cos_box_gph_array = np.reshape(cos_box_gph_array, (cos_box_gph_array.shape[0], -1))
        
            
            Regions_lag_i = func_mcK.define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid)
            
            if Regions_lag_i.max()> 0:
                n_regions_lag_i = int(Regions_lag_i.max())
    #            print(('{} regions detected for lag {}, variable {}'.format(n_regions_lag_i, lag,var)))
    #            x_reg = numpy.max(Regions_lag_i)
    			
    #            levels = numpy.arange(x, x + x_reg +1)+.5
                A_r = numpy.reshape(Regions_lag_i, (la_gph, lo_gph))
                A_r + x
            
    
    
            x = A_r.max() 
    
            # this array will be the time series for each region
            if n_regions_lag_i < n_strongest:
                n_strongest = n_regions_lag_i
            ts_regions_lag_i = np.zeros((actbox.shape[0], n_strongest))
    				
            for j in range(n_strongest):
                B = np.zeros(Regions_lag_i.shape)
                B[Regions_lag_i == j+1] = 1	
                ts_regions_lag_i[:,j] = np.mean(actbox[:, B == 1] * cos_box_gph_array[:, B == 1], axis =1)
            
            
            # creating arrays of output
            npmap = np.ma.reshape(Regions_lag_i, (len(lat_grid), len(lon_grid)))
            mask_strongest = (npmap!=0.) & (npmap <= n_strongest)
            npmap[mask_strongest==False] = 0
            xrnpmap = mean.copy()
            xrnpmap.values = npmap
            
            mask = (('latitude', 'longitude'), mask_strongest)
            mean.coords['mask'] = mask
            xrnpmap.coords['mask'] = mask
            xrnpmap = xrnpmap.where(xrnpmap.mask==True)
            # normal mean of extracted regions
            norm_mean = mean.where(mean.mask==True)
            
            coeff_features = func_mcK.train_weights_LogReg(ts_regions_lag_i, event_binary)
            features = np.arange(xrnpmap.min(), xrnpmap.max() + 1 ) 
            weights = npmap.copy()
            for f in features:
                mask_single_feature = (npmap==f)
                weight = int(round(coeff_features[int(f-1)], 2) * 100)
                np.place(arr=weights, mask=mask_single_feature, vals=weight)
    #            weights = weights/weights.max()
            
    
            weighted_mean = norm_mean * abs(weights)
            
            
            
            return weighted_mean, xrnpmap, ts_regions_lag_i
        commun_mean, commun_numbered, ts_regions_lag_i = extract_commun(
                                        sample, event_binary, n_std, n_strongest)  
        commun_comp[idx] = commun_mean
        commun_num[idx]  = commun_numbered
        
        
    #    print(commun_mean.max(), commun_numbered.max())
    
        
    
    #    print(commun_comp[idx].max().values, commun_num[idx].max().values)
    
        
    
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
        func_mcK.finalfigure(plotarr, file_name, kwrgs) 
    
    
    #foldername = 'communities_Marlene'
    ##plotting_wrapper(commun_comp, foldername, kwrgs=None)
    #
    #kwrgs = dict( {'vmin' : 0, 'vmax' : n_strongest, 
    #                   'title' : title, 'clevels' : 'notdefault', 'steps':n_strongest+1,
    #                   'map_proj' : map_proj, 'cmap' : plt.cm.Dark2, 'column' : 2} )
    #plotting_wrapper(commun_num, foldername, kwrgs=kwrgs)
    #
    #plotting_wrapper(commun_comp, foldername)
    
    #%%
    
    for lag in lags:
        idx = lags.index(lag)
    
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(Prec_test, region='PEPrectangle')[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
        full_timeserie = Prec_test.sel(time=dates_min_lag)
        
        # select test event predictand series
        RV_ts_test = RV_test
        crosscorr_mcK = func_mcK.cross_correlation_patterns(var_test_mcK, mcK_mean.sel(lag=lag))
        crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, commun_comp.sel(lag=lag))
        
        ROC_mcK, ROC_boot_mcK = ROC_score(predictions=crosscorr_mcK, observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
        ROC_Sem, ROC_boot_Sem = ROC_score(predictions=crosscorr_Sem, observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
        ROC_std = 2 * np.std([ROC_boot_mcK, ROC_boot_Sem])
        print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
            '\t ±{:.2f} 2*std random events'.format(region, 
              lag, ROC_mcK, ROC_Sem, ROC_std))
        if ROC_mcK-0.5 > ROC_std:
            mcK_ROCS.append(mcK_mean.sel(lag=lag))
        if ROC_Sem-0.5 > ROC_std:
            Sem_ROCS.append(commun_comp.sel(lag=lag))
            

#%%        
for lag in lags:
    idx = lags.index(lag)

    # select antecedant SST pattern to summer days:
    dates_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
    var_test_mcK = func_mcK.find_region(Prec_reg, region='PEPrectangle')[0]
    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    full_timeserie = Prec_reg.sel(time=dates_min_lag)
    
    # select test event predictand series
    RV_ts_test = mcKts
    crosscorr_mcK = func_mcK.cross_correlation_patterns(full_timeserie_regmck, mcK_mean.sel(lag=lag))
    crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, commun_comp.sel(lag=lag))
    
    ROC_mcK, ROC_boot_mcK = ROC_score(predictions=crosscorr_mcK, observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
    ROC_Sem, ROC_boot_Sem = ROC_score(predictions=crosscorr_Sem, observed=RV_ts_test, threshold_event=hotdaythreshold, lag=lag)
    ROC_std = 2 * np.std([ROC_boot_mcK, ROC_boot_Sem])
    print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
        '\t ±{:.2f} 2*std random events'.format(region, 
          lag, ROC_mcK, ROC_Sem, ROC_std))
          

