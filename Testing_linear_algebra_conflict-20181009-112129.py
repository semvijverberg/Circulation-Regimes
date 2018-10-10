#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:20:26 2018

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import func_mcK
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

import matplotlib.pyplot as plt

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
map_proj = ccrs.PlateCarree(central_longitude=240)  

ex = dict(
     {'grid_res'     :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'sstartdate'   :       '1982-06-24',
     'senddate'     :       '1982-08-22',
     'fig_path'     :       "/Users/semvijverberg/surfdrive/McKinRepl/T95_ERA-I"}
     )

if os.path.isdir(ex['fig_path']) == False: os.makedirs(ex['fig_path'])
# Load in mckinnon Time series
T95name = 'PEP-T95TimeSeries.txt'
mcKtsfull, datesmcK = func_mcK.read_T95(T95name, ex)
datesmcK = func_mcK.make_datestr(datesmcK, ex)
# Load in external ncdf
sst_ERAname = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
#t2m_ERAname = 't2mmax_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
# full globe - full time series
varfullgl = func_mcK.import_array(sst_ERAname, ex)
matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, varfullgl.time[0].dt.hour)
mcKts = mcKtsfull.sel(time=datesmcK)
# only hot days
var = varfullgl.sel(time=matchdaysmcK)
# region mckinnon - full time series
varfull = func_mcK.find_region(varfullgl, region='Mckinnonplot')[0]

hotts = mcKts.where( mcKts.std().values>mcKts.values) 
plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
                                freq=(datesmcK[1] - datesmcK[0])))
plotpaper.plot()
std = var.std(dim='time')
# not merging hot days which happen consequtively
hotdates = hotts.dropna(how='all', dim='time').time
matchhotdates = func_mcK.to_datesmcK(hotdates, hotdates[0].dt.hour, varfullgl.time[0].dt.hour)


#%% Perform eof from package for Region Mckinnon
eof_output, solver = func_mcK.EOF(varfull, neofs=12)
lag=50
title = 'EOFs on full variability'
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'EOFs on full variability region mcK - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)


#%% Verify covariance matrix using F.T dot F and using SVD.

# =============================================================================
# Remember, here taking SVD of matrix_sst (not the same as eof analysis, that is: SVD(F.T*F))
# =============================================================================
dimspace = varfull[0].size
dimtime  = varfull.time.size


matrix_sst = np.reshape( varfull.values , (varfull.time.size, varfull[0].size) )
matrix_sst = np.nan_to_num(matrix_sst)
dimtime = matrix_sst.shape[0]
dimspace = matrix_sst.shape[1]

U, s, V = np.linalg.svd(matrix_sst, full_matrices=True)
# classic way
R = np.dot( np.transpose(matrix_sst), matrix_sst )
# SVD way
Rsvd = np.dot(np.dot(V.T,np.diag(s**2)),V)
i=0
plt.figure()
plt.imshow(np.reshape( R[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( Rsvd[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()

#%% check if RC = RLambda  with SVD
U, s, V = np.linalg.svd(R, full_matrices=True)

Csvd = V.T
Lambda = np.dot( np.dot(Csvd.T, R), Csvd )
I = np.dot( Csvd.T, Csvd )
#    RC = np.dot( R[:maxspace,:maxspace], Ceof[:maxspace,:maxspace] )
RC = np.dot( R, Csvd )
CLambda = np.dot(  Csvd, Lambda )

#plot
i = i
plt.figure()
plt.imshow(np.reshape( RC.T[i,:], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( CLambda.T[i], (varfull.latitude.size, varfull.longitude.size) ) )
print('Above figure should be the same, (and noisy).')

#%% Lambda values of package eof versus SVD
Lambda_eof = np.diag(solver.eigenvalues().values)
plt.figure()
plt.plot( Lambda.diagonal()[:] / np.mean(Lambda.diagonal()) )
plt.figure()
plt.plot( Lambda_eof.diagonal()[:] )




#%% check if C == V
eofs = solver.eofs().values
maxspace = eofs.shape[0]
#eof_output, solver = func_mcK.EOF(varfull)
i = i
plt.figure()
plt.imshow(eofs[i] ); plt.colorbar()
plt.figure()
plt.imshow(np.reshape( V[i], (varfull.latitude.size, varfull.longitude.size) ) ); plt.colorbar()

print('C and V not the same due to filling in zeros into the matrix_sst for SVD')

#%% Check if A = FC and F = AC.T


C = Csvd

A = np.dot( matrix_sst, C )
ATA = np.dot( A.T, A) 
F = np.dot( A, C.T)
ATA2 = np.dot( np.dot(F,C).T, np.dot(F,C) )

plt.figure()
plt.imshow(np.reshape( F[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( matrix_sst[i], (varfull.latitude.size, varfull.longitude.size) ) )

#%% find expansion coefficients with eofs from package:
n_eofs = 10
PCi = np.zeros( (dimtime, n_eofs))
C = Csvd
for i in range(n_eofs):
    PCi[:,i] = np.dot( matrix_sst, C[:,i] )
plt.plot(PCi[:,0])
plt.plot(A[:,0])
print('matric A (from matrix_sst[:,:] dot C[:,:]), is the same as matrix_sst[:,:] dot C[:,i]' )
plt.plot(A[:,0] - PCi[:,0])

#%% find expansion coefficients with eofs from SVD:   
Ceof = np.reshape( eofs, (maxspace, dimspace) ) 
#C = Ceof
C = Csvd
n_eofs = 10
PCi_eofs = np.zeros( (dimtime, n_eofs))
PCi_svd = np.zeros( (dimtime, n_eofs))
for i in range(n_eofs):
    PCi_eofs[:,i] = np.dot( matrix_sst[:,:maxspace], C[:maxspace,i] )
    PCi_svd[:,i] = np.dot( matrix_sst[:,:maxspace], V[:maxspace,i] )
plt.plot(PCi_eofs[:,0])
plt.plot(PCi_svd[:,0])
print('matric A (from matrix_sst[:,:] dot C[:,:]), is the same as matrix_sst[:,:] dot C[:,i]' )
plt.plot(PCi_eofs[:,0] - PCi_svd[:,0]) 
#%% What is U
plt.plot(np.dot( U[:,0]))



#%% Check C = V
# calcuate covariance matrix normal way (Ft*F)
R = np.dot( np.transpose(matrix_sst), matrix_sst )
# Perform SVD to get obtain matrix V
Lamdasvd = np.diag(s**2)
Lambdacheck = np.dot( np.diag(s).T, np.diag(s) )
# R = Vt * st * s * V
Rsvd = np.dot( np.dot( np.dot(V.T, np.diag(s).T ), np.diag(s) ), V )
Rsvd = np.dot( np.dot( V.T, Lambdacheck ), V )
i=1
plt.figure()
plt.imshow(np.reshape( R[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( Rsvd[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()

plt.figure()
plt.imshow(np.reshape( C[i]*-1, (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( V[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()

#%% Rebuild matrix_sst from svd 
s_nxm = np.zeros( (dimtime, dimspace) )
for i in range(dimspace):
    s_nxm[i,i] = s[i]
F = np.dot( np.dot(U, s_nxm), V )

i=100
plt.figure()
plt.imshow(np.reshape( F[i], (varfull.latitude.size, varfull.longitude.size) ) )
plt.figure()
plt.imshow(np.reshape( matrix_sst[i], (varfull.latitude.size, varfull.longitude.size) ) )

#%% Create Prewhitening operator
eofs_kept = 100
s_nxm_inv = np.zeros( (dimtime, dimspace) )
for i in range(eofs_kept):
    s_nxm_inv[i,i] = s[i]**-1
PW = (dimtime)**(0.5) *  np.dot(U, s_nxm_inv)
Trans = np.dot( PW.T, matrix_sst )
F = np.dot( np.dot(U, s_nxm_inv), V )

for i in range(4):
    plt.figure()
    plt.imshow(np.reshape( Trans[i], (varfull.latitude.size, varfull.longitude.size) ) )
    plt.figure()
    plt.imshow(np.reshape( V[i], (varfull.latitude.size, varfull.longitude.size) ) )

#%% Rebuilt time series with inverse of eigenvalues

for i in range(4):
    plt.figure()
    plt.imshow(np.reshape( F[i], (varfull.latitude.size, varfull.longitude.size) ) )
    plt.figure()
    plt.imshow(np.reshape( matrix_sst[i], (varfull.latitude.size, varfull.longitude.size) ) )
    
#%% Rebuilt time series with giving equal weight to dominant eigenvalues
eofs_kept = 100
Ucopy = np.dot( matrix_sst, matrix_sst.T )
s_nxm_1 = np.zeros( (dimtime, dimspace) )
for i in range(eofs_kept):
    s_nxm_1[i,i] = np.mean(s[:eofs_kept])
F = np.dot( np.dot(U, s_nxm_1), V )

#%%
F_lonlat = np.reshape(F, (dimtime, varfull.latitude.size, varfull.longitude.size))
Transfull = xr.DataArray(F_lonlat, coords=[varfull.time, varfull.latitude, varfull.longitude], 
                         dims=['time', 'latitude', 'longitude'])

eof_output, solver = func_mcK.EOF(Transfull, neofs=4)
title = 'EOFs on full variability'
kwrgs = dict( {'vmin' : -1*(1*np.std(F)), 'vmax' : 1*np.std(F), 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'EOFs on full variability global - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)

lag = 50
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
Transhotdays = Transfull.sel(time=dates_min_lag)

eof_output, solver = func_mcK.EOF(Transhotdays, neofs=4)
title = 'EOFs on full variability'
kwrgs = dict( {'vmin' : -1*(1*np.std(F)), 'vmax' : 1*np.std(F), 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'EOFs on full variability global - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)

eof_output, solver = func_mcK.EOF(varfull, neofs=4)
title = 'EOFs on full variability'
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'EOFs on full variability global - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)

#for i in range(2):
#    plt.figure()
#    plt.imshow(np.reshape( F[i], (varfull.latitude.size, varfull.longitude.size) ) )
#    plt.figure()
#    plt.imshow(np.reshape( matrix_sst[i], (varfull.latitude.size, varfull.longitude.size) ) )

#%% Recreate composite:
F_lonlat = np.reshape(F, (dimtime, varfull.latitude.size, varfull.longitude.size))
Transfull = xr.DataArray(F_lonlat, coords=[varfull.time, varfull.latitude, varfull.longitude], 
                         dims=['time', 'latitude', 'longitude'])
lag = 50
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
Transhotdays = Transfull.sel(time=dates_min_lag)


lags = [0, 5, 10, 15, 20, 30, 40, 50]
varfull = func_mcK.find_region(varfull, region='Mckinnonplot')[0]
array = np.zeros( (len(lags),varfull.latitude.size, varfull.longitude.size) )
xrdata = xr.DataArray(data=array, coords=[lags, varfull.latitude, varfull.longitude], 
                      dims=['lag','latitude','longitude'], name='Trans_hotdays')
         
for lag in lags:
    idx = lags.index(lag)
    dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
    Transhotdays = Transfull.sel(time=dates_min_lag).mean(dim='time')
#    Transhotdays = Transfull.sel(time=dates_min_lag).mean(dim='time')
    xrdata[idx] = Transhotdays


xrdata.attrs['units'] = 'Kelvin (normalized by std)'
file_name = '~/Downloads/finalfiguremckinnon'
title = 'normalized by total std \nT95 McKinnon data - ERA-I SST'
kwrgs = dict( {'vmin' : -1*(2*np.std(F)), 'vmax' : 2*np.std(F), 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'normalized by std values lag{}-{}.png'.format(lags[0], lags[-1]))
func_mcK.finalfigure(xrdata, file_name, kwrgs) 


#%% calculate std of first 20 dominant eofs



#%% Depricated:
#%% Global patterns
eof_output, solver = func_mcK.EOF(varfullgl, neofs=12)
title = 'EOFs on full variability'
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             'EOFs on full variability global - lag {}.png'.format(lag))
plotting = eof_output
func_mcK.finalfigure(plotting, file_name, kwrgs)

#%% EOFS on composite hot days
lag = 50
dates_min_lag = matchhotdates - pd.Timedelta(int(lag), unit='d')
varhotdays = varfull.sel(time=dates_min_lag)
eof_output, solver = func_mcK.EOF(varhotdays, neofs=6)
title = 'EOFs on composite (hot days, n={})'.format(matchhotdates.time.size) 
kwrgs = dict( {'vmin' : -1, 'vmax' : 1, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
file_name = os.path.join(ex['fig_path'], 
             title + ' - lag {}.png'.format(lag))
plotting = eof_output.rename( {'mode':'lag'} )
func_mcK.finalfigure(plotting, file_name, kwrgs)