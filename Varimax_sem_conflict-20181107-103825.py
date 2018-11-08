#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:47:10 2018

@author: semvijverberg
"""
#%%
import os
import numpy as np
import generate_varimax
from geo_field_jakob import GeoField


base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
path_pp  = os.path.join(base_path, 'input_pp/')

load_filename = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
folder_name = path_pp
varname = 'sst'

geo_object = generate_varimax.load_data(load_filename, folder_name, varname)

geo_data = geo_object.data()
flattened = np.reshape(np.array(geo_data), 
                       (geo_data.shape[0], np.prod(geo_data.shape[1:])))
nonmask_flat = np.where(np.array(flattened)[0] != 0.)[0]
nonmasked = flattened[:,nonmask_flat]


# convert to nonmasked array
data = nonmasked
truncate_by = 'max_comps'
max_comps=60
fraction_explained_variance=0.9
verbosity=0

var_standard = generate_varimax.get_varimax_loadings_standard(data=data,
                    truncate_by = truncate_by, 
                    max_comps=max_comps,
                    fraction_explained_variance=fraction_explained_variance,
                    verbosity=verbosity,
                    )

# Plug in nonmasked values into the original lonlat with mask
npcopy = np.array(flattened[:max_comps])
npcopy[:,nonmask_flat] = np.swapaxes(var_standard['weights'],1,0)
# convert to xarray

xr.DataArray(npcopy, coords=[np.arange(0,max_comps), geo_object.lats.data, 
                             geo_object.lons.data], 
                            dims=['modes', 'latitude','longitude'], 
                            name='rot_pca_standard')
                                         


mcK_mean = xr.DataArray(data=array, coords=[lags, varsumreg.latitude, varsumreg.longitude], 
                      dims=['lag','latitude','longitude'], name='McK_Composite_diff_lags')

