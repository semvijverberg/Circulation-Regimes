#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:47:10 2018

@author: semvijverberg
"""
#%%
import os
import generate_varimax
from geo_field_jakob import GeoField

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
path_pp  = os.path.join(base_path, 'input_pp/')

load_filename = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
folder_name = path_pp
varname = 'sst'

geo_object = generate_varimax.load_data(load_filename, folder_name, varname)

data = sst.data()
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
