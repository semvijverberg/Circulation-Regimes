import os
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Circulation-Regimes')
script_dir = os.getcwd()
import what_variable
#import retrieve_ERA_i
import functions
import numpy as np
import plotting
Variable = what_variable.Variable
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
xarray_plot = plotting.xarray_plot
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region


#%% assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=3, endmonth=9, grid='2.5/2.5', stream='oper', units='K')






#%%
# =============================================================================
# clustering tests
# =============================================================================

cls = temperature
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
method = methods[1] ; n_clusters = 4; month=6
data = anom
#%% clustering temporal
output = functions.clustering_temporal(data, method, n_clusters, temperature, region='U.S.', month=6)

#%% clustering spatial
output = functions.clustering_spatial(data, method, n_clusters, temperature)


#%% save to github
import os
import subprocess
runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
subprocess.call(runfile)


#%%
# =============================================================================
# Simple example of cdo commands within python by calling bash script
# =============================================================================
infile = os.path.join(temperature.base_path, 'input_raw', temperature.filename)
outfile = os.path.join(temperature.base_path, 'input_pp', 'output')
args = ['cdo eofspatial,3 {} {}'.format(infile, outfile+'1.nc', outfile+'2.nc')]
functions.kornshell_with_input(args)











#%%
# =============================================================================
# Depricated tests with EOF
# =============================================================================

# input data EOF
region_values, region_coords = find_region(anom, region='EU')
from eofs.examples import example_data_path
# Read geopotential height data using the xarray module. The file contains
# December-February averages of geopotential height at 500 hPa for the
# European/Atlantic domain (80W-40E, 20-90N).
filename = example_data_path('hgt_djf.nc')
z_djf = xr.open_dataset(filename)['z']
# Compute anomalies by removing the time-mean.
z_djf = z_djf - z_djf.mean(dim='time')



eof_output = functions.EOF(data, neofs=4)
eof_output = functions.EOF(region_values, neofs=4)
PlateCarree_timesteps(eof_output.rename( {'mode':'time'}), temperature, cbar_mode='individual')
plotting.xarray_plot(eof_output)

exit()











