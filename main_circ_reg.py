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

#%%
# assign instance
sst = Variable(name='SST', dataset='ERA-i', var_cf_code='34.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=1, endmonth=12, grid='2.5/2.5', stream='oper', units='K')

##%%
## assign instance
#temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
#                       startyear=1979, endyear=2017, startmonth=1, endmonth=12, grid='2.5/2.5', stream='moda', units='K')
#
#%%
#%% assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='oper', units='K')



#%% load in array
temperature.filename = '2_metre_temperature_1979-2017_6_8_dt-20days_2.5x2.5.nc'
marray, temperature = import_array(temperature, path='pp')




#%% clustering test
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
infile = os.path.join(temperature.base_path, 'input', temperature.filename)
out_path = '/Users/semvijverberg/surfdrive/Data/Tigramite/'
outfile = os.path.join(out_path, 'input', temperature.filename[:-3]+'_5d_mean.nc')



args = ['cdo timselmean,5 {} {}'.format(infile, outfile)]
functions.kornshell_with_input(args)




#%%





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











