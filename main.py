import os
import what_variable
import retrieve_ERA_i
import functions
import numpy as np
import plotting
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
import_array = functions.import_array
calc_anomaly = functions.calc_anomaly
PlateCarree_timesteps = plotting.PlateCarree_timesteps
PlateCarree = plotting.PlateCarree
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region


# assign instance
temperature = Variable(name='2_metre_temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth', units='K')
# Download variable
retrieve_ERA_i_field(temperature)
marray, temperature = import_array(temperature, decode_cf=True, decode_coords=True)

clim, anom, std = calc_anomaly(marray=marray, cls=temperature)

#%%

def clustering_temporal(data, method, n_clusters, cls):
    input = np.squeeze(data.isel(data.get_axis_num(cls.name)))
    import sklearn.cluster as cluster
    import xarray as xr
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    def clustering_plotting(cluster_method, input):        
        region_values, region_coords = find_region(input,region='EU')
        print region_values.mean()
        output = region_values.copy()
        X_vectors = np.reshape(region_values.values, (len(input.time), np.size(region_values.isel(time=0))))
        # Make sure field has mean of 0 and unit variance (std = 1)
        X = StandardScaler().fit_transform(X_vectors)
        out_clus = cluster_method.fit(X)
        labels = out_clus.labels_
        labels_clusters = xr.DataArray(labels, [input.coords['time'][:]], name='time')
        labels_dates = xr.DataArray(input.time.values, [input.coords['time'][:]], name='time')
        output['cluster'] = labels_clusters
        output['time_dates'] = labels_dates
        output = output.set_index(time=['cluster','time_dates'])
        output.name = method + '_' + data.name
#        functions.quicksave_ncdf(output, cls, path=folder, name=output.name)

        for n in range(0,n_clusters):
            folder = os.path.join(cls.base_path,'Clustering_temporal/', name_method)
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            group_clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True)
            group_clusters.name = 'cluster_{}_{}'.format(n, input['2_metre_temperature'].values)
            plotting.xarray_plot(group_clusters.sel(cluster=n), path=folder, saving=True)
#            folder = os.path.join(cls.base_path,'Clustering_temporal/', name_method, str(n))
#            print folder
#            if os.path.isdir(folder):
#                pass
#            else:
#                os.makedirs(folder)
#            idx = np.where(output['cluster'] ==  n)[0]
#            dates_in_cluster = output['time'].isel(time=idx).time_dates.values
#            for t in dates_in_cluster:
#                output.name = 'cluster_{}_{}'.format(n, str(t)[:7])
#                plotting.xarray_plot(output.sel(time_dates=t), path=folder, saving=True)
        return output
  
    algorithm = cluster.__dict__[method]
    if method == 'KMeans':
        cluster_method = algorithm(n_clusters)
        name_method = method
        output = clustering_plotting(cluster_method, input)
    if method == 'AgglomerativeClustering':
        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
        #                   'average' is standard.
        linkage = ['ward','complete', 'average']
        linkage = ['ward']
        for link in linkage:
            name_method = os.path.join(method, link)
            print name_method
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            output = clustering_plotting(cluster_method, input)
    folder = os.path.join(cls.base_path, 'Clustering_temporal', method)
    functions.quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
   
    return output

#%%
cls = temperature
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
method = methods[1] ; n_clusters = 4
data = anom.isel(time=np.array(np.where(anom['time.month']==6)).reshape(int(anom['time.year'].max()-anom['time.year'].min()+1)))

#%%
output = clustering_temporal(data, method, n_clusters, temperature)
plottable = output.groupby('cluster').mean(dim='time', keep_attrs=True) 
PlateCarree_timesteps(plottable.rename( {'cluster':'time'} ), temperature, path='default', saving=True)
#%%

#%%
output = functions.clustering_spatial(data, method, n_clusters, temperature)


# small changes blabla


#%%
# PlateCarree timesteps different output then xarray_plot..
import os
import subprocess
cwd = os.getcwd()
runfile = os.path.join(cwd, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
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



# palette = sns.color_palette('deep', np.unique(labels).max() + 1)
# colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
# plt.figure()
# for i in range(n_clusters):
#     # select only data observations with cluster label == i
#     ds = test[np.where(labels == i)]
#     # plot the data observations
#     # plt.plot(np.arange(0,np.size(ds)), ds[:], 'o')
#     plt.scatter(np.arange(0, np.size(ds)), ds[:], c=colors)
#     # plot the centroids
#     lines = plt.plot(find_nearest(ds, centroids[i]),centroids[i], 'kx')
#     # make the centroid x's bigger
#     plt.setp(lines, ms=15.0)
#     plt.setp(lines, mew=2.0)
# plt.show()

# plt.colorbar()









