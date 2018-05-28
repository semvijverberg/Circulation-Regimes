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
temperature = Variable(name='2 metre temperature', dataset='ERA-i', var_cf_code='167.128', levtype='sfc', lvllist=0,
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth', units='K')
# Download variable
retrieve_ERA_i_field(temperature)
marray, temperature = import_array(temperature, decode_cf=True, decode_coords=True)
clim, anom, std = calc_anomaly(marray=marray, cls=temperature)

#%%

def clustering(data, method, n_clusters, cls):
    input = np.squeeze(data.isel(data.get_axis_num(cls.name)))
    import sklearn.cluster as cluster
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    def clustering_plotting(cluster_method, input):
        region_values, region_coords = find_region(input.isel(time=0), region='EU')
        output = np.repeat(region_values.expand_dims('time', axis=0).copy(), len(data.time), axis=0)
        output['time'] = data['time']
        for t in data['time'].values:
            # t = data['time'].values[0]
            region_values, region_coords = find_region(input.sel(time=t),region='EU')
            X_vectors = np.reshape(region_values.values, (np.size(region_values),  1))
            # X_vectors = np.reshape(region_values, (shape[0]*shape[1], 2))
            X = StandardScaler().fit_transform(X_vectors)
            out_clus = cluster_method.fit(X)
            # centroids = out_clus.cluster_centers_
            labels = out_clus.labels_
            # print np.max(labels)
            lonlat_cluster = np.reshape(labels, region_values.shape)
            idx = int(np.where(output['time'] ==  t)[0])
            output[idx,:,:] = lonlat_cluster

        output.name = method + '_' + data.name
        for t in data['time'].values[:3]:
            folder = os.path.join(cls.base_path,'Clustering/' + method, str(t)[:7])
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            plotting.xarray_plot(output.sel(time=t), path=folder, saving=True)
            region_values, region_coords = find_region(input.sel(time=t), region='EU')
            plotting.xarray_plot(region_values, path=folder, saving=True)
        return output

        
    algorithm = cluster.__dict__[method]
    print algorithm
    if method == 'KMeans':
        cluster_method = algorithm(n_clusters)
        output = clustering_plotting(cluster_method, input)
    if method == 'AgglomerativeClustering':
        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
        #                   'average' is standard.
        linkage = ['ward','complete', 'average']
        for link in linkage:
            print link
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            clustering_plotting(cluster_method, input)
    folder = os.path.join(cls.base_path, 'Clustering', method)
    functions.quicksave_ncdf(region_values, cls, path=folder, name=region_values.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
   
    return output

#%%
cls = temperature
methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
method = methods[1] ; n_clusters = 4
data = anom.isel(time=np.array(np.where(anom['time.month']==6)).reshape(int(anom['time.year'].max()-anom['time.year'].min()+1)))
#%%
output = clustering(data, method, n_clusters, temperature)

#%%

data = clim.isel(time=np.array(np.where(anom['time.year']==1979)).reshape(3))
PlateCarree_timesteps(data, temperature)



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

eof_output = functions.EOF(z_djf, neofs=4)
eof_output = functions.EOF(region_values, neofs=4)
PlateCarree_timesteps(eof_output.rename( {'mode':'time'}), temperature, cbar_mode='individual')
functions.(region_values)
plotting.xarray_plot(eof_output)



#%%
import os
import subprocess
cwd = os.getcwd()
runfile = os.path.join(cwd, 'saving_repository_to_Github.sh')
subprocess.call(runfile)
#%%


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









