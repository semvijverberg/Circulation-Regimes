import os
import what_variable
import retrieve_ERA_i
import computations
import numpy as np
import plotting
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
import_array = computations.import_array
calc_anomaly = computations.calc_anomaly
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


def clustering(data, method, n_clusters, cls):
    input = np.squeeze(data.isel(data.get_axis_num(cls.name)))
    import sklearn.cluster as cluster
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    region_values, region_coords = find_region(input.isel(time=0), region='EA')
    output = np.repeat(region_values.expand_dims('time', axis=0).copy(), len(data.time), axis=0)
    # output = region_values.values.copy()
    output['time'] = data['time']
    algorithm = cluster.__dict__[method]
    print algorithm
    for t in data['time'].values:
        # t = data['time'].values[0]
        region_values, region_coords = find_region(input.sel(time=t),region='EU')
        X_vectors = np.reshape(region_values.values, (np.size(region_values),  1))
        # X_vectors = np.reshape(region_values, (shape[0]*shape[1], 2))
        X = StandardScaler().fit_transform(X_vectors)
        if method == 'KMeans':
            out_clus = algorithm(n_clusters).fit(X)
        if method == 'AgglomerativeClustering':
            out_clus = algorithm(linkage='complete', n_clusters=n_clusters).fit(X)
        elif method == 'DBSCAN':
            out_clus = algorithm(eps=0.05, min_samples=3).fit(test)
        elif method == 'hierarchical':
            pass
        # centroids = out_clus.cluster_centers_
        labels = out_clus.labels_
        # print np.max(labels)
        lonlat_cluster = np.reshape(labels, region_values.shape)
        idx = int(np.where(output['time'] ==  t)[0])
        print idx
        output[idx,:,:] = lonlat_cluster

    folder = os.path.join(cls.base_path,'Clustering/' + method)
    if os.path.isdir(folder):
        pass
    else:
        os.makedirs(folder)
    namefile = cls.name + '_' + method + '_' + '{}-{}.nc'.format(int(data['time.year'].min()),
                                                                 int(data['time.year'].max()))
    output.to_netcdf(folder + '/'+ namefile.replace(' ', '_'))
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
    return output



methods = ['KMeans', 'DBSCAN','AgglomerativeClustering','hierarchical']
method = methods[0] ; n_clusters = 5
data = anom.isel(time=np.array(np.where(anom['time.month']==6)).reshape(int(anom['time.year'].max()-anom['time.year'].min()+1)))
out_cluster = clustering(data, method, n_clusters, temperature)
PlateCarree_timesteps(out_cluster, temperature)
PlateCarree_timesteps(anom, temperature)



data = clim.isel(time=np.array(np.where(anom['time.year']==1979)).reshape(3))
PlateCarree_timesteps(data, temperature)




# PlateCarree_timesteps(clim, temperature, type='norm')
single_timestep_1 = clim.sel(time=clim['time'])[0,0]
single_timestep_2 = anom.sel(time=anom['time'])[0,0]
PlateCarree(single_timestep_1)
PlateCarree(single_timestep_2, valueformat='norm')
exit()


single_timestep_2 = data[0,0]
PlateCarree(single_timestep_2, valueformat='norm', region='EU')
PlateCarree(plot[0])

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









