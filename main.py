import what_variable
import retrieve_ERA_i
import computations
import numpy as np
import plotting
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
calc_anomaly = computations.calc_anomaly
PlateCarree = plotting.PlateCarree
PlateCarree_timesteps = plotting.PlateCarree_timesteps
LamberConformal = plotting.LamberConformal
find_region = plotting.find_region


# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# Download variable
retrieve_ERA_i_field(temperature)
clim, anom, std = calc_anomaly(cls=temperature)
cls = temperature

# PlateCarree(clim-273, temperature)
data = anom.isel(time=np.array(np.where(anom['time.year']==1979)).reshape(3))
PlateCarree_timesteps(data, temperature, valueformat='norm', region='EA')


plottable = data.sel(time=data['time'][0])






def clustering(data, method, n_clusters, cls):
    input = np.squeeze(data.isel(data.get_axis_num(cls.name)))
    import sklearn.cluster as cluster
    import seaborn as sns
    algorithm = cluster.__dict__[method]
    output = input.copy()
    output[:,:,:] =1.
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    # for t in data['time'].values:
    t = data['time'].values[0]
    region_values = find_region(input.sel(time=t),region='EA')
    output.where()
    test = np.reshape(region_values.values, (np.size(region_values),  1))
    func = algorithm(n_clusters)
    out_clus = func.fit(test)
    centroids = out_clus.cluster_centers_
    labels = out_clus.labels_
    lonlat_cluster = np.reshape(labels, region_values.shape)
    region_values.values = lonlat_cluster


    lons, lats = np.meshgrid(output.longitude, output.latitude)
    test = output.sel(time=t, longitude=region_values.longitude, latitude=region_values.latitude) #lonlat_cluster
    # PlateCarree(output, temperature)
    plt.pcolormesh(lons, lats, output.sel(time=t))
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

methods = ['KMeans', 'DBSCAN']
method = methods[0] ; n_clusters = 4
out_cluster = clustering(anom, methods[0], n_clusters, temperature)

data = anom.isel(time=np.array(np.where(anom['time.year']==1979)).reshape(3))
PlateCarree(data, temperature, valueformat='norm')


exit()










