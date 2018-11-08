# import numpy as np
# from netCDF4 import Dataset
# import os
# import xarray as xr
# import argparse
# import sys
# import os
# import IPython
# %run retrieve_ERA_i_field.py
import plotting

def import_array(cls, path='pp'):
    import os
    import xarray as xr
    from netCDF4 import num2date
    import pandas as pd
    if path == 'pp':
        file_path = os.path.join(cls.path_pp, cls.filename_pp)
    else:
        file_path = os.path.join(cls.path_raw, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name.replace(' ', '_')}))
    marray.name = cls.name
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates_np = pd.to_datetime(dates)
    print(('temporal frequency \'dt\' is: \n{}'.format(dates_np[1]- dates_np[0])))
    marray['time'] = dates_np
    cls.dates_np = dates_np
    
    return marray, cls

def kornshell_with_input(args):
    '''some kornshell with input '''
    import os
    import subprocess
    cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = os.path.join(cwd,'bash_scripts', "bash_script.sh")
    # example syntax
    # arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
    # arg_selbox = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)
    
    # append all commands into basch script: $1, $2 etc...
    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo starting bash script\n")
        for No_args in range(len(bash_and_args)):
            if No_args != 0:
                file.write("${}\n".format(No_args))     
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) 
    out = p.communicate()[0]
    print(out.decode())




def normalize(marray, with_mean=True, with_std=True):
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    output = marray.copy()
    vector_marray_to_norm = np.reshape(marray.values, (len(marray.time), np.size(marray.isel(time=0))))
    vectorn_marray_norm = StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(vector_marray_to_norm)
    output.values = np.reshape(vectorn_marray_norm, marray.shape)
    return output
    
def calc_anomaly(marray, cls, q = 0.95):
    import xarray as xr
    import numpy as np
    print(("calc_anomaly called for {}".format(cls.name, marray.shape)))
    clim = marray.groupby('time.month').mean('time', keep_attrs=True)
    clim.name = 'clim_' + marray.name
    anom = marray.groupby('time.month') - clim
    anom['time_multi'] = anom['time']
    anom['time_date'] = anom['time']
    anom = anom.set_index(time_multi=['time_date','month'])
    anom.attrs = marray.attrs
#    substract = lambda x, y: (x - y)
#    anom = xr.apply_ufunc(substract, marray, np.tile(clim,(1,(cls.endyear+1-cls.startyear),1,1)), keep_attrs=True)
    anom.name = 'anom_' + marray.name
    std = anom.groupby('time.month').reduce(np.percentile, dim='time', keep_attrs=True, q=q)
#    std = anom.groupby('time.month').reduce(np.percentile, dim='time', keep_attrs=True, q=q)
    std.name = 'std_' + marray.name
    return clim, anom, std

def EOF(data, neofs=1, center=False, weights=None):
    import numpy as np
    from eofs.xarray import Eof
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data.coords['latitude'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(np.squeeze(data), center=False)
    eof_output = solver.eofsAsCovariance(neofs=4)
    eof_output.attrs['units'] = 'mode'
    return eof_output

def quicksave_ncdf(data, cls, path, name):
    import os
    if 'path' in locals():
        pass
    else:
        path = '/Users/semvijverberg/Downloads'
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if data.name != '':
        name = cls.name.replace(' ', '_')
    if 'name' in locals():
        print('input name is: '.format(name))
        name = name + '_' + today + '.nc'
    else:
        name = 'netcdf_' + today + '.nc'
    data.to_netcdf(os.path.join(path, name))
    print(('{} to path {}'.format(name, path)))

def clustering_spatial(data, ex, n_clusters, region):
    import numpy as np
    import xarray as xr
    import os
    import sklearn.cluster as cluster
    input = data.squeeze()
    def clustering_plotting(cluster_method, input, n_clusters):
        region_values = input.where(input.mask == True)
        output = region_values[0].copy()
        def station_time_shape(array, timesteps):
            X_vec = np.reshape( array.values, (timesteps, array.longitude.size*array.latitude.size) )
            X_station_time = np.swapaxes(X_vec, 1,0 )
            return X_station_time
        # reshape array to [station, time] (to create dendogram timeseries)
        
        mask_station_time = station_time_shape(region_values.mask, 1)
        mask_station_time = np.array(np.tile(mask_station_time, (1,data.time.size)), dtype=int)
        output_lonlat = np.array(mask_station_time[:,0].copy(), dtype=int)
        indic_station_time = station_time_shape(region_values, region_values.time.size)
        # only keep the land gridcells
        indic_st_land = indic_station_time[mask_station_time == 1]
        n_land_stations = output_lonlat[output_lonlat==1].size
        indic_st_land = indic_st_land.reshape( (n_land_stations, data.time.size)  )

#        X = StandardScaler().fit_transform(X_station_time)
        out_clus = cluster_method.fit(indic_st_land)
        labels = out_clus.labels_
        # plug in the land gridcell with the labels
        land_gcs = np.argwhere(output_lonlat == 1)[:,0]
        output_lonlat[land_gcs] = labels
        output_lonlat[land_gcs] = output_lonlat[land_gcs] + 1
        output_lonlat = np.reshape(np.array(output_lonlat), region_values[0].shape)
        output.values = output_lonlat
#        output.name = cls.name + '_' + str(n_clusters) + '_' + name_method
        output.name = ex['name'] + '_' + str(n_clusters) + '_' + name_method
        
        folder = os.path.join(ex['base_path'],'Clustering_spatial/', 
                              '{}deg_tfreq{}'.format(ex['grid_res'],ex['tfreq']))
        if os.path.isdir(folder) != True : os.makedirs(folder)
#        savepath = os.path.join(folder, output.name)
        plotting.xarray_mask_plot(output, path=folder, saving=True)
        return output   
    
    algorithm = cluster.__dict__[ex['clusmethod']]
    if ex['clusmethod'] == 'KMeans':
        cluster_method = algorithm(n_clusters)
        name_method = ex['clusmethod']
        output = clustering_plotting(cluster_method, input, n_clusters)
    if ex['clusmethod'] == 'AgglomerativeClustering':
        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
        #                   'average' is standard.
        if np.size(ex['linkage']) == 1:
            link = ex['linkage']
            name_method = 'AgglomerativeClustering' + '_' + link
            print(name_method)
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters, 
                                           affinity=ex['distmetric'])
            output = clustering_plotting(cluster_method, input, n_clusters)
        else:
            for link in ex['linkage']:
                name_method = 'AgglomerativeClustering' + '_' + link
                print(name_method)
                cluster_method = algorithm(linkage=link, n_clusters=n_clusters, 
                                           affinity=ex['distmetric'])
                output = clustering_plotting(cluster_method, input, n_clusters)
                
#    folder = os.path.join(cls.base_path, 'Clustering_spatial', method)
#    quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters) 
    return output

def clustering_temporal(data, ex, n_clusters, cls, tfreq, region):
    import sklearn.cluster as cluster
    import xarray as xr
    import numpy as np
    import os
    input = data.squeeze()

    def clustering_plotting(cluster_method, input):    
        region_values, region_coords = plotting.find_region(input, region=region)
#        region_values = input.where(input.mask == True)

        def time_station_shape(array, timesteps):
            X_vec = np.reshape( array.values, (timesteps, array.longitude.size*array.latitude.size) )
            return X_vec

#        mask_time_station = time_station_shape(region_values.mask, 1)
#        mask_time_station = np.array(np.tile(mask_time_station, (data.time.size,1)), dtype=int)
#        output_lonlat = np.array(mask_time_station[0,:].copy(), dtype=int)
        # input shape for clustering
        X_time_station = time_station_shape(region_values, region_values.time.size)
        # only keep the land gridcells
#        X_st_land = X_time_station[mask_time_station == 1]
#        n_land_stations = output_lonlat[output_lonlat==1].size
#        X_st_land = X_st_land.reshape( (data.time.size, n_land_stations )  )

#        out_clus = cluster_method.fit(X_st_land)
#        labels = out_clus.labels_

        out_clus = cluster_method.fit(X_time_station)
        labels = out_clus.labels_       
        output = region_values.copy()
        labels_clusters = xr.DataArray(labels, [input.coords['time'][:]], name='time')
        labels_dates = xr.DataArray(input.time.values, [input.coords['time'][:]], name='time')
        output['cluster'] = labels_clusters
        output['time_date'] = labels_dates
        output = output.set_index(time=['cluster','time_date'])
        output.name = '{}_tf{}_{}'.format(data.name, tfreq, ex['clusmethod'][:4])
        Nocluster = {}
        group_clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True)
#        functions.quicksave_ncdf(output, cls, path=folder, name=output.name)

        group_clusters = group_clusters.drop('mask')
        labels = []
        for n in range(0,n_clusters):
            folder = os.path.join(cls.base_path,'Clustering_temporal/',
                              '{}deg_tfreq{}'.format(ex['grid_res'],tfreq),
                              name_method +'_'+ region)
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            perc_of_cluster = str(100*float(output.sel(cluster=n).time_date.size)/float(input.time.size))[:2]+'%'
            Nocluster['cluster_{}'.format(n)] = perc_of_cluster
            cluster_n = group_clusters.sel(cluster=n)
            cluster_n.name = '{}_cl_{}of{}_{}_tf{}_{}'.format(data.name, n, n_clusters, 
                              perc_of_cluster, tfreq, name_method)
            labels.append('cluster {}, {}'.format(n, perc_of_cluster))
            plotting.xarray_plot(cluster_n, path=folder, saving=True)
        # updating name (name of saved figure)
        group_clusters.name = '{}_{}_clusters_tf{}_{}'.format(data.name, n_clusters, 
                              tfreq, name_method)
        # updating labels (name of title in figure)
        group_clusters['cluster'] = xr.DataArray(labels, dims='cluster')
        plotting.PlateCarree_timesteps(group_clusters.rename({'cluster':'time'}), cls, 
                                       path=folder, region=region, saving=True)

        return output
  
    algorithm = cluster.__dict__[ex['clusmethod']]
    if ex['clusmethod'] == 'KMeans':
        cluster_method = algorithm(n_clusters)
        name_method = ex['clusmethod']
        output = clustering_plotting(cluster_method, input)
    if ex['clusmethod'] == 'AgglomerativeClustering':
        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
        #                   'average' is standard.
        if np.size(ex['linkage']) == 1:
            link = ex['linkage']
            name_method = 'Aggl' + '_' + link[:3]
            print(name_method)
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            output = clustering_plotting(cluster_method, input)
        else:
            for link in ex['linkage']:
                name_method = 'Aggl' + '_' + link[:3]
                print(name_method)
                cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
                output = clustering_plotting(cluster_method, input)
#    folder = os.path.join(cls.base_path, 'Clustering_temporal', method)
#    quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
    return output


def Mckin_timeseries(marray, RV):
    import numpy as np
    perc = marray.reduce(np.percentile, dim='time', keep_attrs=True, q=95)
    rep_perc = np.tile(perc, (marray.time.size,1,1))
    indic = np.squeeze(marray.where(np.squeeze(marray.values) > rep_perc))
    indic.values = np.nan_to_num(indic)
    indic.values[indic.values > 0 ] = 1
    return indic
#%%
# clim = xr.DataArray(input_array_3Dims, dims=('time', 'latitude', 'longitude'), coords=[months, latitudes,longitudes])