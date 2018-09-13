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

def clustering_temporal(data, method, linkage, n_clusters, cls, region, RV_period):
    import sklearn.cluster as cluster
    import xarray as xr
    import numpy as np
    import os
    from sklearn.preprocessing import StandardScaler
    input = data.isel(time=RV_period)
#    input = data.sel(month=month).drop('time_date')
#    input = np.squeeze(input.isel(data.get_axis_num(cls.name))).drop(cls.name)

    def clustering_plotting(cluster_method, input):        
        region_values, region_coords = plotting.find_region(input,region=region)
        print(region_values.mean())
        X_vectors = np.reshape(region_values.values, (len(input.time), np.size(region_values.isel(time=0))))
        # Make sure field has mean of 0 and unit variance (std = 1)
        X = StandardScaler().fit_transform(X_vectors)
        out_clus = cluster_method.fit(X)
        labels = out_clus.labels_       
        output = region_values.copy()
        labels_clusters = xr.DataArray(labels, [input.coords['time'][:]], name='time')
        labels_dates = xr.DataArray(input.time.values, [input.coords['time'][:]], name='time')
        output['cluster'] = labels_clusters
        output['time_date'] = labels_dates
        output = output.set_index(time=['cluster','time_date'])
        output.name = method + '_' + cls.name
        Nocluster = {}
        group_clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True)
#        functions.quicksave_ncdf(output, cls, path=folder, name=output.name)

        for n in range(0,n_clusters):
            folder = os.path.join(cls.base_path,'Clustering_temporal/', name_method)
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            perc_of_cluster = str(100*float(output.sel(cluster=n).time_date.size)/float(input.time.size))[:2]+'%'
            Nocluster['cluster_{}'.format(n)] = perc_of_cluster
            group_clusters.name = 'cluster_{}_{}_{}'.format(n, perc_of_cluster, str(input[cls.name].values[0]))
            plotting.xarray_plot(group_clusters.sel(cluster=n), path=folder, saving=True)
        group_clusters.name = name_method.replace('/','_') + '_' + cls.name
        plotting.PlateCarree_timesteps(group_clusters.rename({'cluster':'time'}), cls, path=folder, region=region, saving=True)

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
        if np.size(linkage) == 1:
            link = linkage
            name_method = 'AgglomerativeClustering' + '_' + link
            print(name_method)
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            output = clustering_plotting(cluster_method, input)
        else:
            for link in linkage:
                name_method = 'AgglomerativeClustering' + '_' + link
                print(name_method)
                cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
                output = clustering_plotting(cluster_method, input)
#    folder = os.path.join(cls.base_path, 'Clustering_temporal', method)
#    quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
    return output

def clustering_spatial(data, ex, n_clusters, region, cls):
    import numpy as np
    import xarray as xr
    import os
    import sklearn.cluster as cluster
    input = np.squeeze(data)
    def clustering_plotting(ex, input):
        region_values, region_coords = plotting.find_region(input, region=region)
        
        path_lsm = '/Users/semvijverberg/surfdrive/Data_ERAint/input_raw/grids/landseamask_2.5deg.nc'
        ncdf = xr.open_dataset(path_lsm, decode_cf=True, decode_coords=True, decode_times=False)
        lsm_array = np.squeeze(ncdf.to_array(path_lsm))
        lsm_region, bounds = plotting.find_region(lsm_array, region=region)
        def station_time_shape(array, timesteps):
            X_vec = np.reshape( array.values, (timesteps, array.longitude.size*array.latitude.size) )
            X_station_time = np.swapaxes(X_vec, 1,0 )
            return X_station_time
        lsm_station_time = station_time_shape(lsm_region, 1)
        lsm_station_time = np.array(np.tile(lsm_station_time, (1,data.time.size)), dtype=bool)
    
        indic_station_time = station_time_shape(indic, indic.time.size)
        indic_st_land = indic_station_time[lsm_station_time]
        n_land_stations = lsm_station_time[:,0][lsm_station_time[:,0]==1].size
        indic_st_land.reshape( (n_land_stations, indic.time.size)  ), lsm_station_time
        output = region_values[0].copy()
        X_vec = np.reshape( region_values.values, (region_values.time.size, region_values[0].size) )
        X_station_time = np.swapaxes(X_vec, 1,0 )
#        X = StandardScaler().fit_transform(X_station_time)
        out_clus = cluster_method.fit(X_station_time)
        labels = out_clus.labels_
        lonlat_cluster = np.reshape(labels, region_values[0].shape)
        output.values = lonlat_cluster
        output.name = name_method + '_' + cls.name
        
        folder = os.path.join(cls.base_path,'Clustering_spatial/')
        if os.path.isdir(folder) != True : os.makedirs(folder)
#        savepath = os.path.join(folder, output.name)
        plotting.xarray_plot(output, path=folder, saving=True)
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
            name_method = 'AgglomerativeClustering' + '_' + link
            print(name_method)
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            output = clustering_plotting(cluster_method, input)
        else:
            for link in ex['linkage']:
                name_method = 'AgglomerativeClustering' + '_' + link
                print(name_method)
                cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
                output = clustering_plotting(cluster_method, input)
                
#    folder = os.path.join(cls.base_path, 'Clustering_spatial', method)
#    quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters) 
    return output


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


#%%
# clim = xr.DataArray(input_array_3Dims, dims=('time', 'latitude', 'longitude'), coords=[months, latitudes,longitudes])