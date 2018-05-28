# import numpy as np
# from netCDF4 import Dataset
# import os
# import xarray as xr
# import argparse
# import sys
# import os
# import IPython
# %run retrieve_ERA_i_field.py




def import_array(cls, decode_cf=True, decode_coords=True):
    import xarray as xr
    import numpy as np
    import os
    # load in file
    file_path = os.path.join(cls.base_path, 'input', cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))
    marray.attrs['units'] = cls.units
    for dims in marray.dims:
        if dims == 'lon':
            marray = marray.rename(({'lon': 'longitude'}))
        if dims == 'lat':
            marray = marray.rename(({'lat': 'latitude'}))
        else:
            pass
    marray.attrs['units'] = cls.units
    marray.attrs['dataset'] = cls.dataset
    print("import array {}".format(cls.name))
    if 'units' in marray.time.attrs:
        if marray.time.attrs['units'] == 'months since 1861-01-01':
            print('original timeformat: months since 1861-01-01')
            cls = month_since_to_datetime(marray, cls)
        if marray.time.attrs['units'] == 'hours since 1900-01-01 00:00:0.0':
            marray['time'] = cls.dates_np
            print("Taken numpy dates from what variable function, check time units")
    return marray, cls

def month_since_to_datetime(marray, cls):
    import datetime as datetime
    datelist = [datetime.date(year=cls.startyear, month=1, day=1)]
    year = cls.startyear
    for steps in marray['time'].values[1:]+1:
        step = int(steps % 12)
        datelist.append(datetime.date(year=year, month=step, day=1))
        # print("year is {}, steps is {}".format(year, steps % 12))
        if steps % 12 == 10.:
            year = year + 1
    cls.datelist = datelist
    return cls


def calc_anomaly(marray, cls, q = 0.95):
    import xarray as xr
    import numpy as np
    print("calc_anomaly called for {}".format(cls.name, marray.shape))
    steps_per_year = len(marray.sel(time=str(cls.startyear))['time'])
    month_index = marray.sel(time=str(cls.startyear))['time.month'].values
    months_string = {1:'jan', 2:'feb', 3:'mar', 4:'april', 5:'may', 6:'june', 7:'juli', 8:'aug', 9:'sep', 10:'okt', 11:'nov', 12:'dec'}
    months=[]
    for keys in month_index:
        months.append(months_string[keys])
    months_group = np.tile(months, len(marray['time.year'])/steps_per_year)
    labels = xr.DataArray(months_group, [marray.coords['time']], name='labels')

    clim = marray.groupby(labels).mean('time', keep_attrs=True).rename({'labels': 'time'})
    clim.name = 'clim'
    substract = lambda x, y: (x - y)
    anom = xr.apply_ufunc(substract, marray, np.tile(clim,(1,(cls.endyear+1-cls.startyear),1,1)), keep_attrs=True)
    anom.name = 'anom'
    std = anom.groupby(labels).reduce(np.percentile, dim='time', keep_attrs=True, q=q).rename({'labels': 'time'})
    std.name = 'std'
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
        name = data.name.replace(' ', '_')
    if 'name' in locals():
        print 'input name is: '.format(name)
        name = name + '_' + today + '.nc'
        pass
    else:
        name = 'netcdf_' + today + '.nc'
    print('{} to path {}'.format(name, path))
    data.to_netcdf(os.path.join(path, name))

def clustering_spatial(data, method, n_clusters, cls):
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
            X = StandardScaler().fit_transform(X_vectors)
            out_clus = cluster_method.fit(X)
            labels = out_clus.labels_
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
            method = 'AgglomerativeClustering' + '_' + link
            print method
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            clustering_plotting(cluster_method, input)
    folder = os.path.join(cls.base_path, 'Clustering', method)
    functions.quicksave_ncdf(input, cls, path=folder, name=region_values.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters) 
    return output
#%%
# clim = xr.DataArray(np.zeros([steps_per_year, len(marray['latitude']), len(marray['longitude'])]), dims=('time', 'latitude', 'longitude'), coords=[months, marray['latitude'].values,marray['longitude'].values])