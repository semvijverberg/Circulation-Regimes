def quickplot(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    data.plot.contourf()




def PlateCarree(data, cls, type='abs', west_lon=-30, east_lon=40, south_lat=35, north_lat=65):
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    if len(data['time']) < 4:
        fig = plt.figure(figsize=(10, 8))
        pass
    else:
        "select less time steps to plot"
    for i in data['time'].values:
        rows = 2 if len(data['time']) / 2. > 1 else 1
        columns = int(len(data['time'])/rows + len(data['time']) % float(rows))
        gs = gridspec.GridSpec(rows, columns)
        fig_grid = np.concatenate([data['time'].values, np.array([np.datetime64('1900-01-01')])]) if len(data['time']) % float(rows) != 0 else data['time'].values
        r,c = np.where(fig_grid.reshape((gs._nrows,gs._ncols)) == i)[0:2]
        proj = ccrs.PlateCarree()
        # proj = ccrs.Mollweide()
        ax = fig.add_subplot(gs[int(r),int(c)], projection=proj)
        ax.add_feature(cfeat.COASTLINE)
        # ax.background_img(name='ne_shaded')
        # ax.add_feature(cfeat.LAND); # ax.add_feature(cfeat.OCEAN) # ax.add_feature(cfeat.LAKES, alpha=0.5)
        ax.add_feature(cfeat.BORDERS, linestyle=':')
        ax.set_xlim(west_lon, east_lon) # west lon, east_lon
        ax.set_ylim(south_lat,north_lat) # south_lat, north_lat
        values_region = find_region(data, west_lon,east_lon,south_lat, north_lat)
        if type=='norm':
            std_region = np.std(values_region).values
            min_region = np.min(values_region/std_region).values ; max_region = np.max(values_region/std_region).values
            plottable = np.squeeze(data.sel(time=i)/std_region)
            unit = r"T2m $\sigma$ "+"= {:}".format(round(std_region,1))
        elif type == 'abs':
            min_region = np.min(values_region) ; max_region = np.max(values_region)
            plottable = np.squeeze(data.sel(time=i))
            unit = r'T2m'

        lons, lats = np.meshgrid(np.linspace(0,360, len(data['longitude'].values)+1), data['latitude'].values)
        norm = colors.BoundaryNorm(boundaries=np.linspace(round(min_region),round(max_region),11), ncolors=256)
        map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree(), vmin=min_region, vmax=max_region, cmap=plt.cm.coolwarm, norm=norm)
        ax.set_title(np.str(i).split(':')[0]) ; cb = plt.colorbar(map, ax=ax, orientation='horizontal', use_gridspec=True, fraction = 0.1, pad=0.1, label=unit)
        # cb.set_label('label', rotation=0, position=(0.5, 0.5))
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False ; gl.xformatter = LONGITUDE_FORMATTER
        gl.ylabels_right= False ; gl.yformatter = LATITUDE_FORMATTER



def LamberConformal(data, cls, west_lon=-120, east_lon=-70, south_lat=20, north_lat=50):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    if len(data['time']) < 4:
        fig = plt.figure(figsize=(10, 8))
        pass
    else:
        "select less time steps to plot"
    for i in data['time'].values:
        rows = 2 if len(data['time']) / 2. > 1 else 1
        columns = int(len(data['time'])/rows + len(data['time']) % float(rows))
        gs = gridspec.GridSpec(rows, columns)
        fig_grid = np.concatenate([data['time'].values, np.array([0])]) if len(data['time']) % float(rows) != 0 else data['time'].valuesfe
        r,c = np.where(fig_grid.reshape((gs._nrows,gs._ncols)) == i)

        proj = ccrs.LambertConformal(central_latitude=25, central_longitude=265, standard_parallels=(25,25))
        ax = fig.add_subplot(gs[int(r),int(c)], projection=proj)
        ax.add_feature(cfeat.COASTLINE)
        # ax.add_feature(cfeat.LAND)
        # ax.add_feature(cfeat.OCEAN)
        # ax.add_feature(cfeat.LAKES, alpha=0.5)
        ax.add_feature(cfeat.BORDERS, linestyle=':')
        ax.set_title(np.str(i).split(':')[0])
        ax.set_extent({west_lon,east_lon, south_lat, north_lat})
        max_region, min_region, std_region = find_max_min_in_region(data, west_lon, east_lon, south_lat, north_lat)
        state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                  scale='50m', facecolor='none')
        ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
        plottable = np.squeeze(data.sel(time=i))
        map = plottable.plot.contourf(ax=ax, cmap=plt.cm.RdBu_r, levels=np.linspace(min_region, max_region,11),
                      add_colorbar=False, transform=ccrs.PlateCarree())  # levels=clevs,
        # lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
        # map = ax.pcolormesh(lons ,lats ,np.squeeze(data[i]), transform=ccrs.PlateCarree())
        cb = plt.colorbar(map, ax=ax, use_gridspec =True)
        cb.set_label('label', rotation=0, position=(1,1))
        plt.show()



def find_region(data, region='EA', west_lon = -30, east_lon = 40, south_lat = 35, north_lat = 65):
    if region == 'EA':
        west_lon = -30; east_lon = 40; south_lat = 35; north_lat = 65
    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))

        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
    return all_values

# west_lon=-30; east_lon=40; south_lat=35; north_lat=65