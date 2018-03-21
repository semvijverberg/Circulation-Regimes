import what_variable
import retrieve_ERA_i
import computations
import numpy as np
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
computations = computations




# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# Download variable
retrieve_ERA_i_field(temperature)

clim, anom, upperquan = computations.calc_anomaly(cls=temperature)

cls = temperature
from datetime import datetime, timedelta




def PlateCarree(data, cls, west_lon=-30, east_lon=40, south_lat=35, north_lat=65):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as colors
    if len(clim['time']) < 4:
        fig = plt.figure(figsize=(10, 8))
        pass
    else:
        "select less time steps to plot"
    for i in clim['time'].values:
        rows = 2 if len(clim['time']) / 2. > 1 else 1
        columns = int(len(clim['time'])/row + len(clim['time']) % float(row))
        gs = gridspec.GridSpec(rows, columns)
        fig_grid = np.concatenate([clim['time'].values[::-1], np.array([0])]) if len(clim['time']) % float(row) != 0 else clim['time'].values
        r,c = np.where(fig_grid.reshape((gs._nrows,gs._ncols)) == i)

        import cartopy.mpl.gridliner as cartgrid
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        proj = ccrs.PlateCarree()
        # proj = ccrs.Mollweide()

        ax = fig.add_subplot(gs[int(r),int(c)], projection=proj)
        ax.add_feature(cfeat.COASTLINE)
        # ax.add_feature(cfeat.LAND); # ax.add_feature(cfeat.OCEAN) # ax.add_feature(cfeat.LAKES, alpha=0.5)
        ax.add_feature(cfeat.BORDERS, linestyle=':')
        ax.set_xlim(west_lon, east_lon) # west lon, east_lon
        ax.set_ylim(south_lat,north_lat) # south_lat, north_lat
        max_region, min_region = find_max_min_in_region(west_lon,east_lon,south_lat, north_lat)
        ax.background_img(name='ne_shaded')
        lons, lats = np.meshgrid(np.linspace(0,360, len(data['longitude'].values)+1), data['latitude'].values)
        plottable = np.squeeze(data.sel(time=i))
        norm = colors.BoundaryNorm(boundaries=np.linspace(round(min_region),round(max_region),11), ncolors=256)
        map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree(), vmin=min_region, vmax=max_region, cmap=cm.coolwarm, norm=norm)
        ax.set_title(i) ; plt.colorbar(map, ax=ax, orientation='horizontal', use_gridspec=True, fraction = 0.1, pad=0.1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False ; gl.xformatter = LONGITUDE_FORMATTER
        gl.ylabels_right= False ; gl.yformatter = LATITUDE_FORMATTER


PlateCarree(clim-273, temperature)

PlateCarree(anom.sel(time=anom['time'][0]), temperature)




def quickplot(data):
    plt.figure(figsize=(10,8))
    data.plot.contourf()
quickplot(clim[0]-273)



def LamberConformal(data, cls, west_lon=-120, east_lon=-70, south_lat=20, north_lat=50):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    if len(clim['time']) < 4:
        fig = plt.figure(figsize=(10, 8))
        pass
    else:
        "select less time steps to plot"
    for i in clim['time'].values:
        rows = 2 if len(clim['time']) / 2. > 1 else 1
        columns = int(len(clim['time'])/row + len(clim['time']) % float(row))
        gs = gridspec.GridSpec(rows, columns)
        fig_grid = np.concatenate([clim['time'].values, np.array([0])]) if len(clim['time']) % float(row) != 0 else clim['time'].valuesfe
        r,c = np.where(fig_grid.reshape((gs._nrows,gs._ncols)) == i)

        proj = ccrs.LambertConformal(central_latitude=25, central_longitude=265, standard_parallels=(25,25))
        ax = fig.add_subplot(gs[int(r),int(c)], projection=proj)
        ax.add_feature(cfeat.COASTLINE)
        # ax.add_feature(cfeat.LAND)
        # ax.add_feature(cfeat.OCEAN)
        # ax.add_feature(cfeat.LAKES, alpha=0.5)
        ax.add_feature(cfeat.BORDERS, linestyle=':')
        # ax.set_title(str(labels))
        ax.set_extent({west_lon,east_lon, south_lat, north_lat})
        max_region, min_region = find_max_min_in_region(data, west_lon, east_lon, south_lat, north_lat)
        state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                                  scale='50m', facecolor='none')
        ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
        plottable = np.squeeze(data.sel(time=i))
        map = plottable.plot.contourf(ax=ax, cmap=plt.cm.RdBu_r, levels=np.linspace(min_region, max_region,11),
                      add_colorbar=False, transform=ccrs.PlateCarree())  # levels=clevs,
        # lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
        # map = ax.pcolormesh(lons ,lats ,np.squeeze(data[i]), transform=ccrs.PlateCarree())
        plt.colorbar(map, ax=ax )
        plt.show()
LamberConformal(clim-273.15, temperature)

def find_max_min_in_region(data, west_lon, east_lon, south_lat, north_lat):
    if west_lon <0 and east_lon > 0:
        max_region = np.max([np.max(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon))),
            np.max(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))])
        min_region = np.min([np.max(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon))),
            np.min(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))])
    if west_lon < 0 and east_lon < 0:
        max_region = np.max(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon)))
        min_region = np.min(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon)))
    return max_region, min_region





exit()