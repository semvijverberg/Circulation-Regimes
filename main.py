import what_variable
import retrieve_ERA_i
import computations
import numpy as np
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field

# do changes


# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2.5/2.5', stream='moda')
# Download variable
retrieve_ERA_i_field(temperature)
# retrieve_ERA_i_field(temperature)
clim, anom, upperquan = computations.calc_anomaly(cls=temperature)



def PlateCarree(data, cls):
    # proj = ccrs.Mollweide()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import cartopy.mpl.gridliner as cartgrid
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0,0], projection=proj)
    ax.add_feature(cfeat.COASTLINE)
    # ax.add_feature(cfeat.LAND)
    # ax.add_feature(cfeat.OCEAN)
    # ax.add_feature(cfeat.LAKES, alpha=0.5)
    ax.add_feature(cfeat.BORDERS, linestyle=':')
    ax.set_xlim(-60, 30) # west lon, east_lon
    ax.set_ylim(35,70) # south_lat, north_lat
    state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                              scale='50m', facecolor='none')
    ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
    ax.background_img(name='ne_shaded')
    lons, lats = np.meshgrid(np.linspace(0,360, len(data['longitude'].values)+1), data['latitude'].values)
    plottable = np.squeeze(data)
    # map = ax.contourf(longitude, latitude, plottable, transform=ccrs.PlateCarree())
    map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree())
    plt.colorbar(map, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    # cartgrid.Gridliner(ax, ccrs.PlateCarree(), draw_labels=True)

PlateCarree(clim-273, temperature)

PlateCarree(anom.sel(time=anom['time'][0]), temperature)








def LamberConformal(data, cls):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(1, 1)
    proj = ccrs.LambertConformal(central_latitude=25, central_longitude=265, standard_parallels=(25,25))
    ax = fig.add_subplot(gs[0,0], projection=proj)
    ax.add_feature(cfeat.COASTLINE)
    # ax.add_feature(cfeat.LAND)
    # ax.add_feature(cfeat.OCEAN)
    # ax.add_feature(cfeat.LAKES, alpha=0.5)
    ax.add_feature(cfeat.BORDERS, linestyle=':')
    ax.set_extent({-120,-70, 20, 50}) # lon west, lon east, lat north, lat south.
    state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                              scale='50m', facecolor='none')
    ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
    lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
    map = ax.pcolormesh(lons ,lats ,np.squeeze(data), transform=ccrs.PlateCarree())
    plt.colorbar(map, ax=ax )
LamberConformal(clim-273.15, temperature)







#
# from mpl_toolkits.basemap import Basemap
# import matplotlib.gridspec as gridspec


# m = Basemap(projection='mill') # 'mill works'
# # m = Basemap(projection='robin', lon_0=0, lat_0=0) # 'robin works'
# # m = Basemap(projection='nplaea',boundinglat=10,lon_0=270,resolution='l')
# m.drawcoastlines()
# # m.fillcontinents()
# plt.title('test')
# # latitude = marray['latitude'].values
# # longitude = marray['longitude'].values
# lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
# cs = m.pcolormesh(lons,lats,data, latlon=True)
# plt.show()


# # def plot_2D(cls):
# data = clim
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# import numpy as np
# plottable = np.squeeze(data)
# lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title("Robinson projection")
# map = Basemap(projection='merc', lon_0 = 10, lat_0 = 0)
# map.drawmapboundary(fill_color='aqua')
# map.fillcontinents(color='coral')
# map.drawcoastlines()
# map.pcolormesh(lons, lats, plottable, vmin=plottable.min(), vmax=plottable.max(), latlon=True)
# plt.show()
#

exit()