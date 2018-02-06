
%load_ext autoreload
%autoreload 2
import what_variable
import retrieve_ERA_i
import computations
import numpy as np
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field


# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# Download variable
retrieve_ERA_i_field(temperature)
# retrieve_ERA_i_field(temperature)
clim, anom, upperquan = computations.calc_anomaly(cls=temperature)



# def PlateCarree(data, cls):

# proj = ccrs.Mollweide()
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.mpl.gridliner as cartgrid
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



data = clim
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 1)
# ax = plt.axes(projection=proj)
ax = plt.subplot(gs[0,0], projection=proj)
ax.add_feature(cfeat.COASTLINE)
# ax.add_feature(cfeat.LAND)
# ax.add_feature(cfeat.OCEAN)
# ax.add_feature(cfeat.LAKES, alpha=0.5)
ax.add_feature(cfeat.BORDERS, linestyle=':')
ax.set_extent({-40,40, 20, 50}) # lon west, lon east, lat north, lat south.
state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                          scale='50m', facecolor='none')
ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
ax.background_img(name='ne_shaded')
longitude = data['longitude'].values
latitude = data['latitude'].values
lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
plottable = np.squeeze(data)
# map = ax.contourf(longitude, latitude, plottable, transform=ccrs.PlateCarree())
map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree())
plt.colorbar(map, ax=ax)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
# cartgrid.Gridliner(ax, ccrs.PlateCarree(), draw_labels=True)
plt.show()




from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec


data = np.squeeze(clim)
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

def LamberConformal(data, cls):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0,0], projection=ccrs.LambertConformal())
    ax.add_feature(cfeat.COASTLINE)
    # ax.add_feature(cfeat.LAND)
    # ax.add_feature(cfeat.OCEAN)
    # ax.add_feature(cfeat.LAKES, alpha=0.5)
    ax.add_feature(cfeat.BORDERS, linestyle=':')
    # ax.set_extent({-120,-70, 20, 50}) # lon west, lon east, lat north, lat south.
    state_borders = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes',
                                              scale='50m', facecolor='none')
    ax.add_feature(state_borders, linestyle='dotted', edgecolor='black')
    longitude = data['longitude'].values-180
    latitude = data['latitude'].values
    lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
    # ax.contourf(longitude,latitude,data, 60, transform=ccrs.PlateCarree())
    map = ax.contourf(lons ,lats ,np.squeeze(data), transform=ccrs.PlateCarree())
    plt.colorbar(map, ax=ax )
    plt.show()
LamberConformal(clim-273.15, temperature)







# marray = clim
# latcorners = marray['latitude'].values
# loncorners = marray['longitude'].values
# lon_0 = 0
# lat_0 = 0
# # create figure and axes instances
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_axes([0.1,0.1,0.8,0.8])
# # create polar stereographic Basemap instance.
# m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
#             llcrnrlat=latcorners[0],urcrnrlat=latcorners[-1],\
#             llcrnrlon=loncorners[0],urcrnrlon=loncorners[-1])#,\
#             # rsphere=6371200.,resolution='l')
# # draw coastlines, state and country boundaries, edge of map.
# m.drawcoastlines()
# m.drawstates()
# m.drawcountries()
# # draw parallels.
# parallels = np.arange(0.,90,10.)
# m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# # draw meridians
# meridians = np.arange(180.,360.,10.)
# m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
# ny = data.shape[0]; nx = data.shape[1]
# lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
# x, y = m(lons, lats) # compute map proj coordinates.
# # draw filled contours.
# clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
# cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
# # add colorbar.
# cbar = m.colorbar(cs,location='bottom',pad="5%")
# cbar.set_label('mm')
# # add title
# plt.title(prcpvar.long_name+' for period ending '+prcpvar.dateofdata)
# plt.show()
#
#
#
#
#
#
#
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
# # import cartopy.crs as ccrs
# # import matplotlib.pyplot as plt
# # clim, anom, upperquan = calc_anomaly(temperature)
# # ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
# # plt.contourf(np.squeeze(clim))
# # plt.contourf(np.mean(np.squeeze(anom),axis=0))
# # ax.set_global() ; ax.coastlines
#
#
exit()