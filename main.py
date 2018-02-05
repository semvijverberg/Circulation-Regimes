import what_variable
import retrieve_ERA_i_field_class
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i_field_class.retrieve_ERA_i_field
if Variable is not what_variable.Variable:
    Variable = reload(Variable)
if retrieve_ERA_i_field is not retrieve_ERA_i_field_class.retrieve_ERA_i_field:
    print "reloaded"
    retrieve_ERA_i_field = reload(retrieve_ERA_i_field_class)




# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')

# Download variable
retrieve_ERA_i_field(temperature)



# retrieve_ERA_i_field(temperature)

def calc_anomaly(cls, decode_cf=True, decode_coords=True):
    import xarray as xr
    import os
    # load in file
    file_path = os.path.join(cls.base_path, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))

    print "dimensions {}".format(cls.name, marray.shape)
    print marray.dims
    clim = marray.mean(dim='time', attrs=True)
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time")
    return clim, anom, upperquan



clim, anom, upperquan = calc_anomaly(temperature)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


data = clim
m = Basemap(projection='mill')
m.drawcoastlines()
m.fillcontinents()
plt.title('test')
cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
plt.show()



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
