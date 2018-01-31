from what_variable import Variable
from retrieve_ERA_i_field_class import retrieve_ERA_i_field


# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1980, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')

# Download variable
retrieve_ERA_i_field(temperature)



# retrieve_ERA_i_field(temperature)

def calc_anomaly(cls, decode_cf=True, decode_coords=True):
    import xarray as xr
    # load in file
    file_path = os.path.join(cls.base_path, cls.filename)
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True)
    marray = ncdf.to_array(file_path).rename(({file_path: cls.name}))

    print "dimensions {}".format(cls.name, marray.shape)
    print marray.dims
    clim = marray.mean(dim='time')
    anom = marray - clim
    upperquan = marray.quantile(0.95, dim="time", keep_attrs=True)
    return clim, anom, upperquan



clim, anom, upperquan = calc_anomaly(temperature)

# def plot_2D(cls):
data = clim
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
plottable = np.squeeze(data)
lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Robinson projection")
map = Basemap(projection='merc', lon_0 = 10, lat_0 = 0)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral')
map.drawcoastlines()
map.pcolormesh(lons, lats, plottable, vmin=plottable.min(), vmax=plottable.max(), latlon=True)
plt.show()

# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# clim, anom, upperquan = calc_anomaly(temperature)
# ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
# plt.contourf(np.squeeze(clim))
# plt.contourf(np.mean(np.squeeze(anom),axis=0))
# ax.set_global() ; ax.coastlines


