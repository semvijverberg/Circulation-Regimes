# import what_variable
# import retrieve_ERA_i
# import computations
# import numpy as np
#
#
# # assign instance
# temperature = what_variable.Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
#                        startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# # Download variable
# retrieve_ERA_i.retrieve_ERA_i_field(temperature)
# # retrieve_ERA_i_field(temperature)
# clim, anom, upperquan = computations.calc_anomaly(cls=temperature)


# np.save(temperature.base_path + 'clim_T2m', np.squeeze(clim))
data = np.load(temperature.base_path + 'clim_T2m.npy')
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 1)
# ax = plt.axes(projection=proj)
ax = plt.subplot(gs[0,0], projection=proj)
ax.add_feature(cfeat.COASTLINE)

lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
plt.show()