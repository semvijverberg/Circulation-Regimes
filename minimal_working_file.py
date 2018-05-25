import what_variable
import retrieve_ERA_i
import functions
import numpy as np


# assign instance
temperature = what_variable.Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=1979, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# Download variable
retrieve_ERA_i.retrieve_ERA_i_field(temperature)
# retrieve_ERA_i_field(temperature)
clim, anom, upperquan = functions.calc_anomaly(cls=temperature)


data = np.squeeze(clim)
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 1)
# ax = plt.axes(projection=proj)
ax = plt.subplot(gs[0,0], projection=proj)
ax.add_feature(cfeat.COASTLINE)
lons = np.arange(0,360.01,2.5)
lats = np.arange(-90,90.01,2.5)
lons, lats = np.meshgrid(lons, lats)
# lons, lats = np.meshgrid(data['longitude'].values, data['latitude'].values)
map = ax.pcolormesh(lons, lats, plottable, transform=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
plt.show()