import what_variable
import retrieve_ERA_i
import computations
import numpy as np
import plotting
Variable = what_variable.Variable
retrieve_ERA_i_field = retrieve_ERA_i.retrieve_ERA_i_field
computations = computations
plotting = plotting


# assign instance
temperature = Variable(name='2 metre temperature', levtype='sfc', lvllist=0, var_cf_code='167.128',
                       startyear=1979, endyear=2017, startmonth=6, endmonth=8, grid='2.5/2.5', stream='mnth')
# Download variable
retrieve_ERA_i_field(temperature)
clim, anom, upperquan = computations.calc_anomaly(cls=temperature)
cls = temperature

# PlateCarree(clim-273, temperature)
data = anom.isel(time=np.array(np.where(anom['time.year']==1980)).reshape(3))
plotting.PlateCarree(data, temperature)



exit()










