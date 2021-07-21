# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:46:16 2021
Updated on Wed Jul 21 11:44:00 2021
@author: Erin Walker
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

"""
P = Present
F = 3Deg
C = Control
DA = Double All
DT = Double Tropopause
Q = Quadruple All
NS = No Stratopshere
"""
filepath = "" # Filepath to ensemble mean files 
p_file_c = xr.open_dataset(filepath + "daily_ensmean_pres_c.nc",chunks={'time':30})
f_file_c = xr.open_dataset(filepath + "daily_ensmean_3deg_c.nc")

p_file_da = xr.open_dataset(filepath + "daily_ensmean_pres_da.nc",chunks={'time':30})
f_file_da = xr.open_dataset(filepath + "daily_ensmean_3deg_da.nc")

p_file_dt = xr.open_dataset(filepath + "daily_ensmean_pres_dt.nc",chunks={'time':30})
f_file_dt = xr.open_dataset(filepath + "daily_ensmean_3deg_dt.nc")

p_file_ns = xr.open_dataset(filepath + "daily_ensmean_pres_ns.nc",chunks={'time':30})
f_file_ns = xr.open_dataset(filepath + "daily_ensmean_3deg_ns.nc")

p_file_q = xr.open_dataset(filepath + "daily_ensmean_pres_q.nc",chunks={'time':30})
f_file_q = xr.open_dataset(filepath + "daily_ensmean_3deg_q.nc")


###############################################
## Zonal mean Temperature Profile Difference ##
###############################################

# Select Northern Hemisphere region and convert to degrees celsius
p_temp_c = p_file_c.temp[:].sel(lat=slice(0,90)) - 273.15
f_temp_c = f_file_c.temp[:].sel(lat=slice(0,90)) - 273.15

p_temp_da = p_file_da.temp[:].sel(lat=slice(0,90)) - 273.15
f_temp_da = f_file_da.temp[:].sel(lat=slice(0,90)) - 273.15

p_temp_dt = p_file_dt.temp[:].sel(lat=slice(0,90)) - 273.15
f_temp_dt = f_file_dt.temp[:].sel(lat=slice(0,90)) - 273.15

p_temp_ns = p_file_ns.temp[:].sel(lat=slice(0,90)) - 273.15
f_temp_ns = f_file_ns.temp[:].sel(lat=slice(0,90)) - 273.15

p_temp_q = p_file_q.temp[:].sel(lat=slice(0,90)) - 273.15
f_temp_q = f_file_q.temp[:].sel(lat=slice(0,90)) - 273.15


#Check shapes are the same
print(p_temp_c.shape, f_temp_c.shape,
      p_temp_da.shape, f_temp_da.shape,
      p_temp_dt.shape, f_temp_dt.shape,
      p_temp_ns.shape, f_temp_ns.shape,
      p_temp_q.shape, f_temp_q.shape)

#Difference of 3deg from present 
diff_temp_c = f_temp_c - p_temp_c
diff_temp_da = f_temp_da - p_temp_da
diff_temp_dt = f_temp_dt - p_temp_dt
diff_temp_ns = f_temp_ns - p_temp_ns
diff_temp_q = f_temp_q - p_temp_q

#Difference of difference from control
diff_temp_c_da = diff_temp_da - diff_temp_c
diff_temp_c_dt = diff_temp_dt - diff_temp_c
diff_temp_c_ns = diff_temp_ns - diff_temp_c
diff_temp_c_q = diff_temp_q - diff_temp_c

# Plot

fig ,ax = plt.subplots(ncols=5,nrows=1,figsize=((16,7)),sharex=True,sharey=True)

#DECEMBER
c1 = diff_temp_c.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0])
p_temp_c.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0])

c2 = diff_temp_c_da.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1],vmin=-3,vmax=3,cmap='RdBu_r') #vmin=-3,vmax=3 for diff from control
p_temp_da.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1])

c3 = diff_temp_c_q.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_q.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2])

c4 = diff_temp_c_dt.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[3],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_dt.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[3])

c5 = diff_temp_c_ns.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[4],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_ns.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[4])


ylabels=['1000','100','10','1']
yticks=[1000,100,10,1]
 
ax[0].set_yticks(yticks)
ax[0].set_yticklabels(ylabels,fontsize=12)
ax[0].set_ylabel('Pressure, hPa',fontsize=12)
ax[0].set_xlabel('')
ax[0].set_title('Control',fontsize=12)
ax[0].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[1].set_yticks(yticks)
ax[1].set_yticklabels(ylabels)
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Double All - Control',fontsize=12) #- Control 
ax[1].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[2].set_yticks(yticks)
ax[2].set_yticklabels(ylabels)
ax[2].set_ylabel('')
ax[2].set_xlabel('')
ax[2].set_title('Quadruple All - Control',fontsize=12)
ax[2].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[3].set_yticks(yticks)
ax[3].set_yticklabels(ylabels)
ax[3].set_ylabel('')
ax[3].set_xlabel('')
ax[3].set_title('Double Tropo - Control',fontsize=12)
ax[3].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[4].set_yticks(yticks)
ax[4].set_yticklabels(ylabels)
ax[4].set_ylabel('')
ax[4].set_xlabel('')
ax[4].set_title('No Strat - Control',fontsize=12)
ax[4].set_xlabel('Latitude, $^\circ$N',fontsize=12)

#Control colorbar
p0 = ax[0].get_position().get_points().flatten()
p1 = ax[1].get_position().get_points().flatten()
p2 = ax[4].get_position().get_points().flatten()

cbar_ax = fig.add_axes([p0[0],0,p0[2]-p0[0],0.05])#1, 0.15, 0.05, 0.7])
cb1 = plt.colorbar(c1, orientation='horizontal',cax=cbar_ax,label='Temperature,$^\circ$C')

#Difference of Difference colorbar
cbar_ax = fig.add_axes([p1[0],0,p2[2]-p1[0],0.05])
cb2 = plt.colorbar(c2, orientation='horizontal',cax=cbar_ax,label='Temperature,$^\circ$C')

plt.savefig("Fig_3\\zonal_djf_mean_temp_3deg_minus_present_minus_control_c_da_dt_ns_q_vertical_profile.png", bbox_inches = "tight")

plt.show()
