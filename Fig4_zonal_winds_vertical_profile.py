# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:46:16 2021
Updated on Wed Jul 21 11:52:00 2021
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
## Zonal mean Wind Profile Difference ##
###############################################

#Select Northern Hemisphere and take zonal average
p_uwind_z_c = p_file_c.ucomp.sel(lat=slice(0,90)).mean(('lon'))
f_uwind_z_c = f_file_c.ucomp.sel(lat=slice(0,90)).mean(('lon'))

p_uwind_z_da = p_file_da.ucomp.sel(lat=slice(0,90)).mean(('lon'))
f_uwind_z_da = f_file_da.ucomp.sel(lat=slice(0,90)).mean(('lon'))

p_uwind_z_dt = p_file_dt.ucomp.sel(lat=slice(0,90)).mean(('lon'))
f_uwind_z_dt = f_file_dt.ucomp.sel(lat=slice(0,90)).mean(('lon'))

p_uwind_z_ns = p_file_ns.ucomp.sel(lat=slice(0,90)).mean(('lon'))
f_uwind_z_ns = f_file_ns.ucomp.sel(lat=slice(0,90)).mean(('lon'))

p_uwind_z_q = p_file_q.ucomp.sel(lat=slice(0,90)).mean(('lon'))
f_uwind_z_q = f_file_q.ucomp.sel(lat=slice(0,90)).mean(('lon'))


#Check shapes are the same
print(p_uwind_z_c.shape, f_uwind_z_c.shape,
      p_uwind_z_da.shape, f_uwind_z_da.shape,
      p_uwind_z_dt.shape, f_uwind_z_dt.shape,
      p_uwind_z_ns.shape, f_uwind_z_ns.shape,
      p_uwind_z_q.shape, f_uwind_z_q.shape)

#Difference of 3deg from present 
diff_ucomp_z_c = f_uwind_z_c - p_uwind_z_c
diff_ucomp_z_da = f_uwind_z_da - p_uwind_z_da
diff_ucomp_z_dt = f_uwind_z_dt - p_uwind_z_dt
diff_ucomp_z_ns = f_uwind_z_ns - p_uwind_z_ns
diff_ucomp_z_q = f_uwind_z_q - p_uwind_z_q

#Difference of difference from control
diff_ucomp_z_c_da = diff_ucomp_z_da - diff_ucomp_z_c
diff_ucomp_z_c_dt = diff_ucomp_z_dt - diff_ucomp_z_c
diff_ucomp_z_c_ns = diff_ucomp_z_ns - diff_ucomp_z_c
diff_ucomp_z_c_q = diff_ucomp_z_q - diff_ucomp_z_c

#Plot
fig ,ax = plt.subplots(ncols=5,nrows=1,figsize=((16,7)),sharex=True,sharey=True)

#DECEMBER
c1 = diff_ucomp_z_c.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0])
c = p_uwind_z_c.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0])

c2 = diff_ucomp_z_da.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1],vmin=-10,vmax=10,cmap='RdBu_r') #vmin=-8,vmax=8 for diff from control
p_uwind_z_da.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1])

diff_ucomp_z_q.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2],vmin=-10,vmax=10,cmap='RdBu_r')
p_uwind_z_q.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2])

diff_ucomp_z_dt.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[3],vmin=-10,vmax=10,cmap='RdBu_r')
p_uwind_z_dt.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[3])

diff_ucomp_z_ns.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[4],vmin=-10,vmax=10,cmap='RdBu_r')
p_uwind_z_ns.sel(time=slice('0001-12-01','0002-02-30'),pfull=slice(1,1000)).mean(('time')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[4])



ylabels=['1000','100','10','1']
yticks=[1000,100,10,1]
 
ax[0].set_yticks(yticks)
ax[0].set_yticklabels(ylabels)
ax[0].set_ylabel('Pressure, hPa',fontsize=12)
ax[0].set_xlabel('')
ax[0].set_title('Control',fontsize=12)
ax[0].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[1].set_yticks(yticks)
ax[1].set_yticklabels(ylabels)
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Double All',fontsize=12) # - Control
ax[1].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[2].set_yticks(yticks)
ax[2].set_yticklabels(ylabels)
ax[2].set_ylabel('')
ax[2].set_xlabel('')
ax[2].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2].set_title('Quadruple All',fontsize=12)

ax[3].set_yticks(yticks)
ax[3].set_yticklabels(ylabels)
ax[3].set_ylabel('')
ax[3].set_xlabel('')
ax[3].set_title('Double Tropo',fontsize=12)
ax[3].set_xlabel('Latitude, $^\circ$N',fontsize=12)

ax[4].set_yticks(yticks)
ax[4].set_yticklabels(ylabels)
ax[4].set_ylabel('')
ax[4].set_xlabel('')
ax[4].set_title('No Strat',fontsize=12)
ax[4].set_xlabel('Latitude, $^\circ$N',fontsize=12)


#Control colorbar
p0 = ax[0].get_position().get_points().flatten()
p1 = ax[1].get_position().get_points().flatten()
p2 = ax[4].get_position().get_points().flatten()

cbar_ax = fig.add_axes([p0[0],0,p2[2]-p0[0],0.05])#1, 0.15, 0.05, 0.7])
cb1 = plt.colorbar(c1, orientation='horizontal',cax=cbar_ax,label='Zonal wind speed, ms$^{-1}$')

plt.savefig("Fig_4\\zonal_djf_mean_winds_3deg_minus_present_c_da_dt_ns_q_vertical_profile.png", bbox_inches = "tight")

plt.show()
