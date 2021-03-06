# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:26:38 2021
Updated on Wed Jul 21 12:29:00 2021
@author: at18707
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

#Select Northern Hemisphere and convert to degree celsius

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


#Plot

fig ,ax = plt.subplots(ncols=5,nrows=3,figsize=((13,10)),sharex=True,sharey=True)

#DECEMBER
c1 = diff_temp_c.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0,0])
p_temp_c.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0,0])

c2 = diff_temp_c_da.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0,1],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_da.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0,1])

diff_temp_c_q.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0,2],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_q.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0,2])

diff_temp_c_dt.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0,3],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_dt.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0,3])

diff_temp_c_ns.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[0,4],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_ns.sel(time=slice('0001-12-01','0001-12-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[0,4])

#JANUARY
diff_temp_c.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1,0])
p_temp_c.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1,0])

diff_temp_c_da.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1,1],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_da.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1,1])

diff_temp_c_q.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1,2],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_q.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1,2])

diff_temp_c_dt.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1,3],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_dt.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1,3])

diff_temp_c_ns.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[1,4],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_ns.sel(time=slice('0002-01-01','0002-01-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[1,4])

#FEBRUARY
diff_temp_c.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2,0])
p_temp_c.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2,0])

diff_temp_c_da.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2,1],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_da.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2,1])

diff_temp_c_q.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2,2],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_q.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2,2])

diff_temp_c_dt.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2,3],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_dt.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2,3])

diff_temp_c_ns.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contourf(yincrease=False, yscale='log',levels=21,add_colorbar=False,ax=ax[2,4],vmin=-3,vmax=3,cmap='RdBu_r')
p_temp_ns.sel(time=slice('0002-02-01','0002-02-30'),pfull=slice(1,1000)).mean(('time','lon')).plot.contour(yincrease=False, yscale='log',levels=21,colors='grey',linewidths=1,ax=ax[2,4])


ylabels=['1000','100','10','1']
yticks=[1000,100,10,1]
 
ax[0,0].set_yticks(yticks)
ax[0,0].set_yticklabels(ylabels)
ax[0,0].set_ylabel('Pressure, hPa',fontsize=12)
ax[0,0].set_xlabel('')
ax[0,0].set_title('Control',fontsize=12)

ax[0,1].set_yticks(yticks)
ax[0,1].set_yticklabels(ylabels)
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
ax[0,1].set_title('Double All - Control',fontsize=12)

ax[0,2].set_yticks(yticks)
ax[0,2].set_yticklabels(ylabels)
ax[0,2].set_ylabel('')
ax[0,2].set_xlabel('')
ax[0,2].set_title('Quadruple All - Control',fontsize=12)

ax[0,3].set_yticks(yticks)
ax[0,3].set_yticklabels(ylabels)
ax[0,3].set_ylabel('')
ax[0,3].set_xlabel('')
ax[0,3].set_title('Double Tropo - Control',fontsize=12)

ax[0,4].set_yticks(yticks)
ax[0,4].set_yticklabels(ylabels)
ax[0,4].set_ylabel('')
ax[0,4].set_xlabel('')
ax[0,4].set_title('No Strat - Control',fontsize=12)

ax[1,0].set_yticks(yticks)
ax[1,0].set_yticklabels(ylabels)
ax[1,0].set_ylabel('Pressure, hPa',fontsize=12)
ax[1,0].set_xlabel('')
ax[1,0].set_title('')

ax[1,1].set_yticks(yticks)
ax[1,1].set_yticklabels(ylabels)
ax[1,1].set_ylabel('')
ax[1,1].set_xlabel('')
ax[1,1].set_title('')

ax[1,2].set_yticks(yticks)
ax[1,2].set_yticklabels(ylabels)
ax[1,2].set_ylabel('')
ax[1,2].set_xlabel('')
ax[1,2].set_title('')

ax[1,3].set_yticks(yticks)
ax[1,3].set_yticklabels(ylabels)
ax[1,3].set_ylabel('')
ax[1,3].set_xlabel('')
ax[1,3].set_title('')

ax[1,4].set_yticks(yticks)
ax[1,4].set_yticklabels(ylabels)
ax[1,4].set_ylabel('')
ax[1,4].set_xlabel('')
ax[1,4].set_title('')

ax[2,0].set_yticks(yticks)
ax[2,0].set_yticklabels(ylabels)
ax[2,0].set_ylabel('Pressure, hPa',fontsize=12)
ax[2,0].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2,0].set_title('')

ax[2,1].set_yticks(yticks)
ax[2,1].set_yticklabels(ylabels)
ax[2,1].set_ylabel('')
ax[2,1].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2,1].set_title('')

ax[2,2].set_yticks(yticks)
ax[2,2].set_yticklabels(ylabels)
ax[2,2].set_ylabel('')
ax[2,2].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2,2].set_title('')

ax[2,3].set_yticks(yticks)
ax[2,3].set_yticklabels(ylabels)
ax[2,3].set_ylabel('')
ax[2,3].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2,3].set_title('')

ax[2,4].set_yticks(yticks)
ax[2,4].set_yticklabels(ylabels)
ax[2,4].set_ylabel('')
ax[2,4].set_xlabel('Latitude, $^\circ$N',fontsize=12)
ax[2,4].set_title('')

#Control colorbar
p0 = ax[2,0].get_position().get_points().flatten()
p1 = ax[2,1].get_position().get_points().flatten()
p2 = ax[2,4].get_position().get_points().flatten()

cbar_ax = fig.add_axes([p0[0],0,p0[2]-p0[0],0.05])#1, 0.15, 0.05, 0.7])
cb1 = plt.colorbar(c1, orientation='horizontal',cax=cbar_ax,label='Temperature,$^\circ$C')

#Difference of Difference colorbar
cbar_ax = fig.add_axes([p1[0],0,p2[2]-p1[0],0.05])
cb2 = plt.colorbar(c2, orientation='horizontal',cax=cbar_ax,label='Temperature,$^\circ$C')

ax[0,0].text(-60, 50, 'Dec', fontsize=14)
ax[1,0].text(-60, 50, 'Jan', fontsize=14)
ax[2,0].text(-60, 50, 'Feb', fontsize=14)

plt.savefig("SupFig1\zonal_djf_temp_3deg_minus_present_minus_control_vertical_profile_ind_mon.png", bbox_inches = "tight")

plt.show()
