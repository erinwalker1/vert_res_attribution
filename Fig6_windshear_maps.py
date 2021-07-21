# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:54:52 2021
Updated on Wed Jul 21 12:06:00 2021
@author: Erin Walker
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.stats import ks_2samp


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

pval_filepath = "" # Filepath to the pvals

c_pval_dec = xr.open_dataset(pval_filepath + "c_pval_windshear_dec.nc")
c_pval_jan = xr.open_dataset(pval_filepath + "c_pval_windshear_jan.nc")
c_pval_feb = xr.open_dataset(pval_filepath + "c_pval_windshear_feb.nc")

da_pval = xr.open_dataset(pval_filepath + "da_pval_windshear.nc")

dt_pval = xr.open_dataset(pval_filepath + "dt_pval_windshear.nc")

q_pval = xr.open_dataset(pval_filepath + "q_pval_windshear.nc")

ns_pval = xr.open_dataset(pval_filepath + "ns_pval_windshear.nc")

##############################
## Calculate the wind speed ##
##############################
p_ucomp_c = p_file_c.ucomp
f_ucomp_c = f_file_c.ucomp
p_vcomp_c = p_file_c.vcomp
f_vcomp_c = f_file_c.vcomp

p_ucomp_da = p_file_da.ucomp
f_ucomp_da = f_file_da.ucomp
p_vcomp_da = p_file_da.vcomp
f_vcomp_da = f_file_da.vcomp

p_ucomp_dt = p_file_dt.ucomp
f_ucomp_dt = f_file_dt.ucomp
p_vcomp_dt = p_file_dt.vcomp
f_vcomp_dt = f_file_dt.vcomp

p_ucomp_ns = p_file_ns.ucomp
f_ucomp_ns = f_file_ns.ucomp
p_vcomp_ns = p_file_ns.vcomp
f_vcomp_ns = f_file_ns.vcomp

p_ucomp_q = p_file_q.ucomp
f_ucomp_q = f_file_q.ucomp
p_vcomp_q = p_file_q.vcomp
f_vcomp_q = f_file_q.vcomp


p_ws_c = np.sqrt(p_ucomp_c**2+p_vcomp_c**2)
f_ws_c = np.sqrt(f_ucomp_c**2+f_vcomp_c**2)
p_ws_da = np.sqrt(p_ucomp_da**2+p_vcomp_da**2)
f_ws_da = np.sqrt(f_ucomp_da**2+f_vcomp_da**2)
p_ws_dt = np.sqrt(p_ucomp_dt**2+p_vcomp_dt**2)
f_ws_dt = np.sqrt(f_ucomp_dt**2+f_vcomp_dt**2)
p_ws_ns = np.sqrt(p_ucomp_ns**2+p_vcomp_ns**2)
f_ws_ns = np.sqrt(f_ucomp_ns**2+f_vcomp_ns**2)
p_ws_q = np.sqrt(p_ucomp_q**2+p_vcomp_q**2)
f_ws_q = np.sqrt(f_ucomp_q**2+f_vcomp_q**2)

#Check shapes are the same
print(p_ws_c.shape, f_ws_c.shape,
      p_ws_da.shape, f_ws_da.shape,
      p_ws_dt.shape, f_ws_dt.shape,
      p_ws_ns.shape, f_ws_ns.shape,
      p_ws_q.shape, f_ws_q.shape)

#######################
## Open up the pvals ##
#######################

c_pval_dec = c_pval_dec.__xarray_dataarray_variable__
c_pval_jan = c_pval_jan.__xarray_dataarray_variable__
c_pval_feb = c_pval_feb.__xarray_dataarray_variable__

da_pval = da_pval.pval
dt_pval = dt_pval.pval
q_pval = q_pval.pval
ns_pval = ns_pval.pval


#############################
## Calculate the wind shear##
#############################

#Calculate wind shear between 850-200hPa

p_850_c = p_ws_c.sel(pfull=850, method='nearest')
p_200_c = p_ws_c.sel(pfull=200, method='nearest')
p_850_da = p_ws_da.sel(pfull=850, method='nearest')
p_200_da = p_ws_da.sel(pfull=200, method='nearest')
p_850_dt = p_ws_dt.sel(pfull=850, method='nearest')
p_200_dt = p_ws_dt.sel(pfull=200, method='nearest')
p_850_ns = p_ws_ns.sel(pfull=850, method='nearest')
p_200_ns = p_ws_ns.sel(pfull=200, method='nearest')
p_850_q = p_ws_q.sel(pfull=850, method='nearest')
p_200_q = p_ws_q.sel(pfull=200, method='nearest')

f_850_c = f_ws_c.sel(pfull=850, method='nearest')
f_200_c = f_ws_c.sel(pfull=200, method='nearest')
f_850_da = f_ws_da.sel(pfull=850, method='nearest')
f_200_da = f_ws_da.sel(pfull=200, method='nearest')
f_850_dt = f_ws_dt.sel(pfull=850, method='nearest')
f_200_dt = f_ws_dt.sel(pfull=200, method='nearest')
f_850_ns = f_ws_ns.sel(pfull=850, method='nearest')
f_200_ns = f_ws_ns.sel(pfull=200, method='nearest')
f_850_q = f_ws_q.sel(pfull=850, method='nearest')
f_200_q = f_ws_q.sel(pfull=200, method='nearest')

p_windshear_c = p_200_c - p_850_c 
f_windshear_c = f_200_c - f_850_c 

p_windshear_da = p_200_da - p_850_da 
f_windshear_da = f_200_da - f_850_da 

p_windshear_dt = p_200_dt - p_850_dt
f_windshear_dt = f_200_dt - f_850_dt

p_windshear_ns = p_200_ns - p_850_ns 
f_windshear_ns = f_200_ns - f_850_ns 

p_windshear_q = p_200_q - p_850_q 
f_windshear_q = f_200_q - f_850_q 

#change_lon
def change_lon(data):
    change_lon = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    sort_lon = change_lon.sortby('lon',ascending=True)
    return sort_lon

p_windshear_c = change_lon(p_windshear_c)
f_windshear_c = change_lon(f_windshear_c)

p_windshear_da = change_lon(p_windshear_da)
f_windshear_da = change_lon(f_windshear_da)

p_windshear_dt = change_lon(p_windshear_dt)
f_windshear_dt = change_lon(f_windshear_dt)

p_windshear_q = change_lon(p_windshear_q)
f_windshear_q = change_lon(f_windshear_q)

p_windshear_ns = change_lon(p_windshear_ns)
f_windshear_ns = change_lon(f_windshear_ns)

c_pval_dec = change_lon(c_pval_dec)

#Difference of 3deg from present 

diff_windshear_c = f_windshear_c - p_windshear_c
diff_windshear_da = f_windshear_da - p_windshear_da
diff_windshear_dt = f_windshear_dt - p_windshear_dt
diff_windshear_ns = f_windshear_ns - p_windshear_ns
diff_windshear_q = f_windshear_q - p_windshear_q


#Difference of difference from control
diff_windshear_c_da = diff_windshear_da - diff_windshear_c
diff_windshear_c_dt = diff_windshear_dt - diff_windshear_c
diff_windshear_c_ns = diff_windshear_ns - diff_windshear_c
diff_windshear_c_q = diff_windshear_q - diff_windshear_c



#Add cyclic point
def add_cyclic(data):
    lon = data.lon
    lon_idx = data.dims.index('lon')
    new, lons = add_cyclic_point(data.values,coord=lon, axis=lon_idx)
    new_data = xr.DataArray(new,coords=[data.time,data.lat,lons],dims=('time','lat','lon'))
    return new_data
 
diff_windshear_c = add_cyclic(diff_windshear_c)
diff_windshear_c_da = add_cyclic(diff_windshear_c_da)
diff_windshear_c_dt = add_cyclic(diff_windshear_c_dt)
diff_windshear_c_ns = add_cyclic(diff_windshear_c_ns)
diff_windshear_c_q = add_cyclic(diff_windshear_c_q)

diff_windshear_c = add_cyclic(diff_windshear_c)
diff_windshear_da = add_cyclic(diff_windshear_da)
diff_windshear_dt = add_cyclic(diff_windshear_dt)
diff_windshear_ns = add_cyclic(diff_windshear_ns)
diff_windshear_q = add_cyclic(diff_windshear_q)


##############
## Plotting ##
##############

fig = plt.figure(figsize=(17,2))
ax1 = plt.subplot(1,5,1,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax2 = plt.subplot(1,5,2,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax3 = plt.subplot(1,5,3,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax4 = plt.subplot(1,5,4,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax5 = plt.subplot(1,5,5,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))#,sharex=False, sharey=False)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()
ax5.coastlines()

ax1.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax2.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax3.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax4.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax5.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())

c1 = diff_windshear_c.sel(time=slice('0001-12-01','0002-02-30')).mean('time').plot.contourf(levels=21,add_colorbar=False,ax=ax1,transform=ccrs.PlateCarree(),vmin=-7,vmax=7,cmap='RdBu_r')
c2 = diff_windshear_c_da.sel(time=slice('0001-12-01','0002-02-30')).mean('time').plot.contourf(levels=21,add_colorbar=False,ax=ax2,transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='RdBu_r')
diff_windshear_c_q.sel(time=slice('0001-12-01','0002-02-30')).mean('time').plot.contourf(levels=21,add_colorbar=False,ax=ax3,transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='RdBu_r')
diff_windshear_c_dt.sel(time=slice('0001-12-01','0002-02-30')).mean('time').plot.contourf(levels=21,add_colorbar=False,ax=ax4,transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='RdBu_r')
diff_windshear_c_ns.sel(time=slice('0001-12-01','0002-02-30')).mean('time').plot.contourf(levels=21,add_colorbar=False,ax=ax5,transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='RdBu_r')

c_pval_dec.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax1,transform=ccrs.PlateCarree())
c_pval_jan.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax1,transform=ccrs.PlateCarree())
c_pval_feb.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax1,transform=ccrs.PlateCarree())

da_pval.plot.contourf(colors='none',hatches=['....'],alpha=0,add_colorbar=False,levels=2,ax=ax2,transform=ccrs.PlateCarree())
levels=2,ax=ax2,transform=ccrs.PlateCarree())

q_pval.plot.contourf(colors='none',hatches=['....'],alpha=0,add_colorbar=False,levels=2,ax=ax3,transform=ccrs.PlateCarree())

dt_pval.plot.contourf(colors='none',hatches=['....'],alpha=0,add_colorbar=False,levels=2,ax=ax4,transform=ccrs.PlateCarree())

ns_pval.plot.contourf(colors='none',hatches=['....'],alpha=0,add_colorbar=False,levels=2,ax=ax5,transform=ccrs.PlateCarree())

p0 = ax1.get_position().get_points().flatten()
p1 = ax2.get_position().get_points().flatten()
p2 = ax5.get_position().get_points().flatten()

cbar_ax = fig.add_axes([p0[0],0,p0[2]-p0[0],0.05])#1, 0.15, 0.05, 0.7])
cb1 = plt.colorbar(c1, orientation='horizontal',cax=cbar_ax,label='ms$^{-1}$')

cbar_ax = fig.add_axes([p1[0],0,p2[2]-p1[0],0.05])
cb1 = plt.colorbar(c2, orientation='horizontal',cax=cbar_ax,label='Windshear, ms$^{-1}$')

#fig.suptitle('3deg - Present DJF Windshear (200-850hPa)')#,x=0.55,y=0.99)
ax1.set_title('Control',fontsize=12)
ax2.set_title('Double All - Control',fontsize=12)
ax3.set_title('Quadruple All - Control',fontsize=12)
ax4.set_title('Double Tropo - Control',fontsize=12)
ax5.set_title('No Strat - Control',fontsize=12)

#plt.tight_layout()

plt.savefig("Fig6/windshear_3deg_minus_present_minus_control_maps_djf_mean.png", bbox_inches = "tight")

plt.show()
