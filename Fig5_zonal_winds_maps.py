# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:00:10 2021
Updated on Wed Jul 21 11:55:13 2021
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

pval_filepath = "" # Filepath to the pvals
c_pval_10 = xr.open_dataset(pval_filepath + "c_pval_10hpa_uwinds.nc")
c_pval_500 = xr.open_dataset(pval_filepath + "c_pval_500hpa_uwinds1.nc")
c_pval_850 = xr.open_dataset(pval_filepath + "c_pval_850hpa_uwinds1.nc")

da_pval_10 = xr.open_dataset(pval_filepath + "da_pval_10hpa.nc")
da_pval_500 = xr.open_dataset(pval_filepath + "da_pval_500hpa.nc")
da_pval_850 = xr.open_dataset(pval_filepath + "da_pval_850hpa.nc")

dt_pval_10 = xr.open_dataset(pval_filepath + "dt_pval_10hpa.nc")
dt_pval_500 = xr.open_dataset(pval_filepath + "dt_pval_500hpa.nc")
dt_pval_850 = xr.open_dataset(pval_filepath + "dt_pval_850hpa.nc")

q_pval_10 = xr.open_dataset(pval_filepath + "q_pval_10hpa.nc")
q_pval_500 = xr.open_dataset(pval_filepath + "q_pval_500hpa.nc")
q_pval_850 = xr.open_dataset(pval_filepath + "q_pval_850hpa.nc")

ns_pval_10 = xr.open_dataset(pval_filepath + "ns_pval_10hpa.nc")
ns_pval_500 = xr.open_dataset(pval_filepath + "ns_pval_500hpa.nc")
ns_pval_850 = xr.open_dataset(pval_filepath + "ns_pval_850hpa.nc")

###############################################
## Zonal mean Wind Profile Difference ##
###############################################

#Select Northern Hemisphere and DJF 

p_uwind_z_c = p_file_c.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')
f_uwind_z_c = f_file_c.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')

p_uwind_z_da = p_file_da.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')
f_uwind_z_da = f_file_da.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')

p_uwind_z_dt = p_file_dt.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')
f_uwind_z_dt = f_file_dt.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')

p_uwind_z_ns = p_file_ns.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')
f_uwind_z_ns = f_file_ns.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')

p_uwind_z_q = p_file_q.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')
f_uwind_z_q = f_file_q.ucomp.sel(lat=slice(0,90),time=slice('0001-12-01','0002-02-30')).mean('time')

c_pval_10 = c_pval_10.__xarray_dataarray_variable__
c_pval_500 = c_pval_500.pvals
c_pval_850 = c_pval_850.pvals

da_pval_10 = da_pval_10.pval
da_pval_500 = da_pval_500.pval
da_pval_850 = da_pval_850.pval

dt_pval_10 = dt_pval_10.pval
dt_pval_500 = dt_pval_500.pval
dt_pval_850 = dt_pval_850.pval

q_pval_10 = q_pval_10.pval
q_pval_500 = q_pval_500.pval
q_pval_850 = q_pval_850.pval

ns_pval_10 = ns_pval_10.pval
ns_pval_500 = ns_pval_500.pval
ns_pval_850 = ns_pval_850.pval

#Check shapes are the same
print(p_uwind_z_c.shape, f_uwind_z_c.shape,
      p_uwind_z_da.shape, f_uwind_z_da.shape,
      p_uwind_z_dt.shape, f_uwind_z_dt.shape,
      p_uwind_z_ns.shape, f_uwind_z_ns.shape,
      p_uwind_z_q.shape, f_uwind_z_q.shape)

#Difference of 3deg from present 
diff_ucomp_c = f_uwind_z_c - p_uwind_z_c
diff_ucomp_da = f_uwind_z_da - p_uwind_z_da
diff_ucomp_dt = f_uwind_z_dt - p_uwind_z_dt
diff_ucomp_ns = f_uwind_z_ns - p_uwind_z_ns
diff_ucomp_q = f_uwind_z_q - p_uwind_z_q

#Difference of difference from control
diff_ucomp_c_da = diff_ucomp_da - diff_ucomp_c
diff_ucomp_c_dt = diff_ucomp_dt - diff_ucomp_c
diff_ucomp_c_ns = diff_ucomp_ns - diff_ucomp_c
diff_ucomp_c_q = diff_ucomp_q - diff_ucomp_c

#Add cyclic point to data
def add_cyclic(data):
    lon = data.lon
    lon_idx = data.dims.index('lon')
    new, lons = add_cyclic_point(data.values,coord=lon, axis=lon_idx)
    new_data = xr.DataArray(new,coords=[data.pfull,data.lat,lons],dims=('pfull','lat','lon'))
    return new_data
 

diff_ucomp_c = add_cyclic(diff_ucomp_c)
diff_ucomp_c_da = add_cyclic(diff_ucomp_c_da)
diff_ucomp_c_dt = add_cyclic(diff_ucomp_c_dt)
diff_ucomp_c_ns = add_cyclic(diff_ucomp_c_ns)
diff_ucomp_c_q = add_cyclic(diff_ucomp_c_q)

diff_ucomp_da = add_cyclic(diff_ucomp_da)
diff_ucomp_dt = add_cyclic(diff_ucomp_dt)
diff_ucomp_ns = add_cyclic(diff_ucomp_ns)
diff_ucomp_q = add_cyclic(diff_ucomp_q)

#Add cyclic point for pvals
def add_cyclic(data):
    lon = data.lon
    lon_idx = data.dims.index('lon')
    new, lons = add_cyclic_point(data.values,coord=lon, axis=lon_idx)
    new_data = xr.DataArray(new,coords=[data.lat,lons],dims=('lat','lon'))
    return new_data

c_pval_10 = add_cyclic(c_pval_10)
c_pval_500 = add_cyclic(c_pval_500)
c_pval_850 = add_cyclic(c_pval_850)

da_pval_10 = add_cyclic(da_pval_10)
da_pval_500 = add_cyclic(da_pval_500)
da_pval_850 = add_cyclic(da_pval_850)

dt_pval_10 = add_cyclic(dt_pval_10)
dt_pval_500 = add_cyclic(dt_pval_500)
dt_pval_850 = add_cyclic(dt_pval_850)

q_pval_10 = add_cyclic(q_pval_10)
q_pval_500 = add_cyclic(q_pval_500)
q_pval_850 = add_cyclic(q_pval_850)

ns_pval_10 = add_cyclic(ns_pval_10)
ns_pval_500 = add_cyclic(ns_pval_500)
ns_pval_850 = add_cyclic(ns_pval_850)

def store_sig_points(pval):
    lon = list()
    lat = list()
    for i in range(0,pval.shape[0]):
        for j in range(0,pval.shape[1]):
            if pval[i,j] <= 0.05:
                lon.append(pval[i,j].lon.values.item())
                lat.append(pval[i,j].lat.values.item())
    return lon,lat

da_10_pval_lon, da_10_pval_lat = store_sig_points(da_pval_10)
dt_10_pval_lon, dt_10_pval_lat = store_sig_points(dt_pval_10)
q_10_pval_lon, q_10_pval_lat = store_sig_points(q_pval_10)
ns_10_pval_lon, ns_10_pval_lat = store_sig_points(ns_pval_10)

da_500_pval_lon, da_500_pval_lat = store_sig_points(da_pval_500)
dt_500_pval_lon, dt_500_pval_lat = store_sig_points(dt_pval_500)
q_500_pval_lon, q_500_pval_lat = store_sig_points(q_pval_500)
ns_500_pval_lon, ns_500_pval_lat = store_sig_points(ns_pval_500)

da_850_pval_lon, da_850_pval_lat = store_sig_points(da_pval_850)
dt_850_pval_lon, dt_850_pval_lat = store_sig_points(dt_pval_850)
q_850_pval_lon, q_850_pval_lat = store_sig_points(q_pval_850)
ns_850_pval_lon, ns_850_pval_lat = store_sig_points(ns_pval_850)

#Masking where is not significant
#mask_c = diff_ucomp_c.sel(pfull=10,method='nearest').where(c_pval_10>0)
#mask_da = diff_ucomp_da.sel(pfull=10,method='nearest').where(da_pval_10>0)
#mask_new = mask_da - mask_c
#mask_new.plot.contourf(levels=21)

#Maps 3deg minus present

fig = plt.figure(figsize=(13,6))

ax1 = plt.subplot(3,5,1,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax2 = plt.subplot(3,5,2,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax3 = plt.subplot(3,5,3,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax4 = plt.subplot(3,5,4,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax5 = plt.subplot(3,5,5,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))

ax6 = plt.subplot(3,5,6,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax7 = plt.subplot(3,5,7,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax8 = plt.subplot(3,5,8,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax9 = plt.subplot(3,5,9,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax10 = plt.subplot(3,5,10,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))

ax11 = plt.subplot(3,5,11,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax12 = plt.subplot(3,5,12,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax13 = plt.subplot(3,5,13,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax14 = plt.subplot(3,5,14,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))
ax15 = plt.subplot(3,5,15,projection=ccrs.Orthographic(central_longitude=-30, central_latitude=50))

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()
ax5.coastlines()

ax6.coastlines()
ax7.coastlines()
ax8.coastlines()
ax9.coastlines()
ax10.coastlines()

ax11.coastlines()
ax12.coastlines()
ax13.coastlines()
ax14.coastlines()
ax15.coastlines()

ax1.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax2.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax3.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax4.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax5.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())

ax6.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax7.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax8.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax9.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax10.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())

ax11.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax12.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax13.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax14.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())
ax15.set_extent((0, 500000, 20, 500000),crs=ccrs.PlateCarree())

c1 = diff_ucomp_c.sel(pfull=10,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax1,transform=ccrs.PlateCarree(),vmin=-15,vmax=15,cmap='RdBu_r')
c2 = diff_ucomp_c.sel(pfull=500,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax6,transform=ccrs.PlateCarree(),vmin=-5,vmax=5,cmap='RdBu_r')
c3 = diff_ucomp_c.sel(pfull=850,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax11,transform=ccrs.PlateCarree(),vmin=-4,vmax=4,cmap='RdBu_r')

c4 = diff_ucomp_c_da.sel(pfull=10,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax2,transform=ccrs.PlateCarree(),vmin=-7,vmax=7,cmap='RdBu_r')
c5 = diff_ucomp_c_da.sel(pfull=500,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax7,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')
c6 = diff_ucomp_c_da.sel(pfull=850,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax12,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')

diff_ucomp_c_q.sel(pfull=10,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax3,transform=ccrs.PlateCarree(),vmin=-7,vmax=7,cmap='RdBu_r')
diff_ucomp_c_q.sel(pfull=500,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax8,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')
diff_ucomp_c_q.sel(pfull=850,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax13,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')

diff_ucomp_c_dt.sel(pfull=10,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax4,transform=ccrs.PlateCarree(),vmin=-7,vmax=7,cmap='RdBu_r')
diff_ucomp_c_dt.sel(pfull=500,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax9,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')
diff_ucomp_c_dt.sel(pfull=850,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax14,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')

diff_ucomp_c_ns.sel(pfull=10,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax5,transform=ccrs.PlateCarree(),vmin=-7,vmax=7,cmap='RdBu_r')
diff_ucomp_c_ns.sel(pfull=500,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax10,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')
diff_ucomp_c_ns.sel(pfull=850,method='nearest').plot.contourf(levels=21,add_colorbar=False,ax=ax15,transform=ccrs.PlateCarree(),vmin=-3,vmax=3,cmap='RdBu_r')

ax2.scatter(x=da_10_pval_lon,y=da_10_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax3.scatter(q_10_pval_lon,q_10_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax4.scatter(dt_10_pval_lon,dt_10_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax5.scatter(ns_10_pval_lon,ns_10_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())

ax7.scatter(da_500_pval_lon,da_500_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax8.scatter(q_500_pval_lon,q_500_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax9.scatter(dt_500_pval_lon,dt_500_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax10.scatter(ns_500_pval_lon,ns_500_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())

ax12.scatter(da_850_pval_lon,da_850_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax13.scatter(q_850_pval_lon,q_850_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax14.scatter(dt_850_pval_lon,dt_850_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())
ax15.scatter(ns_850_pval_lon,ns_850_pval_lat,color='k',s=2,marker='.',transform=ccrs.PlateCarree())

c_pval_10.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax1,transform=ccrs.PlateCarree())
c_pval_500.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax6,transform=ccrs.PlateCarree())
c_pval_850.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax11,transform=ccrs.PlateCarree())

#da_pval_10.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax2,transform=ccrs.PlateCarree())
#da_pval_500.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax7,transform=ccrs.PlateCarree())
#da_pval_850.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax12,transform=ccrs.PlateCarree())
#
#q_pval_10.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax3,transform=ccrs.PlateCarree())
#q_pval_500.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax8,transform=ccrs.PlateCarree())
#q_pval_850.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax13,transform=ccrs.PlateCarree())
#
#dt_pval_10.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax4,transform=ccrs.PlateCarree())
#dt_pval_500.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax9,transform=ccrs.PlateCarree())
#dt_pval_850.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax14,transform=ccrs.PlateCarree())
#
#ns_pval_10.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax5,transform=ccrs.PlateCarree())
#ns_pval_500.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax10,transform=ccrs.PlateCarree())
#ns_pval_850.plot.contourf(colors='none',hatches=['...'],alpha=0,add_colorbar=False,levels=2,ax=ax15,transform=ccrs.PlateCarree())


ax1.set_title('Control',fontsize =14)
ax2.set_title('Double All - Control',fontsize =14)
ax3.set_title('Quadruple - Control',fontsize =14)
ax4.set_title('Double Tropo - Control',fontsize =14)
ax5.set_title('No Strat - Control',fontsize =14)
ax6.set_title('')
ax7.set_title('')
ax8.set_title('')
ax9.set_title('')
ax10.set_title('')
ax11.set_title('')
ax12.set_title('')
ax13.set_title('')
ax14.set_title('')
ax15.set_title('')

ax1.annotate('10hPa',xy=(0,0), xytext=(-125, 0),textcoords='offset points', fontsize=14,size='large')
ax6.annotate('500hPa',xy=(0,0), xytext=(-125, 0),textcoords='offset points', fontsize=14,size='large')
ax11.annotate('850hPa',xy=(0,0), xytext=(-125, 0),textcoords='offset points', fontsize=14,size='large')

p0 = ax1.get_position().get_points().flatten()
p1 = ax6.get_position().get_points().flatten()
p2 = ax11.get_position().get_points().flatten()

p3 = ax2.get_position().get_points().flatten()
p4 = ax7.get_position().get_points().flatten()
p5 = ax12.get_position().get_points().flatten()

p6 = ax5.get_position().get_points().flatten()
p7 = ax10.get_position().get_points().flatten()
p8 = ax15.get_position().get_points().flatten()

plt.tight_layout()

#Set colorbars add_axes([left,bottom,width,height])
cbar_ax = fig.add_axes([p0[0]-0.05,0.68,0.16,0.03])
cb1 = plt.colorbar(c1, orientation='horizontal',cax=cbar_ax,label='')

cbar_ax = fig.add_axes([p1[0]-0.05,0.35,0.16,0.03])
cb2 = plt.colorbar(c2, orientation='horizontal',cax=cbar_ax,label='')

cbar_ax = fig.add_axes([p2[0]-0.05,0,0.16,0.03])
cb3 = plt.colorbar(c3, orientation='horizontal',cax=cbar_ax,label='ms$^{-1}$').set_label(label='ms$^{-1}$',size=12)

cbar_ax = fig.add_axes([p3[0],0.68,0.7,0.03])#1, 0.15, 0.05, 0.7])
cb1 = plt.colorbar(c4, orientation='horizontal',cax=cbar_ax,label='')

cbar_ax = fig.add_axes([p4[0],0.35,0.7,0.03])
cb2 = plt.colorbar(c5, orientation='horizontal',cax=cbar_ax,label='')

cbar_ax = fig.add_axes([p5[0],0,0.7,0.03])
cb3 = plt.colorbar(c6, orientation='horizontal',cax=cbar_ax,label='ms$^{-1}$').set_label(label='ms$^{-1}$',size=12)

plt.savefig("Fig5/zonal_global_uwinds_3deg_minus_present_minus_control_NA_maps_djf.png",bbox_inches = "tight")
plt.show()


