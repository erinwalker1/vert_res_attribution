# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:07:12 2021
Updated on Wed Jul 21 11:42:00 2021
@author: Erin Walker
"""

##Import modules and packages 
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter, FormatStrFormatter, ScalarFormatter
from cftime import utime
import pandas as pd
from scipy.stats import ttest_ind,ttest_1samp
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy import stats
import matplotlib.dates as mdates

print('Opening Files')

#######################################################
## Polar Cap PV at 10hPa for ERA5 2013/14 timeseries ##
#######################################################

###filepaths
PV_filepath = "" #create pathway to PV data
uwind_filepath = "" #create pathway to uwind data
era5_1979_2020_clim = PV_filepath + "era5_lait_scaled_1979_2019_clim_djf_PV.nc"
era5_13_14 = PV_filepath + "era5_lait_scaled_2013_2014_djf_PV.nc"

###Load ERA5 1979-2020 DJF climatology PV
era5_1979_2020_clim = xr.open_dataset(era5_1979_2020_clim)
era5_1979_2020_clim = era5_1979_2020_clim.PV
###Load ERA5 2013/2014 DJF lait scaled data
era5_13_14 = xr.open_dataset(era5_13_14)
era5_13_14 = era5_13_14.PV.sel(lat=slice(0,90))
###Load ERA5 2013/2014 uwind
era5_uwind = xr.open_dataset(uwind_filepath + "era5_2013_2014_djf_uwinds_60N_10hPa_anom_from_1979_2018.nc")
era5_uwind = era5_uwind.u

###Interpolate Lat to same as ERA5 2013/2014
era5_1979_2020_clim = era5_1979_2020_clim.interp({'lat':era5_13_14.lat,'lon':era5_13_14.lon})

###Calculate the difference between climatology and 2013/2014 djf
era5_PV_anom = era5_13_14 - era5_1979_2020_clim

### Create dates for DJF1314 for xaxis
days = ['15-12-2013','01-01-2014','15-01-2014','01-02-2014','15-02-2014']

###Select 60N-90N  10hPa for Polar Cap PV
era5_pc = era5_13_14.sel(pfull= 10, lat= slice(59,90)).mean(('lat','lon'))
era5_PC_PV_anom = era5_PV_anom.sel(pfull=10,lat=slice(59,90)).mean(('lat','lon'))

###Select 60N 10hPa for uwind
era5_10hPa = era5_uwind.sel(level=10, latitude=60,method='nearest').sel(time=slice('2013-12-01','2014-02-28')).mean('longitude')

### PLOT
fig ,ax = plt.subplots(ncols=1,nrows=1,figsize=((12,6)),sharex=True,sharey=True)

ln1 = era5_PC_PV_anom.plot(ax=ax,color='k',label='PV')

ax2 = ax.twinx()
ln2 =era5_10hPa.plot(ax=ax2,color='k',linestyle='--',label='Uwind')

ax.set_xticklabels(days,ha='center',rotation=0)
ax.tick_params(axis='x',labelsize=14,rotation=0,labelbottom=True)
ax.tick_params(axis='y',labelsize=14)
ax2.tick_params(axis='y',labelsize=14)
ax.set_xlim(era5_pc[0].time.values,era5_pc[-1].time.values)

ax.axhline(y=0,color='grey')

ax.set_title('Polar Cap (60-90$^\circ$N, 10hPa) PV and Uwind (60$^\circ$N, 10hPa) anomalies',fontsize=14)
ax2.set_title('')

ax.set_xlabel('')
ax.set_ylabel('PVU, 10$^{-6}m^{2}s^{-1}Kkg^{-1}$',fontsize=14)
ax2.set_ylabel('U wind, ms$^{-1}$',fontsize=14)

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0,prop={'size': 12})

### Make zero cross at same point in figure
ax1_ylims = ax.axes.get_ylim()           # Find y-axis limits set by the plotter
ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  # Calculate ratio of lowest limit to highest limit

ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit


# If the plot limits ratio of plot 1 is smaller than plot 2, the first data set has
# a wider range range than the second data set. Calculate a new low limit for the
# second data set to obtain a similar ratio to the first data set.
# Else, do it the other way around

if ax1_yratio < ax2_yratio: 
    ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
else:
    ax.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

#Option to format dates
#dtFmt = mdates.DateFormatter('%d-%m-%Y') # define the formatting
#plt.gca().xaxis.set_major_formatter(dtFmt)

plt.savefig('Fig_1a\\era5_polar_cap_PV_60N10hpa_uwinds_timeseries.png')
plt.show()

#############################
## ERA5 Wind Speed over UK ##
#############################

u10_filepath = "" #Insert path to u10 wind data
v10_filepath = "" #Insert path to v10 wind data

era5_u10wind = xr.open_dataset(u10_filepath + "era5_10m_uwind_1979_2021_djf_uk_daily.nc")
era5_v10wind = xr.open_dataset(v10_filepath + "era5_10m_vwind_1979_2021_djf_uk_daily.nc")
era5_u10wind = era5_u10wind.u10[:,0]
era5_v10wind = era5_v10wind.v10[:,0]

### Select southern UK
era5_suk_uwind_79_20 = era5_u10wind.sel(latitude=slice(52,50),longitude=slice(-6.,2.),time=slice('1979-12-01','2018-02-28'))
era5_suk_vwind_79_20 = era5_v10wind.sel(latitude=slice(52,50),longitude=slice(-6.,2.),time=slice('1979-12-01','2018-02-28'))

era5_suk_uwind_13_14 = era5_u10wind.sel(latitude=slice(52,50),longitude=slice(-6.,2.),time=slice('2013-12-01','2014-02-28'))
era5_suk_vwind_13_14 = era5_v10wind.sel(latitude=slice(52,50),longitude=slice(-6.,2.),time=slice('2013-12-01','2014-02-28'))

### Calculate windspeed
era5_suk_ws_79_20 = np.sqrt(era5_suk_uwind_79_20**2+era5_suk_vwind_79_20**2)
era5_suk_ws_13_14 = np.sqrt(era5_suk_uwind_13_14**2+era5_suk_vwind_13_14**2)

era5_suk_ws_79_20 = pd.DataFrame(era5_suk_ws_79_20.mean(('latitude','longitude')))
groups = era5_suk_ws_79_20.groupby(era5_suk_ws_79_20.index // 90)
size_index = groups.size().index

step = np.arange(0,3520,90)
daysi = (14,31,45,62,76)

### Plot
fig ,ax = plt.subplots(ncols=1,nrows=1,figsize=((12,6)),sharex=True,sharey=True)
for i in range(0,len(step)-1):
    group_x = groups.get_group(size_index[i])
    group_x.index = np.arange(0,90)
    z = ax.plot(group_x,color='grey',label='1979-2018',linewidth=2,alpha=0.2)

ln1 = ax.plot(np.arange(0,90),era5_suk_ws_13_14.mean(('latitude','longitude')),color='k',linewidth=2,label='2013/2014')  
 
ax.set_ylim(0,14)
ax.set_xticks(daysi)
ax.set_xticklabels(days,ha = 'center', rotation=0)
ax.tick_params(axis='x',labelsize=14,rotation=0,labelbottom=True)
ax.tick_params(axis='y',labelsize=14)
ax.set_xlim(0,89)

ax.set_title('Southern UK 10m Windspeeds',fontsize=14)

ax.set_xlabel('')
ax.set_ylabel('Windspeed, ms$^{-1}$',fontsize=14)

##DATES of major storms
dos = ['5Dec','18/19 Dec','23/24 Dec','26/27Dec','30/31 Dec',
       '3Jan','5Jan', '25/26 Jan','31/1Feb',
       ' 4/5 Feb',' 8/9 Feb',' 12 Feb',' 14/15 Feb']
#Dec
plt.axvline(x=4,color='r',alpha=0.5,linewidth=4)
plt.axvspan(17,18,color='r',alpha=0.5,lw=2)
plt.axvspan(22,23,color='r',alpha=0.5,lw=2)
plt.axvspan(25,26,color='r',alpha=0.5,lw=2)
plt.axvspan(29,30,color='r',alpha=0.5,lw=2)
#Jan
plt.axvline(x=33,color='r',alpha=0.5,linewidth=4)
plt.axvline(x=35,color='r',alpha=0.5,linewidth=4)
plt.axvspan(55,56,color='r',alpha=0.5,lw=2)
plt.axvspan(61,62,color='r',alpha=0.5,lw=2)
#Feb
plt.axvspan(65,66,color='r',alpha=0.5,lw=2)
plt.axvspan(69,70,color='r',alpha=0.5,lw=2)
plt.axvline(x=73,color='r',alpha=0.5,linewidth=4)
plt.axvspan(75,76,color='r',alpha=0.5,lw=2)

lns = z + ln1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')

plt.savefig('Fig_1c\\era5_suk_10m_windspeed_timeseries.png')
plt.show()



###########################################################
## ERA5 Precipitation amounts for southern UK timeseries ##
###########################################################

ppt_filepath = '' # Filepath to UK precipitation data
era5_ppt_79_21 = xr.open_dataset(ppt_filepath + 'era5_tp_1979_2021_djf_daily_uk.nc')
era5_ppt_79_21 = era5_ppt_79_21.tp[:,0] * 1000 #convert m to mm

### Select for southern UK and timeperiods
era5_ppt_79_21_s_uk = era5_ppt_79_21.sel(latitude=slice(52,50),longitude=slice(-6,2),time=slice('1979-12-01','2018-02-28'))
era5_ppt_13_14_s_uk = era5_ppt_79_21.sel(latitude=slice(52,50),longitude=slice(-6,2),time=slice('2013-12-01','2014-02-28'))

era5_ppt_79_21_s_uk = pd.DataFrame(era5_ppt_79_21_s_uk.mean(('latitude','longitude')))
groups = era5_ppt_79_21_s_uk.groupby(era5_ppt_79_21_s_uk.index // 90)
size_index = groups.size().index

step = np.arange(0,3558,90)

### Plot
fig ,ax = plt.subplots(ncols=1,nrows=1,figsize=((12,6)),sharex=True,sharey=True)
for i in range(0,len(step)-1):
    group_x = groups.get_group(size_index[i])
    group_x.index = np.arange(0,90)
    z = ax.plot(group_x,color='grey',linewidth=2,alpha=0.2,label='1979-2018')

ln1 = ax.plot(np.arange(0,90),era5_ppt_13_14_s_uk.mean(('latitude','longitude')),linewidth=2,color='k',label='2013/2014')

ax.set_xlabel('')
ax.set_ylabel('Daily Precipitation, mm/day',fontsize=14)
ax.set_xticks(daysi)
ax.set_xticklabels(days,ha = 'center', rotation=0)
ax.tick_params(axis='x',labelsize=14,rotation=0,labelbottom=True)
ax.tick_params(axis='y',labelsize=14)
ax.set_xlim(0,89)
ax.set_ylim(0,30)

ax.set_title('Southern UK Daily Precipitation',fontsize=14)

#Dec
plt.axvline(x=4,color='r',alpha=0.5,linewidth=4)
plt.axvspan(17,18,color='r',alpha=0.5,lw=2)
plt.axvspan(22,23,color='r',alpha=0.5,lw=2)
plt.axvspan(25,26,color='r',alpha=0.5,lw=2)
plt.axvspan(29,30,color='r',alpha=0.5,lw=2)
#Jan
plt.axvline(x=33,color='r',alpha=0.5,linewidth=4)
plt.axvline(x=35,color='r',alpha=0.5,linewidth=4)
plt.axvspan(55,56,color='r',alpha=0.5,lw=2)
plt.axvspan(61,62,color='r',alpha=0.5,lw=2)
#Feb
plt.axvspan(65,66,color='r',alpha=0.5,lw=2)
plt.axvspan(69,70,color='r',alpha=0.5,lw=2)
plt.axvline(x=73,color='r',alpha=0.5,linewidth=4)
plt.axvspan(75,76,color='r',alpha=0.5,lw=2)

lns = z + ln1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')

plt.savefig('Fig_1d\\era5_suk_daily_rainfall_timeseries.png')
plt.show()


#######################
## ERA5 UK Surf Temp ##
#######################

tmp_filepath = "" # Filepath to UK surface temperature file
era5_79_20 = xr.open_dataset(tmp_filepath + "era5_2mt_1979_2021_djf_daily_uk.nc",chunks={'time':30},decode_times=True)
era5_temp_79_20 = era5_79_20.t2m[:,0] -273.15 #Convert to degrees Celsius

### Select for southern UK and timeperiods
era5_79_20_temp_suk = era5_temp_79_20.sel(latitude=slice(52,50),longitude=slice(-6,2),time=slice('1979-12-01','2018-02-28'))
era5_13_14_temp_suk = era5_temp_79_20.sel(latitude=slice(52,50),longitude=slice(-6,2),time=slice('2013-12-01','2014-02-28'))

era5_temp_clim = pd.DataFrame(era5_79_20_temp_suk.mean(('latitude','longitude')))
groups = era5_temp_clim.groupby(era5_temp_clim.index // 90)
size_index = groups.size().index

step = np.arange(0,3520,90)

### Plot

fig ,ax = plt.subplots(ncols=1,nrows=1,figsize=((12,6)),sharex=True,sharey=True)
for i in range(0,len(step)-1):
    group_x = groups.get_group(size_index[i])
    group_x.index = np.arange(0,90)
    z = ax.plot(group_x,color='grey',label='1979-2018',linewidth=2,alpha=0.2)
   
ln1 = ax.plot(np.arange(0,90),era5_13_14_temp_suk.mean(('latitude','longitude')),color='k',linewidth=2,label='2013/2014')  

ax.set_xlabel('')
ax.set_ylabel('Temperature, $^\circ$C',fontsize=14)
ax.set_xticks(daysi)
ax.set_xticklabels(days,ha = 'center', rotation=0)
ax.tick_params(axis='x',labelsize=14,rotation=0,labelbottom=True)
ax.tick_params(axis='y',labelsize=14)
ax.set_xlim(0,89)
plt.axhline(y=0,color = 'grey')

ax.set_title('Southern UK 2m Temperautre',fontsize=14)

lns = z + ln1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

plt.savefig('Fig_1b\\era5_suk_2m_daily_temp_timeseries.png')
plt.show()

