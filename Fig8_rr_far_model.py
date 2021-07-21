# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:26:16 2021
Updated on Wed Jul 21 12:20:00 2021
@author: Erin Walker
"""

"""
 Adapted from Sarah Sparrow Code:
https://github.com/snsparrow/Attribution_workshop.git
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import random
from scipy import stats
import pandas as pd

"""
P = Present
F = 3Deg
C = Control
DA = Double All
DT = Double Tropopause
Q = Quadruple All
NS = No Stratopshere
"""
filepath = "" # Filepath to southern UK Isca DJF precipitation 
p_file_c = xr.open_dataset(filepath + "present_control_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_c = xr.open_dataset(filepath + "future_control_suk_pr_djf_all_ens.nc")

p_file_da = xr.open_dataset(filepath + "present_double_all_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_da = xr.open_dataset(filepath + "future_double_all_suk_pr_djf_all_ens.nc")

p_file_dt = xr.open_dataset(filepath + "present_double_tropo_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_dt = xr.open_dataset(filepath + "future_double_tropo_suk_pr_djf_all_ens.nc")

p_file_q = xr.open_dataset(filepath + "present_quadruple_all_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_q = xr.open_dataset(filepath + "future_quadruple_all_suk_pr_djf_all_ens.nc")

p_file_ns = xr.open_dataset(filepath + "present_no_strat_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_ns = xr.open_dataset(filepath + "future_no_strat_suk_pr_djf_all_ens.nc")

# Load precipitation 
p_c = p_file_c['precipitation'][:] 
p_da = p_file_da['precipitation'][:]
p_dt = p_file_dt['precipitation'][:]
p_q = p_file_q['precipitation'][:]
p_ns =p_file_ns['precipitation'][:]

f_c = f_file_c['precipitation'][:]
f_da = f_file_da['precipitation'][:]
f_dt = f_file_dt['precipitation'][:]
f_q = f_file_q['precipitation'][:]
f_ns = f_file_ns['precipitation'][:]

#Resample for 2 daily sum
p_c_2day = p_c.resample(time='2D').sum(dim='time')
p_da_2day = p_da.resample(time='2D').sum(dim='time')
p_dt_2day = p_dt.resample(time='2D').sum(dim='time')
p_q_2day = p_q.resample(time='2D').sum(dim='time')
p_ns_2day = p_ns.resample(time='2D').sum(dim='time')

f_c_2day = f_c.resample(time='2D').sum(dim='time')
f_da_2day = f_da.resample(time='2D').sum(dim='time')
f_dt_2day = f_dt.resample(time='2D').sum(dim='time')
f_q_2day = f_q.resample(time='2D').sum(dim='time')
f_ns_2day = f_ns.resample(time='2D').sum(dim='time')

#Calculate two daily max for each ensemble
p_twoday_max_c = []
p_twoday_max_da = []
p_twoday_max_dt = []
p_twoday_max_q = []
p_twoday_max_ns = []

f_twoday_max_c = []
f_twoday_max_da = []
f_twoday_max_dt = []
f_twoday_max_q = []
f_twoday_max_ns = []

for ens in range(0,len(p_c.ensemble)):
    p_twoday_max_c.append(p_c_2day.mean(('lat','lon'))[ens].max().values.item())
    p_twoday_max_da.append(p_da_2day.mean(('lat','lon'))[ens].max().values.item())
    p_twoday_max_dt.append(p_dt_2day.mean(('lat','lon'))[ens].max().values.item())
    p_twoday_max_q.append(p_q_2day.mean(('lat','lon'))[ens].max().values.item())
    p_twoday_max_ns.append(p_ns_2day.mean(('lat','lon'))[ens].max().values.item())
    f_twoday_max_c.append(f_c_2day.mean(('lat','lon'))[ens].max().values.item())
    f_twoday_max_da.append(f_da_2day.mean(('lat','lon'))[ens].max().values.item())
    f_twoday_max_dt.append(f_dt_2day.mean(('lat','lon'))[ens].max().values.item())
    f_twoday_max_q.append(f_q_2day.mean(('lat','lon'))[ens].max().values.item())
    f_twoday_max_ns.append(f_ns_2day.mean(('lat','lon'))[ens].max().values.item())

#calculate percentiles
x2 = np.array(p_twoday_max_c).flatten()
x2.sort()
i2_95 =  int(0.95 * len(x2))
i2_90 = int(0.90 * len(x2))
c2_95 = x2[i2_95]
c2_90 = x2[i2_90]

##################################
## Calculate Risk Ratio and FAR ##
##################################


def far(dataPres,data3Deg,threshold):
    CountPres=sum(np.array(dataPres)>threshold)
    Count3Deg=sum(np.array(data3Deg)>threshold)
    P_Pres=float(CountPres)/float(len(dataPres))
    P_3Deg=float(Count3Deg)/float(len(data3Deg))
    try:
        FAR= 1 - P_Pres/P_3Deg
    except:
        FAR=None
    return FAR

def probability_ratio(dataPres,data3Deg,threshold):
    CountPres=sum(np.array(dataPres)>threshold)
    Count3Deg=sum(np.array(data3Deg)>threshold)
    P_Pres=float(CountPres)/float(len(dataPres))
    P_3Deg=float(Count3Deg)/float(len(data3Deg))
    try:
        RR= P_3Deg/P_Pres
    except:
        RR=None
    return RR


def calc_PR_conf(BootPres,Boot3Deg,percentile,threshold,bsn=10000):
    PR=[]
    for ib in range(0,bsn):
        PR.append(probability_ratio(BootPres[ib,:].flatten(),Boot3Deg[ib,:].flatten(),threshold))
    return PR

def calc_bootstrap_ensemble(em, direction="ascending", bsn=1e5, slen=0):
        # bsn = boot strap number, number of times to resample the distribution
        ey_data = np.array(em).flatten()
        if slen==0:
                slen=ey_data.shape[0]
        print(slen)
        # create the store
        sample_store = np.zeros((int(bsn), int(slen)), 'f')
        # do the resampling
        for s in range(0, int(bsn)):
                t_data = np.zeros((slen), 'f')
                for y in range(0, slen):
                        x = random.uniform(0, slen)
                        t_data[y] = ey_data[int(x)]
                t_data.sort()
                # reverse if necessary
                if direction == "descending":
                        t_data = t_data[::-1]
                sample_store[s] = t_data
        return sample_store

def percentiles(data):
    """ Calculate the 10th, 90th, 95th and 99th Perctile
    """
    mean = np.array(data).mean()
    std = np.array(data).std()
    np.array(data).sort()  
    i_5 = int(0.05 * len(data))
    i_50 = int(0.50 * len(data))
    i_95 = int(0.95 * len(data))
    p_5 = data[i_5]
    p_50 = data[i_50]
    p_95 = data[i_95]
    return p_5, p_50,p_95


################################
### WORKING = ALL ENSEMBLES ####
################################
c_RR = list()
c_FAR = list()

da_RR = list()
da_FAR = list()

q_RR = list()
q_FAR = list()

dt_RR = list()
dt_FAR = list()

ns_RR = list()
ns_FAR = list()


for i in np.arange(22,32,0.5): 
    c_rr = probability_ratio(p_twoday_max_c,f_twoday_max_c,i)
    da_rr = probability_ratio(p_twoday_max_da,f_twoday_max_da,i)
    dt_rr = probability_ratio(p_twoday_max_dt,f_twoday_max_dt,i)
    q_rr = probability_ratio(p_twoday_max_q,f_twoday_max_q,i)
    ns_rr = probability_ratio(p_twoday_max_ns,f_twoday_max_ns,i)

    c_far = far(p_twoday_max_c,f_twoday_max_c,i)
    da_far = far(p_twoday_max_da,f_twoday_max_da,i)
    dt_far = far(p_twoday_max_dt,f_twoday_max_dt,i)
    q_far = far(p_twoday_max_q,f_twoday_max_q,i)
    ns_far = far(p_twoday_max_ns,f_twoday_max_ns,i)
    
    c_RR.append(c_rr)
    c_FAR.append(c_far)
    
    da_RR.append(da_rr)
    da_FAR.append(da_far)

    dt_RR.append(dt_rr)
    dt_FAR.append(dt_far)

    q_RR.append(q_rr)
    q_FAR.append(q_far)

    ns_RR.append(ns_rr)
    ns_FAR.append(ns_far)



style=['-','--',':','-.','.-']

ppt = np.arange(22,32, 0.5)
fig, axes = plt.subplots(1,2,figsize=(20,6),sharex=False,sharey=False)

axes[0].plot(ppt,c_RR,style[0],label='Control',linewidth=3)
axes[0].plot(ppt,da_RR,style[1],label='Double All',linewidth=3)
axes[0].plot(ppt,q_RR,style[2],label='Quadruple All',linewidth=3)
axes[0].plot(ppt,dt_RR,style[3],label='Double Tropo',linewidth=3)
axes[0].plot(ppt,ns_RR,style[4],label='No Strat',linewidth=3,markersize=2)

axes[1].plot(ppt,c_FAR,style[0],label='Control',linewidth=3)
axes[1].plot(ppt,da_FAR,style[1],label='Double All',linewidth=3)
axes[1].plot(ppt,q_FAR,style[2],label='Quadruple All',linewidth=3)
axes[1].plot(ppt,dt_FAR,style[3],label='Double Tropo',linewidth=3)
axes[1].plot(ppt,ns_FAR,style[4],label='No Strat',linewidth=3,markersize=2)

axes[0].vlines(x=30.04,ymin=0,ymax=5)
axes[1].vlines(x=30.04,ymin=0,ymax=1)

axes[0].set_ylim(1,5)
axes[1].set_ylim(0,1)
axes[0].set_xlim(22,31)
axes[1].set_xlim(22,31)

axes[0].set_ylabel('RR',fontsize=14)
axes[1].set_ylabel('FAR',fontsize=14)
axes[0].set_xlabel('Precipitation, mm',fontsize=14)
axes[1].set_xlabel('Precipitation, mm',fontsize=14)
axes[0].tick_params(axis='both', labelsize=12)
axes[1].tick_params(axis='both', labelsize=12)
axes[0].legend()

plt.savefig("Fig8\RR_FAR_vs_precip.png", bbox_inches = "tight")
plt.show()

