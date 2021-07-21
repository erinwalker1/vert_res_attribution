# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:44:44 2021

@author: at18707
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import stats

p_file_c = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\present_control_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_c = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\future_control_suk_pr_djf_all_ens.nc")

p_file_da = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\present_double_all_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_da = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\future_double_all_suk_pr_djf_all_ens.nc")

p_file_dt = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\present_double_tropo_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_dt = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\future_double_tropo_suk_pr_djf_all_ens.nc")

p_file_q = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\present_quadruple_all_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_q = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\future_quadruple_all_suk_pr_djf_all_ens.nc")

p_file_ns = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\present_no_strat_suk_pr_djf_all_ens.nc",chunks={'time':30})
f_file_ns = xr.open_dataset("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\return_periods\\future_no_strat_suk_pr_djf_all_ens.nc")

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

p_twoday_max_c.sort()
p_twoday_max_da.sort()
p_twoday_max_dt.sort()
p_twoday_max_q.sort()
p_twoday_max_ns.sort()

f_twoday_max_c.sort()
f_twoday_max_da.sort()
f_twoday_max_dt.sort()
f_twoday_max_q.sort()
f_twoday_max_ns.sort()

def gev_fit_RR(Pres, Deg3):
    pres_params=stats.genextreme.fit(Pres)
    Deg3_params=stats.genextreme.fit(Deg3)
    T=np.r_[1:10000]*0.01
    rt=stats.genextreme.isf(1./T,pres_params[0],loc=pres_params[1],scale=pres_params[2])
    cdf=stats.genextreme.cdf(rt,pres_params[0],loc=pres_params[1],scale=pres_params[2])
    probPres=1-cdf
    cdf=stats.genextreme.cdf(rt,Deg3_params[0],loc=Deg3_params[1],scale=Deg3_params[2])
    prob3Deg=1-cdf
    RR = prob3Deg/probPres
    return rt, RR

def gev_fit_FAR(Pres, Deg3):
    pres_params=stats.genextreme.fit(Pres)
    Deg3_params=stats.genextreme.fit(Deg3)
    T=np.r_[1:10000]*0.01
    rt=stats.genextreme.isf(1./T,pres_params[0],loc=pres_params[1],scale=pres_params[2])
    cdf=stats.genextreme.cdf(rt,pres_params[0],loc=pres_params[1],scale=pres_params[2])
    probPres=1-cdf
    cdf=stats.genextreme.cdf(rt,Deg3_params[0],loc=Deg3_params[1],scale=Deg3_params[2])
    prob3Deg=1-cdf
    FAR= 1 - probPres/prob3Deg
    return rt, FAR

#RR
prt_c, pRR_c = gev_fit_RR(p_twoday_max_c,f_twoday_max_c)

prt_da, pRR_da = gev_fit_RR(p_twoday_max_da,f_twoday_max_da)

prt_dt, pRR_dt = gev_fit_RR(p_twoday_max_dt,f_twoday_max_dt)

prt_q, pRR_q = gev_fit_RR(p_twoday_max_q,f_twoday_max_q)

prt_ns, pRR_ns = gev_fit_RR(p_twoday_max_ns,f_twoday_max_ns)

#FAR
fart_c, FAR_c = gev_fit_FAR(p_twoday_max_c,f_twoday_max_c)

fart_da, FAR_da = gev_fit_FAR(p_twoday_max_da,f_twoday_max_da)

fart_dt, FAR_dt = gev_fit_FAR(p_twoday_max_dt,f_twoday_max_dt)

fart_q, FAR_q = gev_fit_FAR(p_twoday_max_q,f_twoday_max_q)

fart_ns, FAR_ns = gev_fit_FAR(p_twoday_max_ns,f_twoday_max_ns)

#Plotting
style=['-','--',':','-.','.-']

fig, axes = plt.subplots(1,2,figsize=(20,6),sharex=False,sharey=False)

axes[0].plot(prt_c,pRR_c,style[0],label='Control',linewidth=3)
axes[0].plot(prt_da,pRR_da,style[1],label='Double All',linewidth=3)
axes[0].plot(prt_q,pRR_q,style[2],label='Quadruple All',linewidth=3)
axes[0].plot(prt_dt,pRR_dt,style[3],label='Double Tropo',linewidth=3)
axes[0].plot(prt_ns,pRR_ns,style[4],label='No Strat',linewidth=3,markersize=2)

axes[1].plot(fart_c,FAR_c,style[0],label='Control',linewidth=3)
axes[1].plot(fart_da,FAR_da,style[1],label='Double All',linewidth=3)
axes[1].plot(fart_q,FAR_q,style[2],label='Quadruple All',linewidth=3)
axes[1].plot(fart_dt,FAR_dt,style[3],label='Double Tropo',linewidth=3)
axes[1].plot(fart_ns,FAR_ns,style[4],label='No Strat',linewidth=3,markersize=2)

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

#plt.savefig("C:\\Users\\at18707\\OneDrive - University of Bristol\\Python\\Isca\\3deg\\50_ens\\gev_RR_FAR_vs_precip.png", bbox_inches = "tight")

plt.show()



###############################################


def probability_ratio(dataPres,data3Deg,threshold):
    CountPres=sum(np.array(dataPres)>threshold)
    Count3Deg=sum(np.array(data3Deg)>threshold)
    P_Pres=float(CountPres)/float(len(dataPres))
    P_3Deg=float(Count3Deg)/float(len(data3Deg))
    try:
        RR= P_3Deg/P_Pres
        #RR= 1 - P_Pres/P_3Deg
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
#BootPres = calc_bootstrap_ensemble(p_ns.mean(('lat','lon')),direction="descending",bsn=10000, slen=0)
#Boot3Deg = calc_bootstrap_ensemble(f_ns.mean(('lat','lon')),direction="descending",bsn=10000, slen=0)

BootPres = calc_bootstrap_ensemble(prt,direction="descending",bsn=10000, slen=0)
Boot3Deg = calc_bootstrap_ensemble(frt,direction="descending",bsn=10000, slen=0)

PR = calc_PR_conf(BootPres,Boot3Deg,[5,50,95],30.04)
#30.046202551350497
len(PR)

PR.sort()
p_5, p_50,p_95 = percentiles(PR)

print(p_50, p_5, p_95)


####################
## Varying Precip ##
####################

c_gev_rr = list()
c_data_rr = list()

for i in np.arange(22,32,0.5): 
    c_rr_gev = probability_ratio(prt_ns,frt_ns,i)
    c_gev_rr.append(c_rr_gev)
    c_rr = probability_ratio(p_twoday_max_ns,f_twoday_max_ns,i)
    c_data_rr.append(c_rr)
    #da_rr = probability_ratio(p_twoday_max_da,f_twoday_max_da,i)
    #dt_rr = probability_ratio(p_twoday_max_dt,f_twoday_max_dt,i)
    #q_rr = probability_ratio(p_twoday_max_q,f_twoday_max_q,i)
    #ns_rr = probability_ratio(p_twoday_max_ns,f_twoday_max_ns,i)
       
    #print('Control RR: ', c_rr)
    #print('Double All RR: ', da_rr)
    #print('Quadruple All RR: ', q_rr)
    #print('Double Tropo RR: ', dt_rr)
    #print('No Strat RR: ', ns_rr)

plt.plot(c_data_rr,label='data rr')
plt.plot(c_gev_rr,label='gev rr')
plt.legend()