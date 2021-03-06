# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:36:38 2021
Updated on Wed Jul 21 12:14:00 2021
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
print(c2_90)

#Now calc return plots 

def calc_return_times(em, direction="ascending", period=1):
	ey_data = np.array(em).flatten()
	ey_data.sort()
	# reverse if necessary
	if direction == "descending":	# being beneath a threshold value
		ey_data = ey_data[::-1]
	# create the n_ens / rank_data
	val = float(len(ey_data)) * 1.0/period
	end = float(len(ey_data)) * 1.0/period
	start = 1.0
	step = (end - start) / (len(ey_data)-1)
	ranks = [x*step+start for x in range(0, len(ey_data))]
	ex_data = val / np.array(ranks, dtype=np.float32)
	return ey_data, ex_data

def calc_return_time_confidences(em, direction="ascending", c=[0.05, 0.95], bsn=1e5):
	# c = confidence intervals (percentiles) to calculate
	# bsn = boot strap number, number of times to resample the distribution
	ey_data = np.array(em).flatten()
	# create the store
	sample_store = np.zeros((int(bsn), ey_data.shape[0]), 'f')
	# do the resampling
	for s in range(0, int(bsn)):
		t_data = np.zeros((ey_data.shape[0]), 'f')
		for y in range(0, ey_data.shape[0]):
			x = random.uniform(0, ey_data.shape[0])
			t_data[y] = ey_data[int(x)]
		t_data.sort()
		# reverse if necessary
		if direction == "descending":
			t_data = t_data[::-1]
		sample_store[s] = t_data
	# now for each confidence interval find the  value at the percentile
	conf_inter = np.zeros((len(c), ey_data.shape[0]), 'f')
	for c0 in range(0, len(c)):
		for y in range(0, ey_data.shape[0]):
			data_slice = sample_store[:,y]
			conf_inter[c0,y] = stats.scoreatpercentile(data_slice, c[c0]*100)
	return conf_inter

def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def plot_return_time(exp,variable,dataAct,dataNat,threshold,dirn,fig,row,col):
    # Setup the plot parameters and axis limits
    ax = plt.subplot2grid((2,3),(row,col))
    plt.title(exp,fontsize=16)
    ax.set_ylabel(variable,fontsize=16)
    ax.set_xlabel("Chance of event occurring in a given year",fontsize=16)
    plt.setp(ax.get_xticklabels(),fontsize=16)
    plt.setp(ax.get_yticklabels(),fontsize=16)
    ax.set_xlim(1,1e2)
    ax.set_ylim(16,45)
    # Plot the return time for the historical and historicalNat simulations
    plot_rt(dataAct,["royalblue","cornflowerblue","mediumblue"],"Present",ax,"both",dirn,threshold,"",'--')
    plot_rt(dataNat,["orange","gold","darkorange"],"3Deg",ax,"both",dirn,threshold,"Threshold",'--')
    labels=['','1/1','1/10','1/100']
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(),fontsize=16)
    plt.setp(ax.get_yticklabels(),fontsize=16)
    ll=ax.legend(loc="upper left",prop={"size": 14},fancybox=True,numpoints=1)
    

def plot_rt(data,cols,plabel,ax,errb,dirn,threshold,tlabel,tstyle):
    # Plot the return times with bootstrap 9-95% confience intervals.
    # Calculate the return times
    y_data_all, x_data_all = calc_return_times(data,direction=dirn,period=1)    
    # Calculate the bootstrap confidences in both the x and y directions
    conf_all = calc_return_time_confidences(data,direction=dirn,bsn=1000)
    conf_all_x = calc_return_time_confidences(x_data_all,direction="descending",bsn=1000)
    # Plot  the return time curve
    l1=ax.semilogx(x_data_all,y_data_all, marker='o',markersize=4,
                       linestyle='None',mec=cols[0],mfc=cols[0],
                       color=cols[0],fillstyle='full',
                       label=plabel,zorder=2)    
    conf_all_5=conf_all[0,:].squeeze()
    conf_all_95=conf_all[1,:].squeeze()
    conf_all_x_5=conf_all_x[0,:].squeeze()
    conf_all_x_95=conf_all_x[1,:].squeeze()
    #ax.grid(b=True,which='major')
    #ax.grid(b=True, which='minor',linestyle='--')
    # Plot the error bars onn the return times
    #if errb=="both":
    #	cl0=ax.fill_between(x_data_all,conf_all_5,conf_all_95,color=cols[1],alpha=0.2,linewidth=1.,zorder=0)
    if errb=="magnitude" or errb=="both":
    	cl1=ax.semilogx([x_data_all,x_data_all],[conf_all_5,conf_all_95],color=cols[1],linewidth=1.,zorder=1)
    if errb=="return_time" or errb=="both":
	    cl2=ax.semilogx([conf_all_x_5,conf_all_x_95],[y_data_all,y_data_all],color=cols[1],linewidth=1.,zorder=1)
    # Calculate GEV fit to data 
    shape,loc,scale=stats.genextreme.fit(data)
    T=np.r_[1:10000]*0.01
    # Perform K-S test and print goodness of fit parameters
    D, p = stats.kstest(np.array(data).flatten(), 'genextreme', args=(shape, loc,scale));
    print(plabel+' GEV fit, K-S test parameters p: '+str(p)+" D: "+str(D))
    # Plot fit line
    if dirn=="ascending":
        rt=stats.genextreme.ppf(1./T,shape,loc=loc,scale=scale)
    else:
        rt=stats.genextreme.isf(1./T,shape,loc=loc,scale=scale)
    l1=ax.semilogx(T,rt,color=cols[2],label="GEV fit")
    
    # Highlight where the threshold is and where the return time curve bisects this threshold
    xmin,xmax=ax.get_xlim()
    ymin,ymax=ax.get_ylim()
    ax.semilogx([xmin,xmax],[threshold,threshold],color="Grey",linestyle=tstyle,linewidth=2.5,label=tlabel, zorder=2)
    nidx=find_nearest(y_data_all,threshold)
    nidx1=np.where(rt>=threshold)[0][0]
    ax.axvspan(conf_all_x_5[nidx],conf_all_x_95[nidx],ymin=0,ymax=(threshold-ymin)/(ymax-ymin),facecolor=cols[1],edgecolor=cols[2],linewidth=1.5,alpha=0.1,zorder=0)
    #ax.axvline(x_data_all[nidx],ymin=0,ymax=(threshold-ymin)/(ymax-ymin),linestyle=tstyle,color=cols[0],linewidth=1.5,zorder=0)
    ax.axvline(T[nidx1],ymin=0,ymax=(threshold-ymin)/(ymax-ymin),linestyle=tstyle,color=cols[0],linewidth=1.5,zorder=0)

   # print(plabel+" return time with 5-95% range: "+str(x_data_all[nidx])+" ("+str(conf_all_x_5[nidx])+" "+str(conf_all_x_95[nidx])+")")
    print(plabel+" return time with 5-95% range: "+str(T[nidx1])+" ("+str(conf_all_x_5[nidx])+" "+str(conf_all_x_95[nidx])+")")
   
    # Return the fit parameters
    return [shape,loc,scale]

fig = plt.figure()
fig.set_size_inches(21,16)
#ymin = 16,40, legend is lower right
plot_return_time('Control',' Max 2-day Precipitaiton, mm',p_twoday_max_c,f_twoday_max_c,30.04,"descending",fig,0,0)
plot_return_time('Double All','Max 2-day Precipitaiton, mm',p_twoday_max_da,f_twoday_max_da,30.04,"descending",fig,0,1)
plot_return_time('Quadruple All','Max 2-day Precipitaiton, mm',p_twoday_max_q,f_twoday_max_q,30.04,"descending",fig,0,2)
plot_return_time('Double Tropo','Max 2-day Precipitaiton, mm',p_twoday_max_dt,f_twoday_max_dt,30.04,"descending",fig,1,0)
plot_return_time('No Strat','Max 2-day Precipitaiton, mm',p_twoday_max_ns,f_twoday_max_ns,30.04,"descending",fig,1,1)

plt.savefig("Fig7/suk_djf_2day_max_pr_return_period_90th_percentile.png",dpi=600)
plt.show()


