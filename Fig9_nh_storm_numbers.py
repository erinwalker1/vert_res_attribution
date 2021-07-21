# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:12:11 2021
Updated on Wed Jul 21 12:25:00 2021
@author: Erin Walker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
import scipy.stats as sp
import seaborn as sns

filepath = "" # filepath to storm track numbers
NH_pres = pd.ExcelFile(filepath + "isca_vert_res_exp_storm_track_numbers_NA.xlsx")

short = pd.read_excel(NH_pres,'24hrs',index_col=0,header=[0,1])
medium = pd.read_excel(NH_pres,'48hrs',index_col=0,header=[0,1])
long = pd.read_excel(NH_pres,'72hrs',index_col=0,header=[0,1])

short_nh =  short.unstack(level=0).reset_index(level=2,drop=True).reset_index(name='data')                
medium_nh =  medium.unstack(level=0).reset_index(level=2,drop=True).reset_index(name='data')                
long_nh =  long.unstack(level=0).reset_index(level=2,drop=True).reset_index(name='data')                

fig, axes = plt.subplots(1,3,figsize=(20,6),sharex=False,sharey=False)

a=sns.boxplot(x='Experiment', y='data', hue="Scenario",data=short_nh,ax=axes[0],showfliers=False)
ax = sns.stripplot(x='Experiment',y='data',hue='Scenario',data=short_nh,dodge=True,alpha=1,ax=axes[0])

b=sns.boxplot(x='Experiment', y='data', hue="Scenario",data=medium_nh,ax=axes[1],showfliers=False)
ax = sns.stripplot(x='Experiment',y='data',hue='Scenario',data=medium_nh,dodge=True,alpha=1,ax=axes[1])
ax.get_legend().remove()

c=sns.boxplot(x='Experiment', y='data', hue="Scenario",data=long_nh,ax=axes[2],showfliers=False)
ax = sns.stripplot(x='Experiment',y='data',hue='Scenario',data=long_nh,dodge=True,alpha=1,ax=axes[2])
ax.get_legend().remove()

axes[0].set_title('24hrs',fontsize=12)
axes[0].set_xlabel('')
axes[0].set_ylabel('Number of tracks',fontsize=12)

axes[1].set_title('48hrs',fontsize=12)
axes[1].set_xlabel('')
axes[1].set_ylabel('')

axes[2].set_title('72hrs',fontsize=12)
axes[2].set_xlabel('')
axes[2].set_ylabel('')

a.tick_params(labelsize=12)
b.tick_params(labelsize=12)
c.tick_params(labelsize=12)
axes[0].set_xticklabels(a.get_xticklabels(),rotation=20)
axes[1].set_xticklabels(b.get_xticklabels(),rotation=20)
axes[2].set_xticklabels(c.get_xticklabels(),rotation=20)

handles, labels = axes[2].get_legend_handles_labels()
axes[0].legend([handles[0],handles[1]],[labels[0],labels[1]],loc='upper left')#, bbox_to_anchor=(1,0.5))

plt.tight_layout()

plt.savefig('Fig9\isca_experiments_storm_track_numbers_NA_DJF_boxplot.png')

plt.show()

