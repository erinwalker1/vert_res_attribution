# Vertical Resolution on extreme event attribution and climate projections

This directory contains the scripts to make the figures for Walker et al. 2021, "Increasing vertical resolution and the implications on extreme event attribution and climate projections", submitted to Quarterly Journal of the Royal Meteorological Society on 23-07-2021.

***

To access and use the data we recommend using xarray, numpy and matplotlib packages in Python. Further metadata are included in the individual scripts. Data can be found at:

***

The climate model used is Isca and can be found here: https://github.com/ExeClim/Isca 
The set up used the realistic earth configuration. Prescribed sea-surface temperatures and sea-ice concentrations can be found at: 
https://www.climateprediction.net/
and:
https://www.happimip.org/happi_data/

***

## Script names and description

Script names correspond to the figure numbers in the paper. 

**[Fig1_era5_figs.py](Fig1_era5_figs.py):**
DJF PV and zonal wind anomalies, 2m daily temperature from 1979-2018, daily total precipitation from 1979-2018, and 10m windspeeds from 1979-2018 for the southern UK.
  
**[Fig3_zonal_temp_vertical_profile.py](Fig3_zonal_temp_vertical_profile.py):**
DJF Zonal mean vertical temperature profile. Difference between 3Deg and Present experiment and the difference from the Control.

**[Fig4_zonal_winds_vertical_profile.py](Fig4_zonal_winds_vertical_profile.py):**
DJF Zonal mean zonal winds vertical profile. Difference between 3Deg and Present experiment and the difference from the Control.
  
**[Fig5_zonal_winds_maps.py](Fig5_zonal_winds_maps.py):**
DJF Zonal mean zonal winds over North Atlantic at 10, 500 and 850hPa. Difference between 3Deg and Present experiment and the difference from the Control.

**[Fig6_windshear_maps.py](Fig6_windshear_maps.py):**
DJF Windshear over North Atlantic. Difference between 3Deg and Present experiment and the difference from the Control.

**[Fig7_return_period.py](Fig7_return_period.py):**
Return period plot of maximum 2-daily precipitation, including GEV fit for Present and 3Deg experiments.
  
**[Fig8_rr_far_model.py](Fig8_rr_far_model.py):**
Plot varying thresholds of precipitation against the corresponding Risk Ratio (RR) and Fraction of Attributal Risk (FAR).
  
**[Fig9_nh_storm_numbers.py](Fig9_nh_storm_numbers.py):**
The number of Northern Hemisphere storms sorted by length of track in hours; 24, 48 and 72hrs. Substitude the NH data for North Atlantic data to create S3 Figure.

**[Fig10_nh_storm_intensity.py](Fig10_nh_storm_intensity.py):**
The intensity of Northern Hemisphere storms sorted by length of track in hours; 24, 48 and 72hrs. Substitude the NH data for North Atlantic data to create S3 Figure.

**[Sup_Fig1_zonal_temp_vertical_profile.py](Sup_Fig1_zonal_temp_vertical_profile.py):**
Zonal mean vertical temperature profile for individual months December, January and February. Difference between 3Deg and Present experiment and the difference from the Control.
  
**[Sup_Fig2_zonal_winds_vertical_profile.py](Sup_Fig2_zonal_winds_vertical_profile.py):**
Zonal mean zonal mean zonal winds vertical temperature profile for individual months December, January and February. Difference between 3Deg and Present experiment and the difference from the Control.
