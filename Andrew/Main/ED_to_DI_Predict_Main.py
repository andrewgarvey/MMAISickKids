"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: July 1st, 2019
Client: Hospital for Sick Children

Title: ED_Review_Main

Purpose:
-   Develop a model to predict DI demand using ED triage data, also try to add outside available factors
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name
## Edited to remove ED things and change

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import datetime as dt
import os

#Set dir
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/')

#Import Cleaned Datasets
ED_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/ED_Clean')
DI_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/DI_Clean')

ED_Clean.shape

# Check in on the amount of people that actually match between the two, same MRN
All_Clean = ED_Clean.merge(DI_Clean, how='left', on = 'MRN' )
All_Clean.shape #plenty because MRN is not distinct in either table,
All_Clean.dtypes

# Check in on the stats not being hilariously wrong particularly for the 3 key joining dates (arrived,order,discharge)
All_Clean.isna().sum() # because we have many ED visitors who went without tests , keeping those as it is a valid result

# Restrict the Joined rows to be based on order dates that are acceptable (arrived -> order -> discharge)
# Or just no tests for that visit is ok too
Arrived_before_Order = All_Clean['Arrived'] < All_Clean['Order Time']
Discharge_after_Order = All_Clean['Order Time'] < All_Clean['Disch Date/Time']
No_Tests_Performed = (All_Clean['Order Time'].isna())

Logical_Timing_Index = (Arrived_before_Order & Discharge_after_Order) | No_Tests_Performed
All_Clean_Logical = All_Clean[Logical_Timing_Index]

All_Clean_Logical.shape

# Drop rows that we cannot possible have at the time this model aims to be used (nearly all of DI, some of ED)
# Later we might have to attempt to quantify how paths of people work out



