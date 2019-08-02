"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: July 1st, 2019
Client: Hospital for Sick Children

Title: DI_Review_Main

Purpose:
-   Look for any trends in DI data that might be useful as it relates to staffing on a long term basis
-   Prepare visualizations for a 1 time presentation
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name

# Import Packages
import os
import numpy as np
import pandas as pd
import datetime as dt

# Import Custom Functions
from Data_Functions import test_func

# Import Custom Files
import Data_Functions

# Import DI Data
DI_2018_Q3 = pd.read_excel('/home/andrew/Public/DI/ED DI 2018 - Q3.xlsx')  # only 1 sheet
DI_2018_Q4 = pd.read_excel('/home/andrew/Public/DI/ED DI 2018 - Q4.xlsx')
DI_2019_Q1 = pd.read_excel('/home/andrew/Public/DI/ED DI 2019 - Q1.xlsx')
DI_2019_Q2 = pd.read_excel('/home/andrew/Public/DI/ED DI 2019 - Q2 20190621.xlsx')

# All results will go to output path
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/DI_Review_Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration

# Merge into 1 dataset
DI_Full = DI_2018_Q3.append([DI_2018_Q4,DI_2019_Q1,DI_2019_Q2],ignore_index=True)

# basic explorations
DI_Full.describe()
DI_Full.isna().sum()
DI_Full.shape
DI_Full.dtypes

# check for full row dupes
DI_dup_index = DI_Full.duplicated(keep='first')
sum(DI_dup_index) # no entire row dupes

sum(DI_Full.duplicated(subset=('Order ID'),keep='first'))  # no duped orders
sum(DI_Full.duplicated(subset=('MRN'),keep='first'))  # yes plenty of patient with multiple tests
sum(DI_Full.duplicated(subset=('MRN','Order Time','Procedure'),keep='first'))
# can have some with same order time, cannot have same procedure at same time

# Remove columns that

# Format date acceptably
DI_Full['Order Time'] = pd.to_datetime(DI_Full['Order Time'])

# Check in on the amount of people that actually match between the two, same MRN
All_Full = DI_Full.merge(ED_Reduced, how='inner', on = 'MRN' )
All_Full.shape #plenty because MRN is not distinct in either table
All_Full.dtypes


# Restrict the Joined to be based on Order time < 24 hours AFTER Arrived
Time_Difference = (All_Full['Order Time'] - All_Full['Arrived']).astype('timedelta64[s]')
Check1 = Time_Difference < (24*60*60)
Check2 = Time_Difference > 0
Logical_Time_Index = Check1&Check2

All_Restricted = All_Full[Logical_Time_Index]

test =

All_Restricted.shape
