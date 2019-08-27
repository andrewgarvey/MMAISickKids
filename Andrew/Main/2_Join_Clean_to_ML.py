"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: Aug 23, 2019
Client: Hospital for Sick Children

Title: ED_DI_to_ML

Purpose:
-   Turn cleaned data into usable ml data
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import datetime as dt
import os
from pandasql import sqldf


#Set dir
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/')
#------------------------------------------------------------------------------------------------------------------------
# Final Cleaning towards ML usable Model DF

#Import Cleaned Datasets
ED_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ED_Clean.csv')
DI_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/DI_Clean.csv')

ED_Clean.shape

# Restrict the Joined rows to be based on order dates that are acceptable (arrived -> order -> discharge)
# Or just no tests for that visit is ok too, this will show up as null
# DI Timeframe entirely encompasses ED, so if they got a test they should be here.

# Could not find a clean way to do this that wouldn't take a bunch of extra work in python, using sql
pysqldf = lambda q: sqldf(q, globals())  # Imports all current global variables to be able to be used in sql as df

All_Clean = pysqldf("SELECT * FROM ED_Clean AS e "
                    "LEFT JOIN DI_Clean AS d " #  left join because NO tests is a valid answer to incoming patient
                    "ON e.MRN = d.MRN "  #  same person
                    "AND e.Arrived < d.[Order Time]"  #  arrived before order
                    "AND e.[Disch Date/Time] > d.[Order Time]")  # discharged after order

All_Clean.isna().sum()

# Drop rows that we cannot possibly have AT THE TIME this Model aims to be used (nearly all of DI, some of ED)
All_Clean_Reduced = All_Clean.drop(['ED Completed Length of Stay (Minutes)', 'Roomed', 'Disch Date/Time', 'Dispo',
                                    'Roomed to Discharge', 'Roomed to Discharge', 'Arrived to Discharge',
                                    'End Exam Time', 'Order Time', 'Finalized Time', 'Finalizing Physician', 'Order ID',
                                    'Order to Protocolled (min)', 'Protocolled to Begin (min)', 'Order to Begin (min)',
                                    'Begin to End (min)', 'End to Prelim (min)', 'End to Sign (min)',
                                    'Order to End (min)', 'Order to Sign (min)', 'Protocolling Instant', 'Procedure id',
                                    'Authorizing Provider id', 'Finalizing Physician id', 'Arrived to Roomed' ], axis=1)

# drop second mrn column
di_mrn = len(All_Clean_Reduced.columns) -2 #  second last column is dupe mrn
All_Clean_Reduced = All_Clean_Reduced.drop(All_Clean_Reduced.columns[di_mrn], axis=1)

# Arrived should focus hour of the day arrived, datetime format not likely useful for model
All_Clean_Reduced.dtypes
All_Clean_Reduced['Arrived'] = pd.to_datetime(All_Clean_Reduced['Arrived']).dt.hour  #make this a dummy variable

# Replace category nan with "none" text, (shows warning)
All_Clean_Reduced['Category id'].loc[All_Clean_Reduced['Category id'].isna()] = 'none'

# Aggregate by everything except category or just csn, make a delimited column for this, (takes a few minutes)
All_Clean_Condensed_orig = All_Clean_Reduced.groupby('CSN', as_index=False).agg(lambda x: ', '.join(set(x.astype(str))))
All_Clean_Condensed = All_Clean_Condensed_orig  # because it takes a while

# Dummy Variable all the things of relevance that should be converted to dummy variables
# not viable for CC, postal code, maybe later.
# Arrived(the hours one), day of arrival, province,

dummies = pd.get_dummies(All_Clean_Condensed['Province']).rename(columns=lambda x: 'Province_' + str(x))
All_Clean_Condensed = pd.concat([All_Clean_Condensed, dummies], axis=1)

dummies = pd.get_dummies(All_Clean_Condensed['Arrived']).rename(columns=lambda x: 'Arrived_Hour' + str(x))
All_Clean_Condensed = pd.concat([All_Clean_Condensed, dummies], axis=1)

dummies = pd.get_dummies(All_Clean_Condensed['Day of Arrival']).rename(columns=lambda x: 'Day_of_Arrival' + str(x))
All_Clean_Condensed = pd.concat([All_Clean_Condensed, dummies], axis=1)

dummies = pd.get_dummies(All_Clean_Condensed['Gender']).rename(columns=lambda x: 'Gender_' + str(x))
All_Clean_Condensed = pd.concat([All_Clean_Condensed, dummies], axis=1)

# Arrival Method simplified greatly , find the big ones , those get a 1/0 for containing
Arrival_Method_Options = All_Clean_Condensed.groupby('Arrival Method').count().sort_values('CSN',ascending = False)

"""
Biggest Options: 
Ambula  (covers ambulance/ambulatory)
Walk
Car
"""

All_Clean_Condensed['Method_Ambulance'] = (All_Clean_Condensed['Arrival Method'].str.contains('Ambula'))
All_Clean_Condensed['Method_Walk'] = (All_Clean_Condensed['Arrival Method'].str.contains('Walk'))
All_Clean_Condensed['Method_Car'] = (All_Clean_Condensed['Arrival Method'].str.contains('Car'))

## CC simplified Greatly, find big key words,
CC_Options = All_Clean_Condensed.groupby('CC').count().sort_values('CSN',ascending = False)
CC_Options = CC_Options.loc[CC_Options['CSN']>15]

# capture each index option that has more than 15 instances
cc_list =  CC_Options.index.values.astype(str)

# do a regex check for each of those columns
for x in cc_list:
    All_Clean_Condensed[x] = All_Clean_Condensed['CC'].str.contains(x)

## Complaint simplified Greatly, find big key words,
Complaint_Options = All_Clean_Condensed.groupby('ED Complaint').count().sort_values('CSN',ascending = False)
Complaint_Options = Complaint_Options.loc[Complaint_Options['CSN']>15]

# capture each index option that has more than 500 people
Complaint_list =  Complaint_Options.index.values.astype(str)

# do a regex check for each of those columns
for x in Complaint_list:
    All_Clean_Condensed[x] = All_Clean_Condensed['ED Complaint'].str.contains(x)

"""
Notable Categories for prediction
10 = X-Ray
9 = UltraSound
7 =  MRI
2 = CT
"""

# Convert the category id column into 4 columns based on delimiter
All_Clean_Condensed['X-Ray'] = (All_Clean_Condensed['Category id'].str.contains('10.0'))
All_Clean_Condensed['US'] = (All_Clean_Condensed['Category id'].str.contains('9.0'))
All_Clean_Condensed['MRI'] = (All_Clean_Condensed['Category id'].str.contains('7.0'))
All_Clean_Condensed['CT'] = (All_Clean_Condensed['Category id'].str.contains('2.0'))

# Remove columns if no longer needed for whatever reason
All_Clean_Dropped = All_Clean_Condensed.drop(['CSN', 'Arrival Method', 'CC', 'Postal Code',
                                              'Province','Category id','Day of Arrival', 'Gender','Arrived' ], axis=1)
# Confirm all the columns are in use-able format
test = All_Clean_Dropped.dtypes

# convert everything that is objects to floats or int

All_Clean_Dropped['Last Weight formatted'] = pd.to_numeric(All_Clean_Dropped['Last Weight formatted'], errors='coerce')
All_Clean_Dropped['Pulse Formatted'] = pd.to_numeric(All_Clean_Dropped['Pulse Formatted'], errors='coerce')
All_Clean_Dropped['Resp Formatted'] = pd.to_numeric(All_Clean_Dropped['Resp Formatted'], errors='coerce')
All_Clean_Dropped['Temp Formatted'] = pd.to_numeric(All_Clean_Dropped['Temp Formatted'], errors='coerce')



# Confirm all the columns are without nulls
All_Clean_Dropped = All_Clean_Dropped.dropna()
All_Clean_Dropped.isna().sum()

# Write it to csv for easy reference
All_Clean_Dropped.to_csv(r'/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv', index = None, header=True)
# -----------------------------------------------------------------------------------------------------------------------
print("done 2")