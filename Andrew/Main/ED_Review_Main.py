"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: July 1st, 2019
Client: Hospital for Sick Children

Title: ED_Review_Main

Purpose:
-   Look for any trends in ED data that might be useful as it relates to staffing on a long term basis
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
from datetime import datetime

# Import Custom Functions
from Data_Functions import test_func

# Import Custom Files
import Data_Functions

# Import ED Data

ED_2018_Aug_2019_Feb = pd.read_csv('/home/dsingh/Public/ED_Epic_Data/ED_DATA_EPIC_AUG18_TO_FEB19.csv'
                                   , encoding ='latin-1') # unsure exactly whats wrong with this one
ED_2019_Feb = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/Feb13_ClinData_2019.xlsx')  # only 1 sheet
ED_2019_Mar = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/March_ClinData_2019.xlsx')
ED_2019_Apr = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/April_ClinData_2019.xlsx')
ED_2019_May = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/May_ClinData_2019.xlsx')
ED_2019_Jun = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/June_ClinData_2019.xlsx')

# All results will go to output path
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/ED_Review_Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Data Merging

# Are the files the same columns?
ED_2018_Aug_2019_Feb.shape  # (48803, 52)
ED_2019_Feb.shape  # (3312, 50)
ED_2019_Mar.shape  # (6779, 50)

# are off by a few, csv has 2 extra columns
ED_2018_Aug_2019_Feb = ED_2018_Aug_2019_Feb.drop(['X','Unnamed: 0'], axis=1)

# csv columns are still misnamed, but have same order
ED_2018_Aug_2019_Feb.columns = ED_2019_Feb.columns

# merge everything together
ED_Full = ED_2018_Aug_2019_Feb.append([ED_2019_Feb,ED_2019_Mar,ED_2019_Apr,ED_2019_May,ED_2019_Jun],ignore_index=True)

ED_Full.shape

# check for dupes, incase the timeframe overlapped, check for duplicated MRN
sum(ED_Full.duplicated(subset=None, keep='first')) # 0, no dupes across whole row
sum(ED_Full.duplicated(subset='MRN', keep='first')) # 212000 are duplicated Patients
sum(ED_Full.duplicated(subset='CSN', keep=False)) # 174 duplicated Visits, gonna remove entirely small enough

# cleaning
lst =  ED_2018_Aug_2019_Feb, ED_2019_Apr, ED_2019_Feb, ED_2019_Mar, ED_2019_May, ED_2019_Jun
del ED_2018_Aug_2019_Feb, ED_2019_Apr, ED_2019_Feb, ED_2019_Mar, ED_2019_May, ED_2019_Jun
del lst
# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration Cleaning

# CSN dups removed
ED_dup_index = ED_Full.duplicated(subset='CSN', keep=False)
ED_Full = ED_Full.loc[~ED_dup_index]

# Columns
ED_Full.columns
ED_Full.isna().sum()

# Remove entirely useless columns, recall purpose is to 'predict demand as function of month/day' and 'predict DI'
ED_Reduced = ED_Full.drop(['Registration Number','Pref Language','Acuity','Care Area','ED Complaint',
                           'Diagnosis','First ED Provider','Last ED Provider','ED Longest Attending ED Provider',
                           'Treatment Team','Last Attending Provider','Discharge-Admit Time','Door to PIA',
                           'ED PIA Threshold', 'ED Completed Length of Stay (Hours)','LOS','ED LWBS','Arrival to Room',
                           'Door to Pain Med','Hour of Arrival','Triage Complete User',
                           'Arrival to Initial Nursing Assessment','Door to Doc','CC.1','Primary Dx','Diagnoses',
                           'Admitting Provider','Lab Status','Rad Status','BP'],axis = 1)


# Counts
ED_Reduced.columns
ED_Reduced.isna().sum()
ED_Reduced.dtypes

# Encounter Number, pick only multi MRN, sort by mrn,
multi_mrn_index = ED_Full.duplicated(subset='MRN', keep=False)

multi_mrn = ED_Full.loc[multi_mrn_index]
multi_mrn = multi_mrn.sort_values(by = ['MRN'])
multi_mrn = multi_mrn.loc[:,['MRN','Encounter Number','Roomed']]  # conclusion, make our own, visits since Aug 2018

# format arrived datetime
ED_Reduced.loc[ED_Reduced['Arrived']==' ','Arrived'] = np.nan
ED_Reduced['Arrived'] = pd.to_datetime(ED_Reduced['Arrived'].astype(str), format=' %d/%m/%y %H%M')

# format discharge datetime, replace empty spaces w nan first
ED_Reduced.loc[ED_Reduced['Disch Date/Time']==' ','Disch Date/Time'] = np.nan
ED_Reduced['Disch Date/Time'] = pd.to_datetime(ED_Reduced['Disch Date/Time'].astype(str), format=' %d/%m/%Y %H%M')

# format roomed datetime, this one is silly, no padded dates, no year...
roomed_year = ED_Reduced['Arrived'].dt.year.astype(str) #double check dec 31 no issue...
roomed_year = roomed_year.str.extract(pat = '(^[0-9]{4})')
roomed_year = roomed_year.iloc[:,0]

roomed_month = ED_Reduced['Roomed'].str.extract(pat = '(/[0-9]+)')
roomed_month = roomed_month.iloc[:,0].str.replace('/','',regex=False)

roomed_day = ED_Reduced['Roomed'].str.extract(pat = '([0-9]+/)')
roomed_day = roomed_day.iloc[:,0].str.replace('/','',regex=False)

roomed_time = ED_Reduced['Roomed'].str.extract(pat = '([0-9]{4})')

roomed_hour = roomed_time.iloc[:,0].str.extract(pat = '(^[0-9]{2})')
roomed_hour = roomed_hour.iloc[:,0]

roomed_minute = roomed_time.iloc[:,0].str.extract(pat = '([0-9]{2}$)')
roomed_minute = roomed_minute.iloc[:,0]

# now we have a bunch of series to make into a datetime
arrived_df = pd.DataFrame({'year': roomed_year,
                           'month': roomed_month,
                           'day': roomed_day,
                           'hour': roomed_hour,
                           'minute': roomed_minute})

ED_Reduced['Roomed'] = pd.to_datetime(arrived_df[['year', 'month', 'day', 'hour', 'minute']])

# generate time between arrived/roomed/discharge in minutes

ED_Reduced['Arrived to Roomed'] = ED_Reduced['Roomed']-ED_Reduced['Arrived']
ED_Reduced['Arrived to Roomed'] = ED_Reduced['Arrived to Roomed']/np.timedelta64(1,'h')

ED_Reduced['Roomed to Discharge'] = ED_Reduced['Disch Date/Time']-ED_Reduced['Roomed']
ED_Reduced['Roomed to Discharge'] = ED_Reduced['Roomed to Discharge']/np.timedelta64(1,'h')

ED_Reduced['Arrived to Discharge'] = ED_Reduced['Disch Date/Time']-ED_Reduced['Arrived']
ED_Reduced['Arrived to Discharge'] = ED_Reduced['Arrived to Discharge']/np.timedelta64(1,'h')

# use that arrived datetime to generate a "since Aug 2018 date" ie Visits in last year, MRN is patient, CRN is visit
ED_Reduced['Visits Since Aug 2018'] = ED_Reduced.groupby(by='MRN')['Roomed'].transform(lambda x: x.rank())

# keep old number of visits to see if helpful (remove 2k)
ED_Reduced['Encounter Number'] = ED_Reduced['Encounter Number']-2000

# Address - > change to postal code and province,
ED_Reduced['Province_PostalCode'] = ED_Reduced['Address'].str.extract(pat='([A-Z]{2} [A-Z][0-9][A-Z] [0-9][A-Z][0-9]$)')

ED_Reduced['Postal Code'] = ED_Reduced['Province_PostalCode'].str.extract(pat='([A-Z][0-9][A-Z] [0-9][A-Z][0-9]$)')
ED_Reduced['Province'] = ED_Reduced['Province_PostalCode'].str.extract(pat='(^[A-Z]{2})')

ED_Reduced = ED_Reduced.drop(['Address','Province_PostalCode'],axis = 1)

# Age needs to to be in 1 unit (gonna choose days), y.o. vs m.o. vs wk.o. vs days
ED_Reduced['Age at Visit Number'] = ED_Reduced['Age at Visit'].str.extract(pat='( [0-9]+)')
ED_Reduced['Age at Visit denomination'] = ED_Reduced['Age at Visit'].str.extract(pat='([A-z].+)')

# convert string to proper time
year_index = (ED_Reduced['Age at Visit denomination']=='y.o.')
month_index = (ED_Reduced['Age at Visit denomination']=='m.o.')
week_index = (ED_Reduced['Age at Visit denomination']=='wk.o.')
days_index = (ED_Reduced['Age at Visit denomination']=='days')

ED_Reduced['Age at Visit denomination'].loc[year_index] = 365 # gives warning , still works fine
ED_Reduced['Age at Visit denomination'].loc[month_index] = 30
ED_Reduced['Age at Visit denomination'].loc[week_index] = 7
ED_Reduced['Age at Visit denomination'].loc[days_index] = 1

ED_Reduced['Age at Visit denomination'] = ED_Reduced['Age at Visit denomination'].astype(float)
ED_Reduced['Age at Visit Number'] = ED_Reduced['Age at Visit Number'].astype(float)

ED_Reduced['Age at Visit in days'] = ED_Reduced['Age at Visit Number']*ED_Reduced['Age at Visit denomination']

# Last weight has a few "none", strip the kg
ED_Reduced['Last Weight formatted'] = ED_Reduced['Last Weight'].str.replace('kg','',regex=False)

none_index = (ED_Reduced['Last Weight formatted']=='None')
ED_Reduced['Last Weight formatted'].loc[none_index] = np.nan

# Current Medications, list the number of them, nan = 0?
ED_Reduced['Current Medications Number'] = ED_Reduced['Current Medications']

ED_Reduced['Current Medications Number'] = ED_Reduced['Current Medications Number'].str.count(';') +1

nan_index = ED_Reduced['Current Medications Number'].isnull()
ED_Reduced['Current Medications Number'].loc[nan_index] = 0

#pulse and resp and temp, use just number no text, rounded
ED_Reduced['Pulse'] = ED_Reduced['Pulse'].astype(str)
ED_Reduced['Resp'] = ED_Reduced['Resp'].astype(str)
ED_Reduced['Temp'] = ED_Reduced['Temp'].astype(str)

ED_Reduced['Pulse Formatted'] = ED_Reduced['Pulse'].str.extract(pat='([0-9]*)')
ED_Reduced['Resp Formatted'] = ED_Reduced['Resp'].str.extract(pat='([0-9]*)')
ED_Reduced['Temp Formatted'] = ED_Reduced['Temp'].str.extract(pat='([0-9]*)') # rounded

# Replace empty with nan
empty_index_pulse = (ED_Reduced['Pulse Formatted'] == '')
empty_index_resp = (ED_Reduced['Resp Formatted'] == '')
empty_index_temp = (ED_Reduced['Temp Formatted'] == '')

ED_Reduced['Pulse Formatted'].loc[empty_index_pulse] = np.nan
ED_Reduced['Resp Formatted'].loc[empty_index_resp] = np.nan
ED_Reduced['Temp Formatted'].loc[empty_index_temp] = np.nan

# Remove the columns that are no longer needed
ED_Clean_w_null = ED_Reduced.drop(['Age at Visit','Last Weight','Current Medications','Pulse','Resp','Temp',
                                   'Age at Visit Number','Age at Visit denomination'] , axis = 1)
# ----------------------------------------------------------------------------------------------------------------------
## Basic Stats
# We can have a overall, but for statistics that are appropriate everything should be grouped by gender and age buckets
ED_Clean_w_null.describe().transpose()

# ----------------------------------------------------------------------------------------------------------------------
## Making the last step towards a perfect ML usable Dataframe

# Remove or replace all Nulls, either impute or remove rows or remove columns
ED_Clean = ED_Clean_w_null.dropna()

ED_Clean_w_null.shape
ED_Clean.shape # i consider this acceptable loses for a proof of concept

# Check out formats and what not for proper data types, want integers/factors
ED_Clean.dtypes # looks fine

# Check in on unique values of each column to make sure they are ok
