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
# Import Packages
import os
import numpy as np
import pandas as pd

# Import Custom Functions
from Data_Functions import test_func

# Import Custom Files
import Data_Functions

# Import ED Data

ED_2018_Aug_2019_Feb = pd.read_csv('/home/dsingh/Public/ED_Epic_Data/ED_DATA_EPIC_AUG18_TO_FEB19.csv'
                                   , encoding ='latin-1') # i believe this is a property of linux being different
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

# CSN dups
dup_index = ED_Full.duplicated(subset='CSN', keep=False)
ED_Full = ED_Full.loc[~dup_index]

# Columns
ED_Full.columns
ED_Full.isna().sum()

ED_Full['Door to PIA'].value_counts()

# Remove entirely useless columns, recall purpose is to 'predict demand as function of month/day' and 'predict DI'
ED_Reduced = ED_Full.drop(['Encounter Number','Registration Number','Pref Language','Acuity','Care Area','ED Complaint',
                           'Diagnosis','First ED Provider','Last ED Provider','ED Longest Attending ED Provider',
                           'Treatment Team','Last Attending Provider','Discharge-Admit Time','Door to PIA',
                           'ED PIA Threshold', 'ED Completed Length of Stay (Hours)','LOS','ED LWBS','Arrival to Room',
                           'Door to Pain Med','Hour of Arrival','Triage Complete User',
                           'Arrival to Initial Nursing Assessment','Door to Doc','CC.1','Primary Dx','Diagnoses',
                           'Admitting Provider','Lab Status','Rad Status'],axis = 1)


# Counts
ED_Reduced.columns
ED_Reduced.isna().sum()


# Encounter Number, pick only multi MRN, sort by mrn,
multi_mrn_index = ED_Full.duplicated(subset='MRN', keep=False)

multi_mrn = ED_Full.loc[multi_mrn_index]
multi_mrn = multi_mrn.sort_values(by = ['MRN'])
multi_mrn = multi_mrn.loc[:,['MRN','Encounter Number','Roomed']]  # conclusion, make our own, visits since Aug 2018

# format roomed datetime

# format arrived datetime

# format discharge datetime

# generate time between arrived > roomed > discharge

# use that arrived datetime to generate a "since Aug 2018 date" ie Visits in last year

# use that to generate visits in last 6mo/ 3mo/ 1mo

# Address - > change to postal code and province

# Age needs to incorporate month vs wk vs year old,

# Weight has a few "none"

# CTA made more clear as a number

# Current Medications, List the number?

# BP has many many NA...

# Weight has to have the (!) removed

# ----------------------------------------------------------------------------------------------------------------------
# We can have a overall, but for statistics that are appropriate everything should be grouped by gender and age buckets

# Probably most
ED_Reduced['Last Weight'].unique()

ED_Reduced.groupby('Dispo').count()