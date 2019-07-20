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

ED_2018_Aug_2019_Feb = pd.read_csv('/home/dsingh/Public/ED_Epic_Data/ED_DATA_EPIC_AUG18_TO_FEB19.csv', encoding ='latin-1') # i believe this is a property of linux being different
ED_2019_Feb = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/Feb13_ClinData_2019.xlsx')  # only 1 sheet
ED_2019_Mar = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/March_ClinData_2019.xlsx')
ED_2019_Apr = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/April_ClinData_2019.xlsx')
ED_2019_May = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/May_ClinData_2019.xlsx')
ED_2019_Jun = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/June_ClinData_2019.xlsx')

# All results will go to output path
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/ED_Review_Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration

# are the files the same columns?
ED_2018_Aug_2019_Feb.shape  # (48803, 52)
ED_2019_Feb.shape  # (3312, 50)
ED_2019_Mar.shape  # (6779, 50)

# are off by a few, csv has 2 extra columns
ED_2018_Aug_2019_Feb = ED_2018_Aug_2019_Feb.drop(['X','Unnamed: 0'], axis=1)

# csv columns are still misnamed, but have same order
ED_2018_Aug_2019_Feb.columns = ED_2019_Feb.columns

# merge everything together

ED_Full = ED_2018_Aug_2019_Feb.append([ED_2019_Feb,ED_2019_Mar,ED_2019_Apr,ED_2019_May,ED_2019_Jun],ignore_index=True)

# check for dupes, incase the timeframe overlapped, check for duplicated MRN

sum(ED_Full.duplicated(subset=None, keep='first')) # 0, no dupes across whole row
sum(ED_Full.duplicated(subset='MRN', keep='first'))
sum(ED_Full.duplicated(subset='CSN', keep='first')) # MRN vs CRN


# ----------------------------------------------------------------------------------------------------------------------

