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
ED_2019_Feb = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/Feb13_ClinData_2019.xlsx') # only 1 sheet
ED_2019_Mar = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/March_ClinData_2019.xlsx')
ED_2019_Apr = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/April_ClinData_2019.xlsx')
ED_2019_May = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/May_ClinData_2019.xlsx')
ED_2019_Jun = pd.read_excel ('/home/andrew/Public/ED_Epic_Data/June_ClinData_2019.xlsx')

# All results will go to output path
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/ED_Review_Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration

# are the files the same shape?
ED_2018_Aug_2019_Feb.shape # (48803, 52)

ED_2019_Feb.shape


# ----------------------------------------------------------------------------------------------------------------------

