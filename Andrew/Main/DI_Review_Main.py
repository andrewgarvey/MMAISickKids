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
# Import Packages
import os
import numpy as np
import pandas as pd

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

# Just check that not everything is horribly wrong

DI_2018_Q3.describe()
DI_2018_Q3.columns.values

DI_2018_Q3.shape
DI_2018_Q4.shape
DI_2019_Q1.shape
DI_2019_Q2.shape  # looks like 22 k rows ish, 21 columns

# As far as DI goes, seems pretty good to me

# ---------------------------

# words