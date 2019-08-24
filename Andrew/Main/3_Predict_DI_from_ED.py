"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: Aug 24, 2019
Client: Hospital for Sick Children

Title: Predict_DI_from_ED

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
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Actually Do a model with purely the info we have here, LR or Random Forest Sounds good, multi-classification
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/ML_Clean')

# May have to go back and change things

#




