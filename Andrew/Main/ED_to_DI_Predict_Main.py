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
from pandasql import sqldf


#Set dir
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/')

#Import Cleaned Datasets
ED_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/ED_Clean')
DI_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Output/DI_Clean')

ED_Clean.shape

# Restrict the Joined rows to be based on order dates that are acceptable (arrived -> order -> discharge)
# Or just no tests for that visit is ok too, this will show up as null
# DI Timeframe entirely encompasses ED, so if they got a test they should be here.

# Could not find a clean way to do this that wouldnt take a bunch of extra work in python, using sql
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
                                    'Authorizing Provider id', 'Finalizing Physician id' ], axis=1)
#drop second mrn
# Tiny bit of renaming,


# Arrived might want to be put into hours of the day, dummy variable wise