"""
Author: Sargon Morad
Partner: Andrew Garvey, edited slightly
Date: July 1st, 2019
Client: Hospital for Sick Children

Title: DI_Review_Main

Purpose:
- Clean the DI data for future analysis
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

os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/')

#import ED DI Data

di2018q3 = pd.read_excel('/home/andrew/Public/DI/ED DI 2018 - Q3.xlsx')  # only 1 sheet
di2018q4 = pd.read_excel('/home/andrew/Public/DI/ED DI 2018 - Q4.xlsx')
di2019q1 = pd.read_excel('/home/andrew/Public/DI/ED DI 2019 - Q1.xlsx')
di2019q2 = pd.read_excel('/home/andrew/Public/DI/ED DI 2019 - Q2 20190621.xlsx')

di_data = pd.merge(di2018q3, di2018q4, how='outer')
di_data = pd.merge(di_data, di2019q1, how='outer')
di_data = pd.merge(di_data, di2019q2, how='outer')

#make master copies of data sets - everything below can be done with following dataframes
di_data_master = di_data.copy()

#use di_data for operations below

#drop Accession # column
di_data = di_data.drop(['Accession #'], axis=1)
#drop 'Reason for Exam' column
di_data = di_data.drop(['Reason for Exam'], axis=1)

#Convert string time entries into datetime objects
di_data['End Exam Time'] = pd.to_datetime(di_data['End Exam Time'], format="%a %d %b %Y %I:%M %p")
di_data['Order Time'] = pd.to_datetime(di_data['Order Time'], format="%Y/%m/%d %H:%M")
di_data['Finalized Time'] = pd.to_datetime(di_data['Finalized Time'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
di_data['Protocolling Instant'] = pd.to_datetime(di_data['Protocolling Instant'], format="%d/%m/%Y %I:%M %p", errors='coerce')

# create a categories df
categories = pd.DataFrame(di_data[['Category']])
categories['Category id'] = categories.groupby(['Category']).ngroup()
categories = categories.drop_duplicates()
categories = categories.sort_values('Category id')



# create a procedure  df
procedures = pd.DataFrame(di_data[['Procedure']])
procedures['Procedure id'] = procedures.groupby(['Procedure']).ngroup()
procedures = procedures.drop_duplicates()
procedures = procedures.sort_values('Procedure id')

# create Authorizing Provider df
authorizing_provider = pd.DataFrame(di_data[['Authorizing Provider']])
authorizing_provider['Authorizing Provider id'] = authorizing_provider.groupby(['Authorizing Provider']).ngroup()
authorizing_provider = authorizing_provider.drop_duplicates()
authorizing_provider = authorizing_provider.sort_values('Authorizing Provider id')

# combine di_data set with newly created dfs that have ids
di_data = pd.merge(di_data, categories, how='left', on='Category')
di_data = pd.merge(di_data, procedures, how='left', on='Procedure')
di_data = pd.merge(di_data, authorizing_provider, how='left', on='Authorizing Provider')

finalizing_physician = authorizing_provider.copy()
finalizing_physician = finalizing_physician.rename(columns={'Authorizing Provider': 'Finalizing Physician', 'Authorizing Provider id': 'Finalizing Physician id'})

di_data = pd.merge(di_data, finalizing_physician, how='left', on='Finalizing Physician')

# Andrew Check what is what
di_data.groupby('Category')['Category id'].count()

di_data = di_data.drop(['Category', 'Procedure', 'Authorizing Provider', 'Name'], axis=1)

di_data.to_csv(r'/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/DI_Clean.csv', index = None, header=True)

print("done 1b")