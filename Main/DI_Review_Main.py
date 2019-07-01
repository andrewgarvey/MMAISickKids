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
# Set paths
custom_func_path = '/home/andrew/PycharmProjects/SickKidsMMAI/MMAISickKids/Helper_Functions/'
data_path = '/home/andrew/Public/DI'
output_path = '/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/DI_Review_Output/'


# Import Packages
import os
import numpy as np
import pandas as pd


# import custom functions
os.chdir(custom_func_path)


# Import DI data
os.chdir(data_path)


# Reset to output path
os.chdir(output_path)
#-----------------------------------------------------------------------------------------------------------------------
# Data Exploration



