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
data_path = '/home/andrew/Public/DI'
output_path = '/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/DI_Review_Output/'

# Import Packages
import os
import numpy as np
import pandas as pd

# Import Custom Functions
from Data_Functions import test_func
from Visualization_Functions import test

# Import DI Data
os.chdir(data_path)


# All results go to output path
os.chdir(output_path)

# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration


os.getcwd()
