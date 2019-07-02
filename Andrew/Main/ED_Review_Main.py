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

# Import ED Data
# ED_data = ???

# All results will go to output path
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/ED_Review_Output/')

# ----------------------------------------------------------------------------------------------------------------------
# Data Exploration

# Just check that not everything is horribly wrong,

# um should have seperate triage based data somewhere?