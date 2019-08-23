"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: July 1st, 2019
Client: Hospital for Sick Children

Title: ED_Review_Main

Purpose:
-   Develop a model to predict DI demand using ED triage data and any outside available factors
"""

# Format date acceptably
DI_Full['Order Time'] = pd.to_datetime(DI_Full['Order Time'])

# Check in on the amount of people that actually match between the two, same MRN
All_Full = DI_Full.merge(ED_Reduced, how='inner', on = 'MRN' )
All_Full.shape #plenty because MRN is not distinct in either table
All_Full.dtypes


# Restrict the Joined to be based on Order time < 24 hours AFTER Arrived
Time_Difference = (All_Full['Order Time'] - All_Full['Arrived']).astype('timedelta64[s]')
Check1 = Time_Difference < (24*60*60)
Check2 = Time_Difference > 0
Logical_Time_Index = Check1&Check2

All_Restricted = All_Full[Logical_Time_Index]

All_Restricted.shape