import math
import csv
import pandas
import mlutilities.types


# Read in water years as ordered from driest to wettest for the Sacramento by NOAA
basePath = 'Data/'
waterYears = []
with open(basePath + 'NOAAWaterYearsDriestToWettest.csv') as file:
    for line in file.readlines():
        year = int(line)
        waterYears.append(year)

# Get water years of interest (drier years)
proportionOfInterest = 0.5
numberToExtract = math.ceil(len(waterYears) * proportionOfInterest)
dryWaterYears = waterYears[:numberToExtract]

# Read in original dataset with all years
myfeaturesIndex = 5
myLabelIndex = 4
month = 'jul'
fullDataSet = mlutilities.types.DataSet('All Years',
                                        basePath + month + '_IntMnt_ref.csv',
                                        featuresIndex=myfeaturesIndex,
                                        labelIndex=myLabelIndex)

# Get appropriate calendar years for the month of interest
# (Oct, Nov, and Dec: calendar year = water year - 1. Ex: Oct 1976 is water year 1977.)
if month in ['oct', 'nov', 'dec']:
    calendarYears = [x - 1 for x in dryWaterYears]
else:
    calendarYears = dryWaterYears

# Subset full dataset to those years of interest
dryDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(calendarYears)]

# Next step: split dry into test and train, then create the fullDataSet version of train.
