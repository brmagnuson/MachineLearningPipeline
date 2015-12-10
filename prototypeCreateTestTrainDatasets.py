import os
import math
import mlutilities.types
import mlutilities.dataTransformation


# Read in water years as ordered from driest to wettest for the Sacramento by NOAA
basePath = 'MasterData/'
waterYears = []
with open(basePath + 'NOAAWaterYearsDriestToWettest.csv') as file:
    for line in file.readlines():
        year = int(line)
        waterYears.append(year)

# Get water years of interest (drier years)
proportionOfInterest = 0.5
numberToExtract = math.ceil(len(waterYears) * proportionOfInterest)
dryWaterYears = waterYears[:numberToExtract]

# Read in original dataset with all years (with ObsID column added at the beginning before running code)
myFeaturesIndex = 6
myLabelIndex = 5
month = 'jul'
fullDataSet = mlutilities.types.DataSet('All Years',
                                        basePath + month + '_IntMnt_ref.csv',
                                        featuresIndex=myFeaturesIndex,
                                        labelIndex=myLabelIndex)

# Get appropriate calendar years for the month of interest
# (Oct, Nov, and Dec: calendar year = water year - 1. Ex: Oct 1976 is water year 1977.)
if month in ['oct', 'nov', 'dec']:
    calendarYears = [x - 1 for x in dryWaterYears]
else:
    calendarYears = dryWaterYears

# Subset full dataset to those years of interest
dryDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(calendarYears)]
dryDataSet = mlutilities.types.DataSet('Dry Years',
                                       basePath + month + '_IntMnt_dry.csv',
                                       'w',
                                       dataFrame=dryDataFrame,
                                       featuresIndex=myFeaturesIndex,
                                       labelIndex=myLabelIndex)

# Split dry data set into test and train
testProportion = 0.2
testPath = os.path.dirname(dryDataSet.path) + '/' + month + '_IntMnt_test.csv'
trainPath = os.path.dirname(dryDataSet.path) + '/' + os.path.basename(dryDataSet.path).split('.')[0] + '_train.csv'
splitDryDataSet = mlutilities.dataTransformation.splitDataSet(dryDataSet,
                                                              testProportion,
                                                              randomSeed=20150112,
                                                              trainPath=trainPath,
                                                              testPath=testPath)
universalTestDataSet = splitDryDataSet.testDataSet
dryTrainDataSet = splitDryDataSet.trainDataSet

# Use ObsIDs of the universal test set to subset full data set to everything else, which creates its training set
universalTestObsIds = universalTestDataSet.dataFrame.ObsID.values
fullTrainDataFrame = fullDataSet.dataFrame.loc[~fullDataSet.dataFrame.ObsID.isin(universalTestObsIds)]
fullTrainDataSet = mlutilities.types.DataSet('All Years Training Set',
                                             basePath + month + '_IntMnt_ref_train.csv',
                                             'w',
                                             dataFrame=fullTrainDataFrame,
                                             featuresIndex=myFeaturesIndex,
                                             labelIndex=myLabelIndex)

# Make sure all the numbers add up
print('Dry data should be equal to test + dry train:',
      dryDataSet.dataFrame.shape[0] == universalTestDataSet.dataFrame.shape[0] + dryTrainDataSet.dataFrame.shape[0])
print('Full data - test should be equal to full train:',
      fullDataSet.dataFrame.shape[0] - universalTestDataSet.dataFrame.shape[0] == fullTrainDataFrame.shape[0])
