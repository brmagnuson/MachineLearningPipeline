import os
import shutil
import fnmatch
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import thesisFunctions

# Parameters
randomSeed = 47392
masterDataPath = 'MasterData/'
month = 'jul'
dryProportionOfInterest = 0.5
myfeaturesIndex = 6
myLabelIndex = 5
kFolds = 5

# Get dry water years
dryYears = thesisFunctions.getDryYears(masterDataPath + 'NOAAWaterYearsDriestToWettest.csv',
                                       month,
                                       dryProportionOfInterest)

# Read in original dataset with all years (with ObsID column added at the beginning before running code)
fullDataSet = mltypes.DataSet('All Years',
                              masterDataPath + month + '_IntMnt_ref.csv',
                              featuresIndex=myfeaturesIndex,
                              labelIndex=myLabelIndex)

# Subset full dataset to those years of interest
dryDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(dryYears)]
dryDataSet = mltypes.DataSet('Dry Years',
                             masterDataPath + month + '_IntMnt_dry.csv',
                             'w',
                             dataFrame=dryDataFrame,
                             featuresIndex=myfeaturesIndex,
                             labelIndex=myLabelIndex)

testPathPrefix = os.path.dirname(dryDataSet.path) + '/' + month + '_IntMnt'

# From the dryDataSet, create k universal test sets and corresponding k dry training sets
splitDryDataSets = mldata.kFoldSplitDataSet(dryDataSet, 5, randomSeed=randomSeed,
                                            testPathPrefix=testPathPrefix)

# Use ObsIDs of each universal test set to subset full data set to everything else, creating k full training sets
for fold in range(len(splitDryDataSets)):
    universalTestDataSet = splitDryDataSets[fold].testDataSet
    universalTestObsIds = universalTestDataSet.dataFrame.ObsID.values
    fullTrainDataFrame = fullDataSet.dataFrame.loc[~fullDataSet.dataFrame.ObsID.isin(universalTestObsIds)]
    fullTrainDataSet = mltypes.DataSet('All Years Training Set',
                                       masterDataPath + month + '_IntMnt_ref_' + str(fold) + '_train.csv',
                                       'w',
                                       dataFrame=fullTrainDataFrame,
                                       featuresIndex=myfeaturesIndex,
                                       labelIndex=myLabelIndex)

# Run pipeline for each fold of the data
for fold in range(len(splitDryDataSets)):

    # Get the datasets from this fold
    foldDataSets = []
    for root, directories, files in os.walk(masterDataPath):
        if root != masterDataPath:
            continue
        filesToCopy = fnmatch.filter(files, '*_' + str(fold) + '_*')
    if len(filesToCopy) == 0:
        raise Exception('No matching files found for fold', fold)

    # Copy them to CurrentFoldData folder, removing the _Number in their name
    for fileToCopy in filesToCopy:
        newFilePath = masterDataPath + 'CurrentFoldData/' + fileToCopy.replace('_' + str(fold), '')
        shutil.copyfile(masterDataPath + fileToCopy, newFilePath)

    # Run pipeline for those datasets
    scoreModelResultsDF = thesisFunctions.flowModelPipeline(universalTestSetFileName='jul_IntMnt_test.csv',
                                                            universalTestSetDescription='Jul IntMnt Test',
                                                            basePath=masterDataPath + 'CurrentFoldData/',
                                                            picklePath=masterDataPath + 'Pickles/',
                                                            statusPrintPrefix='K-fold #' + str(fold),
                                                            randomSeed=randomSeed)

    # Aggregate results



scoreModelResultsDF.to_csv('Output/cvScoreModelResults.csv', index=False)
