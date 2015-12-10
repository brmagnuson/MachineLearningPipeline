import os
import thesisFunctions
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata

# Parameters
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
splitDryDataSets = mldata.kFoldSplitDataSet(dryDataSet, 5, randomSeed=47283,
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

# For each fold of the pipeline
# currentFold = 1
# while currentFold < 6:

    # Copy those datasets to CurrentFoldData, removing the _Number in their name
    # Run pipeline for those datasets
    # Aggregate results


# scoreModelResultsDF = thesisFunctions.flowModelPipeline(universalTestSetFileName='jul_IntMnt_test.csv',
#                                                         universalTestSetDescription='Jul IntMnt Test',
#                                                         basePath='Data/',
#                                                         picklePath='Pickles/',
#                                                         statusPrintPrefix='K-fold #1',
#                                                         randomSeed=47392)
# scoreModelResultsDF.to_csv('Output/cvScoreModelResults.csv', index=False)
