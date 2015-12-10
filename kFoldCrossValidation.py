import pandas
import thesisFunctions

# Parameters
randomSeed = 17
masterDataPath = 'MasterData/'
month = 'jul'
dryProportionOfInterest = 0.5
myFeaturesIndex = 6
myLabelIndex = 5
kFolds = 5

# Create my 5 test/train folds
# thesisFunctions.createKFoldDataSets(kFolds, masterDataPath, month, dryProportionOfInterest,
#                                     myFeaturesIndex, myLabelIndex, randomSeed)

# Run pipeline for each fold of the data
allFoldScoreModelResultsDFs = []
for fold in range(kFolds):

    thesisFunctions.copyFoldDataSets(fold, masterDataPath)

    # Run pipeline for those datasets
    foldScoreModelResultsDF = thesisFunctions.flowModelPipeline(universalTestSetFileName='jul_IntMnt_test.csv',
                                                                universalTestSetDescription='Jul IntMnt Test',
                                                                basePath=masterDataPath + 'CurrentFoldData/',
                                                                picklePath=masterDataPath + 'Pickles/',
                                                                outputFilePath=masterDataPath + 'Output/scoreModelResults_' + str(fold) + '.csv',
                                                                statusPrintPrefix='K-fold #' + str(fold),
                                                                randomSeed=randomSeed)

    allFoldScoreModelResultsDFs.append(foldScoreModelResultsDF)

# Aggregate results into a single DataFrame
allResultsDF = pandas.DataFrame()
for fold in allFoldScoreModelResultsDFs:
    allResultsDF = allResultsDF.append(fold, ignore_index=True)
allResultsDF.to_csv(masterDataPath + 'Output/scoreModelResults_all.csv', index=False)

# allResultsDF = pandas.read_csv(masterDataPath + 'Output/scoreModelResults_all.csv')

# Group by unique model & dataset combinations to average
averageResultsDF = allResultsDF.groupby(['Base DataSet', 'Model Method']).mean().reset_index()
sortedAverageResultsDF = averageResultsDF.sort(columns='R Squared', ascending=False)
sortedAverageResultsDF.to_csv(masterDataPath + 'Output/scoreModelResults_average.csv', index=False)\
