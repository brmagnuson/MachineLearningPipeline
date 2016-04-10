import os
import shutil
import time
import pandas
import sklearn.ensemble
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions
import constants


def setUpFiles(basePath):
    """
    Creates folder structure in OriginalModelAssessment folder and copies all years training data + test set
    :return:
    """

    allMonthsPath = 'AllMonthsDryHalf/'
    regions = ['IntMnt', 'Xeric']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    subFolders = ['CurrentFoldData', 'Output', 'Prediction']

    for region in regions:
        for month in months:

            print('Processing:', region, month.capitalize())

            # Create folder for each month & region in original model folder
            newFolderPath = basePath + region + '/' + month
            if not os.path.exists(newFolderPath):
                os.makedirs(newFolderPath)

            misnamed = newFolderPath + '/Predictions'
            if os.path.exists(misnamed):
                os.rmdir(misnamed)

            # Create subfolders used in model pipeline
            for subFolder in subFolders:
                newSubFolderPath = newFolderPath + '/' + subFolder
                if not os.path.exists(newSubFolderPath):
                    os.makedirs(newSubFolderPath)

            # Copy all years training sets and test sets
            for i in range(5):

                # Build paths
                trainFileName = '{}_{}_all_{}_train.csv'.format(month, region, i)
                testFileName = '{}_{}_{}_test.csv'.format(month, region, i)
                sourceTrainFilePath = allMonthsPath + region + '/' + month + '/' + trainFileName
                newTrainFilePath = newFolderPath + '/' + trainFileName
                sourceTestFilePath = allMonthsPath + region + '/' + month + '/' + testFileName
                newTestFilePath = newFolderPath + '/' + testFileName

                # Copy files
                shutil.copyfile(sourceTrainFilePath, newTrainFilePath)
                shutil.copyfile(sourceTestFilePath, newTestFilePath)

            # Add in full dataset and Sacramento data for prediction
            fullFileName = '{}_{}_all.csv'.format(month, region)
            sourceFullFilePath = allMonthsPath + region + '/' + month + '/' + fullFileName
            newFullFilePath = newFolderPath + '/Prediction/' + fullFileName
            shutil.copyfile(sourceFullFilePath, newFullFilePath)
            sacData = thesisFunctions.prepSacramentoData(month,
                                                         region)
            predictionFilePath = newFolderPath + '/Prediction/sacramentoData.csv'
            sacData.to_csv(predictionFilePath, index=False)


def getMonthVars(basepath, month, region):
    """
    Gets the list of variables chosen for that month and region from a text file
    """

    # Find month number
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthNumber = months.index(month) + 1
    monthNumber = str(monthNumber)
    if len(monthNumber) < 2:
        monthNumber = '0' + monthNumber

    # Get month variables from file
    monthVarPath = basepath + region + '/Vars/mvar' + str(monthNumber) + '.txt'
    with open(monthVarPath) as monthVarFile:
        monthVarString = '[' + monthVarFile.read() + ']'

    # Get rid of TOPWET because I don't have that data
    monthVarString = monthVarString.replace(',"TOPWET"', '')
    monthVars = eval(monthVarString)

    # Drop the first 4, because they are ID variables and the label
    monthVars = monthVars[4:]

    return monthVars


def runModels(basePath, performanceEstimation=True, prediction=False):

    randomSeed = constants.randomSeed
    myFeaturesIndex = 6
    myLabelIndex = 5
    kFolds = 5
    regions = ['IntMnt', 'Xeric']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
    meanOEMethod = mltypes.ModelScoreMethod('Mean O/E', mlmodel.meanObservedExpectedScore)
    sdOEMethod = mltypes.ModelScoreMethod('Standard Deviation O/E', mlmodel.sdObservedExpectedScore)
    mseMethod = mltypes.ModelScoreMethod('Mean Squared Error (cfs)', sklearn.metrics.mean_squared_error)
    testScoreMethods = [r2Method, meanOEMethod, sdOEMethod, mseMethod]

    randomForestParameters = {'n_estimators': 2000,
                              'max_features': .333,
                              'random_state': randomSeed,
                              'n_jobs': -1}
    randomForestMethod = mltypes.ModellingMethod(constants.randomForest,
                                                 sklearn.ensemble.RandomForestRegressor)

    for region in regions:
        for month in months:

            print('Processing:', region, month.capitalize())

            # Get expert features from text files
            selectedFeatures = getMonthVars(basePath, month, region)
            expertSelectedConfig = mltypes.FeatureEngineeringConfiguration('Expert Selection',
                                                                           'selection',
                                                                           mltypes.ExtractSpecificFeatures,
                                                                           {'featureList': selectedFeatures})

            modelFolder = basePath + region + '/' + month + '/'

            # Run model once on each fold to get estimates of test metrics
            if performanceEstimation:
                allFoldScoreModelResultsDFs = []

                for fold in range(kFolds):

                    # Get dataset info
                    foldTestFilePath = modelFolder + '{}_{}_{}_test.csv'.format(month, region, fold)
                    foldTrainFilePath = modelFolder + '{}_{}_all_{}_train.csv'.format(month, region, fold)
                    testDescription = month.capitalize() + ' ' + region + ' Test'
                    trainDescription = month.capitalize() + ' ' + region + ' Train'

                    # Copy to CurrentFoldDataFolder
                    testFilePath = modelFolder + 'CurrentFoldData/' + '{}_{}_test.csv'.format(month, region)
                    trainFilePath = modelFolder + 'CurrentFoldData/' + '{}_{}_all_train.csv'.format(month, region)
                    shutil.copyfile(foldTestFilePath, testFilePath)
                    shutil.copyfile(foldTrainFilePath, trainFilePath)

                    # Get datasets
                    fullTestDataSet = mltypes.DataSet(testDescription,
                                                      testFilePath,
                                                      featuresIndex=myFeaturesIndex,
                                                      labelIndex=myLabelIndex)
                    fullTrainDataSet = mltypes.DataSet(trainDescription,
                                                       trainFilePath,
                                                       featuresIndex=myFeaturesIndex,
                                                       labelIndex=myLabelIndex)

                    # Select features
                    trainDataSet, transformer = mldata.engineerFeaturesForDataSet(fullTrainDataSet,
                                                                                  expertSelectedConfig)
                    testDataSet = mldata.engineerFeaturesByTransformer(fullTestDataSet,
                                                                       transformer)

                    # Apply model
                    applyRFModelConfig = mltypes.ApplyModelConfiguration('Apply ' + constants.randomForest,
                                                                         randomForestMethod,
                                                                         randomForestParameters,
                                                                         trainDataSet,
                                                                         testDataSet)
                    randomForestResult = mlmodel.applyModel(applyRFModelConfig)
                    applyModelResults = [randomForestResult]

                    # Score model and convert results to data frame
                    scoreModelResults = mlmodel.scoreModels(applyModelResults, testScoreMethods)
                    scoreModelResultsDF = mlutils.createScoreDataFrame(scoreModelResults)

                    # Add RMSE, then add to list of results for this month
                    scoreModelResultsDF['RMSE (cfs)'] = scoreModelResultsDF['Mean Squared Error (cfs)'].map(lambda x: x ** (1/2))
                    allFoldScoreModelResultsDFs.append(scoreModelResultsDF)

                    print(region, month, fold, 'processed')

                # Aggregate results into a single DataFrame
                allResultsDF = pandas.DataFrame()
                for fold in allFoldScoreModelResultsDFs:
                    allResultsDF = allResultsDF.append(fold, ignore_index=True)
                allResultsDF.to_csv(modelFolder + 'Output/scoreModelResults_all.csv', index=False)

                # Group by unique model & dataset combinations to average
                averageResultsDF = allResultsDF.groupby(['Base DataSet', 'Model Method']).mean().reset_index()
                sortedAverageResultsDF = averageResultsDF.sort(columns='R Squared', ascending=False)
                sortedAverageResultsDF.to_csv(modelFolder + 'Output/scoreModelResults_average.csv', index=False)

            # Prediction
            # if prediction:
            #
            #     predictionFolder = modelFolder + 'Prediction/'
            #
            #     # Get data
            #     fullTrainingDataSet = mltypes.DataSet(month.capitalize() + ' Training Data',
            #                                       predictionFolder + '{}_{}_all.csv'.format(month, region),
            #                                       featuresIndex=myFeaturesIndex,
            #                                       labelIndex=myLabelIndex)
            #     fullPredictionDataSet = mltypes.DataSet(month.capitalize() + ' Prediction Data',
            #                                         predictionFolder + 'sacramentoData.csv',
            #                                         featuresIndex=3,
            #                                         labelIndex=None)
            #
            #     # Select features
            #     trainDataSet, transformer = mldata.engineerFeaturesForDataSet(fullTrainingDataSet,
            #                                                                   expertSelectedConfig)
            #     predictionDataSet = mldata.engineerFeaturesByTransformer(fullPredictionDataSet,
            #                                                        transformer)
            #
            #     # Train model and predict for the Sacramento region
            #     applyRFModelConfig = mltypes.ApplyModelConfiguration('Apply ' + constants.randomForest,
            #                                                          randomForestMethod,
            #                                                          randomForestParameters,
            #                                                          trainDataSet,
            #                                                          predictionDataSet)
            #     applyRFModelResult = mlmodel.applyModel(applyRFModelConfig)
            #     predictionOutputPath = predictionFolder + 'sacramentoPredictions.csv'
            #     thesisFunctions.outputPredictions(applyRFModelResult, predictionOutputPath)

    if prediction:
        print('Aggregating predictions.')
        aggregateFile = thesisFunctions.aggregateSacPredictions([basePath],
                                                                'Output/',
                                                                'RandomForestData.csv',
                                                                months,
                                                                regions)
        waterYear = 1977
        thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)


# Initial setup
startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')
originalModelPath = 'ThesisAnalysis/OriginalModelAssessment/'

# Modeling process
# setUpFiles(originalModelPath)
runModels(originalModelPath, performanceEstimation=False, prediction=True)

# Report finish
endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond
print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))

