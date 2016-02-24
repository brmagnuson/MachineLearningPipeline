import os
import shutil
import time
import threading
import math
import fnmatch
import copy
import re
import pandas
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import constants


# The following are functions specifically for my thesis and data, rather than generalizable functions as in the
# mlutilities library.

def getYearsOfInterest(filePath, month, proportionOfInterest, wetOrDry='dry'):
    """
    Given a ranking of driest water years from driest to wettest, extract the calendar years for the driest proportion.
    :param filePath:
    :param month: string. should be written as first three letters of month, lowercase. ex: 'jul'
    :param proportionOfInterest: float between 0 and 1
    :return: list of years of interest
    """

    # Read in water years as ordered from driest to wettest for the Sacramento by NOAA
    waterYears = []
    with open(filePath) as file:
        for line in file.readlines():
            year = int(line)
            waterYears.append(year)

    # Get water years of interest (drier years)
    if wetOrDry == 'dry':
        stopIndex = math.ceil(len(waterYears) * proportionOfInterest)
        waterYearsOfInterest = waterYears[:stopIndex]
    elif wetOrDry == 'wet':
        startIndex = math.ceil(len(waterYears) * proportionOfInterest)
        waterYearsOfInterest = waterYears[startIndex:]
    else:
        raise ValueError('wetOrDry had value other than \'wet\' or \'dry\'.')

    # Get appropriate calendar years for the month of interest
    # (Oct, Nov, and Dec: calendar year = water year - 1. Ex: Oct 1976 is water year 1977.)
    if month in ['oct', 'nov', 'dec']:
        calendarYears = [x - 1 for x in waterYearsOfInterest]
    else:
        calendarYears = waterYearsOfInterest
    return calendarYears


def createDescriptionFromFileName(fileName):
    """
    Takes in a file name (without any directory path) and turns it into a pretty string
    :param fileName:
    :return:
    """
    fileNameWithoutExtension = fileName.split('.')[0]
    fileNamePieces = fileNameWithoutExtension.split('_')

    capitalizedFileNamePieces = []
    for fileNamePiece in fileNamePieces:
        firstLetter = fileNamePiece[0]
        theRest = fileNamePiece[1:]
        firstLetter = firstLetter.capitalize()
        capitalizedFileNamePieces.append(firstLetter + theRest)

    prettyDescription = ' '.join(capitalizedFileNamePieces)
    return prettyDescription


def createKFoldDataSets(kFolds, masterDataPath, myFeaturesIndex, myLabelIndex, randomSeed=None,
                        modelApproach=None, proportionOfInterest=None, month=None, region=None):

    if modelApproach not in ['wet', 'dry', 'sacramento']:
        raise ValueError("modelApproach must be either 'wet', 'dry', or 'sacramento'.")

    # Read in original dataset with all years
    if modelApproach in ['wet', 'dry']:

        fullDataSet = mltypes.DataSet('All Years',
                                      masterDataPath + month + '_' + region + '_all.csv',
                                      featuresIndex=myFeaturesIndex,
                                      labelIndex=myLabelIndex)
    else:

        fullDataSet = mltypes.DataSet('Sacramento Basin',
                                      masterDataPath + 'Sacramento_Basin.csv',
                                      featuresIndex=myFeaturesIndex,
                                      labelIndex=myLabelIndex)

    # Split DataSet k times
    if modelApproach in ['wet', 'dry']:

        # Get water years of interest
        yearsOfInterest = getYearsOfInterest(masterDataPath + 'NOAAWaterYearsDriestToWettest.csv',
                                             month,
                                             proportionOfInterest,
                                             modelApproach)

        # Subset full dataset to those years of interest
        yearsOfInterestDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(yearsOfInterest)]
        if modelApproach == 'dry':
            yearsOfInterestDescription = 'Dry Years'
        else:
            yearsOfInterestDescription = 'Wet Years'

        yearsOfInterestDataSet = mltypes.DataSet(yearsOfInterestDescription,
                                                 masterDataPath + month + '_' + region + '_' + modelApproach + '.csv',
                                                 'w',
                                                 dataFrame=yearsOfInterestDataFrame,
                                                 featuresIndex=myFeaturesIndex,
                                                 labelIndex=myLabelIndex)

        testPathPrefix = os.path.dirname(yearsOfInterestDataSet.path) + '/' + month + '_' + region

        # From the subset DataSet, create k universal test sets and corresponding k wet/dry (depending on wetOrDry)
        # training sets
        splitDataSets = mldata.kFoldSplitDataSet(yearsOfInterestDataSet, kFolds, randomSeed=randomSeed,
                                                 testPathPrefix=testPathPrefix)

    else:

        # When running the Sacramento Basin approach, we don't need to subset to dry/wet years. We just split it.
        splitDataSets = mldata.kFoldSplitDataSet(fullDataSet, kFolds, randomSeed=randomSeed)

    # If doing the wet/dry approach, use ObsIDs of each universal test set to subset full data set to everything else,
    # creating k full training sets
    if modelApproach in ['wet', 'dry']:
        for fold in range(kFolds):
            universalTestDataSet = splitDataSets[fold].testDataSet
            universalTestObsIds = universalTestDataSet.dataFrame.ObsID.values
            fullTrainDataFrame = fullDataSet.dataFrame.loc[~ fullDataSet.dataFrame.ObsID.isin(universalTestObsIds)]

            # Write this out to the proper folder.
            mltypes.DataSet('All Years Training Set',
                            masterDataPath + month + '_' + region + '_all_' + str(fold) + '_train.csv',
                            'w',
                            dataFrame=fullTrainDataFrame,
                            featuresIndex=myFeaturesIndex,
                            labelIndex=myLabelIndex)

    return


def copyFoldDataSets(fold, masterDataPath):

    # Get the datasets from this fold
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

    return


def getResultsFromThreads(function, arguments, listForAppending, statusPrint=None):
    if statusPrint is not None:
        print(statusPrint)
    listForAppending.append(function(**arguments))

def flowModelPipeline(universalTestSetFileName, universalTestSetDescription, basePath, scoreOutputFilePath,
                      myFeaturesIndex, myLabelIndex, selectedFeatureList, statusPrintPrefix='', subTaskPrint=True,
                      randomSeed=None, runScaleDatasets=True, runFeatureEngineering=True, runEnsembleModels=True,
                      multiThreadApplyModels=False):

    """
    Runs the pipeline for a given universal test set.
    :param universalTestSetFileName:
    :param universalTestSetDescription:
    :param basePath:
    :param scoreOutputFilePath:
    :param statusPrintPrefix:
    :param subTaskPrint:
    :param randomSeed:
    :param runScaleDatasets:
    :param runFeatureEngineering:
    :param runEnsembleModels:
    :return:
    """

    # Parameters
    tuneScoreMethod = 'r2'
    # tuneScoreMethod = 'mean_squared_error'
    r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
    meanOEMethod = mltypes.ModelScoreMethod('Mean O/E', mlmodel.meanObservedExpectedScore)
    sdOEMethod = mltypes.ModelScoreMethod('Standard Deviation O/E', mlmodel.sdObservedExpectedScore)
    mseMethod = mltypes.ModelScoreMethod('Mean Squared Error (cfs)', sklearn.metrics.mean_squared_error)
    testScoreMethods = [r2Method, meanOEMethod, sdOEMethod, mseMethod]

    # Prepare datasets
    print(statusPrintPrefix, 'Preparing input data sets.')

    # Get base test set from folder
    universalTestDataSet = mltypes.DataSet(universalTestSetDescription,
                                           basePath + universalTestSetFileName,
                                           featuresIndex=myFeaturesIndex,
                                           labelIndex=myLabelIndex)

    # Get all base training sets from folder
    baseTrainingDataSets = []
    for root, directories, files in os.walk(basePath):
        for file in fnmatch.filter(files, '*_train.csv'):
            description = createDescriptionFromFileName(file)
            baseTrainingDataSet = mltypes.DataSet(description,
                                                  basePath + file,
                                                  featuresIndex=myFeaturesIndex,
                                                  labelIndex=myLabelIndex)
            baseTrainingDataSets.append(baseTrainingDataSet)

    # Associate each base training set with its own copy of the universal test set
    dataSetAssociations = []
    for baseTrainingDataSet in baseTrainingDataSets:
        # Build new versions of DataSet attributes
        copyDescription = baseTrainingDataSet.description + '\'s Copy Of Test Set'
        copyPath = basePath + \
                   os.path.basename(universalTestDataSet.path).split('.')[0] + '_' + \
                   os.path.basename(baseTrainingDataSet.path).split('.')[0].split('_')[2] + '_copy.csv'
        copyOfUniversalTestDataSet = mltypes.DataSet(copyDescription,
                                                     copyPath,
                                                     'w',
                                                     dataFrame=universalTestDataSet.dataFrame,
                                                     featuresIndex=myFeaturesIndex,
                                                     labelIndex=myLabelIndex)
        dataSetAssociation = mltypes.SplitDataSet(baseTrainingDataSet, copyOfUniversalTestDataSet)
        dataSetAssociations.append(dataSetAssociation)

    # Scale data sets based on the training set
    scaledDataSetAssociations = []
    if runScaleDatasets:
        print(statusPrintPrefix, 'Scaling data sets.')
        for dataSetAssociation in dataSetAssociations:
            # Scale training data and get scaler
            scaledTrainDataSet, scaler = mldata.scaleDataSet(dataSetAssociation.trainDataSet)

            # Scale testing data using scaler
            scaledTestDataSet = mldata.scaleDataSetByScaler(dataSetAssociation.testDataSet, scaler)

            # Associate the data sets
            scaledDataSetAssociation = mltypes.SplitDataSet(scaledTrainDataSet, scaledTestDataSet)
            scaledDataSetAssociations.append(scaledDataSetAssociation)

    dataSetAssociations += scaledDataSetAssociations

    # Perform feature engineering
    featureEngineeredDataSetAssociations = []
    if runFeatureEngineering:
        print(statusPrintPrefix, 'Engineering features.')
        varianceThresholdConfig = mltypes.FeatureEngineeringConfiguration('Variance Threshold .08',
                                                                          'selection',
                                                                          sklearn.feature_selection.VarianceThreshold,
                                                                          {'threshold': .08})
        pca20Config = mltypes.FeatureEngineeringConfiguration('PCA n20',
                                                              'extraction',
                                                              sklearn.decomposition.PCA,
                                                              {'n_components': 20})
        pca50Config = mltypes.FeatureEngineeringConfiguration('PCA n50',
                                                              'extraction',
                                                              sklearn.decomposition.PCA,
                                                              {'n_components': 50})
        expertSelectedConfig = mltypes.FeatureEngineeringConfiguration('Expert Selection',
                                                                       'selection',
                                                                       mltypes.ExtractSpecificFeatures,
                                                                       {'featureList': selectedFeatureList})
        featureEngineeringConfigs = [varianceThresholdConfig, pca20Config, pca50Config, expertSelectedConfig]

        for dataSetAssociation in dataSetAssociations:

            for featureEngineeringConfig in featureEngineeringConfigs:

                # Feature engineer training data and get transformer
                featureEngineeredTrainDataSet, transformer = mldata.engineerFeaturesForDataSet(dataSetAssociation.trainDataSet,
                                                                                               featureEngineeringConfig)
                # Transform testing data using transformer
                featureEngineeredTestDataSet = mldata.engineerFeaturesByTransformer(dataSetAssociation.testDataSet,
                                                                                    transformer)

                # Associate the data sets
                featureEngineeredDataSetAssociation = mltypes.SplitDataSet(featureEngineeredTrainDataSet,
                                                                           featureEngineeredTestDataSet)
                featureEngineeredDataSetAssociations.append(featureEngineeredDataSetAssociation)

    dataSetAssociations += featureEngineeredDataSetAssociations

    # Tune models
    print(statusPrintPrefix, 'Tuning models.')

    ridgeParameters = [{'alpha': [0.0, 0.1, 0.5, 1.0],
                        'normalize': [True, False]}]
    ridgeMethod = mltypes.ModellingMethod(constants.ridgeRegression,
                                          getSKLearnFunction(constants.ridgeRegression))
    ridgeConfig = mltypes.TuneModelConfiguration(constants.ridgeRegression,
                                                 ridgeMethod,
                                                 ridgeParameters,
                                                 tuneScoreMethod)
    randomForestParameters = [{'n_estimators': [50, 75, 100],
                               'max_features': [10, 'sqrt'],
                               'random_state': [randomSeed]}]
    randomForestMethod = mltypes.ModellingMethod(constants.randomForest,
                                                 getSKLearnFunction(constants.randomForest))
    randomForestConfig = mltypes.TuneModelConfiguration(constants.randomForest,
                                                        randomForestMethod,
                                                        randomForestParameters,
                                                        tuneScoreMethod)
    kNeighborsParameters = [{'n_neighbors': [2, 5, 10],
                             'metric': ['minkowski'],
                             'weights': ['uniform', 'distance']}]
    kNeighborsMethod = mltypes.ModellingMethod(constants.kNeighbors,
                                               getSKLearnFunction(constants.kNeighbors))
    kNeighborsConfig = mltypes.TuneModelConfiguration(constants.kNeighbors,
                                                      kNeighborsMethod,
                                                      kNeighborsParameters,
                                                      tuneScoreMethod)
    svmParameters = [{'C': [1.0, 10.0],
                      'epsilon': [0.1, 0.2],
                      'kernel': ['rbf', 'sigmoid']}]
    svmMethod = mltypes.ModellingMethod(constants.supportVectorMachine,
                                        getSKLearnFunction(constants.supportVectorMachine))
    svmConfig = mltypes.TuneModelConfiguration(constants.supportVectorMachine,
                                               svmMethod,
                                               svmParameters,
                                               tuneScoreMethod)
    decisionTreeParameters = [{'max_features': ['sqrt', 'auto'],
                               'random_state': [randomSeed]}]
    decisionTreeMethod = mltypes.ModellingMethod(constants.decisionTree,
                                                 getSKLearnFunction(constants.decisionTree))
    decisionTreeConfig = mltypes.TuneModelConfiguration(constants.decisionTree,
                                                        decisionTreeMethod,
                                                        decisionTreeParameters,
                                                        tuneScoreMethod)
    adaBoostParameters = [{'n_estimators': [50, 100],
                           'learning_rate': [0.5, 1.0],
                           'random_state': [randomSeed]}]
    adaBoostMethod = mltypes.ModellingMethod(constants.adaBoost,
                                             getSKLearnFunction(constants.adaBoost))
    adaBoostConfig = mltypes.TuneModelConfiguration(constants.adaBoost,
                                                    adaBoostMethod,
                                                    adaBoostParameters,
                                                    tuneScoreMethod)

    tuneModelConfigs = [ridgeConfig, randomForestConfig, kNeighborsConfig,
                        svmConfig, decisionTreeConfig, adaBoostConfig]

    # Build tune model configurations
    counter = 1
    total = len(dataSetAssociations) * len(tuneModelConfigs)
    tuneModelResults = []
    for dataSetAssociation in dataSetAssociations:
        for tuneModelConfig in tuneModelConfigs:

            if subTaskPrint:
                print(statusPrintPrefix, 'Tuning (%s of %s):' % (counter, total),
                      tuneModelConfig.description, 'for', dataSetAssociation.trainDataSet.description)
            tuneModelResult = mlmodel.tuneModel(dataSetAssociation.trainDataSet,
                                                tuneModelConfig,
                                                randomSeed,
                                                constants.n_jobs)
            tuneModelResults.append(tuneModelResult)
            counter += 1

    # Apply models
    print(statusPrintPrefix, 'Applying models to test data.')

    # Build single-model ApplyModelConfigurations
    applyModelConfigs = []
    for tuneModelResult in tuneModelResults:

        trainDataSet = tuneModelResult.dataSet
        testDataSet = None
        for dataSetAssociation in dataSetAssociations:
            if dataSetAssociation.trainDataSet == trainDataSet:
                testDataSet = dataSetAssociation.testDataSet
                break

        # Make sure we found a match
        if testDataSet is None:
            raise Exception('No SplitDataSet found matching this training DataSet:\n' + trainDataSet)

        applyModelConfig = mltypes.ApplyModelConfiguration('Apply ' + tuneModelResult.description.replace('Train',
                                                                                                          'Test'),
                                                           tuneModelResult.modellingMethod,
                                                           tuneModelResult.parameters,
                                                           trainDataSet,
                                                           testDataSet)
        applyModelConfigs.append(applyModelConfig)

    # Build ensemble ApplyModelConfigurations
    if runEnsembleModels:

        # Find the maximum mean squared error for use in weighting
        maximumMSE = None
        if tuneScoreMethod == 'mean_squared_error':
            maximumMSE = max([tuneModelResult.bestScore for tuneModelResult in tuneModelResults])

        # For each base DataSet, find its matching model functions and parameters
        ensembleApplyModelConfigs = []
        for dataSetAssociation in dataSetAssociations:

            predictorConfigs = []
            weights = []
            bestWeight = float('-inf')
            stackingPredictorConfig = None

            # Find models associated with that DataSet and get their information to build predictor configs
            # for ensembles
            for tuneModelResult in tuneModelResults:
                if dataSetAssociation.trainDataSet == tuneModelResult.dataSet:

                    # Build Predictor Config
                    predictorConfig = mltypes.PredictorConfiguration(tuneModelResult.modellingMethod.description,
                                                                     tuneModelResult.modellingMethod.function,
                                                                     tuneModelResult.parameters)
                    predictorConfigs.append(predictorConfig)

                    # Make sure all weights are all positive
                    if tuneScoreMethod == 'mean_squared_error':
                        # The higher MSE is, the worse it is, so we want to invert its weight
                        weight = maximumMSE + 1 - tuneModelResult.bestScore
                    else:
                        # R squared can be negative, and weights should all be zero or positive.
                        if tuneModelResult.bestScore < 0:
                            weight = 0
                        else:
                            weight = tuneModelResult.bestScore

                    weights.append(weight)

                    # If tuneModelResult has a better score than previously seen, make it the stacked predictor config
                    if weight > bestWeight:
                        bestWeight = weight
                        stackingPredictorConfig = copy.deepcopy(predictorConfig)

                        # Hack: If stacking with a RandomForestRegressor and the number of models I'm stacking is fewer
                        # than max_features (which might occur when max_features was set to a specific number),
                        # RandomForestRegressor will error out.
                        if type(stackingPredictorConfig.predictorFunction()) == \
                                type(sklearn.ensemble.RandomForestRegressor()):
                            if isinstance(stackingPredictorConfig.parameters['max_features'], int):
                                stackingPredictorConfig.parameters['max_features'] = None

            # Create averaging ensemble
            averagingEnsembleModellingMethod = mltypes.ModellingMethod('Averaging Ensemble',
                                                                       mltypes.AveragingEnsemble)
            averagingEnsembleParameters = {'predictorConfigurations': predictorConfigs,
                                           'weights': weights}
            averagingEnsembleApplyModelConfig = mltypes.ApplyModelConfiguration(
                'Apply Averaging Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train',
                                                                                                               'Test'),
                averagingEnsembleModellingMethod,
                averagingEnsembleParameters,
                dataSetAssociation.trainDataSet,
                dataSetAssociation.testDataSet
            )
            ensembleApplyModelConfigs.append(averagingEnsembleApplyModelConfig)

            # Create stacking ensemble
            stackingEnsembleModellingMethod = mltypes.ModellingMethod('Stacking Ensemble',
                                                                      mltypes.StackingEnsemble)
            stackingEnsembleParameters = {'basePredictorConfigurations': predictorConfigs,
                                          'stackingPredictorConfiguration': stackingPredictorConfig}
            stackingEnsembleApplyModelConfig = mltypes.ApplyModelConfiguration(
                'Apply Stacking Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train',
                                                                                                              'Test'),
                stackingEnsembleModellingMethod,
                stackingEnsembleParameters,
                dataSetAssociation.trainDataSet,
                dataSetAssociation.testDataSet
            )
            ensembleApplyModelConfigs.append(stackingEnsembleApplyModelConfig)

            stackingOFEnsembleModellingMethod = mltypes.ModellingMethod('Stacking OF Ensemble',
                                                                        mltypes.StackingEnsemble)
            stackingOFEnsembleParameters = {'basePredictorConfigurations': predictorConfigs,
                                            'stackingPredictorConfiguration': stackingPredictorConfig,
                                            'includeOriginalFeatures': True}
            stackingOFEnsembleApplyModelConfig = mltypes.ApplyModelConfiguration(
                'Apply OF Stacking Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train',
                                                                                                                 'Test'),
                stackingOFEnsembleModellingMethod,
                stackingOFEnsembleParameters,
                dataSetAssociation.trainDataSet,
                dataSetAssociation.testDataSet
            )
            ensembleApplyModelConfigs.append(stackingOFEnsembleApplyModelConfig)

        # Add ensemble configs to the rest of the ApplyModelConfigs
        applyModelConfigs += ensembleApplyModelConfigs

    # Apply models to test data
    if multiThreadApplyModels:
        counter = 1
        total = len(applyModelConfigs)
        applyModelResults = []
        applyModelResultThreads = []

        for applyModelConfig in applyModelConfigs:

            arguments = {'applyModelConfiguration': applyModelConfig}
            if subTaskPrint:
                statusPrint = statusPrintPrefix + ' Applying ({} of {})'.format(counter, total)
            else:
                statusPrint = None
            applyModelResultThread = threading.Thread(target=getResultsFromThreads,
                                                      args=(mlmodel.applyModel, arguments, applyModelResults, statusPrint))
            applyModelResultThreads.append(applyModelResultThread)
            counter += 1

        # Start all threads
        for applyModelResultThread in applyModelResultThreads:
            applyModelResultThread.start()

        # Wait for all threads to finish populating applyModelResults before continuing
        for applyModelResultThread in applyModelResultThreads:
            applyModelResultThread.join()

    else:
        applyModelResults = mlmodel.applyModels(applyModelConfigs, subTaskPrint=subTaskPrint)

    # Score models
    # if runScoreModels:
    print(statusPrintPrefix, 'Scoring models on test data.')
    testScoreModelResults = mlmodel.scoreModels(applyModelResults, testScoreMethods)

    # Model testing result reporting
    if testScoreMethods[0].function == sklearn.metrics.mean_squared_error:
        sortedTestScoreModelResults = sorted(testScoreModelResults, key=lambda x: x.modelScores[0].score)
    else:
        sortedTestScoreModelResults = sorted(testScoreModelResults, key=lambda x: -x.modelScores[0].score)

    # Convert to data frame for tabulation and visualization
    scoreModelResultsDF = mlutils.createScoreDataFrame(sortedTestScoreModelResults)

    # Add RMSE and arrange column
    scoreModelResultsDF['RMSE (cfs)'] = scoreModelResultsDF['Mean Squared Error (cfs)'].map(lambda x: x ** (1/2))

    # Output to file
    scoreModelResultsDF.to_csv(scoreOutputFilePath, index=False)
    return scoreModelResultsDF


def runKFoldPipeline(baseDirectoryPath, myFeaturesIndex, myLabelIndex, selectedFeaturesList, kFolds=5,
                     modelApproach=None, month=None, region=None, randomSeed=None, multiThreadApplyModels=False):

    """
    Splits each region-month base dataset into k-fold test/train sets and runs the pipeline for each one.
    :param month:
    :param region:
    :param randomSeed:
    :return:
    """

    if modelApproach not in ['wet', 'dry', 'sacramento']:
        raise ValueError("Model approach must be either 'wet', 'dry', or 'sacramento'.")

    # Set parameters
    if modelApproach in ['wet', 'dry']:
        masterDataPath = baseDirectoryPath + region + '/' + month + '/'
        proportionOfInterest = 0.5
    else:
        masterDataPath = baseDirectoryPath

    # Create my 5 test/train folds
    if modelApproach in ['wet', 'dry']:
        createKFoldDataSets(kFolds,
                            masterDataPath,
                            myFeaturesIndex,
                            myLabelIndex,
                            randomSeed,
                            modelApproach=modelApproach,
                            proportionOfInterest=proportionOfInterest,
                            month=month,
                            region=region)
    else:
        createKFoldDataSets(kFolds,
                            masterDataPath,
                            myFeaturesIndex,
                            myLabelIndex,
                            randomSeed,
                            modelApproach=modelApproach)

    # Run pipeline for each fold of the data
    allFoldScoreModelResultsDFs = []
    for fold in range(kFolds):

        copyFoldDataSets(fold, masterDataPath)

        # Run pipeline for those datasets
        if modelApproach in ['wet', 'dry']:
            universalTestSetFileName = month + '_' + region + '_test.csv'
            universalTestSetDescription = month.capitalize() + ' ' + region + ' Test'
            statusPrintPrefix = region + ' ' + month.capitalize() + ' K-fold #' + str(fold)
        else:
            universalTestSetFileName = 'Sacramento_Basin_test.csv'
            universalTestSetDescription = 'Sacramento Basin Test'
            statusPrintPrefix = 'Sacramento Basin K-fold #' + str(fold)

        foldScoreModelResultsDF = flowModelPipeline(universalTestSetFileName=universalTestSetFileName,
                                                    universalTestSetDescription=universalTestSetDescription,
                                                    basePath=masterDataPath + 'CurrentFoldData/',
                                                    scoreOutputFilePath=masterDataPath + 'Output/scoreModelResults_' +
                                                                        str(fold) + '.csv',
                                                    myFeaturesIndex=myFeaturesIndex,
                                                    myLabelIndex=myLabelIndex,
                                                    selectedFeatureList=selectedFeaturesList,
                                                    statusPrintPrefix=statusPrintPrefix,
                                                    subTaskPrint=False,
                                                    randomSeed=randomSeed,
                                                    multiThreadApplyModels=multiThreadApplyModels)

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
    sortedAverageResultsDF.to_csv(masterDataPath + 'Output/scoreModelResults_average.csv', index=False)

    return


def getSKLearnFunction(description):
    """
    Matches a model description with an sklearn function object.
    :param description:
    :return:
    """
    if description == constants.ridgeRegression:
        predictorFunction = sklearn.linear_model.Ridge
    elif description == constants.randomForest:
        predictorFunction = sklearn.ensemble.RandomForestRegressor
    elif description == constants.kNeighbors:
        predictorFunction = sklearn.neighbors.KNeighborsRegressor
    elif description == constants.supportVectorMachine:
        predictorFunction = sklearn.svm.SVR
    elif description == constants.decisionTree:
        predictorFunction = sklearn.tree.DecisionTreeRegressor
    elif description == constants.adaBoost:
        predictorFunction = sklearn.ensemble.AdaBoostRegressor
    else:
        raise Exception('No matching sklearn function found.')
    return predictorFunction


def parseDescriptionToBuildFeatureEngineeringConfig(dataSetDescription, selectedFeaturesList, randomSeed):

    # Extract feature engineering information
    featureEngineeringDescription = dataSetDescription.split('via')[1].strip().split(',')[0]

    # Build feature engineering config
    if any(x in featureEngineeringDescription for x in ['ICA', 'PCA']):
        selectionOrExtraction = 'extraction'
        n_components = int(featureEngineeringDescription.split('n')[1])

        if 'PCA' in featureEngineeringDescription:
            featureEngineeringMethod = sklearn.decomposition.PCA
            featureEngineeringParameters = {'n_components': n_components}
        else:
            featureEngineeringMethod = sklearn.decomposition.FastICA
            featureEngineeringParameters = {'n_components': n_components, 'max_iter': 2500, 'random_state': randomSeed}

    elif any(x in featureEngineeringDescription for x in ['Variance Threshold', 'Expert Selection']):
        selectionOrExtraction = 'selection'

        if 'Variance Threshold' in featureEngineeringDescription:
            featureEngineeringMethod = sklearn.feature_selection.VarianceThreshold
            threshold = float(featureEngineeringDescription.split()[-1])
            featureEngineeringParameters = {'threshold': threshold}
        else:
            featureEngineeringMethod = mltypes.ExtractSpecificFeatures
            featureEngineeringParameters = {'featureList': selectedFeaturesList}

    else:
        raise Exception('Feature engineering method not recognized.')

    featureEngineeringConfig = mltypes.FeatureEngineeringConfiguration(featureEngineeringDescription,
                                                                       selectionOrExtraction,
                                                                       featureEngineeringMethod,
                                                                       featureEngineeringParameters)
    return featureEngineeringConfig


def parseDescriptionToBuildApplyModelConfig(modelDescription, modelParameters, trainDataSet, predictionDataSet):

    if 'Ensemble' in modelDescription:

        # Build model method and parameters for averaging or stacking ensembles
        if modelDescription == 'Averaging Ensemble':

            # Parse trainModelParameters string for averaging ensemble so that its pieces can be correctly evaluated.

            try:
                # Find weights in trainModelParameters string and convert to list
                weights = re.search("'weights': (\[.*?])", modelParameters).group(1)
                weights = eval(weights)

            except AttributeError:
                raise Exception('Weights not found in Averaging Ensemble.')

            try:
                # Find predictor configurations
                predictorConfigsString = re.search("'predictorConfigurations': (.*?])", modelParameters).group(1)

            except AttributeError:
                raise Exception('Predictor configurations not found in Averaging Ensemble.')

            # Parse each predictor configuration
            predictorConfigs = []
            predictorTypeList = [constants.ridgeRegression, constants.randomForest, constants.kNeighbors,
                                 constants.supportVectorMachine, constants.decisionTree, constants.adaBoost]
            for predictorType in predictorTypeList:

                # Get function object that matches predictorType
                predictorFunction = getSKLearnFunction(predictorType)

                # Get dictionary of predictor configuration's parameters
                predictorParams = eval(re.search(predictorType + ' (.*?})', predictorConfigsString).group(1))

                predictorConfig = mltypes.PredictorConfiguration(predictorType,
                                                                 predictorFunction,
                                                                 predictorParams)
                predictorConfigs.append(predictorConfig)

            # Build pieces for applyModelConfig
            trainModelParameters = {'predictorConfigurations': predictorConfigs,
                                    'weights': weights}
            modelMethod = mltypes.ModellingMethod(modelDescription, mltypes.AveragingEnsemble)

        else:

            # Parse trainModelParameters string for stacking ensemble so that its pieces can be correctly evaluated.
            originalFeaturesSearch = re.search("'includeOriginalFeatures': (.*?),", modelParameters)

            if originalFeaturesSearch is None:

                # Then the stacking ensemble config must be using the default value, False
                includeOriginalFeatures = False

            else:
                includeOriginalFeatures = originalFeaturesSearch.group(1)
                includeOriginalFeatures = eval(includeOriginalFeatures)

            try:
                # Find predictor configurations
                predictorConfigsString = re.search("'basePredictorConfigurations': (.*?])", modelParameters).group(1)

            except AttributeError:
                raise Exception('Base predictor configurations not found in Stacking Ensemble.')

            # Parse each base predictor configuration
            predictorConfigs = []
            predictorTypeList = [constants.ridgeRegression, constants.randomForest, constants.kNeighbors,
                                 constants.supportVectorMachine, constants.decisionTree, constants.adaBoost]
            for predictorType in predictorTypeList:

                # Get function object that matches predictorType
                predictorFunction = getSKLearnFunction(predictorType)

                # Get dictionary of predictor configuration's parameters
                predictorParams = eval(re.search(predictorType + ' (.*?})', predictorConfigsString).group(1))

                predictorConfig = mltypes.PredictorConfiguration(predictorType,
                                                                 predictorFunction,
                                                                 predictorParams)
                predictorConfigs.append(predictorConfig)

            # Find the stacking predictor configuration

            try:
                # Which of the base predictor configs do we use to stack the predictions?
                stackingPredictorDescription = re.search("'stackingPredictorConfiguration': "
                                                         "(.*?) {", modelParameters).group(1)

            except AttributeError:
                raise Exception('Stacking Predictor not found in Stacking Ensemble.')

            # Match the stacking predictor to its base predictor config
            for predictorConfig in predictorConfigs:
                if stackingPredictorDescription == predictorConfig.description:
                    stackingPredictorConfig = copy.deepcopy(predictorConfig)
                    break

            # Hack: If stacking with a RandomForestRegressor and the number of models I'm stacking is fewer than
            # max_features (which might occur when max_features was set to a specific number), RandomForestRegressor
            # will error out.
            if type(stackingPredictorConfig.predictorFunction()) == type(sklearn.ensemble.RandomForestRegressor()):
                if isinstance(stackingPredictorConfig.parameters['max_features'], int):
                    stackingPredictorConfig.parameters['max_features'] = None


            # Build pieces for applyModelConfig
            trainModelParameters = {'basePredictorConfigurations': predictorConfigs,
                                    'stackingPredictorConfiguration': stackingPredictorConfig,
                                    'includeOriginalFeatures': includeOriginalFeatures}
            modelMethod = mltypes.ModellingMethod(modelDescription, mltypes.StackingEnsemble)

    elif any(x in modelDescription for x in [constants.randomForest, constants.ridgeRegression, constants.kNeighbors,
                                             constants.supportVectorMachine, constants.decisionTree,
                                             constants.adaBoost]):

        # Get model parameters from text string in dictionary form
        trainModelParameters = eval(modelParameters)

        # Build model method object
        modelFunction = getSKLearnFunction(modelDescription)
        modelMethod = mltypes.ModellingMethod(modelDescription, modelFunction)
    else:
        raise Exception('Model method not recognized.')

    applyModelConfig = mltypes.ApplyModelConfiguration(modelDescription,
                                                       modelMethod,
                                                       trainModelParameters,
                                                       trainDataSet,
                                                       predictionDataSet)
    return applyModelConfig


def outputPredictionLog(logPath, applyModelConfig, statistics=None):

    """
    Outputs a log of the applyModelConfig being used to predict.
    :param logPath:
    :param applyModelConfig:
    :param statistics:
    :return:
    """

    # Build text string
    text = 'Date: ' + time.strftime('%a, %d %b %Y %X') + '\n\n'
    text += 'Prediction dataset: ' + str(applyModelConfig.testDataSet) + '\n'
    text += 'Prediction method: ' + applyModelConfig.modellingMethod.description + '\n'
    text += 'Prediction parameters: ' + str(applyModelConfig.parameters) + '\n\n'

    if statistics is not None:
        text += 'Estimated Statistics:\n'
        for statistic, value in statistics.iteritems():
            text += statistic + ': ' + str(value) + '\n'

    # Output to file
    with open(logPath, 'w') as logFile:
        logFile.write(text)


def findModelAndPredict(predictionDataSet, masterDataPath, randomSeed, myFeaturesIndex, myLabelIndex,
                        selectedFeaturesList, month, region=None, modelRowIndex=0, printLog=False, logPath=None):

    # Read in score model results files.
    averageFile = masterDataPath + 'Output/scoreModelResults_average.csv'
    allFile = masterDataPath + 'Output/scoreModelResults_all.csv'
    averageResults = pandas.read_csv(averageFile)
    allResults = pandas.read_csv(allFile)

    # Get model with highest average R2
    bestModel = averageResults.iloc[modelRowIndex]
    trainDataSetDescription = bestModel.loc['Base DataSet']
    trainModelDescription = bestModel.loc['Model Method']

    # From all the fold results that match the best model, extract the parameters of the one with the highest R2
    bestModelFolds = allResults.loc[(allResults['Model Method'] == trainModelDescription) &
                                    (allResults['Base DataSet'] == trainDataSetDescription)]
    sortedBestModelFolds = bestModelFolds.sort(columns='R Squared', ascending=False)
    trainModelParameters = sortedBestModelFolds.iloc[0].loc['Parameters']

    # Find appropriate dataset based on the description and copy to Prediction folder
    if 'Sacramento' in trainDataSetDescription:
        trainingFileName = 'Sacramento_Basin.csv'
    elif 'Wet' in trainDataSetDescription:
        trainingFileName = month + '_' + region + '_wet.csv'
    elif 'Dry' in trainDataSetDescription:
        trainingFileName = month + '_' + region + '_dry.csv'
    else:
        trainingFileName = month + '_' + region + '_all.csv'

    trainingFilePath = masterDataPath + trainingFileName
    copiedTrainingFilePath = masterDataPath + 'Prediction/' + trainingFileName
    shutil.copyfile(trainingFilePath, copiedTrainingFilePath)

    trainDataSet = mltypes.DataSet(month.title() + ' Training Set',
                                   copiedTrainingFilePath,
                                   featuresIndex=myFeaturesIndex,
                                   labelIndex=myLabelIndex)

    # Scale if necessary
    if 'Scaled' in trainDataSetDescription:
        scaledTrainDataSet, scaler = mldata.scaleDataSet(trainDataSet)
        trainDataSet = scaledTrainDataSet
        predictionDataSet = mldata.scaleDataSetByScaler(predictionDataSet, scaler)

    # Feature engineer if necessary
    if 'features selected' in trainDataSetDescription:

        featureEngineeringConfig = parseDescriptionToBuildFeatureEngineeringConfig(trainDataSetDescription,
                                                                                   selectedFeaturesList,
                                                                                   randomSeed)
        featureEngineeredTrainDataSet, transformer = mldata.engineerFeaturesForDataSet(trainDataSet,
                                                                                       featureEngineeringConfig)
        trainDataSet = featureEngineeredTrainDataSet
        predictionDataSet = mldata.engineerFeaturesByTransformer(predictionDataSet, transformer)

    # Build apply model configuration
    applyModelConfig = parseDescriptionToBuildApplyModelConfig(trainModelDescription,
                                                               trainModelParameters,
                                                               trainDataSet,
                                                               predictionDataSet)

    # Train model and predict dataset
    if printLog:
        statistics = bestModel[2:]
        outputPredictionLog(logPath, applyModelConfig, statistics)
    applyModelResult = mlmodel.applyModel(applyModelConfig)
    return applyModelResult


def prepSacramentoData(month, region, wetOrDry=None, waterYearsFilePath=None, proportionOfInterest=None):

    hucFile = '../SacramentoData/Sacramento_basin_huc12_v3.csv'
    hucRegionFile = '../SacramentoData/Sacramento_huc12_ecoregions.csv'
    staticFile = '../SacramentoData/static_vars/h12.static.vars.csv'
    climateBasePath = '../SacramentoData/climate_vars/'

    # Find month number to build climate file path
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthNumber = months.index(month) + 1
    monthNumber = str(monthNumber)
    if len(monthNumber) < 2:
        monthNumber = '0' + monthNumber
    climateFile = climateBasePath + 'm' + monthNumber + '_HUC12.clim.data.' + month + '.csv'

    # Get HUCs for the Sacramento basin for the region of interest
    sacHUCs = pandas.read_csv(hucFile)
    sacHUCsWithRegions = pandas.read_csv(hucRegionFile)
    sacHUCsWithRegions = sacHUCsWithRegions.loc[:, ['HUC_12', 'AggEcoreg']]
    sacHUCsWithRegions.rename(columns={'HUC_12': 'HUC12'}, inplace=True)
    sacHUCsWithRegions.drop_duplicates(subset='HUC12', inplace=True)
    sacHUCs = pandas.merge(sacHUCs, sacHUCsWithRegions, on='HUC12')
    regionHUCs = sacHUCs[sacHUCs.AggEcoreg == region]

    # Drop region column so that we don't have an extra column and confuse the models
    regionHUCs = regionHUCs.loc[:, ['HUC12']]

    # Import prediction data
    staticData = pandas.read_csv(staticFile)
    climateData = pandas.read_csv(climateFile)

    # Build sum variables based on climateData
    climateData['p6sum'] = climateData['p1'] + climateData['p2'] + climateData['p3'] + climateData['p4'] + \
                           climateData['p5'] + climateData['p6']
    climateData['p3sum'] = climateData['p1'] + climateData['p2'] + climateData['p3']
    climateData['p2sum'] = climateData['p1'] + climateData['p2']

    # Fill missing values in static data using the HUC12 above's data, since this only happens in two variables and for
    # 3 HUC12s that are also in sacHUCs
    staticData['PERDUN'].fillna(method='ffill', inplace=True)
    staticData['PERHOR'].fillna(method='ffill', inplace=True)

    # Join to build data set for region of Sacramento basin
    climateData.rename(columns={'SITE': 'HUC12'}, inplace=True)
    regionData = pandas.merge(regionHUCs, climateData, on='HUC12')
    regionData = pandas.merge(regionData, staticData, on='HUC12')

    # Get rid of 1949 and 2011, since they have a bunch of missing climate data for oct and dec
    regionData = regionData[~ regionData.YEAR.isin([1949, 2011])]

    # Subset to just wet or dry years when trying to create a specific wet/dry model
    if wetOrDry is not None:
        yearsOfInterest = getYearsOfInterest(waterYearsFilePath, month, proportionOfInterest, wetOrDry)
        regionData = regionData[regionData.YEAR.isin(yearsOfInterest)]

    # Reorder columns to match training dataset
    columns = regionData.columns.tolist()
    newColumns = columns[:3] + columns[29:42] + columns[3:29] + columns[42:]
    regionData = regionData[newColumns]

    return regionData


def outputPredictions(applyModelResult, outputPath):

    # Build predictions output data frame
    nonFeatures = applyModelResult.testDataSet.nonFeaturesDataFrame
    predictions = applyModelResult.testPredictions
    predictionsDataFrame = pandas.DataFrame(predictions, columns=['qpred'])
    outputDataFrame = pandas.concat([nonFeatures, predictionsDataFrame], axis=1)

    # Output to csv
    outputDataFrame.to_csv(outputPath, index=False)
    return


def processSacPredictions(basePath, trainFeaturesIndex, trainLabelIndex, modelIndex, selectedFeaturesList,
                          randomSeed, modelApproach, region=None, month=None, printLog=False):

    if modelApproach not in ['wet', 'dry', 'sacramento']:
        raise ValueError("Model approach must be either 'wet', 'dry', or 'sacramento'.")

    # Get prediction DataSet information and decide where to put output
    if modelApproach in ['wet', 'dry']:
        print('Predicting for %s, %s' % (region, month.capitalize()))

        masterDataPath = basePath + region + '/' + month + '/'
        predictionDataPath = masterDataPath + 'Prediction/sacramentoData.csv'
        predictionOutputPath = masterDataPath + 'Prediction/sacramentoPredictions.csv'

    else:
        print('Predicting for %s' % (month.capitalize()))

        masterDataPath = basePath
        predictionDataPath = masterDataPath + 'Prediction/sacramentoData_' + month + '.csv'
        predictionOutputPath = masterDataPath + 'Prediction/sacramentoPredictions_' + month + '.csv'

    logOutputPath = masterDataPath + 'Prediction/predictionMethod.txt'

    predictionDataSet = mltypes.DataSet(month.capitalize() + ' Prediction Data',
                                        predictionDataPath,
                                        featuresIndex=3,
                                        labelIndex=None)

    # Train the best model and predict for the Sacramento region
    if modelApproach in ['wet', 'dry']:
        sacResult = findModelAndPredict(predictionDataSet, masterDataPath, randomSeed, trainFeaturesIndex,
                                        trainLabelIndex, selectedFeaturesList, month, region, modelIndex,
                                        printLog, logOutputPath)
    else:
        sacResult = findModelAndPredict(predictionDataSet, masterDataPath, randomSeed, trainFeaturesIndex,
                                        trainLabelIndex, selectedFeaturesList, month, modelRowIndex=modelIndex,
                                        printLog=printLog, logPath=logOutputPath)

    # Save the output
    outputPredictions(sacResult, predictionOutputPath)
    return


def aggregateSacPredictions(baseFolderList, outputFolder, outputPrefix, months, region=None):

    allPredictions = pandas.DataFrame()

    # For each half:
    for baseFolder in baseFolderList:

        # Create DataFrame for predictions for this half of the years (wet or dry)
        halfPredictions = pandas.DataFrame()

        # Concatenate month results
        for month in months:

            # Read in predictions
            if region is not None:
                predictionsFile = baseFolder + region + '/' + month + '/Prediction/sacramentoPredictions.csv'
            else:
                predictionsFile = baseFolder + 'Prediction/sacramentoPredictions_' + month + '.csv'
            monthPredictions = pandas.read_csv(predictionsFile)

            # Aggregate results for this half
            halfPredictions = halfPredictions.append(monthPredictions, ignore_index=True)

        allPredictions = allPredictions.append(halfPredictions, ignore_index=True)

    # Sort by month, then year, then HUC12
    sortedAllPredictions = allPredictions.sort(columns=['MONTH', 'YEAR', 'HUC12'])

    # Output results for this regional model to file
    if region is not None:
        outputFile = outputFolder + outputPrefix + region + '.csv'
    else:
        outputFile = outputFolder + outputPrefix + '.csv'
    sortedAllPredictions.to_csv(outputFile, index=False)

    # Return aggregated filename
    return outputFile


def formatWaterYearPredictions(waterYear, predictionsFile):

    monthNums = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    monthTexts = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']

    # Read in aggregated predictions
    allPredictionsDF = pandas.read_csv(predictionsFile)

    # Build base of dataFrame for the water year
    huc12s = allPredictionsDF.HUC12.unique()
    yearPredictionsDF = pandas.DataFrame(huc12s, columns=['HUC12'])

    for monthNum, monthText in zip(monthNums, monthTexts):

        # Find calendar year
        if monthText in ['oct', 'nov', 'dec']:
            calendarYear = waterYear - 1
        else:
            calendarYear = waterYear

        monthPredictions = allPredictionsDF[(allPredictionsDF.YEAR == calendarYear) &
                                            (allPredictionsDF.MONTH == monthNum)]
        yearPredictionsDF[monthText] = monthPredictions.qpred.values.tolist()

    yearPath = predictionsFile.split('.')[0] + str(calendarYear) + '.csv'
    yearPredictionsDF.to_csv(yearPath, index=False)
    return



