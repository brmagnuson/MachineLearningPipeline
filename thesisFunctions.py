import sys
import os
import shutil
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
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils


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


def createKFoldDataSets(kFolds, masterDataPath, month, region, proportionOfInterest, myFeaturesIndex, myLabelIndex,
                        wetOrDry='dry', randomSeed=None):

    # Get water years of interest
    yearsOfInterest = getYearsOfInterest(masterDataPath + 'NOAAWaterYearsDriestToWettest.csv',
                                  month,
                                  proportionOfInterest,
                                  wetOrDry)

    # Read in original dataset with all years (with ObsID column added at the beginning before running code)
    fullDataSet = mltypes.DataSet('All Years',
                                  masterDataPath + month + '_' + region + '_all.csv',
                                  featuresIndex=myFeaturesIndex,
                                  labelIndex=myLabelIndex)

    # Subset full dataset to those years of interest
    yearsOfInterestDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(yearsOfInterest)]
    if wetOrDry == 'dry':
        yearsOfInterestDescription = 'Dry Years'
    elif wetOrDry == 'wet':
        yearsOfInterestDescription = 'Wet Years'
    else:
        raise ValueError('wetOrDry had value other than "wet" or "dry".')

    yearsOfInterestDataSet = mltypes.DataSet(yearsOfInterestDescription,
                                             masterDataPath + month + '_' + region + '_' + wetOrDry + '.csv',
                                             'w',
                                             dataFrame=yearsOfInterestDataFrame,
                                             featuresIndex=myFeaturesIndex,
                                             labelIndex=myLabelIndex)

    testPathPrefix = os.path.dirname(yearsOfInterestDataSet.path) + '/' + month + '_' + region

    # From the subset DataSet, create k universal test sets and corresponding k wet/dry (depending on wetOrDry) training sets
    splitDataSets = mldata.kFoldSplitDataSet(yearsOfInterestDataSet, 5, randomSeed=randomSeed,
                                                testPathPrefix=testPathPrefix)

    # Use ObsIDs of each universal test set to subset full data set to everything else, creating k full training sets
    for fold in range(kFolds):
        universalTestDataSet = splitDataSets[fold].testDataSet
        universalTestObsIds = universalTestDataSet.dataFrame.ObsID.values
        fullTrainDataFrame = fullDataSet.dataFrame.loc[~fullDataSet.dataFrame.ObsID.isin(universalTestObsIds)]

        # Write this out to the proper folder.
        fullTrainDataSet = mltypes.DataSet('All Years Training Set',
                                           masterDataPath + month + '_' + region + '_all_' + str(fold) + '_train.csv',
                                           'w',
                                           dataFrame=fullTrainDataFrame,
                                           featuresIndex=myFeaturesIndex,
                                           labelIndex=myLabelIndex)


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


def flowModelPipeline(universalTestSetFileName, universalTestSetDescription, basePath, scoreOutputFilePath,
                      myFeaturesIndex, myLabelIndex, statusPrintPrefix='', subTaskPrint=True, randomSeed=None,
                      runScaleDatasets=True, runFeatureEngineering=True, runEnsembleModels=True):

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

    # runPrepareDatasets=True
    # runTuneModels=True
    # runApplyModels=True
    # runScoreModels=True

    # Parameters
    selectedFeatureList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                           't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                           'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                           'WD_BASIN']
    tuneScoreMethod = 'r2'
    # tuneScoreMethod = 'mean_squared_error'
    r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
    mseMethod = mltypes.ModelScoreMethod('Mean Squared Error', sklearn.metrics.mean_squared_error)
    testScoreMethods = [mseMethod, r2Method]

    # Prepare datasets
    # if runPrepareDatasets:
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
        varianceThresholdPoint1Config = mltypes.FeatureEngineeringConfiguration('Variance Threshold .08',
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
        # ica20Config = mltypes.FeatureEngineeringConfiguration('ICA n20',
        #                                                     'extraction',
        #                                                     sklearn.decomposition.FastICA,
        #                                                     {'n_components': 20, 'max_iter': 2500, 'random_state': randomSeed})
        ica50Config = mltypes.FeatureEngineeringConfiguration('ICA n50',
                                                              'extraction',
                                                              sklearn.decomposition.FastICA,
                                                              {'n_components': 50, 'max_iter': 2500, 'random_state': randomSeed})
        expertSelectedConfig = mltypes.FeatureEngineeringConfiguration('Expert Selection',
                                                                       'selection',
                                                                       mltypes.ExtractSpecificFeatures,
                                                                       {'featureList': selectedFeatureList})
        featureEngineeringConfigs = [varianceThresholdPoint1Config, pca20Config, pca50Config, ica50Config, expertSelectedConfig]

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
    # if runTuneModels:
    print(statusPrintPrefix, 'Tuning models.')

    ridgeParameters = [{'alpha': [0.1, 0.5, 1.0],
                        'normalize': [True, False]}]
    ridgeMethod = mltypes.ModellingMethod('Ridge Regression',
                                          sklearn.linear_model.Ridge)
    ridgeConfig = mltypes.TuneModelConfiguration(ridgeMethod.description,
                                                 ridgeMethod,
                                                 ridgeParameters,
                                                 tuneScoreMethod)
    randomForestParameters = [{'n_estimators': [10, 20, 50],
                               'max_features': [10, 'sqrt'],
                               'random_state': [randomSeed]}]
    randomForestMethod = mltypes.ModellingMethod('Random Forest',
                                                 sklearn.ensemble.RandomForestRegressor)
    randomForestConfig = mltypes.TuneModelConfiguration(randomForestMethod.description,
                                                        randomForestMethod,
                                                        randomForestParameters,
                                                        tuneScoreMethod)
    kNeighborsParameters = [{'n_neighbors': [2, 5, 10],
                             'metric': ['minkowski', 'euclidean']}]
    kNeighborsMethod = mltypes.ModellingMethod('K Nearest Neighbors',
                                               sklearn.neighbors.KNeighborsRegressor)
    kNeighborsConfig = mltypes.TuneModelConfiguration(kNeighborsMethod.description,
                                                      kNeighborsMethod,
                                                      kNeighborsParameters,
                                                      tuneScoreMethod)

    tuneModelConfigs = [ridgeConfig, randomForestConfig, kNeighborsConfig]

    counter = 1
    total = len(dataSetAssociations) * len(tuneModelConfigs)
    tuneModelResults = []
    for dataSetAssociation in dataSetAssociations:
        for tuneModelConfig in tuneModelConfigs:

            if subTaskPrint:
                print(statusPrintPrefix, 'Tuning (%s of %s):' % (counter, total), tuneModelConfig.description, 'for', dataSetAssociation.trainDataSet.description)
            tuneModelResult = mlmodel.tuneModel(dataSetAssociation.trainDataSet, tuneModelConfig, randomSeed)
            tuneModelResults.append(tuneModelResult)
            counter += 1

    # # Model tuning result reporting
    # if tuneScoreMethod == 'mean_squared_error':
    #     sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: x.bestScore)
    # else:
    #     sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: -x.bestScore)

    # Apply models
    # if runApplyModels:
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
        if testDataSet == None:
            raise Exception('No SplitDataSet found matching this training DataSet:\n' + trainDataSet)

        applyModelConfig = mltypes.ApplyModelConfiguration('Apply ' + tuneModelResult.description.replace('Train', 'Test'),
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

            # Find models associated with that DataSet and get their information to build predictor configs for ensembles
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
                        # R squared can be negative, and weights should all be positive
                        weight = tuneModelResult.bestScore
                    weights.append(weight)

                    # If tuneModelResult has a better score than previously seen, make it the stacked predictor config
                    if weight > bestWeight:
                        bestWeight = weight
                        stackingPredictorConfig = copy.deepcopy(predictorConfig)

                        # Hack: If the number of models I'm stacking is fewer than max_features, RandomForestRegressor
                        # will error out.
                        if type(stackingPredictorConfig.predictorFunction()) == type(sklearn.ensemble.RandomForestRegressor()):
                            stackingPredictorConfig.parameters['max_features'] = None


            # Create averaging ensemble
            averagingEnsembleModellingMethod = mltypes.ModellingMethod('Averaging Ensemble',
                                                                       mltypes.AveragingEnsemble)
            averagingEnsembleParameters = {'predictorConfigurations': predictorConfigs,
                                           'weights': weights}
            averagingEnsembleApplyModelConfig = mltypes.ApplyModelConfiguration(
                'Apply Averaging Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train', 'Test'),
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
                'Apply Stacking Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train', 'Test'),
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
                'Apply OF Stacking Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train', 'Test'),
                stackingOFEnsembleModellingMethod,
                stackingOFEnsembleParameters,
                dataSetAssociation.trainDataSet,
                dataSetAssociation.testDataSet
            )
            ensembleApplyModelConfigs.append(stackingOFEnsembleApplyModelConfig)

        # Add ensemble configs to the rest of the ApplyModelConfigs
        applyModelConfigs += ensembleApplyModelConfigs

    # Apply models to test data
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
    scoreModelResultsDF.to_csv(scoreOutputFilePath, index=False)
    return scoreModelResultsDF


def runKFoldPipeline(month, region, baseDirectoryPath, myFeaturesIndex, myLabelIndex,
                     kFolds=5, wetOrDry='dry', randomSeed=None):

    """
    Splits each region-month base dataset into k-fold test/train sets and runs the pipeline for each one.
    :param month:
    :param region:
    :param randomSeed:
    :return:
    """

    # Set parameters
    masterDataPath = baseDirectoryPath + region + '/' + month + '/'
    proportionOfInterest = 0.5

    # Create my 5 test/train folds
    createKFoldDataSets(kFolds, masterDataPath, month, region, proportionOfInterest,
                        myFeaturesIndex, myLabelIndex, wetOrDry, randomSeed)

    # Run pipeline for each fold of the data
    allFoldScoreModelResultsDFs = []
    for fold in range(kFolds):

        copyFoldDataSets(fold, masterDataPath)

        # Run pipeline for those datasets
        universalTestSetFileName = month + '_' + region + '_test.csv'
        universalTestSetDescription = month.capitalize() + ' ' + region + ' Test'
        foldScoreModelResultsDF = flowModelPipeline(universalTestSetFileName=universalTestSetFileName,
                                                    universalTestSetDescription=universalTestSetDescription,
                                                    basePath=masterDataPath + 'CurrentFoldData/',
                                                    scoreOutputFilePath=masterDataPath + 'Output/scoreModelResults_' + str(fold) + '.csv',
                                                    myFeaturesIndex=myFeaturesIndex,
                                                    myLabelIndex=myLabelIndex,
                                                    statusPrintPrefix=region + ' ' + month.capitalize() + ' K-fold #' + str(fold),
                                                    subTaskPrint=False,
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
    sortedAverageResultsDF.to_csv(masterDataPath + 'Output/scoreModelResults_average.csv', index=False)


def getSKLearnFunction(description):
    """
    Matches a model description with an sklearn function object.
    :param description:
    :return:
    """
    if description == 'Ridge Regression':
        predictorFunction = sklearn.linear_model.Ridge
    elif description == 'Random Forest':
        predictorFunction = sklearn.ensemble.RandomForestRegressor
    elif description == 'K Nearest Neighbors':
        predictorFunction = sklearn.neighbors.KNeighborsRegressor
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
                weights = re.search("'weights': (.*?])", modelParameters).group(1)
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
            predictorTypeList = ['Ridge Regression', 'Random Forest', 'K Nearest Neighbors']
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

            if originalFeaturesSearch == None:

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
            predictorTypeList = ['Ridge Regression', 'Random Forest', 'K Nearest Neighbors']
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
                stackingPredictorDescription = re.search("'stackingPredictorConfiguration': (.*?) {", modelParameters).group(1)

            except AttributeError:
                raise Exception('Stacking Predictor not found in Stacking Ensemble.')

            # Match the stacking predictor to its base predictor config
            for predictorConfig in predictorConfigs:
                if stackingPredictorDescription == predictorConfig.description:
                    stackingPredictorConfig = copy.deepcopy(predictorConfig)
                    break

            # Hack: If the number of models I'm stacking in a random forest stackingPredictorConfig is fewer than
            # max_features, RandomForestRegressor will error out.
            if type(stackingPredictorConfig.predictorFunction()) == type(sklearn.ensemble.RandomForestRegressor()):
                stackingPredictorConfig.parameters['max_features'] = None

            # Build pieces for applyModelConfig
            trainModelParameters = {'basePredictorConfigurations': predictorConfigs,
                                    'stackingPredictorConfiguration': stackingPredictorConfig,
                                    'includeOriginalFeatures': includeOriginalFeatures}
            modelMethod = mltypes.ModellingMethod(modelDescription, mltypes.StackingEnsemble)

    elif any(x in modelDescription for x in ['Random Forest', 'Ridge Regression', 'K Nearest Neighbors']):

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


def findModelAndPredict(predictionDataSet, basePath, month, region, randomSeed, myFeaturesIndex, myLabelIndex,
                        selectedFeaturesList, modelRowIndex=0):

    masterDataPath = basePath + region + '/' + month + '/'

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

    # Find appropriate dataset (either all or dry, depending on above results) and copy to Prediction folder
    if 'Dry' in trainDataSetDescription:
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
    applyModelResult = mlmodel.applyModel(applyModelConfig)
    return applyModelResult


def prepSacramentoData(month, region, basePath):

    hucFile = '../SacramentoData/Sacramento_basin_huc12_v2.csv'
    staticFile = '../SacramentoData/static_vars/h12.static.vars.csv'
    climateBasePath = '../SacramentoData/climate_vars/'

    # Find month number to build climate file path
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthNumber = months.index(month) + 1
    monthNumber = str(monthNumber)
    if len(monthNumber) < 2:
        monthNumber = '0' + monthNumber
    climateFile = climateBasePath + 'm' + monthNumber + '_HUC12.clim.data.' + month + '.csv'

    # Import HUCs for the Sacramento basin
    sacHUCs = pandas.read_csv(hucFile)

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

    # Join to build data set for Sacramento basin
    climateData.rename(columns={'SITE':'HUC12'}, inplace=True)
    sacData = pandas.merge(sacHUCs, climateData, on='HUC12')
    sacData = pandas.merge(sacData, staticData, on='HUC12')

    # Get rid of 1949 and 2011, since they have a bunch of missing climate data for oct and dec
    sacData = sacData[ ~ sacData.YEAR.isin([1949, 2011])]

    # Reorder columns to match training dataset
    columns = sacData.columns.tolist()
    newColumns = columns[:3] + columns[29:42] + columns[3:29] + columns[42:]
    sacData = sacData[newColumns]

    # Output to Prediction folder
    predictionFilePath = basePath + region + '/' + month + '/Prediction/sacramentoData.csv'
    sacData.to_csv(predictionFilePath, index=False)

def outputPredictions(applyModelResult, outputPath):

    # Build predictions output data frame
    nonFeatures = applyModelResult.testDataSet.nonFeaturesDataFrame
    predictions = applyModelResult.testPredictions
    predictionsDataFrame = pandas.DataFrame(predictions, columns=['qmeas'])
    outputDataFrame = pandas.concat([nonFeatures, predictionsDataFrame], axis=1)

    # Output to csv
    outputDataFrame.to_csv(outputPath, index=False)

def processSacPredictions(basePath, region, month, randomSeed, trainFeaturesIndex, trainLabelIndex,
                          selectedFeaturesList, modelIndex):

    print('Predicting for %s, %s' % (region, month.capitalize()))

    predictionPath = basePath + region + '/' + month + '/Prediction/sacramentoData.csv'
    predictionDataSet = mltypes.DataSet(month.title() + ' Prediction Data',
                                        predictionPath,
                                        featuresIndex=3,
                                        labelIndex=None)

    sacResult = findModelAndPredict(predictionDataSet, basePath, month, region, randomSeed, trainFeaturesIndex,
                                    trainLabelIndex, selectedFeaturesList, modelIndex)

    outputPath = basePath + region + '/' + month + '/Prediction/sacramentoPredictions.csv'
    outputPredictions(sacResult, outputPath)
