import os
import shutil
import math
import fnmatch
import copy
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

def getDryYears(filePath, month, proportionOfInterest):
    """
    Given a ranking of driest water years from driest to wettest, extract the calendar years for the driest proportion.
    :param filePath:
    :param month: string. should be written as first three letters of month, lowercase. ex: 'jul'
    :param proportionOfInterest: float between 0 and 1
    :return:
    """

    # Read in water years as ordered from driest to wettest for the Sacramento by NOAA
    waterYears = []
    with open(filePath) as file:
        for line in file.readlines():
            year = int(line)
            waterYears.append(year)

    # Get water years of interest (drier years)
    numberToExtract = math.ceil(len(waterYears) * proportionOfInterest)
    dryWaterYears = waterYears[:numberToExtract]

    # Get appropriate calendar years for the month of interest
    # (Oct, Nov, and Dec: calendar year = water year - 1. Ex: Oct 1976 is water year 1977.)
    if month in ['oct', 'nov', 'dec']:
        calendarYears = [x - 1 for x in dryWaterYears]
    else:
        calendarYears = dryWaterYears
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


def createKFoldDataSets(kFolds, masterDataPath, month, region, dryProportionOfInterest, myFeaturesIndex, myLabelIndex,
                        randomSeed=None):

    # Get dry water years
    dryYears = getDryYears(masterDataPath + 'NOAAWaterYearsDriestToWettest.csv',
                           month,
                           dryProportionOfInterest)

    # Read in original dataset with all years (with ObsID column added at the beginning before running code)
    fullDataSet = mltypes.DataSet('All Years',
                                  masterDataPath + month + '_' + region + '_all.csv',
                                  featuresIndex=myFeaturesIndex,
                                  labelIndex=myLabelIndex)

    # Subset full dataset to those years of interest
    dryDataFrame = fullDataSet.dataFrame.loc[fullDataSet.dataFrame['Year'].isin(dryYears)]
    dryDataSet = mltypes.DataSet('Dry Years',
                                 masterDataPath + month + '_' + region + '_dry.csv',
                                 'w',
                                 dataFrame=dryDataFrame,
                                 featuresIndex=myFeaturesIndex,
                                 labelIndex=myLabelIndex)

    testPathPrefix = os.path.dirname(dryDataSet.path) + '/' + month + '_' + region

    # From the dryDataSet, create k universal test sets and corresponding k dry training sets
    splitDryDataSets = mldata.kFoldSplitDataSet(dryDataSet, 5, randomSeed=randomSeed,
                                                testPathPrefix=testPathPrefix)

    # Use ObsIDs of each universal test set to subset full data set to everything else, creating k full training sets
    for fold in range(kFolds):
        universalTestDataSet = splitDryDataSets[fold].testDataSet
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


def flowModelPipeline(universalTestSetFileName, universalTestSetDescription, basePath, outputFilePath,
                      statusPrintPrefix=None, subTaskPrint=True, randomSeed=None, runScaleDatasets=True,
                      runFeatureEngineering=True, runEnsembleModels=True):

    # # Parameters
    # runPrepareDatasets=True
    # runTuneModels=True
    # runApplyModels=True
    # runScoreModels=True

    tuneScoreMethod = 'r2'
    # tuneScoreMethod = 'mean_squared_error'
    r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
    mseMethod = mltypes.ModelScoreMethod('Mean Squared Error', sklearn.metrics.mean_squared_error)
    testScoreMethods = [mseMethod, r2Method]

    myFeaturesIndex = 6
    myLabelIndex = 5

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
        varianceThresholdPoint1Config = mltypes.FeatureEngineeringConfiguration('Variance Threshold .1',
                                                                          'selection',
                                                                          sklearn.feature_selection.VarianceThreshold,
                                                                          {'threshold': .1})
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
        featureEngineeringConfigs = [varianceThresholdPoint1Config, pca20Config, pca50Config, ica50Config]

        for dataSetAssociation in dataSetAssociations:

            for featureEngineeringConfig in featureEngineeringConfigs:
                # Feature engineer training data and get transformer
                featureEngineeredTrainDataSet, transformer = mldata.engineerFeaturesForDataSet(
                    dataSetAssociation.trainDataSet,
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
    randomForestParameters = [{'n_estimators': [10, 20],
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

        # Find the maximum mean squared error
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
    scoreModelResults = mlmodel.scoreModels(applyModelResults, testScoreMethods)

    # Model testing result reporting
    if testScoreMethods[0].function == sklearn.metrics.mean_squared_error:
        sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: x.modelScores[0].score)
    else:
        sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: -x.modelScores[0].score)

    # Convert to data frame for tabulation and visualization
    scoreModelResultsDF = mlutils.createScoreDataFrame(sortedScoreModelResults)
    scoreModelResultsDF.to_csv(outputFilePath, index=False)
    return scoreModelResultsDF


def runAllModels(month, region, randomSeed=None):

    # Set parameters
    masterDataPath = 'AllMonths/' + region + '/' + month + '/'
    dryProportionOfInterest = 0.5
    myFeaturesIndex = 6
    myLabelIndex = 5
    kFolds = 5

    # Create my 5 test/train folds
    createKFoldDataSets(kFolds, masterDataPath, month, region, dryProportionOfInterest,
                        myFeaturesIndex, myLabelIndex, randomSeed)

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
                                                    outputFilePath=masterDataPath + 'Output/scoreModelResults_' + str(fold) + '.csv',
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



