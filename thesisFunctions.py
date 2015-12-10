import os
import math
import fnmatch
import pickle
import copy
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


def flowModelPipeline(universalTestSetFileName, universalTestSetDescription, basePath, picklePath, statusPrintPrefix=None,
                      randomSeed=None):

    # Parameters
    runPrepareDatasets = True
    runScaleDatasets = True
    runFeatureEngineering = True
    runTuneModels = True
    runApplyModels = True
    runEnsembleModels = True
    runScoreModels = True

    tuneScoreMethod = 'r2'
    # tuneScoreMethod = 'mean_squared_error'
    r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
    mseMethod = mltypes.ModelScoreMethod('Mean Squared Error', sklearn.metrics.mean_squared_error)
    testScoreMethods = [mseMethod, r2Method]

    myFeaturesIndex = 6
    myLabelIndex = 5

    if runPrepareDatasets:
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

        pickle.dump(dataSetAssociations, open(picklePath + 'dataSetAssociations.p', 'wb'))

    dataSetAssociations = pickle.load(open(picklePath + 'dataSetAssociations.p', 'rb'))

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

        pickle.dump(scaledDataSetAssociations, open(picklePath + 'scaledDataSetAssociations.p', 'wb'))

    scaledDataSetAssociations = pickle.load(open(picklePath + 'scaledDataSetAssociations.p', 'rb'))

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

        pickle.dump(featureEngineeredDataSetAssociations, open(picklePath + 'featureEngineeredDataSetAssociations.p', 'wb'))

    featureEngineeredDataSetAssociations = pickle.load(open(picklePath + 'featureEngineeredDataSetAssociations.p', 'rb'))
    dataSetAssociations += featureEngineeredDataSetAssociations

    # Tune models
    if runTuneModels:
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

                print(statusPrintPrefix, 'Tuning (%s of %s):' % (counter, total), tuneModelConfig.description, 'for', dataSetAssociation.trainDataSet.description)
                tuneModelResult = mlmodel.tuneModel(dataSetAssociation.trainDataSet, tuneModelConfig, randomSeed)
                tuneModelResults.append(tuneModelResult)
                counter += 1

        pickle.dump(tuneModelResults, open(picklePath + 'tuneModelResults.p', 'wb'))

    tuneModelResults = pickle.load(open(picklePath + 'tuneModelResults.p', 'rb'))

    # Model tuning result reporting
    if tuneScoreMethod == 'mean_squared_error':
        sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: x.bestScore)
    else:
        sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: -x.bestScore)

    # Apply models
    if runApplyModels:

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
        applyModelResults = mlmodel.applyModels(applyModelConfigs)

        pickle.dump(applyModelResults, open(picklePath + 'applyModelResults.p', 'wb'))

    applyModelResults = pickle.load(open(picklePath + 'applyModelResults.p', 'rb'))

    # Score models
    if runScoreModels:
        print(statusPrintPrefix, 'Scoring models on test data.')
        scoreModelResults = mlmodel.scoreModels(applyModelResults, testScoreMethods)
        pickle.dump(scoreModelResults, open(picklePath + 'scoreModelResults.p', 'wb'))

    scoreModelResults = pickle.load(open(picklePath + 'scoreModelResults.p', 'rb'))

    # Model testing result reporting
    if testScoreMethods[0].function == sklearn.metrics.mean_squared_error:
        sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: x.modelScores[0].score)
    else:
        sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: -x.modelScores[0].score)

    # Convert to data frame for tabulation and visualization
    scoreModelResultsDF = mlutils.createScoreDataFrame(sortedScoreModelResults)
    return scoreModelResultsDF
