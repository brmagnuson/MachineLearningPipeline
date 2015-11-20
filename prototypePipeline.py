import os
import fnmatch
import pickle
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldataTrans
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions

# Parameters
runPrepareDatasets = False
runScaleDatasets = False
runFeatureEngineering = False
runTestTrainSplit = False
runTuneModels = False
runApplyModels = False
runEnsembleModels = False
runScoreModels = False
runVisualization = False

# tuneScoreMethod = 'r2'
tuneScoreMethod = 'mean_squared_error'
r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
mseMethod = mltypes.ModelScoreMethod('Mean Squared Error', sklearn.metrics.mean_squared_error)
testScoreMethods = [mseMethod, r2Method]

picklePath = 'Pickles/'
basePath = 'Data/'
myFeaturesIndex = 6
myLabelIndex = 5

if runPrepareDatasets:
    print('Preparing input data sets.')

    # Get base test set from Data folder
    universalTestDataSet = mltypes.DataSet('Jul IntMnt Test',
                                           basePath + 'jul_IntMnt_test.csv',
                                           featuresIndex=myFeaturesIndex,
                                           labelIndex=myLabelIndex)

    # Get all base training sets from Data folder
    baseTrainingDataSets = []
    for root, directories, files in os.walk(basePath):
        for file in fnmatch.filter(files, '*_train.csv'):
            description = thesisFunctions.createDescriptionFromFileName(file)
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
    print('Scaling data sets.')
    for dataSetAssociation in dataSetAssociations:
        # Scale training data and get scaler
        scaledTrainDataSet, scaler = mldataTrans.scaleDataSet(dataSetAssociation.trainDataSet)

        # Scale testing data using scaler
        scaledTestDataSet = mldataTrans.scaleDataSetByScaler(dataSetAssociation.testDataSet, scaler)

        # Associate the data sets
        scaledDataSetAssociation = mltypes.SplitDataSet(scaledTrainDataSet, scaledTestDataSet)
        scaledDataSetAssociations.append(scaledDataSetAssociation)

    pickle.dump(scaledDataSetAssociations, open(picklePath + 'scaledDataSetAssociations.p', 'wb'))

scaledDataSetAssociations = pickle.load(open(picklePath + 'scaledDataSetAssociations.p', 'rb'))

dataSetAssociations += scaledDataSetAssociations

# Perform feature engineering
featureEngineeredDataSetAssociations = []
if runFeatureEngineering:
    print('Engineering features.')
    varianceThresholdConfig = mltypes.FeatureEngineeringConfiguration('Variance Threshold 1',
                                                                      'selection',
                                                                      sklearn.feature_selection.VarianceThreshold,
                                                                      {'threshold': .1})
    pcaConfig = mltypes.FeatureEngineeringConfiguration('PCA n10',
                                                        'extraction',
                                                        sklearn.decomposition.PCA,
                                                        {'n_components': 10})
    featureEngineeringConfigs = [varianceThresholdConfig, pcaConfig]

    for dataSetAssociation in dataSetAssociations:

        for featureEngineeringConfig in featureEngineeringConfigs:
            # Feature engineer training data and get transformer
            featureEngineeredTrainDataSet, transformer = mldataTrans.engineerFeaturesForDataSet(
                dataSetAssociation.trainDataSet,
                featureEngineeringConfig)
            # Transform testing data using transformer
            featureEngineeredTestDataSet = mldataTrans.engineerFeaturesByTransformer(dataSetAssociation.testDataSet,
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
    print('Tuning models.')

    ridgeParameters = [{'alpha': [0.1, 0.5, 1.0],
                        'normalize': [True, False]}]
    ridgeMethod = mltypes.ModellingMethod('Ridge Regression',
                                          sklearn.linear_model.Ridge)
    ridgeConfig = mltypes.TuneModelConfiguration('Ridge Regression',
                                                 ridgeMethod,
                                                 ridgeParameters,
                                                 tuneScoreMethod)
    randomForestParameters = [{'n_estimators': [10, 20],
                               'max_features': [10, 'sqrt']}]
    randomForestMethod = mltypes.ModellingMethod('Random Forest',
                                                 sklearn.ensemble.RandomForestRegressor)
    randomForestConfig = mltypes.TuneModelConfiguration('Random Forest',
                                                        randomForestMethod,
                                                        randomForestParameters,
                                                        tuneScoreMethod)
    tuneModelConfigs = [ridgeConfig, randomForestConfig]

    tuneModelResults = []
    for dataSetAssociation in dataSetAssociations:
        for tuneModelConfig in tuneModelConfigs:
            tuneModelResult = mlmodel.tuneModel(dataSetAssociation.trainDataSet, tuneModelConfig)
            tuneModelResults.append(tuneModelResult)

    pickle.dump(tuneModelResults, open(picklePath + 'tuneModelResults.p', 'wb'))

tuneModelResults = pickle.load(open(picklePath + 'tuneModelResults.p', 'rb'))


# Model tuning result reporting
if tuneScoreMethod == 'mean_squared_error':
    sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: x.bestScore)
else:
    sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: -x.bestScore)
# for item in sortedTuneModelResults:
#     print(item)
#     print()

# Apply models
if runApplyModels:

    print('Applying models to test data.')

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

    # Build ensemble-averaging ApplyModelConfigurations
    if runEnsembleModels:

        # For each base DataSet, find its matching model functions and parameters
        ensembleModellingMethod = mltypes.ModellingMethod('Averaging Ensemble',
                                                          mltypes.AveragingEnsemble)
        ensembleApplyModelConfigs = []
        for dataSetAssociation in dataSetAssociations:
            predictorConfigs = []

            # Find models associated with that DataSet and get their information
            for applyModelConfig in applyModelConfigs:
                if dataSetAssociation.trainDataSet == applyModelConfig.trainDataSet:
                    predictorConfig = mltypes.PredictorConfiguration(applyModelConfig.modellingMethod.description,
                                                                     applyModelConfig.modellingMethod.function,
                                                                     applyModelConfig.parameters)
                    predictorConfigs.append(predictorConfig)

            # Create dictionary of ensemble parameters that will be unpacked in applyModel()
            ensembleParameters = {'predictorConfigurations': predictorConfigs}

            ensembleApplyModelConfig = mltypes.ApplyModelConfiguration(
                'Apply Averaging Ensemble for DataSet: ' + dataSetAssociation.trainDataSet.description.replace('Train', 'Test'),
                ensembleModellingMethod,
                ensembleParameters,
                dataSetAssociation.trainDataSet,
                dataSetAssociation.testDataSet
            )
            ensembleApplyModelConfigs.append(ensembleApplyModelConfig)

        # Add ensemble configs to the rest of the ApplyModelConfigs
        applyModelConfigs += ensembleApplyModelConfigs

    # Apply models to test data
    applyModelResults = mlmodel.applyModels(applyModelConfigs)

    pickle.dump(applyModelConfigs, open(picklePath + 'applyModelConfigs.p', 'wb'))
    pickle.dump(applyModelResults, open(picklePath + 'applyModelResults.p', 'wb'))

applyModelConfigs = pickle.load(open(picklePath + 'applyModelConfigs.p', 'rb'))
applyModelResults = pickle.load(open(picklePath + 'applyModelResults.p', 'rb'))

# Score models
if runScoreModels:
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
scoreModelResultsDF.to_csv('Output/scoreModelResults.csv', index=False)

# Visualization
scoreModelResultsDF['RMSE'] = scoreModelResultsDF['Mean Squared Error'].map(lambda x: x ** (1 / 2))
dryYearScoreModelResultsDF = scoreModelResultsDF[scoreModelResultsDF['Base DataSet'].str.contains('Dry')]

if runVisualization:
    mlutils.scatterPlot(scoreModelResultsDF,
                        'Mean Squared Error',
                        'R Squared',
                        'MSE by R Squared for Each Model',
                        'Output/mseByR2AllModels.png')
    mlutils.scatterPlot(dryYearScoreModelResultsDF,
                        'Mean Squared Error',
                        'R Squared',
                        'MSE by R Squared for Each Model (Dry Year Models Only',
                        'Output/mseByR2DryModels.png')
    mlutils.scatterPlot(scoreModelResultsDF,
                        'RMSE',
                        'R Squared',
                        'RMSE by R Squared for Each Model',
                        'Output/rmseByR2AllModels.png')
    mlutils.scatterPlot(dryYearScoreModelResultsDF,
                        'RMSE',
                        'R Squared',
                        'RMSE by R Squared for Each Model (Dry Year Models Only',
                        'Output/rmseByR2DryModels.png')

    mlutils.barChart(scoreModelResultsDF,
                     'Mean Squared Error',
                     'MSE for Each Model',
                     'Output/meanSquaredError.png')
    mlutils.barChart(scoreModelResultsDF,
                     'RMSE',
                     'Root Mean Squared Error for Each Model',
                     'Output/rootMeanSquaredError.png')
    mlutils.barChart(scoreModelResultsDF,
                     'R Squared',
                     'R Squared for Each Model',
                     'Output/rSquared.png')
