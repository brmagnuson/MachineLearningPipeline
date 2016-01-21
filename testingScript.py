import os
import shutil
import re
import fnmatch
import copy
import pandas
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions


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

basePath = 'AllMonthsDryHalf/'
month = 'apr'
region = 'IntMnt'
randomSeed = 47392
myFeaturesIndex = 6
myLabelIndex = 5
modelIndex = 0

selectedFeatureList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                       'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                       'WD_BASIN']

masterDataPath = basePath + region + '/' + month + '/'

# Read in score model results files.
averageFile = masterDataPath + 'Output/scoreModelResults_average.csv'
allFile = masterDataPath + 'Output/scoreModelResults_all.csv'
averageResults = pandas.read_csv(averageFile)
allResults = pandas.read_csv(allFile)

# Get model with highest average R2
bestModel = averageResults.iloc[modelIndex]
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
    print('Scaling training data')
    print()
    scaledTrainDataSet, scaler = mldata.scaleDataSet(trainDataSet)
    trainDataSet = scaledTrainDataSet

# Feature engineer if necessary
if 'features selected' in trainDataSetDescription:

    # Extract feature engineering information
    featureEngineeringDescription = trainDataSetDescription.split('via')[1].strip().split(',')[0]
    print('Feature engineering training data via', featureEngineeringDescription)
    print()

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
            featureEngineeringParameters = {'threshold': .1}
        else:
            featureEngineeringMethod = mltypes.ExtractSpecificFeatures
            featureEngineeringParameters = {'featureList': selectedFeatureList}

    else:
        raise Exception('Feature engineering method not recognized.')

    featureEngineeringConfig = mltypes.FeatureEngineeringConfiguration(featureEngineeringDescription,
                                                                       selectionOrExtraction,
                                                                       featureEngineeringMethod,
                                                                       featureEngineeringParameters)
    featureEngineeredTrainDataSet, transformer = mldata.engineerFeaturesForDataSet(trainDataSet,
                                                                                   featureEngineeringConfig)
    trainDataSet = featureEngineeredTrainDataSet

print('Training:', trainDataSet)
print()

# Build apply model configuration
if 'Ensemble' in trainModelDescription:

    # Build model method and parameters for averaging or stacking ensembles
    if trainModelDescription == 'Averaging Ensemble':

        # Parse trainModelParameters string for averaging ensemble so that its pieces can be correctly evaluated.

        try:
            # Find weights in trainModelParameters string and convert to list
            weights = re.search("'weights': (.*?])", trainModelParameters).group(1)
            weights = eval(weights)

        except AttributeError:
            raise Exception('Weights not found in Averaging Ensemble.')

        try:
            # Find predictor configurations
            predictorConfigsString = re.search("'predictorConfigurations': (.*?])", trainModelParameters).group(1)

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
        modelMethod = mltypes.ModellingMethod(trainModelDescription, mltypes.AveragingEnsemble)

    else:

        # Parse trainModelParameters string for stacking ensemble so that its pieces can be correctly evaluated.

        try:
            # Should we include original features?
            includeOriginalFeatures = re.search("'includeOriginalFeatures': (.*?),", trainModelParameters).group(1)
            includeOriginalFeatures = eval(includeOriginalFeatures)

        except AttributeError:
            raise Exception('Include Original Features option not found in Stacking Ensemble.')

        try:
            # Find predictor configurations
            predictorConfigsString = re.search("'basePredictorConfigurations': (.*?])", trainModelParameters).group(1)

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
            stackingPredictor = re.search("'stackingPredictorConfiguration': (.*?) {", trainModelParameters).group(1)

        except AttributeError:
            raise Exception('Stacking Predictor not found in Stacking Ensemble.')

        # Match the stacking predictor to its base predictor config
        for predictorConfig in predictorConfigs:
            if stackingPredictor == predictorConfig.description:
                stackingPredictorConfig = copy.deepcopy(predictorConfig)
                break


        # Build pieces for applyModelConfig
        trainModelParameters = {'basePredictorConfigurations': predictorConfigs,
                           'stackingPredictorConfiguration': stackingPredictorConfig,
                           'includeOriginalFeatures': includeOriginalFeatures}
        modelMethod = mltypes.ModellingMethod(trainModelDescription, mltypes.StackingEnsemble)

elif any(x in trainModelDescription for x in ['Random Forest', 'Ridge Regression', 'K Nearest Neighbors']):

    # Get model parameters from text string in dictionary form
    trainModelParameters = eval(trainModelParameters)

    # Build model method object
    modelFunction = getSKLearnFunction(trainModelDescription)
    modelMethod = mltypes.ModellingMethod(trainModelDescription, modelFunction)
else:
    raise Exception('Model method not recognized.')

applyModelConfig = mltypes.ApplyModelConfiguration(trainModelDescription,
                                                   modelMethod,
                                                   trainModelParameters,
                                                   trainDataSet)

print(applyModelConfig)


# Train model and predict dataset
applyModelResult = mlmodel.applyModel(applyModelConfig)





