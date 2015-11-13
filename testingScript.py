import os
import fnmatch
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldataTrans
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions


basePath = 'Data/'
myFeaturesIndex = 6
myLabelIndex = 5

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

# Scale data based on the training set
scaledDataSetAssociations = []
for dataSetAssociation in dataSetAssociations:

    # Scale training data and get scaler
    scaledTrainDataSet, scaler = mldataTrans.scaleDataSet(dataSetAssociation.trainDataSet)

    # Scale testing data using scaler
    scaledTestDataSet = mldataTrans.scaleDataSetByScaler(dataSetAssociation.testDataSet, scaler)

    # Associate the data sets
    scaledDataSetAssociation = mltypes.SplitDataSet(scaledTrainDataSet, scaledTestDataSet)
    scaledDataSetAssociations.append(scaledDataSetAssociation)

dataSetAssociations += scaledDataSetAssociations

# Feature engineering
featureEngineeredDataSetAssociations = []
varianceThresholdConfig = mltypes.FeatureEngineeringConfiguration('Variance Threshold 1',
                                                                  'selection',
                                                                  sklearn.feature_selection.VarianceThreshold,
                                                                  {'threshold':.1})
pcaConfig = mltypes.FeatureEngineeringConfiguration('PCA n10',
                                                    'extraction',
                                                    sklearn.decomposition.PCA,
                                                    {'n_components':10})
featureEngineeringConfigs = [varianceThresholdConfig, pcaConfig]
for dataSetAssociation in dataSetAssociations:

    for featureEngineeringConfig in featureEngineeringConfigs:

        # Feature engineer training data and get transformer
        featureEngineeredTrainDataSet, transformer = mldataTrans.engineerFeaturesForDataSet(dataSetAssociation.trainDataSet,
                                                                                            featureEngineeringConfig)
        # Transform testing data using transformer
        featureEngineeredTestDataSet = mldataTrans.engineerFeaturesByTransformer(dataSetAssociation.testDataSet,
                                                                                 transformer)

        # Associate the data sets
        featureEngineeredDataSetAssociation = mltypes.SplitDataSet(featureEngineeredTrainDataSet,
                                                                   featureEngineeredTestDataSet)
        featureEngineeredDataSetAssociations.append(featureEngineeredDataSetAssociation)

dataSetAssociations += featureEngineeredDataSetAssociations

for item in dataSetAssociations:
    print(item)
    print()

# Tune model
# parameters = [{'alpha': [0.1, 0.5, 1.0], 'normalize': [True, False]}]
# ridgeMethod = mlutilities.types.ModellingMethod('Ridge Regression',
#                                                 sklearn.linear_model.Ridge)
# ridgeConfig = mlutilities.types.TuneModelConfiguration('Ridge regression scored by mean_squared_error',
#                                                        ridgeMethod,
#                                                        parameterGrid=parameters,
#                                                        scoreMethod='mean_squared_error')
#
# tuneModelResult = mlutilities.modeling.tuneModel(trainDataSet, ridgeConfig)
# print(tuneModelResult)
# print()
#
# # Apply model
# applyRidgeConfig = mlutilities.types.ApplyModelConfiguration('Apply ' + tuneModelResult.description.replace('Training Set', 'Testing Set'),
#                                                              tuneModelResult.modellingMethod,
#                                                              tuneModelResult.parameters,
#                                                              trainDataSet,
#                                                              testDataSet)
# print(applyRidgeConfig)
# print()
#
# applyRidgeResult = mlutilities.modeling.applyModel(applyRidgeConfig)
# print(applyRidgeResult)
# print()
#
# mseMethod = mlutilities.types.ModelScoreMethod('Mean Squared Error', sklearn.metrics.mean_squared_error)
# testScoreMethods = [mseMethod]
# applyModelResult = mlutilities.modeling.scoreModel(applyRidgeResult, testScoreMethods)
# print(applyModelResult)
