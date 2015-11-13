import os
import fnmatch
import pandas
import sklearn.linear_model
import sklearn.metrics
import mlutilities.types
import mlutilities.dataTransformation
import mlutilities.modeling
import mlutilities.utilities
import thesisFunctions


basePath = 'Data/'
myFeaturesIndex = 6
myLabelIndex = 5

# Get base test set from Data folder
universalTestDataSet = mlutilities.types.DataSet('Jul IntMnt Test',
                                                 basePath + 'jul_IntMnt_test.csv',
                                                 featuresIndex=myFeaturesIndex,
                                                 labelIndex=myLabelIndex)

# Get all base training sets from Data folder
baseTrainingDataSets = []
for root, directories, files in os.walk(basePath):
    for file in fnmatch.filter(files, '*_train.csv'):
        description = thesisFunctions.createDescriptionFromFileName(file)
        baseTrainingDataSet = mlutilities.types.DataSet(description,
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
    copyOfUniversalTestDataSet = mlutilities.types.DataSet(copyDescription,
                                                           copyPath,
                                                           'w',
                                                           dataFrame=universalTestDataSet.dataFrame,
                                                           featuresIndex=myFeaturesIndex,
                                                           labelIndex=myLabelIndex)
    dataSetAssociation = mlutilities.types.SplitDataSet(baseTrainingDataSet, copyOfUniversalTestDataSet)
    dataSetAssociations.append(dataSetAssociation)

# Scale data based on the training set
scaledDataSetAssociations = []
for dataSetAssociation in dataSetAssociations:

    # Scale training data and get scaler
    scaledTrainDataSet, scaler = mlutilities.dataTransformation.scaleDataSet(dataSetAssociation.trainDataSet)

    # Scale testing data using scaler
    scaledTestDataSet = mlutilities.dataTransformation.scaleDataSetByScaler(dataSetAssociation.testDataSet, scaler)

    # Associate the data sets
    scaledDataSetAssociation = mlutilities.types.SplitDataSet(scaledTrainDataSet, scaledTestDataSet)
    scaledDataSetAssociations.append(scaledDataSetAssociation)

dataSetAssociations += scaledDataSetAssociations

# Feature engineering
# featureEngineeredDataSetAssociations = []
# for dataSetAssociation in dataSetAssociations:
#
#     # Feature engineer training data and get

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
