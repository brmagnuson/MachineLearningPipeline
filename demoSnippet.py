import sklearn.feature_selection
import sklearn.decomposition
import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
import mlutilities.types
import mlutilities.dataTransformation
import mlutilities.modeling
import mlutilities.utilities

# Paths
pathToData = 'ExampleData/myData.csv'

# Read data sets
myData = mlutilities.types.DataSet('My Training Data',
                                   pathToData)
splitData = mlutilities.dataTransformation.splitDataSet(myData,
                                                        testProportion=0.3,
                                                        randomSeed=89271)
trainingData = splitData.trainDataSet
testingData = splitData.testDataSet

# Tune models for training data set
tuneScoringMethod = 'r2'
rfParameters = [{'n_estimators': [50, 75, 100]}]
rfMethod = mlutilities.types.ModellingMethod('Random Forest',
                                             sklearn.ensemble.RandomForestRegressor)
rfConfig = mlutilities.types.TuneModelConfiguration('Tune Random Forest',
                                                    rfMethod,
                                                    rfParameters,
                                                    tuneScoringMethod)
knnParameters = [{'n_neighbors': [2, 5]}]
knnMethod = mlutilities.types.ModellingMethod('K Nearest Neighbors',
                                              sklearn.neighbors.KNeighborsRegressor)
knnConfig = mlutilities.types.TuneModelConfiguration('Tune KNN',
                                                     knnMethod,
                                                     knnParameters,
                                                     tuneScoringMethod)

predictorConfigs = [rfConfig, knnConfig]
tunedModelResults = mlutilities.modeling.tuneModels([trainingData],
                                                    predictorConfigs)

# Apply the tuned models to some test data
applyModelConfigs = []
for tunedModelResult in tunedModelResults:
    applyModelConfig = mlutilities.types.ApplyModelConfiguration(tunedModelResult.description,
                                                                 tunedModelResult.modellingMethod,
                                                                 tunedModelResult.parameters,
                                                                 trainingData,
                                                                 testingData)
    applyModelConfigs.append(applyModelConfig)
applyModelResults = mlutilities.modeling.applyModels(applyModelConfigs)

# Score the test results
r2Method = mlutilities.types.ModelScoreMethod('R Squared',
                                              sklearn.metrics.r2_score)
meanOEMethod = mlutilities.types.ModelScoreMethod('Mean O/E',
                                                  mlutilities.modeling.meanObservedExpectedScore)
testScoringMethods = [r2Method, meanOEMethod]
testScoreModelResults = mlutilities.modeling.scoreModels(applyModelResults,
                                                         testScoringMethods)

# Create a dataframe where each row is a different dataset-model combination
scoreModelResultsDF = mlutilities.utilities.createScoreDataFrame(testScoreModelResults)
print(scoreModelResultsDF)

# Visualization
mlutilities.utilities.barChart(scoreModelResultsDF,
                               'R Squared',
                               'R Squared for Each Model',
                               'ExampleData/rSquared.png',
                               '#2d974d')

mlutilities.utilities.scatterPlot(scoreModelResultsDF,
                                  'Mean O/E',
                                  'R Squared',
                                  'Mean O/E by R Squared for Each Model',
                                  'ExampleData/meanOEbyRSquared.png',
                                  '#2d974d')


# Some other things you can do

# Scale data
scaledTrainingData, scaler = mlutilities.dataTransformation.scaleDataSet(trainingData)
scaledTestingData = mlutilities.dataTransformation.scaleDataSetByScaler(testingData, scaler)

# Perform feature engineering
pcaConfig = mlutilities.types.FeatureEngineeringConfiguration('PCA n5',
                     'extraction', sklearn.decomposition.PCA, {'n_components': 5})

pcaTrainingData, transformer = mlutilities.dataTransformation.\
    engineerFeaturesForDataSet(trainingData, pcaConfig)
pcaTestingData = mlutilities.dataTransformation.engineerFeaturesByTransformer(
                                                         testingData, transformer)

# Create stacking ensemble
predictorConfigs = []
for tunedModelResult in tunedModelResults:
    predictorConfig = mlutilities.types.PredictorConfiguration(tunedModelResult.modellingMethod.description,
                                                     tunedModelResult.modellingMethod.function,
                                                     tunedModelResult.parameters)
    predictorConfigs.append(predictorConfig)

stackMethod = mlutilities.types.ModellingMethod('Stacking Ensemble',
                                                mlutilities.types.StackingEnsemble)
stackParameters = {'basePredictorConfigurations': predictorConfigs,
                   'stackingPredictorConfiguration': predictorConfigs[0]}
stackApplyModelConfig = mlutilities.types.ApplyModelConfiguration(
    'Stacking Ensemble',
    stackMethod,
    stackParameters,
    trainingData,
    testingData)

stackResult = mlutilities.modeling.applyModel(stackApplyModelConfig)

