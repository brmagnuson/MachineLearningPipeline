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
pathToTrainingData = 'ExampleData/myTrainingData.csv'
pathToTestingData = 'ExampleData/myTestingData.csv'

# Read data sets
trainingData = mlutilities.types.DataSet('My Training Data',
                                         pathToTrainingData)
testingData = mlutilities.types.DataSet('My Testing Data',
                                        pathToTestingData)

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
tunedModelResults = mlutilities.modeling.tuneModels([trainingData],
                                                    [rfConfig, knnConfig])

# Apply the tuned models to some test data
applyModelConfigs = []
for tuneModelResult in tunedModelResults:
    applyModelConfig = mlutilities.types.ApplyModelConfiguration(tuneModelResult.description,
                                                                 tuneModelResult.modellingMethod,
                                                                 tuneModelResult.parameters,
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
                               'ExampleData/rSquared.png')


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


