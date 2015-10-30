import pickle
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.ensemble
import mlutilities.types
import mlutilities.dataTransformation
import mlutilities.modeling


# Get list of data sets.
print('Reading input data sets.')
picklePath = 'Pickles/'
basePath = 'Data/'
myfeaturesIndex = 5
myLabelIndex = 4
allYearsDataSet = mlutilities.types.DataSet('All Years',
                                            basePath + 'jul_IntMnt_ref.csv',
                                            featuresIndex=myfeaturesIndex,
                                            labelIndex=myLabelIndex)
dryYearsDataSet = mlutilities.types.DataSet('Dry Years',
                                            basePath + 'jul_IntMnt_driest31.csv',
                                            featuresIndex=myfeaturesIndex,
                                            labelIndex=myLabelIndex)
# regularDataSets = [allYearsDataSet, dryYearsDataSet]

# pickle.dump(regularDataSets, open(picklePath + 'regularDataSets.p', 'wb'))
regularDataSets = pickle.load(open(picklePath + 'regularDataSets.p', 'rb'))

# Get scaled data sets
print('Scaling data sets.')
# scaledDataSets = mlutilities.dataTransformation.scaleDataSets(regularDataSets)
#
# pickle.dump(scaledDataSets, open(picklePath + 'scaledDataSets.p', 'wb'))
scaledDataSets = pickle.load(open(picklePath + 'scaledDataSets.p', 'rb'))

allDataSets = regularDataSets + scaledDataSets

# Perform feature engineering
print('Engineering features.')
varianceThresholdConfiguration = mlutilities.types.FeatureEngineeringConfiguration('Variance Threshold 1',
                                                                                   'selection',
                                                                                   sklearn.feature_selection.VarianceThreshold,
                                                                                   {'threshold':.1})
pcaConfiguration = mlutilities.types.FeatureEngineeringConfiguration('PCA n10',
                                                                     'extraction',
                                                                     sklearn.decomposition.PCA,
                                                                     {'n_components':10})
featureEngineeringConfigurations = [varianceThresholdConfiguration, pcaConfiguration]
# featureEngineeredDatasets = mlutilities.dataTransformation.engineerFeaturesForDataSets(allDataSets, featureEngineeringConfigurations)
# allDataSets += featureEngineeredDatasets
#
# pickle.dump(allDataSets, open(picklePath + 'allDataSets.p', 'wb'))
allDataSets = pickle.load(open(picklePath + 'allDataSets.p', 'rb'))

# Train/test split
print('Splitting into testing & training data.')
testProportion = 0.25
# splitDataSets = mlutilities.dataTransformation.splitDataSets(allDataSets, testProportion)
#
# pickle.dump(splitDataSets, open(picklePath + 'splitDataSets.p', 'wb'))
splitDataSets = pickle.load(open(picklePath + 'splitDataSets.p', 'rb'))

trainDataSets = [splitDataSet.trainDataSet for splitDataSet in splitDataSets]

# Tune models
print('Tuning models.')
scoreMethod = 'mean_squared_error'
# scoreMethod = 'r2'
ridgeParameters = [{'alpha': [0.1, 0.5, 1.0],
                    'normalize': [True, False]}]
ridgeConfig = mlutilities.types.ModelCreationConfiguration('Ridge regression scored by mean_squared_error',
                                                           sklearn.linear_model.Ridge,
                                                           ridgeParameters,
                                                           scoreMethod)
randomForestParameters = [{'n_estimators': [10, 20],
                           'max_features': [10, 'sqrt']}]
randomForestConfig = mlutilities.types.ModelCreationConfiguration('Random Forest scored by mean_squared_error',
                                                                  sklearn.ensemble.RandomForestRegressor,
                                                                  randomForestParameters,
                                                                  scoreMethod)
modelCreationConfigs = [ridgeConfig, randomForestConfig]

# tunedModelConfigs = mlutilities.modeling.tuneModels(trainDataSets, modelCreationConfigs)
#
# pickle.dump(tunedModelConfigs, open(picklePath + 'tunedModelConfigs.p', 'wb'))
tunedModelConfigs = pickle.load(open(picklePath + 'tunedModelConfigs.p', 'rb'))

# Model tuning result reporting
if scoreMethod == 'mean_squared_error':
    sortedTunedModelConfigs = sorted(tunedModelConfigs, key=lambda x: x.bestScore)
else:
    sortedTunedModelConfigs = sorted(tunedModelConfigs, key=lambda x: -x.bestScore)
for item in sortedTunedModelConfigs:
    print('Entry:')
    print(item.description)
    print(item.parameters)
    print('Training score:', item.bestScore)
    print()

# Apply models

