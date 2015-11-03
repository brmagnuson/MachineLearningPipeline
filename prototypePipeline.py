import pickle
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import mlutilities.types
import mlutilities.dataTransformation
import mlutilities.modeling

# True to run all steps, False to just use previously pickled output
pickling = False

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

if pickling:
    regularDataSets = [allYearsDataSet, dryYearsDataSet]
    pickle.dump(regularDataSets, open(picklePath + 'regularDataSets.p', 'wb'))
regularDataSets = pickle.load(open(picklePath + 'regularDataSets.p', 'rb'))

# Get scaled data sets
print('Scaling data sets.')

if pickling:
    scaledDataSets = mlutilities.dataTransformation.scaleDataSets(regularDataSets)
    pickle.dump(scaledDataSets, open(picklePath + 'scaledDataSets.p', 'wb'))
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

if pickling:
    featureEngineeredDatasets = mlutilities.dataTransformation.engineerFeaturesForDataSets(allDataSets, featureEngineeringConfigurations)
    allDataSets += featureEngineeredDatasets
    pickle.dump(allDataSets, open(picklePath + 'allDataSets.p', 'wb'))
allDataSets = pickle.load(open(picklePath + 'allDataSets.p', 'rb'))

# Train/test split
print('Splitting into testing & training data.')
testProportion = 0.25

if pickling:
    splitDataSets = mlutilities.dataTransformation.splitDataSets(allDataSets, testProportion)
    pickle.dump(splitDataSets, open(picklePath + 'splitDataSets.p', 'wb'))
splitDataSets = pickle.load(open(picklePath + 'splitDataSets.p', 'rb'))

trainDataSets = [splitDataSet.trainDataSet for splitDataSet in splitDataSets]

# Tune models
print('Tuning models.')

# scoreMethod = 'r2'
scoreMethod = 'mean_squared_error'
ridgeParameters = [{'alpha': [0.1, 0.5, 1.0],
                    'normalize': [True, False]}]
ridgeConfig = mlutilities.types.TuneModelConfiguration('Ridge regression scored by mean_squared_error',
                                                           sklearn.linear_model.Ridge,
                                                           ridgeParameters,
                                                           scoreMethod)
randomForestParameters = [{'n_estimators': [10, 20],
                           'max_features': [10, 'sqrt']}]
randomForestConfig = mlutilities.types.TuneModelConfiguration('Random Forest scored by mean_squared_error',
                                                                  sklearn.ensemble.RandomForestRegressor,
                                                                  randomForestParameters,
                                                                  scoreMethod)
tuneModelConfigs = [ridgeConfig, randomForestConfig]

if pickling:
    tuneModelResults = mlutilities.modeling.tuneModels(trainDataSets, tuneModelConfigs)
    pickle.dump(tuneModelResults, open(picklePath + 'tuneModelResults.p', 'wb'))
tuneModelResults = pickle.load(open(picklePath + 'tuneModelResults.p', 'rb'))

# Model tuning result reporting
if scoreMethod == 'mean_squared_error':
    sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: x.bestScore)
else:
    sortedTuneModelResults = sorted(tuneModelResults, key=lambda x: -x.bestScore)
for item in sortedTuneModelResults:
    print(item)
    print()

# Create ApplyModelConfigurations
if pickling:

    applyModelConfigs = []
    for tuneModelResult in tuneModelResults:

        trainDataSet = tuneModelResult.dataSet
        testDataSet = None
        for splitDataSet in splitDataSets:
            if splitDataSet.trainDataSet == trainDataSet:
                testDataSet = splitDataSet.testDataSet

        # Make sure we found a match
        if testDataSet == None:
            raise Exception('No SplitDataSet found matching this training DataSet:\n' + trainDataSet)

        applyModelConfig = mlutilities.types.ApplyModelConfiguration('Apply ' + tuneModelResult.description.replace('Training Set', 'Testing Set'),
                                                                     tuneModelResult.modelMethod,
                                                                     tuneModelResult.parameters,
                                                                     trainDataSet,
                                                                     testDataSet)
        applyModelConfigs.append(applyModelConfig)

    pickle.dump(applyModelConfigs, open(picklePath + 'applyModelConfigs.p', 'wb'))
applyModelConfigs = pickle.load(open(picklePath + 'applyModelConfigs.p', 'rb'))

# Apply models
print('Applying models to test data.')
applyModelResults = mlutilities.modeling.applyModels(applyModelConfigs)

# Score models
# testScoreMethod = sklearn.metrics.r2_score
testScoreMethod = sklearn.metrics.mean_squared_error
scoreModelResults = mlutilities.modeling.scoreModels(applyModelResults, testScoreMethod)

# Model testing result reporting
if testScoreMethod == sklearn.metrics.mean_squared_error:
    sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: x.score)
else:
    sortedScoreModelResults = sorted(scoreModelResults, key=lambda x: -x.score)
for item in sortedScoreModelResults:
    print(item)
    print()
