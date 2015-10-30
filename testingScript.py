import sklearn.linear_model
import mlutilities.types
import mlutilities.dataTransformation
import mlutilities.modeling


# Get list of data sets.
basePath = 'Data/'
myfeaturesIndex = 5
myLabelIndex = 4
allYearsDataSet = mlutilities.types.DataSet('All Years',
                                            basePath + 'jul_IntMnt_ref.csv',
                                            featuresIndex=myfeaturesIndex,
                                            labelIndex=myLabelIndex)

# Train/test split
testProportion = 0.25
splitDataSet = mlutilities.dataTransformation.splitDataSet(allYearsDataSet, testProportion)
trainDataSet = splitDataSet.trainDataSet
testDataSet = splitDataSet.testDataSet

# Tune model
parameters = [{'alpha': [0.1, 0.5, 1.0], 'normalize': [True, False]}]
ridgeConfig = mlutilities.types.TuneModelConfiguration('Ridge regression scored by mean_squared_error',
                                                       sklearn.linear_model.Ridge,
                                                       parameterGrid=parameters,
                                                       scoreMethod='mean_squared_error')

tuneModelResult = mlutilities.modeling.tuneModel(trainDataSet, ridgeConfig)
print(tuneModelResult)
print()

# Apply model
applyRidgeConfig = mlutilities.types.ApplyModelConfiguration(tuneModelResult.description.replace('Training Set', 'Testing Set'),
                                                             tuneModelResult.modelMethod,
                                                             tuneModelResult.parameters,
                                                             trainDataSet,
                                                             testDataSet)
print(applyRidgeConfig)
print()

applyRidgeResult = mlutilities.modeling.applyModel(applyRidgeConfig)
print(applyRidgeResult)
print()
