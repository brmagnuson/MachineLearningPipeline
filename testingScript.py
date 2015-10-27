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
print(allYearsDataSet)

# Train/test split
testProportion = 0.25
trainDataSet, testDataSet = mlutilities.dataTransformation.splitDataSet(allYearsDataSet, testProportion)

parameters = [{'alpha': [0.1, 0.5, 1.0], 'normalize': [True, False]}]
ridgeConfig = mlutilities.types.ModelCreationConfiguration('Ridge regression scored by mean_squared_error',
                                                           sklearn.linear_model.Ridge,
                                                           parameterGrid=parameters,
                                                           scoreMethod='mean_squared_error')
print(ridgeConfig)

print()
tunedRidgeConfig = mlutilities.modeling.tuneModel(trainDataSet, ridgeConfig)
print(tunedRidgeConfig)
