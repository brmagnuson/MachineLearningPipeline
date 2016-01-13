import os
import fnmatch
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions


basePath = 'Data/'
myFeaturesIndex = 6
myLabelIndex = 5

trainingDataSet = mltypes.DataSet('All years training',
                                  basePath + 'jul_IntMnt_ref_train.csv',
                                  featuresIndex=myFeaturesIndex,
                                  labelIndex=myLabelIndex)
trainFeatures = trainingDataSet.featuresDataFrame
trainLabel = trainingDataSet.labelSeries

# expertSelectedConfig = mltypes.FeatureEngineeringConfiguration('Expert Selection',
#                                                                'selection',
#                                                                mltypes.ExtractSpecificFeatures,
#                                                                {'featureList': ['wb0', 'wb1']})
# featureEngineeredTrainingDataSet, transformer = mldata.engineerFeaturesForDataSet(trainingDataSet,
#                                                                                expertSelectedConfig)

ridgeModellingMethod = mltypes.ModellingMethod('Ridge Reg',
                                               sklearn.linear_model.Ridge)
applyModelConfig = mltypes.ApplyModelConfiguration('Apply Ridge Regression',
                                                   ridgeModellingMethod,
                                                   {'alpha': 0.1, 'normalize': True},
                                                   trainingDataSet)

applyModelResult = mlmodel.applyModel(applyModelConfig)
r2Method = mltypes.ModelScoreMethod('R Squared', sklearn.metrics.r2_score)
scoreModelResult = mlmodel.scoreModel(applyModelResult, [r2Method])
print(scoreModelResult)
print()

# tuneScoreMethod = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
tuneScoreMethod = 'r2'
ridgeParameters = [{'alpha': [0.1, 0.5, 1.0],
                    'normalize': [True, False]}]
ridgeConfig = mltypes.TuneModelConfiguration('Ridge Regression',
                                             ridgeModellingMethod,
                                             ridgeParameters,
                                             tuneScoreMethod)
tuneModelResult = mlmodel.tuneModel(trainingDataSet, ridgeConfig)
print(tuneModelResult.bestScore)
for score in tuneModelResult.gridScores:
    print(score)
