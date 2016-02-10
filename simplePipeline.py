import os
import fnmatch
import pickle
import copy
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions

# Parameters
runScaleDatasets = False
runFeatureEngineering = True
runEnsembleModels = False

randomSeed = 47392

month = 'jul'
region = 'IntMnt'
basePath = 'Data/'
universalTestSetFileName = month + '_' + region + '_test.csv'
universalTestSetDescription = month.capitalize() + ' ' + region + ' Test'
scoreOutputFilePath = 'Output/testResults.csv'
myFeaturesIndex = 6
myLabelIndex = 5

selectedFeatureList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                       'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                       'WD_BASIN']

flowModelResult = thesisFunctions.flowModelPipeline(universalTestSetFileName,
                                                    universalTestSetDescription,
                                                    basePath,
                                                    scoreOutputFilePath,
                                                    myFeaturesIndex,
                                                    myLabelIndex,
                                                    selectedFeatureList=selectedFeatureList,
                                                    randomSeed=randomSeed,
                                                    runScaleDatasets=runScaleDatasets,
                                                    runFeatureEngineering=runFeatureEngineering,
                                                    runEnsembleModels=runEnsembleModels)


# # Visualization
# scoreModelResultsDF['RMSE'] = scoreModelResultsDF['Mean Squared Error'].map(lambda x: x ** (1 / 2))
# dryYearScoreModelResultsDF = scoreModelResultsDF[scoreModelResultsDF['Base DataSet'].str.contains('Dry')]

# if runVisualization:
#     mlutils.scatterPlot(scoreModelResultsDF,
#                         'Mean Squared Error',
#                         'R Squared',
#                         'MSE by R Squared for Each Model',
#                         'Output/mseByR2AllModels.png')
#     mlutils.scatterPlot(dryYearScoreModelResultsDF,
#                         'Mean Squared Error',
#                         'R Squared',
#                         'MSE by R Squared for Each Model (Dry Year Models Only',
#                         'Output/mseByR2DryModels.png')
#     mlutils.scatterPlot(scoreModelResultsDF,
#                         'RMSE',
#                         'R Squared',
#                         'RMSE by R Squared for Each Model',
#                         'Output/rmseByR2AllModels.png')
#     mlutils.scatterPlot(dryYearScoreModelResultsDF,
#                         'RMSE',
#                         'R Squared',
#                         'RMSE by R Squared for Each Model (Dry Year Models Only',
#                         'Output/rmseByR2DryModels.png')
#
#     mlutils.barChart(scoreModelResultsDF,
#                      'Mean Squared Error',
#                      'MSE for Each Model',
#                      'Output/meanSquaredError.png')
#     mlutils.barChart(scoreModelResultsDF,
#                      'RMSE',
#                      'Root Mean Squared Error for Each Model',
#                      'Output/rootMeanSquaredError.png')
#     mlutils.barChart(scoreModelResultsDF,
#                      'R Squared',
#                      'R Squared for Each Model',
#                      'Output/rSquared.png')
