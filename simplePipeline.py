import time
import mlutilities.utilities as mlutils
import thesisFunctions
import constants


startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

# Parameters
runScaleDatasets = True
runFeatureEngineering = True
runEnsembleModels = True
multiThreadApplyModels = True

randomSeed = constants.randomSeed

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
                                                    runEnsembleModels=runEnsembleModels,
                                                    multiThreadApplyModels=multiThreadApplyModels)

endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print()
print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))

# Visualization
dryYearScoreModelResultsDF = flowModelResult[flowModelResult['Base DataSet'].str.contains('Dry')]

mlutils.scatterPlot(flowModelResult,
                    'Mean Squared Error',
                    'R Squared',
                    'MSE by R Squared for Each Model',
                    'Output/mseByR2AllModels.png',
                    '#2d974d')
mlutils.scatterPlot(dryYearScoreModelResultsDF,
                    'Mean Squared Error',
                    'R Squared',
                    'MSE by R Squared for Each Model (Dry Year Models Only',
                    'Output/mseByR2DryModels.png',
                    '#2d974d')
mlutils.scatterPlot(flowModelResult,
                    'RMSE',
                    'R Squared',
                    'RMSE by R Squared for Each Model',
                    'Output/rmseByR2AllModels.png',
                    '#2d974d')
mlutils.scatterPlot(dryYearScoreModelResultsDF,
                    'RMSE',
                    'R Squared',
                    'RMSE by R Squared for Each Model (Dry Year Models Only',
                    'Output/rmseByR2DryModels.png',
                    '#2d974d')

mlutils.barChart(flowModelResult,
                 'Mean Squared Error',
                 'MSE for Each Model',
                 'Output/meanSquaredError.png',
                 '#2d974d')
mlutils.barChart(flowModelResult,
                 'RMSE',
                 'Root Mean Squared Error for Each Model',
                 'Output/rootMeanSquaredError.png',
                 '#2d974d')
mlutils.barChart(flowModelResult,
                 'R Squared',
                 'R Squared for Each Model',
                 'Output/rSquared.png',
                 '#2d974d')
