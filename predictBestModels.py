import mlutilities.types as mltypes
import thesisFunctions

basePath = 'AllMonthsDryHalf/'
month = 'apr'
region = 'IntMnt'
randomSeed = 47392
myFeaturesIndex = 6
myLabelIndex = 5
modelIndex = 0
selectedFeaturesList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                        't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                        'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                        'WD_BASIN']

predictionPath = basePath + region + '/' + month + '/Prediction/sacramentoData.csv'
predictionDataSet = mltypes.DataSet(month.title() + ' Prediction Data',
                                    predictionPath,
                                    featuresIndex=3,
                                    labelIndex=None)

result = thesisFunctions.findModelAndPredict(predictionDataSet, basePath, month, region, randomSeed, myFeaturesIndex,
                                             myLabelIndex, selectedFeaturesList, modelIndex)