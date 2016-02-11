import time
import thesisFunctions
import constants

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

baseDirectoryPath = 'SacramentoModel/'
myFeaturesIndex = 6
myLabelIndex = 5
kFolds = 5
modelApproach = 'sacramento'
randomSeed = constants.randomSeed
selectedFeatureList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                       'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                       'WD_BASIN', 'IntMnt', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                       'sep', 'oct', 'nov']

# Run the flow model pipeline for five folds
thesisFunctions.runKFoldPipeline(baseDirectoryPath,
                                 myFeaturesIndex,
                                 myLabelIndex,
                                 selectedFeatureList,
                                 kFolds,
                                 modelApproach,
                                 randomSeed=randomSeed)


endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print()
print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))