import time
import thesisFunctions
import constants

basePath = 'SacramentoModel/'

# Parameters
scaleLabel = False
trainFeaturesIndex = 6
trainLabelIndex = 5
modelIndex = 0
modelApproach = 'sacramento'
randomSeed = constants.randomSeed
selectedFeaturesList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                       'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'ELEV_MEAN_M_BASIN_30M',
                       'WD_BASIN', 'IntMnt', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                       'sep', 'oct', 'nov']

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Predict for the Sacramento river
for month in months:
    if month == 'jan':
        printLog = True
    else:
        printLog = False
    thesisFunctions.processSacPredictions(basePath, trainFeaturesIndex, trainLabelIndex, modelIndex,
                                          selectedFeaturesList, randomSeed, modelApproach,
                                          month=month, printLog=printLog, scaleLabel=scaleLabel)

print()


endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))
