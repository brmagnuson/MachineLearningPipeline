import time
import threading
import thesisFunctions
import constants

# basePath = 'AllMonthsDryHalf/'
basePath = 'AllMonthsWetHalf/'
# modelApproach = 'dry'
modelApproach = 'wet'

# Parameters
multiThreading = True
printLog = False
singleModel = True
scaleLabel = True
randomSeed = constants.randomSeed
trainFeaturesIndex = 6
trainLabelIndex = 5
modelIndex = 0
selectedFeaturesList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                        't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                        'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'ELEV_MEAN_M_BASIN_30M',
                        'WD_BASIN']

regions = ['IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

threads = []
for region in regions:

    for month in months:

        if multiThreading:

            # Build threads
            t = threading.Thread(target=thesisFunctions.processSacPredictions,
                                 args=(basePath,
                                       trainFeaturesIndex,
                                       trainLabelIndex,
                                       modelIndex,
                                       selectedFeaturesList,
                                       randomSeed,
                                       modelApproach,
                                       region,
                                       month,
                                       printLog,
                                       singleModel,
                                       scaleLabel))
            threads.append(t)

        else:

            # Run each prediction process in sequence
            thesisFunctions.processSacPredictions(basePath, trainFeaturesIndex, trainLabelIndex, modelIndex,
                                                  selectedFeaturesList, randomSeed, modelApproach, region, month,
                                                  printLog, singleModel, scaleLabel)

if multiThreading:

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to finish before continuing
    for t in threads:
        t.join()

# Pull all results together

endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))
