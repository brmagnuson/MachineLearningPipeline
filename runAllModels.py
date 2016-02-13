import time
import threading
import thesisFunctions
import constants

baseDirectoryPath = 'AllMonthsDryHalf/'
wetOrDry = 'dry'
# baseDirectoryPath = 'AllMonthsWetHalf/'
# wetOrDry = 'wet'
myFeaturesIndex = 6
myLabelIndex = 5
kFolds = 5
randomSeed = constants.randomSeed
multiThreading = False

# regions = ['CoastMnt', 'IntMnt', 'Xeric']
regions = ['IntMnt']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

selectedFeatureList = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
                       't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12',
                       'p2sum', 'p3sum', 'p6sum', 'PERMAVE', 'RFACT', 'DRAIN_SQKM', 'ELEV_MEAN_M_BASIN_30M',
                       'WD_BASIN']

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

threads = []
for region in regions:
    for month in months:

        if multiThreading:

            # Build threads
            t = threading.Thread(target=thesisFunctions.runKFoldPipeline, args=(baseDirectoryPath,
                                                                                myFeaturesIndex,
                                                                                myLabelIndex,
                                                                                selectedFeatureList,
                                                                                kFolds,
                                                                                wetOrDry,
                                                                                month,
                                                                                region,
                                                                                randomSeed))
            threads.append(t)

        else:

            # Run each pipeline in sequence
            print('Running pipeline for %s, %s' % (region, month.capitalize()))
            thesisFunctions.runKFoldPipeline(baseDirectoryPath,
                                             myFeaturesIndex,
                                             myLabelIndex,
                                             selectedFeatureList,
                                             kFolds,
                                             wetOrDry,
                                             month,
                                             region,
                                             randomSeed)
            print()

if multiThreading:

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to finish before continuing
    for t in threads:
        t.join()

endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))
