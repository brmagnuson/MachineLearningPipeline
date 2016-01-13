import time
import threading
import thesisFunctions

baseDirectoryPath = 'AllMonths/'
randomSeed = 47392
multiThreading = True

# regions = ['CoastMnt', 'IntMnt', 'Xeric']
regions = ['IntMnt']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

threads = []
for region in regions:
    for month in months:

        if multiThreading:

            # Build threads
            t = threading.Thread(target=thesisFunctions.runKFoldPipeline, args=(month,
                                                                                region,
                                                                                baseDirectoryPath,
                                                                                randomSeed))
            threads.append(t)

        else:

            # Run each pipeline in sequence
            print('Running pipeline for %s, %s' % (region, month.capitalize()))
            thesisFunctions.runKFoldPipeline(month, region, baseDirectoryPath, randomSeed)
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
