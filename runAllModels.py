import time
import threading
import thesisFunctions

allMonthsPath = 'AllMonths/'
randomSeed = 47392
multiThreading = True

# regions = ['CoastMnt', 'IntMnt', 'Xeric']
regions = ['IntMnt']
# months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
months = ['jan', 'feb']

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')

threads = []
for region in regions:
    for month in months:

        if multiThreading:

            # Build threads
            t = threading.Thread(target=thesisFunctions.runAllModels, args=(month, region, randomSeed))
            threads.append(t)

        else:

            # Run each pipeline
            print('Running pipeline for %s, %s' % (region, month.capitalize()))
            thesisFunctions.runAllModels(month, region, randomSeed)
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
