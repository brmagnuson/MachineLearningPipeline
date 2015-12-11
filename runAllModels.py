import time
import threading
import thesisFunctions

allMonthsPath = 'AllMonths/'
randomSeed = 47392
multiThreading = True

# regions = ['CoastMnt', 'IntMnt', 'Xeric']
regions = ['IntMnt']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

startTime = time.strftime('%a, %d %b %Y %X')

for region in regions:
    for month in months:

        if multiThreading:
            t = threading.Thread(target=thesisFunctions.runAllModels, args=(month, region, randomSeed))
            t.start()
        else:
            print('Running pipeline for %s, %s' % (region, month.capitalize()))
            thesisFunctions.runAllModels(month, region, randomSeed)
            print()

endTime = time.strftime('%a, %d %b %Y %X')

print('Start time:', startTime)
print('End time:', endTime)
