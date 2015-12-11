import time
import thesisFunctions

allMonthsPath = 'AllMonths/'

regions = ['CoastMnt', 'IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

startTime = time.strftime('%a, %d %b %Y %X')

for region in regions:
    for month in months:

        print('Running pipeline for %s, %s' % (region, month.capitalize()))
        print('Current time:', time.strftime('%a, %d %b %Y %X'))
        thesisFunctions.runAllModels(month, region, randomSeed=47392)
        print()

endTime = time.strftime('%a, %d %b %Y %X')

print('Start time:', startTime)
print('End time:', endTime)
