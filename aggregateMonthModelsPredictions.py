import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
regions = ['IntMnt']

dryFolder = 'AllMonthsDryHalf/'
wetFolder = 'AllMonthsWetHalf/'
baseFolders = [dryFolder, wetFolder]

outputFolder = 'Output/'
outputPrefix = 'AllMonthsData_'

# Aggregate each half-region-month's predictions into one file.
for region in regions:
    thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputPrefix, months, region)

# Output IntMnt file for specific water year for use in DWRAT
aggregateFile = outputFolder + outputPrefix + regions[0] + '.csv'
waterYear = 1977
thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)
