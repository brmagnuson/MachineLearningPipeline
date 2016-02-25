import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
regions = ['IntMnt', 'Xeric']

dryFolder = 'AllMonthsDryHalf/'
wetFolder = 'AllMonthsWetHalf/'
baseFolders = [dryFolder, wetFolder]

outputFolder = 'Output/'
outputFileName = 'AllMonthsData.csv'

# Aggregate each half-region-month's predictions into one file.
thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputFileName, months, regions)

# Output IntMnt file for specific water year for use in DWRAT
aggregateFile = outputFolder + outputFileName
waterYear = 1977
thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)
