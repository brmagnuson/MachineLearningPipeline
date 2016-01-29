import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
regions = ['IntMnt']

dryFolder = 'AllMonthsDryHalf/'
wetFolder = 'AllMonthsWetHalf/'
baseFolders = [dryFolder, wetFolder]

outputFolder = 'Output/'
outputPrefix = 'SacramentoData_'

# Aggregate each half-region-month's predictions into one file.
thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputPrefix, months, regions)

aggregateFile = outputFolder + outputPrefix + regions[0] + '.csv'
waterYear = 1977
thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)
