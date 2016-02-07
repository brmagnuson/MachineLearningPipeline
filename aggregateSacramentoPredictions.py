import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

sacFolder = 'SacramentoModel/'
baseFolders = [sacFolder]

outputFolder = 'Output/'
outputPrefix = 'SacramentoData'

# Aggregate predictions into one file
aggregateFile = thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputPrefix, months)

# Output file for specific water year for use in DWRAT
# aggregateFile = outputFolder + outputPrefix + region + '.csv'
waterYear = 1977
thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)
