import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

sacFolder = 'SacramentoModel/'
baseFolders = [sacFolder]

outputFolder = 'Output/'
outputFileName = 'SacramentoData.csv'

# Aggregate predictions into one file
aggregateFile = thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputFileName, months)

# Output file for specific water year for use in DWRAT
waterYear = 1977
thesisFunctions.formatWaterYearPredictions(waterYear, aggregateFile)
