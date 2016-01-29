import thesisFunctions

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
regions = ['IntMnt']

dryFolder = 'AllMonthsDryHalf/'
wetFolder = 'AllMonthsWetHalf/'
baseFolders = [dryFolder, wetFolder]

outputFolder = 'Output/'
outputPrefix = 'SacramentoData_'

thesisFunctions.aggregateSacPredictions(baseFolders, outputFolder, outputPrefix, months, regions)
