import pandas


# Read in original dataset with all years
basePath = 'Data/'
waterYears = pandas.read_csv(basePath + 'NOAAWaterYearsDriestToWettest.csv')
print(waterYears)
