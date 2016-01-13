import os
import fnmatch
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions


filePath = 'MasterData/NOAAWaterYearsDriestToWettest.csv'
month = 'jul'
proportionOfInterest = 0.5
wetOrDry = 'dry'

calendarYears = thesisFunctions.getYearsOfInterest(filePath, month, proportionOfInterest, wetOrDry)
print(calendarYears)
