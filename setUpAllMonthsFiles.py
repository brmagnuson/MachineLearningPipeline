import os
import shutil
import pandas
import thesisFunctions

# allMonthsPath = 'AllMonthsDryHalf/'
# wetOrDry = 'dry'
allMonthsPath = 'AllMonthsWetHalf/'
wetOrDry = 'wet'
rfDataPath = '../RF_model_data/data/model_training/'

proportionOfInterest = 0.5

# The Sacramento region only belongs to the IntMnt and Xeric ecoregions.
regions = ['IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Create folder for each month & region in allMonths
for region in regions:
    for month in months:
        newFolderPath = allMonthsPath + region + '/' + month
        if not os.path.exists(newFolderPath):
            os.makedirs(newFolderPath)

# Populate folders with necessary files and folders
for region in regions:
    for month in months:

        print('Processing', region, month.title())

        destinationFolderPath = allMonthsPath + region + '/' + month + '/'

        # Copy files from matching directory used to create the original random forest model, writing "ObsID" as
        # header for first column and deleting the 'X.1' mystery variable when it shows up
        sourceFilePath = rfDataPath + region.lower() + '/' + month + '_' + region + '_ref.csv'
        destinationFilePath = destinationFolderPath + month + '_' + region + '_all.csv'

        sourceFile = pandas.read_csv(sourceFilePath)

        sourceFile.rename(columns={'Unnamed: 0': 'ObsID'}, inplace=True)
        badVariable = 'X.1'
        if badVariable in sourceFile.columns.values:
            sourceFile.drop(badVariable, axis=1, inplace=True)

        sourceFile.to_csv(destinationFilePath, index=False)

        # Create subfolders used in model pipeline
        subFolders = ['CurrentFoldData', 'Output', 'Prediction']
        for subFolder in subFolders:
            newSubFolderPath = destinationFolderPath + '/' + subFolder
            if not os.path.exists(newSubFolderPath):
                os.makedirs(newSubFolderPath)

        # Also add a copy of NOAA water years
        waterYearsSourceFilePath = allMonthsPath + 'NOAAWaterYearsDriestToWettest.csv'
        waterYearsDestinationFilePath = destinationFolderPath + 'NOAAWaterYearsDriestToWettest.csv'
        shutil.copyfile(waterYearsSourceFilePath, waterYearsDestinationFilePath)

        # Add in Sacramento data for prediction
        sacData = thesisFunctions.prepSacramentoData(month,
                                                     region,
                                                     wetOrDry,
                                                     waterYearsDestinationFilePath,
                                                     proportionOfInterest)
        predictionFilePath = allMonthsPath + region + '/' + month + '/Prediction/sacramentoData.csv'
        sacData.to_csv(predictionFilePath, index=False)
