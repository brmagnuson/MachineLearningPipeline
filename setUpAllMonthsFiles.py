import os
import shutil
import csv

allMonthsPath = 'AllMonths/'
rfDataPath = '../RF_model_data/data/model_training/'

regions = ['CoastMnt', 'IntMnt', 'Xeric']
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

        destinationFolderPath = allMonthsPath + region + '/' + month + '/'

        # Copy files from matching directory used to create the original random forest model, writing "ObsID" as
        # header for first column
        sourceFilePath = rfDataPath + region.lower() + '/' + month + '_' + region + '_ref.csv'
        destinationFilePath = destinationFolderPath + month + '_' + region + '_all.csv'

        with open(sourceFilePath, 'r') as sourceFile:
            reader = csv.reader(sourceFile)
            lines = [line for line in reader]

        lines[0][0] = 'ObsID'

        with open(destinationFilePath, 'w') as destinationFile:
            writer = csv.writer(destinationFile)
            writer.writerows(lines)

        # Create subfolders used in model pipeline
        subFolders = ['CurrentFoldData', 'Output', 'Pickles']
        for subFolder in subFolders:
            newSubFolderPath = destinationFolderPath + '/' + subFolder
            if not os.path.exists(newSubFolderPath):
                os.makedirs(newSubFolderPath)

        # Also add a copy of NOAA water years
        sourceFilePath = allMonthsPath + 'NOAAWaterYearsDriestToWettest.csv'
        destinationFilePath = destinationFolderPath + 'NOAAWaterYearsDriestToWettest.csv'
        shutil.copyfile(sourceFilePath, destinationFilePath)



