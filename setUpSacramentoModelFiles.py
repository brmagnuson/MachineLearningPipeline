import pandas
import thesisFunctions

# Build dataset of all months & regions

sacBasePath = 'SacramentoModel/'
rfDataPath = '../RF_model_data/data/model_training/'

regions = ['IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
# regions = ['CoastMnt']
# months = ['jan', 'feb']

# Build DataFrames in which to accumulate training and prediction data
allTrainingDF = pandas.DataFrame()
predictionDF = pandas.DataFrame()

for region in regions:

    print()
    print('Processing region:', region)

    for month in months:

        print(month, end=' ', flush=True)
        
        # Get training data from matching directory used to create the original random forest model, writing "ObsID" as
        # header for first column and deleting the 'X.1' mystery variable when it shows up
        sourceFile = rfDataPath + region.lower() + '/' + month + '_' + region + '_ref.csv'
        monthTrainingDF = pandas.read_csv(sourceFile)
        monthTrainingDF.rename(columns={'Unnamed: 0': 'ObsID'}, inplace=True)
        badVariable = 'X.1'
        if badVariable in monthTrainingDF.columns.values:
            monthTrainingDF.drop(badVariable, axis=1, inplace=True)

        # Drop "T" from beginning of STAID column (someone added it as a string casting hack before I got the data)
        monthTrainingDF['STAID'] = monthTrainingDF['STAID'].map(lambda x: x[1:])

        # Get prediction data
        monthPredictionDF = thesisFunctions.prepSacramentoData(month, region, sacBasePath)

        # Add columns that ID current region and month (1 if True, 0 if False) to each DataFrame
        for regionColumn in regions:

            if regionColumn == region:
                monthTrainingDF[regionColumn] = 1
                monthPredictionDF[regionColumn] = 1
            else:
                monthTrainingDF[regionColumn] = 0
                monthPredictionDF[regionColumn] = 0

        for monthColumn in months:

            if monthColumn == month:
                monthTrainingDF[monthColumn] = 1
                monthPredictionDF[monthColumn] = 1
            else:
                monthTrainingDF[monthColumn] = 0
                monthPredictionDF[monthColumn] = 0

        # Append to accumulator DataFrames
        allTrainingDF = allTrainingDF.append(monthTrainingDF, ignore_index=True)
        predictionDF = predictionDF.append(monthPredictionDF, ignore_index=True)

# Subset all training data to just those observations in the Sacramento basin
sacGagesFile = sacBasePath + 'Sac_gages_list.csv'
sacGagesDF = pandas.read_csv(sacGagesFile)
sacRefGages = sacGagesDF.STAID[sacGagesDF.CLASS == 'Ref'].map(lambda x: str(x)).values.tolist()
sacTrainingDF = allTrainingDF[allTrainingDF.STAID.isin(sacRefGages)]

# Write Sacramento training data out to file
sacTrainingDataFile = sacBasePath + 'Sacramento_Basin.csv'
sacTrainingDF.to_csv(sacTrainingDataFile, index=False)

# Write Sacramento prediction data out to file
predictionFilePath = sacBasePath + '/Prediction/sacramentoData.csv'
predictionDF.to_csv(predictionFilePath, index=False)
