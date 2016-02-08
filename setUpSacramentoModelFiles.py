import pandas
import thesisFunctions

# Build dataset of all months & regions

sacBasePath = 'SacramentoModel/'
rfDataPath = '../RF_model_data/data/model_training/'

# Just use IntMnt and Xeric regions, because nothing from the CoastMnt dataset is in the Sacramento watershed
regions = ['IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
# months = ['jan', 'feb']

# Get gages for the Sacramento basin
sacGagesFile = sacBasePath + 'Sac_gages_list.csv'
sacGagesDF = pandas.read_csv(sacGagesFile)
sacRefGages = sacGagesDF.STAID[sacGagesDF.CLASS == 'Ref'].map(lambda x: str(x)).values.tolist()

# Build DataFrame in which to accumulate training data
allTrainingDF = pandas.DataFrame()

for month in months:

    print()
    print('Processing month:', month.title())
    
    # Build DataFrame in which to accumulate prediction data
    monthPredictionDF = pandas.DataFrame()

    for region in regions:

        print(region, end=' ', flush=True)
        
        # Get training data from matching directory used to create the original random forest model, writing "ObsID" as
        # header for first column and deleting the 'X.1' mystery variable when it shows up
        sourceFile = rfDataPath + region.lower() + '/' + month + '_' + region + '_ref.csv'
        regionTrainingDF = pandas.read_csv(sourceFile)
        regionTrainingDF.rename(columns={'Unnamed: 0': 'ObsID'}, inplace=True)
        badVariable = 'X.1'
        if badVariable in regionTrainingDF.columns.values:
            regionTrainingDF.drop(badVariable, axis=1, inplace=True)

        # Drop "T" from beginning of STAID column (someone added it as a string casting hack before I got the data)
        regionTrainingDF['STAID'] = regionTrainingDF['STAID'].map(lambda x: x[1:])

        # Subset all training data to just those observations in the Sacramento basin
        regionTrainingDF = regionTrainingDF[regionTrainingDF.STAID.isin(sacRefGages)]

        # Get prediction data
        regionPredictionDF = thesisFunctions.prepSacramentoData(month, region)

        # Add columns that ID current region and month (1 if True, 0 if False) to each DataFrame, leaving off last one
        # in each list to prevent model being over-specified
        for regionColumn in regions[:-1]:

            if regionColumn == region:
                regionTrainingDF[regionColumn] = 1
                regionPredictionDF[regionColumn] = 1
            else:
                regionTrainingDF[regionColumn] = 0
                regionPredictionDF[regionColumn] = 0

        for monthColumn in months[:-1]:

            if monthColumn == month:
                regionTrainingDF[monthColumn] = 1
                regionPredictionDF[monthColumn] = 1
            else:
                regionTrainingDF[monthColumn] = 0
                regionPredictionDF[monthColumn] = 0

        # Append data to accumulator DataFrames
        allTrainingDF = allTrainingDF.append(regionTrainingDF, ignore_index=True)
        monthPredictionDF = monthPredictionDF.append(regionPredictionDF, ignore_index=True)

    # Write month's prediction data out to file (too big to deal with all months in one csv)
    predictionFilePath = sacBasePath + '/Prediction/sacramentoData_' + month + '.csv'
    monthPredictionDF.to_csv(predictionFilePath, index=False)

# Write all Sacramento training data out to file
sacTrainingDataFilePath = sacBasePath + 'Sacramento_Basin.csv'
allTrainingDF.to_csv(sacTrainingDataFilePath, index=False)
