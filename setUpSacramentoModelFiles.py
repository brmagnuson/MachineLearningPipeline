import pandas
import thesisFunctions

# Build dataset of all months & regions

sacBasePath = 'SacramentoModel/'
rfDataPath = '../RF_model_data/data/model_training/'

# Just use IntMnt and Xeric regions, because nothing from the CoastMnt dataset is in the Sacramento watershed
regions = ['IntMnt', 'Xeric', 'CoastMnt']
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

        # Hack: If we're in the CoastMnt region loop, reassign region as 'IntMnt' because the only Sacramento reference
        # gage in that region's source file, 11371000, is actually in the West Mnt region in the Gages II database. None
        # of the prediction data is in the CoastMnt region so it won't affect anything in that dataframe.
        if region == 'CoastMnt':
            regionForDF = 'IntMnt'
        else:
            regionForDF = region

        # Add columns that ID current region and month (1 if True, 0 if False) to each DataFrame, leaving off last one
        # in each list to prevent model being over-specified (actually the last two for the regions, because of the
        # 11371000 misclassification as CoastMnt explained above
        for regionColumn in regions[:-2]:

            if regionColumn == regionForDF:
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

        # CoastMnt's DataFrame being empty means it has a different column order (HUC12 in the middle rather than at the
        # end), so don't append it, because the different column orders mess up the output.
        if region != 'CoastMnt':
            monthPredictionDF = monthPredictionDF.append(regionPredictionDF, ignore_index=True)

    # Write month's prediction data out to file (too big to deal with all months in one csv)
    predictionFilePath = sacBasePath + '/Prediction/sacramentoData_' + month + '.csv'
    monthPredictionDF.to_csv(predictionFilePath, index=False)

# Write all Sacramento training data out to file
sacTrainingDataFilePath = sacBasePath + 'Sacramento_Basin.csv'
allTrainingDF.to_csv(sacTrainingDataFilePath, index=False)
