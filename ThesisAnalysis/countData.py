import os
import pandas

regions = ['IntMnt', 'Xeric']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

allGages = pandas.DataFrame()

for region in regions:

    print(region)

    # Build full data frame of training gages used in the region
    regionGages = pandas.DataFrame()
    for month in months:

        # Read in data file and fix STAID (remove T from beginning that somebody added along the way)
        fileName = os.path.join('..', 'AllMonthsDryHalf', region, month, month + '_' + region + '_all.csv')
        monthGages = pandas.read_csv(fileName)
        monthGages['STAID'] = monthGages['STAID'].map(lambda x: str(x)[1:])
        regionGages = regionGages.append(monthGages, ignore_index=True)

    # Count number of gages
    regionGageList = regionGages['STAID'].unique()
    count = len(regionGageList)

    print('Number of gages:', count)
    print('Number of observations:', regionGages.shape[0])
    print('Number of years:', len(regionGages['Year'].unique()))
    print()

    # Create list of gages with their region
    regionGageDF = pandas.DataFrame(regionGageList, columns=['STAID'])
    regionGageDF['Region'] = region
    allGages = allGages.append(regionGageDF, ignore_index=True)

# Add that one CoastMnt gage in the Sacramento basin to the list
specialGage = 11371000
allGages.loc[len(allGages)] = [specialGage, 'CoastMnt']

# Output to CSV
allGages.to_csv('refGageRegions.csv', index=False)

