import pandas
import thesisFunctions

def prepareFlowData(flowFilePath):

    # Get Sacramento basin data and keep just the identifying info and the gage measurements (qmeas)
    flowData = pandas.read_csv('SacramentoModel/Sacramento_Basin.csv')
    flowData = flowData.loc[:, ['STAID', 'Month', 'Year', 'qmeas']]

    # Delete 2011 and 2012 data since we don't have predictive data for those years
    flowData = flowData[ ~ flowData.Year.isin([2011, 2012])]

    # Write out to file to save
    flowData.to_csv(flowFilePath, index=False)


# Parameters
runPrepFlowData = False
month = 1
year = 1977
flowFile = 'ScalingWaterYear/sacramentoFlowData.csv'

if runPrepFlowData:
    prepareFlowData(flowFile)

# Read in data from file
flowData = pandas.read_csv(flowFile)

# Get lists of months and years to iterate over
# months = flowData.Month.unique().tolist()
# years = flowData.Year.unique().tolist()
# years.sort()

# Calculate gage ratio for given year and month


# Pull in present data for those gages
print('done')
