import sys
import os
import time
import numpy
import pandas
import matplotlib.pyplot as plt
# import seaborn


def partitionDataFrame(dataFrame, method):

    if method == 'dry':
        dataFrame1 = dataFrame[dataFrame['Base DataSet'].str.contains('Dry')]
        dataFrame2 = dataFrame[~ dataFrame['Base DataSet'].str.contains('Dry')]
        label1 = 'Dry Years'
        label2 = 'All Years'
    elif method == 'wet':
        dataFrame1 = dataFrame[dataFrame['Base DataSet'].str.contains('Wet')]
        dataFrame2 = dataFrame[~ dataFrame['Base DataSet'].str.contains('Wet')]
        label1 = 'Wet Years'
        label2 = 'All Years'
    elif method == 'ensemble':
        dataFrame1 = dataFrame[dataFrame['Model Method'].str.contains('Ensemble')]
        dataFrame2 = dataFrame[~ dataFrame['Model Method'].str.contains('Ensemble')]
        label1 = 'Ensemble Models'
        label2 = 'Non-ensemble Models'
    elif method == 'stacking':
        dataFrame1 = dataFrame[dataFrame['Model Method'].str.contains('Stacking')]
        dataFrame2 = dataFrame[~ dataFrame['Model Method'].str.contains('Stacking')]
        label1 = 'Stacked Models'
        label2 = 'Non-stacked Models'
    else:
        sys.exit('Dataframe partitioning method not recognized.')

    return dataFrame1, dataFrame2, label1, label2


def twoClassScatterPlot(dataPath, title, partitionMethod, outputPath=None, zoom=None, legendLocation='upper left'):

    # Get data
    results = pandas.read_csv(dataPath)
    dataFrame1, dataframe2, label1, label2 = partitionDataFrame(results, partitionMethod)

    # Make scatterplot
    xColumn = 'R Squared'
    yColumn = 'Mean O/E'
    plt.axhline(1, color='#000000', linestyle='dotted', zorder=0)
    plt.scatter(dataFrame1[xColumn], dataFrame1[yColumn], s=110, facecolors='#2d974d', edgecolors='#2d974d', label=label1,
                linewidths='2', marker='.')
    plt.scatter(dataframe2[xColumn], dataframe2[yColumn], s=40, facecolors='none', edgecolors='#96d98e', label=label2,
                linewidths='2', marker='.', alpha=0.9)

    # Formatting
    if zoom is None:
        plt.xlim(plt.xlim()[0], 1.0)
    else:
        plt.xlim(zoom[0], 1.0)
        plt.ylim(zoom[1], zoom[2])
    plt.xlabel(xColumn)
    plt.ylabel(yColumn)
    plt.legend(loc=legendLocation)
    plt.title(title)

    if outputPath is None:
        plt.show()
    else:
        plt.savefig(outputPath, bbox_inches='tight')

    plt.close()
    return


def subsetToBestModelsRelatives(dataFrame):

    bestModel = dataFrame.iloc[0]
    bestModelBase = 'Dry' if 'Dry' in bestModel['Base DataSet'] else 'All'
    bestModelMethod = bestModel['Model Method']
    relatives = dataFrame[(dataFrame['Base DataSet'].str.contains(bestModelBase)) &
                          (dataFrame['Model Method'] == bestModelMethod)]
    return relatives


def makeLabels(dataFrame):

    datasetInfo = dataFrame['Base DataSet']
    dataFrame['Label'] = datasetInfo.str.split(',').str.get(0).str.slice(20).str.replace('features selected via ', '')
    for index, row in dataFrame.iterrows():
        if row['Label'] == '':
            dataFrame.set_value(index, 'Label', 'Original DataSet')
    return


def labeledDatasetScatterplot(dataPath, title, outputPath=None):

    # Get data
    dataFrame = pandas.read_csv(dataPath)
    makeLabels(dataFrame)
    relatives = subsetToBestModelsRelatives(dataFrame)

    # Make scatterplot
    xColumn = 'R Squared'
    yColumn = 'Mean O/E'
    plt.scatter(relatives[xColumn], relatives[yColumn], color='#2d974d')
    plt.axhline(1, color='#000000', linestyle='dotted', zorder=0)

    offsets = [(4, 4), (-20, -17), (-142, -7), (4, 4), (-53, 4), (-99, -5), (-102, 4), (-114, -15), (4, 4), (4, 4)]
    counter = 0
    for index, row in relatives.iterrows():
        plt.annotate(xy=(row[xColumn], row[yColumn]),
                     s=str(row['Label']),
                     xytext=offsets[counter],
                     textcoords='offset points')
        counter += 1

    # Formatting
    plt.xlabel(xColumn)
    plt.ylabel(yColumn)
    plt.title(title)
    plt.savefig(outputPath, bbox_inches='tight')
    plt.close()
    return


def specificFoldGraphs(baseDataPath, pathToFigures):

    # Specific fold parameters
    region = 'IntMnt'
    month = 'jul'
    suffix = '2'
    resultsPath = os.path.join(baseDataPath, region, month, 'Output', 'scoreModelResults_' + suffix + '.csv')
    title = month.title() + ' ' + region + ' Results for Fold #2'
    zoom=[-2, -1, 4]

    # Specific fold: dry vs all split
    partitionMethod = 'dry'
    outputPath = os.path.join(pathToFigures, '4-1 Dry R v meanOE ' + region + month.title() + suffix.title())
    twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)
    twoClassScatterPlot(resultsPath, 'Detail of ' + title, partitionMethod, outputPath + ' zoom', zoom)

    # Specific fold: ensemble vs not split
    partitionMethod = 'ensemble'
    outputPath = os.path.join(pathToFigures, '4-2 Ensemble R v meanOE ' + region + month.title() + suffix.title())
    twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)
    twoClassScatterPlot(resultsPath, 'Detail of ' + title, partitionMethod, outputPath + ' zoom', zoom)
    return


def julyCrossValidatedResults(baseDataPath, pathToFigures):

    # Parameters
    region = 'IntMnt'
    month = 'jul'
    suffix = 'average'
    resultsPath = os.path.join(baseDataPath, region, month, 'Output', 'scoreModelResults_' + suffix + '.csv')
    title = month.title() + ' ' + region + ' Cross-Validation Estimates'
    zoom = [-2, -1, 4]

    # Dry vs all split with zoom
    partitionMethod = 'dry'
    outputPath = os.path.join(pathToFigures, '4-3 Dry R v meanOE ' + region + month.title() + suffix.title())
    twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)
    twoClassScatterPlot(resultsPath, 'Detail of ' + title, partitionMethod, outputPath + ' zoom', zoom)

    # Ensemble vs not split with zoom
    partitionMethod = 'ensemble'
    outputPath = os.path.join(pathToFigures, '4-4 Ensemble R v meanOE ' + region + month.title() + suffix.title())
    twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)
    twoClassScatterPlot(resultsPath, 'Detail of ' + title, partitionMethod, outputPath + ' zoom', zoom)
    return


def appendixCGraphs(pathToAppendixFigures):

    suffix = 'average'

    # Monthly regional models
    scenarios = ['AllMonthsDryHalf', 'AllMonthsWetHalf']
    regions = ['IntMnt', 'Xeric']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for scenario in scenarios:

        baseDataPath = os.path.join('..', scenario)

        for region in regions:
            for month in months:
                # Set parameters
                resultsPath = os.path.join(baseDataPath, region, month, 'Output',
                                           'scoreModelResults_' + suffix + '.csv')
                title = month.title() + ' ' + region + ' Cross-Validation Estimates'

                # Dry vs all split
                partitionMethod = 'dry' if scenario == 'AllMonthsDryHalf' else 'wet'
                outputPath = os.path.join(pathToAppendixFigures,
                                          scenario,
                                          partitionMethod.title() + ' R v meanOE ' + region + month.title() + suffix.title())
                twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)

                time.sleep(1)

                # Ensemble vs non split
                partitionMethod = 'ensemble'
                outputPath = os.path.join(pathToAppendixFigures,
                                          scenario,
                                          'Ensemble R v meanOE ' + region + month.title() + suffix.title())
                twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)

                time.sleep(1)

    # Sacramento Model
    baseDataPath = '../SacramentoModel'
    resultsPath = os.path.join(baseDataPath, 'Output', 'scoreModelResults_' + suffix + '.csv')
    title = 'Sacramento Basin Approach Cross-Validation Estimates'
    partitionMethod = 'ensemble'
    outputPath = os.path.join(pathToAppendixFigures, 'Sacramento Ensemble R v meanOE ' + suffix.title())
    twoClassScatterPlot(resultsPath, title, partitionMethod, outputPath)
    return

def dataTransformationGraph(baseDataPath, pathToFigures):

    # Parameters
    region = 'IntMnt'
    month = 'jul'
    suffix = 'average'
    resultsPath = os.path.join(baseDataPath, region, month, 'Output', 'scoreModelResults_' + suffix + '.csv')
    title = 'Jul IntMnt Dry-Year Datasets with a Stacking Ensemble'
    outputPath = os.path.join(pathToFigures, '4-5 Data Transformation Graph')

    labeledDatasetScatterplot(resultsPath, title, outputPath)
    return


def generalBestResultsGraphs(pathToFigures):

    dataFolders = ['AllMonthsDryHalf', 'AllMonthsWetHalf']
    regions = ['IntMnt', 'Xeric']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for dataFolder in dataFolders:
        basePath = os.path.join('..', dataFolder)
        scenario = 'Dry' if dataFolder == 'AllMonthsDryHalf' else 'Wet'

        for region in regions:

            # Get data frame of best models from each month
            bestModels = pandas.DataFrame(columns=['Month',
                                                   'Base DataSet',
                                                   'Model Method',
                                                   'R Squared',
                                                   'Mean O/E',
                                                   'Standard Deviation O/E',
                                                   'Mean Squared Error',
                                                   'RMSE (cfs)'])
            for month in months:
                resultsPath = os.path.join(basePath, region, month, 'Output',
                                           'scoreModelResults_average.csv')
                results = pandas.read_csv(resultsPath)
                monthBestModel = results.iloc[0]
                bestModels = bestModels.append(monthBestModel, ignore_index=True)

            # Fix Base DataSet names for output
            datasetInfo = bestModels['Base DataSet']
            bestModels['Month'] = datasetInfo.str.slice(0, 4)
            bestModels['Base DataSet'] = datasetInfo.str.split(',').str.get(0).str.split(
                region + ' ').str.get(1).str.replace('Test', 'Years')

            # Output to file for table
            csvPath = os.path.join('BestModels', scenario + region + 'BestModels.csv')
            bestModels.to_csv(csvPath, index=False)

            # Summary statistics
            print(csvPath)
            print(numpy.mean(bestModels))
            print()

            # Graphics
            figurePath = os.path.join(pathToFigures, '4-6 Which Dataset ' + scenario + region)
            twoClassScatterPlot(csvPath, scenario + ' ' + region + ' Best Models', scenario.lower(),
                                outputPath=figurePath, legendLocation='upper right')
            figurePath = os.path.join(pathToFigures, '4-7 Stacked ' + scenario + region)
            if scenario == 'Wet' and region == 'Xeric':
                twoClassScatterPlot(csvPath, scenario + ' ' + region + ' Best Models', 'stacking',
                                    outputPath=figurePath, legendLocation='upper right')
            else:
                twoClassScatterPlot(csvPath, scenario + ' ' + region + ' Best Models', 'stacking',
                                outputPath=figurePath, legendLocation='lower right')
    return


def comparisonLineGraph(myBestModelResults, usgsModelResults, yColumn, title, outputPath=None, legendLoc='lower right'):

    # Data prep
    xColumn = 'Month'
    myXTicks = myBestModelResults[xColumn]
    numericMonths = []
    for month in myXTicks:
        numericMonth = time.strptime(month.strip(), '%b').tm_mon
        numericMonths.append(numericMonth)

    # Make plot
    plt.xticks(numericMonths, myXTicks)
    plt.plot(numericMonths, myBestModelResults[yColumn], label='New Models', color='#2d974d')
    plt.plot(numericMonths, usgsModelResults[yColumn], label='USGS Models', color='#96d98e')
    if yColumn == 'Mean O/E':
        plt.axhline(1, color='#000000', linestyle='dotted', zorder=0)

    # Formatting
    if yColumn == 'R Squared':
        plt.ylim(0, 1)
    plt.xlabel(xColumn)
    plt.ylabel(yColumn)
    plt.legend(loc=legendLoc)
    plt.title(title)

    if outputPath is None:
        plt.show()
    else:
        plt.savefig(outputPath, bbox_inches='tight')

    plt.close()
    return


def meVsUSGSGraphs(pathToFigures):

    scenario = 'Dry'
    regions = ['IntMnt', 'Xeric']
    # regions = ['IntMnt']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    baseUSGSPath = 'OriginalModelAssessment'
    figureNumber = 8

    for region in regions:

        # Get data frame of USGS results
        usgsModelResults = pandas.DataFrame()
        for month in months:
            monthResultsPath = os.path.join(baseUSGSPath, region, month, 'Output',
                                       'scoreModelResults_average.csv')
            monthResults = pandas.read_csv(monthResultsPath)
            monthResults['Month'] = month.title()
            usgsModelResults = usgsModelResults.append(monthResults, ignore_index=True)

        # Fix Base DataSet names for output
        datasetInfo = usgsModelResults['Base DataSet']
        usgsModelResults['Base DataSet'] = datasetInfo.str.split(',').str.get(0)

        # desiredColumnOrder = ['Month', 'Base DataSet', 'Model Method', 'R Squared', 'Mean O/E',
        #                       'Standard Deviation O/E', 'Mean Squared Error', 'RMSE (cfs)']
        # usgsModelResults = usgsModelResults[desiredColumnOrder]

        # Get my best results
        csvPath = os.path.join('BestModels', scenario + region + 'BestModels.csv')
        myBestModelResults = pandas.read_csv(csvPath)

        # Line graphs to compare results by month
        yColumn = 'R Squared'
        title = scenario + ' ' + region + ' R-Squared Comparison'
        outputPath = os.path.join(pathToFigures, '4-' + str(figureNumber) + ' USGS Comparison ' + yColumn + ' ' +
                                  scenario + ' ' + region)
        if region == 'IntMnt':
            comparisonLineGraph(myBestModelResults, usgsModelResults, yColumn, title, outputPath, 'lower left')
        else:
            comparisonLineGraph(myBestModelResults, usgsModelResults, yColumn, title, outputPath)

        yColumn = 'Mean O/E'
        title = scenario + ' ' + region + ' ' + yColumn + ' Comparison'
        outputPath = os.path.join(pathToFigures, '4-' + str(figureNumber) + ' USGS Comparison Mean OE ' +
                                  scenario + ' ' + region)
        if region == 'Xeric':
            comparisonLineGraph(myBestModelResults, usgsModelResults, yColumn, title, outputPath, 'upper left')
        else:
            comparisonLineGraph(myBestModelResults, usgsModelResults, yColumn, title, outputPath, 'lower left')

        figureNumber += 1

    return


# Parameters
pathToFigures = os.path.join('..', '..', '..', 'Figures')
pathToAppendixFigures = os.path.join(pathToFigures, 'AppendixC')
baseDataPath = '../AllMonthsDryHalf'

# Graph production
# specificFoldGraphs(baseDataPath, pathToFigures)
# julyCrossValidatedResults(baseDataPath, pathToFigures)
# appendixCGraphs(pathToAppendixFigures)
# dataTransformationGraph(baseDataPath, pathToFigures)
# generalBestResultsGraphs(pathToFigures)
meVsUSGSGraphs(pathToFigures)

