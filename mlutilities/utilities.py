import pandas
import matplotlib.pyplot

def createScoreDataFrame(scoreModelResults):
    """

    :param scoreModelResults:
    :return:
    """

    columnNames = ['Base DataSet', 'Model Method', 'Parameters']

    # Extract scoring function names
    columnNames += [str(modelScore.modelScoreMethod.description) for modelScore in scoreModelResults[0].modelScores]

    # Build rows of data frame, one for each model result
    rows = []
    for scoreModelResult in scoreModelResults:

        formattedDescription = scoreModelResult.description.split(':')[-1].strip()

        row = [formattedDescription,
               str(scoreModelResult.modellingMethod.description),
               str(scoreModelResult.parameters)]

        # Add each score
        for modelScore in scoreModelResult.modelScores:
            row.append(modelScore.score)

        rows.append(row)

    return pandas.DataFrame(rows, columns=columnNames)


def barChart(originalDataFrame, column, title, outputPath=None):
    """
    Creates a simple bar chart
    :param dataframe:
    :param columnName:
    :param outputPath:
    :return:
    """
    # Create new dataframe of just the desired column and sort accordingly
    if column == 'R Squared':
        ascendingBoolean = False
    else:
        ascendingBoolean = True
    plotDataFrame = originalDataFrame[[column]].sort(columns=column, ascending=ascendingBoolean)

    # Create graphic
    plotDataFrame[column].plot(kind='bar', edgecolor='w')
    matplotlib.pyplot.xlabel('Models')
    matplotlib.pyplot.ylabel(column)
    matplotlib.pyplot.title(title)

    # Display or save to file
    if outputPath == None:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.savefig(outputPath, bbox_inches='tight')

    # Close window
    matplotlib.pyplot.close()


def scatterPlot(dataFrame, xColumn, yColumn, title, outputPath=None):
    """

    :param dataFrame:
    :param xColumn:
    :param yColumn:
    :param outputPath:
    :return:
    """
    # Create graphic
    matplotlib.pyplot.scatter(dataFrame[xColumn], dataFrame[yColumn])
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel(xColumn)
    matplotlib.pyplot.ylabel(yColumn)

    # Display or save to file
    if outputPath == None:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.savefig(outputPath, bbox_inches='tight')

    # Close window
    matplotlib.pyplot.close()


