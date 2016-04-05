import pandas
import matplotlib.pyplot


def createScoreDataFrame(scoreModelResults):
    """
    Creates a pandas dataframe from the scored results of various models for easier comparison
    :param scoreModelResults:
    :return: DataFrame
    """

    columnNames = ['Base DataSet', 'Model Method', 'Parameters']

    # Extract scoring function names
    columnNames += [str(modelScore.modelScoreMethod.description) for modelScore in scoreModelResults[0].modelScores]

    # Build rows of data frame, one for each model result
    rows = []
    for scoreModelResult in scoreModelResults:

        formattedDescription = scoreModelResult.description.split(':')[-1].strip()

        row = [formattedDescription,
               scoreModelResult.modellingMethod.description,
               scoreModelResult.parameters]

        # Add each score
        for modelScore in scoreModelResult.modelScores:
            row.append(modelScore.score)

        rows.append(row)

    return pandas.DataFrame(rows, columns=columnNames)


def barChart(dataFrame, column, title, outputPath=None, color='b'):
    """
    Creates a simple bar chart from a single column of a pandas dataframe
    :param dataframe:
    :param column:
    :param title:
    :param outputPath: Optional. If None, graphic is displayed rather than saved to file.
    """
    # Create new dataframe of just the desired column and sort accordingly
    if column == 'R Squared':
        ascendingBoolean = False
    else:
        ascendingBoolean = True
    plotDataFrame = dataFrame[[column]].sort(columns=column, ascending=ascendingBoolean)

    # Create graphic
    plotDataFrame[column].plot(kind='bar', color=color, edgecolor='w')
    matplotlib.pyplot.xlabel('Models')
    matplotlib.pyplot.ylabel(column)
    matplotlib.pyplot.title(title)

    # Display or save to file
    if outputPath is None:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.savefig(outputPath, bbox_inches='tight')

    # Close window
    matplotlib.pyplot.close()


def scatterPlot(dataFrame, xColumn, yColumn, title, outputPath=None, color='b'):
    """
    Uses two columns of a pandas dataframe to make a scatterplot
    :param dataFrame:
    :param xColumn:
    :param yColumn:
    :param title:
    :param outputPath: Optional. If None, graphic is displayed rather than saved to file.
    """
    # Create graphic
    matplotlib.pyplot.scatter(dataFrame[xColumn], dataFrame[yColumn], color=color)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel(xColumn)
    matplotlib.pyplot.ylabel(yColumn)

    # Display or save to file
    if outputPath is None:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.savefig(outputPath, bbox_inches='tight')

    # Close window
    matplotlib.pyplot.close()

