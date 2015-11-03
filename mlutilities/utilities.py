import pandas

def createScoreDataFrame(scoreModelResults):
    """

    :param scoreModelResults:
    :return:
    """

    columnNames = ['Description', 'Model Method', 'Parameters']

    # Extract scoring function names
    columnNames += [str(modelScore.modelScoreMethod.description) for modelScore in scoreModelResults[0].modelScores]

    # Build rows of data frame, one for each model result
    rows = []
    for scoreModelResult in scoreModelResults:

        row = [scoreModelResult.description,
               str(scoreModelResult.modellingMethod.description),
               str(scoreModelResult.parameters)]

        # Add each score
        for modelScore in scoreModelResult.modelScores:
            row.append(modelScore.score)

        rows.append(row)

    return pandas.DataFrame(rows, columns=columnNames)




