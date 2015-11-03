import pandas


class DataSet:
    """
    Everything we need to know about a data set to use it in our pipeline. It either reads in 
     a csv at a specific file location and creates a Pandas data frame or uses a Pandas data frame 
     to write out a csv to a specific file location.
    featuresIndex should be index at which your features start, counting from 0.
    """
    def __init__(self, description, path, mode='r', dataFrame=None, featuresIndex=None, labelIndex=None):
        self.description = description
        self.path = path
        self.mode = mode
        self.dataFrame = dataFrame
        self.featuresIndex = featuresIndex
        self.labelIndex = labelIndex

        if mode == 'r':
            # Read csv into pandas frame
            self.dataFrame = pandas.read_csv(self.path)
        else:
            # Write pandas frame into csv
            self.dataFrame.to_csv(path, index=False)

        self.nonFeaturesDataFrame = self.dataFrame.ix[:, :self.featuresIndex]
        self.featuresDataFrame = self.dataFrame.ix[:, self.featuresIndex:]
        self.labelSeries = self.dataFrame.iloc[:, labelIndex]

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + ', Path: \'' + self.path + '\''

    def __eq__(self, other):
        """
        Compares two data sets on the basis of their paths
        """
        return (isinstance(other, self.__class__)
            and self.path == other.path)

    def __ne__(self, other):
        return not self.__eq__(other)


class SplitDataSet:
    """

    """
    def __init__(self, trainDataSet, testDataSet):
        self.trainDataSet = trainDataSet
        self.testDataSet = testDataSet

    def __str__(self):
        return self.__class__.__name__ + '\n' + \
               'Training: ' + self.trainDataSet + '\n' + \
               'Testing: ' + self.testDataSet


class FeatureEngineeringConfiguration:
    """
    Everything we need to know to perform a specific type of feature selection on a DataSet.
    """
    def __init__(self, description, selectionOrExtraction, method, parameters):
        self.description = description
        self.selectionOrExtraction = selectionOrExtraction
        self.method = method
        self.parameters = parameters

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description


class ModellingMethod:
    """

    """
    def __init__(self, description, function):
        self.description = description
        self.function = function

    def __str__(self):
        return self.__class__.__name__ + ' ' + self.description


class TuneModelConfiguration:
    """
    Note: scoreMethod can be a callable object/function or a string ('r2', 'mean_absolute_error',
    'mean_squared_error', or 'median_absolute_error')
    """
    def __init__(self, description, modellingMethod, parameterGrid, scoreMethod):
        self.description = description
        self.modellingMethod = modellingMethod
        self.parameterGrid = parameterGrid
        self.scoreMethod = scoreMethod

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Parameter Grid: ' + str(self.parameterGrid) + '\n' + \
               'Scoring Method: ' + self.scoreMethod


class TuneModelResult:
    """

    """
    def __init__(self, description, dataSet, modellingMethod, parameters, scoreMethod, bestScore, gridScores):
        self.description = description
        self.dataSet = dataSet
        self.modellingMethod = modellingMethod
        self.parameters = parameters
        self.scoreMethod = scoreMethod
        self.bestScore = bestScore
        self.gridScores = gridScores

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Dataset: ' + str(self.dataSet) + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Tuned Parameters: ' + str(self.parameters) + '\n' + \
               'Scoring Method: ' + self.scoreMethod + '\n' + \
               'Tuned Training Score: ' + str(self.bestScore) + '\n' + \
               'Grid Scores: ' + str(self.gridScores)


class ApplyModelConfiguration:
    """

    """
    def __init__(self, description, modellingMethod, parameters, trainDataSet, testDataSet):
        self.description = description
        self.modellingMethod = modellingMethod
        self.parameters = parameters
        self.trainDataSet = trainDataSet
        self.testDataSet = testDataSet

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Parameters: ' + str(self.parameters) + '\n' + \
               'Training Data Set: ' + str(self.trainDataSet) + '\n' + \
               'Testing Data Set: ' + str(self.testDataSet)

class ApplyModelResult:
    """

    """
    def __init__(self, description, testPredictions, testDataSet, modellingMethod, parameters):
        self.description = description
        self.testPredictions = testPredictions
        self.testDataSet = testDataSet
        self.modellingMethod = modellingMethod
        self.parameters = parameters

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Testing Data Set: ' + str(self.testDataSet) + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Parameters: ' + str(self.parameters)


class ScoreModelResult:
    """

    """
    def __init__(self, description, modellingMethod, parameters, modelScores):
        self.description = description
        self.modellingMethod = modellingMethod
        self.parameters = parameters
        self.modelScores = modelScores

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Parameters: ' + str(self.parameters) + '\n' + \
               'Model Scores:\n' + '\n'.join(map(str, self.modelScores)) + '\n'


class ModelScore:
    """

    """
    def __init__(self, score, modelScoreMethod):
        self.score = score
        self.modelScoreMethod = modelScoreMethod

    def __str__(self):
        return self.__class__.__name__ + ' Scoring Function: ' + str(self.scoringFunction) + ' Score: ' + str(self.score)


class ModelScoreMethod:
    """

    """
    def __init__(self, description, function):
        self.description = description
        self.function = function

    def __str__(self):
        return self.__class__.__name__ + ' ' + self.description

