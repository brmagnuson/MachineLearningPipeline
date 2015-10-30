import pandas
import copy

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


class ModelCreationConfiguration:
    """
    Note: scoreMethod can be a callable object/function or a string ('r2', 'mean_absolute_error',
    'mean_squared_error', or 'median_absolute_error')
    """
    def __init__(self, description, modelMethod, parameterGrid, scoreMethod):
        self.description = description
        self.modelMethod = modelMethod
        self.parameterGrid = parameterGrid
        self.scoreMethod = scoreMethod

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Model: ' + str(self.modelMethod) + '\n' + \
               'Parameter Grid: ' + str(self.parameterGrid) + '\n' + \
               'Scoring Method: ' + self.scoreMethod


class TunedModelConfiguration:
    """

    """
    def __init__(self, description, dataSet, modelMethod, parameters, scoreMethod, bestScore, gridScores):
        self.description = description
        self.dataSet = dataSet
        self.modelMethod = modelMethod
        self.parameters = parameters
        self.scoreMethod = scoreMethod
        self.bestScore = bestScore
        self.gridScores = gridScores

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Dataset: ' + str(self.dataSet) + '\n' + \
               'Model: ' + str(self.modelMethod) + '\n' + \
               'Tuned Parameters: ' + str(self.parameters) + '\n' + \
               'Scoring Method: ' + self.scoreMethod + '\n' + \
               'Tuned Training Score: ' + str(self.bestScore) + '\n' + \
               'Grid Scores: ' + str(self.gridScores)


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

