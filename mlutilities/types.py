import pandas
import numpy
import copy


class DataSet:
    """
    Everything we need to know about a data set to use it in our pipeline. It either reads in 
     a csv at a specific file location and creates a Pandas data frame or uses a Pandas data frame 
     to write out a csv to a specific file location.
    featuresIndex should be index of column at which your features start, counting from 0. The default assumption is
     that the first column is the label and the second column starts the features.
    """
    def __init__(self, description, path, mode='r', dataFrame=None, featuresIndex=1, labelIndex=0):
        self.description = description
        self.path = path
        self.mode = mode
        self.dataFrame = copy.deepcopy(dataFrame)
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

        # As long as there is a label, create the labelSeries
        if not labelIndex == None:
            self.labelSeries = self.dataFrame.iloc[:, labelIndex]

    def __str__(self):
        return self.__class__.__name__ + ' ' + self.description + ', Path: \'' + self.path + '\''

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
    This associates both pieces of a DataSet split into testing and training.
    """
    def __init__(self, trainDataSet, testDataSet):
        self.trainDataSet = trainDataSet
        self.testDataSet = testDataSet

    def __str__(self):
        return self.__class__.__name__ + '\n' + \
               'Training: ' + str(self.trainDataSet) + '\n' + \
               'Testing: ' + str(self.testDataSet)


class Scaler:
    """
    This associates a MinMaxScaler object with the original, unscaled DataSet used for fitting it.
    """
    def __init__(self, dataSetUsedToFit, scalingObject):
        self.dataSetUsedToFit = dataSetUsedToFit
        self.scalingObject = scalingObject

    def __str__(self):
        return self.__class__.__name__ + ' for ' + str(self.dataSetUsedToFit)


class ExtractSpecificFeatures:

    def __init__(self, featureList):
        self.featureList = featureList

    def fit_transform(self, dataFrame):
        return dataFrame[self.featureList]

    def transform(self, dataFrame):
        return dataFrame[self.featureList]


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


class Transformer:
    """
    This associates a decomposition or feature_selection object with the original, untransformed DataSet
    used for fitting it.
    """
    def __init__(self, description, selectionOrExtraction, dataSetUsedToFit, transformingObject):
        self.description = description
        self.selectionOrExtraction = selectionOrExtraction
        self.dataSetUsedToFit = dataSetUsedToFit
        self.transformingObject = transformingObject

    def __str__(self):
        return self.__class__.__name__ + ' ' + str(self.description) + \
               ' for ' + str(self.dataSetUsedToFit)


class ModellingMethod:
    """
    This associates an sklearn modelling function with a description for easier processing.
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
    The outcome of tuneModel(), which contains everything important found in tuning parameters for a ModellingMethod.
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
    Everything needed to use applyModel(): the modelling method, its parameters, the DataSet to train on, and
    the DataSet to predict for. (Note that if no test DataSet is passed in, the applyModel() will predict for the
    training DataSet by default.)
    """
    def __init__(self, description, modellingMethod, parameters, trainDataSet, testDataSet=None):
        self.description = description
        self.modellingMethod = modellingMethod
        self.parameters = parameters
        self.trainDataSet = trainDataSet
        if testDataSet == None:
            testDataSet = trainDataSet
        self.testDataSet = testDataSet

    def __str__(self):
        return self.__class__.__name__ + ' Description: ' + self.description + '\n' + \
               'Model: ' + str(self.modellingMethod) + '\n' + \
               'Parameters: ' + str(self.parameters) + '\n' + \
               'Training Data Set: ' + str(self.trainDataSet) + '\n' + \
               'Testing Data Set: ' + str(self.testDataSet)

class ApplyModelResult:
    """
    The outcome of applyModel(), which can then be scored.
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
    The outcome of scoreModel: how a ModellingMethod performed on various ModelScoreMethods
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
    Relates a score to a ModelScoreMethod.
    """
    def __init__(self, score, modelScoreMethod):
        self.score = score
        self.modelScoreMethod = modelScoreMethod

    def __str__(self):
        return self.__class__.__name__ + ' Scoring Function: ' + str(self.modelScoreMethod) + ' Score: ' + str(self.score)


class ModelScoreMethod:
    """
    This associates an sklearn.metrics scoring function with a description to facilitate processing.
    """
    def __init__(self, description, function):
        self.description = description
        self.function = function

    def __str__(self):
        return self.__class__.__name__ + ' ' + self.description


class AveragingEnsemble:
    """
    This takes a list of models and averages their predictions for a test dataset (optionally as a weighted average),
    mimicking a generic sklearn regression object.
    Predictor configurations contain the functions and parameters necessary to initialize predictors.
    If no weights are passed in, a regular arithmetic mean is calculated.
    Weights should be a list of the same length as predictorConfigurations and in the same order.
    """
    def __init__(self, predictorConfigurations, weights=None):
        if weights != None:
            if len(predictorConfigurations) != len(weights):
                raise Exception('Each predictor configuration needs a weight.\n' +
                                'Number of predictor configurations: ' + str(len(predictorConfigurations)) + '\n' +
                                'Number of weights: ' + str(len(weights)))
        predictors = []
        for predictorConfiguration in predictorConfigurations:
            predictor = predictorConfiguration.predictorFunction(**predictorConfiguration.parameters)
            predictors.append(predictor)
        self.predictors = predictors
        self.weights = weights

    def fit(self, X, y):
        for predictor in self.predictors:
            predictor.fit(X, y)

    def predict(self, X):

        # Make prediction using each predictor
        self.predictions = []
        for predictor in self.predictors:
            prediction = predictor.predict(X)
            self.predictions.append(prediction)

        # Find the average prediction for each observation
        meanPrediction = numpy.average(self.predictions, axis=0, weights=self.weights)
        return meanPrediction

    def __str__(self):
        return self.__class__.__name__ + '\n' + \
               'Predictors: ' + str(self.predictors) + '\n' + \
               'Weights: ' + str(self.weights)


class StackingEnsemble:
    """
    This takes a list of models and stacks their predictions for a dataset, mimicking a generic sklearn regression object.
    Predictor configurations contain the functions and parameters necessary to initialize predictors.
    """
    def __init__(self, basePredictorConfigurations, stackingPredictorConfiguration, includeOriginalFeatures=False):
        basePredictors = []
        for basePredictorConfiguration in basePredictorConfigurations:
            basePredictor = basePredictorConfiguration.predictorFunction(**basePredictorConfiguration.parameters)
            basePredictors.append(basePredictor)
        self.basePredictors = basePredictors
        self.stackingPredictor = stackingPredictorConfiguration.predictorFunction(**stackingPredictorConfiguration.parameters)
        self.includeOriginalFeatures = includeOriginalFeatures

    def fit(self, X, y):

        # Building a dataframe of each base predictor's predictions
        basePredictions = pandas.DataFrame()
        counter = 1
        for basePredictor in self.basePredictors:
            basePredictor.fit(X, y)
            basePrediction = basePredictor.predict(X)
            basePredictions['Base Predictor ' + str(counter)] = basePrediction
            counter += 1

        if self.includeOriginalFeatures:
            basePredictions = pandas.concat([basePredictions, X], axis=1)

        # Use this new dataframe to fit the stacking predictor
        self.stackingPredictor.fit(basePredictions, y)


    def predict(self, X):

        # Building a dataframe of each base predictor's predictions
        basePredictions = pandas.DataFrame()
        for basePredictor in self.basePredictors:
            basePrediction = basePredictor.predict(X)
            basePredictions[str(basePredictor)] = basePrediction

        if self.includeOriginalFeatures:
            basePredictions = pandas.concat([basePredictions, X], axis=1)

        # Use this new dataframe to predict using the stacking predictor
        stackedPrediction = self.stackingPredictor.predict(basePredictions)
        return stackedPrediction

    def __str__(self):
        return self.__class__.__name__ + '\n' + \
               'Stacking predictor: ' + str(self.stackingPredictor) + '\n' + \
               'Base predictors: ' + str(self.basePredictors)


class PredictorConfiguration:
    """
    If no parameters are provided, defaults for that predictor function are used
    """
    def __init__(self, description, predictorFunction, parameters=None):
        self.description = description
        self.predictorFunction = predictorFunction
        self.parameters = parameters

    def __str__(self):
        return self.__class__.__name__ + ' ' + self.description + ' ' + str(self.parameters)

    def __repr__(self):
        return self.description + ' ' + str(self.parameters)
