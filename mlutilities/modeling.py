import numpy
import sklearn.grid_search
import sklearn.cross_validation
import mlutilities.types


def tuneModel(dataSet, tuneModelConfiguration, randomSeed=None, n_jobs=1):
    """
    Finds the best combination of a set of model parameters for a DataSet.
    :param dataSet: Presumes that the last column in nonFeaturesDataFrame is the label.
    :param modelCreationConfiguration:
    :return: TuneModelResult
    """
    # Get features and label from dataSet
    features = dataSet.featuresDataFrame
    label = dataSet.labelSeries

    # Grid search to find best parameters.
    gridSearchPredictor = sklearn.grid_search.GridSearchCV(tuneModelConfiguration.modellingMethod.function(),
                                                           tuneModelConfiguration.parameterGrid,
                                                           scoring=tuneModelConfiguration.scoreMethod,
                                                           cv=sklearn.cross_validation.KFold(len(label),
                                                                                             n_folds=5,
                                                                                             shuffle=True,
                                                                                             random_state=randomSeed),
                                                           refit=False,
                                                           n_jobs=n_jobs)
    gridSearchPredictor.fit(features, label)

    # GridSearchCV returns negative scores for loss functions (like MSE) so that highest score is best, so this
    # must be corrected for reporting
    if tuneModelConfiguration.scoreMethod == 'mean_squared_error':
        bestScore = -gridSearchPredictor.best_score_
    else:
        bestScore = gridSearchPredictor.best_score_

    # Create new TunedModelConfiguration object
    tuneModelResult = mlutilities.types.TuneModelResult('Tuned ' + tuneModelConfiguration.description \
                                                            + ' for DataSet: ' + dataSet.description,
                                                        dataSet,
                                                        tuneModelConfiguration.modellingMethod,
                                                        gridSearchPredictor.best_params_,
                                                        tuneModelConfiguration.scoreMethod,
                                                        bestScore,
                                                        gridSearchPredictor.grid_scores_)
    return tuneModelResult


def tuneModels(dataSets, tuneModelConfigurations, randomSeed=None):
    """
    Wrapper function to loop through multiple data sets and model creation configurations
    :param dataSets:
    :param modelCreationConfigurations:
    :return: list of TuneModelResults
    """
    tuneModelResults = []
    counter = 1
    total = len(dataSets) * len(tuneModelConfigurations)
    for dataSet in dataSets:
        for tuneModelConfiguration in tuneModelConfigurations:
            print('Tuning (%s of %s):' % (counter, total), tuneModelConfiguration.description, 'for', dataSet.description)
            tuneModelResult = tuneModel(dataSet, tuneModelConfiguration, randomSeed)
            tuneModelResults.append(tuneModelResult)
            counter += 1
    return tuneModelResults


def applyModel(applyModelConfiguration):
    """
    Given a model, its parameters, a training set, and a test set, train the model and apply to the test data
    :param applyModelConfiguration
    :return: ApplyModelResult
    """
    # Get features and label from DataSets
    trainFeatures = applyModelConfiguration.trainDataSet.featuresDataFrame
    trainLabel = applyModelConfiguration.trainDataSet.labelSeries
    testFeatures = applyModelConfiguration.testDataSet.featuresDataFrame

    # Train model on training set
    predictor = applyModelConfiguration.modellingMethod.function(**applyModelConfiguration.parameters)
    predictor.fit(trainFeatures, trainLabel)

    # Predict for testing set
    testPredictions = predictor.predict(testFeatures)

    # Build ApplyModelResult
    applyModelResult = mlutilities.types.ApplyModelResult(applyModelConfiguration.description.replace('Apply', 'Result:'),
                                                          testPredictions,
                                                          applyModelConfiguration.testDataSet,
                                                          applyModelConfiguration.modellingMethod,
                                                          applyModelConfiguration.parameters)
    return applyModelResult


def applyModels(applyModelConfigurations, subTaskPrint=True):
    """
    Wrapper function to loop through multiple ApplyModelConfigurations
    :param applyModelConfigurations:
    :return: list of ApplyModelResults
    """
    applyModelResults = []
    counter = 1
    total = len(applyModelConfigurations)
    for applyModelConfiguration in applyModelConfigurations:
        if subTaskPrint:
            print('Applying (%s of %s):' % (counter, total), applyModelConfiguration.description)
        applyModelResult = applyModel(applyModelConfiguration)
        applyModelResults.append(applyModelResult)
        counter += 1
    return applyModelResults


def scoreModel(applyModelResult, modelScoreMethods):
    """
    Scores the result of applying a model based on various sklearn.metrics scoring methods
    :param applyModelResult:
    :param scoringFunction: list of ModelScoreMethods
    :return: ScoreModelResult
    """
    testLabel = applyModelResult.testDataSet.labelSeries
    testPredictions = applyModelResult.testPredictions

    modelScores = []
    for modelScoreMethod in modelScoreMethods:
        score = modelScoreMethod.function(testLabel, testPredictions)
        modelScore = mlutilities.types.ModelScore(score, modelScoreMethod)
        modelScores.append(modelScore)

    scoreModelResult = mlutilities.types.ScoreModelResult(applyModelResult.description + ', Test Score',
                                                          applyModelResult.modellingMethod,
                                                          applyModelResult.parameters,
                                                          modelScores)
    return scoreModelResult


def scoreModels(applyModelResults, modelScoreMethods):
    """
    Wrapper function to loop through multiple ApplyModelResult objects
    :param applyModelResults:
    :param scoringFunction:
    :return: list of ScoreModelResults
    """
    scoreModelResults = []
    for applyModelResult in applyModelResults:
        scoreModelResult = scoreModel(applyModelResult, modelScoreMethods)
        scoreModelResults.append(scoreModelResult)
    return scoreModelResults


def meanObservedExpectedScore(y_true, y_pred):

    """
    Function modeled after sklearn.metrics scoring functions to calculate mean O/E score for a model.
    Works when y_true and y_pred are vector-like pandas DataFrames or numpy Arrays.
    Will ignore any NaN or inf values.
    """
    oeRatios = y_true / y_pred
    meanOE = numpy.nanmean(oeRatios[~ numpy.isinf(oeRatios)])
    return float(meanOE)


def sdObservedExpectedScore(y_true, y_pred):
    """
    Function modeled after sklearn.metrics scoring functions to calculate the standard deviation of the O/E score for a
    model.
    Works when y_true and y_pred are vector-like pandas DataFrames or numpy Arrays.
    Will ignore any NaN or inf values.
    """
    oeRatios = y_true / y_pred
    standardDeviationOE = numpy.nanstd(oeRatios[~ numpy.isinf(oeRatios)])
    return float(standardDeviationOE)
