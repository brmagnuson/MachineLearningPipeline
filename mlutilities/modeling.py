import sklearn.grid_search
import sklearn.metrics
import mlutilities.types


def tuneModel(dataSet, tuneModelConfiguration):
    """

    :param dataSet: Presumes that the last column in nonFeaturesDataFrame is the label.
    :param modelCreationConfiguration:
    :return:
    """
    # Get features and label from dataSet
    features = dataSet.featuresDataFrame
    label = dataSet.labelSeries

    # Grid search to find best parameters.
    gridSearchPredictor = sklearn.grid_search.GridSearchCV(tuneModelConfiguration.modelMethod(),
                                                           tuneModelConfiguration.parameterGrid,
                                                           scoring=tuneModelConfiguration.scoreMethod,
                                                           cv=5,
                                                           refit=False)
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
                                                                 tuneModelConfiguration.modelMethod,
                                                                 gridSearchPredictor.best_params_,
                                                                 tuneModelConfiguration.scoreMethod,
                                                                 bestScore,
                                                                 gridSearchPredictor.grid_scores_)
    return tuneModelResult


def tuneModels(dataSets, tuneModelConfigurations):
    """
    Wrapper function to loop through multiple data sets and model creation configurations
    :param dataSets:
    :param modelCreationConfigurations:
    :return:
    """
    tuneModelResults = []
    for dataSet in dataSets:
        for tuneModelConfiguration in tuneModelConfigurations:
            tuneModelResult = tuneModel(dataSet, tuneModelConfiguration)
            tuneModelResults.append(tuneModelResult)
    return tuneModelResults


def applyModel(applyModelConfiguration):
    """

    :param tunedModelConfig:
    :param trainDataSet:
    :param testDataSet:
    :return:
    """
    # Get features and label from dataSet
    trainFeatures = applyModelConfiguration.trainDataSet.featuresDataFrame
    trainLabel = applyModelConfiguration.trainDataSet.labelSeries
    testFeatures = applyModelConfiguration.testDataSet.featuresDataFrame
    testLabel = applyModelConfiguration.testDataSet.labelSeries

    # Train model
    predictor = applyModelConfiguration.modelMethod(**applyModelConfiguration.parameters)
    predictor.fit(trainFeatures, trainLabel)

    # Predict
    testPredictions = predictor.predict(testFeatures)

    # Build ApplyModelResult
    applyModelResult = mlutilities.types.ApplyModelResult(applyModelConfiguration.description + ' Result',
                                                          testPredictions,
                                                          applyModelConfiguration.testDataSet,
                                                          applyModelConfiguration.modelMethod,
                                                          applyModelConfiguration.parameters)
    return applyModelResult


def applyModels(applyModelConfigurations):
    """
    Wrapper function to loop through multiple ApplyModelConfigurations
    :param applyModelConfigurations:
    :return:
    """
    applyModelResults = []
    for applyModelConfiguration in applyModelConfigurations:
        applyModelResult = applyModel(applyModelConfiguration)
        applyModelResults.append(applyModelResult)
    return applyModelResults
