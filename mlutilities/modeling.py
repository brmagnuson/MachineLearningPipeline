import sklearn.grid_search
import mlutilities.types

def tuneModel(dataSet, modelCreationConfiguration):
    """

    :param dataSet: Presumes that the last column in nonFeaturesDataFrame is the label.
    :param modelCreationConfiguration:
    :return:
    """
    # Get features and label from dataSet
    # A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    X = dataSet.featuresDataFrame
    y = dataSet.labelSeries

    # Grid search to find best parameters.
    gridSearchPredictor = sklearn.grid_search.GridSearchCV(modelCreationConfiguration.modelMethod(),
                                                           modelCreationConfiguration.parameterGrid,
                                                           scoring=modelCreationConfiguration.scoreMethod,
                                                           cv=5,
                                                           refit=False)
    gridSearchPredictor.fit(X, y)

    # GridSearchCV returns negative scores for loss functions (like MSE) so that highest score is best, so this
    # must be corrected for reporting
    if modelCreationConfiguration.scoreMethod == 'mean_squared_error':
        bestScore = -gridSearchPredictor.best_score_
    else:
        bestScore = gridSearchPredictor.best_score_

    # Create new TunedModelConfiguration object
    tunedModelConfiguration = mlutilities.types.TunedModelConfiguration('Tuned ' + modelCreationConfiguration.description \
                                                                            + ' for DataSet: ' + dataSet.description,
                                                                        dataSet,
                                                                        modelCreationConfiguration.modelMethod,
                                                                        gridSearchPredictor.best_params_,
                                                                        modelCreationConfiguration.scoreMethod,
                                                                        bestScore,
                                                                        gridSearchPredictor.grid_scores_)
    return tunedModelConfiguration


def tuneModels(dataSets, modelCreationConfigurations):
    """
    Wrapper function to loop through multiple data sets and model creation configurations
    :param dataSets:
    :param modelCreationConfigurations:
    :return:
    """

    tunedModelConfigurations = []
    for dataSet in dataSets:
        for modelCreationConfiguration in modelCreationConfigurations:
            tunedModelConfiguration = tuneModel(dataSet, modelCreationConfiguration)
            tunedModelConfigurations.append(tunedModelConfiguration)
    return tunedModelConfigurations
