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

    # Create new TunedModelConfiguration object
    tunedModelConfiguration = mlutilities.types.TunedModelConfiguration('Tuned ' + modelCreationConfiguration.description \
                                                                            + ' for DataSet: ' + dataSet.description,
                                                                        dataSet,
                                                                        modelCreationConfiguration.modelMethod,
                                                                        gridSearchPredictor.best_params_,
                                                                        modelCreationConfiguration.scoreMethod,
                                                                        gridSearchPredictor.best_score_,
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
