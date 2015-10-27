import os
import pandas
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.cross_validation
import mlutilities.types


def scaleDataSet(dataSet):
    """
    Scale columns of data set from 0 to 1
    :param dataSet:
    :return:
    """
    # Scale dataset from 0 to 1 using sklearn
    minMaxScaler = sklearn.preprocessing.MinMaxScaler()
    scaledValues = minMaxScaler.fit_transform(dataSet.featuresDataFrame)

    # Translate values back into pandas data frame
    scaledValuesDataFrame = pandas.DataFrame(scaledValues, columns=dataSet.featuresDataFrame.columns)

    # Concatenate values and ids into new pandas data frame
    completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, scaledValuesDataFrame], axis=1)

    # Assign values to new DataSet object
    newDescription = dataSet.description + ' Scaled'
    newPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + '_scaled.csv'
    scaledDataSet = mlutilities.types.DataSet(newDescription,
                                              newPath,
                                              'w',
                                              completeDataFrame,
                                              dataSet.featuresIndex,
                                              dataSet.labelIndex)

    return scaledDataSet


def scaleDataSets(dataSets):
    """
    Wrapper function to loop through scaling multiple data sets
    :param dataSets: list of DataSet objects
    :return: list of scaled DataSet objects
    """
    scaledDataSets = []
    for dataSet in dataSets:
        scaledDataSet = scaleDataSet(dataSet)
        scaledDataSets.append(scaledDataSet)
    return scaledDataSets


def engineerFeaturesForDataSet(dataSet, featureEngineeringConfiguration):
    """

    :param dataSet:
    :param featureEngineeringConfiguration:
    :return:
    """
    # Create feature selector using method function and unpacking parameters
    selector = featureEngineeringConfiguration.method(**featureEngineeringConfiguration.parameters)

    # Select features from pandas data frame
    fitTransformValues = selector.fit_transform(dataSet.featuresDataFrame)

    # Build new pandas data frame based on selected/extracted features
    if featureEngineeringConfiguration.selectionOrExtraction == 'selection':
        selectedFeatureIndices = selector.get_support(indices=True)
        columnNames = [dataSet.featuresDataFrame.columns.values[i] for i in selectedFeatureIndices]
        selectedFeaturesDataFrame = dataSet.featuresDataFrame[columnNames]
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, selectedFeaturesDataFrame], axis=1)
    else:
        extractedFeaturesDataFrame = pandas.DataFrame(fitTransformValues)
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, extractedFeaturesDataFrame], axis=1)

    # Assign values to new DataSet object
    newDescription = dataSet.description + ', features selected via ' + featureEngineeringConfiguration.description
    newPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + \
              '_' + featureEngineeringConfiguration.description.replace(' ', '_') + '.csv'
    selectedFeaturesDataSet = mlutilities.types.DataSet(newDescription,
                                                        newPath,
                                                        'w',
                                                        completeDataFrame,
                                                        dataSet.featuresIndex,
                                                        dataSet.labelIndex)

    return selectedFeaturesDataSet


def engineerFeaturesForDataSets(dataSets, featureEngineeringConfigurations):
    """
    Wrapper function to loop through multiple data sets and feature engineering configurations
    :param dataSets: list of DataSet objects
    :param featureEngineeringConfigurations: list of FeatureEngineeringConfiguration objects
    :return: list of feature engineered DataSet objects
    """
    featureEngineeredDatasets = []
    for dataSet in dataSets:
        for featureEngineeringConfiguration in featureEngineeringConfigurations:
            featureEngineeredDataset = engineerFeaturesForDataSet(dataSet, featureEngineeringConfiguration)
            featureEngineeredDatasets.append(featureEngineeredDataset)
    return featureEngineeredDatasets


def splitDataSet(dataSet, testProportion):
    """
    Splits a DataSet's data frame into a test set and a training set based on the given proportion
    :param dataSet: input DataSet
    :param testProportion: float between 0 and 1; proportion that will be held back in the test set
    :return: tuple of training DataSet and testing DataSet
    """
    originalDataFrame = dataSet.dataFrame
    trainDataFrame, testDataFrame = sklearn.cross_validation.train_test_split(originalDataFrame,
                                                                              test_size=testProportion,
                                                                              random_state=1000)
    # Assign values to new DataSet objects
    baseNewDescription = dataSet.description + ', '
    baseNewPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + '_'
    trainDataSet = mlutilities.types.DataSet(baseNewDescription + 'Training Set',
                                             baseNewPath + 'train.csv',
                                             'w',
                                             trainDataFrame,
                                             dataSet.featuresIndex,
                                             dataSet.labelIndex)
    testDataSet = mlutilities.types.DataSet(baseNewDescription + 'Testing Set',
                                            baseNewPath + 'test.csv',
                                            'w',
                                            testDataFrame,
                                            dataSet.featuresIndex,
                                            dataSet.labelIndex)

    return (trainDataSet, testDataSet)


def splitDataSets(dataSets, testProportion):
    """
    Wrapper function to loop through multiple data sets and split them into train/test data
    :param dataSets: list of DataSet objects
    :param testProportion: float between 0 and 1; proportion that will be held back in the test set
    :return: tuple of lists of training DataSets and testing DataSets
    """
    trainDataSets = []
    testDataSets = []
    for dataSet in dataSets:
        trainDataSet, testDataSet = splitDataSet(dataSet, testProportion)
        trainDataSets.append(trainDataSet)
        testDataSets.append(testDataSets)
    return (trainDataSets, testDataSets)
