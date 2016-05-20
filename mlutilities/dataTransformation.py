import os
import pandas
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.cross_validation
import mlutilities.types


def scaleDataSet(dataSet):
    """
    Scale columns of data set from 0 to 1
    :param dataSet: DataSet object
    :return: a tuple containing the scaled data set and the scaling object for use on another data set, if desired
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

    # Create Scaler object to associate original dataSet and minMaxScaler
    scaler = mlutilities.types.Scaler(dataSet, minMaxScaler)

    return scaledDataSet, scaler


def scaleDataSets(dataSets):
    """
    Wrapper function to loop through scaling multiple DataSets
    :param dataSets: list of DataSet objects
    :return: list of scaled DataSet objects and list of Scaler objects
    """
    scaledDataSets = []
    scalers = []
    for dataSet in dataSets:
        scaledDataSet, scaler = scaleDataSet(dataSet)
        scaledDataSets.append(scaledDataSet)
        scalers.append(scaler)
    return scaledDataSets, scalers


def scaleDataSetByScaler(dataSet, scaler):
    """
    Scales columns of data according to a scaler already fit on another (usually associated) DataSet
    :param dataSet: DataSet
    :param scaler: Scaler, which consists of the original DataSet used for fitting and the scaling function
    :return: scaled DataSet
    """
    scaledValues = scaler.scalingObject.transform(dataSet.featuresDataFrame)

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


def engineerFeaturesForDataSet(dataSet, featureEngineeringConfiguration):
    """
    Transforms a DataSet according to a FeatureEngineeringConfiguration's parameters.
    :param dataSet:
    :param featureEngineeringConfiguration: Can be either a selection or a decomposition config
    :return: the transformed DataSet and a Transformer available for use on other DataSets
    """
    # Create feature selector using method function and unpacking parameters
    selector = featureEngineeringConfiguration.method(**featureEngineeringConfiguration.parameters)

    # Select features from pandas data frame
    fitTransformValues = selector.fit_transform(dataSet.featuresDataFrame)

    # Build new pandas data frame based on selected/extracted features
    if featureEngineeringConfiguration.selectionOrExtraction == 'selection':
        if featureEngineeringConfiguration.method == mlutilities.types.ExtractSpecificFeatures:
            selectedFeaturesDataFrame = fitTransformValues
        else:
            selectedFeatureIndices = selector.get_support(indices=True)
            columnNames = [dataSet.featuresDataFrame.columns.values[i] for i in selectedFeatureIndices]
            selectedFeaturesDataFrame = dataSet.featuresDataFrame[columnNames]
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, selectedFeaturesDataFrame], axis=1)
    else:
        extractedFeaturesDataFrame = pandas.DataFrame(fitTransformValues)
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, extractedFeaturesDataFrame], axis=1)

    # Assign values to new DataSet object
    newDescription = dataSet.description + ' features selected via ' + featureEngineeringConfiguration.description
    newPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + \
              '_' + featureEngineeringConfiguration.description.replace(' ', '_') + '.csv'
    selectedFeaturesDataSet = mlutilities.types.DataSet(newDescription,
                                                        newPath,
                                                        'w',
                                                        completeDataFrame,
                                                        dataSet.featuresIndex,
                                                        dataSet.labelIndex)

    # Create Transformer to associate original dataSet and selector
    transformer = mlutilities.types.Transformer(featureEngineeringConfiguration.description,
                                                featureEngineeringConfiguration.selectionOrExtraction,
                                                dataSet,
                                                selector)

    return selectedFeaturesDataSet, transformer


def engineerFeaturesForDataSets(dataSets, featureEngineeringConfigurations):
    """
    Wrapper function to loop through multiple data sets and feature engineering configurations
    :param dataSets: list of DataSets
    :param featureEngineeringConfigurations: list of FeatureEngineeringConfiguration objects
    :return: list of feature engineered DataSets
    """
    featureEngineeredDatasets = []
    transformers = []
    for dataSet in dataSets:
        for featureEngineeringConfiguration in featureEngineeringConfigurations:
            featureEngineeredDataset, transformer = engineerFeaturesForDataSet(dataSet, featureEngineeringConfiguration)
            featureEngineeredDatasets.append(featureEngineeredDataset)
            transformers.append(transformer)
    return featureEngineeredDatasets, transformers


def engineerFeaturesByTransformer(dataSet, transformer):
    """
    Transforms a DataSet's features according to transformer (either a selector or decomposer) already fit on
    another (usually associated) DataSet
    :param dataSet: DataSet
    :param transformer: Transformer, which consists of the original DataSet used for fitting and the selector or
    decomposer created on it
    :return: feature engineered DataSet
    """
    transformedValues = transformer.transformingObject.transform(dataSet.featuresDataFrame)

    # Build new pandas data frame based on selected/extracted features
    if transformer.selectionOrExtraction == 'selection':
        if type(transformer.transformingObject) == mlutilities.types.ExtractSpecificFeatures:
            selectedFeaturesDataFrame = transformedValues
        else:
            selectedFeatureIndices = transformer.transformingObject.get_support(indices=True)
            columnNames = [dataSet.featuresDataFrame.columns.values[i] for i in selectedFeatureIndices]
            selectedFeaturesDataFrame = dataSet.featuresDataFrame[columnNames]
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, selectedFeaturesDataFrame], axis=1)
    else:
        extractedFeaturesDataFrame = pandas.DataFrame(transformedValues)
        completeDataFrame = pandas.concat([dataSet.nonFeaturesDataFrame, extractedFeaturesDataFrame], axis=1)

    # Assign values to new DataSet object
    newDescription = dataSet.description + ' features selected via ' + transformer.description
    newPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + \
              '_' + transformer.description.replace(' ', '_') + '.csv'
    selectedFeaturesDataSet = mlutilities.types.DataSet(newDescription,
                                                        newPath,
                                                        'w',
                                                        completeDataFrame,
                                                        dataSet.featuresIndex,
                                                        dataSet.labelIndex)
    return selectedFeaturesDataSet


def splitDataSet(dataSet, testProportion, randomSeed=None, trainPath=None, testPath=None):
    """
    Splits a DataSet's data frame into a test set and a training set based on the given proportion
    :param dataSet: input DataSet
    :param testProportion: float between 0 and 1; proportion that will be held back in the test set
    :param randomSeed: Optional. Integer. If None, test-train split randomness will not be controlled and replicable.
    :param trainPath: Optional. If None, automatically creates new files in same location and appends _test and _train
    to the end. If specified, you can control location and/or name of the new files.
    :return: SplitDataSet
    """
    originalDataFrame = dataSet.dataFrame
    trainDataFrame, testDataFrame = sklearn.cross_validation.train_test_split(originalDataFrame,
                                                                              test_size=testProportion,
                                                                              random_state=randomSeed)

    # Assign values to new DataSets, using default path option if none was specified
    baseNewDescription = dataSet.description
    baseNewPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + '_'
    if trainPath == None:
        trainPath = baseNewPath + 'train.csv'
    if testPath == None:
        testPath = baseNewPath + 'test.csv'
    trainDataSet = mlutilities.types.DataSet(baseNewDescription + ' Training Set',
                                             trainPath,
                                             'w',
                                             trainDataFrame,
                                             dataSet.featuresIndex,
                                             dataSet.labelIndex)
    testDataSet = mlutilities.types.DataSet(baseNewDescription + ' Testing Set',
                                            testPath,
                                            'w',
                                            testDataFrame,
                                            dataSet.featuresIndex,
                                            dataSet.labelIndex)

    # Join into new SplitDataSet
    theSplitDataSet = mlutilities.types.SplitDataSet(trainDataSet, testDataSet)

    return theSplitDataSet


def splitDataSets(dataSets, testProportion, randomSeed=None):
    """
    Wrapper function to loop through multiple data sets and split them into train/test data
    :param dataSets: list of DataSet objects
    :param testProportion: float between 0 and 1; proportion that will be held back in the test set
    :param seed: Optional. Integer. If None, test-train split randomness will not be controlled and replicable.
    :return: tuple of lists of training DataSets and testing DataSets
    """
    theSplitDataSets = []
    for dataSet in dataSets:
        theSplitDataSet = splitDataSet(dataSet, testProportion, randomSeed)
        theSplitDataSets.append(theSplitDataSet)
    return theSplitDataSets


def kFoldSplitDataSet(dataSet, numberOfFolds=3, randomSeed=None, trainPathPrefix=None, testPathPrefix=None):
    """
    Splits a DataSet's data frame into k test/training sets
    :param dataSet: input DataSet
    :param numberOfFolds: Optional. Integer. Default is 3.
    :param randomSeed: Optional. Integer. If None, test-train split randomness will not be controlled and replicable.
    :return: list of SplitDataSets, each representing one test-train fold.
    """
    originalDataFrame = dataSet.dataFrame
    numberOfObs = len(originalDataFrame)
    kFoldIndices = sklearn.cross_validation.KFold(numberOfObs, numberOfFolds, shuffle=True, random_state=randomSeed)

    # Build k DataSets

    baseNewDescription = dataSet.description
    baseNewPath = os.path.dirname(dataSet.path) + '/' + os.path.basename(dataSet.path).split('.')[0] + '_'
    foldDataSets = []
    currentFold = 0

    for trainIndex, testIndex in kFoldIndices:

        trainDataFrame = originalDataFrame.iloc[trainIndex]
        testDataFrame = originalDataFrame.iloc[testIndex]

        if trainPathPrefix == None:
            trainPath = baseNewPath + str(currentFold) + '_train.csv'
        else:
            trainPath = trainPathPrefix + '_' + str(currentFold) + '_train.csv'
        if testPathPrefix == None:
            testPath = baseNewPath + str(currentFold) + '_test.csv'
        else:
            testPath = testPathPrefix + '_' + str(currentFold) + '_test.csv'

        trainDataSet = mlutilities.types.DataSet(baseNewDescription + ' ' + str(currentFold) + ' Training Set',
                                                 trainPath,
                                                 'w',
                                                 trainDataFrame,
                                                 dataSet.featuresIndex,
                                                 dataSet.labelIndex)
        testDataSet = mlutilities.types.DataSet(baseNewDescription + ' ' + str(currentFold) + ' Testing Set',
                                                testPath,
                                                'w',
                                                testDataFrame,
                                                dataSet.featuresIndex,
                                                dataSet.labelIndex)

        # Join into new SplitDataSet
        foldDataSet = mlutilities.types.SplitDataSet(trainDataSet, testDataSet)
        foldDataSets.append(foldDataSet)

        currentFold += 1

    return foldDataSets
