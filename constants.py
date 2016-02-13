"""
This script contains references I want to use in many functions.
"""

ridgeRegression = 'Ridge Regression'
randomForest = 'Random Forest'
kNeighbors = 'K Nearest Neighbors'
supportVectorMachine = 'Support Vector Machine'
decisionTree = 'Decision Tree'
adaBoost = 'Ada Boost'

randomSeed = 47392

# For my computer, use n_jobs=1 for the tune model step because anything else seems to break the apply model step
# whenever runFeatureEngineering is set to True. This is completely illogical but the truth. For AWS, use n_jobs=-1
# to set n_jobs equal to the number of cores.
n_jobs = -1
