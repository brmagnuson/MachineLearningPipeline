import time
import thesisFunctions

baseDirectoryPath = 'SacramentoModel/'
myFeaturesIndex = 6
myLabelIndex = 5
kFolds = 5
modelApproach = 'sacramento'
randomSeed = 47392

startSecond = time.time()
startTime = time.strftime('%a, %d %b %Y %X')


# Run the flow model pipeline for five folds
thesisFunctions.runKFoldPipeline(baseDirectoryPath,
                                 myFeaturesIndex,
                                 myLabelIndex,
                                 kFolds,
                                 modelApproach,
                                 randomSeed=randomSeed)


endSecond = time.time()
endTime = time.strftime('%a, %d %b %Y %X')
totalSeconds = endSecond - startSecond

print()
print('Start time:', startTime)
print('End time:', endTime)
print('Total: {} minutes and {} seconds'.format(int(totalSeconds // 60), round(totalSeconds % 60)))