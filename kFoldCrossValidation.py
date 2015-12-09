import thesisFunctions

scoreModelResultsDF = thesisFunctions.flowModelPipeline(universalTestSetFileName='jul_IntMnt_test.csv',
                                                        universalTestSetDescription='Jul IntMnt Test',
                                                        basePath='Data/',
                                                        picklePath='Pickles/',
                                                        statusPrintPrefix='K-fold #1',
                                                        randomSeed=47392)
scoreModelResultsDF.to_csv('Output/cvScoreModelResults.csv', index=False)