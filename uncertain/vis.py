import pandas as pd
import matplotlib.pyplot as plt
import sys

numTrials = 3
# the paths to the different trials, should be able to append a number ot the
# end of each path to get the folder
pathToTrialsData = sys.argv[1]
outputName = sys.argv[2]
# a list of the run
runTypes = ["_augmented.csv", "_noprob.csv", "_notaugmented.csv"]

allDFs = [[pd.read_csv(pathToTrialsData + str(outputName) + runTypes) for y in runTypes] for x in numTrials]

trialIdx = 0
for dfsForTrial in allDFs:
    trialIdx += 1
    plt.figure()
    for i in range(numTrials):
        if i == 0:
            ax = dfsForTrial[i].plot(x=1, y=2, kind="bar")
        else:
            dfsForTrial[i].plot(x=1, y=2, kind="bar", ax=ax)
    plt.savefig(pathToTrialsData + str(outputName) + "vis_trial" + str(trialIdx))