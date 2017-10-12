import pandas as pd
import matplotlib.pyplot as plt
import sys

numTrials = [1, 2, 3]
# the paths to the different trials, should be able to append a number ot the
# end of each path to get the folder
pathToTrialsData = sys.argv[1]
outputName = sys.argv[2]
# a list of the run
runTypes = ["_augmented.csv", "_noprob.csv", "_notaugmented.csv"]

allDFs = [[pd.read_csv(pathToTrialsData + str(x) + y) for y in runTypes] for x in numTrials]

trialIdx = 0
for dfsForTrial in allDFs:
    trialIdx += 1
    plt.figure()
    for i in range(len(numTrials)):
        columns = dfsForTrial[i].columns
        if i == 0:
            ax = dfsForTrial[i].plot(x=columns[1], y=columns[2])
            plt.title(outputName + " trial " + str(trialIdx))
        else:
            dfsForTrial[i].plot(x=columns[1], y=columns[2], ax=ax)
    plt.savefig(pathToTrialsData + str(outputName) + "vis_trial" + str(trialIdx))