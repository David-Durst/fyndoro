import pandas as pd
import matplotlib.pyplot as plt
import sys

numTrials = ['']
# the paths to the different trials, should be able to append a number ot the
# end of each path to get the folder
pathToTrialsData = sys.argv[1]
outputName = sys.argv[2]
# a list of the run
runTypes = ["1", "2", "3", "4"]

allDFs = [[pd.read_csv(pathToTrialsData + str(x) + y + ".csv") for y in runTypes] for x in numTrials]

trialIdx = 0
for dfsForTrial in allDFs:
    trialIdx += 1
    plt.figure()
    for i in range(len(runTypes)):
        columns = dfsForTrial[i].columns
        if i == 0:
            ax = dfsForTrial[i].plot(x=columns[1], y=columns[2], linestyle='--', marker='o')
            ax.set_ylabel('Validation Accuracy', fontsize=15)
            ax.set_xlabel('Imagenet Images', fontsize=15)
            plt.title('Male or Female', fontsize=15)
        else:
            dfsForTrial[i].plot(x=columns[1], y=columns[2], ax=ax, linestyle='--', marker='o')
    plt.savefig(pathToTrialsData + "vis_trial" + str(trialIdx))


# acerage the dataframes across all trials
averagedDFs = []
plt.figure()
for i in range(len(runTypes)):
    runType = runTypes[i]
    for j in range(len(numTrials)):
        if j == 0:
            avgCol = allDFs[j][i][allDFs[j][i].columns[2]] / 3
        else:
            avgCol += allDFs[j][i][allDFs[j][i].columns[2]] / 3
    plottingColumns = allDFs[0][i].columns[1:3]
    dfToPlot = pd.concat([allDFs[0][i][allDFs[0][i].columns[1]],avgCol], axis=1, keys=plottingColumns)
    if i == 0:
        ax = dfToPlot.plot(x=plottingColumns[0], y=plottingColumns[1], linestyle='--', marker='o')
        plt.title(outputName + str("avg"))
    else:
        dfToPlot.plot(x=plottingColumns[0], y=plottingColumns[1], ax=ax, linestyle='--', marker='o')
plt.savefig(pathToTrialsData + "vis_avg")
