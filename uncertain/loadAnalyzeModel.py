import pandas as pd
import os

def loadModel(modelLocation):
    with open(modelLocation, 'r') as f:
        trainAndVal = eval(f.read())
        results = {}
        results["train"] = pd.DataFrame.from_dict(trainAndVal["train"])
        results["val"] = pd.DataFrame.from_dict(trainAndVal["val"])
        return results


def uncertain(valAndTrain):
    val = valAndTrain["val"]
    valUncertain = val[(val[2] > 0.45) & (val[2] < 0.55)][0].tolist()
    for fName in valUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/uncertainPics/val/" + os.path.basename(fName))

    train = valAndTrain["train"]
    trainUncertain = train[(train[2] > 0.45) & (train[2] < 0.55)][0].tolist()
    for fName in trainUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/uncertainPics/train/" + os.path.basename(fName))


def certainRight(valAndTrain):
    val = valAndTrain["val"]
    valUncertain = val[val[2] > 0.90][0].tolist()[:100]
    for fName in valUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/certainPics/right/val/" + os.path.basename(fName))

    train = valAndTrain["train"]
    trainUncertain = train[train[1] > 0.90][0].tolist()[:100]
    for fName in trainUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/certainPics/right/train/" + os.path.basename(fName))

def certainWrong(valAndTrain):
    val = valAndTrain["val"]
    valUncertain = val[val[1] > 0.90][0].tolist()[:100]
    for fName in valUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/certainPics/wrong/val/" + os.path.basename(fName))

    train = valAndTrain["train"]
    trainUncertain = train[train[2] > 0.90][0].tolist()[:100]
    for fName in trainUncertain:
        os.system("cp " + fName + " mOrFRecNoFreezeModels/certainPics/wrong/train/" + os.path.basename(fName))
