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


def certainRightAndWrong(valAndTrain):
    for type in ['train', 'val']:
        dset = valAndTrain[type]
        certain = dset[(dset[1] > 0.90) | (dset[1] < 0.1)]

        destDirRightCat1 = " mOrFRecNoFreezeModels/certainPics/right/1.0,0.0/" + type + "/"
        os.system("mkdir -p " + destDirRightCat1)
        certainRightCat1 = certain[(dset[1] > 0.9) & (dset[0].str.contains("1.0,0.0"))][0].tolist()[:100]
        for fname in certainRightCat1:
            os.system(
                "cp " + fname + destDirRightCat1 + os.path.basename(fname))

        destDirRightCat2 = " mOrFRecNoFreezeModels/certainPics/right/0.0,1.0/" + type + "/"
        os.system("mkdir -p " + destDirRightCat2)
        certainRightCat2 = certain[(dset[2] > 0.9) & (dset[0].str.contains("0.0,1.0"))][0].tolist()[:100]
        for fname in certainRightCat2:
            os.system(
                "cp " + fname + destDirRightCat2 + os.path.basename(fname))

        destDirWrongCat1 = " mOrFRecNoFreezeModels/certainPics/wrong/1.0,0.0/" + type + "/"
        os.system("mkdir -p " + destDirWrongCat1)
        certainWrongCat1 = certain[(dset[1] < 0.1) & (dset[0].str.contains("1.0,0.0"))][0].tolist()[:100]
        for fname in certainWrongCat1:
            os.system(
                "cp " + fname + destDirWrongCat1 + os.path.basename(fname))

        destDirWrongCat2 = " mOrFRecNoFreezeModels/certainPics/wrong/0.0,1.0/" + type + "/"
        os.system("mkdir -p " + destDirWrongCat2)
        certainWrongCat2 = certain[(dset[2] < 0.1) & (dset[0].str.contains("0.0,1.0"))][0].tolist()[:100]
        for fname in certainWrongCat2:
            os.system(
                "cp " + fname + destDirWrongCat2 + os.path.basename(fname))
