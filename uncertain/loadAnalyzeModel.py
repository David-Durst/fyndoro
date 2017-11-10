import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import os

def loadModelApplicationResults(modelLocation):
    with open(modelLocation, 'r') as f:
        trainAndVal = eval(f.read())
        results = {}
        results["train"] = pd.DataFrame.from_dict(trainAndVal["train"])
        results["val"] = pd.DataFrame.from_dict(trainAndVal["val"])
        return results

def getEmbeddingsNPArr(modelResults, trainOrValKey):
    # need to go DF of lists of floats to list of lists of floats to np array of floats
    # as otherwise to np converter choses lists as its base type
    embeddingsDF = modelResults[trainOrValKey][3].apply(lambda x: np.array(x))
    return np.array(embeddingsDF.tolist())

class valElementMostSimilar(object):
    def __init__(self, valIndex, trainIndex, distance, valFilename, trainFilename):
        self.valIndex = valIndex
        self.trainIndex = trainIndex
        self.distance = distance
        self.valFilename = valFilename
        self.trainFilename = trainFilename

    def to_dict(self):
        return {
            "valIndex": self.valIndex,
            "trainIndex": self.trainIndex,
            "distance": self.distance,
            "valFilename": self.valFilename,
            "trainFilename": self.trainFilename
        }

    def __str__(self):
        return "valIndex: {}, trainIndex: {}, distance: {}, valFilename: {}, trainFilename: {}".format(self.valIndex, self.trainIndex, self.distance, self.valFilename, self.trainFilename)

#import loadAnalyzeModel
#res = loadAnalyzeModel.loadModelApplicationResults("birds2Models_4/dataToLabels_10_2755_1227")
#loadAnalyzeModel.getMultipleExamplesBySimilarity(res, "birds2Models_4/multiple_examples_10_2755_1227")
def getMultipleSimilarTrainPointsForEachVal(modelResults):
    trainEmbeddingsNP = getEmbeddingsNPArr(modelResults, "train")
    valEmbeddingsNP = getEmbeddingsNPArr(modelResults, "val")
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(np.array(trainEmbeddingsNP))
    distances, nearestTrainIndicesForEachValNestedArr = nbrs.kneighbors(valEmbeddingsNP)
    mostSimilarClassesList = [valElementMostSimilar(valIndex, trainIndices.tolist(), distances[valIndex].tolist(),
                                                    modelResults["val"][0][valIndex],
                                                    modelResults["train"][0][trainIndices].tolist()) for
                              valIndex, trainIndices in enumerate(nearestTrainIndicesForEachValNestedArr)]
    return pd.DataFrame.from_dict([mostSimilarEl.to_dict() for mostSimilarEl in mostSimilarClassesList])


def getMultipleExamplesBySimilarity(modelResults, outputLocation):
    increments = 0.1
    os.system("mkdir " + outputLocation)
    splitPoints = np.arange(0.0, 1.5, increments)
    mostSimilarDF = getMultipleSimilarTrainPointsForEachVal(modelResults).sort_values('distance')
    for pnt in splitPoints:
        # a series containg the first, smallest value of the distance column for each row in mostSimilarDF
        firstDistanceValueSeries = mostSimilarDF['distance'].apply(lambda xs: xs[0])
        mostSimilarAtPntDF = mostSimilarDF[(firstDistanceValueSeries >= pnt) & (firstDistanceValueSeries < pnt + increments)].head(5)
        pntFolderPath = outputLocation + "/" + str(pnt)
        os.system("mkdir " + pntFolderPath)
        # i iterates over the validation points, each validation has multiple training points
        # associated with it. (number of val points is number of neighbors above) j iterates
        # over training points
        for i in range(len(mostSimilarAtPntDF)):
            ithSimilarAtPnt = mostSimilarAtPntDF.iloc[i]
            valPointOutputPath = pntFolderPath + "/" + str(i) + "_" + str(ithSimilarAtPnt['distance'][0])
            os.system("mkdir " + valPointOutputPath)
            os.system("cp \"" + ithSimilarAtPnt['valFilename'] + "\" " + valPointOutputPath + "/")
            for j in range(len(ithSimilarAtPnt['distance'])):
                trainPointOutputPath = valPointOutputPath + "/" + str(j) + "_" + str(ithSimilarAtPnt['distance'][j])
                os.system("mkdir " + trainPointOutputPath)
                os.system("cp \"" + ithSimilarAtPnt['trainFilename'][j] + "\" " + trainPointOutputPath + "/")

def getMostSimilarTrainPointForEachVal(modelResults):
    trainEmbeddingsNP = getEmbeddingsNPArr(modelResults, "train")
    valEmbeddingsNP = getEmbeddingsNPArr(modelResults, "val")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(trainEmbeddingsNP))
    distances, nearestTrainIndexForEachValNestedArr = nbrs.kneighbors(valEmbeddingsNP)
    # normally, each element is an array (as may have more than 1 nearest neighbor)
    # remove the array per val element as only 1 nearest neighbor in train
    nearestTrainIndexForEachVal = np.apply_along_axis(lambda x: x[0], 1, nearestTrainIndexForEachValNestedArr)
    mostSimilarClassesList = [valElementMostSimilar(valIndex, trainIndex, distances[valIndex][0], modelResults["val"][0][valIndex], modelResults["train"][0][trainIndex]) for valIndex, trainIndex in enumerate(nearestTrainIndexForEachVal)]
    return pd.DataFrame.from_dict([mostSimilarEl.to_dict() for mostSimilarEl in mostSimilarClassesList])

def getMostSimilarDistanceDistibution(modelResults):
    mostSimilarDF = getMostSimilarTrainPointForEachVal(modelResults)
    plt.close()
    mostSimilarDF.hist("distance")
    plt.savefig('mostSimilarDistances.pdf')

def getExamplesBySimilarity(modelResults, outputLocation):
    increments = 0.01
    os.system("mkdir " + outputLocation)
    splitPoints = np.arange(1.0, 1.08, increments)
    mostSimilarDF = getMostSimilarTrainPointForEachVal(modelResults).sort_values('distance')
    for pnt in splitPoints:
        mostSimilarAtPntDF = mostSimilarDF[(mostSimilarDF['distance'] >= pnt) & (mostSimilarDF['distance'] < pnt + increments)].head(5)
        pntFolderPath = outputLocation + "/" + str(pnt)
        os.system("mkdir " + pntFolderPath)
        for i in range(len(mostSimilarAtPntDF)):
            ithSimilarAtPnt = mostSimilarAtPntDF.iloc[i]
            ithOutputPath = pntFolderPath + "/" + str(i) + "_" + str(ithSimilarAtPnt['distance'])
            os.system("mkdir " + ithOutputPath)
            os.system("cp \"" + ithSimilarAtPnt['trainFilename'] + "\" " + ithOutputPath + "/")
            os.system("cp \"" + ithSimilarAtPnt['valFilename'] + "\" " + ithOutputPath + "/")

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

        destDirRightCat1 = " mOrFRecNoFreezeModels/certainPics/" + type + "/1.0,0.0/right/"
        os.system("mkdir -p " + destDirRightCat1)
        certainRightCat1 = certain[(dset[1] > 0.9) & (dset[0].str.contains("1.0,0.0"))][0].tolist()[:100]
        for fname in certainRightCat1:
            os.system(
                "cp " + fname + destDirRightCat1 + os.path.basename(fname))

        destDirRightCat2 = " mOrFRecNoFreezeModels/certainPics/" + type + "/0.0,1.0/right/"
        os.system("mkdir -p " + destDirRightCat2)
        certainRightCat2 = certain[(dset[2] > 0.9) & (dset[0].str.contains("0.0,1.0"))][0].tolist()[:100]
        for fname in certainRightCat2:
            os.system(
                "cp " + fname + destDirRightCat2 + os.path.basename(fname))

        destDirWrongCat1 = " mOrFRecNoFreezeModels/certainPics/" + type + "/1.0,0.0/wrong/"
        os.system("mkdir -p " + destDirWrongCat1)
        certainWrongCat1 = certain[(dset[1] < 0.1) & (dset[0].str.contains("1.0,0.0"))][0].tolist()[:100]
        for fname in certainWrongCat1:
            os.system(
                "cp " + fname + destDirWrongCat1 + os.path.basename(fname))

        destDirWrongCat2 = " mOrFRecNoFreezeModels/certainPics/" + type + "/0.0,1.0/wrong/"
        os.system("mkdir -p " + destDirWrongCat2)
        certainWrongCat2 = certain[(dset[2] < 0.1) & (dset[0].str.contains("0.0,1.0"))][0].tolist()[:100]
        for fname in certainWrongCat2:
            os.system(
                "cp " + fname + destDirWrongCat2 + os.path.basename(fname))
