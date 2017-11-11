from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import os
import sys

cutoff = 0.0012

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
    def __init__(self, trainIndex, valIndex, distance, trainFilename, valFilename):
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

def getMostSimilarValPointForEachTrain(modelResults):
    trainEmbeddingsNP = getEmbeddingsNPArr(modelResults, "train")
    valEmbeddingsNP = getEmbeddingsNPArr(modelResults, "val")
    nbrs = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm='auto').fit(np.array(valEmbeddingsNP))
    distances, nearestValIndexForEachTrainNestedArr = nbrs.kneighbors(trainEmbeddingsNP)
    # normally, each element is an array (as may have more than 1 nearest neighbor)
    # remove the array per val element as only 1 nearest neighbor in train
    nearestValIndexForEachTrain = np.apply_along_axis(lambda x: x[0], 1, nearestValIndexForEachTrainNestedArr)
    mostSimilarClassesList = [valElementMostSimilar(trainIndex, valIndex, distances[valIndex][0], modelResults["train"][0][trainIndex], modelResults["val"][0][valIndex]) for trainIndex, valIndex in enumerate(nearestValIndexForEachTrain)]
    return pd.DataFrame.from_dict([mostSimilarEl.to_dict() for mostSimilarEl in mostSimilarClassesList])

# remove all training points which have a cosine distance of less than
# the cutoff to the closest validation point
def removeTooSimiliar(modelResults):
    similarityDF = getMostSimilarValPointForEachTrain(modelResults)
    tooSimilarDF = similarityDF[similarityDF["distance"] < cutoff]
    for i in range(len(tooSimilarDF)):
        os.system("rm \"" + tooSimilarDF.iloc[i]['trainFilename'] + "\"")

data_embeddings = sys.argv[1]
model_results = loadModelApplicationResults(data_embeddings)
removeTooSimiliar(model_results)

