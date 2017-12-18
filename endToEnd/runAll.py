from endToEnd import makeTrainedModel
from endToEnd.shared import *
import json
import argparse
import os

def runAll(taskName, categoryThreshold, categories, searchwords, keywordFilters, wrongwordFilters, scrapeOnRemote=False,
           scrapingUserHost='durst@dawn4', remoteDir='fyndoro/', numIterations=2):
    makeTrainedModel.makeTrainedModel(taskName, categories, searchwords, keywordFilters, wrongwordFilters, scrapeOnRemote,
           scrapingUserHost, remoteDir, numIterations)

    modelFile = getModelWeightsPath(taskName)

    classes = {int(key): value for (key, value) in json.load(open(getIndexToClassMapPath(taskName))).items()}
    # NEED DATA_DIR FOR INFERENCE
    inferenceImagesPath = getInferenceImagesFolderPath(taskName)
    os.system("mkdir -p %s" % inferenceImagesPath)
    for c in categories[:2]:
        os.system("instagram-scraper %s --tag -d %s/%s --maximum 2000" % (taskName, inferenceImagesPath, taskName))

    # do this here only after makeTrainedModel ensures spn is installed
    from endToEnd import runObjDetectInference
    runObjDetectInference.runObjDetection(inferenceImagesPath, modelFile, classes[0], classes[1], classes[2], categoryThreshold)

def setupArgParserForRunAll(parser):
    requiredNamed = parser.add_argument_group('other required arguments')
    requiredNamed.add_argument('--categoryThreshold', required=True,
                               help='how confident the model should be for it to determine that an image is of a given class. Set this between 0 and 1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download the data for a model, train it, and try it on Instagram data')
    # don't need setup for ObjDetectInference
    setupArgParserForRunAll(parser)
    makeTrainedModel.setupArgParserForMakingTrainedModel(parser)
    args = parser.parse_args()
    runAll(args.taskName, args.categoryThreshold, args.categories, args.searchwords, args.keywordFilters, args.wrongwordFilters, args.scrapeOnRemote,
                     args.scrapingUserHost, args.remoteDir, int(args.numIterations))
