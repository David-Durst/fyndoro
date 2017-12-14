import os

modulePath = os.path.dirname(os.path.realpath(__file__))

def getImageFolderPath(runName):
    return "%s/endToEnd/modelsAndData/%s_data" % (modulePath, runName)

def getModelWeightsPath(runName):
    return "%s/modelsAndData/%s_model" % (modulePath, modelName)
