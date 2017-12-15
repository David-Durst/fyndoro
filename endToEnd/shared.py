import os

modulePath = os.path.dirname(os.path.realpath(__file__))

def getImageFolderPath(taskName):
    return "%s/modelsAndData/%s_data" % (modulePath, taskName)

def getInferenceImagesFolderPath(taskName):
    return "%s/modelsAndData/%s_inference" % (modulePath, taskName)

def getModelWeightsPath(taskName):
    return "%s/modelsAndData/%s_model" % (modulePath, taskName)

def getIndexToClassMapPath(taskName):
    return "%s/modelsAndData/%s_data/indexToClassMap" % (modulePath, taskName)