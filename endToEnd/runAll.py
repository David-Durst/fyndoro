from endToEnd import makeTrainedModel, runObjDetectInference

# download the imagenet category list
    classes = {int(key):value for (key, value)
              in json.load(open(label_map)).items()}