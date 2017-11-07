* How To Install

1. Install Conda - https://www.anaconda.com/download/
2. Checkout this repository - git clone git@github.com:David-Durst/fyndoro.git
3. Create the environment - conda env create -f environment.yml (or mac_env.yml if using a mac)
3. Install gcloud terminal tools - https://cloud.google.com/sdk/downloads
3. Get access to the Google Vision API - https://cloud.google.com/vision/docs/before-you-begin
3. Provide gcloud access to the application - gcloud auth application-default login --no-launch-browser
    a. documentation for command - https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
3. create a google API key and put in in the file API_KEY in the root of this project - https://support.google.com/cloud/answer/6158862?hl=en
3. Install imagemagick - brew install imagemagick for mac
3. Install gshuf (or shuf on linux) - brew install coreutils for mac
4. Run the learner - python -m uncertain.learn


files to look at:

uncertain/learn.py - takes in a dataset (one train folder, one val folder), trains a model, and outputs its best validation accuracy
uncertain/imageCleaningAndGoogleSearching/scrape.py - takes in a folder of images, produces a new folder where it google reverse image searches every image in the input folder and scrapes those that match the filters
uncertain/imageCleaningAndGoogleSearching/clean.py - use imagemagick to make sure all images are good for learn.py
uncertain/makeDataRecursiveGoogle.py - take in a large collection of images, randomly split them into validation and training, then split training into groups and call scape.py on each subset of the training data to get different sized amounts of augmented data, split based on amount of augmentation, for running runLearningonRecrusive.py on
uncertain/runLearningonRecursive.py - take in many different folders containing different amounts of data for training and validating, call learn.py on each amount to measure how well data augmentation improves valdiation accuracy as number of base images and number of augmented images increases
uncertain/createDataToLabelMapping.py - take a model output by learn.py and run it on many training and validation images to record the labels