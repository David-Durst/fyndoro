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

How to run:
You will either need to run the scraper on the same machine as the rest of the pipeline, or have passwordless login to the remote machine. For (almost) everywhere but Stanford, use public/private keys. For Stanford, run `kinit && aklog` to get a 24 hour kerberos ticket and repeat this every 24 hours you need the system to work.
files to look at:
NOTE: two important gotchas with the system currently are: 1. you can only train to detect two classes at a time. 2. There is a third class, random, that is used for everything except the classes you are training for. The name "random" must come after the names of your classes when alphabetically sorted. If it does not, then change the name of random to something that comes later in the alphabet, like "zzzzzz".

uncertain/learn.py - takes in a dataset (one train folder, one val folder), trains a model, and outputs its best validation accuracy
uncertain/imageCleaningAndGoogleSearching/scrape.py - takes in a folder of images, produces a new folder where it google reverse image searches every image in the input folder and scrapes those that match the filters
uncertain/imageCleaningAndGoogleSearching/clean.py - use imagemagick to make sure all images are good for learn.py
uncertain/makeDataRecursiveGoogle.py - take in a large collection of images, randomly split them into validation and training, then split training into groups and call scape.py on each subset of the training data to get different sized amounts of augmented data, split based on amount of augmentation, for running runLearningonRecrusive.py on
uncertain/runLearningonRecursive.py - take in many different folders containing different amounts of data for training and validating, call learn.py on each amount to measure how well data augmentation improves valdiation accuracy as number of base images and number of augmented images increases
uncertain/createDataToLabelMapping.py - take a model output by learn.py and run it on many training and validation images to record the labels