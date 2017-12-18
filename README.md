* How To Install

1. Install Conda - https://www.anaconda.com/download/
2. Checkout this repository - git clone --recursive git@github.com:David-Durst/fyndoro.git
3. Create the environment - conda env create -f environment.yml (or mac_env.yml if using a mac)
3. Install imagemagick
3. If you will be running scraping on a remote server, ensure that you have passwordless ssh login to that server from the server you will be running the main script on. and install this repository on that server. It is recommended that you install fyndoro in the root folder.
3. Run the end-to-end process of initial model training, data collection for a specific task, transfer learning the model for that task, and testing with inference on instagram data. Do this by running endToEnd/runAll.py with the following command:

python -m endToEnd.runAll --taskName testBags --categories handbag louisvuittonbag --searchwords "louis vuitton handbag" "handbag" --keywordFilters "louis vuitton" "handbag" --wrongwordFilters "coach fostello" "louis vuitton" --categoryThreshold 0.8

How to run:
You will either need to run the scraper on the same machine as the rest of the pipeline, or have passwordless login to the remote machine. For (almost) everywhere but Stanford, use public/private keys. For Stanford, run `kinit && aklog` to get a 24 hour kerberos ticket and repeat this every 24 hours you need the system to work.
files to look at:
NOTE: two important gotchas with the system currently are: 1. you can only train to detect two classes at a time. 2. There is a third class, random, that is used for everything except the classes you are training for. The name "random" must come after the names of your classes when alphabetically sorted. If it does not, then change the name of random to something that comes later in the alphabet, like "zzzzzz".
NOTE: on line 9 of endToEnd/createTransferLearningDataSet.sh, there is a variable called randomImages, this must be set to a collection of ~2100 images that aren't in the two classes you are trying to detect. I created this data by downloading 300 images from each of the 7 top level synsets in ImageNet that have images. I can't post this folder due to copyright concerns.
uncertain/learn.py - takes in a dataset (one train folder, one val folder), trains a model, and outputs its best validation accuracy
uncertain/imageCleaningAndGoogleSearching/scrape.py - takes in a folder of images, produces a new folder where it google reverse image searches every image in the input folder and scrapes those that match the filters
uncertain/imageCleaningAndGoogleSearching/clean.py - use imagemagick to make sure all images are good for learn.py
uncertain/makeDataRecursiveGoogle.py - take in a large collection of images, randomly split them into validation and training, then split training into groups and call scape.py on each subset of the training data to get different sized amounts of augmented data, split based on amount of augmentation, for running runLearningonRecrusive.py on
uncertain/runLearningonRecursive.py - take in many different folders containing different amounts of data for training and validating, call learn.py on each amount to measure how well data augmentation improves valdiation accuracy as number of base images and number of augmented images increases
uncertain/createDataToLabelMapping.py - take a model output by learn.py and run it on many training and validation images to record the labels