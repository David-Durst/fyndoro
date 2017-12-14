from subprocess import Popen, PIPE
import argparse
import os
import importlib
import time
from endToEnd.filterImages import filterBasedOnEmbeddingSimilarity, getEmbeddingsForData
from endToEnd.constants import *

parser = argparse.ArgumentParser(description='Download the data for a model and train it.')

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--categories', metavar="category_name", nargs='+',
                   help='the name of a category that the classifier will learn, should have no spaces')
requiredNamed.add_argument('--searchwords', metavar='"category n search terms"', nargs='+',
                   help='a phrase to search google for a category (order of searchwords args should match order of '
                        'categories)')
requiredNamed.add_argument('--keywordFilters', metavar='"category n keyword filters"', nargs='+', required=True,
                   help='filter out google results for a category that don\'t have these words '
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces) (order of keywordFilters args should match order of categories)')
requiredNamed.add_argument('--wrongwordFilters', metavar='"category n wrongword filters"', nargs='+',
                   help='filter out google results for a category that have these words'
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces) (order of keywordFilters args should match order of categories)')


optionalNamed = parser.add_argument_group('optional arguments')
optionalNamed.add_argument('--nameOfOutput', default=None,
                    help='The name for the data and model to be save under in the modelsAndData folder. Model will be '
                         'under name <name>_model.pth. If unset, the categories will be used with spaces removed.')
optionalNamed.add_argument('--scrapeOnRemote', default=False, action='store_true',
                    help='enable this to run the scraper on a different host from the rest of the system')
optionalNamed.add_argument('--scrapingUserHost', metavar="user@host", default='durst@dawn4',
                    help='the remote user@host to run the scraping on.')
optionalNamed.add_argument('--remoteDir', metavar="remote/path/for/fyndoro", default='fyndoro/',
                    help='the fyndoro folder on the remote host for the scraping code and data.')
optionalNamed.add_argument('--numIterations', metavar="N", type=int, default=2,
                   help='how many iterations of augmentation by google searching should be done. First iteration is'
                        'text searching, all others are reverse google image search based.')


args = parser.parse_args()
# python makeTrainedModel.py --categories cat0 cat1 cat2 --searchwords "0hat 0dog omaha0" "1blue 1red 1how" "2in 2out" --keywordFilters "0a 0b" "1c 1d" "2 q" --wrongwordFilters "0no 0" "1no 1 11 1 12 23 1n noasd a" "2 3123123asd"
print("Input args:" + str(args))

shell = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)

def executeShellCommand(commandStr, shell=shell, returnResult=False):
    print("executing command: " + commandStr)
    shell.stdin.write(str.encode(commandStr + '\n'))
    shell.stdin.flush()
    if returnResult:
        return shell.stdout.readline()

def convertListOfStringsToBashArray(inputList, wrapString = "\""):
    return " ".join(list(map(lambda s: wrapString + s + wrapString, inputList)))

#def convertListOfStringsToBashAssociativeArray(keys, values):
#    zipped = zip(keys, values)
#    return " ".join(list(map(lambda kv: '["%s"]="%s"' % (kv[0], kv[1]), zipped)))



# install SPN.pytorch if its not already installed
if importlib.util.find_spec("spn") is None:
    executeShellCommand("cd %s/SPN.pytorch/spnlib" % modulePath)
    executeShellCommand("bash make.sh")
#import after sure installed
import endToEnd.transferLearn as transferLearn

# if model doesn't already exist, make it
executeShellCommand("cd %s/SPN.pytorch/demo" % modulePath)
pathToModelPreTransferLearning = modulePath + "/SPN.pytorch/demo/logs/voc2007/model_best.pth.tar"
if not os.path.isfile(pathToModelPreTransferLearning):
    executeShellCommand("bash runme.sh")
executeShellCommand("cd %s" % modulePath)

# get strings for bash for getting data
categoryGroupsBashStr = "categoryGroups=(%s)" % (convertListOfStringsToBashArray(args.categories, wrapString=""))

outputName = convertListOfStringsToBashArray(args.categories, wrapString="").replace(" ", "") if args.nameOfOutput is None else args.nameOfOutput
outputNameWithDatetime = outputName + "-" + time.strftime("%Y%m%d-%H%M%S")

searchwordsBashStr = "export searchwords=(%s)" % (convertListOfStringsToBashArray(args.categories))
keywordFiltersBashStr = "export keywordFilters=(%s)" % (convertListOfStringsToBashArray(args.categories))
wrongwordFiltersBashStr = "export wrongwordFilters=(%s)" % (convertListOfStringsToBashArray(args.categories))
declareAllVars = "export %s ; export numIterations=%s ; export uniqueNum=$RANDOM ; %s ; %s ; %s ; " % \
                 (categoryGroupsBashStr, args.numIterations, searchwordsBashStr, keywordFiltersBashStr, wrongwordFiltersBashStr)


# get data for transfer learning model
if args.scrapeOnRemote:
    # https://stackoverflow.com/questions/12845206/check-if-file-exists-on-remote-host-with-ssh
    scrapingExistsStr = executeShellCommand(
        """ssh -q %s [[ -f %s/.gitignore ]] && echo "File exists" || echo "File does not exist";""" % (
        args.scrapingHost, args.remoteDir))

    if scrapingExistsStr == "File does not exist\n":
        print("Please install fyndoro on the remote host %s at location %s if you want to run scraping there." % (args.scrapingHost, args.remoteDir))
        exit(1)
    else:
        executeShellCommand(
            "ssh -q %s '%s %s/endToEnd/createTransferLearningDataSet.sh %s/endToEnd/modelsAndData/%s_data &> %s/endToEnd/modelsAndData/%s_data.log'" % (
            args.scrapingHost, declareAllVars, args.remoteDir, args.remoteDir, outputName, outputNameWithDatetime))
        executeShellCommand("rsync -a %s:%s/endToEnd/modelsAndData/%s_data %s/modelsAndData/" %
                            (args.scrapingHost, args.remoteDir, outputName, modulePath))
else:
    executeShellCommand("%s %s/createTransferLearningDataSet.sh %s/modelsAndData/%s_data &> %s/modelsAndData/%s_data.log" % (
        declareAllVars, modulePath, modulePath, outputName, modulePath, outputNameWithDatetime))

pathToImages = "%s/modelsAndData/%s_data/downloadedImages/upto_%s" % (modulePath, outputName, args.numIterations)
pathToEmbeddings = "%s/modelsAndData/%s_data/embeddings" % (modulePath, outputName)
# clean up data for model
getEmbeddingsForData.generateEmbeddings(pathToImages, "", pathToEmbeddings)
filterBasedOnEmbeddingSimilarity.generateEmebeddingsAndFilterTooSimilar(pathToEmbeddings)
# python uncertain/getEmbeddingsForData.py $uptoiDir/ "" $uptoiDir/embeddings
#        python uncertain/imageCleaningAndGoogleSearching/filterBasedOnEmbeddingSimilarity.py $uptoiDir/embeddings

#transfer learn model
pathToModelPostTransferLearning = "%s/modelsAndData/%s_model" % (modulePath, outputName)
transferLearn.transferLearn(pathToImages, pathToModelPreTransferLearning, "%s/modelsAndData/%s_data/indexToClassMap" % (modulePath, outputName),
                            pathToModelPostTransferLearning)
#executeShellCommand("%s/transferLearnModel.sh %s/SPN.pytorch/demo/logs/voc2007/model_best.pth.tar "
#                    "%s/modelsAndData/%s_data %s/modelsAndData/%s_model &> %s/modelsAndData/%s_model" %
#                    (modulePath, modulePath, modulePath, categoryGroupsFileName, modulePath, categoryGroupsFileName,
#                     modulePath, categoryGroupsFileName))

#"../objectDetection/SPN.pytorch/demo/logs/voc2007/model_best.pth.tar"

#categoryGroups=("scarletTanager" "summerTanager")
#    declare -A searchwords=( [${categoryGroups[0]}]="scarlet tanager" [${categoryGroups[1]}]="summer tanager")
#    declare -A keywordFilters=( [${categoryGroups[0]}]="scarlet" [${categoryGroups[1]}]="summer")
#    # wrongwords are words that indicate an image is bad
#    declare -A wrongwordFilters=( [${categoryGroups[0]}]="summer" [${categoryGroups[1]}]="scarlet")
# BASH_ENV=<(declare -pA numb=(["hat"]=1 ["dog"]=2)) ./printNumb.sh


#executeShellCommand()