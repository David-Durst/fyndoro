from subprocess import Popen, PIPE
import argparse
import os
import importlib

scriptPath = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Download the data for a model and train it.')
parser.add_argument('scrapeOnRemote', default=False, action='store_true',
                    help='enable this to run the scraper on a different host from the rest of the system')
parser.add_argument('scrapingUserHost', default='durst@dawn4',
                    help='the remote user@host to run the scraping on.')
parser.add_argument('remoteDir', default='fyndoro/',
                    help='the fyndoro folder on the remote host for the scraping code and data.')
parser.add_argument('categories', nargs='+',
                   help='the name of a category that the classifier will learn')
parser.add_argument('searchwords', nargs='+',
                   help='a phrase to search google for a category (order of searchwords args should match order of '
                        'categories)')
parser.add_argument('keywordFilters', nargs='+',
                   help='filter out google results for a category that don\'t have these words '
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces)'
                        '(order of keywordFilters args should match order of categories)')
parser.add_argument('wrongwordFilters', nargs='+',
                   help='filter out google results for a category that have these words'
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces)'
                        '(order of keywordFilters args should match order of categories)')
parser.add_argument('numIterations', type=int, default=2,
                   help='how many iterations of augmentation by google searching should be done. First iteration is'
                        'text searching, all others are reverse google image search based.')




args = parser.parse_args()

print(args)
exit(0)

shell = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)

def executeShellCommand(commandStr, shell=shell, returnResult=False):
    shell.stdin.write(str.encode(commandStr + '\n'))
    shell.stdin.flush()
    if returnResult:
        return shell.stdout.readline().decode()

# install SPN.pytorch if its not already installed
if importlib.util.find_spec("spn") is None:
    executeShellCommand("cd %s/SPN.pytorch/spnlib")
    executeShellCommand("bash make.sh")

# if model doesn't already exist, make it
executeShellCommand("cd ../demo")
if not os.path.isfile("logs/voc2007/model_best.pth.tar"):
    executeShellCommand("bash runme.sh")
executeShellCommand("cd ../..")

if args.scrapeOnRemote:
    # https://stackoverflow.com/questions/12845206/check-if-file-exists-on-remote-host-with-ssh
    scrapingExistsStr = executeShellCommand(
        """ssh -q %s [[ -f %s/.gitignore ]] && echo "File exists" || echo "File does not exist";""" % (
        args.scrapingHost, args.remoteDir))

    if scrapingExistsStr == "File does not exist\n":
        print("Please install fyndoro on the remote host %s at location %s if you want to run scraping there." % (args.scrapingHost, args.remoteDir))
        exit(1)
    else:
        # get bash export command to define searchwords


        scrapingExistsStr = executeShellCommand(
            """ssh -q %s %s/endToEnd/createTransferLearningDataSet.sh %s/endToEnd/endToEnd/modelsAndData/""" % (
            args.scrapingHost, args.remoteDir))
"../objectDetection/SPN.pytorch/demo/logs/voc2007/model_best.pth.tar"




executeShellCommand()