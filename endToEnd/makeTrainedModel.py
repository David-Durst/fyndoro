from subprocess import Popen, PIPE
import argparse

parser = argparse.ArgumentParser(description='Download the data for a model and train it.')
parser.add_argument('categories', metavar='C', nargs='+',
                   help='the name of a category that the classifier will learn')
parser.add_argument('searchwords', metavar='S', nargs='+',
                   help='a phrase to search google for a category (order of searchwords args should match order of '
                        'categories)')
parser.add_argument('keywordFilters', metavar='C', nargs='+',
                   help='filter out google results for a category that don\'t have these words '
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces)'
                        '(order of keywordFilters args should match order of categories)')
parser.add_argument('wrongwordFilters', metavar='C', nargs='+',
                   help='filter out google results for a category that have these words'
                        '(a list of words for each category, denote each list with quotes, separate words in a list '
                        'with spaces)'
                        '(order of keywordFilters args should match order of categories)')
parser.add_argument('numIterations', metavar='I', type=int, default=2,
                   help='how many iterations of augmentation by google searching should be done. First iteration is'
                        'text searching, all others are reverse google image search based.')




args = parser.parse_args()

shell = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)

def executeShellCommand(commandStr, shell=shell):
    shell.stdin.write(str.encode(commandStr + '\n'))
    shell.stdin.flush()
    return (shell.stdout.readline())

# get the code for training the base model
executeShellCommand("git clone git@github.com:yeezhu/SPN.pytorch.git")

executeShellCommand()