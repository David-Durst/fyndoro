import os
from os import listdir
from os.path import isfile, join, splitext
import sys
def filterImages(dir):
    anyFiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    for f in anyFiles:
        _, fExt = splitext(f)

        relPathAndName = join(dir, f)

        # delete non images (and duplicates, which have a .1 or .n at end, where n is an integer)
        if fExt != ".png" and fExt != ".jpg" and fExt != ".JPEG" and fExt != ".jpeg":
            os.remove(relPathAndName)
        # otherwise, reshape them
        else:
            retVal = os.system("convert -resize 500x500 \'" + relPathAndName + "\' \'" + relPathAndName + "\'")
            if retVal != 0:
                os.remove(relPathAndName)

if __name__ == "__main__":
    # input to script is the directory of images to clean up
    filterImages(sys.argv[1])