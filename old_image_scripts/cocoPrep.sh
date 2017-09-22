#make all necessary directories
mkdir -p coco/2017train
mkdir -p coco/1000val
mkdir -p coco/200train
#get all images
wget --directory-prefix coco/ http://images.cocodataset.org/zips/val2017.zip
unzip coco/ -d mscoco/2014val/

numImagesTotal=$(ls mscoco/2014val/val2014/ -1 | wc -l)
#randomly select 1000 images to test on, 200 to train on
shuf -zn1000 -e *.jpg | xargs -0 cp -vt ta/

