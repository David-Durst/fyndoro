import skimage
import selectivesearch
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# based on https://github.com/AlpacaDB/selectivesearch/blob/develop/example/example.py

def main(imagePath):
    img = skimage.io.imread(imagePath)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        numpy.asarray(img), scale=500, sigma=0.9, min_size=50)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 50:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w == 0 or h == 0 or w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])