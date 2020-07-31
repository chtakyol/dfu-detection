import os
import cv2
import pathlib
import pandas
import numpy as np
import matplotlib.pyplot as plt

raw_images_path = "DFUC2020"
groundtruth = "DFUC2020/groundtruth.csv"


if not os.path.exists("gt_images"):
    os.mkdir("gt_images")

data_dir = pathlib.Path(raw_images_path)
image_paths = list(data_dir.glob('*/*.jpg'))

gt = pandas.read_csv(groundtruth)
gt = gt.set_index(["name"])

for(i, image_path) in enumerate(image_paths):
    print("[INFO] processing image {}/{}...".format(i + 1,len(image_paths)))
    filename = image_path.parts[-1]	# selecting filename
    selected_gt = gt.loc[filename]	# filtering groundtruth trough filename
    gt_boxes = []	# groundtruth box pos

    print(sum(gt.index == filename))
    if sum(gt.index == filename) > 1:
        for index, row in selected_gt.iterrows():
            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]
            gt_boxes.append((xmin, ymin, xmax, ymax))
    else:
        xmin = selected_gt["xmin"]
        ymin = selected_gt["ymin"]
        xmax = selected_gt["xmax"]
        ymax = selected_gt["ymax"]
        gt_boxes.append((xmin, ymin, xmax, ymax))

    index = 0

    for gt_box in gt_boxes:
        index += 1
        new_filename = filename.split(".")[0] + "-" + str(index) + filename.split(".")[1]
        print(gt_box)
        print(new_filename)
        image = cv2.imread(str(image_path))

        (xmin, ymin, xmax, ymax) = gt_box

        roi = image[ymin:ymax, xmin:xmax]
        # cv2.imwrite(new_filename, roi)

        # for investigating where is the groudn truths are.
        p1 = (xmin, ymin)
        p2 = (xmax, ymax)
        image = cv2.rectangle(image, p1, p2, (255, 0, 0), 2)
        imgplot = plt.imshow(image)
        plt.show()
