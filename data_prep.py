import os
import pathlib
import cv2
import pandas


raw_images_path = "DFUC2020"
groundtruth = "DFUC2020/groundtruth.csv"
max_proposal = 2000
INPUT_DIMS = (160,160)

for dirPath in ("wound", "no_wound"):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

data_dir = pathlib.Path(raw_images_path)
image_paths = list(data_dir.glob('*/*.jpg'))
image_count = len(image_paths)
print("[INFO] Total raw image count: {}".format(image_count))
print(image_paths[0])

gt = pandas.read_csv(groundtruth)
image_names = gt["name"]
gt = gt.set_index(["name"])

totalPositive = 0
totalNegative = 0

def calc_IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U

# iterate image paths
for(i, image_path) in enumerate(image_paths):
	print("[INFO] processing image {}/{}...".format(i + 1,len(image_paths)))
	filename = image_path.parts[-1]	# selecting filename
	print(filename)
	selected_gt = gt.loc[filename]	# filtering groundtruth trough filename
	gt_boxes = []	# groundtruth box pos
	
	# if an image include more than one groundtruth iterate over it
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

	image = cv2.imread(str(image_path))
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()
	rects = ss.process()
	proposedRects= []

	for (x, y, w, h) in rects:
		proposedRects.append((x, y, x + w, y + h))
	
	positiveROIs = 0
	negativeROIs = 0

	for proposed_rect in proposedRects:
		(propStartX, propStartY, propEndX, propEndY) = proposed_rect

		for gt_box in gt_boxes:
			iou = calc_IOU(gt_box, proposed_rect)
			(gtStartX, gtStartY, gtEndX, gtEndY) = gt_box
			roi = None
			outputPath = None
			
			if iou > 0.7 and positiveROIs < 30:
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.jpg".format(totalPositive)
				outputPath = os.path.sep.join(["wound", filename])

				positiveROIs += 1
				totalPositive += 1

			fullOverlap = propStartX >= gtStartX
			fullOverlap = fullOverlap and propStartY >= gtStartY
			fullOverlap = fullOverlap and propEndX <= gtEndX
			fullOverlap = fullOverlap and propEndY <= gtEndY
			
			if not fullOverlap and iou < 0.05 and negativeROIs <= 10:
				# extract the ROI and then derive the output path to
				# the negative instance
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.jpg".format(totalNegative)
				outputPath = os.path.sep.join(["no_wound",
					filename])
				# increment the negative counters
				negativeROIs += 1
				totalNegative += 1
			# check to see if both the ROI and output path are valid
			
			if roi is not None and outputPath is not None:
				# resize the ROI to the input dimensions of the CNN
				# that we'll be fine-tuning, then write the ROI to
				# disk
				roi = cv2.resize(roi, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(outputPath, roi)
