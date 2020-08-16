from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import nms

model_path = "model"
encoder_path = "encoder"
input_size = (224, 224)
min_proba = 0.80
max_proposals_infer = 300

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the our fine-tuned model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(model_path)
lb = pickle.loads(open(encoder_path, "rb").read())

# load the input image from disk
image = cv2.imread(args["image"])
image = cv2.resize(image, input_size, interpolation = cv2.INTER_AREA)

# run selective search on the image to generate bounding box proposal
# regions
print("[INFO] running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []
for (x, y, w, h) in rects:
	# extract the region from the input image, convert it from BGR to
	# RGB channel ordering, and then resize it to the required input
	# dimensions of our trained CNN
    roi = image[y:y + h, x:x + w] # check
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, input_size, interpolation=cv2.INTER_CUBIC)
	
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
	
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

# convert the proposals and bounding boxes into NumPy arrays
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")

print("[INFO] proposal shape: {}".format(proposals.shape))

print("[INFO] classifying proposals...")
proba = model.predict(proposals)

# find the index of all predictions that are positive for the "wound" class
print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "wound")[0]

# use the indexes to extract all bounding boxes and associated class 
# label probabilities associated with the "wound" class
boxes = boxes[idxs]
proba = proba[idxs][:, 1]

# filter proba array
idxs = np.where(proba >= min_proba)
boxes = boxes[idxs]
proba = proba[idxs]

print(proba)
print(idxs)
print(boxes)

# clone the original image so that we can draw on it
clone = image.copy()
# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = box
	cv2.rectangle(clone, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Wound: {:.2f}%".format(prob * 100)
	cv2.putText(clone, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output after *before* running NMS
cv2.imshow("Before NMS", clone)

# run non-maxima suppression on the bounding boxes
boxIdxs = nms.non_max_suppression_fast(boxes, proba) # this need to be index
print(boxIdxs)
for i in boxIdxs:
	item_index = np.where(boxes==i)
	(startX, startY, endX, endY) = boxes[item_index[0][0]]
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text = "Wound: {:.2f}%".format(proba[item_index[0][0]] * 100)
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output image *after* running NMS
cv2.imshow("After NMS", image)
cv2.waitKey(0)
