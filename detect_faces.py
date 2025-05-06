import numpy as np
import cv2
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="path to input image")
parser.add_argument("--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
parser.add_argument("--model", required=True, help="path to Caffe pre-trained model")
parser.add_argument("--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = parser.parse_args()

# Load the pre-trained model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

# Load the input image
image = cv2.imread(args.image)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args.confidence:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Show the output
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
