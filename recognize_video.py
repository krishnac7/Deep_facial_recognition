# --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle


from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from pygame import mixer
import pygame
pygame.init()
mixer.music.load('alarm1.wav')

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=False,default="face_models",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=False,default="face_models/openface_nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=False,default="output/recognizer.pickle",
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,default="output/le.pickle",
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("--unauth",required=False,default=None,help ="name an intruder")
args = vars(ap.parse_args())


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


fps = FPS().start()


while True:

	frame = vs.read()


	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]


	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)


	detector.setInput(imageBlob)
	detections = detector.forward()


	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]


		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]


			if fW < 20 or fH < 20:
				continue


			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()


			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]


			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
			if name == args["unauth"] and not mixer.music.get_busy():
				mixer.music.play()



	fps.update()

	cv2.putText(frame, "press ESC to Quit", (25, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key % 256 == 27:
		# ESC pressed
		print("Escape hit, closing...")
		break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
vs.stop()