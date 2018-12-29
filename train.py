#--dataset dataset --embeddings output/embeddings.pickle --detector face_models --embedding-model face_models/openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle


from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,default="dataset",
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=False,default="output/embeddings.pickle",
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=False,default="face_models",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=False,default="face_models/openface_nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-r", "--recognizer", required=False,default="output/recognizer.pickle",
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,default="output/le.pickle",
	help="path to output label encoder")



args = vars(ap.parse_args())

def extract_embeddings():
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(args["dataset"]))


    knownEmbeddings = []
    knownNames = []


    total = 0


    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]


        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]


        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)


        detector.setInput(imageBlob)
        detections = detector.forward()


        if len(detections) > 0:

            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]


            if confidence > args["confidence"]:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")


                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]


                if fW < 20 or fH < 20:
                    continue


                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()


                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1


    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(args["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()

def train_model():

    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args["embeddings"], "rb").read())


    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])


    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)


    f = open(args["recognizer"], "wb")
    f.write(pickle.dumps(recognizer))
    f.close()


    f = open(args["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()


def main():
    extract_embeddings()
    train_model()

if __name__ == "__main__":
    main()