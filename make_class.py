import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument( "--class", required=True,
	help="Name of the class")
args = vars(ap.parse_args())
name = args["class"]
directory = os.path.join('dataset',name)
if not os.path.exists(directory):
    os.makedirs(directory)
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break
    key = cv2.waitKey(1)

    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key%256 == 32:
        # SPACE pressed
        img_name = "{}{}.png".format(name,img_counter)
        cv2.imwrite(os.path.join(directory,img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1
    cv2.putText(frame, "press SPACE to Capture and ESC to Quit", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imshow("Capture", frame)

cam.release()

cv2.destroyAllWindows()