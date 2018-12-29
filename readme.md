1) clone the repo using

    `git clone Deep_facial_recognition`

2) cd into the directory

    `cd Deep_facial_recognition`

3) First run the following in terminal to install the dependencies

    `pip3 install -r requirements.txt`

4) Create a new folder with class name as the directory name eg: Krishna
5) Add images of the New Person into directory
6) Or use make_class.py to create classes

###NOTE: Make sure you have atleast two classes in the dataset folder

7) execute the following to run facial recognition on a image

 `python recognize.py --image <PATH_TO_TEST_IMAGE>`

8) execute the following to run facial recognition on webcam

`python recognize_video.py`

Optionally you can pass in the argument --unauth to sound alarm when that person is in the frame

`python recognize_video.py --unauth Krishna`
