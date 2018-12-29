
This Project uses Opencv's [OpenFace](https://cmusatyalab.github.io/openface/), a pytorch implementation of [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) to train a simple svm that performs deep facial recognition with minimal number (4-5) of images per class

1) clone the repo using

    `git clone Deep_facial_recognition`

2) cd into the directory

    `cd Deep_facial_recognition`

3) First run the following in terminal to install dependencies

    `pip3 install -r requirements.txt`

4) Run make_class.py with class name as argument to create your own class of faces

    `python make_class.py --class Krishna`
    
     _Manual Addition:_   
 
    1) Create a new folder with class name as the directory in dataset folder eg: Krishna
    2) Add images of the New Person into directory ( make sure there is only one person in the image )

NOTE: You need atleast two classes in the dataset folder to proceed with training

5) Run train.py to create corresponding embeddings and pickle files in the output folder

7) Execute the following to run facial recognition on a image

    `python recognize.py --image <PATH_TO_TEST_IMAGE>`

8) Execute the following to run facial recognition on webcam

    `python recognize_video.py`

Optionally you can pass in the argument --unauth <CLASS_NAME> to sound alarm when that particular person is in the frame

    `python recognize_video.py --unauth Krishna`
