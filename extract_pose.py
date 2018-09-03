# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import math
import numpy as np
import glob
from random import randint
from PIL import Image
from sys import platform

# Remember to add your installation path here
# Option a
dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
else: sys.path.append('../../python');
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled

def distance(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["model_pose"] = "BODY_25"
params["net_resolution"] = "-1x176"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

image_file_names = glob.glob("/usr/local/images/*.png")
print("Loaded images: ", image_file_names)
faces = [Image.open(facePath) for facePath in image_file_names]

threshold = 0.3
cap = cv2.VideoCapture(0)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while 1:
    faceImg = faces[randint(0, len(faces) - 1)]
    for i in list(range(100)):
        # Read new image
        ret, img = cap.read()
        if not ret: 
            continue
        # Output keypoints and the image with the human skeleton blended on it
        keypoints, output_image = openpose.forward(img, True)
        # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
        if keypoints.any():
            for person in keypoints:
                nose = person[0]
                neck = person[1]
                lEar = person[18]
                rEar = person[17]
                if lEar[2] > threshold and rEar[2] > threshold and nose[2] > threshold and neck[2] > threshold:
                    width = distance(lEar[0:2], rEar[0:2])
                    degrees = math.degrees(math.asin((rEar[1] - lEar[1]) / width))
                    width = int(1.3 * width)
                    height = int(width * 1.3)
                    startX = int(nose[0] - (width / 2.0))
                    startY = int(nose[1] - 10 - (height / 2.0))

                    res = faceImg.resize((width, height)).rotate(degrees)
                    if isinstance(output_image, np.ndarray):
                        output_image = Image.fromarray(output_image.astype('uint8'), 'RGB')
                    if startX > 0 and startY > 0:
                        output_image.paste(res, (startX, startY), res)
                    else:
                        cropBox = (max(0, -startX), max(0, -startY), width, height)
                        res = res.crop(cropBox)
                        output_image.paste(res, (max(0, startX), max(0, startY)), res)
                    
        if isinstance(output_image, Image.Image):
            output_image = np.array(output_image)

        # Display the image
        cv2.imshow("window", output_image)
        cv2.waitKey(15)
