# Sign_Language_Detection
This repository contains a variety of tools to build up a experimental ecosystem for recognizing signs of the german sign language (DGS). Our claim is an experimental attempt at live subtitling of gestures. For this we train a deep learning model (RNN) for predicting the actualsigns made by a person filmed. Therefore we use MediaPipe, a framework for building ML pipelines, to extract face and hand positions, including multiple coordinates for each finger.
The project is to help the society of deprived people who are facing difficulties to speaking and hearing. Python (3.7.4), cv2 (openCV) (version 3.4.2), numpy, cvzone, Jupyter notebook.

# Here are some Screenshots
![Screenshot (210)](https://github.com/its-deepakdixit/Sign_Language_Detection/assets/92906504/b603b6be-ae62-4d98-a2cf-11bf004e1ebe)

# Its based on American Sign Language
![Sign-language-letters](https://github.com/its-deepakdixit/Sign_Language_Detection/assets/92906504/2b1fb957-ddfc-4fe3-a68c-74a25af5fe0d)

# Reference images
![sam-slr](https://github.com/its-deepakdixit/Sign_Language_Detection/assets/92906504/bae340f0-f3ad-4ed2-a13f-8348a88f090b)
![Capture1](https://github.com/its-deepakdixit/Sign_Language_Detection/assets/92906504/a330cde0-ed98-459c-9a06-8cae5572ddb4)

# Requirements and Docker Image
The code is written using Anaconda Python >= 3.6 and Pytorch 1.7 with OpenCV.
Detailed enviroment requirment can be found in requirement.txt in each code folder.
For convenience, we provide a Nvidia docker image to run our code.
