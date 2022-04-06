#!/usr/bin/python3
"""
Computes HOG features using Python and OpenCV
"""

import cv2
import numpy as np
from Constants import *


def main(verbose, image):
    # Compute the HOG features
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hog_features = hog.compute(image)
    
    if(verbose):
        # Print the length, mean and standard deviation of the HOG features
        print(f"Length: {len(hog_features)}")
        print(f"Mean: {hog_features.mean()}")
        print(f"Std: {hog_features.std()}")
        # Write hog_features to a text file called pyHOG_opencv.txt and store it in the data folder
        with open(os.path.join(FOLDER_DATA_DESCRIPTORS, "pyHOG_opencv.txt"), 'w') as f:
            for i in range(len(hog_features)):
                f.write(f"{hog_features[i]} ")

def importImage(filepath : str):
    """
    Imports an image from filepath and returns it as a numpy array
    """
    return cv2.imread(filepath)

def convertImageToGrayscale(image : np.ndarray):
    """
    Converts an image to grayscale and returns it as a numpy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
    verbose = True
    profile = True
    profileIterations = 100000
    # Import the test image from data folder and convert it to grayscale
    image = importImage(os.path.join(FOLDER_DATA_DESCRIPTORS, "crop1_64128.png"))
    image = convertImageToGrayscale(image)
    # Measure the time it takes to execute main() function
    
    if(profile == False):
        main(verbose, image)
    else:
        averageTime = 0
        totalTime = 0
        for i in range(profileIterations):
            start = cv2.getTickCount()
            main(False, image)
            end = cv2.getTickCount()
            
            timesElapsed = (end - start) / cv2.getTickFrequency()
            averageTime += timesElapsed
            averageTime /= profileIterations
            totalTime += timesElapsed
            
            # Write time elapsed to a file
            with open(os.path.join(FOLDER_DATA_DESCRIPTORS, "pyHOG_times.txt"), 'a') as f:
                f.write(f"{timesElapsed} ")
        print(averageTime)
        print(totalTime)