#!/usr/bin/python3
"""
Tests the OpenCV and custom HOG implementaion descriptors for statistical differences in them.
Checks their mean, length and standard deviation.
"""
from Constants import *
import scipy.stats
import numpy as np

def main():
    # Fills the global variables timesElapsedWithAR and timesElapsedWithoutAR
    mine = getElements(os.path.join(FOLDER_DATA_DESCRIPTORS, "HOG_myImpl.txt"))
    opencv = getElements(os.path.join(FOLDER_DATA_DESCRIPTORS, "HOG_openCV.txt"))

    # Convert to ndarray
    mine_np : np.ndarray=np.array(mine)
    opencv_np : np.ndarray=np.array(opencv)

    # Calculate mean+std
    print(f"Mean Mine       {len(mine_np)}")
    print(f"Length OpenCV   {len(opencv_np)}")
    print(f"Mean Mine       {mine_np.mean()}")
    print(f"Std Mine        {mine_np.std()}")
    print(f"Mean OpenCV     {opencv_np.mean()}")
    print(f"Std OpenCV      {opencv_np.std()}")
    
    # Plot the data and run normality tests
    testPerformMannWhitneyU(mine_np, opencv_np)

def getElements(filepath : str):
    """ Open the file at filepath, split all elements according to whitespace
    type cast them to floats and return them as a list
     and return the first 500 elements as a list 
    """
    with open(filepath, 'r') as f:
        return [float(x) for x in f.read().split()]

def testPerformMannWhitneyU(mine, opencv):
    output = scipy.stats.mannwhitneyu(mine, opencv)
    alpha = 0.05
    if(output.pvalue < alpha):
        print(f"MannWhitneyU: Reject H0, pvalue = {output.pvalue}")
    else:
        print("MannWhitneyU: Fail to reject H0, pvalue = {output.pvalue}")


if __name__ == '__main__':
    main()