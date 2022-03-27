#ifndef TEST_HPP
#define TEST_HPP

#include <string>
#include <opencv2/opencv.hpp>

class hog
{
private:
    /* data */
public:
    hog(std::string path);
    ~hog();

    /**
     * @brief Compresses each colour channel to improve performance at low FPPW
     */
    void squareRootColorGamma();

    /**
     * @brief Gradient computed using 1D gaussian mask (-1; 0; 1)
     * For color images, calculates seperate gradients for each color
     * and takes the one with the largest norm at
     * pixel's gradient vector
     * 
     */
    void gradientComputation();

    /**
     * @brief calculates a weighted vote for an edge
     * orientation histogram channel based on the orientation of the
     * gradient element centred on it, and the votes are accumu-
     * lated into orientation bins over local spatial regions that we call cells
     */
    void orientationHistogram();


};

#endif