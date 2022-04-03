#ifndef HOG_HPP
#define HOG_HPP

#include <string>
#include <opencv2/opencv.hpp>

class HOG
{
private:

    std::vector<std::vector< std::vector<float> >> histogram; // Width x Height x Bins
    std::vector<float> finalDescriptor;
    std::string imgPath;

    cv::Mat magnitude;
    cv::Mat angles;

    size_t numBins = 9; // Number of bins in histogram 
    size_t spatialResolution = 180; // Spatial resolution of histogram
    size_t pixelsPerBlock = 16;
    size_t pixelsPerCell = 8; 
    size_t blockStride = 8; // for a 4 fold coverage
    size_t cellsX;
    size_t cellsY;
    const size_t cellsPerWindow_H = 2;
    const size_t cellsPerWindow_W = 2;
public:
    HOG(std::string path);
    ~HOG();

    cv::Mat inputImgRGB;
    cv::Mat inputImgGray;
    cv::Mat output;

    /**
     * @brief Gradient computed using 1D gaussian mask (-1; 0; 1)
     * For color images, calculates seperate gradients for each color
     * and takes the one with the largest norm at
     * pixel's gradient vector
     */
    void gradientComputation();

    /**
     * @brief calculates a weighted vote for an edge
     * orientation histogram channel based on the orientation of the
     * gradient element centred on it, and the votes are accumu-
     * lated into orientation bins over local spatial regions that we call cells
     */
    void orientationHistogram();

    void process();

    void processCell(cv::Mat &cell, cv::Mat &dstMag, cv::Mat &dstAngle, std::vector<float> &dstHist);

    /** 
     * @brief Lowe-style clipped L2 norm
     */
    void L2blockNormalization();

    void binning();

    /**
     * @brief Concatenates two images horizontally into a single image in a side-by-side manner
     * 
     * @return cv::Mat 
     */
    void displayImage(cv::Mat image1, cv::Mat image2);

    /**
     * @brief Displays the loaded image
     * 
     */
    void displayImage(cv::Mat image1);

    /**
     * @brief Converts the angle from signed (0-360) to unsigned (0-180)
     * 
     * @param angle 
     */
    void convertToUnsignedAngles(cv::Mat &srcMat);

    /**
     * @brief Returns a visualization-ready image of the features
     * 
     * @return cv::Mat 
     */
    cv::Mat getVectorMask();

    void computeAndPrintOpenCV();

    void computeMagnitudeAndAngle();

    void writeToFile(std::string path, std::vector<float> &descriptor);

};

#endif