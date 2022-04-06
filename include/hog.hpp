// Implemented by Ernests Lavrinovics as part of Image Processing and Computer Vision course in Aalborg University, Medialogy 2022.

#ifndef HOG_HPP
#define HOG_HPP

#include <string>
#include <opencv2/opencv.hpp>

class HOG
{
private:

    // CONTAINERS
    /**
     * @brief Holds the histogram for the 2D image. 1D - width; 2D height; 3D - histogram
     */
    std::vector<std::vector< std::vector<float> >> histogram; // Width x Height x Bins
    /**
     * @brief Magnitude of the X-Y gradients
     */
    cv::Mat magnitude;
    /**
     * @brief Angles for the X-Y gradients of the input image
     */
    cv::Mat angles;

    // CONSTANTS
    /**
     * @brief Is verbose mode enabled?
     */
    const bool verbose;
    /**
     * @brief Path to the input image
     */
    const std::string imgPath;
    /**
     *  @brief Relative folder from build/bin/ to the root folder
     */
    const std::string ROOT_FOLDER = "../../";
    /**
     * @brief Folder to which all output files are saved to
     */
    const std::string OUTPUT_FOLDER = ROOT_FOLDER + "output/";
    /**
     * @brief Number of cells in width
     */
    size_t cellsX;
    /**
     * @brief Number of cells in height
     */
    size_t cellsY;


    // HYPERPARAMETERS

    /** 
     * @brief Resolution of the histogram
     */
    const size_t spatialResolution = 180;
    /**
     * @brief Number of pixels per block. Block consists of cells that is used during batch normalization.
     */
    const size_t pixelsPerBlock = 16;
    /**
     * @brief Number of pixels per cell for which a histogram will be computed
     */
    const size_t pixelsPerCell = 8; 
    /**
     * @brief Stride by which the sliding window will slide over the image 
     */
    const size_t blockStride = 8;
    /**
     * @brief Number of bins that represent angles for the gradient orientation
     */
    const size_t numBins = 9;
    /**
     * @brief Number of cells per block in height
     */
    const size_t cellsPerWindow_H = 2;
    /**
     * @brief Number of cells per block in width
     */
    const size_t cellsPerWindow_W = 2;


    // FUNCTIONS

    /**
     * @brief Compute the summed square-root of the given input
     * 
     * @param input 
     * @return float summed-square root of the input
     */
    void L2norm(std::vector<std::vector<float>> &input);
    /**
     * @brief Clip the input to the 0-0.2f range
     * 
     * @param input Float to be cliped
     */
    void clipNumber(std::vector<std::vector<float>> &input);
    /**
     * @brief Gradient computed using 1D gaussian mask (-1; 0; 1)
     * For color images, calculates seperate gradients for each color
     * and takes the one with the largest norm at
     * pixel's gradient vector
     */
    void gradientComputation();
    /**
     * @brief Process a single cell of the image
     * 
     * @param cell Cell to compute a histogram for
     * @param magn Gradient magnitudes of the particular cell
     * @param angle Gradient angles of the particular cell
     * @param dstHist Oriented histogram of the particular cell
     */
    void processCell(const cv::Mat &cell, const cv::Mat &magn, cv::Mat &angle, std::vector<float> &dstHist);
    /** 
     * @brief Lowe-style clipped L2 norm
     */
    void L2blockNormalization();
    /**
     * @brief Converts the angle from signed (0-360) to unsigned (0-180)
     * 
     * @param angle 
     */
    void convToUnsignAngl(cv::Mat &srcAngles);

public:
    /**
     * @brief Construct a new HOG object. (!) Automatically converts the input image to grayscale
     * 
     * @param path path to an image
     * @param verbose verbose mode enabled. Prints additional information to the console, saves data, etc.
     */
    HOG(std::string path, bool verbose);
    ~HOG();

    // CONTAINERS
    
    /**
     * @brief Input image converted to grayscale
     */
    cv::Mat inputImgGray;
    /**
     * @brief Final, normalized 1D feature descriptor of the image 
     */
    std::vector<float> descriptor;


    // FUNCTIONS

    /**
     * @brief Main entry point for the HOG computation. Linearly issues all necessary calls in a sequential manner to compute the final descriptor
     */
    void process();
    /** 
     * Computes and prints the HOG feature descriptor using OpenCV
     */
    void computeAndWriteOpenCV();
    /**
     * @brief Writes a descritor to a text file at the designated path
     * 
     * @param path Path to which the descriptor will be written
     * @param descriptor Descriptor to be written
     */
    void writeToFile(std::string path, std::vector<float> &descriptor);
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
     * @brief Saves the image to the specified path. Specify the .JPG extension
     * 
     * @param pathOfImage 
     * @param image 
     */
    void saveImage(std::string pathOfImage, cv::Mat image);
};

#endif