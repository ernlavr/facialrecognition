// Implemented by Ernests Lavrinovics as part of Image Processing and Computer Vision course in Aalborg University, Medialogy 2022.

#include "hog.hpp"
#include <vector>                                                               
#include <iostream>                                                             
#include <numeric>  
#include <math.h>
#include <filesystem>

HOG::~HOG(){ }

HOG::HOG(std::string path, bool verbose) 
: verbose(verbose), imgPath(path)
{
    // Load image
    auto inputImgRGB = cv::imread(path);

    // Error Handling
    if (inputImgRGB.empty()) {
        std::cout << "Image File Not Found" << std::endl;
    }

    assert(inputImgRGB.rows % 2 == 0);
    assert(inputImgRGB.cols % 2 == 0);
    // Convert inputImgRGB to grayscale
    cv::cvtColor(inputImgRGB, inputImgGray, cv::COLOR_BGR2GRAY);

    if(verbose) {
        saveImage("crop1_64128_gray.jpg", inputImgGray);
    }
}

void HOG::gradientComputation() {
    // Create filters with [-1, 0, 1] mask
    cv::Mat filterX = (cv::Mat_<char>(1, 3) << -1, 0, 1);
    cv::Mat filterY = (cv::Mat_<char>(3, 1) << -1, 0, 1);

    // Compute the gradient and derive its magnitude and angles
    cv::Mat gradX, gradY;
    cv::filter2D(inputImgGray, gradX, CV_32F, filterX);
    cv::filter2D(inputImgGray, gradY, CV_32F, filterY);
    cv::magnitude(gradX, gradY, magnitude);
    cv::phase(gradX, gradY, angles, true);

    // Save images in verbose mode
    if(verbose) {
        cv::Mat gradXY;
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0.0, gradXY);
        saveImage("gradXY.jpg", gradXY);
        saveImage("gradX.jpg", gradX);
        saveImage("gradY.jpg", gradY);
        saveImage("magnitude.jpg", magnitude);
        saveImage("directions.jpg", angles);       
    }
}

void HOG::process() {
    cellsY = static_cast<int>(inputImgGray.rows / pixelsPerCell);
    cellsX = static_cast<int>(inputImgGray.cols / pixelsPerCell);
    histogram.resize(cellsY);

    gradientComputation();

    // process an image block by block
    for (int y = 0; y < cellsY; y += 1) {
        for (int x = 0; x < cellsX; x += 1) {
            // fetch the cell
            cv::Rect cell_rect = cv::Rect(x * pixelsPerCell, y * pixelsPerCell, pixelsPerCell, pixelsPerCell);
            cv::Mat cellMagnitude = magnitude(cell_rect);
            cv::Mat cellAngle = angles(cell_rect);
            cv::Mat cell = inputImgGray(cell_rect);
            
            // Prepare destination container and compute the histogram for the cell
            std::vector<float> dstHist (numBins, 0);
            processCell(cell, cellMagnitude, cellAngle, dstHist);

            // store the histogram in the histogram vector
            histogram.at(y).push_back(dstHist);
        }
    }

    // Perform 16x16 block normalization on the histogram
    L2blockNormalization();
    
    // Print and save if verbose mode is enabled
    if(verbose) {
        float average = std::accumulate( descriptor.begin(), descriptor.end(), 0.0) / descriptor.size();
        std::cout << "Length of final descriptor: " << descriptor.size() << std::endl;
        std::cout << "Average value of final descriptor: " << average << std::endl;
        writeToFile("HOG_myImpl.txt", descriptor);
    }
}

void HOG::computeAndPrintOpenCV() {
    // Initialize
    auto e = cv::HOGDescriptor(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), numBins);
    std::vector<float> desc;
    std::vector<cv::Point> locations;

    // Compute
    e.compute(inputImgGray, desc, cv::Size(8, 8), cv::Size(0, 0), locations);

    float hogAvg = std::accumulate( desc.begin(), desc.end(), 0.0) / desc.size();
    // Print stats
    std::cout << "size of openCV: " << desc.size() << std::endl;
    std::cout << "openCV avg: " << hogAvg << std::endl;
    
    // Write to .txt file
    writeToFile("HOG_openCV.txt", desc);
}

float HOG::computeL2norm(std::vector<float> input) {
    float sum = 0;
    for (auto i : input) {
        sum += i * i;
    }
    return sqrt(sum);
}

void HOG::clipNumber(float &input) {
    if(input > 0.2)
        input = 0.2;
    else if(input < 0)
        input = 0;
}

void HOG::L2blockNormalization() {
    // Loop over the histogram variable with a 16x16 sliding window and normalize the values
    std::vector<std::vector<std::vector<float>>> normalizedHistogram;
    normalizedHistogram.resize(cellsY - 1); // 16x16 block will have 1 less row than the original image
    for (int y = 0; y < cellsY - 1; y += 1) { 
        normalizedHistogram.at(y).resize(cellsX - 1); // 16x16 block will have 1 less column than the original image
        for (int x = 0; x < cellsX - 1; x += 1) {

            std::vector<float> mergedCells;   
            std::vector <float> norms;
            std::vector<std::vector<float>> cellVec;
            
            for (int width = y; width < y + cellsPerWindow_H; ++width) {
                for (int height = x; height < x + cellsPerWindow_H; ++height) {
                    auto cell1 = histogram.at(width).at(height);
                    auto norm1 = computeL2norm(cell1);

                    cellVec.push_back(cell1);
                    norms.push_back(norm1);
                }
            }

            // compute the average of the norms
            // Merge the vectors into 1
            // Loop over normVector and for each cell normalize the values using the correct coefficient from the norms array
            for (int i = 0; i < cellVec.size(); i++) {
                auto cellVecSize = cellVec.at(i).size();
                for (int j = 0; j < cellVecSize; j++) {
                    auto entry = cellVec.at(i).at(j) / norms.at(i);
                    clipNumber(entry);
                    mergedCells.push_back(entry);
                }
            }
            normalizedHistogram.at(y).at(x) = mergedCells;
        }
    }

    // Stretch the normalizedHistogram into a single, 1d vector
    for (int y = 0; y < cellsY - 1; y += 1) {
        for (int x = 0; x < cellsX - 1; x += 1) {
            auto cell = normalizedHistogram.at(y).at(x);
            for (auto i : cell) {
                descriptor.push_back(i);
            }
        }
    }
}

void HOG::processCell(const cv::Mat &cell, const cv::Mat &cellMagn, cv::Mat &cellAngle, std::vector<float> &dstHist) {
    // Convert dstAngle from unsigned to signed if a value is larger than 180
    convertToUnsignedAngles(cellAngle);

    // Calculate the histogram of gradient
    int histogramSize = 9;
    int binSize = 180 / histogramSize;
    for (int i = 0; i < cellAngle.rows; i++) {
        for (int j = 0; j < cellAngle.cols; j++) {
            auto angle = cellAngle.at<float>(i, j);
            auto binUnrounded = static_cast<int>(angle / binSize);
            int binRounded = static_cast<int>(binUnrounded);

            auto mag = cellMagn.at<float>(i, j);
            dstHist[binRounded] += mag;
        }
    }
}

void HOG::writeToFile(std::string filename, std::vector<float> &vector) {
    // Write the descriptor to a file
    std::ofstream file;
    file.open(OUTPUT_FOLDER + filename);
    for (auto i : vector) {
        file << i << " ";
    }
    file.close();
}

void HOG::convertToUnsignedAngles(cv::Mat &srcMat) {
    for (int i = 0; i < srcMat.rows; i++) {
        for (int j = 0; j < srcMat.cols; j++) {
            if (srcMat.at<float>(i, j) >= 180) {
                srcMat.at<float>(i, j) = srcMat.at<float>(i, j) - 180;
            }
        }
    }
}

/**
 * @brief Display image1 colored and image2 grayscale horizontally side by side
 * @param image1 
 * @param image2 
 */
void HOG::displayImage(cv::Mat image1, cv::Mat image2) {
    cv::imshow("img1", image1);
    cv::imshow("img2", image2);
    cv::waitKey(0);
}

/**
 * @brief Display image1 colored and image2 grayscale horizontally side by side
 * @param image1 
 * @param image2 
 */
void HOG::displayImage(cv::Mat image1) {
    cv::Mat toDisplay;
    image1.convertTo(toDisplay, CV_8U);
    cv::imshow("img1", toDisplay);
    cv::waitKey(0);
}

void HOG::saveImage(std::string filename, cv::Mat image) {
    auto outputPath = OUTPUT_FOLDER + filename;
    std::filesystem::path output(OUTPUT_FOLDER);
    std::filesystem::path file(filename);
    auto pathToWrite = (std::filesystem::current_path() / output / file).string();
    cv::imwrite(outputPath, image);
}