#include "hog.hpp"
#include <vector>                                                               
#include <iostream>                                                             
#include <numeric>  
#include <math.h>

HOG::~HOG(){ }

HOG::HOG(std::string path)
{
    // Load image
    inputImgRGB = cv::imread(path);

    // Error Handling
    if (inputImgRGB.empty()) {
        std::cout << "Image File Not Found" << std::endl;
    }

    assert(inputImgRGB.rows % 2 == 0);
    assert(inputImgRGB.cols % 2 == 0);
    // Convert inputImgRGB to grayscale
    cv::cvtColor(inputImgRGB, inputImgGray, cv::COLOR_BGR2GRAY);
    imwrite("crop1_64128_gray.png", inputImgGray);
}

void HOG::computeMagnitudeAndAngle() {
    // Compute a gradient using sobel operation and [-1, 0, 1] filter
    cv::Mat filterX = (cv::Mat_<char>(1, 3) << -1, 0, 1);
    cv::Mat filterY = (cv::Mat_<char>(3, 1) << -1, 0, 1);

    cv::Mat gradX, gradY;
    cv::filter2D(inputImgGray, gradX, CV_32F, filterX);
    cv::filter2D(inputImgGray, gradY, CV_32F, filterY);
    cv::magnitude(gradX, gradY, magnitude);
    cv::phase(gradX, gradY, angles, true);

}

void HOG::process() {
    cellsY = static_cast<int>(inputImgGray.rows / pixelsPerCell);
    cellsX = static_cast<int>(inputImgGray.cols / pixelsPerCell);
    histogram.resize(cellsY);

    computeMagnitudeAndAngle();

    // process an image block by block
    for (int y = 0; y < cellsY; y += 1) {
        for (int x = 0; x < cellsX; x += 1) {
            // fetch the cell
            cv::Rect cell_rect = cv::Rect(x * pixelsPerCell, y * pixelsPerCell, pixelsPerCell, pixelsPerCell);
            cv::Mat cellMagnitude = magnitude(cell_rect);
            cv::Mat cellAngle = angles(cell_rect);

            cv::Mat cell = inputImgGray(cv::Rect(x * pixelsPerCell, y * pixelsPerCell, pixelsPerCell, pixelsPerCell));
            std::vector<float> dstHist (numBins, 0);
            processCell(cell, cellMagnitude, cellAngle, dstHist);

            // store the histogram in the histogram vector
            histogram.at(y).push_back(dstHist);
        }
    }

    // Perform 16x16 block normalization on the histogram
    L2blockNormalization();
    
    float average = std::accumulate( finalDescriptor.begin(), finalDescriptor.end(), 0.0) / finalDescriptor.size();
    std::cout << "final descriptor avg: " << average << std::endl;
    writeToFile("myImpl.txt", finalDescriptor);
    
}

void HOG::computeAndPrintOpenCV() {
    // Initialize
    auto e = cv::HOGDescriptor(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), numBins);
    std::vector<float> desc;
    std::vector<cv::Point> locations;

    // Compute
    e.compute(inputImgGray, desc, cv::Size(8, 8), cv::Size(0, 0), locations);
    
    auto opencvMax = std::max_element(std::begin(desc), std::end(desc));
    auto myMax = std::max_element(std::begin(finalDescriptor), std::end(finalDescriptor));


    float hogAvg = std::accumulate( desc.begin(), desc.end(), 0.0) / desc.size();
    // Print stats
    std::cout << "size of mine: " << finalDescriptor.size() << std::endl;
    std::cout << "size of openCV: " << desc.size() << std::endl;
    std::cout << "openCV avg: " << hogAvg << std::endl;
    
    // Write to .txt file
    writeToFile("opencv.txt", desc);
    writeToFile("mine.txt", finalDescriptor);
}

float computeL2norm(std::vector<float> input) {
    float sum = 0;
    for (auto i : input) {
        sum += i * i;
    }
    return sqrt(sum);
}

void clipNumber(float &input) {
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
            // print all values from block_norm
            for(auto& val : mergedCells)
                std::cout << val << " ";
            std::cout << std::endl;
            normalizedHistogram.at(y).at(x) = mergedCells;
        }
    }

    // Stretch the normalizedHistogram into a single, 1d vector
    for (int y = 0; y < cellsY - 1; y += 1) {
        for (int x = 0; x < cellsX - 1; x += 1) {
            auto cell = normalizedHistogram.at(y).at(x);
            for (auto i : cell) {
                finalDescriptor.push_back(i);
            }
        }
    }
}

void HOG::processCell(cv::Mat &cell, cv::Mat &dstMag, cv::Mat &dstAngle, std::vector<float> &dstHist) {
    // Convert dstAngle from unsigned to signed if a value is larger than 180
    convertToUnsignedAngles(dstAngle);

    // Calculate the histogram of gradient
    int histogramSize = 9;
    int binSize = 180 / histogramSize;
    for (int i = 0; i < dstAngle.rows; i++) {
        for (int j = 0; j < dstAngle.cols; j++) {
            auto angle = dstAngle.at<float>(i, j);
            auto binUnrounded = static_cast<int>(angle / binSize);
            int binRounded = static_cast<int>(binUnrounded);

            auto mag = dstMag.at<float>(i, j);
            dstHist[binRounded] += mag;
        }
    }
}

cv::Mat HOG::getVectorMask() {
    // Create a vector mask
    cv::Mat vectorMask(inputImgGray.rows, inputImgGray.cols, CV_8U, cv::Scalar(0));

    // Get the max value from the histogram
    float maxValue = 0;
    //std::vector<std::vector<float>> cell_hist_maxs(this.ce);
    for (int i = 0; i < histogram.size(); i++) {
        for(int j = 0; j < histogram.at(i).size(); j++) {
            for (int k = 0; k < histogram.at(i).at(j).size(); k++) {
                if (histogram.at(i).at(j).at(k) > maxValue) {
                    maxValue = histogram.at(i).at(j).at(k);
                }
            }
        }
    }

    // iterate over the cells in histogram
    for (int i = 0; i < histogram.size(); i++) {
        for(int j = 0; j < histogram.at(i).size(); j++) {
            int colorMagnitude = static_cast<int>(histogram.at(i).at(j).at(0) / maxValue * 255);
            for (int k = 0; k < histogram.at(i).at(j).size(); k++) {
                // if the value is greater than the threshold, set the mask to 255
                if (histogram.at(i).at(j).at(k) > maxValue * 0.1) {
                    vectorMask.at<uchar>(i * pixelsPerCell, j * pixelsPerCell) = 255;
                }
            }
        }
    }

    return vectorMask;
}

void HOG::writeToFile(std::string filename, std::vector<float> &vector) {
    // Write the descriptor to a file
    std::ofstream file;
    file.open(filename);
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
    cv::imshow("img1", image1);
    cv::waitKey(0);
}

