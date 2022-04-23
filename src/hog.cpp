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
  // Clear containers
  descriptor.clear();
  histogram.clear();
  magnitude.release();
  angles.release();

  cellsY = static_cast<int>(inputImgGray.rows / pixelsPerCell);
  cellsX = static_cast<int>(inputImgGray.cols / pixelsPerCell);
  histogram.resize(cellsY);

  gradientComputation();

  // process an image block by block
  for (int y = 0; y < cellsY; y += 1) {
    for (int x = 0; x < cellsX; x += 1) {
      // Fetch the cell slices of the angle, magnitude and image
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

void HOG::initializeOpenCVHOG() {
  auto winSize = cv::Size(64, 128);
  auto blockSize = cv::Size(16, 16);
  auto stride = cv::Size(8, 8);
  auto cellSize = cv::Size(8, 8);
  auto padding = cv::Size(0, 0);
  opencvHOG = cv::HOGDescriptor(winSize, blockSize, stride, cellSize, numBins);
}

void HOG::computeAndWriteOpenCVHog() {
  std::vector<float> desc;
  std::vector<cv::Point> locations;
  // Compute
  opencvHOG.compute(inputImgGray, desc, cv::Size(8, 8), cv::Size(0, 0), locations);

  // Write to .txt file
  if(verbose) {
    writeToFile("HOG_openCV.txt", desc);
  }
}

void HOG::L2norm(std::vector<std::vector<float>> &input) {
  // L2 Divider
  float sum = 0;
  for(auto &i : input) { // for each cell
    for(auto &j : i) { // for each histogram
      sum += j * j;
    }
  }

  // Divide each cell in input
  for(auto &i : input) {
    for(auto &j : i) {
      j /= sqrt(sum);
    }
  }
}

void HOG::clipNumber(std::vector<std::vector<float>> &input) {
  // Clip each cell to 0.2
  for(auto &i : input) { // for each cell
    for(auto &j : i) { // for each histogram
      if(j > 0.2) {
        j = 0.2;
      }
      else if(j < 0) {
        j = 0;
      }
    }
  }
}

void HOG::L2blockNormalization() {
  for (int y = 0; y < cellsY - 1; y += 1) { 
    for (int x = 0; x < cellsX - 1; x += 1) {
      std::vector<std::vector<float>> window;
      // Fetch the 2 by 2 window of cells and its divisor
      for (int width = y; width < y + cellsPerWindow_H; ++width) {
        for (int height = x; height < x + cellsPerWindow_H; ++height) {
          auto cell = histogram.at(width).at(height);
          window.push_back(cell);
        }
      }

      L2norm(window);
      clipNumber(window);
      L2norm(window);
      
      // Add the normalized values to the final 1D descriptor
      for(auto i : window) {
        for(auto j : i) {
          descriptor.push_back(j);
        }
      }
    }
  }
}

void HOG::processCell(const cv::Mat &cell, const cv::Mat &cellMagn, cv::Mat &cellAngle, std::vector<float> &dstHist) {
  // Convert dstAngle from unsigned to signed if a value is larger than 180
  convToUnsignAngl(cellAngle);

  // Calculate the histogram of gradient
  int binSize = 180 / numBins;
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

void HOG::convToUnsignAngl(cv::Mat &srcAngles) {
  for (int i = 0; i < srcAngles.rows; i++) {
    for (int j = 0; j < srcAngles.cols; j++) {
      if (srcAngles.at<float>(i, j) >= 180) {
        srcAngles.at<float>(i, j) -= 180;
      }
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