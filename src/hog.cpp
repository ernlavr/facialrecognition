#include "hog.hpp"

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

void HOG::process() {
    cellsY = static_cast<int>(inputImgGray.rows / pixelsPerCell);
    cellsX = static_cast<int>(inputImgGray.cols / pixelsPerCell);
    histogram.resize(cellsY);

    // process an image block by block
    for (int y = 0; y < cellsY; y += 1) {
        for (int x = 0; x < cellsX; x += 1) {
            // fetch the cell
            cv::Mat cell = inputImgGray(cv::Rect(x * pixelsPerCell, y * pixelsPerCell, pixelsPerCell, pixelsPerCell));
            std::vector<float> dstHist (numBins, 0);
            processCell(cell, dstHist);

            // store the histogram in the histogram vector
            histogram.at(y).push_back(dstHist);
        }
    }

    // Perform 16x16 block normalization on the histogram
    L2blockNormalization();

    auto e = cv::HOGDescriptor(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), numBins);
    std::vector<float> desc;
    std::vector<cv::Point> locations;
    e.compute(inputImgGray, desc, cv::Size(1, 1), cv::Size(blockStride, blockStride), locations);
    std::cout << "size of desc: " << desc.size() << std::endl;
    std::cout << "dogshit" << std::endl;
} 

float computeL2norm(std::vector<float> input) {
    float sum = 0;
    for (auto i : input) {
        sum += i * i;
    }
    return sqrt(sum);
}

void HOG::L2blockNormalization() {
    // Loop over the histogram variable with a 16x16 sliding window and normalize the values
    std::vector<std::vector<std::vector<float>>> normalizedHistogram;
    normalizedHistogram.resize(cellsY - 1); // 16x16 block will have 1 less row than the original image
    for (int y = 0; y < cellsY - 1; y += 1) { 
        normalizedHistogram.at(y).resize(cellsX - 1); // 16x16 block will have 1 less column than the original image
        for (int x = 0; x < cellsX - 1; x += 1) {
            // fetch the cells
            auto cell1 = histogram.at(y).at(x);
            auto cell2 = histogram.at(y).at(x + 1);
            auto cell3 = histogram.at(y + 1).at(x);
            auto cell4 = histogram.at(y + 1).at(x + 1);
            
            // Merge the vectors into 1
            std::vector<float> mergedCells;   

            // compute the L2 norm of each cell
            auto norm1 = computeL2norm(cell1);
            auto norm2 = computeL2norm(cell2);
            auto norm3 = computeL2norm(cell3);
            auto norm4 = computeL2norm(cell4);

            // compute the average of the norms
            std::vector <float> norms = {norm1, norm2, norm3, norm4};
            std::vector<std::vector<float>> cellVec = {cell1, cell2, cell3, cell4};
            
            // Loop over normVector and for each cell normalize the values using the correct coefficient from the norms array
            for (int i = 0; i < cellVec.size(); i++) {
                auto cellVecSize = cellVec.at(i).size();
                for (int j = 0; j < cellVecSize; j++) {
                    mergedCells.push_back(cellVec.at(i).at(j) / norms.at(i));
                }
            }
            normalizedHistogram.at(y).at(x) = mergedCells;
        }
    }
    std::cout << "dogshit" << std::endl;

    // Stretch the normalizedHistogram into a single, 1d vector
    for (int y = 0; y < cellsY - 1; y += 1) {
        for (int x = 0; x < cellsX - 1; x += 1) {
            auto cell = normalizedHistogram.at(y).at(x);
            for (auto i : cell) {
                finalDescriptor.push_back(i);
            }
        }
    }

    std::cout << "length of catpiss: " << finalDescriptor.size() << std::endl;
}

void HOG::processCell(cv::Mat &cell, std::vector<float> &dstHist) {
    // Compute a gradient using sobel operation and [-1, 0, 1] filter
    cv::Mat filterX = (cv::Mat_<char>(1, 3) << -1, 0, 1);
    cv::Mat filterY = (cv::Mat_<char>(3, 1) << -1, 0, 1);

    cv::Mat gradX, gradY, dstMag, dstAngle;
    cv::filter2D(cell, gradX, CV_32F, filterX);
    cv::filter2D(cell, gradY, CV_32F, filterY);
    cv::magnitude(gradX, gradY, dstMag);
    cv::phase(gradX, gradY, dstAngle, true);

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

void HOG::convertToUnsignedAngles(cv::Mat &srcMat) {
    for (int i = 0; i < srcMat.rows; i++) {
        for (int j = 0; j < srcMat.cols; j++) {
            if (srcMat.at<float>(i, j) > 180) {
                srcMat.at<float>(i, j) = srcMat.at<float>(i, j) - 180;
            }
        }
    }
}

void HOG::printMatrix(cv::Mat &mat) {
    // Print the matrix
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            std::cout << mat.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
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

void HOG::squareRootColorGamma() {
    // Square root of each color channel
    auto tmp = inputImgGray.clone();
    tmp.convertTo(tmp, CV_32F);
    cv::sqrt(tmp, output);
    output.convertTo(output, CV_8U);
    displayImage(output);
}
