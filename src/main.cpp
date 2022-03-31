#include "test.hpp"
#include "hog.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>

#define DATASET_FOLDER "/home/ernests/Documents/Personal/universityNotes/Semester2/imgProc/MiniProject/facialrecognition/dataset/INRIAPerson/64x128"

void display_superimposed(const cv::Mat& A, const cv::Mat& B, const std::string& name) {
    cv::Mat superimposed;
    cv::addWeighted(A, 0.5, B, 0.5, 0.0, superimposed);
    imshow(name, superimposed);
	cv::waitKey(0);
}

cv::Mat custom_normalization(const cv::Mat& src) {
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::Mat dst = src * 200 / (max - min) + 128;
    dst.convertTo(dst, CV_8U);
    return dst;
}

int main(int argc, char** argv){

	std::cout << std::filesystem::current_path() << std::endl;

	std::string buf(DATASET_FOLDER);
	buf.append("/crop1_64128.png");
	HOG hog(buf);
	hog.process();

	display_superimposed(hog.inputImgGray, hog.getVectorMask(), "inputImgRGB");
	
	return 0;
}