// Implemented by Ernests Lavrinovics as part of Image Processing and Computer Vision course in Aalborg University, Medialogy 2022.

#include "test.hpp"
#include "hog.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>

#define DATASET_FOLDER "/home/ernests/Documents/Personal/universityNotes/Semester2/imgProc/MiniProject/facialrecognition/dataset/INRIAPerson/64x128"

int main(int argc, char** argv){
    bool verbose = false;
    if(argc > 1) {
        auto arg1 = std::string(argv[1]);
        if(arg1 == "-v") {
            verbose = true;
        }
    }

	std::string img(DATASET_FOLDER);
	img.append("/crop1_64128.png");
	HOG hog(img, verbose);
	hog.process();
    
    if(verbose) {
        hog.computeAndPrintOpenCV();
    }
	return 0;
}