// Implemented by Ernests Lavrinovics as part of Image Processing and Computer Vision course in Aalborg University, Medialogy 2022.

#include "test.hpp"
#include "hog.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>

#define DATASET_FOLDER "/home/ernests/Documents/Personal/universityNotes/Semester2/imgProc/MiniProject/facialrecognition/dataset/INRIAPerson/64x128"

void profileFunction(std::function<void()> func, int iterations, std::string name) {
    double totalTimeTaken = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        totalTimeTaken += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    // Convert timePerIteration to nanoseconds and print it
    std::cout << name << " Time per iteration: " << totalTimeTaken / iterations << " nanoseconds" << std::endl;

    // Convert total time taken to seconds and store it in a new variable and print the new variable
    double totalTimeTakenInSeconds = totalTimeTaken / 1000000000.0;
    std::cout << name << " Total time taken: " << totalTimeTakenInSeconds << " seconds" << std::endl;
    std::cout << name << " Total time taken: " << totalTimeTaken << " ns" << std::endl;
}

int main(int argc, char** argv){
    bool verbose = false;
    bool profile = false;
    if(argc > 1) {
        auto arg1 = std::string(argv[1]);
        if(arg1 == "-v") {
            verbose = true;
        }
        if(arg1 == "-p") {
            profile = true;
        }
    }

	std::string img(DATASET_FOLDER);
	img.append("/crop1_64128.png");
	HOG hog(img, verbose);
    hog.process();

    if(profile) {
        hog.initializeOpenCVHOG();
        std::cout << "Profiling..." << std::endl;
        // Measure the time it takes to perform 1 iteration of hog.process() 100k times
        profileFunction(std::bind(&HOG::computeAndWriteOpenCVHog, &hog), 100000, "OpenCV");
        profileFunction(std::bind(&HOG::process, &hog), 100000, "Mine");
    }
    
    if(verbose) {
        hog.initializeOpenCVHOG();
        hog.computeAndWriteOpenCVHog();
    }
	return 0;
}