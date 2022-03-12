#include "test.hpp"

TestClass::TestClass() {
    // Constructor
}

TestClass::~TestClass() {
    // Destructor
}

void TestClass::printHelloWorld(std::string input) {
    std::cout << input << std::endl;
}

void TestClass::openCameraStream() {
    cv::VideoCapture cap(0, cv::CAP_ANY);

    cap.set(3, 640);
    cap.set(4, 480);
    
    if(!cap.isOpened()) {
        std::cout << "Change the camera port number!" << std::endl;
        return;
    }
    
    cv::Mat frame;
    cv::Mat dst;
    while(true) {
        cap >> frame;
        cv::imshow("frame", frame);

        if(cv::waitKey(1) == 27) {
            return;
        }
    }
}