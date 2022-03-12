#ifndef TEST_HPP
#define TEST_HPP

#include "iostream"
#include "string.h"
#include "opencv2/opencv.hpp"

class TestClass{
	public:
		TestClass();
		~TestClass();
		void printHelloWorld(std::string input);
		void openCameraStream();
		
	private:
		std::string helloWorld;   
};

#endif