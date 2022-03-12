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
		
	private:
		std::string helloWorld;   
		std::string hi;
};

#endif