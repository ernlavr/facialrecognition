#include "test.hpp"

int main(int argc, char** argv){

	TestClass hello;
    
	hello.printHelloWorld("Hello World!");
    hello.openCameraStream();
	return 0;
}