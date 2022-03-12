#include "main.hpp"

TestClass::TestClass() {
    // Constructor
}

TestClass::~TestClass() {
    // Destructor
}

void TestClass::printHelloWorld(std::string input) {
    std::cout << input << std::endl;
}

int main(int argc, char** argv){

	TestClass hello;

	hello.printHelloWorld("Hello World!");

	exit(0);
}