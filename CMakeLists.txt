cmake_minimum_required(VERSION 3.10) 
set(CMAKE_CXX_STANDARD 17)
PROJECT (HOG)
find_package(OpenCV REQUIRED )
set( NAME_SRC
    src/main.cpp 
    src/hog.cpp
)

set( NAME_HEADERS       
     include/hog.hpp
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(HOG ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries(HOG ${OpenCV_LIBS})