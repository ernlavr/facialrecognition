# Implementation of Histogram of Oriented Gradients
## Project Structure
`src` - Source files of the implementation
`include` - C++ header files
`py_analysis` - Python scripts for data analysis, see `py_analysis/readme.md`
`output` - output of the built binary
`build` - contains built binary
`dataset` - contains INRIA image dataset

## To Use
1. Run the binary `bin/build/HOG` with the following flags \
`-v` to dump the output in `output` directory \
`-p` to profile this against OpenCV implementation

## Note
Tested on Ubuntu LTS 20.04, built using CMake and gcc