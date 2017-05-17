# cudaImageCompare.
compare image "Source" and "Pattern"<br>
Find the most similar part in source image and plot it on source image

# System Requirement
CUDA<br>
OpenCV<br>
source image file<br>
pattern image file

# Compile
vim CMakeLists.txt

cmake_minimum_required(VERSION 2.6)<br>
project(HW4)<br>
find_package(CUDA REQUIRED)<br>
find_package(OpenCV REQUIRED)<br>
set(CUDA_64_BIT_DEVICE_CODE ON)<br>
cuda_add_executable( HW4 HW4.cu )<br>
target_link_libraries( HW4 ${OpenCV_LIBS} )<br>

cmake<br>
make

# Usage
./HW4 sourceImg patternImg
