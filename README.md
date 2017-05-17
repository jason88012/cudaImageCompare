# cudaImageCompare.
compare image "Source" and "Pattern"
Find the most similar part in source image and plot it on source image

#Require
CUDA
OpenCV
source image file
pattern image file

#Compile
vim CMakeLists.txt

cmake_minimum_required(VERSION 2.6)
project(HW4)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_64_BIT_DEVICE_CODE ON)
cuda_add_executable( HW4 HW4.cu )
target_link_libraries( HW4 ${OpenCV_LIBS} )

cmake .
make

#Usage
./HW4 sourceImg patternImg
