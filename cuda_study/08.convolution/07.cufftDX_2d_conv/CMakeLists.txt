cmake_minimum_required(VERSION 3.11)

project(
	TEST
	VERSION 0.1
	DESCRIPTION "for test"
	LANGUAGES CXX CUDA
	)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CUTLASS_DIR "/home/kyp/Workspace/01.libraries/cutlass")
set(CUFFTDX_DIR "/home/kyp/Workspace/01.libraries/nvidia-mathdx-25.01.1/nvidia/mathdx/25.01")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

add_executable (test main.cu)

target_include_directories(test 
    PUBLIC 
    ${CUDA_INCLUDE_DIRS}
    /usr/local/cuda/include 
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ./matplotlib-cpp 
    ${CUFFTDX_DIR}/include
    ${CUTLASS_DIR}/include
)

#include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

target_link_libraries(test
    ${Python3_LIBRARIES}
    cufft
)

#set_target_properties(test PROPERTIES CUDA_STANDARD 11)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES 86)
