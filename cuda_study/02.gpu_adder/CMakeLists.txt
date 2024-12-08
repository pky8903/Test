cmake_minimum_required(VERSION 3.11)

project(
	TEST
	VERSION 0.1
	DESCRIPTION "for test"
	LANGUAGES CXX
    CUDA
	)

add_executable (program main.cu DS_timer.cpp)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

target_include_directories(program PUBLIC ${CUDA_INCLUDE_DIRS})

set_target_properties(program PROPERTIES CUDA_STANDARD 11)
