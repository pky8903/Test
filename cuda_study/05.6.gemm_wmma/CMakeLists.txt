cmake_minimum_required(VERSION 3.11)

project(
	TEST
	VERSION 0.1
	DESCRIPTION "for test"
	LANGUAGES CUDA CXX
    CUDA
	)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CUTLASS_DIR ${CUTLASS_PATH})
if (NOT CUTLASS_DIR)
    set(CUTLASS_DIR "/home/kyp/Workspace/01.libraries/cutlass")
endif()

include_directories(${CUTLASS_DIR}/include)

include_directories(${CUTLASS_DIR}/tools/util/include)

add_executable (program main.cu DS_timer.cpp)

target_compile_options(program PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_70,code=sm_70>)
target_include_directories(program PUBLIC ${CUDA_INCLUDE_DIRS})

set_target_properties(program PROPERTIES CUDA_STANDARD 17)

target_link_libraries(program PRIVATE cudart)

