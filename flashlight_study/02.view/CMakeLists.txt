cmake_minimum_required(VERSION 3.18)
project(MyFlashlightApp LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(flashlight REQUIRED)
find_package(ArrayFire REQUIRED)
find_library(CUDNN_LIB cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib/x86_64-linux-gnu)

add_executable(my_app main.cc)
target_link_libraries(my_app 
    PRIVATE 
    flashlight
    ArrayFire::af
    ArrayFire::afcuda
    CUDA::cudart
    ${CUDNN_LIB}
)

