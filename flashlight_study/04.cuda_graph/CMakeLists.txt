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

# ─────────────── NVTX 처리 (CUDA 11 vs 12) ───────────────
# CMake ≥3.18 의 FindCUDAToolkit 모듈은 `CUDA::nvToolsExt` 임포트 타깃을
# (라이브러리가 존재할 때)만 만들어 줍니다.
if(TARGET CUDA::nvToolsExt)
    message(STATUS "Linking CUDA::nvToolsExt (CUDA ≤11.x)")
    target_link_libraries(my_app PRIVATE CUDA::nvToolsExt)
else()
    message(STATUS "CUDA::nvToolsExt target not found → CUDA 12.x header-only NVTX")
    # 헤더는 이미 cuda include 경로에 있으므로 추가 작업 불필요
endif()
