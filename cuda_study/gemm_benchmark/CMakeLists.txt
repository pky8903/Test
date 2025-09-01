cmake_minimum_required(VERSION 3.11)

project(
    TEST
    VERSION 0.1
    DESCRIPTION "for test"
    LANGUAGES CXX CUDA
)

# CUDA (legacy) finder
find_package(CUDA REQUIRED)

# ---- CUDA language/standard ----
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---- Paths (adjust to your env) ----
set(CUTLASS_DIR "/home/kyp/Workspace/01.libraries/cutlass")
set(CUFFTDX_DIR "/home/kyp/Workspace/01.libraries/nvidia-mathdx-25.01.1/nvidia/mathdx/25.01")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

add_executable(test main.cu)

# ---- Includes ----
target_include_directories(test
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUTLASS_DIR}/include
    ${CUFFTDX_DIR}/include
)

# ---- Link dirs & libraries (legacy style) ----
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(test
    PRIVATE
    cudart           
    cublas            # cuBLAS
    cublasLt          # cuBLASLt (Matmul API)
    cufft           
)

# ---- SM architecture ----
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES 80)

