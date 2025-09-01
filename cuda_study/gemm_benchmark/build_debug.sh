rm -rf build_debug

mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
