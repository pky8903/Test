rm -rf build

mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
