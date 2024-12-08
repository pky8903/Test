rm -rf build_debug

mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$(which g++-10) -DCMAKE_C_COMPILER=$(which gcc-10) -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
