rm -rf build

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(which g++-10) -DCMAKE_C_COMPILER=$(which gcc-10) -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
