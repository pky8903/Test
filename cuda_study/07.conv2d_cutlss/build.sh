mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_C_COMPILER=gcc-10
make -j12
