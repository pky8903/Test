rm -rf build

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release:q .. -DCMAKE_CXX_COMPILER=$(which g++-10) -DCMAKE_C_COMPILER=$(which gcc-10)
make
