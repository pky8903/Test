/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx ./build/my_app 1 1 256 256 20
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx ./build/my_app 4 1 256 256 20
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx ./build/my_app 8 1 256 256 20
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx ./build/my_app 16 1 256 256 20
