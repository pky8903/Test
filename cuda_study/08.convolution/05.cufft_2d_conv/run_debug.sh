#./build/test --width 1024 --mode 0
#./build/test --width 1024 --mode 1
#./build/test --width 2048 --mode 0
#./build/test --width 2048 --mode 1
#./build/test --width 4096 --mode 0
#./build/test --width 4096 --mode 1
#./build/test --width 8192 --mode 0
#./build/test --width 8192 --mode 1
#./build/test --width 16384 --mode 0
#./build/test --width 16384 --mode 1

gdb --args ./build_debug/test --width 16384 --mode 0
