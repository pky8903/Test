rm -rf *.nsys-rep
rm -rf *.sqlite

nsys profile --trace=cuda,nvtx ./build/program
nsys stats report1.nsys-rep
