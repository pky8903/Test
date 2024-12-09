rm -rf *.nsys-rep
rm -rf *.sqlite

nsys profile --stats=true --trace=cuda,nvtx -o report ./build/program
nsys stats report1.nsys-rep
