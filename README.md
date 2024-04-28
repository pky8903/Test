# Test
Test in my linux server

Test result: 

Test Eigen 1
test1 duration: 0.606307 s (matmul)
test2 duration: 1.75874 s (for-loop)
test3 duration: 3.45413 s

Test Eigen 2 (with MKL)
test1 duration: 0.23199 s (matmul)
test2 duration: 1.7653 s (for-loop)
test3 duration: 3.33907 s

Test arma 1 (with openblas and lapack)
test1 duration: 0.0236908 s (matmul)
test2 duration: 0.252405 s (for-loop)
test3 duration: 0.293178 s

Test arma 2 (with ARAM_DONT_USE_WRAPPER)
test1 duration: 0.0586106 s
test2 duration: 0.214528 s
test3 duration: 0.284397 s

Test arma 3 (with -march=native flags)
test1 duration: 0.0449793 s
test2 duration: 0.254239 s
test3 duration: 0.312113 s

* Test arma 1
	ldd binary
	linux-vdso.so.1 (0x00007ffd355e0000)
	libarmadillo.so.12 => /lib/x86_64-linux-gnu/libarmadillo.so.12 (0x00007f5e56ceb000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f5e56a00000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f5e56ccb000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f5e56600000)
	libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0 (0x00007f5e541b0000)
	liblapack.so.3 => /lib/x86_64-linux-gnu/liblapack.so.3 (0x00007f5e53a00000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f5e56919000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f5e56d20000)
	libgfortran.so.5 => /lib/x86_64-linux-gnu/libgfortran.so.5 (0x00007f5e53600000)
	libquadmath.so.0 => /lib/x86_64-linux-gnu/libquadmath.so.0 (0x00007f5e56c81000)
	
