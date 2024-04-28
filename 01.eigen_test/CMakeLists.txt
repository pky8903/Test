cmake_minimum_required(VERSION 3.11)

project(
	TEST
	VERSION 0.1
	DESCRIPTION "for test"
	LANGUAGES CXX
	)

set(MKLROOT "/opt/intel/oneapi/mkl/latest")
link_directories(${MKLROOT}/lib/intel64)

find_package(Threads REQUIRED)
set(MKL_LIBRARIES mkl_rt pthread)

add_executable (program test.cc)

target_compile_options(program PUBLIC -Wall -Werror)

include_directories(${MKLROOT}/include)

target_link_libraries(program ${MKL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
