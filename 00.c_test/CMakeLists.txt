cmake_minimum_required(VERSION 3.11)

project(
	TEST
	VERSION 0.1
	DESCRIPTION "for test"
	LANGUAGES CXX
	)

add_subdirectory(lib)

add_executable (program test.cc foo.cc)

target_compile_options(program PUBLIC -Wall -Werror)

target_link_libraries(program shape)

target_include_directories(program PUBLIC ${CMAKE_SOURCE_DIR}/includes)
