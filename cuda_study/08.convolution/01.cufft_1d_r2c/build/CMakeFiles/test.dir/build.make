# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/main.cu.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/main.cu.o: ../main.cu
CMakeFiles/test.dir/main.cu.o: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test.dir/main.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test.dir/main.cu.o -MF CMakeFiles/test.dir/main.cu.o.d -x cu -c /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/main.cu -o CMakeFiles/test.dir/main.cu.o

CMakeFiles/test.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/main.cu.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/main.cu.o
test: CMakeFiles/test.dir/build.make
test: /usr/lib/x86_64-linux-gnu/libpython3.10.so
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build /home/kyp/Workspace/00.test/Test/cuda_study/08.convolution/01.cufft_1d_r2c/build/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend

