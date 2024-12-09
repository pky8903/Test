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
CMAKE_SOURCE_DIR = /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build

# Include any dependencies generated for this target.
include CMakeFiles/program.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/program.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/program.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/program.dir/flags.make

CMakeFiles/program.dir/main.cu: CMakeFiles/program.dir/flags.make
CMakeFiles/program.dir/main.cu: ../main.cu
CMakeFiles/program.dir/main.cu: CMakeFiles/program.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/program.dir/main.cu"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/program.dir/main.cu -MF CMakeFiles/program.dir/main.cu.d -x cu -c /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/main.cu -o CMakeFiles/program.dir/main.cu

CMakeFiles/program.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/program.dir/main.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/program.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/program.dir/main.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/program.dir/DS_timer.cpp.o: CMakeFiles/program.dir/flags.make
CMakeFiles/program.dir/DS_timer.cpp.o: ../DS_timer.cpp
CMakeFiles/program.dir/DS_timer.cpp.o: CMakeFiles/program.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/program.dir/DS_timer.cpp.o"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/program.dir/DS_timer.cpp.o -MF CMakeFiles/program.dir/DS_timer.cpp.o.d -o CMakeFiles/program.dir/DS_timer.cpp.o -c /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/DS_timer.cpp

CMakeFiles/program.dir/DS_timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/program.dir/DS_timer.cpp.i"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/DS_timer.cpp > CMakeFiles/program.dir/DS_timer.cpp.i

CMakeFiles/program.dir/DS_timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/program.dir/DS_timer.cpp.s"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/DS_timer.cpp -o CMakeFiles/program.dir/DS_timer.cpp.s

# Object files for target program
program_OBJECTS = \
"CMakeFiles/program.dir/main.cu" \
"CMakeFiles/program.dir/DS_timer.cpp.o"

# External object files for target program
program_EXTERNAL_OBJECTS =

program: CMakeFiles/program.dir/main.cu
program: CMakeFiles/program.dir/DS_timer.cpp.o
program: CMakeFiles/program.dir/build.make
program: CMakeFiles/program.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable program"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/program.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/program.dir/build: program
.PHONY : CMakeFiles/program.dir/build

CMakeFiles/program.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/program.dir/cmake_clean.cmake
.PHONY : CMakeFiles/program.dir/clean

CMakeFiles/program.dir/depend:
	cd /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build /home/kyp/Workspace/00.test/Test/cuda_study/compute_ai_batch/build/CMakeFiles/program.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/program.dir/depend
