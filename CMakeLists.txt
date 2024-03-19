cmake_minimum_required(VERSION 3.12)

project(MyMPICUDAProject LANGUAGES CXX CUDA)

# Find the MPI package
find_package(MPI REQUIRED)

# Ensure that the CUDA toolkit is available
find_package(CUDA REQUIRED)

# Add the include directories for MPI and CUDA
include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files and headers
include_directories(include)
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# Define the executable to be built from the specified sources
add_executable(MyMPICUDAExec ${SOURCES})

# Link the MPI and CUDA libraries
target_link_libraries(MyMPICUDAExec ${MPI_LIBRARIES} ${CUDA_LIBRARIES})

# If using an MPI compiler, specify the compiler to CMake
if(MPI_COMPILE_FLAGS)
  set_target_properties(MyMPICUDAExec PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(MyMPICUDAExec PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
