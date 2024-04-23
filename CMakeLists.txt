cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_COMPILER mpicxx)
project(Graph500 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

set(CMAKE_BUILD_TYPE Debug)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(MPI REQUIRED)
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")

enable_testing()

include(FetchContent)

# Check if the policy exists before setting it
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
add_subdirectory(tests)
