cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# set(CMAKE_CXX_COMPILER mpicxx)
project(Graph500 LANGUAGES CXX)

# Check if GPTL environment variable is set
if(NOT DEFINED ENV{GPTL})
    message(FATAL_ERROR "The GPTL environment variable is not set. Please define it before configuring this project.")
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

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

# Centralize all binary output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

option(ENABLE_PROFILING "Enable profiling with GPTL" OFF)

add_subdirectory(src)
add_subdirectory(tests)

# Custom target for cleaning the build directory
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/build"
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaned up the build directory.")
