include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(graph500 main.cpp)

add_library(graphlib graph.h graph.cpp hello.cu)

set_target_properties(graphlib PROPERTIES CUDA_ARCHITECTURES "60;70")

# Specify include directories for this target
target_include_directories(graphlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(graph500 ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
if(MPI_COMPILE_FLAGS)
  set_target_properties(graph500 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(graph500 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
