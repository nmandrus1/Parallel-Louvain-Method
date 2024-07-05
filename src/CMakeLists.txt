# include_directories(${MPI_INCLUDE_PATH})

# #include GPTL paths


# # add_executable(graph500 main.cpp)
# add_executable(graph500 main_distributed_community.cpp)
# add_executable(comm main_community.cpp)

# add_library(graphlib graph.h graph.cpp util.h util.cpp community.h community.cpp)

# set_target_properties(graphlib PROPERTIES CUDA_ARCHITECTURES "60;70")

# # Specify include directories for this target
# target_include_directories(graphlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# target_link_libraries(graph500 ${MPI_LIBRARIES} ${CUDA_LIBRARIES} graphlib)
# target_link_libraries(comm ${MPI_LIBRARIES} ${CUDA_LIBRARIES} graphlib)

# if(MPI_COMPILE_FLAGS)
#   set_target_properties(graph500 PROPERTIES
#     COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()

# if(MPI_LINK_FLAGS)
#   set_target_properties(graph500 PROPERTIES
#     LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()

include_directories(${MPI_INCLUDE_PATH})

# Include GPTL paths
if(ENABLE_PROFILING)
  include_directories($ENV{GPTL}/include)
  link_directories($ENV{GPTL}/lib)
  add_definitions(-DPROFILE_FNS)
endif()

add_executable(graph500 main_distributed_community.cpp)
add_executable(comm main_community.cpp)

add_library(graphlib graph.h graph.cpp util.h util.cpp community.h community.cpp)

if(ENABLE_PROFILING)
  target_link_libraries(graphlib -L$ENV{GPTL}/lib -lgptl -lunwind)
  target_compile_options(graphlib PRIVATE -rdynamic)
endif()

# Specify include directories for this target
target_include_directories(graphlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link with MPI libraries and GPTL libraries
target_link_libraries(graph500 ${MPI_LIBRARIES} graphlib)
target_link_libraries(comm ${MPI_LIBRARIES} graphlib)

# Adding the GPTL libraries
if(ENABLE_PROFILING)
  target_link_libraries(graph500 -L$ENV{GPTL}/lib -lgptl -lunwind)
  target_compile_options(graph500 PRIVATE -rdynamic)
endif()

# Apply compile options to enable instrumentation and rdynamic for GPTL

if(MPI_COMPILE_FLAGS)
  set_target_properties(graph500 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(graph500 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

