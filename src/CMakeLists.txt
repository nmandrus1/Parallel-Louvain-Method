include_directories(${MPI_INCLUDE_PATH})

# Include GPTL paths
if(ENABLE_PROFILING)
  include_directories($ENV{GPTL}/include)
  link_directories($ENV{GPTL}/lib)
  add_definitions(-DPROFILE_FNS)
endif()

add_executable(community main_distributed_community.cpp)
add_executable(comm main_community.cpp)

add_library(graphlib graph.cpp util.cpp community.cpp distcommunity.cpp)

if(ENABLE_PROFILING)
  target_link_libraries(graphlib -L$ENV{GPTL}/lib -lgptl -lunwind)
  target_compile_options(graphlib PRIVATE -rdynamic)
endif()

# Specify include directories for this target
target_include_directories(graphlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link with MPI libraries and GPTL libraries
target_link_libraries(community ${MPI_LIBRARIES} graphlib)
target_link_libraries(comm ${MPI_LIBRARIES} graphlib)

# Adding the GPTL libraries
if(ENABLE_PROFILING)
  target_link_libraries(community -L$ENV{GPTL}/lib -lgptl -lunwind)
  target_compile_options(community PRIVATE -rdynamic)
endif()

# Apply compile options to enable instrumentation and rdynamic for GPTL

if(MPI_COMPILE_FLAGS)
  set_target_properties(community PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(community PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

