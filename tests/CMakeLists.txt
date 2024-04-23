include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(runUnitTests test_main.cpp)

add_executable(mpiUnitTests mpi_test.cpp)

# target_link_libraries(runUnitTests gtest gtest_main)
target_link_libraries(runUnitTests PRIVATE gtest gtest_main graphlib ${MPI_LIBRARIES})
target_link_libraries(mpiUnitTests PRIVATE gtest gtest_main graphlib ${MPI_LIBRARIES})

add_test(NAME that_test COMMAND runUnitTests)

add_custom_target(mpi_test
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpiUnitTests> ${MPIEXEC_POSTFLAGS}
    DEPENDS mpi_test
    COMMENT "Running specific_test using MPI"
    USES_TERMINAL
)


add_custom_target(basic_test
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:runUnitTests> ${MPIEXEC_POSTFLAGS}
    DEPENDS basic_test
    COMMENT "Running unit tests using MPI"
    USES_TERMINAL
)
