include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(runUnitTests test_main.cpp)

# target_link_libraries(runUnitTests gtest gtest_main)
target_link_libraries(runUnitTests PRIVATE gtest gtest_main graphlib)

add_test(NAME that_test COMMAND runUnitTests)
