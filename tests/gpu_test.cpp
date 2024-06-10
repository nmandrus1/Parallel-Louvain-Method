#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <utility>
#include "graph.h"
#include <mpi.h>


class MPIEnvironment : public ::testing::Environment {
public:
  int *argc;
  char ***argv;

  MPIEnvironment(int *_argc, char ***_argv) : argc(_argc), argv(_argv) {}

  virtual void SetUp() {
    int mpiError = MPI_Init(argc, argv);
    ASSERT_FALSE(mpiError);
  }

  virtual void TearDown() {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
};


TEST(CudaTests, CudaEdgeList) {
  auto list = generate_kronecker_list_cuda(5, 16, 123);
}

TEST(CudaTests, GraphFromCuda) {
  Graph g;
  g.from_kronecker_cuda(2, 1, 123);
  g.print_graph();
}

TEST(CudaTests, BigGraphFromCuda) {
  Graph g;
  g.from_kronecker_cuda(5, 8, 123);
  g.print_graph();
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
  return RUN_ALL_TESTS();
}
