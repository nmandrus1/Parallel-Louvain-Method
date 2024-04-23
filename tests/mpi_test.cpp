
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

TEST(MPITests, ParallelBfsTest) {
  // Create kronecker graph
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Graph g = Graph::from_kronecker(2, 2, 123);
  auto parents = g.parallel_top_down_bfs(0);

  if(rank == 0) {
    std::vector<int> expected = {0, -1, 0, 2};
    ASSERT_EQ(parents, expected);
  }

  if(rank == 1) {
    std::vector<int> expected = {-1, -1, -1, -1};
    ASSERT_EQ(parents, expected);
  }

  if(rank == 2) {
    std::vector<int> expected = {0, -1, -1, -1};
    ASSERT_EQ(parents, expected);
  }

  if(rank == 3) {
    std::vector<int> expected = {6, -1, 0, 6};
    ASSERT_EQ(parents, expected);
  }
}

TEST(MPITests, VertexOwnership) {
  // Create kronecker graph
  int rank, scale = 2;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // kronecker of scale 2 -> 4 vertices per graph
  Graph g = Graph::from_kronecker(scale, 1, 123);

  if(rank == 0) {
    ASSERT_TRUE(g.contains(0));
    ASSERT_TRUE(g.contains(1));
    ASSERT_FALSE(g.contains(9));
  }

  if(rank == 3) {
    ASSERT_TRUE(g.contains(7));
    ASSERT_FALSE(g.contains(3));
  }
  
}


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
  return RUN_ALL_TESTS();
}
