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

TEST(BasicGraphTests, AddEdge) {
  Graph g(5); 
  g.add_edge(1, 2);

  auto list = g.get_edges(1);
  ASSERT_EQ(list[0], 2);

  list = g.get_edges(2);
  ASSERT_EQ(list[0], 1);
}

TEST(BasicGraphTests, FromEdgeList) {
  std::vector<std::pair<int, int>> edges = { {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto list = g.get_edges(0);
  std::vector<int> expected = {1, 2};
  ASSERT_EQ(list, expected);

  list = g.get_edges(1);
  expected = {0, 2, 3};
  ASSERT_EQ(list, expected);

  list = g.get_edges(2);
  expected = {0, 1, 4};
  ASSERT_EQ(list, expected);

  list = g.get_edges(3);
  expected = {1};
  ASSERT_EQ(list, expected);

  list = g.get_edges(4);
  expected = {2};
  ASSERT_EQ(list, expected);
}

TEST(BasicGraphTests, TopDownBfsTest) {
  std::vector<std::pair<int, int>> edges = { {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto parent = g.top_down_bfs(0);
  std::vector<int> expected = {0, 0, 0, 1, 2};
  ASSERT_EQ(parent, expected);
}

TEST(BasicGraphTests, BtmDownBfsTest) {
  std::vector<std::pair<int, int>> edges = { {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto parent = g.btm_down_bfs(0);
  std::vector<int> expected = {0, 0, 0, 1, 2};
  ASSERT_EQ(parent, expected);
}

TEST(KroneckerTest, BasicKroneckerEdgeList) {
  auto list = generate_kronecker_list(5, 16, 123);
}

TEST(CudaTests, CudaEdgeList) {
  auto list = generate_kronecker_list_cuda(5, 16, 123);
}

TEST(CudaTests, GraphFromCuda) {
  Graph g = Graph::from_kronecker_cuda(2, 1, 123);
  g.print_graph();
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
  return RUN_ALL_TESTS();
}
