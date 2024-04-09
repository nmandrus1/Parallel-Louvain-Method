#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <utility>
#include "graph.h"

TEST(BasicGraphTests, AddEdge) {
  Graph g(5); 
  g.add_edge(1, 2);

  auto list = g.get_edges(1);
  ASSERT_EQ(list[0], 2);

  list = g.get_edges(2);
  ASSERT_EQ(list[0], 1);
}

TEST(BasicGraphTests, FromEdgeList) {
  std::vector<std::pair<size_t, size_t>> edges = { {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto list = g.get_edges(0);
  std::vector<size_t> expected = {1, 2};
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

TEST(BasicGraphTests, BfsTest) {
  std::vector<std::pair<size_t, size_t>> edges = { {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto parent = g.bfs(0);
  std::vector<size_t> expected = {0, 0, 0, 1, 2};
  ASSERT_EQ(parent, expected);
}

TEST(CudaTests, HelloCuda) {
  Graph g(0);
  g.hello_cuda();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
