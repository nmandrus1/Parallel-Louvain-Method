#include "graph.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>
#include <vector>

// TEST(BasicGraphTests, AddEdge) {
//   Graph g(5);
//   g.add_edge(1, 2);

//   auto list = g.get_edges(1);
//   ASSERT_EQ(list[0], 2);

//   list = g.get_edges(2);
//   ASSERT_EQ(list[0], 1);
// }

TEST(BasicGraphTests, FromEdgeList) {
  std::vector<std::pair<int, int>> edges = {
      {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  std::vector<unsigned> expected_col_index = {1, 2, 0, 2, 3, 0, 1, 4, 1, 2};
  std::vector<unsigned> expected_row_index = {0, 2, 5, 8, 9, 10};

  ASSERT_EQ(g.data.size(), 10);
  ASSERT_EQ(g.column_index, expected_col_index);
  ASSERT_EQ(g.row_index, expected_row_index);

  // auto list = g.get_edges(0);
  // std::vector<int> expected = {1, 2};
  // ASSERT_EQ(list, expected);

  // list = g.get_edges(1);
  // expected = {0, 2, 3};
  // ASSERT_EQ(list, expected);

  // list = g.get_edges(2);
  // expected = {0, 1, 4};
  // ASSERT_EQ(list, expected);

  // list = g.get_edges(3);
  // expected = {1};
  // ASSERT_EQ(list, expected);

  // list = g.get_edges(4);
  // expected = {2};
  // ASSERT_EQ(list, expected);
}

TEST(BasicGraphTests, TopDownBfsTest) {
  std::vector<std::pair<int, int>> edges = {
      {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto parent = g.top_down_bfs(0);
  std::vector<int> expected = {0, 0, 0, 1, 2};
  ASSERT_EQ(parent, expected);
}

TEST(BasicGraphTests, BtmDownBfsTest) {
  std::vector<std::pair<int, int>> edges = {
      {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}};
  Graph g(edges, 5);

  auto parent = g.btm_down_bfs(0);
  std::vector<int> expected = {0, 0, 0, 1, 2};
  ASSERT_EQ(parent, expected);
}

TEST(KroneckerTest, BasicKroneckerEdgeList) {
  auto list = generate_kronecker_list(5, 16, 123);
}

TEST(CommunityDetection, Small) {
  std::vector<std::pair<int, int>> edges = {
      {1, 2},  {1, 4},  {1, 7},   {2, 0},   {2, 4},   {2, 5},   {2, 6},
      {3, 0},  {3, 7},  {4, 0},   {4, 10},  {5, 0},   {5, 7},   {5, 11},
      {6, 7},  {6, 11}, {8, 9},   {8, 10},  {8, 11},  {8, 14},  {8, 15},
      {9, 12}, {9, 14}, {10, 11}, {10, 12}, {10, 13}, {10, 14}, {11, 13},
  };

  
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
