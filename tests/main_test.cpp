#include "graph.h"
#include "community.h"
#include <cstdlib>
#include <gtest/gtest.h>
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
  Graph::EdgeList edges = {
      {0, 1, 1.0}, {0, 2, 1.0}, {1, 2, 1.0}, {1, 3, 1.0}, {2, 4, 1.0}};
  Graph g(edges);

  std::vector<unsigned> expected_col_index = {1, 2, 0, 2, 3, 0, 1, 4, 1, 2};
  std::vector<unsigned> expected_row_index = {0, 2, 5, 8, 9, 10};
  std::vector<double> expected_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  ASSERT_EQ(g.ecount, 5);
  ASSERT_EQ(g.column_index, expected_col_index);
  ASSERT_EQ(g.row_index, expected_row_index);
  ASSERT_EQ(g.weights, expected_weights);

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

TEST(CommunityDetection, Small) {
  Graph::EdgeList edges = {
      {1, 2, 1.0},  {1, 4, 1.0},  {1, 7, 1.0},   {2, 0, 1.0},   {2, 4, 1.0},   {2, 5, 1.0},   {2, 6, 1.0},
      {3, 0, 1.0},  {3, 7, 1.0},  {4, 0, 1.0},   {4, 10, 1.0},  {5, 0, 1.0},   {5, 7, 1.0},   {5, 11, 1.0},
      {6, 7, 1.0},  {6, 11, 1.0}, {8, 9, 1.0},   {8, 10, 1.0},  {8, 11, 1.0},  {8, 14, 1.0},  {8, 15, 1.0},
      {9, 12, 1.0}, {9, 14, 1.0}, {10, 11, 1.0}, {10, 12, 1.0}, {10, 13, 1.0}, {10, 14, 1.0}, {11, 13, 1.0},
  };

  Graph g(edges);
  g.print_graph();

  ASSERT_TRUE(g.ecount == 28);
  
  Communities c(g);

  double eps = 0.000001;

  ASSERT_TRUE(std::abs(c.modularity() - -0.0714286) < eps);
  // ASSERT_DOUBLE_EQ(c.modularity() , -0.0714286);

  ASSERT_TRUE(c.iterate());
  c.print_comm_membership();

  ASSERT_TRUE(std::abs(c.modularity() - 0.346301) < eps);

  Graph g2 = c.into_new_graph();
  g2.print_graph();

  ASSERT_EQ(g2.local_vcount, 4);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
