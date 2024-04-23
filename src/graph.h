#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>
#include "util.h"

// Adjacency Matrix Graph Represenation
class Graph {

  public:
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const std::vector<std::pair<int, int>> & edge_list, const size_t vcount);

  // generate a graph using kronecker algorithm
  static Graph from_kronecker(int scale, int edgefactor, unsigned long seed);

  // add an edge between two vertices
  void add_edge(const int v1, const int v2);

  // get the list of vertices vertex is connected to (local indexing)
  std::vector<int> get_edges(const int vert) const;

  // use global indexing
  std::vector<int> get_edges_distributed(const int vert) const;

  // Perform a Top Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> top_down_bfs(const int src) const;

  // Perform a Bottom Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> btm_down_bfs(const int src) const;
  
  // Perform a Parallel Top Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> parallel_top_down_bfs(const int src) const;

  // checks process info to determine if this graph has a partial edge list for v
  bool in_column(int v) const { return v >= this->columns.first && v <= this->columns.second; }
  bool in_row(int v) const { return (v >= this->rows.first && v <= this->rows.second); }
  bool contains(int v) const { return this->in_column(v) || this->in_row(v); }

  // print adj. mat. 
  void print_graph() const;
  void hello_cuda();
  
  private:
  // Bitfield Adjacency Matrix
  std::vector<bool> data;
  // number of vertices
  size_t vcount;
  // info on MPI Processes/Topology
  ProcInfo info;

  // start and end (inclusive) ownership for rows and columns
  std::pair<int, int> columns;
  std::pair<int, int> rows;
};


std::vector<std::vector<int>> generate_kronecker_list(int scale, int edgefactor, unsigned long seed);

#endif

