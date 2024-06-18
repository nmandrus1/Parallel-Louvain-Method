#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>

#include "util.h"

// Adjacency Matrix Graph Represenation
class Graph {

  public:
  // Initialize to create an empty graph with no vertices.}
  Graph() : vcount(0), ecount(0), data(), info(), columns(), rows(), output(false) {}
  
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const std::vector<std::pair<int, int>> & edge_list, const size_t vcount);
  // creates a graph from just an edge list
  Graph(const std::vector<std::pair<int,int>> edge_list);

  Graph(const std::string& fname, bool distributed);

  // generate a graph using kronecker algorithm
  void from_kronecker(int scale, int edgefactor, unsigned long seed);
  void from_kronecker_cuda(int scale, int edgefactor, unsigned long seed);

  // add an edge between two vertices
  // void add_edge(const int v1, const int v2);

  // get the list of vertices vertex is connected to (local indexing)
  std::vector<int> neighbors(const int vert) const;

  // Perform a Top Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> top_down_bfs(const int src);

  // Perform a Bottom Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> btm_down_bfs(const int src) const;
  
  // Perform a Parallel Top Down BFS from the specified source vertex and return the Parent Array
  std::vector<int> parallel_top_down_bfs(const int src, int checkpoint_int);
  // helper function to broadcast candidate parents across rows
  std::vector<int> parallel_top_down_bfs_driver(std::vector<int> &parents, std::vector<int> &local_frontier, int checkpoint_int); 
  void broadcast_to_row(std::unordered_map<int, int> &candidate_parents);
  bool gather_global_frontier(const std::vector<int> local_frontier, std::vector<int>& global_frontier);
  void checkpoint_data(const std::vector<int>& data, const int iteration, MPI_Comm comm, const char* filename) const;

  // Community detection algorithm
  Graph louvain(const int iterations) const;
  float modularity(std::unordered_map<int, int>& communities) const;
  int degree(int v) const;
  int get_edge(int v1, int v2) const;

  // checks process info to determine if this graph has a partial edge list for v
  bool in_column(int v) const { return v >= this->columns.first && v <= this->columns.second; }
  bool in_row(int v) const { return (v >= this->rows.first && v <= this->rows.second); }
  bool contains(int v) const { return this->in_column(v) || this->in_row(v); }

  // int getRowOwner(int v) const { return (v / vcount) * info.width; }
  // int getColOwner(int v) const { return v / vcount; }
  int makeLocal(int v) const { return v % vcount; }

  // print adj. mat. 
  void print_graph() const;
  void toggle_output() { output = true; }

  // CSR 
  std::vector<int> data;
  std::vector<unsigned> row_index;
  std::vector<unsigned> column_index;

  // number of vertices
  size_t vcount;
  // number of vertices
  size_t ecount;
  // info on MPI Processes/Topology
  ProcInfo info;

  // start and end (inclusive) ownership for rows and columns
  std::pair<int, int> columns;
  std::pair<int, int> rows;
  bool output;
};


std::vector<std::vector<int>> generate_kronecker_list(int scale, int edgefactor, unsigned long seed);
std::vector<std::vector<int>> generate_kronecker_list_cuda(int scale, int edgefactor, unsigned long long seed);

#endif

