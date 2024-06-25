#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#include "util.h"

// Adjacency Matrix Graph Represenation
class Graph {

  public:
  // Initialize to create an empty graph with no vertices.}
  Graph() : vcount(0), ecount(0), data(), info(ProcInfo::getInstance()), columns(), rows(), output(false) {}
  
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const std::vector<std::pair<int, int>> & edge_list, const size_t vcount);
  // creates a graph from just an edge list
  Graph(const std::vector<std::pair<int,int>> edge_list);

  Graph(const std::string& fname, bool distributed);

  // get the list of vertices vertex is connected to (local indexing)
  std::vector<int> neighbors(const int vert) const;
  // get a list of vertices vert is connected to in the local graph (global indexing)
  std::vector<int> neighborsGlobalIdxs(const int vert) const;

  int degree(int v) const;
  int get_edge(int v1, int v2) const;

  // checks process info to determine if this graph has a partial edge list for v
  bool in_column(int v) const { return v >= this->columns.first && v <= this->columns.second; }
  bool in_row(int v) const { return (v >= this->rows.first && v <= this->rows.second); }
  bool contains(int v) const { return this->in_column(v) || this->in_row(v); }

  int getRowOwner(int v) const { return (v / vcount) * info->width; }
  // int getColOwner(int v) const { return v / vcount; }
  int makeLocal(int v) const { return v % vcount; }
  int localRowToGlobal(int v) const { assert(v < vcount); return v + (info->grid_row * vcount); }
  int localColToGlobal(int v) const { assert(v < vcount); return v + (info->grid_col * vcount); }

  
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
  const ProcInfo* info;

  // start and end (inclusive) ownership for rows and columns
  std::pair<int, int> columns;
  std::pair<int, int> rows;
  bool output;
};


#endif

