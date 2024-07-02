#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cstddef>
#include <utility>
#include <vector>
#include <cassert>
#include <map>
#include <set>

#include "util.h"

// Adjacency Matrix Graph Represenation
class Graph {

  public:
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const std::vector<std::pair<int, int>> & edge_list, const size_t vcount);
  // creates a graph from just an edge list
  Graph(const std::vector<std::pair<int,int>> edge_list);

  Graph(const std::string& fname, bool distributed);

  // get the list of vertices vertex is connected to (local indexing)
  std::vector<int> neighbors(const int vert) const;
  inline int degree(int v) const { 
    assert(in_row(v)); 
    int vert = makeLocal(v);
    return this->row_index[vert + 1] - this->row_index[vert];
  }

  int get_edge(int v1, int v2) const;

  bool in_row(int v) const { return (v >= this->rows.first && v < this->rows.second); }

  // Assuming global indexing
  int getRankOfOwner(int v) const { return v / local_vcount; }
  int makeLocal(int v) const { return v % local_vcount; }
  int localRowToGlobal(int v) const { assert(v < local_vcount); return v + (info.rank * local_vcount); }
  
  // print adj. mat. 
  void print_graph() const;

  // CSR 
  std::vector<int> data;
  std::vector<unsigned> row_index;
  std::vector<unsigned> column_index;

  // number of vertices
  size_t local_vcount;
  size_t global_vcount;
  // number of vertices
  size_t ecount;
  // info on MPI Processes/Topology
  ProcInfo info;

  // start and end (inclusive) ownership for rows
  std::pair<int, int> rows;

  private: 
    // convert adj_list into CSR
    void sparsify(const std::map<int, std::set<unsigned>>& adj_list);
    void initializeFromAdjList(const std::map<int, std::set<unsigned>>& adj_list);
    void distributedGraphInit(const std::vector<std::pair<int, int>>& edge_list, ProcInfo info);
};


#endif

