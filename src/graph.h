#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cassert>
#include <cstddef>
#include <iterator>
#include <map>
#include <mpi.h>
#include <set>
#include <utility>
#include <vector>

#include "util.h"

struct Edge {
  int v1, v2;
  double weight;
};

// Adjacency Matrix Graph Represenation
class Graph {

public:
  typedef std::vector<Edge> EdgeList;
  typedef std::map<int, std::set<std::pair<unsigned, double>>> AdjacencyList;

  class NeighborIterator {
  public:
    typedef std::pair<unsigned, double> value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const value_type *pointer;
    typedef const value_type &reference;
    typedef std::input_iterator_tag iterator_category;

    NeighborIterator(const std::vector<unsigned> &column_index,
                  const std::vector<double> &weights, size_t start, size_t end)
        : column_index_(column_index), weights_(weights), start_(start),
          current_(start), end_(end) {}

    bool operator!=(const NeighborIterator &other) const {
      return current_ != other.current_;
    }

    value_type operator*() const {
      return {column_index_[current_], weights_[current_]};
    }

    const NeighborIterator &operator++() {
      current_++;
      return *this;
    }

    NeighborIterator begin() { return *this; }
    NeighborIterator end() { return NeighborIterator(column_index_, weights_, end_, end_); }
 
    
  private:
    const std::vector<unsigned> &column_index_;
    const std::vector<double> &weights_;
    size_t start_;
    size_t current_;
    size_t end_;
  };

  // typedef std::pair<VtxIterator, VtxIterator> VertexIterator;
  // typedef std::pair<NeighIterator, NeighIterator> NeighborIterator;

  // helper function to construct an edgelist from an input file
  static EdgeList edge_list_from_file(const std::string &fname);

  // write the distributed graph to a file
  void write_edges_to_file(const std::string& directory) const;

  Graph();
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const EdgeList &edge_list, const size_t vcount);
  // creates a graph from just an edge list
  Graph(const EdgeList &edge_list);

  // creates a graph from an adjacency list
  Graph(const AdjacencyList &adj_list);

  Graph(const std::string &fname, bool distributed);
  ~Graph();

  void distributedGraphInit(const EdgeList &edge_list, ProcInfo info);
  void distributedGraphInitWithGlobalVcount(const EdgeList &edge_list, ProcInfo info, int vcount);

  // return iterator over neighbors and the edge weight
  NeighborIterator neighbors(const int vert) const;

  int get_edge(int v1, int v2) const;

  bool in_row(int v) const {
    return (v >= this->rows.first && v < this->rows.second);
  }

  double weighted_degree(int vtx) const;

  // Assuming global indexing
  int getRankOfOwner(int v) const { return v / local_vcount; }
  int makeLocal(int v) const { return v % local_vcount; }
  int localRowToGlobal(int v) const {
    assert(v < local_vcount);
    return v + (info.rank * local_vcount);
  }

  // print adj. mat.
  void print() const;
  void distributed_print() const;

  // CSR
  std::vector<unsigned> row_index;
  std::vector<unsigned> column_index;
  std::vector<double> weights;

  // number of vertices
  size_t local_vcount;
  size_t global_vcount;
  // number of vertices
  size_t ecount;
  // info on MPI Processes/Topology
  ProcInfo info;

  // start and end (inclusive) ownership for rows
  std::pair<int, int> rows;

  MPI_Datatype MPI_EDGE;

private:
  // convert adj_list into CSR and return the number of edegs in the list
  int sparsify(const AdjacencyList &adj_list);
  void initializeFromAdjList(const AdjacencyList &adj_list);
  // Create an MPI_Datatype for Edge struct
  void create_edge_datatype(MPI_Datatype* dt); 
};

#endif
