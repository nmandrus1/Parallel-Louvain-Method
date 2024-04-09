#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <cstddef>
#include <utility>
#include <vector>

// Adjacency Matrix Graph Represenation
class Graph {

  public:
  // constructor creates adj. mat. with vcount^2 elements in data vector
  Graph(size_t vcount);
  // creates graph from list of edges and a vcount
  Graph(const std::vector<std::pair<size_t, size_t>> & edge_list, const size_t vcount);

  // add an edge between two vertices
  void add_edge(const size_t v1, const size_t v2);

  // get the list of vertices vertex is connected to
  std::vector<size_t> get_edges(const size_t vert) const;

  // Perform a BFS from the specified source vertex and return the Parent Array
  std::vector<size_t> bfs(const size_t src) const;

  // print adj. mat. 
  void print_graph() const;
  void hello_cuda();
  
  private:
  // 1D Adjacency Matrix
  std::vector<size_t> data;
  // number of vertices
  size_t vcount;
};

#endif
