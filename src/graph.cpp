#include "graph.h"
#include <cstddef>
#include <iostream>
#include <unordered_set>
#include <queue>

void runHelloCuda();

// default constructor
Graph::Graph(size_t vcount) : vcount(vcount) {
  this->data.resize(vcount*vcount, 0);
}

// construct a graph from an edge list

Graph::Graph(const std::vector<std::pair<size_t, size_t>> &edge_list, const size_t vcount): Graph(vcount) {
  // loop over every edge pair and add it to the graph
  for (auto pair: edge_list) {
    this->add_edge(pair.first, pair.second); 
  }
}

// add an edge between v1 and v2
void Graph::add_edge(const size_t v1, const size_t v2) {
  // 1-d indexing
  this->data[v1 * this->vcount + v2] += 1;
  this->data[v2 * this->vcount + v1] += 1;
}


// get the list of vertices vert is connected to
std::vector<size_t> Graph::get_edges(const size_t vert) const {
  std::vector<size_t> ret;  

  // loop over every vertex and push back those vert is adjacent to
  for(unsigned i = 0; i < vcount; i++) {
    if (this->data[vert * vcount + i] > 0) 
      ret.push_back(i);
  }

  return ret;
}

// Perform a BFS from the specified source vertex and return the Parent Array
std::vector<size_t> Graph::bfs(const size_t src) const {
  // queue for storing future nodes to explore
  std::queue<size_t> q;
  q.push(src);
  // Hashset to store visited nodes
  std::unordered_set<size_t> visited;
  // vector of parents 
  std::vector<size_t> parents(this->vcount, 0);

  while(!q.empty()) {
    size_t parent = q.front();    
    q.pop();

    // get list of adjacent vertices
    auto connections = this->get_edges(parent);
    for (size_t v : connections) {
      if (!visited.contains(v)) {
        parents[v] = parent;
        q.push(v);
        visited.insert(v);
      }
    }

    // insert parent into visited list
    visited.insert(parent);
  }

  return parents;
}

void Graph::print_graph() const {
  for(unsigned i = 0; i < this->vcount; i++) {
    for(unsigned j = 0; j < this->vcount; j++) {
      std::cout << this->data[this->vcount * i + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

// test CUDA function
void Graph::hello_cuda() {
  runHelloCuda();  
}
