#ifndef __COMM_H__
#define __COMM_H__

#include "graph.h"

struct Communities {
  std::vector<int> node_to_comm_map;
  std::vector<int> comm_to_degree_map;

  // stores running summations used to quickly calculate community modularity
  std::vector<double> in, total;

  // these are repopulated for every node 
  // when computing neighboring community weights
  std::vector<int> neighbor_comms;
  std::vector<double> neighbor_weights;

  Communities(Graph& g);
  
  void insert(int node, int community, int node_comm_degree);
  void remove(int node, int community, int node_comm_degree);
  int compute_best_community(int node, int node_comm);
  void compute_neighbors(int node);
  double modularity_gain(int node, int comm, double node_comm_degree);
  bool iterate();

  double modularity();

  Graph into_new_graph();

  Graph& g;
};

// Distributed Community Detection
struct DistCommunities {
  std::vector<int> node_to_comm_map;
  std::vector<int> comm_to_degree_map;
  std::vector<int> node_to_global_degree;

  // stores running summations used to quickly calculate community modularity
  std::vector<double> in, total;

  // these are repopulated for every node 
  // when computing neighboring community weights
  std::vector<int> neighbor_comms;
  std::vector<double> neighbor_weights;

  DistCommunities(Graph& g);
  
  void insert(int node, int community, int node_comm_degree);
  void remove(int node, int community, int node_comm_degree);
  int compute_best_community(int node, int node_comm);
  void compute_neighbors(int node);
  double modularity_gain(int node, int comm, double node_comm_degree);
  bool iterate();

  double modularity();

  Graph into_new_graph();

  Graph& g;
};

#endif
