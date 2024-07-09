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
  std::vector<double> edges_to_other_comms;

  Communities(Graph& g);
  
  void insert(int node, int community, int node_comm_degree);
  void remove(int node, int community, int node_comm_degree);
  int compute_best_community(int node, int node_comm);
  void compute_neighbors(int node);
  double modularity_gain(int node, int comm, double node_comm_degree);
  bool iterate();

  void print_comm_membership();

  double modularity();

  Graph into_new_graph();

  Graph& g;
};

#endif
