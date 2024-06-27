#ifndef __COMM_H__
#define __COMM_H__

#include "graph.h"
#include <unordered_map>
#include <unordered_set>

#define MPI_REMOVAL_TAG 0
#define MPI_ADDITION_TAG 1
#define MPI_COMM_SYNC 2
#define MPI_NEIGHBOR_SYNC 3


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


  double modularity();

  Graph into_new_graph();

  Graph& g;
};


struct CommunityUpdate {
  int node;
  int global_degree;
  int old_comm;
  int new_comm;
  double edges_within_old_comm;
  double edges_within_new_comm;
};

struct CommunityInfo {
  int comm;
  double in;
  double total;
};

struct NeighborUpdate {
  int node;
  CommunityInfo comm_info;
};

// Distributed Community Detection
struct DistCommunities {
  
  // It is EXTREMEMLY important to distinguish between row and column vertices, 
  // since our graph representation stores nodes with local indecies, neighbors of 

  // will map neighboring nodes to communities
  std::unordered_map<int, int> gbl_vtx_to_comm_map;

  std::unordered_map<int,int> neighbor_degree;

  // stores running summations used to quickly calculate community modularity
  std::unordered_map<int, double> in, total;

  // these are repopulated for every node 
  // when computing neighboring community weights
  std::vector<int> neighbor_comms;
  std::unordered_map<int, double> edges_to_other_comms;

  // a map of communities to the ranks that need updated community information 
  std::unordered_map<int, std::unordered_set<int>> comm_subscribers;

  // maps local vertices to ranks that need updates on this vertex's community
  std::unordered_map<int, std::unordered_set<int>> neighbor_subscribers;

  DistCommunities(Graph& g);
  
  void insert(int node, int community, int node_comm_degree);
  void remove(int node, int community, int node_comm_degree);
  int compute_best_community(int node, int node_comm);
  void compute_neighbors(int node);
  double modularity_gain(int node, int comm, double node_comm_degree);
  bool iterate();


  void process_incoming_updates();
  void process_local_removal(const CommunityUpdate& update);
  void process_local_addition(const CommunityUpdate& update);
  void update_subscribers();
  void update_neighbors();

  double modularity();

  Graph into_new_graph();

  Graph& g;
};


struct DistCommunityUpdate {
  int node;
  int node_comm;
  int best_comm;
  double node_comm_degree;
  double best_comm_degree;
};


void serialize(DistCommunityUpdate& data, std::vector<char>& buffer);
void deserialize(std::vector<char>& buffer, DistCommunityUpdate& data);

#endif
