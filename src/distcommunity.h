#ifndef __DIST_COMM_H__
#define __DIST_COMM_H__

#include "graph.h"
#include <unordered_map>
#include <unordered_set>

#define MPI_REMOVAL_TAG 0
#define MPI_ADDITION_TAG 1
#define MPI_DATA_TAG 2
#define MPI_NEIGHBOR_SYNC 3
#define MPI_COMM_SYNC 4

#ifdef PROFILE_FNS
#include <gptl.h>
#include <gptlmpi.h>
#endif

struct CommunityUpdate {

  enum UpdateType {
    Addition,
    Removal
  };

  UpdateType type;
  int node;
  double global_degree;
  int old_comm;
  int new_comm;
  double edges_within_old_comm;
  double edges_within_new_comm;
  int num_ranks_bordering_node;
};

struct CommunityInfo {
  int comm;
  double in;
  double total;
};


// Distributed Community Detection
struct DistCommunities {
  
  // It is EXTREMEMLY important to distinguish between row and column vertices, 
  // since our graph representation stores nodes with local indecies, neighbors of 

  // will map neighboring nodes to communities
  std::unordered_map<int, int> gbl_vtx_to_comm_map;
  // map a community to its size
  std::unordered_map<int, int> comm_size;

  // stores running summations used to quickly calculate community modularity
  std::unordered_map<int, double> in, total;

  // these are repopulated for every node 
  // when computing neighboring community weights
  std::vector<int> neighbor_comms;
  std::unordered_map<int, double> edges_to_other_comms;

  // maps local vertices to ranks that need updates on this vertex's community
  std::unordered_map<int, std::unordered_set<int>> rank_to_border_vertices;
  
  // comm/vtx -> { rank -> int }
  std::unordered_map<int, std::unordered_map<int, int>> vtx_rank_degree;
  std::unordered_map<int, std::unordered_map<int, int>> comm_ref_count;

  std::unordered_set<int> comms_updated_this_iter;

  DistCommunities(Graph& g);
  ~DistCommunities();

  void init();
  
  double modularity();
  void insert(int node, int community, double degree, int edges_within_comm, std::unordered_map<int, int>& rank_counts);
  void remove(int node, int community, double degree, int edges_within_comm, std::unordered_map<int, int>& rank_counts);
  // int compute_best_community(int node, int node_comm);
  int compute_best_community(int node, int node_comm, double temperature);
  void compute_neighbors(int node);
  double modularity_gain(int node, int comm, double node_comm_degree);
  bool iterate();
  bool one_level();

  void process_incoming_updates();
  void update_subscribers();
  void update_neighbors(int vtx);
  void send_community_update(int dest, const CommunityUpdate& update);


  // I/O
  void write_communities_to_file(const std::string& directory) const;
  void print_comm_ref_counts() const;
  void print_comm_membership() const;

  // Reconstruction
  void into_new_graph();
  
  Graph& g;

  // MPI CUSTOM Datatype
  MPI_Datatype MPI_DEGREE_INFO;
  MPI_Datatype MPI_COMMUNITY_UPDATE;
};


#endif
