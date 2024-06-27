#include "community.h"
#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <mpi.h>
#include <iostream>

// Constructor for the Communities class.
// Initializes internal data structures and sets up initial community
// assignments where each node is its own community.
Communities::Communities(Graph &g) : g(g) {
  // Resize all vectors to accommodate the graph's vertex count.
  node_to_comm_map.resize(g.local_vcount);
  comm_to_degree_map.resize(g.local_vcount);
  in.resize(g.local_vcount);
  total.resize(g.local_vcount);
  neighbor_comms.resize(g.local_vcount);
  edges_to_other_comms.resize(g.local_vcount);

  // Initialize communities such that each node is in its own community.
  for (int i = 0; i < g.local_vcount; i++) {
    node_to_comm_map[i] = i;
    total[i] =
        g.degree(i); // Total degree of the community is the degree of the node.
    in[i] = 0;       // Initially, no internal edges within the community.
  }
}

// Inserts a node into a community and updates relevant metrics.
void Communities::insert(int node, int community, int node_comm_degree) {
  node_to_comm_map[node] = community;
  total[community] += g.degree(node);
  in[community] += 2 * node_comm_degree;
}

// Removes a node from a community, updating the internal community structure
// and degree information.
void Communities::remove(int node, int community, int node_comm_degree) {
  node_to_comm_map[node] = -1;
  total[community] -= g.degree(node);
  in[community] -= 2 * node_comm_degree;
}

// Computes the overall modularity of the graph based on current community
// assignments.
double Communities::modularity() {
  double q = 0.0;
  double m2 = static_cast<double>(g.ecount) *
              2; // Total weight of all edges in the graph, multiplied by 2.

  for (int i = 0; i < g.local_vcount; i++) {
    if (total[i] > 0)
      q += in[i] / m2 -
           (total[i] / m2) * (total[i] / m2); // Modularity formula as sum of
                                              // each community's contribution.
  }

  return q;
}

// Executes one pass of the Louvain method algorithm, attempting to improve the
// modularity by moving nodes between communities.
bool Communities::iterate() {
  int total_num_moves = 0;
  int prev_num_moves = 0;
  bool improvement = false;

  while (true) {
    prev_num_moves = total_num_moves;

    for (int node = 0; node < g.local_vcount; node++) {
      int node_comm = node_to_comm_map[node];

      compute_neighbors(node); // Update the weights to all neighboring
                               // communities of the node.

      // Temporarily remove the node from its current community.
      remove(node, node_comm, edges_to_other_comms[node_comm]); 

      // Determine the best community for this node based on potential modularity gain.
      int best_comm = compute_best_community(node, node_comm); 

      // Insert the node into the best community found.
      insert(node, best_comm, edges_to_other_comms[best_comm]); 

      if (best_comm != node_comm)
        total_num_moves++;
    }

    if (total_num_moves > 0)
      improvement = true;

    if (total_num_moves == prev_num_moves)
      return improvement; // Return whether there was any improvement in this iteration.
  }
}

// Given a node and its current community, computes the best community for this
// node that would increase the modularity the most.
int Communities::compute_best_community(int node, int node_comm) {
  int best_comm = node_comm;
  double best_increase = 0.0;
  for (auto neighbor_comm : neighbor_comms) {
    double increase = modularity_gain(node, neighbor_comm, edges_to_other_comms[neighbor_comm]);
    if (increase > best_increase) {
      best_comm = neighbor_comm;
      best_increase = increase;
    }
  }
  return best_comm;
}

// Computes and updates internal structures with weights corresponding to each
// neighboring community of a given node.
void Communities::compute_neighbors(int node) {
  neighbor_comms.clear();
  std::fill(edges_to_other_comms.begin(), edges_to_other_comms.end(), -1.0);

  for (int neighbor : g.neighbors(node)) {
    int neighbor_comm = node_to_comm_map[neighbor];

    if (node != neighbor) {
      // if this neighbor community hasn't been seen yet, 
      // initialize it before adding weight to it
      if (edges_to_other_comms[neighbor_comm] == -1) {
        edges_to_other_comms[neighbor_comm] = 0;
        neighbor_comms.push_back(neighbor_comm);
      }

      // Increment the edge weight to the neighboring community.
      // this value is used to store the edges from this node to 
      // other nodes in whichever community we decide to join
      edges_to_other_comms[neighbor_comm] += 1.0; 
    }
  }
}

// Computes the potential modularity gain from moving a node to a new community.
double Communities::modularity_gain(int node, int comm, double node_comm_degree) {
  double totc = static_cast<double>(total[comm]);
  double degc = static_cast<double>(g.degree(node));
  double m2 = static_cast<double>(g.ecount) * 2;
  double dnc = static_cast<double>(node_comm_degree);

  return (dnc - totc * degc / m2); // Modularity gain formula.
}

// Constructs a new graph based on the current community assignments.
Graph Communities::into_new_graph() {
  std::unordered_map<int, std::vector<int>> map;

  for (int i = 0; i < node_to_comm_map.size(); i++) {
    map[node_to_comm_map[i]].push_back(i);
  }

  // Renumber communities to be consecutive starting from 0.
  std::fill(node_to_comm_map.begin(), node_to_comm_map.end(), -1);
  int new_comm_number = 0;
  for (auto pair : map) {
    for (auto node : pair.second)
      node_to_comm_map[node] = new_comm_number;
    new_comm_number++;
  }

  // Create new edges based on community connections.
  std::vector<std::unordered_set<int>> comm_edges(new_comm_number);
  for (int node = 0; node < node_to_comm_map.size(); node++) {
    int comm = node_to_comm_map[node];
    for (auto neighbor : g.neighbors(node))
      comm_edges[comm].insert(node_to_comm_map[neighbor]);
  }

  std::vector<std::pair<int, int>> edge_list;
  for (int node = 0; node < comm_edges.size(); node++) {
    for (auto neighbor : comm_edges[node]) {
      edge_list.push_back(std::make_pair(node, neighbor));
    }
  }

  return Graph(edge_list); // Return a new graph representing the compressed
                           // community structure.
}


// Constructor for the DistCommunities class.
// Initializes internal data structures and sets up initial community
// assignments where each node is its own community.
DistCommunities::DistCommunities(Graph &g) : g(g) {
  // Resize all vectors to accommodate the graph's vertex count.
  gbl_vtx_to_comm_map.reserve(g.local_vcount * 2);
  in.reserve(g.local_vcount * 2);
  total.reserve(g.local_vcount * 2);
  neighbor_comms.resize(g.local_vcount);
  edges_to_other_comms.reserve(g.local_vcount * 2);
  neighbor_degree.reserve(g.local_vcount * 2);

  std::unordered_map<int, std::vector<int>> msg_map;
  
  // exchange the degree of our local vertices to their neighbors
  for(int v = g.rows.first; v< g.rows.second; v++) {
   for(auto n: g.neighbors(v)) {
     int owner = g.getRankOfOwner(n);
      if(owner != g.info->rank) {      
        msg_map[owner].push_back(v);
        msg_map[owner].push_back(g.degree(v));
      }   
    }
  }

  int degree_buf[2];


  for(auto& [rank, degrees]: msg_map) {
    for(int i = 0; i < degrees.size(); i += 2) {
      degree_buf[0] = degrees[i];
      degree_buf[1] = degrees[i + 1];
      MPI_Send(&degree_buf, 2, MPI_INT, rank, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int flag;
  MPI_Status status;
  while(true) {
      MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);

      if (flag) {
          MPI_Recv(&degree_buf, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
          neighbor_degree.insert({degree_buf[0] ,degree_buf[1]});
      } else {
          // No more updates available at this time
          break;
      }
  }

  // Initialize communities such that each row node is in its own community.
  // global indexing
  for (int v = g.rows.first; v < g.rows.second; v++) {
    gbl_vtx_to_comm_map[v] = v;
    total[v] = g.degree(v);  // Total degree of the community is the degree of the node.
    in[v] = 0;                            // Initially, no internal edges within the community.

    for(int n: g.neighbors(v)) {
     if(gbl_vtx_to_comm_map.contains(n)) continue;
     gbl_vtx_to_comm_map[n] = n;
     total[n] = neighbor_degree[n];
     in[n] = 0;

     int neighbor_owner = g.getRankOfOwner(n);
     // if there is a remote neighbor that needs to be kept up to date add them to subscriber list
     if(neighbor_owner != g.info->rank)
       neighbor_subscribers[v].insert(neighbor_owner);
    }
  }
}

// Inserts a node into a community and updates relevant metrics.
void DistCommunities::insert(int node, int community, int edges_within_comm) {
  gbl_vtx_to_comm_map[node] = community;
  total[community] += g.degree(node);
  in[community] += 2 * edges_within_comm;
}

// Removes a node from a community, updating the internal community structure
// and degree information.
void DistCommunities::remove(int node, int community, int edges_within_comm) {
  gbl_vtx_to_comm_map[node] = -1;
  total[community] -= g.degree(node);
  in[community] -= 2 * edges_within_comm;
}

// Computes the overall modularity of the graph based on current community
// assignments.
double DistCommunities::modularity() {
  double q = 0.0;
  double m2 = static_cast<double>(g.ecount) * 2; // Total weight of all edges in the graph, multiplied by 2.

  for (int v = g.rows.first; v < g.rows.second; v++) {
    if (total[v] > 0)
      q += in[v] / m2 - (total[v] / m2) * (total[v] / m2); // Modularity formula as sum of
                                                           // each community's contribution.
  }

  MPI_Allreduce(&q, &q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return q;
}

// Executes one pass of the Louvain method algorithm, attempting to improve the
// modularity by moving nodes between communities.
bool DistCommunities::iterate() {
  int total_num_moves = 0;
  int prev_num_moves = 0;
  bool improvement = false, all_finished = false;
  // std::vector<char> community_update_buf(sizeof(DistCommunityUpdate));

  while (true) {
    prev_num_moves = total_num_moves;

    for (int node = g.rows.first; node < g.rows.second; node++) {
      int node_comm = gbl_vtx_to_comm_map[node];

      std::cout << "RANK " << g.info->rank << ": Computing neighbors" << std::endl;
      compute_neighbors(node); // Update the weights to all neighboring
                               // communities of the node.

      // Temporarily remove the node from its current community.
      std::cout << "RANK " << g.info->rank << ": Removing vtx " << node << " from comm " << node_comm << std::endl;
      remove(node, node_comm, edges_to_other_comms[node_comm]); 

      // Determine the best community for this node based on potential modularity gain.
      std::cout << "RANK " << g.info->rank << ": computing best comm" << std::endl;
      auto best_comm = compute_best_community( node, node_comm); 
      std::cout << "RANK " << g.info->rank << ": best comm = " << best_comm << std::endl;

      if(best_comm != node_comm) {
        // MPI_Request req;
        CommunityUpdate update = {node, g.degree(node), node_comm, best_comm, edges_to_other_comms[node_comm], edges_to_other_comms[best_comm]};
        // calculate the ranks that own the old and new communities
        int old_comm_owner = g.getRankOfOwner(node_comm);
        int new_comm_owner = g.getRankOfOwner(best_comm);

        std::cout << "RANK " << g.info->rank << ": Communicating removal/addition \t old: " << old_comm_owner << " new: " << new_comm_owner << std::endl;
        if(old_comm_owner != g.info->rank) 
          MPI_Send(&update, sizeof(CommunityUpdate), MPI_BYTE, old_comm_owner, MPI_REMOVAL_TAG, MPI_COMM_WORLD);
        // No else required since if the old owner was this rank then it was removed properly by the remove() call 
        
        if(new_comm_owner != g.info->rank) {
          MPI_Send(&update, sizeof(CommunityUpdate), MPI_BYTE, new_comm_owner, MPI_ADDITION_TAG, MPI_COMM_WORLD);
          gbl_vtx_to_comm_map[node] = best_comm;
        } else insert(node, best_comm, edges_to_other_comms[best_comm]);
        
        total_num_moves++;
      } else insert(node, best_comm, edges_to_other_comms[best_comm]);

      MPI_Barrier(MPI_COMM_WORLD);
      // recieve updates from remote process about nodes joining local community
      std::cout << "RANK " << g.info->rank << ": Processing remote updates" << std::endl;
      process_incoming_updates();

      std::cout << "RANK " << g.info->rank << ": Updating subscriber procs" << std::endl;
      update_subscribers();
      update_neighbors();

      if (best_comm != node_comm)
        total_num_moves++;

      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (total_num_moves > 0)
      improvement = true;

    if (total_num_moves == prev_num_moves) 
      all_finished = true;

    MPI_Allreduce(&all_finished, &all_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(all_finished) return improvement; // Return whether there was any improvement in this iteration.


    auto mod = modularity();
    if(g.info->rank == 0)
      std::cout << "Modularity: " << mod << std::endl; 

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void DistCommunities::process_incoming_updates() {
    MPI_Status status;
    int flag;
    CommunityUpdate update;

    // Continue processing as long as there are messages
    while (true) {
        // Check for any incoming message
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            // Receive the update
            MPI_Recv(&update, sizeof(update), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == MPI_ADDITION_TAG) {
                // Handle addition
                process_local_addition(update);
                comm_subscribers[update.new_comm].insert(status.MPI_SOURCE);
                std::cout << "RANK " << g.info->rank << ": Adding rank " << status.MPI_SOURCE << " to subscriber list" << std::endl;
            } else if (status.MPI_TAG == MPI_REMOVAL_TAG) {
                // Handle removal
                process_local_removal(update);
                comm_subscribers[update.old_comm].erase(status.MPI_SOURCE);
            }
        } else {
            // No more messages to process
            break;
        }
    }
}

void DistCommunities::process_local_addition(const CommunityUpdate& update) {
    // Assuming gbl_vtx_to_comm_map, total, and in are accessible
    gbl_vtx_to_comm_map[update.node] = update.new_comm;
    total[update.new_comm] += update.global_degree;
    in[update.new_comm] += 2 * update.edges_within_new_comm;  // Assuming edge weight needs to be doubled
}

void DistCommunities::process_local_removal(const CommunityUpdate& update) {
    if (gbl_vtx_to_comm_map[update.node] == update.old_comm) {
        total[update.old_comm] -= update.global_degree;
        in[update.old_comm] -= 2 * update.edges_within_old_comm;  // Assuming edge weight needs to be doubled
        std::cout << "RANK " << g.info->rank << ": Node " << update.node << " removed from community " << update.old_comm << std::endl;
    }
}


void DistCommunities::update_subscribers() {
  MPI_Status status;
  int flag;
  CommunityInfo comm_info;

  for (auto& [comm, ranks]: comm_subscribers) {
   CommunityInfo updated_info = {comm, in[comm], total[comm]}; 
   for(int rank: ranks) 
     MPI_Send(&updated_info, sizeof(CommunityInfo), MPI_BYTE, rank, MPI_COMM_SYNC, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Check for incoming community updates
  while(true) {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_COMM_SYNC, MPI_COMM_WORLD, &flag, &status);

      if (flag) {
          MPI_Recv(&comm_info, sizeof(CommunityInfo), MPI_BYTE, MPI_ANY_SOURCE, MPI_COMM_SYNC, MPI_COMM_WORLD, &status);

          // Process the received community info
          in[comm_info.comm] = comm_info.in;
          total[comm_info.comm] = comm_info.total;
      } else {
          // No more updates available at this time
          break;
      }
  }
}


void DistCommunities::update_neighbors() {
  MPI_Status status;
  int flag;
  NeighborUpdate updated_info;

  for (auto& [vtx, ranks]: neighbor_subscribers) {
    int comm = gbl_vtx_to_comm_map[vtx];
    updated_info = {vtx, {comm, in[comm], total[comm]}}; 
    for(int rank: ranks) 
      MPI_Send(&updated_info, sizeof(NeighborUpdate), MPI_BYTE, rank, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Check for incoming community updates
  while(true) {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD, &flag, &status);

      if (flag) {
          MPI_Recv(&updated_info, sizeof(NeighborUpdate), MPI_BYTE, MPI_ANY_SOURCE, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD, &status);

          // Process the received community info
          gbl_vtx_to_comm_map[updated_info.node] = updated_info.comm_info.comm;
          in[updated_info.comm_info.comm] = updated_info.comm_info.in;
          total[updated_info.comm_info.comm] = updated_info.comm_info.total;
      } else {
          // No more updates available at this time
          break;
      }
  }
}
// Given a node and its current community, computes the best community for this
// node that would increase the modularity the most.
int DistCommunities::compute_best_community(int node, int node_comm) {
  int best_comm = node_comm;
  double best_increase = 0.0;
  for (auto neighbor_comm : neighbor_comms) {
    double increase = modularity_gain(node, neighbor_comm, edges_to_other_comms[neighbor_comm]);
    if (increase > best_increase) {
      best_comm = neighbor_comm;
      best_increase = increase;
    }
  }

  return best_comm;
}

// Computes and updates internal structures with weights corresponding to each
// neighboring community of a given node.
void DistCommunities::compute_neighbors(int node) {
  neighbor_comms.clear();
  edges_to_other_comms.clear();

  for (int neighbor : g.neighbors(node)) {
    int neighbor_comm = gbl_vtx_to_comm_map[neighbor];

    if (node != neighbor) {
      // if this neighbor community hasn't been seen yet, 
      // initialize it before adding weight to it
      if (!edges_to_other_comms.contains(neighbor_comm)) {
        edges_to_other_comms[neighbor_comm] = 0;
        neighbor_comms.push_back(neighbor_comm);
      }

      // Increment the edge weight to the neighboring community.
      edges_to_other_comms[neighbor_comm] += 1.0; 
    }
  }
}

// Computes the potential modularity gain from moving a node to a new community.
double DistCommunities::modularity_gain(int node, int comm, double node_comm_degree) {
  double totc = static_cast<double>(total[comm]);
  double degc = static_cast<double>(g.degree(node));
  double m2 = static_cast<double>(g.ecount) * 2;
  double dnc = static_cast<double>(node_comm_degree);

  return (dnc - totc * degc / m2); // Modularity gain formula.
}

// Constructs a new graph based on the current community assignments.
// Graph DistCommunities::into_new_graph() {
//   std::unordered_map<int, std::vector<int>> map;

//   for (int i = 0; i < gbl_vtx_to_comm_map.size(); i++) {
//     map[gbl_vtx_to_comm_map[i]].push_back(i);
//   }

//   // Renumber communities to be consecutive starting from 0.
//   std::fill(gbl_vtx_to_comm_map.begin(), gbl_vtx_to_comm_map.end(), -1);
//   int new_comm_number = 0;
//   for (auto pair : map) {
//     for (auto node : pair.second)
//       gbl_vtx_to_comm_map[node] = new_comm_number;
//     new_comm_number++;
//   }

//   // Create new edges based on community connections.
//   std::vector<std::unordered_set<int>> comm_edges(new_comm_number);
//   for (int node = 0; node < gbl_vtx_to_comm_map.size(); node++) {
//     int comm = gbl_vtx_to_comm_map[node];
//     for (auto neighbor : g.neighbors(node))
//       comm_edges[comm].insert(gbl_vtx_to_comm_map[neighbor]);
//   }

//   std::vector<std::pair<int, int>> edge_list;
//   for (int node = 0; node < comm_edges.size(); node++) {
//     for (auto neighbor : comm_edges[node]) {
//       edge_list.push_back(std::make_pair(node, neighbor));
//     }
//   }

//   return Graph(edge_list); // Return a new graph representing the compressed
//                            // community structure.
// }

// void serialize(DistCommunityUpdate& data, std::vector<char>& buffer) {
//     size_t int_size = sizeof(int);
//     size_t double_size = sizeof(double);
//     buffer.resize(3 * int_size + 2 * double_size);
//     char* ptr = buffer.data();

//     // Copy integers
//     std::memcpy(ptr, &data.node, int_size); ptr += int_size;
//     std::memcpy(ptr, &data.node_comm, int_size); ptr += int_size;
//     std::memcpy(ptr, &data.best_comm, int_size); ptr += int_size;

//     // Copy doubles
//     std::memcpy(ptr, &data.node_comm_degree, double_size); ptr += double_size;
//     std::memcpy(ptr, &data.best_comm_degree, double_size);
// }


// void deserialize(std::vector<char>& buffer, DistCommunityUpdate& data) {
//     const char* ptr = buffer.data();
//     size_t int_size = sizeof(int);
//     size_t double_size = sizeof(double);

//     // Copy integers
//     std::memcpy(&data.node, ptr, int_size); ptr += int_size;
//     std::memcpy(&data.node_comm, ptr, int_size); ptr += int_size;
//     std::memcpy(&data.best_comm, ptr, int_size); ptr += int_size;

//     // Copy doubles
//     std::memcpy(&data.node_comm_degree, ptr, double_size); ptr += double_size;
//     std::memcpy(&data.best_comm_degree, ptr, double_size);
// }
