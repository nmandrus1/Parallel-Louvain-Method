#include "community.h"
#include <algorithm>
#include <cstdio>
#include <unordered_set>
#include <utility>
#include <mpi.h>

// Constructor for the Communities class.
// Initializes internal data structures and sets up initial community
// assignments where each node is its own community.
Communities::Communities(Graph &g) : g(g) {
  // Resize all vectors to accommodate the graph's vertex count.
  node_to_comm_map.resize(g.vcount);
  comm_to_degree_map.resize(g.vcount);
  in.resize(g.vcount);
  total.resize(g.vcount);
  neighbor_comms.resize(g.vcount);
  neighbor_weights.resize(g.vcount);

  // Initialize communities such that each node is in its own community.
  for (int i = 0; i < g.vcount; i++) {
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

  for (int i = 0; i < g.vcount; i++) {
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

    for (int node = 0; node < g.vcount; node++) {
      int node_comm = node_to_comm_map[node];

      compute_neighbors(node); // Update the weights to all neighboring
                               // communities of the node.

      // Temporarily remove the node from its current community.
      remove(node, node_comm, neighbor_weights[node_comm]); 

      // Determine the best community for this node based on potential modularity gain.
      int best_comm = compute_best_community(node, node_comm); 

      // Insert the node into the best community found.
      insert(node, best_comm, neighbor_weights[best_comm]); 

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
    double increase =
        modularity_gain(node, neighbor_comm, neighbor_weights[neighbor_comm]);
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
  std::fill(neighbor_weights.begin(), neighbor_weights.end(), -1.0);

  for (int neighbor : g.neighbors(node)) {
    int neighbor_comm = node_to_comm_map[neighbor];

    if (node != neighbor) {
      // if this neighbor community hasn't been seen yet, 
      // initialize it before adding weight to it
      if (neighbor_weights[neighbor_comm] == -1) {
        neighbor_weights[neighbor_comm] = 0;
        neighbor_comms.push_back(neighbor_comm);
      }

      // Increment the edge weight to the neighboring community.
      neighbor_weights[neighbor_comm] += 1.0; 
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
  gbl_vtx_to_comm_map.reserve(g.vcount * 2);
  gbl_comm_to_degree_map.reserve(g.vcount * 2);
  in.reserve(g.vcount * 2);
  total.reserve(g.vcount * 2);
  neighbor_comms.resize(g.vcount);
  neighbor_weights.reserve(g.vcount * 2);
  gbl_vtx_to_gbl_degree.reserve(g.vcount * 2);
  
  // Collect global degrees for every row vtx
  // assuming every mpi proc has the same number of rows

  //   rank 0   rank 1
  // 0 | 0 1 0 || 1 1 0 | -> deg(0) = 3
  // 1 | 0 0 0 || 1 0 0 | -> deg(1) = 1
  // 2 | 0 1 0 || 0 1 0 | -> deg(2) = 2
  //         ....

  // recieve the degree of every vtx in this row and add them 
  // all up for a global degree 
  std::vector<int> send_buf(g.vcount);
  for(int i = 0; i < g.vcount; i++)
    send_buf[i] = g.degree(i);

  std::vector<int> recv_buf(g.vcount * g.info->width, 0);

  MPI_Allgather(send_buf.data(), send_buf.size(), MPI_INT, recv_buf.data(), send_buf.size(), MPI_INT, g.info->row_comm);

  // sum up all degrees for each vertex in this global row
  for(int i = 0; i < g.vcount; i++) {
    for(int r = 0; r < g.info->width; r++)
      gbl_vtx_to_gbl_degree[g.localRowToGlobal(i)] += recv_buf[i + (g.vcount * r)];
  }

  recv_buf.clear();
  if(g.info->grid_col == g.info->grid_row) {
    // globally index row vertices
    // add global degree of row vertex to recv buf and send 
    // to all column processes 
    for(int v = g.rows.first; v < g.rows.second; v++)
      recv_buf.push_back(gbl_vtx_to_gbl_degree[v]);
  }
    
  MPI_Bcast(recv_buf.data(), g.vcount, MPI_INT, g.info->grid_col, g.info->col_comm);

  if(g.info->grid_col != g.info->grid_row) {
    for(int v = 0; v < g.vcount; v++) {
      gbl_vtx_to_gbl_degree[g.localColToGlobal(v)] = recv_buf[v];
    }
  }

  // Initialize communities such that each row node is in its own community.
  // global indexing
  for (int v = g.rows.first; v < g.rows.second; v++) {
    gbl_vtx_to_comm_map[v] = v;
    total[v] = gbl_vtx_to_gbl_degree[v];  // Total degree of the community is the degree of the node.
    in[v] = 0;                            // Initially, no internal edges within the community.
  }

  // do the same for columns of the processes that have different row/col vertices
  if(g.info->grid_row != g.info->grid_col) {
    for(int v = g.columns.first; v < g.columns.second; v++) {
      gbl_vtx_to_comm_map[v] = v;
      total[v] = gbl_vtx_to_gbl_degree[v];  // Total degree of the community is the degree of the node.
      in[v] = 0;                            // Initially, no internal edges within the community.
    }
  }
}

// Inserts a node into a community and updates relevant metrics.
void DistCommunities::insert(int node, int community, int node_comm_degree) {
  gbl_vtx_to_comm_map[node] = community;
  total[community] += gbl_vtx_to_gbl_degree[node];
  in[community] += 2 * node_comm_degree;
}

// Removes a node from a community, updating the internal community structure
// and degree information.
void DistCommunities::remove(int node, int community, int node_comm_degree) {
  gbl_vtx_to_comm_map[node] = -1;
  total[community] -= gbl_vtx_to_gbl_degree[node];
  in[community] -= 2 * node_comm_degree;
}

// Computes the overall modularity of the graph based on current community
// assignments.
double DistCommunities::modularity() {
  double q = 0.0;
  if(g.info->grid_col == 0) {
    double m2 = static_cast<double>(g.ecount) * 2; // Total weight of all edges in the graph, multiplied by 2.

    for (int v = g.rows.first; v < g.rows.second; v++) {
      if (total[v] > 0)
        q += in[v] / m2 - (total[v] / m2) * (total[v] / m2); // Modularity formula as sum of
                                                             // each community's contribution.
    }

    MPI_Allreduce(&q, &q, 1, MPI_DOUBLE, MPI_SUM, g.info->col_comm);
  }
  return q;
}

// Executes one pass of the Louvain method algorithm, attempting to improve the
// modularity by moving nodes between communities.
bool DistCommunities::iterate() {
  int total_num_moves = 0;
  int prev_num_moves = 0;
  bool improvement = false, all_finished = false;
  std::vector<char> community_update_buf(sizeof(DistCommunityUpdate));

  std::cout << "RANK " << g.info->rank << ": Entering iterate() while loop" << std::endl;
  while (true) {
    prev_num_moves = total_num_moves;

    for (int node = g.rows.first; node < g.rows.second; node++) {
      int node_comm = gbl_vtx_to_comm_map[node];

      std::cout << "RANK " << g.info->rank << ": Computing neighbors" << std::endl;
      compute_neighbors(node); // Update the weights to all neighboring
                               // communities of the node.

      // Temporarily remove the node from its current community.
      remove(node, node_comm, neighbor_weights[node_comm]); 

      std::cout << "RANK " << g.info->rank << ": Computing best community" << std::endl;
      // Determine the best community for this node based on potential modularity gain.
      int best_comm = compute_best_community( node, node_comm); 

      // Insert the node into the best community found.
      insert(node, best_comm, neighbor_weights[best_comm]); 

      // Communicate Community updates
      // Every process in this row should have the best community for this iteration
      // so now we just need to update every process that has any neighbor vertices that 
      // are vertices of THIS process's rows.

      DistCommunityUpdate update;
      if(g.info->grid_col == g.info->grid_row) {
        update.node = node;
        update.node_comm = node_comm;
        update.best_comm = best_comm;
        update.node_comm_degree = neighbor_weights[node_comm];
        update.best_comm_degree = neighbor_weights[best_comm];

        serialize(update, community_update_buf);
        std::cout << "RANK " << g.info->rank << ": Broadcasting update" << std::endl;
      }

      MPI_Bcast(community_update_buf.data(), community_update_buf.size(), MPI_CHAR, g.info->grid_col, g.info->col_comm);

      if(g.info->grid_col != g.info->grid_row) {
        std::cout << "RANK " << g.info->rank << ": Receiving update" << std::endl;
        deserialize(community_update_buf, update);
        // remove from old community, and update our total and in vectors
        remove(update.node, update.node_comm, update.node_comm_degree);
        insert(update.node, update.best_comm, update.best_comm_degree);
      }

      if (best_comm != node_comm)
        total_num_moves++;

      std::cout << "RANK " << g.info->rank << ": Waiting at Barrier" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (total_num_moves > 0)
      improvement = true;

    if (total_num_moves == prev_num_moves) 
      all_finished = true;

    std::cout << "RANK " << g.info->rank << ": Communicating completion status" << std::endl;
    MPI_Allreduce(&all_finished, &all_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(all_finished) return improvement; // Return whether there was any improvement in this iteration.


    auto mod = modularity();
    if(g.info->rank == 0)
      std::cout << "Modularity: " << mod << std::endl; 

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// Given a node and its current community, computes the best community for this
// node that would increase the modularity the most.
int DistCommunities::compute_best_community(int node, int node_comm) {
  int best_comm = node_comm;
  double best_increase = 0.0;
  for (auto neighbor_comm : neighbor_comms) {
    double increase = modularity_gain(node, neighbor_comm, neighbor_weights[neighbor_comm]);
    if (increase > best_increase) {
      best_comm = neighbor_comm;
      best_increase = increase;
    }
  }

  // Row wise communicaiton to determine best community 
  std::vector<int> best_comm_recv_buf(g.info->width);
  std::vector<double> best_increase_recv_buf(g.info->width);

  MPI_Allgather(&best_comm, 1, MPI_INT, best_comm_recv_buf.data(), 1, MPI_INT, g.info->row_comm);
  MPI_Allgather(&best_increase, 1, MPI_DOUBLE, best_increase_recv_buf.data(), 1, MPI_DOUBLE, g.info->row_comm);

  for(int rank = 0; rank < g.info->width; rank++) {
    double increase = best_increase_recv_buf[rank];
    if (increase > best_increase) {
      best_comm = best_comm_recv_buf[rank];
      best_increase = increase;
    }
  }

  return best_comm;
}

// Computes and updates internal structures with weights corresponding to each
// neighboring community of a given node.
void DistCommunities::compute_neighbors(int node) {
  neighbor_comms.clear();
  neighbor_weights.clear();

  for (int neighbor : g.neighborsGlobalIdxs(node)) {
    int neighbor_comm = gbl_vtx_to_comm_map[neighbor];

    if (node != neighbor) {
      // if this neighbor community hasn't been seen yet, 
      // initialize it before adding weight to it
      if (!neighbor_weights.contains(neighbor_comm)) {
        neighbor_weights[neighbor_comm] = 0;
        neighbor_comms.push_back(neighbor_comm);
      }

      // Increment the edge weight to the neighboring community.
      neighbor_weights[neighbor_comm] += 1.0; 
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

void serialize(DistCommunityUpdate& data, std::vector<char>& buffer) {
    size_t int_size = sizeof(int);
    size_t double_size = sizeof(double);
    buffer.resize(3 * int_size + 2 * double_size);
    char* ptr = buffer.data();

    // Copy integers
    std::memcpy(ptr, &data.node, int_size); ptr += int_size;
    std::memcpy(ptr, &data.node_comm, int_size); ptr += int_size;
    std::memcpy(ptr, &data.best_comm, int_size); ptr += int_size;

    // Copy doubles
    std::memcpy(ptr, &data.node_comm_degree, double_size); ptr += double_size;
    std::memcpy(ptr, &data.best_comm_degree, double_size);
}


void deserialize(std::vector<char>& buffer, DistCommunityUpdate& data) {
    const char* ptr = buffer.data();
    size_t int_size = sizeof(int);
    size_t double_size = sizeof(double);

    // Copy integers
    std::memcpy(&data.node, ptr, int_size); ptr += int_size;
    std::memcpy(&data.node_comm, ptr, int_size); ptr += int_size;
    std::memcpy(&data.best_comm, ptr, int_size); ptr += int_size;

    // Copy doubles
    std::memcpy(&data.node_comm_degree, ptr, double_size); ptr += double_size;
    std::memcpy(&data.best_comm_degree, ptr, double_size);
}
