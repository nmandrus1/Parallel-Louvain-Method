#include "community.h"
#include "graph.h"
#include <algorithm>
#include <alloca.h>
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <utility>
#include <vector>

// Constructor for the Communities class.
// Initializes internal data structures and sets up initial community
// assignments where each node is its own community.
Communities::Communities(Graph &g) : g(g) {
  // Resize all vectors to accommodate the graph's vertex count.
  node_to_comm_map.resize(g.local_vcount);
  in.resize(g.local_vcount);
  total.resize(g.local_vcount);
  neighbor_comms.resize(g.local_vcount);
  edges_to_other_comms.resize(g.local_vcount);

  // Initialize communities such that each node is in its own community.
  for (int i = 0; i < g.local_vcount; i++) {
    node_to_comm_map[i] = i;
    total[i] = g.weighted_degree(i); // Total degree of the community is the degree of the node.
    in[i] = 0;              // Initially, no internal edges within the community.
  }
}

// Inserts a node into a community and updates relevant metrics.
void Communities::insert(int node, int community, double node_comm_degree) {
  node_to_comm_map[node] = community;
  total[community] += g.weighted_degree(node);
  in[community] += 2 * node_comm_degree;
}

// Removes a node from a community, updating the internal community structure
// and degree information.
void Communities::remove(int node, int community, double node_comm_degree) {
  node_to_comm_map[node] = -1;
  total[community] -= g.weighted_degree(node);
  in[community] -= 2 * node_comm_degree;
}

// Computes the overall modularity of the graph based on current community
// assignments.
double Communities::modularity() {
  double q = 0.0;
  double m2 = static_cast<double>(g.ecount) * 2.0; // Total weight of all edges in the graph, multiplied by 2.

  for (int i = 0; i < g.local_vcount; i++) {
    if (total[i] > 0)
      q += in[i] / m2 - (total[i] / m2) * (total[i] / m2); // Modularity formula as sum of
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

      compute_neighbors(node); // Update the weights to all neighboring // communities of the node.

      // Temporarily remove the node from its current community.
      remove(node, node_comm, edges_to_other_comms[node_comm]);

      // Determine the best community for this node based on potential
      // modularity gain.
      int best_comm = compute_best_community(node, node_comm);
        
      // Insert the node into the best community found.
      insert(node, best_comm, edges_to_other_comms[best_comm]);

      if (best_comm != node_comm)
        total_num_moves++;
    }

    if (total_num_moves > 0)
      improvement = true;

    
    auto mod = modularity();
    std::cout << "Modularity: " << mod  << std::endl;

    if (total_num_moves == prev_num_moves)
      return improvement; // Return whether there was any improvement in this
                          // iteration.
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
  edges_to_other_comms[node] = 0;

  int node_comm = node_to_comm_map[node];
  edges_to_other_comms[node_comm] = 0;
  neighbor_comms.push_back(node_comm);

  for (auto [neighbor, weight]: g.neighbors(node)) {
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
      edges_to_other_comms[neighbor_comm] += weight;
    }
  }
}

// Computes the potential modularity gain from moving a node to a new community.
double Communities::modularity_gain(int node, int comm,
                                    double node_comm_degree) {
  double totc = static_cast<double>(total[comm]);
  double degc = static_cast<double>(g.weighted_degree(node));
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
  std::unordered_map<int, std::unordered_map<int, double>> new_graph;

  for (int vtx = 0; vtx < node_to_comm_map.size(); vtx++) {
    int comm = node_to_comm_map[vtx];

    for (auto [neighbor, weight]: g.neighbors(vtx)) {
      int neighbor_comm = node_to_comm_map[neighbor];
      if(!new_graph[comm].contains(neighbor_comm)) {
        new_graph[comm][neighbor_comm] = 0.0;
      }  
      new_graph[comm][neighbor_comm] += weight;
    }
  }

  // Build Adjacency List
  Graph::AdjacencyList adj_list;
  for(auto& [vtx, edges]: new_graph) {
    for(auto& [child, weight]: edges) adj_list[vtx].insert({child, weight});
  }

  return Graph(adj_list); // Return a new graph representing the compressed
                           // community structure.
}

void Communities::print_comm_membership() {
  // print comm_ref_counts
  for (int v = g.rows.first; v < g.rows.second; v++) 
     std::cout << "Vtx " << v << " Community: " << node_to_comm_map[v] << "\n";
}
