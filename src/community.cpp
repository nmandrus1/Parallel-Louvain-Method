#include "community.h"
#include <algorithm>

Communities::Communities(Graph &g) : g(g) {
  node_to_comm_map.resize(g.vcount);
  comm_to_degree_map.resize(g.vcount);

  in.resize(g.vcount);
  total.resize(g.vcount);

  neighbor_comms.resize(g.vcount);
  neighbor_weights.resize(g.vcount);

  for(int i = 0; i < g.vcount; i++) {
    node_to_comm_map[i] = i;
    total[i] = g.degree(i);
    in[i] = 0;
  }
}

// Utility methods for tracking Community Info
void Communities::insert(int node, int community, int node_comm_degree) {
  node_to_comm_map[node] = community;
  total[community] += g.degree(node);
  in[community] += 2 * node_comm_degree;
}

// Utility methods for tracking Community Info
void Communities::remove(int node, int community, int node_comm_degree) {
  node_to_comm_map[node] = -1;
  total[community] -= g.degree(node);
  in[community] -= 2 * node_comm_degree;
}

double Communities::modularity() {
  double q  = 0.;
  double m2 = (double)g.ecount * 2;

  for (int i=0 ; i<g.vcount; i++) {
    if (total[i]>0)
      q += (double)in[i]/m2 - ((double)total[i]/m2)*((double)total[i]/m2);
  }

  return q;
}

// one pass of the Louvain Method Algorithm
bool Communities::iterate() {
  int total_num_moves = 0;
  int prev_num_moves = 0;
  bool improvement = false;

  while (true) {
    prev_num_moves = total_num_moves;

    for (int node = 0; node < g.vcount; node++) {
      // remove node from its comm and find the best comm for it
      int node_comm = node_to_comm_map[node];

      // write all all weighst to neighboring communites to member vectors
      compute_neighbors(node);
      // neighbor_comms is now good to access and stores the indecies
      // to the vector neighbor_weights that holds actual community weight info

      // remove node from its community
      remove(node, node_comm, neighbor_weights[node_comm]);

      int best_comm = node_comm;
      double best_increase = 0.0;
      for (auto neighbor_comm : neighbor_comms) {
        double increase = modularity_gain(node, neighbor_comm,
                                          neighbor_weights[neighbor_comm]);
        if (increase > best_increase) {
          best_comm = neighbor_comm;
          best_increase = increase;
        }
      }

      insert(node, best_comm, neighbor_weights[best_comm]);

      if (best_comm != node_comm)
        total_num_moves++;
    }

    if(total_num_moves > 0)
      improvement = true;

    // no changes made
    if(total_num_moves == prev_num_moves)
      return improvement;
  }
}

void Communities::compute_neighbors(int node) {
  // reset this field so its ready to be populated with
  // node's specific neighbors
  neighbor_comms.resize(0);
  std::fill(neighbor_weights.begin(), neighbor_weights.end(), -1.0);

  // loop over each neighbor of our node
  for (int neighbor : g.neighbors(node)) {
    int neighbor_comm = node_to_comm_map[neighbor];

    if (node != neighbor) {
      if (neighbor_weights[neighbor_comm] == -1) {
        neighbor_weights[neighbor_comm] = 0;
        neighbor_comms.push_back(neighbor_comm);
      }
      neighbor_weights[neighbor_comm] += 1.0;
    }
  }
}

double Communities::modularity_gain(int node, int comm,
                                    double node_comm_degree) {
  double totc = (double)total[comm];
  double degc = (double)g.degree(node);
  double m2 = (double)g.ecount * 2;
  double dnc = (double)node_comm_degree;

  return (dnc - totc * degc / m2);
}
