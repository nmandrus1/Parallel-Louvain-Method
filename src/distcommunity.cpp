#include "distcommunity.h"
#include "graph.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <alloca.h>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stdlib.h>
#include <random>


void create_degree_info_datatype(MPI_Datatype* dt);
void create_community_update_datatype(MPI_Datatype* dt);


struct DegreeInfo {
  int vtx;
  double degree;
};

// Constructor for the DistCommunities class.
// Initializes internal data structures and sets up initial community
// assignments where each node is its own community.
DistCommunities::DistCommunities(Graph &g) : g(g) { 
  create_degree_info_datatype(&MPI_DEGREE_INFO);
  create_community_update_datatype(&MPI_COMMUNITY_UPDATE);

  init();
}

void DistCommunities::init() {
  // Resize all vectors to accommodate the graph's vertex count.
  gbl_vtx_to_comm_map.reserve(g.local_vcount * 2);
  in.reserve(g.local_vcount * 2);
  total.reserve(g.local_vcount * 2);
  neighbor_comms.resize(g.local_vcount);
  edges_to_other_comms.reserve(g.local_vcount * 2);
  std::unordered_map<int,double> neighbor_degree(g.local_vcount * 2);

  // map rank to list of vertices that have an edge in that rank
  std::unordered_map<int, std::unordered_set<int>> msg_map;

  // 2d array of buffers to send to other ranks idx = rank
  std::vector<std::vector<DegreeInfo>> send_bufs(g.info.comm_size);
  std::vector<int> send_counts(g.info.comm_size, 0);

  for (int v = g.rows.first; v < g.rows.second; v++) {
    double weighted_degree = g.weighted_degree(v);
    for (auto [n, weight] : g.neighbors(v)) {
      int owner = g.getRankOfOwner(n);

      if (owner != g.info.rank) {
        if(!msg_map[owner].insert(v).second) continue;

          std::cout << "RANK " << g.info.rank << ": Owner of vtx " << n << ": " << owner << " \t local_vcount: " << g.local_vcount  << " global_vcount: " << g.global_vcount << std::endl;
          send_bufs[owner].push_back({v, weighted_degree});
          send_counts[owner]++;
      }
    }
  }

  std::vector<int> sdispls(g.info.comm_size);
  int total_send = 0;
  for (int i = 0; i < g.info.comm_size; ++i) {
      sdispls[i] = total_send;
      total_send += send_counts[i];
  }

  // Assuming recv_counts and rdispls are similarly calculated
  std::vector<int> recv_counts(g.info.comm_size, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> rdispls(g.info.comm_size);
  // The total receive size would need to be calculated by all processes collectively, potentially using MPI_Alltoall to share the send_counts
  // Calculate receive displacements
  int total_receive = 0; // This will accumulate the total number of items to receive
  for (int i = 0; i < g.info.comm_size; ++i) {
      rdispls[i] = total_receive;
      total_receive += recv_counts[i];
  }

  // concatenate send_bufs into a single buffer for sending
  std::vector<DegreeInfo> send_data;
  send_data.reserve(total_send);
  for (const auto& buf : send_bufs) {
      send_data.insert(send_data.end(), buf.begin(), buf.end());
  }

  std::vector<DegreeInfo> recv_data(total_receive);  // total_receive needs to be calculated

  MPI_Alltoallv(send_data.data(), send_counts.data(), sdispls.data(), MPI_DEGREE_INFO,
                recv_data.data(), recv_counts.data(), rdispls.data(), MPI_DEGREE_INFO, MPI_COMM_WORLD);



  for(auto info: recv_data) neighbor_degree[info.vtx] = info.degree;

  // Initialize communities such that each row node is in its own community.
  // global indexing
  for (int v = g.rows.first; v < g.rows.second; v++) {
    gbl_vtx_to_comm_map[v] = v;
    comm_size[v] = 1;
    total[v] = g.weighted_degree(v); // Total degree of the community is the degree of the node.
    in[v] = 0;              // Initially, no internal edges within the community.

    for (auto [n, weight] : g.neighbors(v)) {
      int neighbor_owner = g.getRankOfOwner(n);

      vtx_rank_degree[v][neighbor_owner]++;
      comm_ref_count[v][neighbor_owner]++;

      if (neighbor_owner != g.info.rank)
        rank_to_border_vertices[neighbor_owner].insert(v);

      if (gbl_vtx_to_comm_map.contains(n))
        continue;
      gbl_vtx_to_comm_map[n] = n;
      total[n] = neighbor_degree[n];
      in[n] = 0;
    }
  }
}

DistCommunities::~DistCommunities() {
  MPI_Type_free(&MPI_DEGREE_INFO);
  MPI_Type_free(&MPI_COMMUNITY_UPDATE);
}

// Inserts a node into a community and updates relevant metrics.
void DistCommunities::insert(int node, int community, double degree,
                             int edges_within_comm,
                             std::unordered_map<int, int> &rank_counts) {
  gbl_vtx_to_comm_map[node] = community;
  comm_size[community]++;
  total[community] += degree;
  in[community] += 2 * edges_within_comm;
  if(comm_size[community] == 1 | comm_size[community] == 0) in[community] = 0;

  // only update reference counts if we own this row
  if(g.in_row(community)) {
    for (auto &[rank, count] : rank_counts)
      comm_ref_count[community][rank] += count;
  }
}

// Removes a node from a community, updating the internal community structure
// and degree information.
void DistCommunities::remove(int node, int community, double degree,
                             int edges_within_comm,
                             std::unordered_map<int, int> &rank_counts) {
  gbl_vtx_to_comm_map[node] = -1;
  comm_size[community]--;
  total[community] -= degree;
  in[community] -= 2 * edges_within_comm;
  if(comm_size[community] == 1 | comm_size[community] == 0) in[community] = 0;

  if(g.in_row(community)) {
    for (auto &[rank, count] : rank_counts)
      comm_ref_count[community][rank] -= count;
  }
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

  MPI_Allreduce(MPI_IN_PLACE, &q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return q;
}

bool DistCommunities::one_level() {

  #ifdef PROFILE_FNS
  GPTLstart("iterate");
  #endif

  bool ret = iterate();

  #ifdef PROFILE_FNS
  GPTLstop("iterate");
  #endif

  #ifdef PROFILE_FNS
  GPTLstart("reconstruction");
  #endif

  into_new_graph();

  #ifdef PROFILE_FNS
  GPTLstop("reconstruction");
  #endif

  return ret;
}

// Executes one pass of the Louvain method algorithm, attempting to improve the
// modularity by moving nodes between communities.
bool DistCommunities::iterate() {
  int total_num_moves = 0;
  int prev_num_moves = 0;
  bool improvement = false, all_finished = false;

  // create and fill vertices with all our local rows
  std::vector<int> vertices(g.local_vcount);
  std::iota(vertices.begin(), vertices.end(), g.rows.first);

  std::default_random_engine eng;
  eng.seed(g.info.rank);

  // negative for exponential decay of temperature as a function of the number of 
  // iterations over every vertex. Higher the number of iterations, the less likely 
  // it should be for vertices to move around
  int iteration = -1;
  double temperature;

  while (true) {
    temperature = std::exp(static_cast<double>(iteration));

    prev_num_moves = total_num_moves;
    std::shuffle(vertices.begin(), vertices.end(), eng);

    for (int vtx: vertices) {
      int vtx_comm = gbl_vtx_to_comm_map[vtx];

      // std::cout << "RANK " << g.info.rank << ": Computing neighbors" << std::endl;
      #ifdef PROFILE_FNS
      GPTLstart("compute_neighbors");
      #endif
      compute_neighbors(vtx); // Update the weights to all neighboring // communities of the node.
      #ifdef PROFILE_FNS
      GPTLstop("compute_neighbors");
      #endif

      // Temporarily remove the node from its current community.
      // std::cout << "RANK " << g.info.rank << ": Removing vtx " << vtx << " from comm " << vtx_comm << std::endl;

      #ifdef PROFILE_FNS
      GPTLstart("remove");
      #endif
      remove(vtx, vtx_comm, g.weighted_degree(vtx), edges_to_other_comms[vtx_comm], vtx_rank_degree[vtx]);
      #ifdef PROFILE_FNS
      GPTLstop("remove");
      #endif

      // Determine the best community for this node based on potential
      // modularity gain.

      // std::cout << "RANK " << g.info.rank << ": computing best comm" << std::endl;

      #ifdef PROFILE_FNS
      GPTLstart("compute_best_community");
      #endif
      auto best_comm = compute_best_community(vtx, vtx_comm, temperature);
      #ifdef PROFILE_FNS
      GPTLstop("compute_best_community");
      #endif

      // change communities
      if(best_comm != vtx_comm) {
        // std::cout << "RANK " << g.info.rank << ": best comm = " << best_comm << std::endl;

        // calculate the ranks that own the old and new communities
        int old_comm_owner = g.getRankOfOwner(vtx_comm);
        int new_comm_owner = g.getRankOfOwner(best_comm);

        CommunityUpdate update = { CommunityUpdate::Removal,
                                  vtx,
                                  g.weighted_degree(vtx),
                                  vtx_comm,
                                  best_comm,
                                  edges_to_other_comms[vtx_comm],
                                  edges_to_other_comms[best_comm],
                                  (int)vtx_rank_degree[vtx].size()};

        // std::cout << "RANK " << g.info.rank << ": Communicating removal/addition \t old: " << old_comm_owner << " new: " << new_comm_owner << std::endl;

        if (old_comm_owner != g.info.rank) {
          #ifdef PROFILE_FNS
          GPTLstart("send_community_update");
          #endif
          send_community_update(old_comm_owner, update);
          #ifdef PROFILE_FNS
          GPTLstop("send_community_update");
          #endif
        } else comms_updated_this_iter.insert(vtx_comm);
        // No else required since if the old owner was this rank then it was
        // removed properly by the remove() call

        if (new_comm_owner != g.info.rank) {
          update.type = CommunityUpdate::Addition;
          #ifdef PROFILE_FNS
          GPTLstart("send_community_update");
          #endif
          send_community_update(new_comm_owner, update);
          #ifdef PROFILE_FNS
          GPTLstop("send_community_update");
          #endif
        } else comms_updated_this_iter.insert(best_comm);
          total_num_moves++;
      }

      // no matter whether comms changed, vtx must be placed in a community for now
      // NOTE: subject to change if vertex rejections are implemented
      #ifdef PROFILE_FNS
      GPTLstart("insert");
      #endif
      insert(vtx, best_comm, g.weighted_degree(vtx), edges_to_other_comms[best_comm], vtx_rank_degree[vtx]);
      #ifdef PROFILE_FNS
      GPTLstop("insert");
      #endif

      MPI_Barrier(MPI_COMM_WORLD);

      // recieve updates from remote process about nodes joining local community
      #ifdef PROFILE_FNS
      GPTLstart("process_incoming_updates ");
      #endif
      process_incoming_updates();
      #ifdef PROFILE_FNS
      GPTLstop("process_incoming_updates ");
      #endif

      #ifdef PROFILE_FNS
      GPTLstart("update_subscribers");
      #endif
      update_subscribers();
      #ifdef PROFILE_FNS
      GPTLstop("update_subscribers");
      #endif

      comms_updated_this_iter.clear();

      #ifdef PROFILE_FNS
      GPTLstart("update_neighbors");
      #endif
      update_neighbors(vtx);
      #ifdef PROFILE_FNS
      GPTLstop("update_neighbors");
      #endif
    }

    if (total_num_moves > 0)
      improvement = true;

    if (total_num_moves == prev_num_moves)
      all_finished = true;

    MPI_Allreduce(MPI_IN_PLACE, &all_finished, 1, MPI_C_BOOL, MPI_LAND,
                  MPI_COMM_WORLD);
    if (all_finished)
      return improvement; // Return whether there was any improvement in this
                          // iteration.
    // print_comm_membership();

    #ifdef PROFILE_FNS
    GPTLstart("modularity");
    #endif
    auto mod = modularity();
    #ifdef PROFILE_FNS
    GPTLstop("modularity");
    #endif

    if (g.info.rank == 0) {
      std::cout << "Modularity: " << mod  << "\nTemp: " << temperature << std::endl;
      
    }

    // negative for exponential decay, see variable declaration
    iteration -= 1;
  }
}

void DistCommunities::send_community_update(int dest, const CommunityUpdate& update) {
  MPI_Request req;
  MPI_Isend(&update, 1, MPI_COMMUNITY_UPDATE, dest, MPI_DATA_TAG, MPI_COMM_WORLD, &req);

  std::vector<int> send_buf;
  for (auto &[rank, count] : vtx_rank_degree[update.node]) {
    send_buf.push_back(rank);
    send_buf.push_back(count);
  }

  MPI_Wait(&req, MPI_STATUS_IGNORE);

  MPI_Send(send_buf.data(), send_buf.size(), MPI_INT, dest, MPI_DATA_TAG, MPI_COMM_WORLD);
}

void DistCommunities::process_incoming_updates() {
  MPI_Status status;
  int flag;
  CommunityUpdate update;
  std::vector<int> buf;
  std::unordered_map<int, int> rank_counts;


  // Continue processing as long as there are messages
  while (true) {
    // Check for any incoming message
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_DATA_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      MPI_Recv(&update, 1, MPI_COMMUNITY_UPDATE, status.MPI_SOURCE, MPI_DATA_TAG, MPI_COMM_WORLD, &status);

      buf.clear();
      buf.resize(update.num_ranks_bordering_node * 2);
      MPI_Recv(buf.data(), buf.size(), MPI_INT, status.MPI_SOURCE, MPI_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < 2 * update.num_ranks_bordering_node; i += 2) {
        int rank = buf[i], count = buf[i + 1];
        rank_counts[rank] = count;
      }

      if (update.type == CommunityUpdate::Addition) {
        insert(update.node, update.new_comm, update.global_degree, update.edges_within_new_comm, rank_counts);
        comms_updated_this_iter.insert(update.new_comm);
      } else {
        remove(update.node, update.old_comm, update.global_degree, update.edges_within_old_comm, rank_counts);
        comms_updated_this_iter.insert(update.old_comm);
      }

      rank_counts.clear();
    } else {
      // No more messages to process
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
  }
}

void DistCommunities::update_subscribers() {
  MPI_Status status;
  int flag;
  CommunityInfo comm_info;

  for (auto &comm: comms_updated_this_iter) {
    CommunityInfo updated_info = {comm, in[comm], total[comm]};
    for (auto &[rank, count] : comm_ref_count[comm]) {
      if(rank == g.info.rank) continue;

      MPI_Send(&updated_info, sizeof(CommunityInfo), MPI_BYTE, rank,
               MPI_COMM_SYNC, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Check for incoming community updates
  while (true) {
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_COMM_SYNC, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      MPI_Recv(&comm_info, sizeof(CommunityInfo), MPI_BYTE, MPI_ANY_SOURCE,
               MPI_COMM_SYNC, MPI_COMM_WORLD, &status);

      // Process the received community info
      in[comm_info.comm] = comm_info.in;
      total[comm_info.comm] = comm_info.total;
    } else {
      // No more updates available at this time
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
  }

  // cleanup reference counts AFTER final messages have been sent about
  // community information
  for (auto comm_it = comm_ref_count.begin(); comm_it != comm_ref_count.end();
       /* no increment here */) {
    // Iterate over each rank in the inner map
    for (auto rank_it = comm_it->second.begin();
         rank_it != comm_it->second.end();
         /* no increment here */) {
      if (rank_it->second == 0) {
        // Erase the rank entry with zero references
        rank_it = comm_it->second.erase(rank_it);
      } else {
        ++rank_it;
      }
    }

    // After cleaning up ranks, check if the community has any ranks left
    if (comm_it->second.empty()) {
      // If no ranks left, erase the community entry
      comm_it = comm_ref_count.erase(comm_it);
    } else {
      ++comm_it;
    }
  }
}


// update the neighboring ranks of vtx that it's joined a new community
void DistCommunities::update_neighbors(int vtx) {
  MPI_Status status;
  int flag;
  std::vector<int> buf;

  // loop over all the ranks that border this vtx and send them neighbor information
  for (auto& [rank, count]: vtx_rank_degree[vtx]) {
    buf.push_back(vtx);
    buf.push_back(gbl_vtx_to_comm_map[vtx]);

    MPI_Send(buf.data(), buf.size(), MPI_INT, rank, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD);
    buf.clear();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int size;

  // Check for incoming community updates
  while (true) {
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      MPI_Get_count(&status, MPI_INT, &size);
      buf.resize(size);

      MPI_Recv(buf.data(), size, MPI_INT, status.MPI_SOURCE, MPI_NEIGHBOR_SYNC, MPI_COMM_WORLD, &status);

      for(int i = 0; i < buf.size(); i += 2) {
        int neighbor = buf[i], comm = buf[i+1];
        gbl_vtx_to_comm_map[neighbor] = comm;
      }

    } else {
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
  }
}

// Given a node and its current community, computes the best community for this
// node that would increase the modularity the most.
int DistCommunities::compute_best_community(int node, int node_comm, double temperature) {
  int best_comm = node_comm;
  double best_increase = 0.0;
  for (auto neighbor_comm : neighbor_comms) {
    double increase = modularity_gain(node, neighbor_comm, edges_to_other_comms[neighbor_comm]);

    if (increase > best_increase && std::abs(best_increase - increase) > temperature) {
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

  int node_comm = gbl_vtx_to_comm_map[node];
  edges_to_other_comms[node_comm] = 0;
  neighbor_comms.push_back(node_comm);

  for (auto [neighbor, weight]: g.neighbors(node)) {
    int neighbor_comm = gbl_vtx_to_comm_map[neighbor];

    if (node != neighbor) {
      // if this neighbor community hasn't been seen yet,
      // initialize it before adding weight to it
      if (!edges_to_other_comms.contains(neighbor_comm)) {
        edges_to_other_comms[neighbor_comm] = 0.0;
        neighbor_comms.push_back(neighbor_comm);
      }

      // Increment the edge weight to the neighboring community.
      edges_to_other_comms[neighbor_comm] += weight;
    }
  }
}

// Computes the potential modularity gain from moving a node to a new community.
double DistCommunities::modularity_gain(int node, int comm,
                                        double node_comm_degree) {
  double totc = static_cast<double>(total[comm]);
  double degc = static_cast<double>(g.weighted_degree(node));
  double m2 = static_cast<double>(g.ecount) * 2;
  double dnc = static_cast<double>(node_comm_degree);

  return (dnc - totc * degc / m2); // Modularity gain formula.
}

void DistCommunities::print_comm_ref_counts() const {

  // print comm_ref_counts
  for (int i = 0; i < g.info.comm_size; i++) {
    if (g.info.rank == i) {
      std::cout << "RANK " << g.info.rank << ": Community Reference Counts\n";
      for (int v = g.rows.first; v < g.rows.second; v++) {
        std::cout << "\tCommunity " << v << "\n";
        auto refs = comm_ref_count.find(v);
        if(refs == comm_ref_count.end()) continue;

        for (auto &[rank, count] : refs->second)
          std::cout << "\t\tConnections to rank " << rank << ": " << count
                    << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void DistCommunities::print_comm_membership() const {

  // print comm_ref_counts
  for (int i = 0; i < g.info.comm_size; i++) {
    if (g.info.rank == i) {
      for (int v = g.rows.first; v < g.rows.second; v++) 
         std::cout << "RANK " << g.info.rank << ": Vtx " << v << " Community: " << gbl_vtx_to_comm_map.find(v)->second << "\n";
      std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void DistCommunities::write_communities_to_file(const std::string& directory) const {
    // Prepare data for all communities on this rank
    std::unordered_map<int, std::vector<int>> local_community_to_vertices;
    for (int v = g.rows.first; v < g.rows.second; v++) {
        int community_id = gbl_vtx_to_comm_map.find(v)->second;
        local_community_to_vertices[community_id].push_back(v);
    }
    // Each rank prepares a send buffer with all its community data
    std::vector<int> send_data;
    for (const auto& [community_id, vertices] : local_community_to_vertices) {
        // Append community_id, the number of vertices, and the vertices themselves
        send_data.push_back(community_id);
        send_data.push_back(vertices.size());
        send_data.insert(send_data.end(), vertices.begin(), vertices.end());
    }
    
    // Gather sizes to determine buffer sizes
    int local_send_count = send_data.size();
    std::vector<int> recv_counts(g.info.comm_size);
    MPI_Allgather(&local_send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for allgatherv
    std::vector<int> displacements(g.info.comm_size, 0);
    std::partial_sum(recv_counts.begin(), recv_counts.end() - 1, displacements.begin() + 1);
    
    // Prepare receive buffer
    int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
    std::vector<int> recv_data(total_recv);
    
    // Perform the allgatherv
    MPI_Allgatherv(send_data.data(), local_send_count, MPI_INT,
                   recv_data.data(), recv_counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    // Process received data and aggregate communities
    std::unordered_map<int, std::vector<int>> global_community_to_vertices;
    int index = 0;
    while (index < total_recv) {
        int community_id = recv_data[index++];
        int num_vertices = recv_data[index++];
        global_community_to_vertices[community_id].insert(
            global_community_to_vertices[community_id].end(),
            recv_data.begin() + index, recv_data.begin() + index + num_vertices
        );
        index += num_vertices;
    }

    // Each rank writes out the communities it owns
    for (const auto& [community_id, vertices] : global_community_to_vertices) {
        if (g.getRankOfOwner(community_id) == g.info.rank) {
            std::ostringstream filename;
            filename << directory << "/community_" << community_id << ".txt";
            std::ofstream outfile(filename.str());
            if (!outfile.is_open()) {
                std::cerr << "Failed to open file for writing: " << filename.str() << std::endl;
                continue;
            }
            outfile << "Community " << community_id << " (" << vertices.size() << " members): {";
            for (size_t i = 0; i < vertices.size(); ++i) {
                outfile << vertices[i];
                if (i < vertices.size() - 1) {
                    outfile << ", ";
                }
            }
            outfile << "}\n";
            outfile.close();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (g.info.rank == 0) {
        std::cout << "All ranks have finished writing community data to " << directory << std::endl;
    }
}

void create_degree_info_datatype(MPI_Datatype* dt) { 
  int lengths[] = {1, 1};
  // Calculate displacements
  // In C, by default padding can be inserted between fields. MPI_Get_address will allow
  // to get the address of each struct field and calculate the corresponding displacement
  // relative to that struct base address. The displacements thus calculated will therefore
  // include padding if any.
  MPI_Aint displacements[2];
  DegreeInfo dummy_info;
  MPI_Aint base_address;
  MPI_Get_address(&dummy_info, &base_address);
  MPI_Get_address(&dummy_info.vtx, &displacements[0]);
  MPI_Get_address(&dummy_info.degree, &displacements[1]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);

  MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE};
  MPI_Type_create_struct(2, lengths, displacements, types, dt);
  MPI_Type_commit(dt);
}

void create_community_update_datatype(MPI_Datatype* dt) { 
  int lengths[] = {1, 1, 1, 1, 1, 1, 1, 1};
  // Calculate displacements
  // In C, by default padding can be inserted between fields. MPI_Get_address will allow
  // to get the address of each struct field and calculate the corresponding displacement
  // relative to that struct base address. The displacements thus calculated will therefore
  // include padding if any.
  MPI_Aint displacements[8];
  CommunityUpdate dummy_update;
  MPI_Aint base_address;
  MPI_Get_address(&dummy_update, &base_address);
  MPI_Get_address(&dummy_update.type, &displacements[0]);
  MPI_Get_address(&dummy_update.node, &displacements[1]);
  MPI_Get_address(&dummy_update.global_degree, &displacements[2]);
  MPI_Get_address(&dummy_update.old_comm, &displacements[3]);
  MPI_Get_address(&dummy_update.new_comm, &displacements[4]);
  MPI_Get_address(&dummy_update.edges_within_old_comm, &displacements[5]);
  MPI_Get_address(&dummy_update.edges_within_new_comm, &displacements[6]);
  MPI_Get_address(&dummy_update.num_ranks_bordering_node, &displacements[7]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);
  displacements[3] = MPI_Aint_diff(displacements[3], base_address);
  displacements[4] = MPI_Aint_diff(displacements[4], base_address);
  displacements[5] = MPI_Aint_diff(displacements[5], base_address);
  displacements[6] = MPI_Aint_diff(displacements[6], base_address);
  displacements[7] = MPI_Aint_diff(displacements[7], base_address);

  MPI_Datatype types[8] = { MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
  MPI_Type_create_struct(8, lengths, displacements, types, dt);
  MPI_Type_commit(dt);
}

// Build a new graph based on community information gathered during iteration
Graph DistCommunities::into_new_graph() {
  // TODO: I think enough information is sent in CommunityUpdates
  //       to keep track of edges to other communities

  // renumber edges
  //   - Count local communities = N
  //   - Send N to process P + 1 so it can start labeling new comms at N + 1
  // 
  // gather weights of edges into other comms
  //   - self loop = in[comm]
  //   - loop over all vertices and sum edges to other comms
  //   - send data if your comm owner is remote
  // 
  // build new graph from adjacency list

  unsigned long unique_count = comm_ref_count.size(); 
  std::vector<int> recv_counts(g.info.comm_size);

  // share uniuqe counts and start renumbering communities
  MPI_Allgather(&unique_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // the first number to label our unique communities
  int prefix = 0;
  for(unsigned i = 0; i < g.info.rank; i++)
    prefix += recv_counts[i];

  // renumbered
  std::unordered_map<int, int> old_to_new_comm(unique_count);
  std::vector<int> comm_update_send_buf;

  Graph::EdgeList self_edges; // vector of self loops for local comms 
  for(auto& [comm, refs]: comm_ref_count) {   
    old_to_new_comm[comm] = prefix;
    comm_update_send_buf.push_back(comm);
    comm_update_send_buf.push_back(prefix);
    self_edges.push_back({prefix, prefix, in[comm]});

    prefix += 1;
  }

  // Alltoallv common vectors
  std::vector<int> send_counts(g.info.comm_size, 0);
  std::vector<int> sdispls(g.info.comm_size);
  std::vector<int> rdispls(g.info.comm_size);
  
  unsigned num_communities = 0;
  for(int rank = 0; rank < g.info.comm_size; rank++) {
    rdispls[rank] = num_communities * 2; 
    num_communities += recv_counts[rank];
    recv_counts[rank] *= 2;
  }

  old_to_new_comm.reserve(num_communities);

  std::vector<int> comm_update_recv_buf(num_communities * 2);
  MPI_Allgatherv(comm_update_send_buf.data(), comm_update_send_buf.size(), MPI_INT, comm_update_recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);

  // loop over the buffer and update ids
  for(int i = 0; i < comm_update_recv_buf.size(); i+=2) 
    old_to_new_comm[comm_update_recv_buf[i]] = comm_update_recv_buf[i+1];

  recv_counts.clear();
  rdispls.clear();

  // loop over every vertex
  // look for neighbors that arent in its community
  //    - add the edge weight to the current edge sum

  // Map comm to neighbor comm and the weight between them
  std::unordered_map<int, std::unordered_map<int, double>> msg_map;

  for(int v = g.rows.first; v < g.rows.second; v++) {
    int comm = gbl_vtx_to_comm_map[v];
    for(auto [neighbor, weight]: g.neighbors(v)) {
      int neighbor_comm = gbl_vtx_to_comm_map[neighbor];
      if(neighbor_comm == comm) continue;
      msg_map[comm][neighbor_comm] += weight;
    }
  }

  // gather a all the info to send to each rank
  std::vector<std::vector<Edge>> send_bufs(g.info.comm_size);
  for(auto& [comm, edges]: msg_map) {
    int owner = g.getRankOfOwner(comm);
    for(auto& [neighbor_comm, weight]: edges) {
      send_bufs[owner].push_back({old_to_new_comm[comm], old_to_new_comm[neighbor_comm], weight});
      send_counts[owner] += 1;
    }
  }

  // prepare for Alltoallv
  int total_send = 0;
  for (int i = 0; i < g.info.comm_size; ++i) {
      sdispls[i] = total_send;
      total_send += send_counts[i];
  } 

  // Assuming recv_counts and rdispls are similarly calculated
  recv_counts.resize(g.info.comm_size);
  rdispls.resize(g.info.comm_size);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // The total receive size would need to be calculated by all processes collectively, potentially using MPI_Alltoall to share the send_counts
  // Calculate receive displacements
  int total_receive = 0; // This will accumulate the total number of items to receive
  for (int i = 0; i < g.info.comm_size; ++i) {
      rdispls[i] = total_receive;
      total_receive += recv_counts[i];
  }

  Graph::EdgeList send_data;
  send_data.reserve(total_send);
  for (const auto& buf : send_bufs) {
      send_data.insert(send_data.end(), buf.begin(), buf.end());
  }

  Graph::EdgeList recv_data(total_receive);  // total_receive needs to be calculated

  MPI_Alltoallv(send_data.data(), send_counts.data(), sdispls.data(), g.MPI_EDGE,
                recv_data.data(), recv_counts.data(), rdispls.data(), g.MPI_EDGE, MPI_COMM_WORLD);

  // combine multi edges into a single edge
  std::unordered_map<int, std::unordered_map<int, double>> adj;
  for(auto edge: recv_data) adj[edge.v1][edge.v2] += edge.weight;

  Graph::EdgeList final;
  for(auto& [v, edge]: adj) {
    for(auto[neighbor, weight]: edge) 
      final.push_back({v, neighbor, weight});
  }

  final.insert(final.end(), self_edges.begin(), self_edges.end());

  // TODO: Checkpoint edgelist here
  // for(int rank = 0; rank < g.info.comm_size; rank++) {
  //   if (g.info.rank == rank) {
  //     std::cout << "RANK " << rank << " final edgelist: {";
  //     for(auto edge: final) {
  //       std::cout << "\t" << edge.v1 << " " << edge.v2 << " " << edge.weight << "\n";
  //     };
  //     std::cout << "}" << std::endl;
  //   }

    // MPI_Barrier(MPI_COMM_WORLD);
  // }

  Graph new_g; 
  new_g.distributedGraphInit(final, g.info);
  return new_g;
}

