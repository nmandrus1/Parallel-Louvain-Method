#include "graph.h"
#include "util.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unordered_map>
#include <map>
#include <set>

#include <utility>

void Graph::sparsify(const std::map<int, std::set<unsigned>>& adj_list) {
  ecount = 0;
  row_index.push_back(0);
  for (const auto& entry : adj_list) {
      for (int neighbor : entry.second) {
          column_index.push_back(neighbor);
          data.push_back(1); // assuming all edges have weight 1 for simplicity
      }
      row_index.push_back(column_index.size());
      ecount += entry.second.size();
  }
}

void Graph::initializeFromAdjList(const std::map<int, std::set<unsigned>>& adj_list) {
  local_vcount = adj_list.size();
  global_vcount = adj_list.size();
  rows.first = 0;
  rows.second = local_vcount;
  sparsify(adj_list);
}

// default constructor
Graph::Graph(size_t vcount) : local_vcount(vcount), global_vcount(vcount) {
  this->data.resize(vcount * vcount, 0);
}

Graph::Graph(const std::vector<std::pair<int, int>> edge_list) {
  std::map<int, std::set<unsigned>> adj_list;
    for (auto& pair : edge_list) {
        adj_list[pair.first].insert(pair.second);
        adj_list[pair.second].insert(pair.first);
    }
    initializeFromAdjList(adj_list);
}


Graph::Graph(const std::string &fname, bool distributed) {
  auto edges = edge_list_from_file(fname);

  if (distributed == false) {
    *this = Graph(edges);
  } else {
    auto info = ProcInfo();
    distributedGraphInit(edges, info);
  } 
}


void Graph::distributedGraphInit(const std::vector<std::pair<int, int>>& edges, ProcInfo info) {
  // if this is a distributed graph, we need to calculate which edges belong
  // where For now we assume that all edges are numbered 0 to n with no gaps map
  // ranks to the edges that need to go to that rank

  info = info;

  int max_vtx = 0;
  for (auto edge : edges)
    max_vtx = std::max(std::max(max_vtx, edge.first), edge.second);

  // send and recv the maximum vertex number, this + 1 is the total number of
  // vertices in the graph and we can use this information to help us partition
  // the graph
  MPI_Allreduce(&max_vtx, &max_vtx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // we have a vcount x vcount adj matrix locally
  // globally, info.width is the width of the process grid
  // so with 4 mpi procs info.width = 2 and the total number of
  // vertices is (max_vtx + 1) / 2 = 16/2 -> 8
  global_vcount = max_vtx + 1;
  local_vcount = global_vcount / info.comm_size;

  rows.first = info.rank * local_vcount;
  rows.second = rows.first + local_vcount;

  int edge_count = edges.size();
  MPI_Allreduce(&edge_count, &edge_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ecount = edge_count;

  
  std::unordered_map<int, std::vector<int>> msg_map;
  for (auto edge : edges) {
    auto v1 = edge.first;
    auto v2 = edge.second;

    auto rank1 = v1 / local_vcount;
    auto rank2 = v2 / local_vcount;
    msg_map[rank1].push_back(v1);
    msg_map[rank1].push_back(v2);

    msg_map[rank2].push_back(v2);
    msg_map[rank2].push_back(v1);
  }

  // store the length, in number of integers, that this proc is sending to each
  // rank
  int total_ints_to_send = 0;
  std::vector<int> send_buf(info.comm_size, 0);
  for (auto entry : msg_map) {
    send_buf[entry.first] = entry.second.size();
    total_ints_to_send += entry.second.size();
  }

  // send 1 int to each proc
  std::vector<int> counts_send(info.comm_size, 1);

  // displacement for each proc is its own rank
  std::vector<int> displs_send(info.comm_size);
  for (int i = 0; i < info.comm_size; i++)
    displs_send[i] = i;

  std::vector<int> recv_buf(info.comm_size);
  // counts_recv is the same as counts_send
  // displ_recv is the same as displ_send
  MPI_Request req;
  MPI_Ialltoallv(send_buf.data(), counts_send.data(), displs_send.data(),
                 MPI_INT, recv_buf.data(), counts_send.data(),
                 displs_send.data(), MPI_INT, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  send_buf.resize(total_ints_to_send);
  send_buf.clear();
  std::fill(counts_send.begin(), counts_send.end(), 0);

  int send_displ = 0;
  for (int rank = 0; rank < info.comm_size; rank++) {
    auto entry = msg_map.find(rank);
    if (entry == msg_map.end())
      continue;

    auto msg = entry->second;
    send_buf.insert(send_buf.end(), msg.begin(), msg.end());
    counts_send[rank] = msg.size();

    // calculate offset from start
    displs_send[rank] = send_displ;
    send_displ += msg.size();
  }

  int total_ints_to_recv = 0;
  std::vector<int> recv_counts(info.comm_size), recv_displ(info.comm_size);
  for (int rank = 0; rank < info.comm_size; rank++) {
    recv_counts[rank] = recv_buf[rank];

    recv_displ[rank] = total_ints_to_recv;
    total_ints_to_recv += recv_buf[rank];
  }

  // resize recv buffer
  recv_buf.resize(total_ints_to_recv);

  MPI_Request req2;
  MPI_Ialltoallv(send_buf.data(), counts_send.data(), displs_send.data(),
                 MPI_INT, recv_buf.data(), recv_counts.data(),
                 recv_displ.data(), MPI_INT, MPI_COMM_WORLD, &req2);
  MPI_Wait(&req2, MPI_STATUS_IGNORE);

  std::map<int, std::set<unsigned>> adj_list;

  // Deserialize recv_data back into a usable format, e.g., updating
  // candidate_parents or similar structures
  for (int i = 0; i < total_ints_to_recv; i += 2) {
    int vertex = recv_buf[i];
    int neighbor = recv_buf[i + 1];
    adj_list[vertex].insert(neighbor);
  }

  initializeFromAdjList(adj_list);
}

// get the list of vertices vert is connected to
std::vector<int> Graph::neighbors(const int v) const {
  assert(in_row(v));
  int vert = makeLocal(v);


  std::vector<int> ret;

  unsigned row_start = this->row_index[vert];
  unsigned row_end = this->row_index[vert + 1];

  // loop over every vertex and push back those vert is adjacent to
  for (unsigned i = row_start; i < row_end; i++) {
    ret.push_back(this->column_index[i]);
  }

  return ret;
}

void Graph::print_graph() const {
  for (unsigned i = 0; i < this->local_vcount; i++) {
    auto edges = neighbors(i);
    auto edges_iter = edges.begin();

    for (unsigned j = 0; j < this->global_vcount; j++) {
      if (edges_iter != edges.end() && *edges_iter == j) {
        std::cout << 1;
        edges_iter++;
      } else std::cout << "0";

      std::cout << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}
