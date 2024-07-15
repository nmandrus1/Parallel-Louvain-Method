#include "graph.h"
#include "util.h"

#include <iomanip>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include <utility>
#include <gptl.h>


Graph::EdgeList Graph::edge_list_from_file(const std::string& fname) {
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fname << std::endl;
        exit(EXIT_FAILURE);
    }

    EdgeList edges;  // Vector to store the edges
    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        double w;

        if (iss >> u >> v >> w) {  // Read two integers from the line
            edges.push_back({u, v, w});  // Add the edge to the vector
        } else {
            std::cerr << "Error reading line: " << line << std::endl;
        }
    }
    file.close();  // Close the file


    return edges;
}

int Graph::sparsify(const AdjacencyList& adj_list) {
  int edges = 0;
  row_index.push_back(0);
  for (const auto& neighbors : adj_list) {
      for (auto& [neighbor, weight]: neighbors.second) {
          column_index.push_back(neighbor);
          weights.push_back(weight);
      }
      row_index.push_back(column_index.size());
      edges += neighbors.second.size();
  }
  return edges / 2;
}

void Graph::initializeFromAdjList(const AdjacencyList& adj_list) {
  local_vcount = adj_list.size();
  global_vcount = adj_list.size();
  rows.first = 0;
  rows.second = local_vcount;
  ecount = sparsify(adj_list);
}

// default constructor
Graph::Graph(size_t vcount) : local_vcount(vcount), global_vcount(vcount){}

Graph::Graph(const EdgeList& edge_list) {
  AdjacencyList adj_list;
    for (auto& edge : edge_list) {
        adj_list[edge.v1].insert(std::make_pair(edge.v2, edge.weight));
        adj_list[edge.v2].insert(std::make_pair(edge.v1, edge.weight));
    }
    initializeFromAdjList(adj_list);
}

Graph::Graph(const AdjacencyList& adj_list) {
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

Graph::~Graph() {
  MPI_Type_free(&MPI_EDGE);
}

void Graph::distributedGraphInit(const EdgeList& edges, ProcInfo info) {
  // if this is a distributed graph, we need to calculate which edges belong
  // where For now we assume that all edges are numbered 0 to n with no gaps map
  // ranks to the edges that need to go to that rank

  int max_vtx = 0;
  for (auto edge : edges)
    max_vtx = std::max(std::max(max_vtx, edge.v1), edge.v2);

  // send and recv the maximum vertex number, this + 1 is the total number of
  // vertices in the graph and we can use this information to help us partition
  // the graph
  MPI_Allreduce(&max_vtx, &max_vtx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  distributedGraphInitWithGlobalVcount(edges, info, max_vtx + 1);
}

void Graph::distributedGraphInitWithGlobalVcount(const EdgeList &edges, ProcInfo info, int vcount) {
  info = info;
  
  // we have a vcount x vcount adj matrix locally
  // globally, info.width is the width of the process grid
  // so with 4 mpi procs info.width = 2 and the total number of
  // vertices is (max_vtx + 1) / 2 = 16/2 -> 8
  global_vcount = vcount;
  local_vcount = global_vcount / info.comm_size;

  rows.first = info.rank * local_vcount;
  rows.second = rows.first + local_vcount;

  int edge_count = edges.size();
  MPI_Allreduce(&edge_count, &edge_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ecount = edge_count;

  
  std::unordered_map<int, EdgeList> msg_map;

  for (auto edge : edges) {
    auto rank1 = edge.v1 / local_vcount;
    auto rank2 = edge.v2 / local_vcount;

    msg_map[rank1].push_back(edge);
    msg_map[rank2].push_back(Edge { edge.v2, edge.v1, edge.weight});
  }

  // store the length, in number of edges, that this proc is sending to each rank
  int total_edges_to_send = 0;
  std::vector<int> count_send_buf(info.comm_size, 0);
  for (auto entry : msg_map) {
    count_send_buf[entry.first] = entry.second.size();
    total_edges_to_send += entry.second.size();
  }

  // send 1 int to each proc
  std::vector<int> counts_send(info.comm_size, 1);

  // displacement for each proc is its own rank
  std::vector<int> displs_send(info.comm_size);
  for (int i = 0; i < info.comm_size; i++)
    displs_send[i] = i;

  std::vector<int> count_recv_buf(info.comm_size);
  // counts_recv is the same as counts_send
  // displ_recv is the same as displ_send
  MPI_Request req;
  MPI_Ialltoallv(count_send_buf.data(), counts_send.data(), displs_send.data(),
                 MPI_INT, count_recv_buf.data(), counts_send.data(),
                 displs_send.data(), MPI_INT, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  std::fill(counts_send.begin(), counts_send.end(), 0);


  EdgeList send_buf;

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

  int total_edges_to_recv = 0;
  std::vector<int> recv_counts(info.comm_size), recv_displ(info.comm_size);
  for (int rank = 0; rank < info.comm_size; rank++) {
    recv_counts[rank] = count_recv_buf[rank];

    recv_displ[rank] = total_edges_to_recv;
    total_edges_to_recv += count_recv_buf[rank];
  }

  // resize recv buffer
  EdgeList recv_buf(total_edges_to_recv);

  create_edge_datatype(&MPI_EDGE);

  MPI_Request req2;
  MPI_Ialltoallv(send_buf.data(), counts_send.data(), displs_send.data(),
                 MPI_EDGE, recv_buf.data(), recv_counts.data(),
                 recv_displ.data(), MPI_EDGE, MPI_COMM_WORLD, &req2);
  MPI_Wait(&req2, MPI_STATUS_IGNORE);

  AdjacencyList adj_list;

  // Deserialize recv_data back into a usable format, e.g., updating
  // candidate_parents or similar structures
  for (auto& edge: recv_buf) {
    int vertex = makeLocal(edge.v1);
    adj_list[vertex].insert({edge.v2, edge.weight});
  }

  sparsify(adj_list);
}

// get the list of vertices vert is connected to
Graph::NeighborIterator Graph::neighbors(const int vert) const {
   int local_vert = makeLocal(vert);
    size_t start_idx = row_index[local_vert];
    size_t end_idx = row_index[local_vert + 1];
    return NeighborIterator(column_index, weights, start_idx, end_idx);
}

// TODO: Cache weighted degree
double Graph::weighted_degree(int vtx) const {
  assert(in_row(vtx));
  int vert = makeLocal(vtx);

  double sum = 0.0;
  for(int i = row_index[vert]; i < row_index[vert+1]; i++)
    sum += weights[i];

  return sum;
}

void Graph::print() const {
    // Determine the width to use for each number in the matrix, for consistent formatting
    int width = 6; // Adjust this width as needed for better spacing

    std::vector<double> weights_row(global_vcount, 0.0);
    for (auto vtx = rows.first; vtx < rows.second; vtx++) {
        // Fill in the weights from neighbors
        for(auto [neighbor, weight] : neighbors(vtx)) {
            weights_row[neighbor] = weight; // Assign weight to the correct column
        }

        // Print out the row with each weight formatted nicely
        std::cout << vtx << ": ";
        for (int j = 0; j < global_vcount; j++) {
            std::cout << std::setw(width) << std::setprecision(2) << std::fixed << weights_row[j];
            if (j != global_vcount - 1) {
                std::cout << " "; // Space between columns, except after the last one
            }
        }
        std::cout << "\n"; // New line after each row

        std::fill(weights_row.begin(), weights_row.end(), 0.0);
    }
    std::cout << std::endl; // Additional newline for separation after the whole matrix
}

void Graph::distributed_print() const {
  for(int rank = 0; rank < info.comm_size; rank++) {
    if(info.rank == rank) {
      print();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}


void Graph::create_edge_datatype(MPI_Datatype* dt) {
    int lengths[3] = { 1, 1, 1 };
 
    // Calculate displacements
    // In C, by default padding can be inserted between fields. MPI_Get_address will allow
    // to get the address of each struct field and calculate the corresponding displacement
    // relative to that struct base address. The displacements thus calculated will therefore
    // include padding if any.
    MPI_Aint displacements[3];
    Edge dummy_edge;
    MPI_Aint base_address;
    MPI_Get_address(&dummy_edge, &base_address);
    MPI_Get_address(&dummy_edge.v1, &displacements[0]);
    MPI_Get_address(&dummy_edge.v2, &displacements[1]);
    MPI_Get_address(&dummy_edge.weight, &displacements[2]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
 
    MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Type_create_struct(3, lengths, displacements, types, dt);
    MPI_Type_commit(dt);
  
}

