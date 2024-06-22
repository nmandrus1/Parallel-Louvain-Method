#include "graph.h"
#include "util.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <mpi.h>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <utility>

// cuda kernels
int cudaInit(int rank);
void generateKroneckerEdgeList(int scale, int edgefactor, unsigned long seed,
                               int *start, int *end);

// default constructor
Graph::Graph(size_t vcount) : Graph() {
  this->vcount = vcount;
  this->data.resize(vcount * vcount, 0);
}

Graph::Graph(const std::vector<std::pair<int, int>> edge_list) : Graph() {
  std::map<int, std::set<unsigned>> adj_list;
  ecount = edge_list.size();

  for (auto pair : edge_list) {
    adj_list[pair.first].insert((unsigned)pair.second);
    adj_list[pair.second].insert((unsigned)pair.first);
  }

  vcount = adj_list.size();

  int nnz = 0;
  row_index.push_back(nnz);

  for (auto entry : adj_list) {
    data.insert(data.end(), entry.second.size(), 1);
    column_index.insert(column_index.end(), std::begin(entry.second),
                        std::end(entry.second));
    nnz += entry.second.size();
    row_index.push_back(nnz);
  }
}

// construct a graph from an edge list
Graph::Graph(const std::vector<std::pair<int, int>> &edge_list,
             const size_t vcount)
    : Graph() {

  this->ecount = edge_list.size();
  this->vcount = vcount;
  std::vector<bool> adj_mat(vcount * vcount, 0);

  // loop over every edge pair and add it to the graph
  for (auto pair : edge_list) {
    if (pair.first == pair.second)
      continue;

    // 1-d indexing
    adj_mat[pair.first * this->vcount + pair.second] = true;
    adj_mat[pair.second * this->vcount + pair.first] = true;
  }

  // sparsify
  int nnz = 0;
  this->row_index.push_back(nnz);

  for (unsigned i = 0; i < vcount; i++) {
    for (unsigned j = 0; j < vcount; j++) {
      int entry = adj_mat[i * vcount + j];
      if (entry != 0) {
        data.push_back(entry);
        column_index.push_back(j);
        nnz += 1;
      }
    }

    this->row_index.push_back(nnz);
  }

  this->rows.first = vcount * info->grid_row;
  this->rows.second = this->rows.first + vcount;

  this->columns.first = vcount * info->grid_col;
  this->columns.second = this->columns.first + vcount - 1;
}

Graph::Graph(const std::string &fname, bool distributed) : Graph() {
  auto edges = edge_list_from_file(fname);

  if (distributed == false) {
    *this = Graph(edges);
    return;
  }

  // if this is a distributed graph, we need to calculate which edges belong
  // where For now we assume that all edges are numbered 0 to n with no gaps map
  // ranks to the edges that need to go to that rank

  int max_vtx = 0;
  for (auto edge : edges)
    max_vtx = std::max(std::max(max_vtx, edge.first), edge.second);

  // send and recv the maximum vertex number, this + 1 is the total number of
  // vertices in the graph and we can use this information to help us partition
  // the graph
  MPI_Allreduce(&max_vtx, &max_vtx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // we have a vcount x vcount adj matrix locally
  // globally, info->width is the width of the process grid
  // so with 4 mpi procs info->width = 2 and the total number of
  // vertices is (max_vtx + 1) / 2 = 16/2 -> 8
  vcount = (max_vtx + 1) / info->width;

  int edge_count = edges.size();
  MPI_Allreduce(&edge_count, &edge_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ecount = edge_count;

  std::unordered_map<int, std::vector<int>> msg_map;
  for (auto edge : edges) {
    auto v1 = edge.first;
    auto v2 = edge.second;

    auto v1_cord = v1 / vcount;
    auto v2_cord = v2 / vcount;

    int rank1 = (info->width * v1_cord) + v2_cord;
    int rank2 = (info->width * v2_cord) + v1_cord;

    msg_map[rank1].push_back(v1);
    msg_map[rank1].push_back(v2);

    msg_map[rank2].push_back(v2);
    msg_map[rank2].push_back(v1);
  }

  // store the length, in number of integers, that this proc is sending to each
  // rank
  int total_ints_to_send = 0;
  std::vector<int> send_buf(info->comm_size, 0);
  for (auto entry : msg_map) {
    send_buf[entry.first] = entry.second.size();
    total_ints_to_send += entry.second.size();
  }

  // send 1 int to each proc
  std::vector<int> counts_send(info->comm_size, 1);

  // displacement for each proc is its own rank
  std::vector<int> displs_send(info->comm_size);
  for (int i = 0; i < info->comm_size; i++)
    displs_send[i] = i;

  std::vector<int> recv_buf(info->comm_size);
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
  for (int rank = 0; rank < info->comm_size; rank++) {
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
  std::vector<int> recv_counts(info->comm_size), recv_displ(info->comm_size);
  for (int rank = 0; rank < info->comm_size; rank++) {
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

  edges.resize(total_ints_to_recv / 2);
  edges.clear();

  // Deserialize recv_data back into a usable format, e.g., updating
  // candidate_parents or similar structures
  for (int i = 0; i < total_ints_to_recv; i += 2) {
    int vertex = recv_buf[i];
    int parent = recv_buf[i + 1];
    edges.push_back(std::make_pair(makeLocal(vertex), makeLocal(parent)));
  }

  // this->ecount = edges.size();
  std::vector<bool> adj_mat(vcount * vcount, 0);

  // loop over every edge pair and add it to the graph
  for (auto pair : edges) {
    if (pair.first == pair.second)
      continue;

    // 1-d indexing
    adj_mat[pair.first * this->vcount + pair.second] = true;
  }

  // sparsify
  int nnz = 0;
  row_index.push_back(nnz);

  for (unsigned i = 0; i < vcount; i++) {
    for (unsigned j = 0; j < vcount; j++) {
      int entry = adj_mat[i * vcount + j];
      if (entry != 0) {
        data.push_back(entry);
        column_index.push_back(j);
        nnz += 1;
      }
    }

    row_index.push_back(nnz);
  }

  rows.first = vcount * info->grid_row;
  rows.second = rows.first + vcount;
  
  columns.first = vcount * info->grid_col;
  columns.second = columns.first + vcount;
}

// generate a graph using kronecker algorithm
void Graph::from_kronecker(int scale, int edgefactor, unsigned long seed) {

  auto edge_list = generate_kronecker_list(scale, edgefactor, seed);
  int n_vertices = 1 << scale;

  std::vector<int> start = std::move(edge_list[0]);
  std::vector<int> end = std::move(edge_list[1]);

  // create a vector of length of the smaller vector
  std::vector<std::pair<int, int>> target(start.size());

  for (unsigned i = 0; i < target.size(); i++)
    target[i] = std::make_pair(start[i], end[i]);

  Graph result(target, n_vertices);

  *this = result;

  return;
}

// // add an edge between v1 and v2
// void Graph::add_edge(const int v1, const int v2) {
//   // 1-d indexing
//   this->data[v1 * this->vcount + v2] = true;
//   this->data[v2 * this->vcount + v1] = true;
// }

// get the list of vertices vert is connected to
std::vector<int> Graph::neighbors(const int vert) const {
  std::vector<int> ret;

  unsigned row_start = this->row_index[vert];
  unsigned row_end = this->row_index[vert + 1];

  // loop over every vertex and push back those vert is adjacent to
  for (unsigned i = row_start; i < row_end; i++) {
    ret.push_back(this->column_index[i]);
  }

  return ret;
}

// Assuming that vert is the global vertex index
std::vector<int> Graph::neighborsGlobalIdxs(const int vert) const {
  int local = vert % vcount;
  std::vector<int> ret;

  unsigned row_start = this->row_index[local];
  unsigned row_end = this->row_index[local + 1];

  // loop over every vertex and push back those vert is adjacent to
  for (unsigned i = row_start; i < row_end; i++) {
    ret.push_back(localColToGlobal(column_index[i]));
  }

  return ret;
}
// Perform a BFS from the specified source vertex and return the Parent Array
std::vector<int> Graph::top_down_bfs(const int src) {

  // queue for storing future nodes to explore
  std::queue<int> q;
  q.push(src);
  // Hashset to store visited nodes
  std::unordered_set<int> visited;
  // vector of parents
  std::vector<int> parents(this->vcount, -1);
  parents[src] = src;

  while (!q.empty()) {
    int parent = q.front();
    q.pop();

    // get list of adjacent vertices
    auto connections = neighbors(parent);
    for (int v : connections) {
      // we only proceed when v has not been visited which is when it is NOT in
      // our set
      if (parents[v] == -1) {
        parents[v] = parent;
        q.push(v);
      }
    }
  }

  return parents;
}

// Perform a Bottom Down BFS from the specified source vertex and return the
// Parent Array
std::vector<int> Graph::btm_down_bfs(const int src) const {

  // Hashsets to store frontier and next frontier
  std::unordered_set<int> frontier;
  frontier.insert(src);

  std::unordered_set<int> next;

  // vector of parents
  std::vector<int> parents(this->vcount, -1);
  parents[src] = src;

  while (!frontier.empty()) {
    // loop over all vertices in V
    for (int u = 0; u < this->vcount; u++) {
      // the vertex has no parent yet
      if (parents[u] == -1) {
        auto edges = neighbors(u);
        for (int v : edges) {
          if (frontier.find(v) != frontier.end()) {
            next.insert(u);
            parents[u] = v;
            break;
          }
        }
      }
    }

    frontier = std::move(next);
    next.clear();
  }

  return parents;
}

// given a local list of candidate parents, communicate with all processors in
// this row to exhange and merge all candiate parents lists together
void Graph::broadcast_to_row(std::unordered_map<int, int> &candidate_parents) {

  std::vector<std::vector<int>> send_buffers(this->info->width);

  // Organize data into send buffers for each process in the row
  for (const auto &pair : candidate_parents) {
    int vertex = pair.first;
    int parent = pair.second;
    int target_process = vertex / vcount;

    send_buffers[target_process].push_back(vertex);
    send_buffers[target_process].push_back(parent);
  }

  // Prepare for MPI_Alltoallv
  std::vector<int> send_counts(this->info->width, 0);
  std::vector<int> sdispls(this->info->width, 0);
  std::vector<int> total_send_buffer;

  // Flatten send buffers and calculate send counts and displacements
  for (int i = 0; i < this->info->width; ++i) {
    sdispls[i] = total_send_buffer.size();
    send_counts[i] = send_buffers[i].size();
    total_send_buffer.insert(total_send_buffer.end(), send_buffers[i].begin(),
                             send_buffers[i].end());
  }

  // Prepare receive counts and displacements (to be gathered via MPI_Alltoall)
  std::vector<int> recv_counts(this->info->width);

  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               info->row_comm);

  std::vector<int> rdispls(this->info->width);
  int total_recv_size = 0;
  for (int i = 0; i < this->info->width; ++i) {
    rdispls[i] = total_recv_size;
    total_recv_size += recv_counts[i];
  }

  std::vector<int> recv_buffer(total_recv_size);

  // Execute the MPI_Alltoallv
  MPI_Alltoallv(total_send_buffer.data(), send_counts.data(), sdispls.data(),
                MPI_INT, recv_buffer.data(), recv_counts.data(), rdispls.data(),
                MPI_INT, info->row_comm);

  // Deserialize recv_data back into a usable format, e.g., updating
  // candidate_parents or similar structures
  for (int i = 0; i < total_recv_size; i += 2) {
    int vertex = recv_buffer[i];
    int parent = recv_buffer[i + 1];
    candidate_parents[vertex] = parent;
  }
}

// communicate with all processes and return the column frontier, set term_cond
// to true if we are done globally
bool Graph::gather_global_frontier(const std::vector<int> local_frontier,
                                   std::vector<int> &global_frontier) {

  std::vector<int> all_local_sizes(this->info->width);
  int global_frontier_size, local_frontier_size = local_frontier.size();

  MPI_Allreduce(&local_frontier_size, &global_frontier_size, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);

  if (global_frontier_size == 0) {
    return true;
  }

  // Gather the sizes of the local frontiers from all processors in the same
  // column
  MPI_Allgather(&local_frontier_size, 1, MPI_INT, all_local_sizes.data(), 1,
                MPI_INT, this->info->col_comm);

  std::vector<int> displacements(this->info->width);
  int sum = 0;
  for (int i = 0; i < this->info->width; ++i) {
    displacements[i] = sum;
    sum += all_local_sizes[i];
  }

  // nothing for these procs to do
  if (sum != 0) {
    // Resize the global frontier to accommodate all elements
    global_frontier.resize(sum);

    // Perform the allgatherv operation
    MPI_Allgatherv(local_frontier.data(), local_frontier_size, MPI_INT,
                   global_frontier.data(), all_local_sizes.data(),
                   displacements.data(), MPI_INT, this->info->col_comm);
  }
  return false;
}

// given a parents list and local frontier, start BFS
std::vector<int>
Graph::parallel_top_down_bfs_driver(std::vector<int> &parents,
                                    std::vector<int> &local_frontier,
                                    int checkpoint_int) {

  std::stringstream s;
  s << "checkpoint" << info->grid_row << ".bin";
  std::string fname = s.str();

  std::unordered_map<int, int> candidate_parents;
  std::vector<int> global_frontier;

  // outer loop
  // only terminates when all processes have an empty frontier

  int iteration = 1;
  while (true) {

    // gather global frontier returns true when we are done
    if (this->gather_global_frontier(local_frontier, global_frontier))
      break;

    // if our frontier is not empty then inspect adjacent vertices
    for (int U : global_frontier) {
      // only check adj. if U might have an edge in our space
      if (this->in_column(U) && parents[U - columns.first] != -1) {
        // shift the value of U to match local indexing
        auto connections = neighbors(U - columns.first);
        // look through all connections and find any that need exploring
        for (auto v : connections) {
          // V is global index
          int V = v + rows.first;
          // if(candidate_parents[V] != -1) continue; // skip any that have
          // already been visited
          candidate_parents[V] = U;
        }
      }
    }

    // Alltoallv
    this->broadcast_to_row(candidate_parents);
    local_frontier.clear();

    for (auto item : candidate_parents) {
      auto vertex = item.first;
      auto parent = item.second;
      // Update or process the candidate_parents information with received data
      // Be sure to handle global IDs and avoid overwriting any previously set
      // parents unless the algorithm dictates otherwise.
      if (this->in_column(vertex)) {

        // change vertex to local index
        int v = vertex - columns.first;
        if (parents[v] == -1) {
          parents[v] = parent;
          local_frontier.push_back(vertex);
        }
      }
    }

    candidate_parents.clear();
    global_frontier.clear();

    // checkpointing
    if (checkpoint_int != 0 && iteration % checkpoint_int == 0) {
      checkpoint_data(parents, iteration, this->info->col_comm, fname.data());
    }

    iteration++;
  }

  if (info->rank == 0)
    std::cout << iteration << "iterations!\n";

  return parents;
}

// write parents list to file
void Graph::checkpoint_data(const std::vector<int> &data, const int iteration,
                            MPI_Comm comm, const char *filename) const {
  MPI_File fh;
  MPI_Status status;

  // Open the file for writing
  MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  // Calculate the offset for each process
  MPI_Offset offset = info->rank * sizeof(int) * data.size();

  // Write the data
  MPI_File_write_at(fh, offset, data.data(), data.size(), MPI_INT, &status);

  // Close the file
  MPI_File_close(&fh);

  if (info->rank == 0 && output) {
    std::cout << "Checkpoint at iteration " << iteration
              << " written successfully.\n";
  }
}

// Perform a Parallel Top Down BFS from the specified source vertex and return
// the Parent Array
std::vector<int> Graph::parallel_top_down_bfs(const int src,
                                              int checkpoint_int) {
  std::vector<int> parents(this->vcount, -1);
  // candidate_parents[src] = src;
  std::vector<int> local_frontier;

  // only add src to frontier if it is owned by the current graph object
  if (this->in_column(src)) {
    parents[src] = src;
    local_frontier.push_back(src);
  }

  return this->parallel_top_down_bfs_driver(parents, local_frontier,
                                            checkpoint_int);
}

void Graph::print_graph() const {
  for (unsigned i = 0; i < this->vcount; i++) {
    auto edges = neighbors(i);
    auto edges_iter = edges.begin();

    for (unsigned j = 0; j < this->vcount; j++) {
      if (edges_iter != edges.end() && *edges_iter == j) {
        std::cout << 1;
        edges_iter++;
      } else
        std::cout << "0";

      std::cout << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

// test CUDA function
void Graph::from_kronecker_cuda(int scale, int edgefactor, unsigned long seed) {
  int num_vertices = 1 << scale;
  // auto edge_list = generate_kronecker_list_cuda(scale, edgefactor, seed);
  auto edge_list = generate_kronecker_list_cuda(scale, edgefactor, seed);
  std::vector<int> start = std::move(edge_list[0]);
  std::vector<int> end = std::move(edge_list[1]);

  // create a vector of length of the smaller vector
  std::vector<std::pair<int, int>> target(start.size());

  for (unsigned i = 0; i < target.size(); i++)
    target[i] = std::make_pair(start[i], end[i]);

  Graph result(target, num_vertices);

  *this = result;

  return;
}

std::vector<std::vector<int>> generate_kronecker_list(int scale, int edgefactor,
                                                      unsigned long seed) {

  int num_vertices = 1 << scale;
  int num_edges = num_vertices * edgefactor;

  std::vector<std::vector<int>> edge_list(2, std::vector<int>(num_edges, 0));

  double A = 0.57, B = 0.19, C = 0.19; // Initiator probabilities
  double ab = A + B;
  double c_norm = C / (1 - (A + B));
  double a_norm = A / (A + B);

  // std::random_device rd;  // Seed for random number engine
  std::mt19937 gen(seed); // Standard mersenne_twister_engine
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Generate edges
  for (int ib = 0; ib < scale; ++ib) {
    for (int i = 0; i < num_edges; ++i) {
      bool ii_bit = dis(gen) > ab;
      bool jj_bit = dis(gen) > (c_norm * ii_bit + a_norm * !ii_bit);
      edge_list[0][i] += (ii_bit ? 1 : 0) * (1 << ib);
      edge_list[1][i] += (jj_bit ? 1 : 0) * (1 << ib);
    }
  }

  // Permute vertex labels and edge list
  std::vector<int> p(num_vertices);
  std::iota(p.begin(), p.end(), 0);
  std::shuffle(p.begin(), p.end(), gen);
  for (int i = 0; i < num_edges; ++i) {
    edge_list[0][i] = p[edge_list[0][i]];
    edge_list[1][i] = p[edge_list[1][i]];
  }

  std::shuffle(edge_list[0].begin(), edge_list[0].end(), gen);
  std::shuffle(edge_list[1].begin(), edge_list[1].end(), gen);

  return edge_list;
}

std::vector<std::vector<int>>
generate_kronecker_list_cuda(int scale, int edgefactor,
                             unsigned long long seed) {
  // init cuda device by rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaInit(rank);

  int num_vertices = 1 << scale;
  int num_edges = num_vertices * edgefactor;

  // return vector
  std::vector<std::vector<int>> edge_list(2, std::vector<int>(num_edges));
  generateKroneckerEdgeList(scale, edgefactor, seed, edge_list[0].data(),
                            edge_list[1].data());

  // std::random_device rd;  // Seed for random number engine
  std::mt19937 gen(seed); // Standard mersenne_twister_engine
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // shuffle lists

  // Permute vertex labels and edge list
  std::vector<int> p(num_vertices);
  std::iota(p.begin(), p.end(), 0);
  std::shuffle(p.begin(), p.end(), gen);
  for (int i = 0; i < num_edges; ++i) {
    edge_list[0][i] = p[edge_list[0][i]];
    edge_list[1][i] = p[edge_list[1][i]];
  }

  std::shuffle(edge_list[0].begin(), edge_list[0].end(), gen);
  std::shuffle(edge_list[1].begin(), edge_list[1].end(), gen);

  return edge_list;
}

// Return the sum of the edge weights (the number of edges for unweighted graph)
int Graph::degree(int v) const {
  return this->row_index[v + 1] - this->row_index[v];
}

int Graph::get_edge(int v1, int v2) const {
  unsigned row_start = this->row_index[v1];
  unsigned row_end = this->row_index[v1 + 1];

  for (unsigned i = row_start; i < row_end; i++) {
    if (this->column_index[i] == v2)
      return column_index[i];
    else if (this->column_index[i] > v2)
      break;
  }

  return 0;
}
