#include "graph.h"
#include <cstddef>
#include <iostream>
#include <mpi.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>

void runHelloCuda();

// default constructor
Graph::Graph(size_t vcount) : vcount(vcount), info() {
  this->data.resize(vcount * vcount, 0);
}

// construct a graph from an edge list

Graph::Graph(const std::vector<std::pair<int, int>> &edge_list,
             const size_t vcount)
    : Graph(vcount) {
  // loop over every edge pair and add it to the graph
  for (auto pair : edge_list) {
    if (pair.first == pair.second)
      continue;
    this->add_edge(pair.first, pair.second);
  }

  this->rows.first = vcount * this->info.j;
  this->rows.second = this->rows.first + vcount - 1;

  this->columns.first = vcount * this->info.i;
  this->columns.second = this->columns.first + vcount - 1;
}

// generate a graph using kronecker algorithm
Graph Graph::from_kronecker(int scale, int edgefactor, unsigned long seed) {
  auto edge_list = generate_kronecker_list(scale, edgefactor, seed);
  int n_vertices = 1 << scale;

  std::vector<int> start = std::move(edge_list[0]);
  std::vector<int> end = std::move(edge_list[1]);

  // create a vector of length of the smaller vector
  std::vector<std::pair<int, int>> target(start.size());

  for (unsigned i = 0; i < target.size(); i++)
    target[i] = std::make_pair(start[i], end[i]);

  Graph result(target, n_vertices);
  return result;
}

// add an edge between v1 and v2
void Graph::add_edge(const int v1, const int v2) {
  // 1-d indexing
  this->data[v1 * this->vcount + v2] = true;
  this->data[v2 * this->vcount + v1] = true;
}

// get the list of vertices vert is connected to
std::vector<int> Graph::get_edges(const int vert) const {
  std::vector<int> ret;

  // loop over every vertex and push back those vert is adjacent to
  for (unsigned i = 0; i < vcount; i++) {
    // if there exists an edge here
    if (this->data[vert + (vcount * i)])
      ret.push_back(i);
  }

  return ret;
}

std::vector<int> Graph::get_edges_distributed(const int vert) const {
  std::vector<int> ret;

  // loop over every vertex and push back those vert is adjacent to
  for (unsigned i = 0; i < vcount; i++) {
    // if there exists an edge here
    if (this->data[vert + (vcount * i)])
      ret.push_back(i);
  }

  return ret;
}

// Perform a BFS from the specified source vertex and return the Parent Array
std::vector<int> Graph::top_down_bfs(const int src) const {
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
    auto connections = this->get_edges(parent);
    for (int v : connections) {
      // we only proceed when v has not been visited which is when it is NOT in
      // our set
      if (parents[v] == -1) {
        parents[v] = parent;
        q.push(v);
        // visited.insert(v);
      }
    }

    // insert parent into visited list
    // visited.insert(parent);
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
        auto edges = this->get_edges(u);
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

// Perform a Parallel Top Down BFS from the specified source vertex and return
// the Parent Array
std::vector<int> Graph::parallel_top_down_bfs(const int src) const {
  // initialize algorithm
  std::vector<int> parents(this->vcount, -1);
  std::unordered_map<int, int> candidate_parents;
  // candidate_parents[src] = src;

  std::vector<int> local_frontier, global_frontier;

  // only add src to frontier if it is owned by the current graph object
  if (this->in_column(src)) {
    parents[src] = src;
    local_frontier.push_back(src);
  }

  // outer loop
  // only terminates when all processes have an empty frontier
  int iteration = 1;
  while (true) {
    int local_frontier_size = local_frontier.size();
    std::vector<int> all_local_sizes(this->info.width);

    int global_frontier_size;
    MPI_Allreduce(&local_frontier_size, &global_frontier_size, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    // termination condition
    if (global_frontier_size == 0)
      break;

    // Gather the sizes of the local frontiers from all processors in the same
    // column
    MPI_Allgather(&local_frontier_size, 1, MPI_INT, all_local_sizes.data(), 1,
                  MPI_INT, this->info.col_comm);

    std::vector<int> displacements(this->info.width);
    int sum = 0;
    for (int i = 0; i < this->info.width; ++i) {
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
                     displacements.data(), MPI_INT, this->info.col_comm);

      // if our frontier is not empty then inspect adjacent vertices
      for (int U : global_frontier) {
        // only check adj. if U might have an edge in our space
        if (this->in_column(U) && parents[U - columns.first] != -1) {
          // shift the value of U to match local indexing
          auto connections = this->get_edges(U - columns.first);
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
    }

    std::vector<std::vector<int>> send_buffers(this->info.width);
    
    // Organize data into send buffers for each process in the row
    for (const auto& pair : candidate_parents) {
        int vertex = pair.first;
        int parent = pair.second;
        int target_process = vertex / vcount;

        send_buffers[target_process].push_back(vertex);
        send_buffers[target_process].push_back(parent);
    }

    // Prepare for MPI_Alltoallv
    std::vector<int> send_counts(this->info.width, 0);
    std::vector<int> sdispls(this->info.width, 0);
    std::vector<int> total_send_buffer;

    // Flatten send buffers and calculate send counts and displacements
    for (int i = 0; i < this->info.width; ++i) {
        sdispls[i] = total_send_buffer.size();
        send_counts[i] = send_buffers[i].size();
        total_send_buffer.insert(total_send_buffer.end(), send_buffers[i].begin(), send_buffers[i].end());
    }

    // Prepare receive counts and displacements (to be gathered via MPI_Alltoall)
    std::vector<int> recv_counts(this->info.width);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, info.row_comm);

    std::vector<int> rdispls(this->info.width);
    int total_recv_size = 0;
    for (int i = 0; i < this->info.width; ++i) {
        rdispls[i] = total_recv_size;
        total_recv_size += recv_counts[i];
    }
    
    std::vector<int> recv_buffer(total_recv_size);    
   
    // Execute the MPI_Alltoallv
    MPI_Alltoallv(total_send_buffer.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  recv_buffer.data(), recv_counts.data(), rdispls.data(), MPI_INT, info.row_comm);
    // clear local frontier to prepare for next iteration
    local_frontier.clear();
    candidate_parents.clear();

    // Deserialize recv_data back into a usable format, e.g., updating
    // candidate_parents or similar structures
    for (int i = 0; i < total_recv_size; i += 2) {
      int vertex = recv_buffer[i];
      int parent = recv_buffer[i + 1];
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

    // wait here for other processes to finish
    MPI_Barrier(MPI_COMM_WORLD);

    if(info.rank == 0) {
      std::cout << "Iteration: " << iteration << "\n";
      iteration++;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i < info.comm_size; i++) {
      if(info.rank == i) {
        std::cout << "Rank: " << this->info.rank << "\t {";
        for (auto v : global_frontier)
          std::cout << v << ", ";
        std::cout << "}" << std::endl;
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }

    global_frontier.clear();
  }

  
  for(int i = 0; i < info.comm_size; i++) {
    if(info.rank == i) {
      std::cout << "Rank: " << i << " Parents array: { ";
        for(auto v: parents) {
          std::cout << v << ", ";
        }
        std::cout << "}\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  return parents;
}

void Graph::print_graph() const {
  for (unsigned i = 0; i < this->vcount; i++) {
    for (unsigned j = 0; j < this->vcount; j++) {
      std::cout << this->data[this->vcount * i + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

// test CUDA function
void Graph::hello_cuda() { runHelloCuda(); }

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

  // Generate weights
  // for (int i = 0; i < M; ++i) {
  // ijw[2][i] = dis(gen);
  // }

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
  // std::shuffle(ijw[2].begin(), ijw[2].end(), gen);

  std::cout << "Start Vertex: ";
  for (int i = 0; i < (num_edges < 10 ? num_edges : 10);
       ++i) { // Output first 10 edges for preview
    std::cout << edge_list[0][i] << ", ";
  }
  std::cout << "\n";

  std::cout << "End Vertex:   ";
  for (int i = 0; i < (num_edges < 10 ? num_edges : 10);
       ++i) { // Output first 10 edges for preview
    std::cout << edge_list[1][i] << ", ";
  }
  std::cout << "\n" << std::endl;

  return edge_list;
}
