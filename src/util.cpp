#include "util.h"
#include <alloca.h>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

ProcInfo::ProcInfo() {

  int initialized;
  MPI_Initialized(&initialized);

  // only compute MPI values if MPI process is active
  if(!initialized) 
    return;

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int width = sqrt(comm_size);

  if(width * width != comm_size) {
    if(rank == 0)
      std::cout << "ERROR: value of " << comm_size << " is not square and therefore cannot be partitoned properly. Aborting..." << std::endl;

    exit(EXIT_FAILURE);
  }


  this->rank = rank;
  this->i = rank % width;
  this->j = std::floor(rank/width);
  this->comm_size = comm_size;
  this->width = width;

  // ranks for MPI Row and Column groups
  int row_ranks[width];
  int col_ranks[width];

  for (int w =0; w < width; w++) {
    // if you're process 4 in a 3x3 grid (middle grid) then 
    // row ranks should be 3, 4, 5
    // col ranks should be 1, 4, 7
    row_ranks[w] = (this->j * width) + w;
    col_ranks[w] = this->i + (w * width);
  }

  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group_incl(world_group, width, row_ranks, &row_group);
  MPI_Group_incl(world_group, width, col_ranks, &col_group);

  MPI_Comm_create_group(MPI_COMM_WORLD, row_group, 0, &row_comm);
  MPI_Comm_create_group(MPI_COMM_WORLD, col_group, 0,  &col_comm);

  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_rank(col_comm, &col_rank);
}

ProcInfo::~ProcInfo() {
  int initialized;
  MPI_Initialized(&initialized);

  // only compute MPI values if MPI process is active
  if(!initialized) 
    return;

  // MPI_Group_free(&world_group);
  MPI_Group_free(&row_group);
  MPI_Group_free(&col_group);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
}


std::vector<std::pair<int, int>> edge_list_from_file(const std::string& fname) {
    std::cout << "Here!" << std::endl;
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fname << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int, int>> edges;  // Vector to store the edges
    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {  // Read two integers from the line
            edges.push_back({u, v});  // Add the edge to the vector
        } else {
            std::cerr << "Error reading line: " << line << std::endl;
        }
    }
    file.close();  // Close the file


    return edges;
}
