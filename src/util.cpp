#include "util.h"
#include <alloca.h>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>

ProcInfo* ProcInfo::instance = nullptr;

ProcInfo::ProcInfo() {
  int initialized;
  MPI_Initialized(&initialized);

  // only compute MPI values if MPI process is active
  if(!initialized) {
    std::cerr << "MPI not initialized, aborting...\n";
    exit(EXIT_FAILURE);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  width = sqrt(comm_size);

  if(width * width != comm_size) {
    if(rank == 0)
      std::cout << "ERROR: value of " << comm_size << " is not square and therefore cannot be partitoned properly. Aborting..." << std::endl;

    exit(EXIT_FAILURE);
  }


  grid_row = std::floor(rank/width);
  grid_col = rank % width;

  MPI_Comm_split(MPI_COMM_WORLD, grid_row, rank, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, grid_col, rank, &col_comm);

  assert(row_comm != MPI_COMM_NULL);
  assert(col_comm != MPI_COMM_NULL);

  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_rank(col_comm, &col_rank);
}

ProcInfo::~ProcInfo() {
  int initialized;
  MPI_Initialized(&initialized);

  // only compute MPI values if MPI process is active
  if(!initialized) 
    return;

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
}


std::vector<std::pair<int, int>> edge_list_from_file(const std::string& fname) {
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
