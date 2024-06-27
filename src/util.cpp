#include "util.h"
#include <alloca.h>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

ProcInfo* ProcInfo::instance = nullptr;

ProcInfo::ProcInfo() {
  int initialized;
  MPI_Initialized(&initialized);

  // only compute MPI values if MPI process is active
  if(!initialized) {
    std::cerr << "MPI not initialized...\n";
    return;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
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
