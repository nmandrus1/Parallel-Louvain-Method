#include "util.h"
#include <alloca.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

#include <gptl.h>

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


