#ifndef __UTIL_H_
#define __UTIL_H_

#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>


struct ProcInfo {
  // store rank, row, and column of adj matrix
  int rank, row_rank,col_rank, comm_size, i, j, width; 
  MPI_Comm row_comm, col_comm;
  MPI_Group world_group, row_group, col_group;

  ProcInfo();
  ~ProcInfo();

  // calculate what rank owns a given vertex
  int owner_of(int vertex);
};


std::vector<std::pair<int, int>> edge_list_from_file(const std::string& fname);

#endif
