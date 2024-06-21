#ifndef __UTIL_H_
#define __UTIL_H_

#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>


struct ProcInfo {
  // store rank, row, and column of adj matrix
  int rank, row_rank,col_rank, comm_size, grid_row, grid_col, width; 
  MPI_Comm row_comm, col_comm;
  
  static ProcInfo* instance;

  ProcInfo();
  ~ProcInfo();

  
  // Delete copy constructor and copy assignment operator
  ProcInfo(const ProcInfo&) = delete;
  ProcInfo& operator=(const ProcInfo&) = delete;

  // Public static method to access the instance
  static const ProcInfo* getInstance() {
      if (instance == nullptr) {
          instance = new ProcInfo();
      }
      return instance;
  }
};


std::vector<std::pair<int, int>> edge_list_from_file(const std::string& fname);

#endif
