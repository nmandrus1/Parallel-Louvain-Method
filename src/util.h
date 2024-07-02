#ifndef __UTIL_H_
#define __UTIL_H_

#include <mpi.h>
#include <vector>
#include <string>


struct ProcInfo {
  // store rank, row, and column of adj matrix
  int rank, comm_size; 
  
  // static ProcInfo* instance;

  ProcInfo();
};


std::vector<std::pair<int, int>> edge_list_from_file(const std::string& fname);

#endif
