#ifndef __UTIL_H_
#define __UTIL_H_

#include <mpi.h>
#include <vector>
#include <string>


struct ProcInfo {
  // store rank, row, and column of adj matrix
  int rank, comm_size; 
  
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
