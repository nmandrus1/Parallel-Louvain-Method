#include "community.h"
#include "graph.h"
#include <string>
#include <iostream>


int main(int argc, char** argv) {

  std::string f("./data/graph/graph.txt");
  Graph g2(f, false);
  g2.print_graph();

  Communities comm(g2);
  double init_mod = comm.modularity();
  comm.iterate();
  double new_mod = comm.modularity();
  std::cout << "(No MPI) Initial Modularity: " << init_mod << " and after one pass: " << new_mod << std::endl;

  for(int v = 0; v < g2.local_vcount; v++)
    std::cout << "Vtx " << v << " Community: " << comm.node_to_comm_map[v] << std::endl;

  return 0;
}


