#include <mpi.h>
#include "graph.h"


int main(int argc, char** argv) {

  // Init MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create kronecker graph
  Graph g = Graph::from_kronecker(10, 16, 123 + rank);
  // g.print_graph();

  // auto parents = g.parallel_top_down_bfs(0);
  auto parents = g.top_down_bfs(0);

  MPI_Finalize();
}
