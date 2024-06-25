#include <cstdlib>
#include <filesystem>
#include <mpi.h>
#include "community.h"
#include "graph.h"
#include <string>

namespace fs = std::filesystem;

struct Args {
  std::string infile;
};

bool parse_args(int argc, char** argv, Args* args) {
  if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <INFILE>\n";
        return false;
    }

    // Parse arguments
    args->infile= std::string(argv[1]);

  return true;
}

int run(int rank, int comm_size, Args args) {
  fs::path dir(args.infile);
  // files are named 0, 1, 2, ... n where rank n will access the file named n
  dir = dir.append(std::to_string(rank));
  // we just crash if these files don't exist 
  if(!fs::exists(dir)) {
    std::cerr << "Error: Preprocessed data file " << dir.string() << " does not exist for rank " << rank 
    << ". Please use the provided \"renumber.py\" script to process the data." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  Graph g(dir.string(), true);
  DistCommunities dist_comm(g);

  double init_mod = dist_comm.modularity();
  dist_comm.iterate();
  MPI_Barrier(MPI_COMM_WORLD);
  double new_mod = dist_comm.modularity();

  if(rank == 0) 
    std::cout << "Initial Modularity: " << init_mod << " and after one pass: " << new_mod << std::endl;


  for(int i = 0; i < 3; i += 2) {
    if(rank == i) {
        for(int v = g.rows.first; v < g.rows.second; v++)
          std::cout << "RANK " << rank << ": Vtx " << v << " Community: " << dist_comm.gbl_vtx_to_comm_map[v] << "\n";
        std::cout << std::endl;

    }

    MPI_Barrier(g.info->col_comm);
  }


  if(rank == 0) {
    std::string f("./data/graph/graph.txt");
    Graph g2(f, false);
    // g2.print_graph();
    Communities comm(g2);
    init_mod = comm.modularity();
    comm.iterate();
    new_mod = comm.modularity();
    std::cout << "(No MPI) Initial Modularity: " << init_mod << " and after one pass: " << new_mod << std::endl;

    for(int v = 0; v < g2.vcount; v++)
      std::cout << "Vtx " << v << " Community: " << comm.node_to_comm_map[v] << std::endl;
  }


  return 0;
}

int main(int argc, char** argv) {
  
  // Init MPI
  MPI_Init(&argc, &argv);
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  Args args;
  if(!parse_args(argc, argv, &args)) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  run(rank, comm_size, args);

  // if(rank == 0) {
  //   std::string f("./data/graph/graph.txt");
  //   g = Graph(f, false);
  //   g.print_graph();
  // }

  MPI_Finalize();
}


